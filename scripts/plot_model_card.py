"""Render model-card comparisons from a completed run; requires matplotlib.

Run: .cache/plot-venv/bin/python scripts/plot_model_card.py --run runs/v1
Outputs: six PNG/SVG/PDF figures, a PDF collection, data and provenance.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLORS = {
    "dohnuts": "#24677B",
    "jev": "#515D69",
    "laya-multilingual": "#B7692D",
    "laya-vision": "#805881",
}
NAMES = {
    "dohnuts": "Dohnuts",
    "jev": "Jev 1.13.0",
    "laya-multilingual": "Laya multilingual",
    "laya-vision": "Laya Vision",
}
TEXT, MUTED, GRID = "#192B35", "#586671", "#E5E9EC"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def read_json(path):
    return json.loads(path.read_text())


def frame(title, subtitle, height=8):
    fig = plt.figure(figsize=(14, height))
    fig.text(0.045, 0.95, title, fontsize=24, weight="bold", va="top")
    fig.text(0.045, 0.89, subtitle, fontsize=11, color=MUTED, va="top")
    return fig


def legend(fig, systems, y=0.845):
    fig.legend(
        [Patch(color=COLORS[s]) for s in systems],
        [NAMES[s] for s in systems],
        loc="upper left",
        bbox_to_anchor=(0.04, y),
        ncol=len(systems),
        frameon=False,
        handlelength=1.25,
        columnspacing=2.2,
    )


def hbars(ax, labels, series, limit=113, xlabel="Accuracy (%) · higher is better", decimals=1):
    positions = np.arange(len(labels))
    width = 0.72 / len(series)
    for index, (system, values) in enumerate(series.items()):
        yy = positions + (index - (len(series) - 1) / 2) * width
        ax.barh(yy, values, height=width * 0.83, color=COLORS[system])
        for y, value in zip(yy, values):
            ax.text(
                value + limit * 0.012,
                y,
                f"{value:.{decimals}f}",
                fontsize=9.5,
                va="center",
                color=COLORS[system],
            )
    ax.set_yticks(positions, labels)
    ax.set_ylim(len(labels) - 0.45, -0.65)
    ax.set_xlim(0, limit)
    ax.set_xlabel(xlabel, labelpad=13, fontsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(axis="both", length=0, pad=7)
    if limit == 113:
        ax.set_xticks([0, 25, 50, 75, 100])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/v1"))
    run = parser.parse_args().run
    output = run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        run / "checkpoint/dohnuts.json",
        run / "jevbench/comparison.json",
        run / "metrics/summary.json",
        run / "laya-chart/summary.json",
    ]
    sources += [run / f"benchmarks/{s}.jsonl" for s in ["dohnuts", "laya", "laya-vision"]]
    checkpoint, jev, summary, chart = map(read_json, sources[:4])
    if jev["checkpoint"]["checkpoint_sha256"] != checkpoint["weights_sha256"]:
        raise ValueError("JevBench results belong to a different checkpoint")
    if summary["checkpoint"]["weights_sha256"] != checkpoint["weights_sha256"]:
        raise ValueError("Quality results belong to a different checkpoint")
    benchmarks = {}
    for key, source in zip(["dohnuts", "laya-multilingual", "laya-vision"], sources[4:]):
        benchmarks[key] = {
            r["case"]: r
            for r in map(json.loads, source.read_text().splitlines())
            if r["kind"] == "matched_inference"
        }
    stamp = f"{checkpoint['model_id']}  ·  seed {checkpoint['seed']}  ·  step {checkpoint['selected_step']:,}  ·  weights {checkpoint['weights_sha256'][:12]}"
    figures, data = [], []
    collection = PdfPages(output / "model-card-comparisons.pdf")

    def record(figure, task, system, metric, value, n, scope):
        data.append(
            {
                "figure": figure,
                "task": task,
                "system": system,
                "metric": metric,
                "value": value,
                "n": n,
                "scope": scope,
            }
        )

    def save(fig, name, notes, description):
        fig.text(0.045, 0.12, "\n".join(notes), fontsize=9, color=MUTED, va="top", linespacing=1.55)
        fig.text(0.045, 0.025, stamp, fontsize=9, color=MUTED)
        fig.text(0.955, 0.025, name.replace("_", " "), fontsize=8, color=MUTED, ha="right")
        for extension in ["png", "svg", "pdf"]:
            fig.savefig(output / f"{name}.{extension}", dpi=220)
        collection.savefig(fig)
        figures.append({"name": name, "title": description, "notes": notes})
        plt.close(fig)

    # Identical public task IDs; output types and difficulty have separate denominators.
    fig = frame(
        "JevBench · public-task accuracy",
        "231 identical public task IDs. Labels are percentages; sample counts appear beside each group.",
        8.6,
    )
    systems = ["dohnuts", "jev", "laya-multilingual", "laya-vision"]
    legend(fig, systems)
    for position, dimension, groups, title in [
        (
            [0.16, 0.23, 0.33, 0.51],
            "type",
            ["overall", "choice", "noul", "score"],
            "By decision type",
        ),
        ([0.65, 0.23, 0.32, 0.51], "tier", ["easy", "standard", "hard"], "By difficulty"),
    ]:
        metrics = [
            jev["overall"] if g == "overall" else jev["by_group"][dimension][g] for g in groups
        ]
        decision_names = {
            "choice": "Candidate selection",
            "noul": "Truth estimate",
            "score": "Ordered score",
        }
        labels = [
            f"{decision_names.get(g, g.title())}\n(n={m['n']})" for g, m in zip(groups, metrics)
        ]
        values = {s: [100 * m[s]["accuracy"] for m in metrics] for s in systems}
        ax = fig.add_axes(position)
        hbars(ax, labels, values)
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=16)
        for g, m in zip(groups, metrics):
            for s in systems:
                record(
                    "01_jevbench",
                    g,
                    s,
                    "accuracy",
                    m[s]["accuracy"],
                    m["n"],
                    "identical_public_ids",
                )
    save(
        fig,
        "01_jevbench",
        [
            "Jev: published per-task outcomes, v1.2.2. Dohnuts and both Laya checkpoints: local measurements. All 231 local outputs are valid.",
            "Dohnuts context limit: 4,096 tokens. Laya defaults: 1,024 tokens with native truncation. Vision is tested on text here.",
            "303 official tasks are unavailable, including the judge tier. This is not the 534-task leaderboard score. One Dohnuts seed.",
        ],
        "JevBench: decision types and difficulty",
    )

    # The fixed multilingual checkpoint, with paired points and explicit numeric columns.
    apps = chart["published_references"]["applications"]
    titles = [
        "AG News",
        "Emotion",
        "Banking77",
        "Email spam",
        "Phishing",
        "Jailbreak guardrails",
        "Toxicity moderation",
        "RAG relevance",
        "Support triage",
        "Model routing",
        "Typed decisions",
    ]
    fig = frame(
        "Laya application tasks",
        "Dohnuts measurements compared with the published Laya multilingual checkpoint; accuracy, higher is better.",
        9.3,
    )
    fig.legend(
        [
            Line2D([], [], color=COLORS[s], marker=m, linestyle="none", markersize=8)
            for s, m in [("dohnuts", "o"), ("laya-multilingual", "s")]
        ],
        ["Dohnuts · measured", "Laya multilingual · published"],
        loc="upper left",
        bbox_to_anchor=(0.04, 0.845),
        ncol=2,
        frameon=False,
    )
    ax = fig.add_axes([0.24, 0.215, 0.47, 0.54])
    ax.set_xlim(0, 103)
    ax.set_ylim(10.6, -0.7)
    labels = []
    for index, ((task, ref), title) in enumerate(zip(apps.items(), titles)):
        metric = chart["metrics"][task]
        a, b = 100 * metric["accuracy"], 100 * ref["multilingual"]
        ax.plot([a, b], [index, index], color="#BCC5CC", linewidth=2, zorder=1)
        ax.scatter(a, index, color=COLORS["dohnuts"], s=54, zorder=3)
        ax.scatter(b, index, color=COLORS["laya-multilingual"], s=48, marker="s", zorder=2)
        labels.append(f"{title}  ·  {metric['n']:,}")
        for x, value, system in [(1.10, a, "dohnuts"), (1.40, b, "laya-multilingual")]:
            ax.text(
                x,
                index,
                f"{value:.2f}%",
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="center",
                color=COLORS[system],
                fontsize=11,
            )
            record(
                "02_applications",
                task,
                system,
                "accuracy",
                value / 100,
                metric["n"],
                "published_application_protocol",
            )
    for x, label, system in [
        (1.10, "Dohnuts", "dohnuts"),
        (1.40, "Multilingual", "laya-multilingual"),
    ]:
        ax.text(
            x,
            1.05,
            label,
            transform=ax.transAxes,
            ha="center",
            color=COLORS[system],
            fontsize=11,
            weight="bold",
        )
    ax.set_yticks(range(11), labels)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID)
    ax.tick_params(length=0, pad=9)
    ax.set_xlabel("Accuracy (%)", labelpad=14)
    save(
        fig,
        "02_applications",
        [
            "Counts beside tasks are decisions, not training examples. Original upstream question builders, seed 13 and candidate order.",
            "The published application/chart reference is used consistently, including typed decisions (2,000 decisions).",
            "Historical Laya dataset revisions and per-example hashes are unavailable; byte-identical historical inputs cannot be verified.",
        ],
        "Laya multilingual: application tasks",
    )

    fig = frame(
        "Multilingual decision quality",
        "Language-macro accuracy on the upstream evaluation protocols; each language has equal weight.",
        7.5,
    )
    legend(fig, ["dohnuts", "laya-multilingual"])
    names = [
        "MASSIVE intent\n51 languages × 100",
        "MASSIVE intent\n14 languages × 300",
        "MASSIVE scenario\n14 languages × 300",
        "XNLI\n15 languages × 300",
    ]
    ours = [chart["massive51"]["macro_accuracy"]]
    refs = [chart["published_references"]["massive51"]["multilingual"]["macro_accuracy"]]
    for prefix in ["massive_intent.", "massive_scenario.", "xnli."]:
        group = {
            k: v
            for k, v in chart["published_references"]["colab"]["suites"].items()
            if k.startswith(prefix)
        }
        ours.append(float(np.mean([chart["metrics"]["colab/" + k]["accuracy"] for k in group])))
        refs.append(
            float(
                np.mean([v["laya-multilingual"]["calibrated"]["accuracy"] for v in group.values()])
            )
        )
    ax = fig.add_axes([0.24, 0.25, 0.69, 0.49])
    hbars(ax, names, {"dohnuts": np.array(ours) * 100, "laya-multilingual": np.array(refs) * 100})
    for i, name in enumerate(names):
        for s, values in [("dohnuts", ours), ("laya-multilingual", refs)]:
            record(
                "03_multilingual",
                name.replace("\n", "; "),
                s,
                "language_macro_accuracy",
                values[i],
                [5100, 4200, 4200, 4500][i],
                "published_multilingual_protocol",
            )
    save(
        fig,
        "03_multilingual",
        [
            "MASSIVE intent uses 20 candidates; XNLI uses 3. The 51-language and 14-language suites are distinct evaluations.",
            f"On the 51-language suite, accuracy exceeds 15% (3× random) in {chart['massive51']['languages_above_3x_random']}/51 languages for Dohnuts and {chart['published_references']['massive51']['multilingual']['languages_above_random']}/51 for multilingual.",
            "Dohnuts: measured. Laya multilingual: published reference. Historical byte identity cannot be independently verified.",
        ],
        "Laya multilingual: language coverage and accuracy",
    )

    reference = summary["same_id_references"]["laya-vision"]
    tasks = ["aokvqa", "scienceqa", "vqav2_yesno"]
    labels = [
        f"{name}\n(n={reference['metrics'][task]['n']:,})"
        for task, name in zip(tasks, ["A-OKVQA", "ScienceQA", "VQAv2 yes/no"])
    ]
    fig = frame(
        "Visual decisions · accuracy and calibration",
        "The same example IDs for both models. Confidence quality is shown alongside top-label accuracy.",
        8.4,
    )
    legend(fig, ["dohnuts", "laya-vision"])
    for rect, field, title, limit, xlabel, decimals in [
        (
            [0.17, 0.25, 0.32, 0.49],
            "accuracy",
            "Accuracy",
            113,
            "Accuracy (%) · higher is better",
            1,
        ),
        ([0.65, 0.25, 0.31, 0.49], "ece_15", "Calibration", 0.205, "ECE · lower is better", 3),
    ]:
        series = {}
        for system, metrics in [
            ("dohnuts", reference["dohnuts_on_same_ids"]),
            ("laya-vision", reference["metrics"]),
        ]:
            series[system] = [
                metrics[t][field] * (100 if field == "accuracy" else 1) for t in tasks
            ]
            for task in tasks:
                record(
                    "04_vision",
                    task,
                    system,
                    field,
                    metrics[task][field],
                    metrics[task]["n"],
                    "local_same_ids",
                )
        ax = fig.add_axes(rect)
        hbars(ax, labels, series, limit, xlabel, decimals)
        ax.set_title(title, fontsize=14, weight="bold", loc="left", pad=16)
    save(
        fig,
        "04_vision",
        [
            "ScienceQA: official test image subset. VQAv2: independent split of the validation pool. These are not the model-card splits.",
            "Laya Vision has possible VQAv2 training exposure and A-OKVQA checkpoint-selection exposure; this is not a blind comparison.",
            "Laya: FP32 CPU, 2,048-token total/head budgets. Dohnuts: BF16. ECE: 15 bins, deployed temperatures, no per-task refit.",
        ],
        "Laya Vision: accuracy and calibration",
    )

    fig = frame(
        "Inference on one RX 7900 XTX",
        "Warm end-to-end predict latency · BF16 · lower is better. Each call contains distinct questions.",
        8.2,
    )
    legend(fig, ["dohnuts", "laya-multilingual", "laya-vision"])
    ax = fig.add_axes([0.08, 0.25, 0.43, 0.49])
    for system, marker, offsets in [
        ("dohnuts", "o", (0, -18)),
        ("laya-multilingual", "s", (0, -18)),
        ("laya-vision", "^", (0, 12)),
    ]:
        values = [
            benchmarks[system][f"distinct_text_{n}q"]["end_to_end"]["p50_ms"]
            for n in [1, 5, 10, 50]
        ]
        ax.plot(range(4), values, color=COLORS[system], marker=marker, markersize=7, linewidth=2)
        ax.annotate(
            f"{values[-1]:.1f} ms",
            (3, values[-1]),
            xytext=offsets,
            textcoords="offset points",
            ha="center",
            color=COLORS[system],
            fontsize=10,
        )
        for n, value in zip([1, 5, 10, 50], values):
            record(
                "05_latency", f"text_{n}q", system, "p50_ms", value, 20, "warm_same_gpu_native_api"
            )
    ax.set_xticks(range(4), [1, 5, 10, 50])
    ax.set_ylim(0, 150)
    ax.set_xlim(-0.2, 3.3)
    ax.set_xlabel("Questions per call (tested batch sizes)", labelpad=12)
    ax.set_ylabel("p50 latency (ms)")
    ax.yaxis.grid(True, color=GRID)
    ax.tick_params(length=0)
    ax.set_title("Text requests", loc="left", fontsize=14, weight="bold", pad=16)
    ax = fig.add_axes([0.68, 0.25, 0.28, 0.49])
    series = {
        s: [benchmarks[s][f"vision_protocol_image_{n}q"]["end_to_end"]["p50_ms"] for n in [1, 3]]
        for s in ["dohnuts", "laya-vision"]
    }
    hbars(ax, ["1 question", "3 questions"], series, 100, "p50 latency (ms) · lower is better")
    ax.set_title("One image per request", loc="left", fontsize=14, weight="bold", pad=16)
    for s, values in series.items():
        for n, value in zip([1, 3], values):
            record("05_latency", f"image_{n}q", s, "p50_ms", value, 20, "warm_same_gpu_native_api")
    save(
        fig,
        "05_latency",
        [
            "3 warmups and 20 synchronized repetitions. Includes preprocessing and transfers; excludes loading, network and queueing.",
            "Uses each native API, tokenizer and cache behavior. Image timings are warm, not uncached image-encoder timings.",
            f"Laya uses the fixed same-hardware baseline; Dohnuts uses {checkpoint['model_id']}. Jev API latency is not mixed into this chart.",
        ],
        "Same-hardware inference latency",
    )

    fig = frame(
        "JevBench · paired decision outcomes",
        "231 paired decisions for each reference. Correctness agreement and answer agreement are different measurements.",
        7.4,
    )
    segments = [
        ("both_correct", "Both correct", "#24677B"),
        ("dohnuts_only_correct", "Only Dohnuts correct", "#6EADAE"),
        ("reference_only_correct", "Only reference correct", "#C88B4D"),
        ("both_wrong", "Both wrong", "#D5DCE1"),
    ]
    fig.legend(
        [Patch(color=c) for _, _, c in segments],
        [label for _, label, _ in segments],
        loc="upper left",
        bbox_to_anchor=(0.04, 0.845),
        ncol=4,
        frameon=False,
        columnspacing=1.8,
    )
    ax = fig.add_axes([0.205, 0.29, 0.74, 0.43])
    for i, system in enumerate(["jev", "laya-multilingual", "laya-vision"]):
        offset = 0
        for key, label, color in segments:
            value = jev["paired"][system][key]
            ax.barh(i, value, left=offset, height=0.52, color=color, edgecolor="white", linewidth=1)
            if value >= 10:
                ax.text(
                    offset + value / 2,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if key == "both_correct" else TEXT,
                    weight="bold",
                )
            else:
                ax.annotate(
                    str(value),
                    (offset + value / 2, i - 0.26),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7},
                )
            record(
                "06_agreement",
                system,
                "Dohnuts vs " + system,
                key,
                value,
                231,
                "identical_public_ids",
            )
            offset += value
        assert offset == 231
    ax.set_yticks(range(3), [NAMES[s] for s in ["jev", "laya-multilingual", "laya-vision"]])
    ax.set_ylim(2.6, -0.65)
    ax.set_xlim(0, 231)
    ax.set_xticks([0, 50, 100, 150, 200, 231])
    ax.set_xlabel("Number of decisions", labelpad=12)
    ax.xaxis.grid(True, color=GRID)
    ax.tick_params(length=0, pad=8)
    agreement = []
    for system, label in [("laya-multilingual", "multilingual"), ("laya-vision", "Vision")]:
        count = jev["paired"][system]["exact_answer_agreement"]
        agreement.append(f"{label} {count}/231 ({count / 231:.1%})")
        record(
            "06_agreement",
            system,
            "Dohnuts vs " + system,
            "exact_answer_agreement",
            count,
            231,
            "identical_public_ids",
        )
    fig.text(
        0.205, 0.17, "Exact answer agreement:  " + "  ·  ".join(agreement), fontsize=11, color=TEXT
    )
    pair = jev["paired"]["jev"]
    save(
        fig,
        "06_agreement",
        [
            "Jev publishes correctness, not predicted labels: its exact answer-agreement count is unavailable.",
            f"{pair['both_correct']} shared correct answers confirm the same label. The {pair['both_wrong']} shared errors may have different labels; {pair['same_correctness']} is correctness agreement.",
        ],
        "JevBench: paired correctness and answer agreement",
    )
    collection.close()

    with (output / "chart-data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    manifest = {
        "version": checkpoint["version"],
        "model_id": checkpoint["model_id"],
        "checkpoint_sha256": checkpoint["weights_sha256"],
        "seed": checkpoint["seed"],
        "step": checkpoint["selected_step"],
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "figures": figures,
        "sources": {
            str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    index = [
        f"# {checkpoint['model_id']} comparisons",
        "",
        (
            f"Seed {checkpoint['seed']}, update {checkpoint['selected_step']:,}. "
            "Figures use saved evaluation results with fixed Laya multilingual and "
            "Laya Vision references."
        ),
        "",
        "[Complete PDF](model-card-comparisons.pdf) · [Chart data](chart-data.csv) · [Sources and checksums](manifest.json)",
        "",
        (
            "PNG files are high-resolution images; SVG files retain editable text; "
            "PDF files embed fonts. Captions document each evaluation protocol. "
            "One seed does not provide confidence intervals across seeds."
        ),
        "",
    ]
    for item in figures:
        name = item["name"]
        index += [
            f"## {item['title']}",
            "",
            f"![{item['title']}]({name}.png)",
            "",
            f"[PNG]({name}.png) · [SVG]({name}.svg) · [PDF]({name}.pdf)",
            "",
            *[f"- {note}" for note in item["notes"]],
            "",
        ]
    index += [
        "## Reproduce the figures",
        "",
        "From the repository root, use a Python environment with Matplotlib installed:",
        "",
        "```sh",
        f".cache/plot-venv/bin/python scripts/plot_model_card.py --run {run}",
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(index))
    print(json.dumps({"figures": len(figures), "data_rows": len(data), "output": str(output)}))


if __name__ == "__main__":
    main()
