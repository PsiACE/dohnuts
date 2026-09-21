"""Evaluate the frozen Dohnuts checkpoint on the original Laya chart protocols."""

import argparse
import csv
import importlib.util
import json
import random
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from prepare_laya_benchmark import REVISION, SOURCE, UPSTREAM, digest

from dohnuts.experiment import Sampler, environment, latency_stats, memory, timed
from dohnuts.predictor import QTYPES, Predictor

CHECKPOINT = Path("runs/v1/checkpoint")
OUTPUT = Path("runs/v1/laya-chart")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_engine():
    notebook = json.loads((SOURCE / "laya_benchmark_colab.ipynb").read_text())
    source = "".join(notebook["cells"][8]["source"]).split("\n", 1)[1]
    path = OUTPUT / "upstream_engine.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("laya_benchmark_engine", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def add_permutations(suites):
    for name in ["massive_intent.en", "en.emotion", "xnli.en"]:
        original = suites["colab/" + name]
        rng = random.Random(99)
        cases, golds = [], []
        for (state, questions), gold in zip(
            original["cases"][:200], original["gold"][:200], strict=True
        ):
            qid, question = next(iter(questions.items()))
            keys = list(question["criteria"])
            order = list(range(len(keys)))
            rng.shuffle(order)
            criteria = {keys[i]: question["criteria"][keys[i]] for i in order}
            cases.append((state, {qid: {**question, "criteria": criteria}}))
            golds.append({qid: {"idx": order.index(gold[qid]["idx"])}})
        suites["order/" + name] = {"cases": cases, "gold": golds, "meta": {"base": "colab/" + name}}


def evaluate(predictor, suites):
    path = OUTPUT / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed = {(row["suite"], row["case"], row["qid"]) for row in rows}
    captured = []

    def capture(module, inputs, output):
        captured.append(output.detach().float().cpu())

    hook = predictor.model.register_forward_hook(capture)
    with path.open("a", buffering=1) as stream:
        for name, suite in suites.items():
            for index, ((state, questions), gold) in enumerate(
                zip(suite["cases"], suite["gold"], strict=True)
            ):
                if all((name, index, qid) in completed for qid in questions):
                    continue
                captured.clear()
                error, response = None, None
                try:
                    response = predictor.predict(state, questions)
                except ValueError as failure:
                    error = str(failure)
                for offset, (qid, question) in enumerate(questions.items()):
                    if (name, index, qid) in completed:
                        continue
                    kind = question["type"]
                    keys = (
                        ["false", "true"]
                        if kind == "noul"
                        else list(question["criteria"])
                        if kind == "choice"
                        else [str(i) for i in range(len(question["criteria"]))]
                    )
                    row = {
                        "suite": name,
                        "case": index,
                        "qid": qid,
                        "type": kind,
                        "keys": keys,
                        "gold": gold[qid],
                        "error": error,
                    }
                    if response is not None:
                        if len(captured) != 1:
                            raise RuntimeError(
                                "Each request must produce one batch of decision logits"
                            )
                        answer = response["answers"][qid]
                        probabilities = (
                            [1 - answer["noul"], answer["noul"]]
                            if kind == "noul"
                            else [answer["probabilities"][key] for key in keys]
                        )
                        row.update(
                            probabilities=probabilities,
                            logits=captured[0][offset, : len(keys)].tolist(),
                            usage=response["usage"],
                        )
                    workflow = suite["meta"].get("workflows")
                    if workflow:
                        row["workflow"] = workflow[index]
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                if (index + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "suite": name,
                                "cases_done": index + 1,
                                "cases_total": len(suite["cases"]),
                            }
                        ),
                        flush=True,
                    )
            print(json.dumps({"suite_finished": name, "decisions_recorded": len(rows)}), flush=True)
    hook.remove()
    return rows


def performance(predictor):
    import bench_latency

    result = {}
    with Sampler(Path("/sys/class/drm/card1/device")) as sampler:
        for count in [1, 5, 10, 50]:
            questions = bench_latency.qs(count)
            for _ in range(3):
                predictor.predict(bench_latency.STATE_EN, questions)
            result[str(count)] = latency_stats(
                timed(partial(predictor.predict, bench_latency.STATE_EN, questions), 20), count
            )
        rng = random.Random(13)
        mixtures = {}
        questions = bench_latency.qs(5)
        for state in [bench_latency.STATE_EN, bench_latency.STATE_HI]:
            predictor.predict(state, questions)
        for share in [0.0, 0.1, 0.3, 0.5]:
            flags = [rng.random() < share for _ in range(100)]
            durations = []
            for is_hindi in flags:
                state = bench_latency.STATE_HI if is_hindi else bench_latency.STATE_EN
                durations.extend(timed(partial(predictor.predict, state, questions), 1))
            mixtures[str(share)] = {
                **latency_stats(durations, 5),
                "hindi_calls": sum(flags),
                "calls": 100,
            }
    return {
        "batch_latency": result,
        "mixed_language": mixtures,
        "memory": memory(),
        "telemetry": sampler.summary(),
        "scope": "RX 7900 XTX; native predict(), BF16, eight CPU threads; same upstream workload; single resident Dohnuts model, no checkpoint swaps",
    }


def report(rows, suites, metadata, runtime, timing):
    from laya.common import temp_bucket

    engine = load_engine()
    groups = defaultdict(list)
    for row in rows:
        groups[row["suite"]].append(row)
    metrics, calibration = {}, {}
    for name, group in groups.items():
        valid = [r for r in group if not r["error"]]
        value = engine.hard_metrics([(r["gold"]["idx"], r["probabilities"]) for r in valid])
        value.update(n_planned=len(group), n_valid=len(valid), failures=len(group) - len(valid))
        value["accuracy"] = sum(
            int(np.argmax(r["probabilities"])) == r["gold"]["idx"] for r in valid
        ) / len(group)
        if name.startswith("apps/"):
            counts = Counter(r["gold"]["idx"] for r in group)
            value["class_counts"] = dict(sorted(counts.items()))
            value["majority_class_accuracy"] = max(counts.values()) / len(group)
            value["per_class_recall"] = {
                str(label): sum(
                    r["gold"]["idx"] == label and int(np.argmax(r["probabilities"])) == label
                    for r in valid
                )
                / count
                for label, count in counts.items()
            }
            value["balanced_accuracy"] = statistics.mean(value["per_class_recall"].values())
        if name.startswith(("massive51/", "colab/massive")) or name == "colab/typed_decisions":
            value.pop("macro_f1", None)
        metrics[name] = value
        if not name.startswith("colab/") or len(valid) < 60:
            continue
        midpoint = len(valid) // 2
        buckets = defaultdict(list)
        for row in valid[:midpoint]:
            buckets[temp_bucket(QTYPES[row["type"]], len(row["keys"]))].append(
                (row["logits"], row["gold"]["idx"])
            )
        fitted = {
            key: engine.fit_temperature(pairs) for key, pairs in buckets.items() if len(pairs) >= 25
        }
        shipped, refit = [], []
        for row in valid[midpoint:]:
            key = temp_bucket(QTYPES[row["type"]], len(row["keys"]))
            temperature = fitted.get(key, metadata["temperatures"][row["type"]])
            shipped.append((row["gold"]["idx"], row["probabilities"]))
            refit.append((row["gold"]["idx"], engine.softmax_t(row["logits"], temperature)))
        calibration[name] = {
            "n_fit": midpoint,
            "n_heldout": len(refit),
            "fitted_temperatures": fitted,
            "shipped": engine.hard_metrics(shipped),
            "refit": engine.hard_metrics(refit),
        }
        if name.startswith("colab/massive") or name == "colab/typed_decisions":
            for mode in ["shipped", "refit"]:
                calibration[name][mode].pop("macro_f1", None)
    typed = groups["colab/typed_decisions"]
    valid = [r for r in typed if not r["error"]]
    metrics["colab/typed_decisions"].update(
        soft_accuracy=float(
            np.mean(
                [
                    np.dot(r["probabilities"], np.array(r["gold"]["soft"]) / sum(r["gold"]["soft"]))
                    for r in valid
                ]
            )
        ),
        brier_vs_soft=float(
            np.mean(
                [
                    np.square(
                        np.array(r["probabilities"])
                        - np.array(r["gold"]["soft"]) / sum(r["gold"]["soft"])
                    ).sum()
                    for r in valid
                ]
            )
        ),
        score_mae=float(
            np.mean(
                [
                    abs(
                        np.dot(np.arange(len(r["keys"])), r["probabilities"])
                        - r["gold"]["gold_score"]
                    )
                    for r in valid
                    if "gold_score" in r["gold"]
                ]
            )
        ),
        by_workflow={
            workflow: engine.hard_metrics(
                [(r["gold"]["idx"], r["probabilities"]) for r in valid if r["workflow"] == workflow]
            )
            for workflow in sorted({r["workflow"] for r in valid})
        },
    )
    for value in metrics["colab/typed_decisions"]["by_workflow"].values():
        value.pop("macro_f1", None)
    order = {}
    for name, suite in suites.items():
        if not name.startswith("order/"):
            continue
        original = {r["case"]: r for r in groups[suite["meta"]["base"]]}
        comparable = [
            r for r in groups[name] if not r["error"] and not original[r["case"]]["error"]
        ]
        flips = sum(
            r["keys"][int(np.argmax(r["probabilities"]))]
            != original[r["case"]]["keys"][int(np.argmax(original[r["case"]]["probabilities"]))]
            for r in comparable
        )
        order[name] = {"n": len(comparable), "flips": flips, "flip_rate": flips / len(comparable)}
    languages = {
        name.split("/")[1]: value
        for name, value in metrics.items()
        if name.startswith("massive51/")
    }
    english_rest = {}
    for family in ["massive_intent", "massive_scenario", "xnli"]:
        english_rest[family] = {
            "english": metrics[f"colab/{family}.en"]["accuracy"],
            "other_languages_mean": statistics.mean(
                value["accuracy"]
                for name, value in metrics.items()
                if name.startswith(f"colab/{family}.") and name != f"colab/{family}.en"
            ),
        }
    published = json.loads((UPSTREAM / "research/results/t4_colab_benchmark.json").read_text())
    published_cpu = json.loads(
        (UPSTREAM / "research/results/cpu_51_language_sweep.json").read_text()
    )
    matched_calibration = {
        model: {
            "n_suites": len(reference["per_suite"]),
            "dohnuts_ece_shipped": statistics.mean(
                calibration["colab/" + name]["shipped"]["ece"] for name in reference["per_suite"]
            ),
            "dohnuts_ece_refit": statistics.mean(
                calibration["colab/" + name]["refit"]["ece"] for name in reference["per_suite"]
            ),
            "published_ece_shipped": reference["mean_ece_shipped"],
            "published_ece_refit": reference["mean_ece_refit"],
        }
        for model, reference in published["calibration_repair"].items()
    }
    reference_names = {
        "apps/jev.ag_news": "AG News (4 labels)",
        "apps/jev.emotion": "DAIR Emotion (6 labels)",
        "apps/jev.banking77_full": "banking77 (77 labels)",
        "apps/app.email_spam": "Email spam",
        "apps/app.phishing": "Phishing",
        "apps/app.guardrails_jailbreak": "LLM guardrails (jailbreak)",
        "apps/app.moderation_toxicity": "Moderation (toxicity)",
        "apps/app.rag_relevance": "RAG passage relevance",
        "apps/app.support_triage": "Support triage (10-way queue)",
        "apps/app.model_routing_domain": "Model routing (domain)",
    }
    reference_rows = {}
    for line in (UPSTREAM / "BENCHMARKS.md").read_text().splitlines():
        cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
        if len(cells) == 5 and cells[0] in reference_names.values():
            reference_rows[cells[0]] = [float(value) for value in cells[1:4]]
    application_references = {
        name: dict(
            zip(["english", "multilingual", "typed_decisions"], reference_rows[label], strict=True)
        )
        for name, label in reference_names.items()
    }
    typed_reference = {}
    for line in (UPSTREAM / "BENCHMARKS.md").read_text().splitlines():
        cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
        key = {
            "`laya`": "english",
            "`laya-multilingual`": "multilingual",
            "`laya-typed-decisions`": "typed_decisions",
        }.get(cells[0])
        if key and len(cells) == 6:
            typed_reference[key] = float(cells[1])
    application_references["colab/typed_decisions"] = typed_reference
    summary = {
        "checkpoint": metadata,
        "upstream_revision": REVISION,
        "runtime": runtime,
        "metrics": metrics,
        "massive51": {
            "macro_accuracy": statistics.mean(v["accuracy"] for v in languages.values()),
            "macro_ece": statistics.mean(v["ece"] for v in languages.values()),
            "languages_above_3x_random": sum(v["accuracy"] > 0.15 for v in languages.values()),
            "per_language": languages,
        },
        "english_vs_rest": english_rest,
        "calibration": {
            "mean_ece_shipped": statistics.mean(v["shipped"]["ece"] for v in calibration.values()),
            "mean_ece_refit": statistics.mean(v["refit"]["ece"] for v in calibration.values()),
            "per_suite": calibration,
            "matched_published_suites": matched_calibration,
            "scope": "Diagnostic only: per-suite first-half temperature fit, second-half evaluation, upstream cardinality buckets and hard-label NLL objective. Does not modify the deployed checkpoint temperatures.",
        },
        "option_order": order,
        "performance": timing,
        "published_references": {
            "applications": application_references,
            "massive51": published_cpu["part_a"]["by_model"],
            "colab": published,
            "scope": "Published historical Laya measurements, not locally rerun reference checkpoints; dataset revisions and individual predictions are unavailable for exact historical ID verification.",
        },
        "scope": json.loads((OUTPUT / "data-manifest.json").read_text())["scope"],
    }
    write_json(OUTPUT / "summary.json", summary)
    with (OUTPUT / "quality.csv").open("w") as stream:
        fields = ["suite", "n_planned", "n_valid", "accuracy", "ece", "nll", "brier", "failures"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {"suite": name, **{key: value[key] for key in fields[1:]}}
            for name, value in metrics.items()
        )
    with (OUTPUT / "calibration.csv").open("w") as stream:
        fields = [
            "suite",
            "n_fit",
            "n_heldout",
            "ece_shipped",
            "ece_refit",
            "nll_shipped",
            "nll_refit",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, value in calibration.items():
            writer.writerow(
                {
                    "suite": name,
                    "n_fit": value["n_fit"],
                    "n_heldout": value["n_heldout"],
                    **{
                        f"{metric}_{mode}": value[mode][metric]
                        for metric in ["ece", "nll"]
                        for mode in ["shipped", "refit"]
                    },
                }
            )
    with (OUTPUT / "latency-samples.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["workload", "questions", "sample", "end_to_end_ms"]
        )
        writer.writeheader()
        for count, value in timing["batch_latency"].items():
            writer.writerows(
                {"workload": "batch", "questions": count, "sample": i, "end_to_end_ms": duration}
                for i, duration in enumerate(value["samples_ms"])
            )
        for share, value in timing["mixed_language"].items():
            writer.writerows(
                {
                    "workload": f"hindi_share_{share}",
                    "questions": 5,
                    "sample": i,
                    "end_to_end_ms": duration,
                }
                for i, duration in enumerate(value["samples_ms"])
            )
    print(
        json.dumps(
            {
                "summary": str(OUTPUT / "summary.json"),
                "decisions": len(rows),
                "failures": sum(bool(row["error"]) for row in rows),
            }
        ),
        flush=True,
    )


def main():
    global OUTPUT, CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--data", type=Path, default=Path("data/benchmarks/laya"))
    args = parser.parse_args()
    OUTPUT, CHECKPOINT = args.output, args.checkpoint
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ["suites.json", "data-manifest.json"]:
        if (args.data / name).resolve() != (OUTPUT / name).resolve():
            shutil.copyfile(args.data / name, OUTPUT / name)
    torch.set_num_threads(8)
    sys.path[:0] = [str(SOURCE.resolve()), str(UPSTREAM.resolve())]
    suites = json.loads((OUTPUT / "suites.json").read_text())
    manifest = json.loads((OUTPUT / "data-manifest.json").read_text())
    if digest(OUTPUT / "suites.json") != manifest["suite_sha256"]:
        raise ValueError("Evaluation questions differ from the frozen manifest")
    metadata = json.loads((CHECKPOINT / "dohnuts.json").read_text())
    config = {
        "checkpoint_sha256": metadata["weights_sha256"],
        "suite_sha256": manifest["suite_sha256"],
        "temperatures": metadata["temperatures"],
        "inference_limit": 4096,
        "upstream_revision": REVISION,
    }
    config_path = OUTPUT / "evaluation-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Saved evaluation configuration differs from this run")
    write_json(config_path, config)
    add_permutations(suites)
    started = time.perf_counter()
    predictor = Predictor.from_checkpoint(CHECKPOINT)
    runtime = environment(predictor.model, predictor.model.base_path)
    runtime["load_seconds"] = time.perf_counter() - started
    rows = evaluate(predictor, suites)
    timing_path = OUTPUT / "performance.json"
    if not timing_path.exists():
        write_json(timing_path, performance(predictor))
    report(rows, suites, metadata, runtime, json.loads(timing_path.read_text()))


if __name__ == "__main__":
    main()
