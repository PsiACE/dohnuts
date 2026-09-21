"""Evaluate the exported checkpoint on the frozen JevBench v1.2.2 public tasks."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from dohnuts import __version__
from dohnuts.adapters import Qwen35Adapter
from dohnuts.experiment import Sampler, environment, memory
from dohnuts.predictor import Predictor
from dohnuts.train import file_hash

REVISION = "e105a48f8cdb7f3babb3594424f73e5d7bdc97b9"
UPSTREAM = Path(".cache/upstream/jevbench")
COHORTS = {"standard": "original", "easy": "easy", "hard": "hard"}


class DohnutsAdapter:
    name = "dohnuts_local"
    model = f"Dohnuts-{__version__}-0.8B"
    cost_basis = "local_gpu_no_provider_tariff"
    price_input_per_m = None
    price_output_per_m = None

    def __init__(self, predictor):
        self.predictor = predictor

    def reserve_estimate(self, task):
        return 0.0

    def run(self, task):
        from jevbench.adapters.base import DecisionResult, build_question

        body = {"state": task.state, "questions": {"decision": build_question(task)}}
        result = DecisionResult(
            adapter=self.name,
            ok=False,
            model=self.model,
            probs_source="native",
            request_body=body,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            response = self.predictor.predict(body["state"], body["questions"])
        except ValueError as error:
            # The deployed API rejects unsupported inputs. JevBench counts a
            # 422 as a wrong answer, without treating it as an infrastructure outage.
            result.status = 422
            result.error = str(error)
            result.latency_s = time.perf_counter() - started
            return result
        torch.cuda.synchronize()
        result.latency_s = time.perf_counter() - started
        result.raw = {"response": response}
        result.usage = response["usage"]
        answer = response["answers"]["decision"]
        if task.question["type"] == "noul":
            result.probs = {"no": 1.0 - answer["noul"], "yes": answer["noul"]}
        else:
            result.probs = answer["probabilities"]
        result.ok = True
        return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def summarize_run(output, cohorts, records, metadata, runtime, telemetry, upstream):
    from jevbench import composite_v12
    from jevbench.metrics import latency_summary
    from jevbench.summarize import metric, summarize

    seed = metadata.get("seed", Path(metadata["selected_run"]).name.removeprefix("seed-"))
    tasks = [task for group in cohorts.values() for task in group]
    tiers = {tier: metric(group, records) for tier, group in cohorts.items()}
    lookup = {row["task_id"]: row for row in records}
    gold_tasks = [task for task in cohorts["hard"] if task.provenance.get("gold_probs")]
    distances = [
        composite_v12.tvd(lookup[task.id]["probs"], task.provenance["gold_probs"], task.labels)
        for task in gold_tasks
        if lookup.get(task.id, {}).get("valid")
    ]
    mean_tvd = statistics.mean(distances) if distances else None
    ece = tiers["hard"]["ece"]
    original_ids = {task.id for task in cohorts["standard"]}
    original = [row for row in records if row["task_id"] in original_ids]
    latency = latency_summary([row["latency_s"] for row in original])
    long_inputs = [
        row for row in records if row["usage"].get("input_tokens", 0) > metadata["max_length"]
    ]
    references = json.loads((upstream / "results/v1.2/jevbench-v1.2-per-task.json").read_text())
    comparisons = []
    for key, system in references["systems"].items():
        outcomes = system["public_tasks"]
        result = {"model": key, "display": system["display"]}
        for tier, group in cohorts.items():
            codes = [outcomes.get(task.id, ["n"])[0] for task in group]
            result[tier + "_accuracy"] = codes.count("c") / len(group)
        codes = [outcomes.get(task.id, ["n"])[0] for task in tasks]
        result.update(
            planned=len(tasks),
            attempted=sum(code != "n" for code in codes),
            correct=codes.count("c"),
            failed=codes.count("f"),
            accuracy=codes.count("c") / len(tasks),
        )
        comparisons.append(result)
    comparisons.append(
        {
            "model": "dohnuts",
            "display": f"{metadata['model_id']} (seed {seed})",
            **{tier + "_accuracy": result["accuracy"] for tier, result in tiers.items()},
            "planned": len(tasks),
            "attempted": len(records),
            "correct": sum(bool(row["correct"]) for row in records),
            "failed": sum(not row["ok"] for row in records),
            "accuracy": sum(bool(row["correct"]) for row in records) / len(tasks),
        }
    )
    summary = {
        "benchmark": "JevBench v1.2.2 public subset",
        "upstream_revision": REVISION,
        "checkpoint": metadata,
        "coverage": {"available": len(tasks), "official_total": 534, "unavailable": 303},
        "ranked": False,
        "official_jevbench_score": None,
        "scope": [
            "231 public tasks; 303 private or non-redistributed tasks are unavailable, including the entire judge tier.",
            "Official runner and scoring functions; one decision per call, no concurrency, no retries.",
            "State, instructions and criteria are unchanged; labels and rationales are never given as answers.",
            f"Checkpoint, temperatures and the {runtime['input_limit']}-token serving limit are frozen before this evaluation; the training budget remains 2048 tokens.",
            "Input rejections count as wrong; no truncation or exclusion of over-budget questions.",
            "No provider tariff exists for local GPU execution. Cost and the four-axis composite stay null.",
            "Reference quality is recomputed on identical public IDs; published full-suite scores are not comparable.",
            "Public-subset Intelligence renormalizes the available easy/standard/hard weights; it is not leaderboard Intelligence.",
            "Latency is local end-to-end wall time. Model loading is separate; first decision is retained.",
            "The official speed cohort has 242 standard/judge tasks; only its 72 public standard tasks are available.",
        ],
        "metrics": summarize(tasks, records),
        "tiers": tiers,
        "public_subset_axes": {
            "intelligence": composite_v12.intelligence(
                {tier: result["accuracy"] for tier, result in tiers.items()}
            ),
            "calibration": composite_v12.calibration(ece["ece"] if ece else None, mean_tvd),
            "speed": composite_v12.speed(latency["p50_s"], latency["p95_s"], "gpu"),
            "cost": None,
        },
        "probability_fidelity": {
            "n_planned": len(gold_tasks),
            "n_valid": len(distances),
            "mean_tvd": mean_tvd,
        },
        "speed": {
            "public_standard_raw": latency,
            "first_decision_s": original[0]["latency_s"] if original else None,
            "warm_public_standard": latency_summary([row["latency_s"] for row in original[1:]]),
            "adjustment": "2 * raw + 0.15 seconds; upstream assumption, not measured",
        },
        "long_inputs": {
            "definition": f"Inputs longer than the {metadata['max_length']}-token training budget",
            "n": len(long_inputs),
            "n_valid": sum(row["valid"] for row in long_inputs),
            "n_correct": sum(row["correct"] for row in long_inputs),
            "latency": latency_summary([row["latency_s"] for row in long_inputs]),
            "max_input_tokens": max(
                (row["usage"].get("input_tokens", 0) for row in records), default=0
            ),
        },
        "input_rejections": [
            {"task_id": row["task_id"], "error": row["error"]}
            for row in records
            if row["status_code"] == 422
        ],
        "same_public_ids": comparisons,
        "runtime": runtime,
        "telemetry": telemetry,
    }
    write_json(output / "summary.json", summary)
    with (output / "same-public-ids.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    with (output / "decision-samples.csv").open("w") as stream:
        fields = ["task_id", "family", "valid", "correct", "input_tokens", "latency_s"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    **{key: row[key] for key in fields if key != "input_tokens"},
                    "input_tokens": row["usage"].get("input_tokens"),
                }
            )
    lines = [
        f"# {metadata['model_id']}: JevBench public evaluation",
        "",
        f"Checkpoint: seed {seed}, step {metadata['selected_step']}; SHA-256 `{metadata['weights_sha256']}`.",
        f"JevBench v1.2.2, pinned commit `{REVISION}`.",
        "",
        "All 231 public tasks are attempted. The official suite has 534 tasks; 303 are unavailable.",
        "This result has no official leaderboard rank or four-axis score. Local compute has no provider tariff.",
        f"Overall accuracy: {summary['metrics']['n_correct']}/231 ({summary['metrics']['accuracy']:.2%}), including input rejections as wrong answers.",
        "",
        "| Public tier | Tasks | Accuracy | Valid distributions | Brier | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier, result in tiers.items():
        lines.append(
            f"| {tier} | {result['n_planned']} | {result['accuracy']:.2%} | "
            f"{result['n_valid']} | {result['brier_mean']:.4f} | {result['ece']['ece']:.4f} |"
        )
    if long_inputs:
        long_latency = summary["long_inputs"]["latency"]
        lines.extend(
            [
                "",
                f"Inputs above the training length: {len(long_inputs)}; correct {summary['long_inputs']['n_correct']}; maximum input length {summary['long_inputs']['max_input_tokens']} tokens.",
                f"Their latency is p50 {long_latency['p50_s'] * 1000:.2f} ms and p95 {long_latency['p95_s'] * 1000:.2f} ms.",
            ]
        )
    lines.extend(
        [
            "",
            f"Public standard latency: p50 {latency['p50_s'] * 1000:.2f} ms, p95 {latency['p95_s'] * 1000:.2f} ms.",
            f"Inputs rejected by the deployed API: {len(summary['input_rejections'])}; these count as wrong.",
            "Brier and ECE above cover valid distributions only; rejected inputs have no probability distribution.",
            f"Model loading: {runtime['load_seconds']:.2f} s; first decision: {summary['speed']['first_decision_s'] * 1000:.2f} ms.",
            f"Whole-board peak VRAM during evaluation: {telemetry['board_vram_bytes']['max'] / 1024**3:.2f} GiB, including runtime and desktop.",
            "",
            "## Quality on identical public tasks",
            "",
            "Reference outcomes come from the upstream public per-task artifact, not from a new local run.",
            "",
            "| System | Attempted / 231 | Easy | Standard | Hard | Overall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(comparisons, key=lambda row: row["model"]):
        lines.append(
            f"| {row['display']} | {row['attempted']} | {row['easy_accuracy']:.2%} | "
            f"{row['standard_accuracy']:.2%} | {row['hard_accuracy']:.2%} | {row['accuracy']:.2%} |"
        )
    lines.extend(["", "## Measurement scope", "", *[f"- {note}" for note in summary["scope"]]])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"report": str(output / "report.md"), "metrics": summary["metrics"]["accuracy"]})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/v1/checkpoint"))
    parser.add_argument("--output", type=Path, default=Path("runs/v1/jevbench"))
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("JevBench must be pinned to the v1.2.2 commit")
    sys.path.insert(0, str(args.upstream.resolve()))
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.tasks import load_jsonl

    manifest = json.loads((args.upstream / "datasets/manifest.json").read_text())
    expected = {row["name"]: row for row in manifest["splits"]}
    cohorts, hashes = {}, {}
    for tier, name in COHORTS.items():
        path = args.upstream / "datasets/public" / f"{name}.jsonl"
        hashes[name] = file_hash(path)
        if hashes[name] != expected[name]["sha256"]:
            raise ValueError(f"JevBench dataset checksum mismatch: {name}")
        cohorts[tier] = load_jsonl(path)
        if len(cohorts[tier]) != expected[name]["n"]:
            raise ValueError(f"JevBench dataset count mismatch: {name}")
    tasks = [task for group in cohorts.values() for task in group]
    if len(tasks) != 231 or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Expected 231 unique public JevBench tasks")
    metadata = json.loads((args.checkpoint / "dohnuts.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "manifest.json",
        {
            "upstream_revision": REVISION,
            "dataset_sha256": hashes,
            "checkpoint_sha256": metadata["weights_sha256"],
            "temperatures": metadata["temperatures"],
            "mapping": "Unchanged state and build_question(task); noul P(true) maps to yes.",
            "protocol": "Serial, no retries, one native distribution per decision; official Runner.",
            "input_limit": Qwen35Adapter.max_input_tokens,
            "cost_basis": DohnutsAdapter.cost_basis,
        },
    )
    torch.set_num_threads(8)
    started = time.perf_counter()
    predictor = Predictor.from_checkpoint(args.checkpoint)
    runtime = environment(predictor.model, predictor.model.base_path)
    runtime["load_seconds"] = time.perf_counter() - started
    runtime["checkpoint"] = str(args.checkpoint)
    runtime["input_limit"] = predictor.max_length
    runner = Runner(
        DohnutsAdapter(predictor),
        Ledger(args.output / "ledger.jsonl", cap_usd=0),
        args.output / "raw",
        default_reserve_usd=0,
    )
    torch.cuda.reset_peak_memory_stats()
    with Sampler(Path("/sys/class/drm/card1/device"), interval=0.1) as sampler:
        records = runner.run_all(tasks, results_path=args.output / "results.jsonl")
    runtime["inference_memory"] = memory()
    if len(records) != len(tasks):
        raise RuntimeError(
            "JevBench stopped before every public task was attempted; see raw records"
        )
    summarize_run(
        args.output, cohorts, records, metadata, runtime, sampler.summary(), args.upstream
    )


if __name__ == "__main__":
    main()
