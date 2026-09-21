"""Resumable mixed-dataset RLCD with development-only checkpoint selection."""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

import torch

from dohnuts import __version__
from dohnuts.experiment import Sampler, emit, environment, memory
from dohnuts.metrics import by_dataset, by_primitive_and_candidates, fit_temperatures
from dohnuts.model import DecisionModel
from dohnuts.recipe import LR_DECAY_STEPS, training_recipe
from dohnuts.rlcd import RLCDConfig, rlcd_loss
from dohnuts.training_data import (
    DecisionCollator,
    EvaluationBatches,
    TrainingBatches,
    load_records,
    prefetch_batches,
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(path, model, optimizer, step, config, score):
    payload = {
        "format_version": 1,
        "step": step,
        "config": config,
        "dev_macro_accuracy": score,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(model, path, optimizer=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    if set(state["trainable"]) != expected:
        raise ValueError("Checkpoint trainable keys differ")
    with torch.no_grad():
        for name, value in state["trainable"].items():
            params[name].copy_(value)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
    return state


@torch.inference_mode()
def evaluate(model, groups, collator, output, batch_size=16):
    model.eval()
    records = []
    loader = torch.utils.data.DataLoader(
        EvaluationBatches(groups, collator, batch_size),
        batch_size=None,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )
    pending = deque()
    counts = Counter()

    with Path(output).open("w") as stream:

        def collect():
            ready, logits, rows = pending.popleft()
            # D2H is asynchronous: CPU must not inspect pinned results before this event.
            ready.synchronize()
            for i, row in enumerate(rows):
                result = {k: row[k] for k in ["id", "dataset", "group", "target"]}
                result.update(
                    type=row["question"]["type"],
                    logits=logits[i, : len(row["target"])].tolist(),
                )
                if not torch.isfinite(logits[i, : len(row["target"])]).all():
                    raise RuntimeError(f"Non-finite prediction: {row['id']}")
                stream.write(json.dumps(result) + "\n")
                records.append(result)
                counts[row["dataset"]] += 1
            key = rows[-1]["dataset"]
            if counts[key] == len(groups[key]):
                stream.flush()
                print(
                    json.dumps({"kind": "evaluation_progress", "dataset": key, "n": counts[key]}),
                    flush=True,
                )

        for (inputs, positions, _, _, _), rows in prefetch_batches(loader):
            logits = model(inputs, positions)
            host = torch.empty_like(logits, device="cpu", pin_memory=True)
            host.copy_(logits, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
            pending.append((ready, host, rows))
            # Submit the following forward before serializing the preceding results.
            if len(pending) == 2:
                collect()
        while pending:
            collect()
    return records


def train(config, run, *, resume=False, adapter=None, initialize_from=None):
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    policy = RLCDConfig(**config.get("rlcd", {}))
    model = DecisionModel(config["model"], adapter=adapter)
    model.enable_lora()
    parent = None
    if initialize_from is not None:
        metadata = model.load_adapter(initialize_from)
        parent = {"checkpoint": str(initialize_from), "weights_sha256": metadata["weights_sha256"]}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("head.")
                ],
                "lr": config["backbone_lr"],
                "initial_lr": config["backbone_lr"],
            },
            {
                "params": model.head.parameters(),
                "lr": config["head_lr"],
                "initial_lr": config["head_lr"],
            },
        ],
        weight_decay=0.01,
    )
    data = Path(config["data"])
    groups = load_records(data / "train.jsonl", config["train_cap"])
    dev = load_records(data / "dev.jsonl", config["dev_cap"])
    source_hashes = {
        split: file_hash(data / f"{split}.jsonl")
        for split in ["train", "dev", "calibration", "test"]
    }
    frozen = {
        **config,
        "source_hashes": source_hashes,
        "rlcd": asdict(policy),
        "sampling": "uniform dataset, replacement, seed+microbatch index; choice permutation only",
        "dev_selection": "fixed SHA-256 subset per dataset; unweighted dataset macro top-1 accuracy",
    }
    if parent is not None:
        frozen["initialized_from"] = parent
    if model.adapter.name != "qwen3.5":
        frozen["adapter"] = model.adapter.name
        frozen["base_model"] = model.adapter.base_model
    config_path = run / "config.json"
    previous = json.loads(config_path.read_text()) if config_path.exists() else None
    frozen["lr_decay_steps"] = (
        previous["lr_decay_steps"] if previous else min(config["steps"], LR_DECAY_STEPS)
    )
    if previous is not None:
        expected = {
            **previous,
            "lr_decay_steps": frozen["lr_decay_steps"],
            "steps": config["steps"],
        }
        if not resume or expected != frozen or config["steps"] < previous["steps"]:
            raise ValueError("Resume may only extend the step budget of the same recipe and data")
    temporary = config_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(frozen, indent=2) + "\n")
    temporary.replace(config_path)
    collator = DecisionCollator(config["model"], adapter=adapter)
    step = 0
    best = -1.0
    if resume:
        state = load_checkpoint(model, run / "last.pt", optimizer)
        step = state["step"]
        best = state["dev_macro_accuracy"]
        # Only saved updates belong to the resumed trajectory. A stopped process
        # may have logged later steps before its next checkpoint was committed.
        metrics_path = run / "metrics.jsonl"
        committed = [
            line
            for line in metrics_path.read_text().splitlines()
            if json.loads(line).get("step", 0) <= step
            and json.loads(line).get("kind") != "train_complete"
        ]
        temporary = metrics_path.with_suffix(".tmp")
        temporary.write_text("\n".join(committed) + "\n")
        temporary.replace(metrics_path)
        emit(
            metrics_path,
            {
                "kind": "resume",
                "step": step,
                "target_step": config["steps"],
                "lr_decay_steps": frozen["lr_decay_steps"],
                "optimizer_lr": [group["lr"] for group in optimizer.param_groups],
                "checkpoint_sha256": file_hash(run / "last.pt"),
            },
        )
    else:
        if (run / "metrics.jsonl").exists():
            raise ValueError("Existing run requires --resume")
        emit(run / "metrics.jsonl", environment(model, Path(config["model"])))
        (run / "samples.json").write_text(
            json.dumps(
                {
                    "train": {k: [r["id"] for r in v] for k, v in groups.items()},
                    "dev": {k: [r["id"] for r in v] for k, v in dev.items()},
                },
                indent=2,
            )
        )
        baseline = evaluate(
            model, dev, collator, run / "dev-step-000000.jsonl", config["eval_batch_size"]
        )
        baseline_metrics = by_dataset(baseline)
        emit(run / "metrics.jsonl", {"kind": "dev", "step": 0, "metrics": baseline_metrics})
        best = baseline_metrics["macro_accuracy"]
        # Continuing from a good checkpoint may never improve development quality.
        # In that case the untouched parent remains the selected candidate.
        save_checkpoint(run / "best.pt", model, optimizer, 0, frozen, best)
    dataset = TrainingBatches(
        groups,
        collator,
        seed=config["seed"],
        batch_size=config["batch_size"],
        steps=config["steps"],
        accumulation=config["accumulation"],
        start_step=step,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=config["workers"],
        prefetch_factor=2 if config["workers"] else None,
        pin_memory=True,
    )
    iterator = prefetch_batches(loader)
    start_time = time.perf_counter()
    consumed = Counter()
    elapsed_before_resume = 0.0
    if resume:
        # Restore plotting counters from the last logged step at/before the
        # checkpoint; optimizer/RNG and actual data order were restored above.
        for line in (run / "metrics.jsonl").read_text().splitlines():
            previous = json.loads(line)
            if previous.get("kind") == "train" and previous["step"] <= step:
                consumed = Counter(previous["consumed"])
                elapsed_before_resume = previous["elapsed_s"]
    with Sampler(
        Path("/sys/class/drm/card1/device"), interval=1.0, output=run / "resources.jsonl"
    ) as telemetry:
        while step < config["steps"]:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            decay_steps = frozen["lr_decay_steps"]
            warmup = max(1, int(decay_steps * 0.03))
            progress = (min(step, decay_steps) - warmup) / max(1, decay_steps - warmup)
            factor = (
                min(1.0, (step + 1) / warmup)
                if step < warmup
                else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            )
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            stats = Counter()
            finite = torch.ones((), dtype=torch.bool, device="cuda")
            step_start = time.perf_counter()
            for micro in range(config["accumulation"]):
                batch, key, ids = next(iterator)
                inputs, positions, mask, target, ordinal = batch
                logits = model(inputs, positions)
                loss, metrics = rlcd_loss(logits, target, mask=mask, ordinal=ordinal, config=policy)
                finite &= torch.isfinite(loss.detach())
                (loss / config["accumulation"]).backward()
                stats.update({k: v.detach() / config["accumulation"] for k, v in metrics.items()})
                stats["loss"] += loss.detach() / config["accumulation"]
                stats["accuracy"] += (
                    logits.detach().masked_fill(~mask, -torch.inf).argmax(-1) == target.argmax(-1)
                ).float().mean() / config["accumulation"]
                consumed[key] += len(ids)
            if not finite:
                raise RuntimeError(f"Non-finite loss at step {step}; optimizer was not advanced")
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step == 1 or step % config["log_every"] == 0:
                torch.cuda.synchronize()
                emit(
                    run / "metrics.jsonl",
                    {
                        "kind": "train",
                        "step": step,
                        "elapsed_s": elapsed_before_resume + time.perf_counter() - start_time,
                        "step_s": time.perf_counter() - step_start,
                        **dict(zip(stats, torch.stack(list(stats.values())).cpu().tolist())),
                        "grad_norm": float(grad_norm),
                        "lr": optimizer.param_groups[0]["lr"],
                        "consumed": dict(consumed),
                        "memory": memory(),
                    },
                )
            if step % config["eval_every"] == 0 or step == config["steps"]:
                predictions = evaluate(
                    model,
                    dev,
                    collator,
                    run / f"dev-step-{step:06d}.jsonl",
                    config["eval_batch_size"],
                )
                metrics = by_dataset(predictions)
                score = metrics["macro_accuracy"]
                emit(run / "metrics.jsonl", {"kind": "dev", "step": step, "metrics": metrics})
                if score > best:
                    best = score
                    save_checkpoint(run / "best.pt", model, optimizer, step, frozen, best)
                save_checkpoint(run / "last.pt", model, optimizer, step, frozen, best)
            elif step % config["save_every"] == 0:
                save_checkpoint(run / "last.pt", model, optimizer, step, frozen, best)
    emit(
        run / "metrics.jsonl",
        {
            "kind": "train_complete",
            "step": step,
            "best_dev_macro_accuracy": best,
            "telemetry": telemetry.summary(),
            "consumed": dict(consumed),
        },
    )


def final_evaluation(config, run, *, adapter=None):
    frozen = json.loads((run / "config.json").read_text())
    for key in ["model", "data", "lora_rank", "image_pixels", "max_length", "backend"]:
        if config.get(key) != frozen.get(key):
            raise ValueError(f"Evaluation recipe differs from the selected run: {key}")
    data = Path(config["data"])
    for split, expected in frozen["source_hashes"].items():
        if file_hash(data / f"{split}.jsonl") != expected:
            raise ValueError(f"Evaluation data changed since training: {split}")
    torch.set_num_threads(config["cpu_threads"])
    model = DecisionModel(config["model"], adapter=adapter)
    if model.adapter.name != frozen.get("adapter", "qwen3.5"):
        raise ValueError("Evaluation adapter differs from the trained model")
    model.enable_lora(checkpointing=False)
    state = load_checkpoint(model, run / "best.pt")
    # Calibration and final evaluation use exactly the deployed merged model.
    model.merge()
    collator = DecisionCollator(config["model"], adapter=adapter)
    calibration = evaluate(
        model,
        load_records(data / "calibration.jsonl"),
        collator,
        run / "calibration-predictions.jsonl",
        config["eval_batch_size"],
    )
    temperatures = fit_temperatures(calibration)
    (run / "temperatures.json").write_text(json.dumps(temperatures, indent=2) + "\n")
    predictions = evaluate(
        model,
        load_records(data / "test.jsonl"),
        collator,
        run / "test-predictions.jsonl",
        config["eval_batch_size"],
    )
    training_monitor = evaluate(
        model,
        load_records(data / "train.jsonl", config["dev_cap"]),
        collator,
        run / "train-monitor-predictions.jsonl",
        config["eval_batch_size"],
    )
    report = {
        "selected_step": state["step"],
        "checkpoint_sha256": file_hash(run / "best.pt"),
        "source_hashes": frozen["source_hashes"],
        "seed": frozen["seed"],
        "inference": "merged LoRA, BF16, fused operations, shared-prefix parallel candidate scoring",
        "temperatures": temperatures,
        "uncalibrated": by_dataset(predictions),
        "calibrated": by_dataset(predictions, temperatures),
        "primitive_candidate_slices": by_primitive_and_candidates(predictions, temperatures),
        "train_monitor": by_dataset(training_monitor, temperatures),
        "train_monitor_scope": "fixed subset of the capped training pool; diagnostic only, no selection",
    }
    (run / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    export_checkpoint(state, run, run.parent / "checkpoint")
    print(
        json.dumps(
            {
                "kind": "evaluation_complete",
                "run": str(run),
                "macro_accuracy": report["calibrated"]["macro_accuracy"],
            }
        ),
        flush=True,
    )


def export_checkpoint(state, run, output):
    """Export the development-selected, calibrated weights from this run."""
    from safetensors.torch import save_file

    config = state["config"]
    base = Path(config["model"])
    calibration_path = run / "temperatures.json"
    if not calibration_path.exists():
        raise FileNotFoundError("Calibrate the selected run before export")
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "adapter.safetensors"
    temporary = weights.with_suffix(".tmp")
    save_file(
        {name: value.contiguous() for name, value in state["trainable"].items()}, str(temporary)
    )
    temporary.replace(weights)
    metadata = {
        "format_version": 1,
        "project": "dohnuts",
        "distribution": "dohnuts",
        "version": __version__,
        "model_id": f"Dohnuts-{__version__}-0.8B",
        "base_model": config.get("base_model", "Qwen/Qwen3.5-0.8B"),
        "base_path": str(base),
        "base_revision": (base / "revision.txt").read_text().strip(),
        "adapter": config.get("adapter", "qwen3.5"),
        "lora_rank": config["lora_rank"],
        "image_pixels": config["image_pixels"],
        "max_length": config["max_length"],
        "backend": config.get("backend", {}),
        "inference": "merged LoRA, BF16, fused operations, shared-prefix parallel candidate scoring",
        "source_hashes": config["source_hashes"],
        "selected_run": str(run),
        "selected_step": state["step"],
        "selection": "maximum development macro accuracy within the run; no test selection",
        "seed": config["seed"],
        "dev_macro_accuracy": state["dev_macro_accuracy"],
        "temperatures": json.loads(calibration_path.read_text()),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    if "initialized_from" in config:
        metadata["initialized_from"] = config["initialized_from"]
    (output / "dohnuts.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initialize-from", type=Path, help="Exported checkpoint used to initialize training"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    expected = training_recipe(
        model=config["model"],
        data=config["data"],
        seed=config["seed"],
        rlcd=RLCDConfig(**config.get("rlcd", {})),
        steps=config["steps"],
    )
    if config != expected:
        raise ValueError("Training uses the fixed recipe, RLCD controls, and step budget")
    if args.action == "train":
        train(config, args.run, resume=args.resume, initialize_from=args.initialize_from)
    else:
        final_evaluation(config, args.run)


if __name__ == "__main__":
    main()
