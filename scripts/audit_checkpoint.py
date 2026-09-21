"""Held-out option-order, image-dependence, and BF16 batch-size audits.

These diagnostics do not select the checkpoint or change calibration. Raw
logits and IDs accompany every variant. Only A-OKVQA four-choice rows are rotated
to reproduce the upstream four-rotation diagnostic without changing ordinality.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from dohnuts.metrics import by_dataset
from dohnuts.predictor import Predictor
from dohnuts.train import evaluate
from dohnuts.training_data import DecisionCollator, load_records


def paired(reference, actual):
    lookup = {r["id"]: r for r in reference}
    differences, probability_deltas, agreements = [], [], []
    for row in actual:
        original = lookup[row["id"]]
        p, q = (np.array(r["logits"], dtype=float) for r in [original, row])
        p, q = np.exp(p - p.max()), np.exp(q - q.max())
        p, q = p / p.sum(), q / q.sum()
        target = np.argmax(row["target"])
        agreements.append(int(p.argmax() == q.argmax()))
        differences.append(int(q.argmax() == target) - int(p.argmax() == target))
        probability_deltas.append(float(np.abs(p - q).max()))
    rng = np.random.default_rng(20260920)
    delta = np.array(differences)
    means = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(2000)]
    return {
        "n": len(delta),
        "argmax_agreement": float(np.mean(agreements)),
        "accuracy_delta": float(delta.mean()),
        "paired_bootstrap_95_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "max_probability_difference": max(probability_deltas),
        "mean_max_probability_difference": float(np.mean(probability_deltas)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    predictor = Predictor.from_checkpoint(args.checkpoint)
    collator = DecisionCollator(predictor.model.base_path)
    groups = load_records(args.data / "test.jsonl", 256)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = evaluate(predictor.model, groups, collator, args.output / "reference.jsonl", 16)
    results = {
        "reference": by_dataset(
            reference, dict(zip(["choice", "score", "noul"], predictor.temperatures, strict=True))
        )
    }
    # A smaller deterministic subset keeps the single-example audit bounded.
    batch_groups = {k: rows[:64] for k, rows in groups.items()}
    batch16 = evaluate(predictor.model, batch_groups, collator, args.output / "batch16.jsonl", 16)
    batch1 = evaluate(predictor.model, batch_groups, collator, args.output / "batch1.jsonl", 1)
    results["batch1_vs_batch16"] = paired(batch16, batch1)
    for variant in ["missing_image", "mismatched_image"]:
        altered = {}
        excluded = []
        for name, rows in groups.items():
            image_rows = [r for r in rows if r["image"]]
            if len({r["image"] for r in image_rows}) < 2:
                continue
            # Match visual token count, so the ablation cannot cross the frozen
            # sequence budget merely by substituting a different aspect ratio.
            images = {}
            for row in image_rows:
                if row["image"] in images:
                    continue
                with Image.open(row["image"]) as image:
                    height, width = smart_resize(
                        image.height,
                        image.width,
                        factor=32,
                        min_pixels=predictor.image_pixels,
                        max_pixels=predictor.image_pixels,
                    )
                images[row["image"]] = (height * width, row["group"])
            altered[name] = []
            for original in image_rows:
                row = copy.deepcopy(original)
                if variant == "missing_image":
                    row["image"] = None
                else:
                    size, group = images[row["image"]]
                    candidates = [
                        path for path, (s, g) in images.items() if s == size and g != group
                    ]
                    if not candidates:
                        excluded.append(
                            {
                                "id": row["id"],
                                "reason": "no different image group at same visual token count",
                            }
                        )
                        continue
                    # Stable different-image pairing, independent of labels.
                    index = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
                    row["image"] = candidates[index % len(candidates)]
                altered[name].append(row)
        predictions = evaluate(
            predictor.model, altered, collator, args.output / f"{variant}.jsonl", 16
        )
        results[variant] = {
            "metrics": by_dataset(predictions),
            "paired": paired(reference, predictions),
            "excluded": excluded,
        }
    aok = load_records(args.data / "test.jsonl")["aokvqa"]
    cyclic = []
    for shift in range(4):
        rows = copy.deepcopy(aok)
        orders = {}
        for row in rows:
            if len(row["target"]) != 4:
                raise ValueError("The upstream rotation diagnostic requires four choices")
            order = [(i + shift) % 4 for i in range(4)]
            options = row["question"]["criteria"]
            row["question"]["criteria"] = (
                {list(options)[i]: options[list(options)[i]] for i in order}
                if isinstance(options, dict)
                else [options[i] for i in order]
            )
            row["target"] = [row["target"][i] for i in order]
            orders[row["id"]] = order
        predictions = evaluate(
            predictor.model, {"aokvqa": rows}, collator, args.output / f"rotation-{shift}.jsonl", 16
        )
        cyclic.append(by_dataset(predictions)["aokvqa"]["accuracy"])
        # Retain a second file aligned back to the original candidate order.
        with (args.output / f"rotation-{shift}-aligned.jsonl").open("w") as stream:
            for row in predictions:
                inverse = np.argsort(orders[row["id"]])
                for key in ["logits", "target"]:
                    row[key] = [row[key][i] for i in inverse]
                stream.write(json.dumps(row) + "\n")
    results["cyclic_option_accuracy"] = cyclic
    results["cyclic_accuracy_spread"] = max(cyclic) - min(cyclic)
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
