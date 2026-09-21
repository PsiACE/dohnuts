"""Plot-ready distribution metrics; ECE uses top probability, never entropy."""

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def summarize(records, temperatures=None):
    temperatures = temperatures or {"choice": 1.0, "noul": 1.0, "score": 1.0}
    values = defaultdict(list)
    labels, predictions = [], []
    for row in records:
        logits = np.array(row["logits"], dtype=np.float64) / temperatures[row["type"]]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        target = np.array(row["target"], dtype=np.float64)
        label, pred = int(target.argmax()), int(p.argmax())
        labels.append(label)
        predictions.append(pred)
        values["accuracy"].append(float(label == pred))
        values["soft_accuracy"].append(float(target[pred]))
        values["nll"].append(float(-(target * np.log(np.maximum(p, 1e-12))).sum()))
        values["brier_sum"].append(float(((p - target) ** 2).sum()))
        values["brier_per_candidate"].append(float(((p - target) ** 2).mean()))
        values["confidence"].append(float(p.max()))
        values["schema_pass_rate"].append(float(np.isfinite(p).all() and abs(p.sum() - 1) < 1e-6))
        if row["type"] == "score":
            indices = np.arange(len(p))
            values["score_mae"].append(float(abs((indices * p).sum() - (indices * target).sum())))
            values["rps"].append(float(((p.cumsum()[:-1] - target.cumsum()[:-1]) ** 2).mean()))
    result: dict[str, Any] = {k: float(np.mean(v)) for k, v in values.items() if k != "confidence"}
    result["n"] = len(records)
    if not records:
        return result
    # Label-index F1 is meaningful only within a fixed candidate vocabulary.
    result["macro_f1"] = float(f1_score(labels, predictions, average="macro", zero_division=0))
    bins = []
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (np.array(values["confidence"]) > lo) & (np.array(values["confidence"]) <= hi)
        count = int(mask.sum())
        confidence = accuracy = None
        if count:
            confidence = float(np.array(values["confidence"])[mask].mean())
            accuracy = float(np.array(values["accuracy"])[mask].mean())
            ece += count / len(records) * abs(confidence - accuracy)
        bins.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "n": count,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    result["ece_15"] = ece
    result["reliability"] = bins
    return result


def by_dataset(records, temperatures=None):
    groups = defaultdict(list)
    for row in records:
        groups[row["dataset"]].append(row)
    result = {key: summarize(rows, temperatures) for key, rows in sorted(groups.items())}
    # Candidate indices differ by question on these datasets; index macro-F1 is invalid.
    for key, metrics in result.items():
        if key in {"aokvqa", "scienceqa", "typed_decisions", "screenqa_choice", "clevr_attribute"}:
            metrics.pop("macro_f1", None)
    result["macro_accuracy"] = (
        float(np.mean([m["accuracy"] for m in result.values()])) if result else None
    )
    return result


def by_primitive_and_candidates(records, temperatures=None):
    groups = defaultdict(list)
    for row in records:
        groups[row["type"], len(row["target"])].append(row)
    result = []
    for (kind, candidates), rows in sorted(groups.items()):
        metrics = summarize(rows, temperatures)
        metrics.pop("macro_f1", None)  # Different tasks do not share class semantics.
        result.append({"primitive": kind, "candidates": candidates, **metrics})
    return result


def fit_temperatures(records):
    """Laya log-temperature LBFGS per decision type, using independent calibration rows."""
    import torch

    groups = defaultdict(list)
    for row in records:
        groups[row["type"]].append(row)
    result = {key: 1.0 for key in ["choice", "noul", "score"]}
    for kind, rows in groups.items():
        if len(rows) < 10:
            continue
        width = max(len(row["logits"]) for row in rows)
        logits = torch.full((len(rows), width), -1e4)
        targets = torch.zeros_like(logits)
        for i, row in enumerate(rows):
            logits[i, : len(row["logits"])] = torch.tensor(row["logits"])
            targets[i, : len(row["target"])] = torch.tensor(row["target"])
        log_t = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure(optimizer=optimizer, targets=targets, logits=logits, log_t=log_t):
            optimizer.zero_grad()
            loss = -(targets * (logits / log_t.exp().clamp(0.1, 10)).log_softmax(-1)).sum(-1).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        result[kind] = float(log_t.detach().exp().clamp(0.1, 10))
    return result
