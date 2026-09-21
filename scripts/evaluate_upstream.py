"""Pinned upstream quality references on the exact Dohnuts evaluation example IDs.

This is an evaluation utility; Laya Agents are used only to measure the reference checkpoints.
Raw model logits are captured before the API's four-decimal output rounding.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from dohnuts.metrics import by_dataset
from dohnuts.train import file_hash
from dohnuts.training_data import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["laya", "laya-vision"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(".cache/upstream") / args.engine))
    if args.engine == "laya":
        from laya.agent import Agent

        agent = Agent(".cache/models/laya-multilingual", device="cpu")
        names = {
            "ag_news",
            "banking77",
            "boolq",
            "emotion",
            "massive_en-US",
            "massive_zh-CN",
            "typed_decisions",
            "xnli_en",
            "xnli_zh",
        }
    else:
        from laya.vlm import VLMAgent

        agent = VLMAgent(".cache/models/laya-vision-trained", device="cpu")
        names = {"aokvqa", "scienceqa", "vqav2_yesno"}
    from laya.common import QTYPES, temp_bucket

    original = dict(agent.cfg)
    # Keep the entire candidate set; disclose extension beyond published defaults.
    agent.cfg.update(max_len=2048, head_max_len=2048)
    groups = load_records(args.data / "test.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "engine": args.engine,
        "device": "cpu",
        "dtype": str(next(agent.model.parameters()).dtype),
        "split": "test",
        "data": str(args.data),
        "source_sha256": file_hash(args.data / "test.jsonl"),
        "original_config": original,
        "evaluation_config": agent.cfg,
        "probabilities": "raw model logits with published temperatures; no API rounding",
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Reference evaluation recipe or data changed")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    predictions = args.output / "predictions.jsonl"
    results = (
        [json.loads(line) for line in predictions.read_text().splitlines()]
        if predictions.exists()
        else []
    )
    completed = {row["id"]: row for row in results}
    expected = {row["id"]: row for name, rows in groups.items() if name in names for row in rows}
    if len(completed) != len(results) or not completed.keys() <= expected.keys():
        raise ValueError("Saved reference predictions do not match this evaluation")
    for key, saved in completed.items():
        if saved["target"] != expected[key]["target"]:
            raise ValueError("Saved reference target differs from the evaluation data")
    failures = []
    captured = []

    def capture(module, inputs, outputs):
        captured.append(outputs[0].detach().float().cpu())

    hook = agent.model.register_forward_hook(capture)
    with predictions.open("a", buffering=1) as stream:
        for name, rows in groups.items():
            if name not in names:
                continue
            for row in rows:
                if row["id"] in completed:
                    continue
                state = row["state"]
                kind = row["question"]["type"]
                k = len(row["target"])
                if row["image"]:
                    with Image.open(row["image"]) as image:
                        state = {**state, "image": image.convert("RGB")}
                captured.clear()
                try:
                    response = agent.predict(state, {"q": row["question"]})
                    if len(captured) != 1 or captured[0].shape[0] != 1:
                        raise ValueError("Unexpected upstream forward batching")
                    temperature = agent.temperature_by_options.get(
                        temp_bucket(QTYPES[kind], k), agent.temperature[QTYPES[kind]]
                    )
                    raw = captured[0][0, :k]
                    if len(raw) != k or not torch.isfinite(raw).all():
                        raise ValueError(
                            "Missing candidate logits or non-finite upstream distribution"
                        )
                    logits = (raw / max(1e-3, float(temperature))).tolist()
                    result = {key: row[key] for key in ["id", "dataset", "group", "target"]}
                    result.update(
                        type=kind, logits=logits, raw_logits=raw.tolist(), usage=response["usage"]
                    )
                    stream.write(json.dumps(result) + "\n")
                    results.append(result)
                except (ValueError, RuntimeError) as error:
                    failures.append({"id": row["id"], "dataset": name, "error": str(error)})
            stream.flush()
            print(
                json.dumps(
                    {
                        "dataset": name,
                        "attempted": len(rows),
                        "total_complete": len(results),
                        "failures": len(failures),
                    }
                ),
                flush=True,
            )
    hook.remove()
    (args.output / "report.json").write_text(
        json.dumps(
            {"config": config, "metrics": by_dataset(results), "failures": failures}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
