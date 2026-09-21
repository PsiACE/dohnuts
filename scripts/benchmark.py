"""Matched predict() workloads based on pinned Laya / Laya Vision protocols."""

import argparse
import gc
import json
import sys
from functools import partial
from pathlib import Path

import torch
from PIL import Image

from dohnuts.experiment import Sampler, emit, latency_stats, memory, timed


def image_workloads():
    image = Image.new("RGB", (96, 96), "white")
    image.paste(Image.new("RGB", (48, 48), (220, 20, 20)), (24, 24))
    one = {"is_red": {"type": "noul", "instructions": "Is the square red?"}}
    three = dict(
        one,
        color={
            "type": "choice",
            "instructions": "What color?",
            "criteria": ["red", "blue", "green"],
        },
        size={
            "type": "score",
            "instructions": "How big?",
            "criteria": ["small", "medium", "large"],
        },
    )
    for label, state in [("text", "Customer: I was billed twice."), ("image", {"image": image})]:
        yield f"vision_protocol_{label}_1q", state, one
        yield f"vision_protocol_{label}_3q", state, three


def text_workloads():
    """Different questions over one support state, with no duplicate-question shortcut."""
    state = {
        "customer": "Mina",
        "plan": "business",
        "invoice": "4411",
        "currency": "USD",
        "amount": 120,
        "charges": 2,
        "expected_charges": 1,
        "refund_requested": True,
        "deadline": "today",
        "service_available": True,
    }
    specifications = [
        ("customer", ["Mina", "Alex", "Sam", "unknown"]),
        ("plan", ["free", "personal", "business", "enterprise"]),
        ("invoice", ["4400", "4411", "4422", "unknown"]),
        ("currency", ["EUR", "USD", "GBP", "unknown"]),
        ("amount", ["30", "60", "120", "240"]),
        ("charges", ["zero", "one", "two", "three"]),
        ("expected_charges", ["zero", "one", "two", "three"]),
        ("refund_requested", ["yes", "no", "unclear", "contradictory"]),
        ("deadline", ["today", "tomorrow", "next week", "none"]),
        ("service_available", ["yes", "no", "unclear", "contradictory"]),
    ]
    # Different interpretations, not changed IDs on an identical prompt.
    instructions = [
        "What value does the state give for {field}?",
        "Which candidate contradicts the reported {field}?",
        "Which candidate should a support agent preserve when copying {field}?",
        "Which candidate represents the customer's stated {field}?",
        "Which value is supported by the evidence about {field}?",
    ]
    questions = {
        f"field_{field}_interpretation_{j}": {
            "type": "choice",
            "instructions": instruction.format(field=field),
            "criteria": options,
        }
        for j, instruction in enumerate(instructions)
        for field, options in specifications
    }
    for count in [1, 5, 10, 50]:
        yield f"distinct_text_{count}q", state, dict(list(questions.items())[:count])


def measure(args, agent, engine_label):
    model = agent.model
    emit(
        args.output,
        {
            "kind": "upstream_environment",
            "engine": engine_label,
            "dtype": "bf16",
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "config": getattr(agent, "cfg", {"image_pixels": 512**2}),
            "memory": memory(),
            "seed": 0,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "merged_lora": args.engine == "dohnuts" and args.checkpoint is not None,
            "trained_vision": args.engine == "laya-vision",
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "torch_module": torch.__file__,
        },
    )
    for label, state, questions in [*image_workloads(), *text_workloads()]:
        if args.engine == "laya" and isinstance(state, dict) and "image" in state:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        operation = partial(agent.predict, state, questions)
        cold = timed(operation, 1)[0]
        timed(operation, 3)
        with Sampler(Path("/sys/class/drm/card1/device")) as sampler:
            durations = timed(operation, 20)
        result = operation()
        if args.engine == "dohnuts":
            prepared, _, _, _ = agent.prepare(state, questions)
            shape = list(prepared["input_ids"].shape)
            image_grid = prepared.get("image_grid_thw", torch.empty(0)).tolist()
        else:
            shape, image_grid = None, None
        emit(
            args.output,
            {
                "kind": "matched_inference",
                "engine": engine_label,
                "case": label,
                "questions": len(questions),
                "warmup": 3,
                "repeats": 20,
                "cold_ms": cold,
                "dtype": "bf16",
                "end_to_end": latency_stats(durations, len(questions)),
                "usage": result.get("usage"),
                "input_shape": shape,
                "image_grid_thw": image_grid,
                "memory": memory(),
                "telemetry": sampler.summary(),
            },
        )
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["dohnuts", "laya", "laya-vision"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.checkpoint and args.engine != "dohnuts":
        parser.error("--checkpoint currently selects exported Dohnuts weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    if args.engine == "dohnuts":
        from dohnuts.predictor import Predictor

        if args.checkpoint is None:
            parser.error("Dohnuts benchmarks require --checkpoint pointing to trained weights")
        agent = Predictor.from_checkpoint(args.checkpoint)
        model = agent.model
    else:
        sys.path.insert(0, str(Path(".cache/upstream") / args.engine))
        if args.engine == "laya":
            from laya.agent import Agent

            agent = Agent(".cache/models/laya-multilingual", device="cuda")
            agent.dtype = torch.bfloat16
        else:
            from laya.vlm import VLMAgent

            agent = VLMAgent(".cache/models/laya-vision-trained", device="cuda", dtype="bf16")
        model = agent.model
    measure(args, agent, args.engine)
    del model, agent
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": "complete", "engine": args.engine}), flush=True)


if __name__ == "__main__":
    main()
