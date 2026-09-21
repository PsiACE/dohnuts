<img src="assets/dohnuts-logo.png" width="160" align="right" alt="A pink-frosted doughnut doing a facepalm">

# Dohnuts

**D'oh! But dohnuts.**

Dohnuts builds small multimodal models for direct decisions. Give the model a
message, a document, or an image, and ask it to choose, judge, or score. It returns
probabilities in a single forward pass, with multiple questions sharing the same
input. That is our take on System 1.

The Python toolkit covers training, inference, evaluation, and agent integration.
The 0.8B model runs locally on a consumer GPU.

[Checkpoint (restricted access)](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B) · [Model card](MODEL_CARD.md) · [Documentation](docs/README.md) · [Benchmarks](docs/figures/README.md)

## One message, several decisions

The Hub example requires an account with access to the private checkpoint
repository. Locally trained checkpoints use the same loader.

Route a support request and check whether it asks for a refund in the same call:

```python
from dohnuts.predictor import Predictor

model = Predictor.from_checkpoint("PsiACE/Dohnuts-0.1.0-0.8B")
result = model.predict(
    {"message": "I was charged twice. Please refund the duplicate."},
    {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle the request?",
            "criteria": ["billing", "technical support", "sales"],
        },
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    },
)

print(result["answers"]["route"]["choice"])
print(result["answers"]["refund"]["noul"])
```

Each answer has its own distribution. Use `choice` to select a candidate, `noul`
to estimate whether a statement is true, and `score` to rate something on an
ordered scale. Add a PIL image to the state for visual questions.

See [installation and inference](docs/inference.md) for setup, checkpoint loading,
and response fields. The loader downloads and caches the checkpoint and its
pinned base model, then merges LoRA and applies the saved calibration.

## Train, evaluate, integrate

One workflow prepares the public-data mixture, trains with joint RLCD and
cross-entropy, selects a checkpoint, calibrates it, and runs the evaluations:

```bash
.venv/bin/python scripts/run_experiment.py
```

The [training guide](docs/run-experiment.md) covers setup and resuming a run.
The [model card](MODEL_CARD.md) reports quality, latency, and limitations, with
comparisons against Jev, Laya multilingual, and Laya Vision. For agents,
[Bub integration](docs/bub-agent.md) exposes the same interface as one decision
tool through the Bub SDK.

## Built at home

We developed, trained, calibrated, and evaluated Dohnuts on a home PC with one
AMD Radeon RX 7900 XTX (24 GB). We used hardware we already owned and a $0
additional project budget. The main ingredient was time.

## License

The code is licensed under [Apache-2.0](LICENSE). The
[model card](MODEL_CARD.md#license) describes the separate checkpoint terms
and training-data restrictions.
