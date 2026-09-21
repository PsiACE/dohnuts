# Installation and inference

The measured runtime uses Python 3.12, PyTorch 2.9.1 with ROCm 6.4, and an
AMD Radeon RX 7900 XTX. Install [PDM](https://pdm-project.org/en/latest/#installation)
2.29.2 or a newer 2.x release, then install from source:

```bash
git clone https://github.com/PsiACE/dohnuts.git
cd dohnuts
pdm use 3.12
pdm install --check --prod
```

The lock file targets Python 3.12. The package sources in `pyproject.toml` bind
PyTorch, torchvision, and PyTorch's Triton runtime to the ROCm 6.4 index.
Keep PDM's native resolver enabled: its experimental uv resolver does not support
these package-to-index bindings. See [development](development.md) for dependency
groups, checks, and updating the lock.

## Load a checkpoint

Load the model by its Hugging Face repository ID. The loader downloads the
decision weights and the exact Qwen3.5-0.8B revision recorded in the checkpoint.
Downloads use the Hugging Face cache and are reused by later calls. Both
repositories are public; downloading them does not require a Hugging Face login.

Run Python examples with `pdm run python` from the repository root:

```python
from dohnuts.predictor import Predictor

model = Predictor.from_checkpoint("PsiACE/Dohnuts-0.1.0-0.8B")
```

The loader verifies the checkpoint's weight checksum and base revision, merges
LoRA, and applies its saved calibration temperatures. The checkpoint contains
the decision head as well as LoRA; load it through Dohnuts. It is not a standalone
Transformers language model or a standard PEFT adapter export.

The same method accepts a local checkpoint directory. An optional `revision`
pins a Hub commit, and `base_model` supplies an existing local base with its
recorded `revision.txt`. `model.metadata` exposes the loaded model identity,
weight checksum, and calibration. Weights are not bundled with the Python package.

## Ask questions

Pass a shared state and a mapping of question names to question definitions:

```python
result = model.predict(
    {"message": "The replacement arrived, but it is broken too."},
    {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"support": "Product issues", "billing": "Payment issues"},
        },
        "repeat_issue": {
            "type": "noul",
            "instructions": "Has the customer experienced this problem before?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently does this request need attention?",
            "criteria": ["routine", "soon", "immediate"],
        },
    },
)

answers = result["answers"]
```

Questions are independent: each sees the state and its own instructions and
candidates. A question cannot refer to another question's answer in the same call.

| Type | Criteria | Answer fields |
| --- | --- | --- |
| `choice` | Candidate list, or label-to-description mapping | `choice` and `probabilities`, keyed by candidate label |
| `noul` | Optional descriptions under `false` and `true` | `noul`, the estimated probability that the statement is true |
| `score` | Level descriptions in ascending order | `score`, the expected zero-based level, and `probabilities`, keyed by level index |

Each answer also includes `type` and `confidence`. For `choice` and `score`,
confidence is normalized inverse entropy, `1 - H(p) / log(K)`; for `noul`, it is
`max(p, 1 - p)`. This describes how concentrated the distribution is. It is not
an empirical probability that the decision is correct.

The response includes `model` and `usage` alongside `answers`. Usage records
input tokens and image count.

## Include an image

Supply one decoded PIL image under the state's `image` key:

```python
from PIL import Image

with Image.open("parcel.jpg") as image:
    result = model.predict(
        {"image": image.convert("RGB")},
        {"damage": {"type": "noul", "instructions": "Is the packaging damaged?"}},
    )
```

Each question supports 2–128 candidates and a 4,096-token input budget,
including state, instructions, candidates, and image tokens. Inputs that exceed
the budget are rejected; shorten the state or candidate descriptions and retry.
Image features and the shared input prefix are reused where possible. More
questions still require more computation.
