# Model architecture

Dohnuts accepts a state and independent questions and returns temperature-scaled
probabilities over supplied candidates. The training and serving templates are
shared. A request reuses its causal prefix and computes question suffixes in
parallel. Frozen image features are encoded on a cache miss and shared. More
questions still require more suffix compute.

Each sequence contains the state, question, candidate descriptions, and a
reserved marker after each candidate. A shared scalar head scores the marker's
hidden state. A per-question softmax gives the decision distribution. Nominal
candidate order is shuffled during training; binary and ordinal order are fixed.
Candidate-order sensitivity is measured on held-out data.

## Fixed recipe

The Qwen3.5 adapter freezes the base and vision encoder and trains rank-8
language LoRA plus the scorer. The training recipe fixes precision, image and
sequence budgets, optimizer, learning rates, sampling, checkpoint
selection, and calibration. `recipe.py` is the executable specification.
Run metadata records all values and data hashes, including values that callers
cannot change.

`RLCDConfig(sigma=0.3, ce_weight=1.0)` is the complete RLCD configuration surface.
`sigma` controls exploration in logit space; `ce_weight` controls the joint
cross-entropy term. Four samples, the proper-scoring reward, ordinal distance
penalty, baseline estimator, and numerical safeguards are fixed. `--steps`
controls the total update budget.

The release path merges LoRA before calibration and final evaluation. Loading
the compact checkpoint reconstructs that same merged model from the pinned
base. The caller does not choose merge state, precision, attention backend,
LoRA rank, or calibration mode.

## Model adapters

`Qwen35Adapter` contains the backbone-specific code. Another backbone supplies
an adapter object to `DecisionModel`, `DecisionCollator`, `train`, and
`final_evaluation`; `Predictor.from_checkpoint(..., adapter=adapter)` validates
its identity against the saved artifact.

| Adapter responsibility | Contract |
| --- | --- |
| `name`, `base_model`, `marker`, `image_prefix` | Stable identity, one reserved candidate token, image prompt syntax |
| `processor`, `load`, `hidden_size` | Load the pinned processor/backbone and size the shared scorer |
| `adapt_language`, `merge`, `freeze_vision` | Apply the training and deployment lifecycle |
| `batch_inputs`, `shared_image_inputs` | Preserve all candidates and process shared images once |
| `forward` | Return candidate-addressable hidden states and the shared-prefix position offset |

There is one supplied adapter. Supporting a different backbone requires its
implementation and the same checkpoint/quality acceptance; an arbitrary model
path alone does not establish compatibility. Model-specific tokenization and
kernels belong here, without branching the RLCD objective or public decision API.

The [inference reference](inference.md) describes question types, response fields,
and input limits. [Bub integration](bub-agent.md) exposes the same decision API to agents.

## Behavior and regression tests

`tests/acceptance.py` loads the exported checkpoint and exercises the real
prediction and Bub interfaces. It checks text and image decisions, candidate
probabilities, independent requests, reload, and input limits. The training
workflow runs this acceptance on its exported weights.

The CPU tests in `tests/test_*.py` cover reported metric meanings, temperature
calibration, the pinned RLCD objective and gradients, and the regression where
identical screenshots crossed splits under different source IDs. Run them when
changing the corresponding behavior:

```bash
.venv/bin/python -m unittest discover -s tests
```

Tests assert observable behavior or a known failure case. Internal allocation,
helper structure, log keys, file counts and exact error wording are not contracts.
Quality evaluation separately measures held-out accuracy, calibration,
candidate-order sensitivity and image dependence.
