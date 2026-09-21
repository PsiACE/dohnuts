# Latency and resource measurement

The benchmark runs the actual exported, merged checkpoint and the pinned Laya multilingual
and Laya Vision checkpoints on one RX 7900 XTX. Each engine receives
the same state and questions. Tokenizers, model sizes, and internal templates
differ; this is an API workload comparison.

The local protocol uses BF16, eight CPU threads, three warm-up calls, and
20 synchronized repetitions. End-to-end timing includes preprocessing, transfers,
GPU execution, and response assembly. It excludes network and request queues.
Report median, linearly interpolated p95, raw durations, decisions per second,
and amortized latency per decision.

`scripts/benchmark.py` uses one fixed suite: Laya Vision's one/three-question
text and image cases, plus 1, 5, 10 and 50 distinct questions over a shared
support state. Each reference uses its checkpoint's published preprocessing.

PyTorch allocated, reserved, and peak memory are recorded separately from board
VRAM. Board VRAM includes the runtime and desktop. Power, utilization, temperature,
and process RSS accompany timing samples. Shared-prefix execution does not imply
constant latency as batch size grows or predict performance under queued load.

The release workflow writes timing samples to `runs/v1/benchmarks/` and
plotting data to `runs/v1/metrics/latency-samples.csv`. Quality, resource use,
and latency are reported separately. Weight-preserving changes still require
held-out quality checks because BF16 arithmetic need not be bitwise identical.

## JevBench

The release workflow also evaluates the exported checkpoint with the official
[JevBench v1.2.2 harness](https://github.com/fstandhartinger/jevbench/tree/v1.2.2),
pinned at `e105a48f8cdb7f3babb3594424f73e5d7bdc97b9`. Its public tasks contain
48 easy, 72 standard, and 111 hard decisions. The other 303 decisions in the
534-item leaderboard are not distributed, including the entire judge tier.

```bash
.venv/bin/python scripts/run_jevbench.py
```

Each task receives one serial `Predictor.predict` call with its original state,
instructions, criteria, and candidate order. Returned probabilities are native
model distributions. The official runner stores requests, responses, failures,
latencies, and scores. Model loading is timed separately; the first decision is
retained and reported. The adapter's 4,096-token inference limit applies: rejected inputs
count as wrong and are not silently truncated or removed from the denominator.
The benchmark never changes weights, temperatures, or the training recipe.

`runs/v1/jevbench/` contains raw evidence, a JSON summary, a Markdown report,
and `same-public-ids.csv`. Reference accuracy is recomputed from upstream's
published outcomes on the same 231 IDs. The report distinguishes this partial
evaluation from the full leaderboard. It records raw local latency and the
upstream assumed latency adjustment separately. Local execution has no provider
tariff, so cost and the four-axis composite remain unmeasured.

## Laya comparison-chart protocols

The official Laya research branch at `28d43add7e47ce502489c9433310d55276c64e0f`
contains the builders for the application, multilingual, calibration, and latency
measurements in its comparison chart. Run them on the exported Dohnuts checkpoint:

```bash
git clone --branch research https://github.com/NandhaKishorM/laya.git .cache/upstream/laya-research
git -C .cache/upstream/laya-research checkout --detach 28d43add7e47ce502489c9433310d55276c64e0f
uv pip install --python .venv/bin/python datasets==5.0.1
.venv/bin/python scripts/prepare_laya_benchmark.py
.venv/bin/python scripts/run_laya_benchmark.py
```

Preparation preserves the upstream builders, seed 13, candidate order, and
document shortening, and records dataset revisions and file hashes. It requires
all ten application suites, all fifty Colab suites, and all 51 MASSIVE languages
to load successfully. The fixed questions are stored before model evaluation.

Results live in `runs/v1/laya-chart/`: raw logits and probabilities,
per-suite CSV, all language scores, calibration diagnostics, option permutations,
and end-to-end timing samples. Temperature refitting follows the upstream
per-suite first-half/second-half protocol and never updates deployment temperatures.
The 51-language sweep uses 100 cases per language and 20 candidates; the Colab
language suites use 300 cases per language. These differ from the main mixture's
full-label-space evaluation.

Published Laya reference columns are historical measurements, not newly run
checkpoints. Their original dataset revisions and per-example hashes are absent,
so reproducing the builders does not independently establish historical byte
identity. Calibration comparisons use the shared 49 suites because the upstream
result omits BANKING77. Typed-decisions soft accuracy is the inner product of
predicted and teacher distributions, following this benchmark's definition.

Local latency uses the RX 7900 XTX, not the chart's T4. Mixed-language measurements
use the upstream English/Hindi request stream and one resident Dohnuts model; no
checkpoint routing or swapping is added. Local inference has a compute cost even
when no API fee is paid.

## Model card figures

The [comparison gallery](figures/README.md) contains the selected model's figures
and chart values. To regenerate them, use the saved evaluation results below.

Use the saved results to render six comparisons: JevBench accuracy, application
accuracy, multilingual accuracy, vision accuracy and calibration, local latency,
and paired JevBench outcomes. The reference checkpoints are Laya multilingual
and Laya Vision. The English Laya entry in JevBench's published results is a
different checkpoint and must not be relabeled as Laya multilingual.

Install plotting tools in a separate environment to keep the training runtime
unchanged, then render the completed run:

```bash
uv venv --python 3.12 .cache/plot-venv
uv pip install --python .cache/plot-venv/bin/python matplotlib==3.11.2 numpy==2.5.3
.cache/plot-venv/bin/python scripts/plot_model_card.py --run runs/v1
```

`runs/v1/figures/README.md` indexes the PNG, SVG and PDF figures. The directory
also contains a combined PDF, chart values in CSV, and a manifest of source
checksums. Captions identify the selected checkpoint, dataset coverage,
measurement conditions, and whether references were measured locally or taken
from published results. Model identity and the shared project version come from the
selected checkpoint metadata.
