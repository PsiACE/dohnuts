# Train and evaluate Dohnuts

Use Python 3.12 and a working PyTorch ROCm installation on the RX 7900 XTX.
The recorded training environment is preserved in
`data/manifests/environment-freeze.txt`. Install the ROCm PyTorch wheels before
installing the package so dependency resolution does not select CUDA wheels.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/rocm6.4
uv pip install --python .venv/bin/python -e '.[train,agent]'
.venv/bin/python scripts/run_experiment.py
```

The Qwen3.5 adapter limits PyTorch's caching allocator to 80% of visible VRAM,
leaving headroom for the desktop and GPU driver on a shared workstation.

The script downloads the pinned base and public datasets when absent, converts
the complete mixture, isolates related examples and identical images across
splits, and freezes token eligibility. It trains seed 42 for 3,600 optimizer updates,
resuming saved optimizer and sampling state after interruption.

Development accuracy selects the checkpoint within this run. LoRA is merged,
three temperatures are fitted on the calibration partition, and the complete
held-out partition is evaluated. Test scores never choose a model. The workflow
exports the selected checkpoint directly after calibration and evaluation, then
completes acceptance, quality diagnostics and benchmarks. Variation across seeds
is unmeasured.

The script then runs checkpoint/Bub acceptance, quality diagnostics, local
Laya/Laya Vision references, JevBench's frozen public tasks, and the complete
frozen Laya chart protocol. API benchmarks measure latency and resource use.
Every failed stage stops execution.
Completed stages are recorded; rerunning the command resumes the same recipe.
A different recipe or dataset requires a separate experiment directory. The
current method warms up for 72 updates, follows cosine decay through update
2,400, and uses the 10% learning-rate floor through update 3,600. This is the
schedule used by the selected Dohnuts 0.1.0 export.

`--steps` changes an experiment's total budget. Extending a completed run retains
optimizer and random state, sampling position, and the decay horizon. Development
selection includes its existing best checkpoint, followed by calibration,
evaluation, and acceptance. Checkpoint steps are experiment coordinates, not
additional product versions.

The RLCD controls are `--sigma` and `--ce-weight`, defaulting to 0.3 and
1.0; `--steps` sets the total update budget. `--data`, `--model`, and `--output` identify local assets and output locations.
They do not select architectures or change the fixed method. Different backbone
support uses the [adapter interface](design.md).

Base initialization and checkpoint initialization use the same 26-group mixture
and training workflow. To initialize from an exported checkpoint:

```bash
.venv/bin/python scripts/run_experiment.py --initialize-from runs/v1/checkpoint --output runs/domain
```

The checkpoint supplies the starting parameters and a fresh optimizer/schedule.
The runner automatically resumes a saved update in its output directory, restoring
optimizer and sampling state. Both modes use the same fused kernels, differentiable
prefix sharing, frozen-image cache, RLCD objective, development selection and
calibration. There is no separate incremental training program.

## Outputs

The default data directory is `data/processed/v1`; outputs go to `runs/v1`:

| Artifact | Contents |
| --- | --- |
| `checkpoint/` | Selected LoRA/head weights, base revision, temperatures, selection metadata, checksum |
| `seed-42/resources.jsonl` | Timestamped GPU memory, power, utilization, temperature and process RSS |
| `seed-42/metrics.jsonl` | Loss, reward, development accuracy, learning rate, and GPU memory |
| `seed-42/test-predictions.jsonl` | Stable example IDs, targets, raw logits |
| `seed-42/evaluation.json` | Raw/calibrated per-task, decision-type and candidate-count metrics |
| `acceptance/report.json` | Actual checkpoint reload, decision interface, and Bub SDK acceptance |
| `quality-audit/` | Candidate rotations, image ablations, and batch-size sensitivity |
| `benchmarks/` | End-to-end timing samples and resource measurements |
| `laya-chart/` | Frozen application/language suites, calibration diagnostics and parallel decision timing |
| `jevbench/` | JevBench public-task predictions, raw evidence, and same-ID reference comparisons |
| `metrics/` | CSV tables, comparison JSON, and a self-contained results report |
| `figures/` | Separately rendered comparison figures, chart data, and source checksums |
| `recipe.json` | Fixed settings, exposed RLCD values, and source hashes |

```python
from dohnuts.predictor import Predictor

model = Predictor.from_checkpoint("runs/v1/checkpoint")
```

Render comparison figures from the saved results using the
[plotting instructions](local-benchmarks.md#model-card-figures).

The compact checkpoint requires its pinned base model, which defaults to
`.cache/models/Qwen3.5-0.8B`. Raw data, cached base weights, and generated results
are local artifacts rather than package contents.
