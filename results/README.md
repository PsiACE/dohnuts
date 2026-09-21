# Dohnuts evaluation results

These tables describe **Dohnuts-0.1.0-0.8B**, seed 42, selected at update 3,600.
The [manifest](manifest.json) records the base revision, decision-weight checksum,
calibration, dataset-partition hashes, and checksums of the published tables.
All measurements come from one AMD Radeon RX 7900 XTX (24 GB).

| File | Contents |
| --- | --- |
| [training.csv](training.csv) | Sampled training loss, reward, batch accuracy, learning rate, timing, and GPU memory |
| [development.csv](development.csv) | Per-group development metrics at selection checkpoints |
| [quality.csv](quality.csv) | Held-out per-group accuracy, soft accuracy, F1, NLL, Brier, ECE, and decision-type metrics |
| [reliability.csv](reliability.csv) | Calibration-bin counts, observed accuracy, and mean predicted probability |
| [latency-samples.csv](latency-samples.csv) | Individual synchronized inference durations and memory/telemetry summaries |
| [training-resources.csv](training-resources.csv) | Training GPU memory, utilization, power, temperature, and process RSS samples |

## Reading the tables

Accuracy and rates use the range 0–1. Overall quality is an unweighted mean over
dataset groups. Blank CSV fields indicate metrics that do not apply to that
decision type. Schema pass rate checks valid responses; it is not task accuracy.

Training `step` is the optimizer update. Training accuracy describes sampled
minibatches, not held-out performance. `policy_loss` and `ce_loss` are the two
terms of the joint objective; policy loss can be negative. Resume events can
restart elapsed-time counters and repeat a logged step. Preserve row order when
examining timing, and use optimizer steps for learning curves. Training memory
columns use GiB and describe PyTorch allocations, separately from board VRAM.

`quality.csv` and `reliability.csv` retain both uncalibrated and calibrated
measurements. Calibration temperatures are fitted on the independent calibration
partition. ECE uses 15 fixed bins; confidence in these tables means maximum
predicted probability. It is different from the API's entropy-based field.

Latency samples include preprocessing, transfers, model execution, and response
assembly. They use BF16, three warmups, and 20 synchronized repetitions per
workload; image timings use warm caches. Network and queueing are excluded.
Reference engine `laya` denotes Laya multilingual. `input_shape` is JSON inside
the CSV; all memory fields retain their unit suffixes.

Resource samples use Unix seconds, bytes, percent, microwatts, and millidegrees
Celsius as indicated by the column names. Board VRAM includes the desktop and
runtime; utilization alone does not measure hardware efficiency.

The [comparison figures](../docs/figures/README.md) include JevBench and Laya task
suites, with numerical [chart values](../docs/figures/chart-data.csv). Consult the
[model card](../MODEL_CARD.md), [data protocol](../docs/data-and-evaluation.md), and
[benchmark protocol](../docs/local-benchmarks.md) for evaluation scope and limits.

These are numerical measurements. They do not contain source documents, images,
private conversations, or training-machine paths.
