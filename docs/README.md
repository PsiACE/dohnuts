# Dohnuts documentation

## Use Dohnuts

- [Installation and inference](inference.md): load a checkpoint and make text or image decisions.
- [Agent integration](bub-agent.md): expose decisions through the Bub SDK.
- [Training and evaluation](run-experiment.md): run the complete workflow or resume training.

## Understand the model

- [Model card](../MODEL_CARD.md): capabilities, training, measured results, and limitations.
- [Benchmark comparisons](figures/README.md): Jev, Laya multilingual, and Laya Vision results.
- [Architecture](design.md): candidate scoring, shared computation, and model adapters.
- [RLCD](rlcd.md): objective, sampling, and calibration.
- [Compute efficiency](compute-efficiency.md): fused execution and the shared training/inference path.

## Reproduce the measurements

- [Data and evaluation](data-and-evaluation.md): datasets, partitions, and quality metrics.
- [Reference protocols](upstream-alignment.md): Laya and Laya Vision checkpoints and comparison scope.
- [Benchmarking](local-benchmarks.md): latency, resource measurements, JevBench, and figure generation.

The [behavior and regression tests](design.md#behavior-and-regression-tests)
cover the prediction interface and known failure cases.

For website contributors, see [Build the website](website.md).

```{toctree}
:hidden:

Installation and inference <inference>
Agent integration <bub-agent>
Training and evaluation <run-experiment>
Build the website <website>
```
