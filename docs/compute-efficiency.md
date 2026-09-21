# Shared training and inference execution

Dohnuts has one execution policy: differentiable FLA fusion, shared causal prefixes,
a bounded frozen-image cache, and asynchronous input/output transfer. These are
the implemented defaults, with no public backend, stream-count, or cache switches.

## Runtime

Qwen uses fused RMSNorm, RMSNorm with output gating, and SwiGLU in both training
and inference. LoRA projection modules remain in the autograd graph; deployment
merges them into the frozen base. Vision patch projection uses a linear map and
causal convolution uses the installed Triton kernel. These kernels preserve the
mathematical operations, but BF16 rounding need not match an unfused execution.
Train, development selection, calibration and test all use this runtime.

When questions share a sufficiently long causal prefix, the model computes that
prefix once and forks both attention KV and delta-rule convolution/recurrent
states into independent question branches. All candidate positions remain in
the suffix. The split is aligned to 64 tokens. A request uses one batched model
invocation: internally the language backbone processes a shared prefix and
then parallel suffixes. It never generates an autoregressive token stream.
Requests without a reusable prefix use a single batched language pass.

For B questions of length L sharing P tokens, dense projection work is
proportional to `P + B × (L − P)` rather than `B × L`. Attention, recurrent state
forking and launch overhead must still be measured; the token ratio is not an
end-to-end speedup guarantee.

The prefix cache belongs to the current request and weight snapshot. Tensor
replacement preserves the autograd history. Training recomputes the complete
shared request during activation checkpointing, constructing fresh cache state.
Gradients from all branches accumulate into the shared prefix. No language state
survives an optimizer update or another request.

Frozen image features use a model-local 128 MiB LRU keyed by decoded RGB content
and image dimensions under the adapter's fixed preprocessing policy. Batch
misses are encoded together, duplicate images once, and the same features serve
training and prediction. They remain valid across LoRA updates because neither
vision weights nor preprocessing changes. Loading a model creates a fresh cache.

CPU workers prepare pinned batches. One transfer stream prefetches the next
batch while the compute stream executes the current batch. Events preserve
batch order and memory lifetime. Evaluation overlaps result transfer and JSON
serialization with the following forward. The optimizer sees four microbatches
at the same parameter snapshot, averages their loss, clips once and updates once.

## Training lifecycle

The training mixture and trainer serve base initialization and checkpoint
initialization. `--initialize-from` supplies verified unmerged LoRA/head weights;
the data, RLCD objective, schedule, selection and evaluation stages are identical.
`--resume` restores the current run's optimizer, random state and update count.
The development baseline at step zero is an eligible selected checkpoint.

RLCD optimizes the policy produced by the fused runtime. It does not replace
correct dependency handling or held-out quality measurement. Calibration follows
LoRA merge on an independent partition. Test and benchmark scores never select
weights or tune temperatures.

## Hardware bounds

The RX 7900 XTX has 96 compute units, a reference boost frequency of 2.5 GHz,
24 GB GDDR6 and 960 GB/s external memory bandwidth. AMD documents 512 BF16
matrix FLOPs per clock per compute unit. The dense BF16 matrix ceiling is
therefore `96 × 2.5e9 × 512 = 122.88e12 FLOP/s`, counting a multiply-add as
two operations. Its 80% target is 98.304 TFLOP/s. The corresponding bandwidth
target is 768 GB/s. These are different constraints, not two simultaneously
required utilization readings.

Sources: [AMD device specifications](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html)
and [AMD WMMA architecture](https://gpuopen.com/learn/wmma_on_rdna3/).
The advertised cache-adjusted graphics bandwidth is not the GDDR bandwidth
available for streaming arbitrary model tensors. Sustained clocks and kernel
shape efficiency must be measured on the actual machine.

For useful arithmetic `F`, necessary memory traffic `Q`, compute ceiling `P`
and bandwidth `W`, an optimistic device lower bound is:

```text
T_lower = max(F / P, Q / W)
throughput_ceiling = decisions / T_lower
efficiency = measured_throughput / throughput_ceiling = T_lower / T_measured
80% throughput target: T_measured <= T_lower / 0.8
```

A mixed graph needs separate BF16 matrix, FP32/vector, recurrent and transfer
bounds, with dependencies between stages. Summing every device/transfer bound
assumes no overlap; taking just one global maximum assumes ideal overlap. A
critical-path model sits between these extremes. CPU preparation, queueing,
serialization and cold compilation also belong in end-to-end measurements.

The arithmetic count must exclude unused vocabulary projections: Dohnuts gathers
candidate hidden states into a scalar head and never computes token-generation
logits. Embedding storage is not a dense matrix multiplication per token. Image
encoding is counted only when an image is present. Reused prefixes or frozen
image features reduce useful work and require a new bound. Padding and repeated
work must be reported separately; increasing them cannot improve the useful-work
efficiency claim.

Projection-only bounds omit attention, delta-rule recurrence, activations,
normalization and vision. They cannot establish whole-model efficiency or an
exact training ETA. Hardware microbenchmark throughput is not API throughput.

## Measurement

```bash
pdm run python scripts/benchmark.py dohnuts --checkpoint runs/v1/checkpoint --output runs/v1/benchmarks/dohnuts.jsonl
```

The benchmark measures complete prediction calls over text and image workloads,
including preprocessing and transfers. It records raw timings, throughput,
memory and board telemetry. See the [measurement protocol](local-benchmarks.md)
for warmup, precision, batch sizes and cache conditions.

Training records loss, development quality and resource samples. Held-out
evaluation establishes model quality. Eighty percent of the full workload
roofline remains an engineering target that requires measurement.
