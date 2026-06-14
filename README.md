# ESM2 L4 Inference Optimization

> **Status: results complete (8 / 9 milestones).** Benchmarks, correctness checks,
> profiling, and the consolidated report are done; only the final repo-cleanup sweep
> (Milestone 9) remains. See [`docs/PLAN.md`](docs/PLAN.md) for the roadmap and
> [`docs/RESULTS.md`](docs/RESULTS.md) for the full numbers.

## TL;DR

Reproducible performance engineering for **ESM2 protein-language-model embedding
extraction** on a single **NVIDIA L4** GPU: honest baselines, profiler evidence,
variable-length batching / padding analysis, `torch.compile` experiments, correctness
checks, and one small custom Triton kernel for masked mean pooling. Target model:
[`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D).

## Why this project exists

Protein-language models like ESM2 are now a standard front-end for biotech ML: you embed
a protein sequence once and reuse the vector for downstream tasks (function prediction,
variant effect, structure, retrieval). The embedding step is pure inference and is run
over millions of sequences, so its throughput and cost matter. This project documents the
unglamorous-but-real engineering of making that step fast *and* correct on commodity
inference hardware (the L4 is a common, power-efficient inference GPU).

## Why ESM2 and protein embeddings

- ESM2 is open, widely used, and available through Hugging Face `transformers`.
- The workload is representative: variable-length sequences, padding, attention,
  per-token hidden states, and a pooling step to get one vector per protein.
- It exposes classic perf themes: dtype choice, batching, padding waste, static vs
  dynamic shapes, kernel fusion.

## Hardware

Single NVIDIA L4 (24 GB, Ada / compute capability 8.9) on an AWS `g6.2xlarge`. Exact
software versions are captured programmatically — see
[`results/raw/environment.txt`](results/raw/environment.txt).

## Methods

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Benchmarks use CUDA events, explicit
warmup, correct synchronization, deterministic synthetic sequences, graceful OOM capture,
and timestamped CSV outputs. Throughput is reported as both padded-compute tokens/sec and
**real amino-acid tokens/sec**.

## Results

Full tables (generated from the raw CSVs by
[`scripts/07_summarize_results.py`](scripts/07_summarize_results.py)) are in
[`docs/RESULTS.md`](docs/RESULTS.md). Headlines on the L4, bf16, `esm2_t33_650M`:

| Experiment | Result | Honest caveat |
|------------|--------|---------------|
| Baseline | ~22k real tokens/sec, ~2.0 GB of 24 GB peak | throughput saturates by ~batch 8 |
| Length-sorted batching | **~1.7× real-token throughput**, padding waste 41–44% → 2–8% | same padded-compute rate; it just wastes less |
| `torch.compile` | **1.3×–2.2×** steady-state | **~28 s cold-start per shape**; recompiles per shape |
| Triton pooling kernel | fastest in all 6 shapes, ~1e-8 accurate | pooling is <0.1% of runtime → no end-to-end gain |
| Profiling | ~42% matmul / ~43% unfused elementwise | the elementwise share is what `torch.compile` fuses |

The cheapest, most portable win is **length-sorted batching**; the most instructive negative
result is that the **Triton kernel, though correct and fast, doesn't move end-to-end latency**.

## Correctness

Optimized paths are validated against an fp32 eager reference with documented numerical
tolerances. Exact equality is not expected across dtypes/kernels; tolerances and reasoning
are in the correctness milestone.

## Profiling

A `torch.profiler` kernel-level breakdown of the encoder forward (bf16, seq 512 × batch 8)
attributes time to leaf CUDA kernels so the self-times **sum to the measured 225 ms/iter**
(no operator/kernel double-counting). The split is ~42% matmul — dominated by the **FFN
linears, not attention** (flash-attention is only ~4.7%) — and ~43% **unfused elementwise**
(GeLU, bias/residual adds, scalings). That elementwise share is precisely the fusion headroom
that explains the `torch.compile` speedup. Full write-up in
[`docs/PROFILING.md`](docs/PROFILING.md).

## What improved performance

- **Length-sorted batching — ~1.7×, and free.** No kernel or precision change; just group
  similar-length sequences so the GPU stops computing over padding. The best return on effort.
- **bf16** — highest throughput of the precisions tested, half the memory, pooled-embedding
  cosine ≥ 0.9996 vs fp32.
- **`torch.compile` (1.3×–2.2×)** — a genuine steady-state win for **fixed-shape, high-volume**
  serving, once the cold-start compile is amortized.
- **The Triton pooling kernel** — fastest pooling in every shape and ~1e-8 accurate (in
  isolation; see below).

## What did not improve performance

Negative results are reported as plainly as the wins:

- **The Triton kernel does not change end-to-end latency** — pooling is ~0.08 ms against a
  25–225 ms encoder forward (<0.1%). It is a correctness/kernel-authoring demonstration.
- **`torch.compile` cold start is ~28 s per shape** — with static shapes a many-shape workload
  recompiles constantly and can be net-slower than eager.
- **`einsum`/compiled pooling are *slower* than plain PyTorch** at small shapes (~0.74×).
- **Throughput saturates by ~batch 8** — larger batches at long sequences *reduce* real-token
  throughput; the L4 is compute-bound, not memory-bound, for this model.

## Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Reproduce

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). In brief:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/00_check_environment.py
```

## Future work

- **Dynamic-shape `torch.compile`** (`dynamic=True`) and shape bucketing to amortize the
  cold-start cost across a realistic length distribution.
- **Fuse the pooling into the encoder's final block** so the Triton kernel's accuracy benefit
  lands without a separate launch — the only way the pooling work becomes end-to-end relevant.
- **FP8 inference** on the L4 (Ada has FP8 tensor cores) with a correctness study against the
  fp32 reference, mirroring the bf16/fp16 analysis here.
- **Compare against a vendor-optimized stack** (TensorRT / TensorRT-LLM) as an explicit,
  honest baseline rather than leaving it out of scope.
- **Real proteome length distributions** (e.g. a UniRef sample) instead of synthetic
  sequences, to validate the batching win on production-like data.

## What this project does not claim

- It does not claim to beat vendor-optimized inference stacks (e.g. TensorRT, FlashAttention
  kernels) — those comparisons are out of scope unless explicitly benchmarked.
- It does not claim the custom Triton kernel is universally faster than PyTorch; results are
  reported as measured.
- Numbers are specific to this L4 + software stack and synthetic sequence distribution.
