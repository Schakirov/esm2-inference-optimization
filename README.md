# ESM2 L4 Inference Optimization

> **Status: work in progress.** This README is a draft scaffold; it is filled in to
> portfolio quality in the final milestone. See [`docs/PLAN.md`](docs/PLAN.md) for the
> full roadmap.

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

To be populated — see [`docs/RESULTS.md`](docs/RESULTS.md).

## Correctness

Optimized paths are validated against an fp32 eager reference with documented numerical
tolerances. Exact equality is not expected across dtypes/kernels; tolerances and reasoning
are in the correctness milestone.

## Profiling

To be populated — see `docs/PROFILING.md`.

## What improved performance

To be populated.

## What did not improve performance

To be populated. Negative results are reported honestly.

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

To be populated.

## What this project does not claim

- It does not claim to beat vendor-optimized inference stacks (e.g. TensorRT, FlashAttention
  kernels) — those comparisons are out of scope unless explicitly benchmarked.
- It does not claim the custom Triton kernel is universally faster than PyTorch; results are
  reported as measured.
- Numbers are specific to this L4 + software stack and synthetic sequence distribution.
