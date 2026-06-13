# Human review: Profiling notes

## TL;DR

The seventh major milestone — *profiling workflow complete* — is done.
`scripts/06_profile_pytorch.py` profiles the ESM2-650M encoder forward (bf16, seq 512, batch 8)
with `torch.profiler` and reports a kernel-level breakdown whose self-times **sum to 225 ms/iter,
matching the Milestone 2 latency**. The forward is ~42% matmul (FFN/projection linears dominate;
flash-attention only ~4.7%), ~43% unfused elementwise, ~14% copies/casts, ~1% LayerNorm — and the
elementwise share is the headroom that explains the Milestone 5 compile win. Write-up in
`docs/PROFILING.md`; data in `results/raw/profile_20260613T171924Z.csv`. Nothing pushed.

## What changed

- `scripts/06_profile_pytorch.py` — kernel-level profiler + category rollup.
- `docs/PROFILING.md` — the profiling write-up.
- `docs/milestones/07-profiling.md`, `docs/METHODOLOGY.md` (profiling note),
  `docs/commits/0008-profiling.md`.
- `results/raw/profile_20260613T171924Z.csv` — top-15 kernels.

## What I should understand before continuing

- **Why a kernel-only view.** `torch.profiler`'s `key_averages()` lists both the `aten::`
  operator events and the device kernels they launch. Summing both double-counts GPU time (the
  first run read ~449 ms/iter ≈ 2×). Filtering to leaf CUDA kernels makes the self-times sum to
  the *measured* forward time — that equality is the proof the breakdown is honest.
- **What the breakdown says.** Matmul ≈ 42% (two GEMM families = the linears; flash-attention is
  only ~4.7%, so attention is not the bottleneck at this shape), unfused elementwise ≈ 43%,
  copies/casts ≈ 14% (rotary `CatArrayBatchedCopy` + dtype casts), LayerNorm ~1%.
- **It connects to Milestone 5.** The ~43% unfused elementwise (each GeLU/bias/residual/scale a
  separate HBM round-trip) is exactly what `torch.compile` fuses — which is *why* compile gave
  1.63× at this same shape. M7 is the explanation behind the M5 number.
- **The categories are approximate.** Name-substring buckets; trust the matmul-vs-rest headline,
  treat the finer buckets as order-of-magnitude.

## Commands I should run manually

```bash
# Reproduce the profile (a few seconds)
.venv/bin/python scripts/06_profile_pytorch.py --dtype bf16 --seq-len 512 --batch-size 8

# Export a Chrome trace to view in chrome://tracing or perfetto (large; do NOT commit)
.venv/bin/python scripts/06_profile_pytorch.py --trace /tmp/esm2_trace.json

# Inspect the committed top-ops CSV
column -s, -t results/raw/profile_20260613T171924Z.csv | cut -c1-120 | less -S
```

## Questions I should be able to answer

- How do I know the kernel breakdown is correct? (The leaf-kernel self-times sum to 225 ms/iter,
  equal to the Milestone 2 median for this shape — the forward is serial on one stream.)
- Why did the first attempt read ~449 ms/iter? (Double counting: `key_averages()` includes both
  operator and kernel events; the fix filters to leaf kernels.)
- Is attention the bottleneck? (No — fused flash-attention is ~4.7% here; the FFN/projection
  linears dominate the matmul cost.)
- What does the ~43% elementwise mean for optimization? (It is unfused activation/bias/residual
  traffic — the fusion target `torch.compile` exploited in Milestone 5.)

## Possible bugs or misleading benchmark artifacts

- **Double counting** (fixed). If anyone reverts to summing all `key_averages()` rows, GPU time
  roughly doubles. The kernel-only filter is load-bearing.
- **Heuristic categories.** "elementwise" bundles activation/bias/residual/scale; "copy" bundles
  casts and concatenations. Don't quote the sub-buckets to two significant figures.
- **Eager, one shape, one stack.** A compiled run, a different shape, or a different GPU gives a
  materially different split.
- **Chrome trace is large** — it is `--trace`-gated and must not be committed.

## Human notes

_Add review outcome / approval here before Milestone 8 begins._
