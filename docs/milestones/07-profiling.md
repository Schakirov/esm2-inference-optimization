# Milestone 7: Profiling notes

## TL;DR

`scripts/06_profile_pytorch.py` runs the ESM2-650M encoder under `torch.profiler` (bf16, seq
512, batch 8) and reports a kernel-level breakdown of the forward. The kernel self-times sum to
**225 ms/iter — matching the Milestone 2 latency**, validating the attribution. Time splits
~evenly between **matmul (~42%)** — FFN/projection linears dominate, flash-attention is only
~4.7% — and **unfused elementwise (~43%)**, with copies/casts ~14% and LayerNorm ~1%. The big
elementwise share is the fusion headroom that explains the Milestone 5 `torch.compile` win. Full
write-up in [`docs/PROFILING.md`](../PROFILING.md); data in
`results/raw/profile_20260613T171924Z.csv`.

## Short explanation

The profiler lists both operator and device-kernel events; summing both double-counts GPU time
(the first run read ~449 ms/iter ≈ 2×). The script filters to leaf CUDA kernels so the self-times
sum to the true forward time, ranks the top kernels, and rolls them up into coarse categories
(matmul / elementwise / copy / softmax-norm). Only the small top-ops CSV is committed; the full
Chrome trace is optional (`--trace`) and not committed.

## Longer explanation

### Files added / changed

- `scripts/06_profile_pytorch.py` — CLI profiler (`--dtype`, `--seq-len`, `--batch-size`,
  `--warmup`, `--iters`, `--topk`, `--trace`, `--output`); kernel-level view + category rollup.
- `docs/PROFILING.md` — the profiling write-up (TL;DR, method, full breakdown, limitations).
- `docs/METHODOLOGY.md` — short note on the kernel-level (no double-count) profiling approach.
- `results/raw/profile_20260613T171924Z.csv` — top-15 kernels for the profiled shape.

### Key findings

- **The breakdown is trustworthy.** Leaf-kernel self-times sum to 225 ms/iter, equal to the
  Milestone 2 median for seq 512 / batch 8 — the forward is essentially serial on one stream.
- **Matmul ≈ 42%, and it's the linears, not attention.** Two GEMM families (cutlass 24.9% +
  ampere 12.6%) are the FFN/projection matmuls; the fused flash-attention kernel is only ~4.7%.
- **Unfused elementwise ≈ 43%.** GeLU, bias-adds, residual adds, and scalings are each their own
  kernel streaming activations through HBM — the exact work `torch.compile` fuses, which is *why*
  Milestone 5 saw 1.63× at this shape. The profile explains the compile result.
- **Copies/casts ≈ 14%** (rotary-embedding `CatArrayBatchedCopy`, dtype casts); LayerNorm ~1%.
- **Not capacity-bound:** ~1.4 GB of 24 GB.

## Validation

- `python scripts/06_profile_pytorch.py --dtype bf16 --seq-len 512 --batch-size 8` → ran cleanly;
  kernel self-times sum to 225.1 ms/iter (matches Milestone 2), wrote a 15-row CSV.
- The double-counting trap (operator + kernel events) was found and fixed: the kernel-only sum
  equals the measured forward time, the operator-inclusive sum was ~2×.
- `pytest -q` → unchanged (no library code touched; 53 pass).

## Limitations

- The category rollup is heuristic (kernel-name substrings); the matmul-vs-rest split is robust,
  the finer buckets are approximate. Detailed in `docs/PROFILING.md`.
- Eager-only, single shape (seq 512 / batch 8), this L4 + PyTorch 2.12.0+cu130 stack — the split
  shifts with compile, shape, and hardware.

## Next steps

Milestone 8 — result processing and the portfolio README: summarize all milestone CSVs into
`docs/RESULTS.md` and the README. **Awaiting human approval before starting.**
