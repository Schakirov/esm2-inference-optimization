# Commit 7: Triton masked-mean-pooling kernel

## TL;DR

Add a hand-written Triton masked-mean-pooling kernel (`esm2_perf.triton_pooling`), a
`POOLING_COLUMNS` schema, 8 GPU tests against the Milestone 3 reference, and a 4-impl
benchmark. The kernel is the fastest pooling path in all 6 shapes (1.3×–5.4× vs eager
vectorized PyTorch; 1.1×–1.6× vs the best PyTorch option) and near-exact (err ~1e-8 vs ~1e-3),
though pooling is a negligible fraction of encoder time.

## Short explanation

The kernel reduces over the sequence axis in fp32 in a single fused pass (grid `(B, H/BLOCK_H)`,
streaming `T` in `BLOCK_T` chunks), clamping the token count so all-padding rows pool to zero.
It is validated against the per-sequence reference oracle, then benchmarked against vectorized,
`einsum`, and `torch.compile`d PyTorch pooling with correctness recorded alongside latency.

## Longer explanation

Files changed:

- `src/esm2_perf/triton_pooling.py` — `triton_masked_mean_pool` + `@triton.jit` kernel; Triton
  import is guarded so the package imports on CPU-only machines.
- `src/esm2_perf/results.py` — `POOLING_COLUMNS` schema.
- `scripts/05_bench_triton_pooling.py` — CLI benchmark (`--quick`, `--dtype`, `--impls`,
  `--seq-lens`, `--batch-sizes`, `--hidden-size`, `--warmup`, `--iters`, `--output`).
- `tests/test_triton_pooling.py` — 8 CUDA+Triton tests (skipped on CPU).
- `docs/milestones/06-triton-pooling-kernel.md`, `docs/METHODOLOGY.md` (validation section),
  `docs/review_notes/06-triton-pooling-kernel-human-review.md`.
- `results/raw/triton_pooling_20260613T162844Z.csv` — 24-row result.

Design decisions:

- **fp32 accumulation.** The kernel sums in fp32 regardless of input dtype, so it is both more
  accurate than bf16 PyTorch pooling and a clean comparison to the fp32 reference. It returns
  fp32 by default (`out_dtype` configurable), so it is not a bit-identical bf16 drop-in.
- **Validate before benchmarking.** The M3 per-sequence reference is the oracle; the kernel is
  tested against it (and the vectorized path) before any speed claim — the point of M3.
- **Honest baseline.** Speedup is vs eager vectorized PyTorch, but the doc also compares against
  the best PyTorch option (`einsum`), which avoids the large temporary that makes the vectorized
  path slow at big shapes, so the 5.4× is not overstated.
- **Correctness next to speed.** `max_abs_err_vs_ref` is a benchmark column, so a fast-but-wrong
  impl is visible.

## Commands run

- `.venv/bin/python -m pytest -q` — 53 passed.
- `.venv/bin/python scripts/05_bench_triton_pooling.py --quick` — succeeded (smoke).
- `.venv/bin/python scripts/05_bench_triton_pooling.py --dtype bf16 --seq-lens 128 512 1022
  --batch-sizes 8 32` — succeeded; wrote 24-row CSV, no OOM.

## Validation

All tests pass. Triton is fastest in every cell (1.29×–5.39× vs eager vectorized; 1.1×–1.6× vs
einsum) with err ~1e-8 vs the fp32 reference. `einsum`/`compiled` cross over (slower than
vectorized at small shapes, faster at large memory-bound shapes). Pooling is ~0.08 ms — a tiny
fraction of the encoder forward — so end-to-end impact is negligible; documented as such.

## Next steps

Milestone 7 — profiling notes and `docs/PROFILING.md` (awaiting human approval).
