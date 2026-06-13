# Milestone 6: Triton masked-mean-pooling kernel

## TL;DR

A hand-written Triton kernel (`esm2_perf.triton_pooling`) fuses masked mean pooling into a
single fp32-accumulating pass. It is validated against the Milestone 3 per-sequence reference
(8 GPU tests) and benchmarked against the PyTorch pooling paths. On the L4 in bf16 it is the
**fastest implementation in all 6 shapes** and **numerically near-exact**: 1.3×–5.4× faster
than the eager vectorized PyTorch pooling (1.1×–1.6× faster than the *best* PyTorch option,
`einsum`), while its max error vs the fp32 reference is ~1e-8 versus ~1e-3 for the bf16 PyTorch
paths. The honest caveat: pooling is ~0.08 ms — a tiny fraction of the ~25–225 ms encoder
forward — so the end-to-end impact is negligible; the value is a correct, fast, more-accurate
fused kernel and a worked Triton reduction. Results: `results/raw/triton_pooling_20260613T162844Z.csv`.

## Short explanation

The kernel reduces over the sequence axis: grid `(B, ceil(H/BLOCK_H))`, each program owns one
batch row and a block of hidden columns and streams over `T` in `BLOCK_T` chunks, accumulating
the masked sum and token count in fp32, then divides (clamping the count to 1 so all-padding
rows pool to zero). `scripts/05_bench_triton_pooling.py` times four implementations —
`pytorch` (vectorized), `einsum`, `compiled` (`torch.compile`), and `triton` — recording
latency, speedup vs eager PyTorch, and max abs error vs the fp32 reference.

## Longer explanation

### Files added / changed

- `src/esm2_perf/triton_pooling.py` — `triton_masked_mean_pool` + the `@triton.jit` kernel.
  Imports guard on Triton so the package still imports on CPU-only machines.
- `src/esm2_perf/results.py` — `POOLING_COLUMNS` schema.
- `scripts/05_bench_triton_pooling.py` — CLI benchmark (`--quick`, `--dtype`, `--impls`,
  `--seq-lens`, `--batch-sizes`, `--hidden-size`, `--warmup`, `--iters`, `--output`).
- `tests/test_triton_pooling.py` — 8 GPU+Triton tests against the M3 reference (skipped on CPU).
- `docs/METHODOLOGY.md` — new "Triton kernel validation" section.

### Results

Source: `results/raw/triton_pooling_20260613T162844Z.csv` (bf16, hidden 1280). Latency is the
median over 50 iters; `speedup` is vs the eager vectorized `pytorch` row; `err` is max abs
error vs the fp32 per-sequence reference.

| seq | bs | pytorch (ms) | einsum | compiled | triton (ms) | triton speedup | triton err |
|----:|---:|-------------:|-------:|---------:|------------:|---------------:|-----------:|
| 128 | 8  | 0.0974 | 0.74× | 0.78× | 0.0758 | 1.29× | 3.0e-08 |
| 128 | 32 | 0.1004 | 0.74× | 0.78× | 0.0727 | 1.38× | 3.0e-08 |
| 512 | 8  | 0.0993 | 0.74× | 0.72× | 0.0758 | 1.31× | 7.5e-09 |
| 512 | 32 | 0.4577 | 3.31× | 3.06× | 0.0850 | **5.39×** | 3.0e-08 |
| 1022| 8  | 0.1157 | 0.83× | 0.93× | 0.0758 | 1.53× | 1.5e-08 |
| 1022| 32 | 1.0506 | 2.42× | 2.19× | 0.3891 | 2.70× | 2.2e-08 |

Observations:

- **Triton is fastest in every cell** and **near-exact.** Because it accumulates in fp32, its
  output is essentially the fp32 reference (err ~1e-8); the bf16 PyTorch paths sum in bf16 and
  carry ~1e-3 error. So the kernel is simultaneously the fastest *and* the most accurate option.
- **Read the 5.4× honestly — the eager baseline is weak at large sizes.** The vectorized
  `masked_mean_pool` does `(hidden * mask).sum(dim=1)`, which materializes a full `(B,T,H)`
  temporary and is memory-bound; at seq 512 / batch 32 it balloons to 0.46 ms. Against the
  *best* PyTorch option at each shape (usually `einsum`, which avoids the temporary), Triton is
  a more modest but consistent **1.1×–1.6×** faster. Both framings are in the table.
- **`einsum` and `compiled` cross over.** They are *slower* than the simple vectorized path at
  small shapes (fixed overhead/guards dominate a ~0.1 ms op) but much faster at the large
  memory-bound shapes (no big temporary). `torch.compile` does not help a tiny pointwise
  reduction and can hurt it — consistent with the Milestone 5 overhead story.
- **Triton latency is nearly flat (~0.073–0.085 ms) until the largest cell**, where it rises to
  0.39 ms — it is launch/overhead-bound below that, like everything else at this scale.

### Honest framing

Pooling is a tiny, memory-bound tail of the pipeline: ~0.08 ms against a ~25–225 ms encoder
forward (Milestone 2), so **none of these speedups move end-to-end throughput meaningfully**.
The kernel earns its place as (a) a correct, unit-tested Triton reduction validated against an
independent oracle, and (b) a pooling path that is both faster and ~5 orders of magnitude more
accurate than the default bf16 PyTorch pooling — useful if pooled embeddings are stored and
compared downstream. It is not a claim that a custom kernel is the right tool for this op in a
real pipeline.

## Validation

- `pytest -q` → 53 passed (45 prior + 8 new Triton tests; tests skip on CPU-only machines).
- The kernel matches the fp32 reference and the vectorized PyTorch pooling to ≤1e-4 rtol /
  1e-5 atol across shapes, handles all-padding rows (zeros, not NaN), and respects `out_dtype`.
- `python scripts/05_bench_triton_pooling.py --quick` → 4-impl smoke succeeded.
- Full run (3 seq × 2 batch × 4 impls) → 24 rows, no OOM, no per-cell errors.

## Limitations

- Inputs are random hidden states, not real encoder outputs; pooling latency is data-independent
  so this is representative, but the absolute numbers are for synthetic tensors.
- Single dtype (bf16) and fixed `BLOCK_T=128`/`BLOCK_H=256`; not autotuned per shape, so there is
  likely some performance left on the table (and possibly a faster config at the largest cell).
- The kernel always accumulates in fp32; that is a feature for accuracy but means it is not a
  bit-identical drop-in for the bf16-accumulating PyTorch pooling.
- Speedups are for the pooling op in isolation and do not change end-to-end encoder throughput.

## Next steps

Milestone 7 — profiling: capture profiler evidence (kernel timeline, memory) for the encoder
and pooling paths and write `docs/PROFILING.md`. **Awaiting human approval before starting.**
