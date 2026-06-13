# Human review: Triton masked-mean-pooling kernel

## TL;DR

The sixth major milestone — *Triton pooling kernel complete* — is done. A hand-written Triton
masked-mean-pooling kernel (`esm2_perf.triton_pooling`) is validated against the Milestone 3
per-sequence reference (8 GPU tests) and benchmarked against the PyTorch pooling paths. On the
L4 in bf16 it is the **fastest implementation in all 6 shapes** (1.3×–5.4× vs eager vectorized
PyTorch; 1.1×–1.6× vs the best PyTorch option) and **near-exact** (err ~1e-8 vs ~1e-3 for bf16
PyTorch). Honest caveat: pooling is ~0.08 ms — a negligible fraction of the ~25–225 ms encoder
forward — so end-to-end throughput is unchanged. 53 tests pass. Results in
`results/raw/triton_pooling_20260613T162844Z.csv`. Nothing pushed.

## What changed

- `src/esm2_perf/triton_pooling.py` — `triton_masked_mean_pool` + the `@triton.jit` kernel.
- `src/esm2_perf/results.py` — `POOLING_COLUMNS` schema.
- `scripts/05_bench_triton_pooling.py` — 4-impl benchmark (pytorch / einsum / compiled / triton).
- `tests/test_triton_pooling.py` — 8 CUDA+Triton tests against the M3 reference.
- `docs/milestones/06-triton-pooling-kernel.md`, `docs/METHODOLOGY.md` (validation section),
  `docs/commits/0007-triton-pooling-kernel.md`.
- `results/raw/triton_pooling_20260613T162844Z.csv` — 24-row result.

## What I should understand before continuing

- **Why this validates against Milestone 3.** The per-sequence reference built in M3 is the
  oracle: the kernel is tested against it (≤1e-4 rtol) before any speed claim. That is the whole
  reason an independent, obviously-correct pooling implementation existed.
- **Fastest *and* most accurate.** The kernel accumulates in fp32, so it is essentially the
  fp32 reference (err ~1e-8) while bf16 PyTorch pooling carries ~1e-3 — and it is still the
  fastest option. Speed and accuracy are not traded off here.
- **Read the 5.4× carefully.** That peak is partly because the eager vectorized
  `masked_mean_pool` materializes a full `(B,T,H)` temporary and is slow at large shapes.
  Against the *best* PyTorch option (`einsum`, which avoids the temporary), Triton is a more
  modest, consistent 1.1×–1.6×. Both framings are in the milestone doc.
- **It does not matter end-to-end.** Pooling is ~0.08 ms vs a ~25–225 ms encoder forward, so
  none of these speedups move pipeline throughput. The kernel's value is pedagogical (a correct,
  tested Triton reduction) plus the accuracy bonus for stored embeddings — stated plainly, not
  dressed up as a pipeline win.

## Commands I should run manually

```bash
# Fast smoke (all impls, one shape)
.venv/bin/python scripts/05_bench_triton_pooling.py --quick

# A real slice
.venv/bin/python scripts/05_bench_triton_pooling.py --dtype bf16 --seq-lens 128 512 --batch-sizes 8 32 --output results/raw/_review.csv

# Inspect the committed results
column -s, -t results/raw/triton_pooling_20260613T162844Z.csv | less -S

# Unit tests (the Triton tests run on GPU, skip on CPU)
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_triton_pooling.py -q
```

## Questions I should be able to answer

- How is the kernel proven correct? (Tested against the M3 per-sequence reference and the
  vectorized PyTorch pooling to ≤1e-4 rtol on fp32 inputs, plus all-padding and out_dtype edge
  cases; `max_abs_err_vs_ref` is also a benchmark column.)
- Why is Triton both faster and more accurate? (Single fused fp32-accumulating pass: no big
  intermediate tensor like the vectorized path, and fp32 sums beat bf16 sums on accuracy.)
- Why is the 5.4× speedup not the honest headline? (The eager vectorized baseline is weak at
  large shapes; vs the best PyTorch option Triton is 1.1×–1.6×.)
- Why does `torch.compile` not help here? (It is a tiny pointwise reduction; compile
  overhead/guards dominate a ~0.1 ms op — same overhead story as Milestone 5.)
- Does this speed up the pipeline? (No — pooling is ~0.08 ms of a ~25–225 ms forward. The win is
  correctness/accuracy and a worked kernel, not throughput.)

## Possible bugs or misleading benchmark artifacts

- **Weak-baseline inflation.** Reporting only "vs vectorized PyTorch" would overstate the win at
  large shapes; the einsum comparison is the honest anchor.
- **Synthetic inputs.** Random hidden states, not real encoder outputs; pooling latency is
  data-independent so this is fine, but the tensors are synthetic.
- **Not autotuned.** Fixed `BLOCK_T=128`/`BLOCK_H=256`; a per-shape autotune might change the
  largest-cell number. No silent claim that this is the optimal config.
- **fp32 accumulation is not a bit-identical bf16 drop-in.** It is deliberately more accurate; a
  pipeline expecting bf16-accumulated pooling would see a (tiny, beneficial) numerical change.
- Pooling speedups do not change end-to-end encoder throughput — do not quote them as if they do.

## Human notes

_Add review outcome / approval here before Milestone 7 begins._
