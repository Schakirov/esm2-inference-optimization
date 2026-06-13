# Commit 8: Profiling notes

## TL;DR

Add `scripts/06_profile_pytorch.py` and `docs/PROFILING.md`: a `torch.profiler` kernel-level
breakdown of the ESM2 encoder forward. Kernel self-times sum to 225 ms/iter (matching Milestone
2), split ~42% matmul / ~43% unfused elementwise / ~14% copies / ~1% LayerNorm — the elementwise
share explaining the Milestone 5 compile win.

## Short explanation

The profiler lists both operator and device-kernel events; summing both double-counts GPU time.
The script filters to leaf CUDA kernels (so self-times sum to the real forward), ranks the top
kernels, and rolls them into categories. Only the small top-ops CSV is committed; the Chrome
trace is optional and not committed.

## Longer explanation

Files changed:

- `scripts/06_profile_pytorch.py` — CLI profiler with kernel-level view + category rollup
  (`--dtype`, `--seq-len`, `--batch-size`, `--warmup`, `--iters`, `--topk`, `--trace`, `--output`).
- `docs/PROFILING.md` — full write-up.
- `docs/milestones/07-profiling.md`, `docs/METHODOLOGY.md` (profiling note),
  `docs/review_notes/07-profiling-human-review.md`.
- `results/raw/profile_20260613T171924Z.csv` — top-15 kernels.

Design decisions:

- **Kernel-only view to avoid double counting.** `key_averages()` includes operator *and* kernel
  events; the kernel-only sum equals the measured forward time (the operator-inclusive sum was
  ~2×). This is the milestone's main methodological point.
- **Category rollup is honest about being approximate.** Name-substring buckets; the
  matmul-vs-rest split is robust, finer buckets are order-of-magnitude.
- **No huge artifacts.** Only the small CSV is committed; the Chrome trace is `--trace`-gated.
- **Representative shape.** seq 512 / batch 8 (compute-bound) so the breakdown reflects kernel
  work, not launch overhead.

## Commands run

- `.venv/bin/python scripts/06_profile_pytorch.py --dtype bf16 --seq-len 512 --batch-size 8` —
  ran cleanly; 225.1 ms/iter (matches Milestone 2); wrote 15-row CSV.
- `.venv/bin/python -m pytest -q` — 53 passed (no library code changed).

## Validation

Kernel self-times sum to the measured forward latency. Double-counting trap found and fixed.
Matmul ≈ 42% (linears dominate; flash-attention ~4.7%), unfused elementwise ≈ 43%, copies ≈ 14%,
LayerNorm ~1%. Peak memory ~1.4 GB.

## Next steps

Milestone 8 — result processing and portfolio README (awaiting human approval).
