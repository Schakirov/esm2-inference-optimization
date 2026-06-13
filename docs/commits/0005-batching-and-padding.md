# Commit 5: Variable-length batching & padding benchmark

## TL;DR

Add `esm2_perf.batching` (pure, CPU-tested batch planning + padding accounting) and
`scripts/03_bench_batching.py`, comparing naive vs length-sorted batching over a fixed
variable-length pool. New `BATCHING_COLUMNS` schema, 11 unit tests, and a 6-row result CSV.
Headline: length-sorting cuts padding waste from 41–44% to 2–8% and raises real-token
throughput up to 1.75× at a roughly constant padded-token rate.

## Short explanation

`plan_batches` groups sequence indices (naive order or sorted by length) and tallies real vs
padded tokens. The benchmark shares one fixed 256-sequence pool across all cells, pre-tokenizes
each strategy's batches onto the GPU, and times one full pass with CUDA events under
`torch.inference_mode()`, recording padding fraction and both real and padded tokens/sec.

## Longer explanation

Files changed:

- `src/esm2_perf/batching.py` — `STRATEGIES`, `order_indices`, `chunk`, `BatchPlan`,
  `plan_batches`; no torch dependency so the math is CPU-unit-testable.
- `src/esm2_perf/results.py` — `BATCHING_COLUMNS` schema.
- `scripts/03_bench_batching.py` — CLI benchmark (`--quick`, `--dtype`, `--strategies`,
  `--num-seqs`, `--min-len`, `--max-len`, `--batch-sizes`, `--warmup`, `--iters`, `--output`).
- `tests/test_batching.py` — 11 CPU-only tests.
- `docs/milestones/04-batching-and-padding.md`, `docs/METHODOLOGY.md` (padding-waste section),
  `docs/review_notes/04-batching-and-padding-human-review.md`.
- `results/raw/batching_20260613T112856Z.csv` — 6-row result.

Design decisions:

- **Pure planning library.** Ordering and padding accounting carry no torch dependency, so
  every padding number is reproducible and unit-tested without a GPU. The script feeds token
  lengths (amino acids + 2 special tokens), so library `padded_tokens` matches the tokenizer.
- **Shared fixed pool.** One pool of variable-length sequences is reused across all cells, so
  the only variable between rows is the grouping; `actual_tokens` is identical everywhere.
- **Two throughput numbers.** Real (non-pad) tokens/sec is the useful-work rate; padded
  tokens/sec is the raw GPU rate. Reporting both makes the win unambiguous — sorting raises the
  real rate while the padded rate stays about flat.
- **Pre-tokenize then time.** Batches are moved to the GPU before timing so the measured pass is
  encoder compute (including wasted padding compute), not host-side tokenization or H2D copies.

## Commands run

- `.venv/bin/python -m pytest -q` — 42 passed.
- `.venv/bin/python scripts/03_bench_batching.py --quick` — succeeded (smoke).
- `.venv/bin/python scripts/03_bench_batching.py --num-seqs 256 --min-len 32 --max-len 512
  --batch-sizes 8 16 32` — succeeded; wrote 6-row CSV, no OOM.

## Validation

All tests pass. Naive padding fraction 0.41–0.44; sorted 0.02–0.08. Real-token throughput at
batch 8 rises 10,476 → 18,382 tok/s (1.75×) while padded tokens/sec stays ~17.7k → 18.8k.
Memory tracks batch size (1.42–2.01 GB), not strategy. `actual_tokens` constant (72,720) across
all rows, confirming grouping-independence.

## Next steps

Milestone 5 — `torch.compile` & static shapes (awaiting human approval).
