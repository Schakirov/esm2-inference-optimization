# Milestone 4: Variable-length batching & padding efficiency

## TL;DR

Batching variable-length sequences pads each to its batch's longest member, so the GPU burns
compute on padding. A new pure, CPU-tested `esm2_perf.batching` module plans batches under two
strategies (`naive` vs length-`sorted`) and tallies padding waste; `scripts/03_bench_batching.py`
times a full pass over a fixed 256-sequence pool (lengths 32–512 aa) and reports throughput as
both real and padded tokens/sec. On the L4, naive batching wastes **41–44%** of compute on
padding; length-sorting cuts that to **2–8%** and lifts real-token throughput by up to
**1.75×** (10.5k → 18.4k tokens/sec at batch 8) — at a roughly constant padded-token rate,
which is the honest framing: the GPU does the same raw work, just less of it wasted. Results
are in `results/raw/batching_20260613T112856Z.csv`.

## Short explanation

`src/esm2_perf/batching.py` provides `plan_batches(lengths, batch_size, strategy)` returning a
`BatchPlan` with the batch grouping and padding accounting (`actual_tokens`, `padded_tokens`,
`padding_fraction`, `padding_waste_ratio`). The benchmark builds one fixed pool of
variable-length synthetic sequences, then for each (strategy, batch_size) cell pre-tokenizes
the planned batches onto the GPU and times one full pass with CUDA events under
`torch.inference_mode()`. Real tokens are independent of grouping; padded tokens are not, so
the difference between the two throughput numbers is exactly the recovered waste. 11 new
CPU-only unit tests cover the planning and padding math.

## Longer explanation

### Files added / changed

- `src/esm2_perf/batching.py` — `STRATEGIES`, `order_indices`, `chunk`, `BatchPlan`,
  `plan_batches`. Pure Python (no torch), so padding accounting is unit-testable on CPU.
- `src/esm2_perf/results.py` — new `BATCHING_COLUMNS` schema.
- `scripts/03_bench_batching.py` — CLI benchmark (`--quick`, `--dtype`, `--strategies`,
  `--num-seqs`, `--min-len`, `--max-len`, `--batch-sizes`, `--warmup`, `--iters`, `--output`).
- `tests/test_batching.py` — 11 CPU-only tests.
- `docs/METHODOLOGY.md` — new "Padding waste & batching" section.

### Strategies & metrics

- **naive** keeps the pool's arbitrary order; each batch pads to its own max.
- **sorted** sorts by length first, so each batch holds similar-length sequences and pads less.
- Per batch, *padded tokens* = `batch_size × max_length_in_batch`, *real tokens* = sum of token
  counts. `padding_fraction = (padded − real) / padded`; `padding_waste_ratio = padded / real`.

A single fixed pool is shared across all cells so the *only* thing that changes between rows is
how sequences are grouped.

### Results

Source: `results/raw/batching_20260613T112856Z.csv`. Pool: 256 sequences, lengths 32–512 aa
(seed 0), bf16. `actual_tokens` = 72,720 for every row (grouping-independent).

| strategy | bs | padding_fraction | padded_tokens | latency_ms | real tok/s | padded tok/s | peak GB |
|----------|---:|-----------------:|--------------:|-----------:|-----------:|-------------:|--------:|
| naive  | 8  | 0.410 | 123,200 | 6941 | 10,476 | 17,749 | 1.42 |
| naive  | 16 | 0.426 | 126,656 | 8019 |  9,069 | 15,795 | 1.62 |
| naive  | 32 | 0.440 | 129,824 | 8593 |  8,463 | 15,109 | 2.01 |
| sorted | 8  | 0.022 |  74,352 | 3956 | 18,382 | 18,794 | 1.42 |
| sorted | 16 | 0.045 |  76,176 | 4546 | 15,996 | 16,757 | 1.62 |
| sorted | 32 | 0.083 |  79,328 | 5140 | 14,147 | 15,432 | 2.01 |

Observations:

- **Length-sorting is a large, free win.** At batch 8 it drops padding from 41% to 2% and
  raises real-token throughput 1.75× (10,476 → 18,382 tok/s), purely by reordering — no kernel
  change, no precision change. The sorted pool's `padded_tokens` (74,352) is within 2% of the
  irreducible `actual_tokens` (72,720).
- **The win is recovered waste, not a faster GPU.** Padded tokens/sec is similar across
  strategies at each batch size (e.g. ~17.7k naive vs ~18.8k sorted at bs 8). The GPU computes
  padded tokens at about the same rate either way; sorting just stops feeding it padding.
- **Padding grows with batch size.** Bigger batches span a wider length range, so even sorted
  batches pad more (2% → 8% from bs 8 → 32) and naive worsens too (41% → 44%). Here smaller
  batches are more padding-efficient; real throughput is highest at sorted/bs-8.
- **Padded throughput dips slightly at larger batches** (sorted 18.8k → 15.4k tok/s). Larger
  batches are more likely to contain a long sequence, raising the batch's max length and with
  it the O(L²) attention cost per token — the same quadratic effect seen in the baseline.
- **Memory tracks batch size**, not strategy (1.42 → 2.01 GB from bs 8 → 32), well within the
  L4's 24 GB.

## Validation

- `pytest -q` → 42 passed (31 prior + 11 new batching tests).
- `python scripts/03_bench_batching.py --quick` → smoke run (both strategies) succeeded.
- Full run (256 seqs, 32–512 aa, batch {8,16,32}, both strategies) → 6 rows, no OOM.
- Padding accounting is computed by the unit-tested library; `actual_tokens` is identical
  across all rows as expected, and sorted `padded_tokens` ≈ `actual_tokens`.

## Limitations

- Synthetic sequence **lengths are uniform** in [min, max]; real proteomes are not uniform, so
  the absolute padding fractions are illustrative. The mechanism (sorting removes
  intra-batch length variance) and the relative win hold for any non-degenerate length
  distribution.
- Sorting reorders outputs; a production pipeline must **restore the original order** after
  inference (the plan stores original indices, but the benchmark does not exercise a gather
  back). Sorting also assumes a full pool is available (batch inference), not streaming.
- Padding waste here is measured as wasted *tokens/compute*; attention's O(L²) cost means a
  long sequence's padding is disproportionately expensive, so token-fraction slightly
  understates the time cost at large batch sizes.
- Single dtype (bf16) and one pool/seed; no bucketed-by-length-then-shuffled hybrid or
  drop-last variants.

## Next steps

Milestone 5 — `torch.compile` & static shapes: measure compile warmup vs steady-state, and
whether a compiled encoder beats eager on the L4 (documenting honestly if it does not).
**Awaiting human approval before starting.**
