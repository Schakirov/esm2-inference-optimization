# Human review: Variable-length batching & padding benchmark

## TL;DR

The fourth major milestone — *batching / padding benchmark complete* — is done. A pure,
CPU-tested `esm2_perf.batching` module plans naive vs length-sorted batches and tallies
padding waste; `scripts/03_bench_batching.py` times a full pass over a fixed 256-sequence pool
(lengths 32–512 aa) on the L4. Length-sorting cuts padding waste from **41–44%** to **2–8%**
and raises real-token throughput up to **1.75×** (10.5k → 18.4k tok/s at batch 8) at a roughly
constant padded-token rate. 42 unit tests pass. Results in
`results/raw/batching_20260613T112856Z.csv`. Nothing pushed.

## What changed

- `src/esm2_perf/batching.py` — `plan_batches` (naive/sorted), `BatchPlan`, padding accounting.
- `src/esm2_perf/results.py` — `BATCHING_COLUMNS` schema.
- `scripts/03_bench_batching.py` — CLI benchmark with `--quick` and pool/strategy/matrix flags.
- `tests/test_batching.py` — 11 CPU-only tests.
- `docs/milestones/04-batching-and-padding.md`, `docs/METHODOLOGY.md` (padding-waste section),
  `docs/commits/0005-batching-and-padding.md`.
- `results/raw/batching_20260613T112856Z.csv` — 6-row result.

## What I should understand before continuing

- **Where padding waste comes from.** Batching pads every sequence to the longest in its batch.
  With varied lengths, the GPU computes over a lot of padding. `padding_fraction` is the share
  of computed tokens that are padding; `padding_waste_ratio = padded / real` (≥ 1.0).
- **Why two throughput columns.** Real tokens/sec is useful work; padded tokens/sec is the raw
  GPU rate. Sorting raises the real rate while the padded rate stays about flat — the win is
  *recovered waste*, not a faster GPU. Stating only the real-token speedup without that context
  would overclaim.
- **Why sorting helps.** Sorting by length puts similar-length sequences in the same batch, so
  each batch's max ≈ its members' lengths and little padding is added. Sorted `padded_tokens`
  (74,352) is within 2% of the irreducible `actual_tokens` (72,720) at batch 8.
- **Padding grows with batch size.** Bigger batches span a wider length range, so padding rises
  with `batch_size` for both strategies (sorted 2% → 8%, naive 41% → 44%). Smaller batches are
  more padding-efficient here.
- **The library is GPU-free.** All padding math lives in pure Python and is unit-tested without
  a GPU; only the timing path touches CUDA.

## Commands I should run manually

```bash
# Fast smoke (small pool, both strategies)
.venv/bin/python scripts/03_bench_batching.py --quick

# A real slice you can eyeball in ~1–2 min
.venv/bin/python scripts/03_bench_batching.py --num-seqs 128 --min-len 32 --max-len 512 --batch-sizes 8 16 --output results/raw/_review.csv

# Inspect the committed results
column -s, -t results/raw/batching_20260613T112856Z.csv | less -S

# Unit tests
.venv/bin/python -m pytest -q
```

## Questions I should be able to answer

- Why is real-token throughput higher for sorted but padded tokens/sec about the same?
  (The GPU computes padded tokens at a fixed rate; sorting feeds it fewer padding tokens, so
  more of the same compute is useful work.)
- Why does padding fraction rise with batch size even for sorted? (Larger batches span a wider
  length range, so the batch max exceeds more members' lengths.)
- Why does padded tokens/sec dip slightly at larger batches? (Bigger batches more often contain
  a long sequence, raising the batch max and the O(L²) attention cost per token.)
- How do I know the comparison is apples-to-apples? (One fixed pool/seed is shared across all
  cells; `actual_tokens` is identical (72,720) on every row — only the grouping changes.)
- What would break this win in production? (Sorting reorders outputs — you must gather back to
  the original order — and it assumes a full batch pool, not streaming.)

## Possible bugs or misleading benchmark artifacts

- **Uniform synthetic lengths.** Lengths are uniform in [min, max]; real proteomes are not, so
  absolute padding fractions are illustrative. The mechanism and relative win generalize, but
  do not quote 41% as a universal figure.
- **Order restoration not exercised.** `BatchPlan` keeps original indices, but the benchmark
  does not gather outputs back to input order. A real pipeline must, or embeddings will be
  misaligned to inputs.
- **Token-fraction vs time.** Padding waste is reported as wasted tokens; because attention is
  O(L²), a long sequence's padding costs disproportionately more time than its token share —
  so the token fraction slightly understates the time cost at large batches.
- Single dtype (bf16), one pool/seed, two strategies only — no length-bucketing-then-shuffle or
  drop-last variants.

## Human notes

Reviewed milestone 4

Length-sorted batching reduces padding waste of resources. Expectedly.

Let's move to the next more interesting milestones.
