# Milestone 8: Result processing & portfolio README

## TL;DR

`scripts/07_summarize_results.py` turns the timestamped raw CSVs under `results/raw/` into the
portfolio report. For each experiment it selects the **richest** raw CSV (most data rows, so a
`--quick` smoke run never shadows the full matrix), writes a condensed
`results/processed/<key>_summary.csv`, and splices a Markdown table into
[`docs/RESULTS.md`](../RESULTS.md) between `<!-- BEGIN/END key -->` markers — so the report is
**regenerated from data and cannot silently drift** (`--check` fails on staleness). `docs/RESULTS.md`
and the `README.md` results/profiling/future-work sections are now populated to portfolio quality.
The consolidated story: length-sorted batching is the cheapest real win (~1.7×), `torch.compile`
is a real 1.3×–2.2× with a ~28 s/shape cold-start caveat, and the Triton pooling kernel is
correct and fastest-in-shape but end-to-end negligible — reported as honestly as the wins.

## Short explanation

The script is read-only with respect to benchmarks: it never re-runs anything, it condenses
CSVs earlier milestones already produced. Six per-experiment summarizers (baseline, correctness,
batching, compile, triton, profile) each return a Markdown table plus the rows for a processed
CSV. Tables are spliced into `docs/RESULTS.md` so prose and generated data live in one file with
a single source of truth. The headline numbers are duplicated into `README.md` as a one-glance
table with the honest caveat for each row.

## Longer explanation

### Files added / changed

- `scripts/07_summarize_results.py` — the summarizer (`--raw-dir`, `--processed-dir`,
  `--results-md`, `--check`, `--no-write-md`); stdlib-only (`csv`, `re`), no pandas dependency.
- `docs/RESULTS.md` — full report: TL;DR, environment, one section per experiment with a
  generated table, “what improved / did not improve”, reproduce steps.
- `results/processed/*_summary.csv` — six condensed summaries (committed; small).
- `README.md` — populated Results (headline table), Profiling, What improved / did not improve,
  and Future work; status banner updated to “results complete (8 / 9)”.
- `tests/test_summarize.py` — 6 CPU-only tests for the pure helpers (table rendering,
  richest-file selection, marker splice). 59 tests pass.

### File-selection rule

`pick_richest` sorts a family's CSVs by `(row_count, filename)` and takes the max. This is why
the baseline table comes from the 40-row `baseline_20260610T101812Z.csv` and not the later
1-row smoke file — “latest” alone would have picked the wrong one.

### The consolidated findings

- **Baseline:** real-token throughput saturates ~22k tok/s (bf16); peak ~2.0 GB of 24 GB.
- **Batching:** sorting cuts padding waste 41–44% → 2–8%, ~1.7× real-token throughput, free.
- **torch.compile:** 1.3×–2.2× steady, ~28 s cold start per static shape.
- **Triton pooling:** fastest in all 6 shapes, ~1e-8 accurate, but <0.1% of runtime end-to-end.
- **Profiling:** ~42% matmul (FFN linears, not attention) / ~43% unfused elementwise.

## Validation

- `python scripts/07_summarize_results.py` → wrote 6 processed CSVs and regenerated all 6 tables
  in `docs/RESULTS.md`; numbers cross-checked against the raw CSVs by hand.
- `python scripts/07_summarize_results.py --check` → reports “up to date” (idempotent).
- `pytest -q` → 59 pass (53 prior + 6 new summarizer tests).

## Limitations

- The report aggregates existing CSVs; it does not re-run benchmarks, so it inherits each run's
  hardware/software stack and synthetic-sequence caveats (see `docs/LIMITATIONS.md`).
- The profiling category percentages use the profiler's top-15 kernel coverage (~99% of the
  forward), a slightly different normalization than the per-iter-ms view on the site — both are
  labeled where shown.

## Next steps

Milestone 9 — final cleanup: a tests/format/`git status` sweep, wire `07_summarize_results.py
--check` into the cleanup checklist, and the final commit. **Awaiting human approval before
starting.**
