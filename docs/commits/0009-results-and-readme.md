# Commit 9: Result processing & portfolio README

## TL;DR

Add `scripts/07_summarize_results.py`, populate `docs/RESULTS.md`, and fill in the README's
results/profiling/future-work sections (Milestone 8). The summarizer regenerates every report
table from the raw CSVs and `--check`s for drift, so the numbers cannot diverge from the data.

## Short explanation

The script selects the richest raw CSV per experiment, writes a condensed
`results/processed/<key>_summary.csv`, and splices a Markdown table into `docs/RESULTS.md`
between `<!-- BEGIN/END -->` markers. It re-runs nothing — it only condenses existing results.

## Longer explanation

Files changed:

- `scripts/07_summarize_results.py` — stdlib-only summarizer (`--raw-dir`, `--processed-dir`,
  `--results-md`, `--check`, `--no-write-md`); six per-experiment summarizers.
- `docs/RESULTS.md` — full report (prose + 6 generated tables).
- `results/processed/{baseline,correctness,batching,compile,triton,profile}_summary.csv`.
- `README.md` — Results headline table, Profiling, What improved / did not improve, Future work;
  status banner → “results complete (8 / 9)”.
- `docs/milestones/08-results-and-readme.md`, `tests/test_summarize.py`.

Design decisions:

- **Generated tables, single source of truth.** Tables are spliced into `docs/RESULTS.md` from
  the raw CSVs; `--check` makes staleness a failure so the report can't drift.
- **“Richest” file selection, not “latest”.** Sort by `(row_count, filename)` so a 1-row smoke
  run never shadows the 40-row baseline matrix.
- **No new dependency.** Pure `csv`/`re`; pandas is available but unused to keep the script
  portable and the import surface small.
- **Honest negatives are first-class.** A dedicated “what did not improve” section (Triton
  end-to-end, compile cold start, einsum regressions, throughput saturation).

## Commands run

- `.venv/bin/python scripts/07_summarize_results.py` — wrote 6 processed CSVs + regenerated
  `docs/RESULTS.md`.
- `.venv/bin/python scripts/07_summarize_results.py --check` — “up to date” (idempotent).
- `.venv/bin/python -m pytest -q` — 59 passed (53 prior + 6 new).

## Validation

Every generated table cross-checked by hand against its raw CSV. Report regeneration is
idempotent. Test suite green.

## Next steps

Milestone 9 — final cleanup sweep (awaiting human approval).
