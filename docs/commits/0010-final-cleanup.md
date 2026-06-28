# Commit 10: Final cleanup

## TL;DR

Final repo-hygiene milestone (Milestone 9): no behavior changes. Document a runnable cleanup /
verification checklist in `docs/REPRODUCIBILITY.md` (wiring in the `07_summarize_results.py
--check` drift guard), add the M9 milestone note, and move the README banner to "complete
(9 / 9)". All 59 tests pass, `ruff check` is clean, the report is in sync with the raw CSVs.

## Short explanation

Earlier milestones verified themselves ad hoc; this commit collects the "definition of done" into
one checklist — tests, lint, report-drift guard, and a `git status` / `git clean` hygiene pass —
so it can be run before any future commit. No code paths change.

## Longer explanation

Files changed:

- `docs/REPRODUCIBILITY.md` — new **Cleanup / verification checklist** section; wires
  `07_summarize_results.py --check` as the report-drift gate; records the ruff policy.
- `docs/milestones/09-final-cleanup.md` — milestone note.
- `README.md` — status banner "results complete (8 / 9)" → "complete (9 / 9)".

Decisions:

- **`ruff check` is the lint gate; `ruff format` is not applied.** Formatting would rewrite 13
  files (~332-line diff) by collapsing intentionally column-aligned inline comments in the schema
  definitions — a readability regression with no correctness benefit.
- **No new runtime dependency.** `ruff` is dev-only and not pinned in `requirements.txt`.
- **No `docs/review_notes/` gate for M8–M9.** Both are low-risk, code-light increments; the
  reviewer judged the per-gate human-review note unnecessary rather than omitting it by accident.

## Commands run

- `.venv/bin/python -m pytest -q` — 59 passed.
- `.venv/bin/ruff check .` — All checks passed.
- `.venv/bin/python scripts/07_summarize_results.py --check` — "docs/RESULTS.md is up to date."
- `git status --short` / `git clean -nxd` — tree clean; only `.venv`/caches ignored.

## Validation

The checklist commands above all pass on the committed tree. This is the final planned milestone;
the project is feature-complete at 9 / 9.

## Next steps

None — final milestone. Further work would start a new plan.
