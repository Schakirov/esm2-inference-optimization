# Milestone 9: Final cleanup

## TL;DR

The final milestone is a repo-hygiene sweep, not new functionality. All 59 unit tests pass,
`ruff check` is clean, `scripts/07_summarize_results.py --check` confirms `docs/RESULTS.md` is in
sync with the raw CSVs, and the working tree carries nothing but intended files (largest tracked
file is 32 KB; `git clean -nxd` would remove only the gitignored `.venv`/caches). The reusable
verification sweep is documented as a **cleanup / verification checklist** in
[`docs/REPRODUCIBILITY.md`](../REPRODUCIBILITY.md), with the `--check` drift guard wired in. The
status banner moves to **complete (9 / 9)**. Nothing pushed.

## Short explanation

This milestone makes the "definition of done" runnable. Earlier milestones each verified
themselves ad hoc; M9 collects those checks into one checklist anyone can run before a commit:
tests, lint, report-drift guard, and a `git status` / `git clean` hygiene pass. No code behavior
changes.

## Longer explanation

### What was checked

| Gate | Command | Result |
|------|---------|--------|
| Unit tests | `pytest -q` | 59 passed in ~2 s |
| Lint | `ruff check .` | All checks passed |
| Report drift | `scripts/07_summarize_results.py --check` | `docs/RESULTS.md` up to date |
| Tree hygiene | `git status --short` / `git clean -nxd` | clean; only `.venv`/caches ignored |

### Files added / changed

- `docs/REPRODUCIBILITY.md` — new **Cleanup / verification checklist** section; wires
  `07_summarize_results.py --check` in as the report-drift guard and records the ruff policy.
- `docs/milestones/09-final-cleanup.md` — this file.
- `docs/commits/0010-final-cleanup.md` — the commit note.
- `README.md` — status banner updated from "results complete (8 / 9)" to "complete (9 / 9)".

### Formatting decision

`ruff check` (lint) is the enforced gate and passes. `ruff format` is **intentionally not applied**:
it would rewrite 13 files (~332-line diff) almost entirely by collapsing the deliberately
column-aligned inline comments in the schema definitions (e.g. `src/esm2_perf/results.py`), which
is a readability regression with no correctness benefit. Ruff remains an optional dev tool, not a
runtime dependency.

### Human-review note

Unlike milestones 1–7, no `docs/review_notes/` gate file was written for milestones 8 and 9: both
are low-risk, code-light increments (a read-only summarizer and a hygiene sweep), so the per-gate
human-review note was judged unnecessary by the reviewer rather than omitted by accident.

## Validation

- `.venv/bin/pytest -q` → 59 passed.
- `.venv/bin/ruff check .` → All checks passed.
- `.venv/bin/python scripts/07_summarize_results.py --check` → "docs/RESULTS.md is up to date."
- `git status --short` → clean (only the intended M9 doc/README changes staged).

## Limitations

- The checklist verifies the repo's *internal* consistency (tests, lint, report-vs-CSV sync); it
  does not re-run benchmarks, so it cannot detect hardware/stack drift on a different machine.
- `ruff` is not pinned in `requirements.txt` (it is dev-only); a future contributor must
  `pip install ruff` to run the lint gate.

## Next steps

None — this is the final planned milestone. The project is feature-complete at 9 / 9. Any further
work (new shapes, GPUs, or kernels) would start a new plan.
