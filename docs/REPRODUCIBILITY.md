# Reproducibility

## TL;DR

How to recreate the environment and re-run every benchmark in this repo.

## Short explanation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/00_check_environment.py   # writes results/raw/environment.txt
```

## Longer explanation

- Hardware target: single NVIDIA L4 (24 GB, compute capability 8.9), AWS `g6.2xlarge`.
- Software versions are captured by `scripts/00_check_environment.py`; see
  `results/raw/environment.txt` for the exact stack used.
- Synthetic sequence generation uses deterministic seeds, so inputs are reproducible.
- Each benchmark script supports `--quick` for a fast smoke run before the full matrix.
- Per-script reproduction commands are added as each milestone lands.

## Cleanup / verification checklist

Run this sweep before any commit that touches code or results — it is the gate used at the
final-cleanup milestone and is safe to re-run anytime (none of it re-runs a benchmark):

```bash
.venv/bin/pytest -q                              # unit tests — expect "59 passed"
.venv/bin/ruff check .                            # lint — expect "All checks passed!"
.venv/bin/python scripts/07_summarize_results.py --check   # report matches results/raw/ CSVs
git status --short                                # working tree clean / only intended changes
git clean -nxd                                    # confirm only .venv/caches would be removed
```

- `07_summarize_results.py --check` is the drift guard: it re-derives every table in
  `docs/RESULTS.md` from the raw CSVs and exits non-zero if the committed report is stale. If it
  fails, run the script without `--check` to regenerate, then review the diff.
- `ruff` is an optional dev tool (`pip install ruff`); it is not a runtime dependency. Lint is the
  enforced gate. `ruff format` is intentionally **not** applied — the schema column-lists in
  `src/esm2_perf/results.py` use aligned inline comments for readability.
