# Commit 1: Repo scaffold and implementation plan

## TL;DR

Fill the empty project scaffold: package metadata, README draft, documentation stubs, the
nine-milestone `docs/PLAN.md`, and directory keepers. Establishes the structure every later
milestone writes into. No model code or benchmarks yet.

## Short explanation

The initial commit created empty placeholder files and directories. This commit gives them
content: a real `pyproject.toml`, a populated package `__init__`, a portfolio-shaped README
draft, TL;DR-first doc stubs (`METHODOLOGY`, `LIMITATIONS`, `REPRODUCIBILITY`, `RESULTS`),
and the full plan.

## Longer explanation

Files changed:

- `pyproject.toml` — project metadata, `src/` package discovery, pytest + ruff config.
- `src/esm2_perf/__init__.py` — docstring, `__version__`, roadmap of future modules.
- `README.md` — all required portfolio headings as a draft, plus an explicit
  "what this project does not claim" section.
- `docs/PLAN.md` — environment table, design principles, milestone table, repo layout,
  review-gate list.
- `docs/METHODOLOGY.md`, `docs/LIMITATIONS.md`, `docs/REPRODUCIBILITY.md`,
  `docs/RESULTS.md` — stubs to be expanded per milestone.
- `results/raw/.gitkeep`, `results/processed/.gitkeep`, `external/.gitkeep`.

Assumptions/tradeoffs: `requirements.txt` is intentionally unpinned for portability; exact
installed versions are captured in `results/raw/environment.txt` for reproducibility.

## Commands run

- `git status` — succeeded (showed scaffold/docs untracked, `.gitignore` modified).
- File writes via editor — succeeded.

## Validation

Structural only at this stage; the environment script (commit 2) is the first executable
validation. Confirmed `.gitignore` excludes `.venv/`, caches, weights, and large traces.

## Next steps

Add and run the environment check script (commit 2).
