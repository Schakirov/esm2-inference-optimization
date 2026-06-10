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
