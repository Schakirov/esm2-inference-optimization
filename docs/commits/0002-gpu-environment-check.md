# Commit 2: Reproducible L4 environment check

## TL;DR

Add `scripts/00_check_environment.py`, a self-contained, gracefully-degrading environment
probe, run it on the target NVIDIA L4, and capture its output to
`results/raw/environment.txt`. This is the first executable, GPU-touching step.

## Short explanation

The script reports system, PyTorch/CUDA, GPU, memory, and package-version details plus an
`nvidia-smi` snapshot, and can mirror the report to a file via `--output`. It is written to
fail gracefully when torch, CUDA, or `nvidia-smi` are unavailable.

## Longer explanation

Files changed:

- `scripts/00_check_environment.py` — argparse CLI; `collect_report()` builds the text
  report; `_pkg_version()` resolves versions tolerating the biopython→`Bio` import-name
  mismatch and missing packages; per-device GPU properties via
  `torch.cuda.get_device_properties`; `nvidia-smi` queried with a fixed CSV field list and
  a timeout.
- `results/raw/environment.txt` — captured report (small text file, safe to commit).

Design decisions: keep zero hard dependencies beyond the stdlib + torch so the check runs
even in a partial environment; query a stable subset of `nvidia-smi` fields to keep the
output diff-friendly; write through `--output` rather than shell redirection so parent
directories are created and the path is logged.

## Commands run

- `.venv/bin/python scripts/00_check_environment.py --output results/raw/environment.txt`
  — succeeded (exit 0); detected NVIDIA L4, capability 8.9, ~22 GiB, bf16 supported, torch
  2.12.0+cu130 / CUDA 13.0.

## Validation

Output matches the independently-run `nvidia-smi` snapshot (NVIDIA L4, 24 GB, driver
595.71.05). CUDA available and bf16 supported, confirming later GPU benchmarks are viable.
No model weights downloaded.

## Next steps

Milestone 2 — baseline ESM2 inference benchmark (awaiting human approval at the review
gate).
