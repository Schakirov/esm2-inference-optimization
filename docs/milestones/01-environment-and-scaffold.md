# Milestone 1: Environment and scaffold

## TL;DR

The repository scaffold is in place (package skeleton, `pyproject.toml`, pinned-ish
`requirements.txt`, README draft, doc stubs, ignore rules) and a reproducible environment
check (`scripts/00_check_environment.py`) runs cleanly on the target NVIDIA L4, with its
output captured at `results/raw/environment.txt`. This combines task Milestones 0 and 1
into the first major review gate: *environment / scaffold complete*.

## Short explanation

Milestone 0 filled the empty scaffold files and laid out the directory structure and
documentation skeleton described in `docs/PLAN.md`. Milestone 1 added a self-contained
environment-detection script that prints Python, PyTorch, CUDA, GPU, memory, package
versions, and an `nvidia-smi` snapshot, then writes the same report to a small text file
for reproducibility.

## Longer explanation

### Scaffold (Milestone 0)

- `pyproject.toml` — package metadata, `src/` layout, pytest + ruff config.
- `src/esm2_perf/__init__.py` — package docstring + version; future modules listed.
- `README.md` — portfolio-shaped draft with all required headings and an explicit
  "what this project does not claim" section.
- `docs/{METHODOLOGY,LIMITATIONS,REPRODUCIBILITY,RESULTS}.md` — TL;DR-first stubs.
- `docs/PLAN.md` — full nine-milestone plan, detected environment table, review gates.
- Directory keepers for `results/raw`, `results/processed`, `external`.
- `.gitignore` (already present) verified to exclude `.venv/`, caches, weights, and large
  profiler traces.

### Environment check (Milestone 1)

`scripts/00_check_environment.py`:

- Prints system info, PyTorch/CUDA details (version, availability, cuDNN), per-device GPU
  properties (name, compute capability, total/allocated/reserved memory, SM count), bf16
  support, selected package versions, and an `nvidia-smi` CSV snapshot.
- Fails gracefully: a missing torch, no CUDA device, or absent `nvidia-smi` produce
  readable messages rather than tracebacks, so the repo is inspectable on CPU-only hosts.
- `--output` writes the report to a file (creating parent dirs).

### Detected stack (this run)

NVIDIA L4, compute capability 8.9 (Ada), 24 GB (≈22.04 GiB visible to CUDA / 23034 MiB per
`nvidia-smi`), 58 SMs; Python 3.12.3; PyTorch 2.12.0 (CUDA 13.0, cuDNN 92000); transformers
5.10.2; triton 3.7.0; bf16 supported. Full detail in `results/raw/environment.txt`.

## Results

`results/raw/environment.txt` — captured environment report (small text file, committed).

## Validation

- `python scripts/00_check_environment.py --output results/raw/environment.txt` ran with
  exit code 0 and produced the expected report.
- CUDA is available and reports the L4 with bf16 support, confirming the GPU benchmark path
  in later milestones is viable.
- No ESM2 weights were downloaded yet (deferred to the baseline milestone).

## Limitations

- `nvidia-smi` reports total memory as 23034 MiB while PyTorch reports ~22.04 GiB visible;
  the difference is driver/runtime reserved memory, not an error.
- No model load or inference is exercised yet, so this milestone does not validate the
  transformers ESM2 code path.

## Next steps

Milestone 2 — baseline ESM2 inference benchmark: load
`facebook/esm2_t33_650M_UR50D`, generate synthetic sequences, and benchmark latency /
throughput / memory across dtypes, sequence lengths, and batch sizes with CUDA-event
timing. **Awaiting human approval before starting.**
