# Human review: Environment and scaffold

## TL;DR

The first major milestone — *environment / scaffold complete* — is done. The repo has its
full directory structure, package skeleton, documentation skeleton, the nine-milestone
plan, and a working, reproducible environment-check script whose output is saved to
`results/raw/environment.txt`. Nothing has been pushed.

## What changed

Files and functionality added:

- `docs/PLAN.md` — full plan: detected-environment table, design principles, milestone
  table, repo layout, review-gate list.
- `pyproject.toml` — package metadata, `src/` layout, pytest + ruff config.
- `src/esm2_perf/__init__.py` — package docstring, version, module roadmap.
- `README.md` — portfolio-shaped draft (all required headings + "what it does not claim").
- `docs/{METHODOLOGY,LIMITATIONS,REPRODUCIBILITY,RESULTS}.md` — TL;DR-first stubs.
- `scripts/00_check_environment.py` — environment/GPU probe with `--output`.
- `results/raw/environment.txt` — captured environment report.
- `docs/milestones/01-environment-and-scaffold.md`, `docs/commits/0001-*.md`,
  `docs/commits/0002-*.md` — milestone and commit documentation.
- Directory keepers for `results/raw`, `results/processed`, `external`.

## What I should understand before continuing

- **Why a separate environment script?** Reproducible benchmarks must record the exact
  stack they ran on. The script captures Python/PyTorch/CUDA/GPU/package versions in one
  diffable artifact so any later number can be tied to a known environment.
- **The hardware.** NVIDIA L4 = Ada generation, compute capability 8.9, 24 GB, 58 SMs. It
  is a power-efficient *inference* GPU (72 W cap here), which is exactly why ESM2 embedding
  throughput on it is a realistic, interesting target. bf16 is supported in hardware.
- **The workload to come.** ESM2 produces per-token hidden states; downstream uses need one
  vector per protein, obtained by **masked mean pooling** over real amino-acid tokens
  (excluding padding and special tokens). Padding waste and dtype choice are the main perf
  levers we will measure.
- **Why ~22 GiB vs 23034 MiB?** PyTorch reports memory visible to the runtime; `nvidia-smi`
  reports board total. The gap is driver/runtime reservation, not a bug.

## Commands I should run manually

```bash
# 1. Re-run the environment check and view the captured report
.venv/bin/python scripts/00_check_environment.py

# 2. Inspect the saved artifact
cat results/raw/environment.txt

# 3. Confirm nothing large or secret is about to be tracked
git status
git ls-files | sed -n '1,40p'

# 4. Confirm the venv is ignored
git check-ignore .venv && echo ".venv correctly ignored"
```

## Questions I should be able to answer

- Why use `torch.cuda.Event` timing rather than `time.time()` for GPU benchmarks?
  (Asynchronous kernel launches — wall-clock would measure launch, not execution.)
- What is masked mean pooling and why exclude special/padding tokens from it?
- Why distinguish real amino-acid tokens/sec from padded tokens/sec?
- What is the compute capability of the L4 and why does it matter for dtype (bf16/fp16)
  and for Triton kernels?
- What would make a benchmark number misleading on this hardware (no warmup, no sync,
  measuring compile time as inference time, OOM masked as success)?

## Possible bugs or misleading benchmark artifacts

- The environment script reports the *metadata* torch version (`2.12.0`) in the package
  list and the *build* string (`2.12.0+cu130`) under PyTorch/CUDA — same wheel, two views;
  not a discrepancy.
- No ESM2 weights have been downloaded yet, so the transformers ESM2 code path is **not**
  yet validated; a model-load failure could still surface in Milestone 2.
- `requirements.txt` is unpinned; exact reproducibility relies on `environment.txt`.

## Human notes

I reviewed the scaffold and environment output. The machine is correctly detected as NVIDIA L4 with bf16 support, CUDA is visible through PyTorch, and the repo structure looks appropriate for a benchmark/optimization project.

I checked that `.venv/`, caches, model artifacts, secrets, and large profiler outputs are ignored. Milestone 1 is mostly infrastructure, so I am comfortable continuing to the first real benchmark milestone.
