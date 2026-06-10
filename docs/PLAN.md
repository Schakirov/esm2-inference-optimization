# Implementation Plan

## TL;DR

This repo demonstrates reproducible ML performance engineering for **ESM2 protein
embedding extraction** on a single **NVIDIA L4** GPU. The plan proceeds through nine
milestones: scaffold → environment check → baseline benchmark → correctness/pooling →
variable-length batching/padding → `torch.compile` → a custom Triton masked-mean-pooling
kernel → profiling → portfolio report. Work is committed locally after each coherent
increment and **stops at every major-milestone boundary for human review** (no pushes).

## Short explanation

The target model is `facebook/esm2_t33_650M_UR50D` (650M params, hidden size 1280). The
core workload is extracting per-token hidden states and **masked mean-pooled** sequence
embeddings over real amino-acid tokens. Each milestone produces code under
`src/esm2_perf/` and `scripts/`, machine-readable results under `results/raw/`, tests under
`tests/`, and documentation under `docs/`. We prioritize correctness, reproducibility, and
honest reporting over headline-grabbing kernel wins.

## Longer explanation

### Detected environment (2026-06-10)

| Item | Value |
|------|-------|
| Host | AWS `g6.2xlarge` |
| GPU | NVIDIA L4, 24 GB VRAM (23034 MiB), compute capability 8.9 (Ada) |
| Driver / CUDA (smi) | 595.71.05 / 13.2 |
| Python | 3.12.3 (`.venv`) |
| PyTorch | 2.12.0+cu130 (CUDA 13.0) |
| transformers | 5.10.2 |
| triton | 3.7.0 |
| bf16 supported | yes |
| Disk free | ~167 GB |
| System RAM | 30 GB |

These facts are re-captured programmatically in Milestone 1 so results are reproducible.

### Design principles

- **Correctness first.** Optimizations are validated against an fp32 eager reference with
  documented numerical tolerances before any speed claim is made.
- **Real vs padded tokens.** Throughput is reported both as padded compute tokens/sec and
  as *real biological amino-acid tokens/sec*; padding waste is a first-class metric.
- **Honest benchmarking.** CUDA events for GPU timing, explicit warmup, correct
  synchronization, graceful OOM capture, deterministic seeds, timestamped result files.
- **No overclaiming.** If `torch.compile` or the Triton kernel does not beat PyTorch, we
  document why. Limitations live in `docs/LIMITATIONS.md`.
- **Repo hygiene.** No weights, venvs, caches, secrets, or large traces committed. Only
  small summarized artifacts.

### Milestones

| # | Title | Key outputs |
|---|-------|-------------|
| 0 | Safety, inspection, scaffold | dirs, `pyproject.toml`, `requirements.txt`, README draft, `.gitignore`, this plan |
| 1 | Environment & GPU check | `scripts/00_check_environment.py`, `results/raw/environment.txt` |
| 2 | Baseline ESM2 inference | `scripts/01_bench_baseline.py`, `src/esm2_perf/{timing,sequences,results}.py`, baseline CSV |
| 3 | Correctness & pooling harness | `src/esm2_perf/pooling.py`, `scripts/02_check_correctness.py`, tests |
| 4 | Variable-length batching/padding | `src/esm2_perf/batching.py`, `scripts/03_bench_batching.py`, batching CSV |
| 5 | `torch.compile` & static shapes | `scripts/04_bench_compile.py`, compile CSV |
| 6 | Triton masked-mean-pooling kernel | `src/esm2_perf/triton_pooling.py`, tests, `scripts/05_bench_triton_pooling.py` |
| 7 | Profiling notes | `scripts/06_profile_pytorch.py`, `docs/PROFILING.md` |
| 8 | Result processing & portfolio README | `scripts/07_summarize_results.py`, `docs/RESULTS.md`, README |
| 9 | Final cleanup | tests/format/status sweep, final commit |

### Repository layout

```
src/esm2_perf/      # reusable library: timing, sequences, results, pooling, batching, triton
scripts/            # CLI-friendly benchmark/check scripts (NN_*.py), all with --quick
tests/              # small, fast pytest unit tests
results/raw/        # timestamped CSV / txt outputs (committed if small)
results/processed/  # summarized tables
docs/               # PLAN, RESULTS, METHODOLOGY, LIMITATIONS, REPRODUCIBILITY, PROFILING
docs/commits/       # NNNN-*.md per commit
docs/milestones/    # XX-*.md per milestone
docs/review_notes/  # XX-*-human-review.md at each major-milestone gate
```

### Major milestones & human review gates

After each **major milestone** below, work stops and waits for explicit human approval
before the next one begins. A review note is written to `docs/review_notes/` at each gate.

1. Environment / scaffold complete  ← **first gate (current target)**
2. Baseline ESM2 benchmark complete
3. Correctness / pooling harness complete
4. Batching / padding benchmark complete
5. `torch.compile` benchmark complete
6. Triton pooling kernel complete
7. Profiling workflow complete
8. Final README / results report complete

### Conventions

- Benchmark scripts are CLI-friendly with `--quick`, `--model-name`, `--dtype`,
  `--batch-sizes`, `--seq-lens`, `--output` where reasonable.
- Synthetic protein sequences use the 20 standard amino acids with deterministic seeds.
- Result filenames carry timestamps (e.g. `baseline_<timestamp>.csv`).
- Docs start with TL;DR, then short explanation, then longer explanation.
- Always run `git status` before committing; stage only intended files; never push.
