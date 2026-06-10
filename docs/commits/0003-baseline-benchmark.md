# Commit 3: Baseline ESM2 inference benchmark

## TL;DR

Add the baseline benchmark for `facebook/esm2_t33_650M_UR50D`: three reusable library
modules (`sequences`, `timing`, `results`), the CLI driver `scripts/01_bench_baseline.py`,
14 unit tests, and a captured 40-cell result CSV. CUDA-event timed, OOM-safe, deterministic.

## Short explanation

The benchmark sweeps bf16/fp16 × seq_len {128,256,512,1022} × batch {1,2,4,8,16}, timing
the encoder forward pass with CUDA events under `torch.inference_mode()` and recording
latency, real-token/sequence throughput, and peak memory to a timestamped CSV.

## Longer explanation

Files changed:

- `src/esm2_perf/sequences.py` — deterministic synthetic amino-acid sequence generation.
- `src/esm2_perf/timing.py` — `time_cuda` CUDA-event timing with warmup/sync.
- `src/esm2_perf/results.py` — `BASELINE_COLUMNS`, schema-validated `write_csv`,
  timestamped-path helpers.
- `scripts/01_bench_baseline.py` — CLI driver (`--quick`, `--dtypes`, `--seq-lens`,
  `--batch-sizes`, `--warmup`, `--iters`, `--output`).
- `tests/test_sequences.py`, `tests/test_results.py` — fast unit tests.
- `docs/milestones/02-baseline-esm2-inference.md`, `docs/METHODOLOGY.md` (timing protocol),
  `docs/review_notes/02-baseline-human-review.md`.
- `results/raw/baseline_20260610T101812Z.csv` — 40-row result (4.7 KB).

Design decisions and a fixed bug:

- Load `AutoModel` (encoder) only; we read `last_hidden_state`, so the UNEXPECTED `lm_head`
  and MISSING/newly-initialized `pooler` reported at load are harmless.
- **Memory artifact fixed:** the previous dtype's model was not released before the next
  loaded, doubling the reported peak for the second dtype. Now each model is explicitly
  freed (`del`, `gc.collect`, `empty_cache`, `synchronize`) between dtypes; fp16 and bf16
  then report identical memory at equal shapes.

## Commands run

- `.venv/bin/python -m pytest -q` — 14 passed.
- `.venv/bin/python scripts/01_bench_baseline.py --quick` — succeeded (smoke).
- `.venv/bin/python scripts/01_bench_baseline.py --dtypes bf16 fp16 --seq-lens 128 256 512
  1022 --batch-sizes 1 2 4 8 16` — succeeded; wrote 40-row CSV, no OOM.

## Validation

All tests pass. Full matrix completes with no OOM or per-cell errors. Peak memory ~2.0 GB
of 24 GB; real-token throughput saturates ~21–22k tokens/sec; bf16 ≈ fp16. Memory artifact
root-caused and re-verified.

## Next steps

Milestone 3 — correctness harness and embedding pooling (awaiting human approval).
