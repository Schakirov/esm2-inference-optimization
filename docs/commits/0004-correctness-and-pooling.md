# Commit 4: Correctness & pooling harness

## TL;DR

Add masked mean pooling (`esm2_perf.pooling`, three equivalent implementations) and a
correctness harness (`scripts/02_check_correctness.py`) that compares bf16/fp16 encoder
outputs against an fp32 reference for both hidden states and pooled embeddings. 9 new
CPU-only tests, a new `CORRECTNESS_COLUMNS` schema, and a captured 12-row result CSV.
Headline: fp16 tracks fp32 ~6× tighter than bf16; pooled embeddings stay cosine ≥ 0.9996.

## Short explanation

Pooling averages `last_hidden_state` over real amino-acid tokens (special tokens excluded by
default via `special_tokens_mask`). The harness runs the encoder in fp32/bf16/fp16 on
identical token ids, cross-checks the three pooling implementations on fp32 hidden states,
and records absolute/RMS/relative error plus min cosine similarity to a timestamped CSV.

## Longer explanation

Files changed:

- `src/esm2_perf/pooling.py` — `build_pool_mask` and three masked-mean-pool implementations
  (vectorized, `einsum`, explicit per-sequence reference); all-pad rows pool to zero.
- `src/esm2_perf/results.py` — new `CORRECTNESS_COLUMNS` schema (one comparison per row).
- `scripts/02_check_correctness.py` — CLI harness (`--quick`, `--dtypes`, `--seq-lens`,
  `--batch-size`, `--include-special`, `--rtol`, `--atol`, `--output`).
- `tests/test_pooling.py` — 9 CPU-only tests.
- `docs/milestones/03-correctness-and-pooling.md`, `docs/METHODOLOGY.md` (correctness
  protocol), `docs/review_notes/03-correctness-and-pooling-human-review.md`.
- `results/raw/correctness_20260613T102451Z.csv` — 12-row result.

Design decisions:

- **Three implementations on purpose.** The explicit per-sequence loop is an independent
  oracle (not a re-spelling of the vectorized expression), so it can validate the vectorized
  and `einsum` paths now and the Triton kernel in Milestone 6. They agree to ~2e-6 in fp32.
- **Two-phase memory hygiene.** The fp32 reference is computed first, stashed on the CPU, and
  released before bf16/fp16 load — one model resident at a time, mirroring the baseline.
- **Cosine is the headline, `allclose` is a reference flag.** `passed` uses a strict 1e-2
  element-wise gate; for embeddings the trustworthy metric is min cosine similarity. Max
  relative error is recorded but not gated on — near-zero hidden elements inflate it.
- **Special tokens excluded by default.** Pooling averages biological residues only;
  `--include-special` keeps `<cls>`/`<eos>`.

## Commands run

- `.venv/bin/python -m pytest -q` — 23 passed (14 prior + 9 new).
- `.venv/bin/python scripts/02_check_correctness.py --quick` — succeeded (smoke).
- `.venv/bin/python scripts/02_check_correctness.py --dtypes bf16 fp16 --seq-lens 128 512
  --batch-size 8` — succeeded; wrote 12-row CSV.

## Validation

All tests pass. Implementation equivalence confirmed at ~2e-6 (fp32). bf16 pooled max abs
error ~0.06–0.11 (cosine ≥ 0.9996, `passed=False` at the strict gate); fp16 pooled max abs
error ~0.01 (cosine ≥ 0.99999, `passed=True`). The fp16 ≫ bf16 accuracy gap is root-caused
to mantissa width (10-bit vs 7-bit) and documented rather than smoothed over.

## Next steps

Milestone 4 — variable-length batching and padding efficiency (awaiting human approval).
