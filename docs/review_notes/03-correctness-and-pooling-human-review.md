# Human review: Correctness & pooling harness

## TL;DR

The third major milestone — *correctness / pooling harness complete* — is done. A new
`esm2_perf.pooling` module (three equivalent masked-mean-pool implementations) and
`scripts/02_check_correctness.py` validate that (a) the pooling implementations agree to
~2e-6 in fp32 and (b) bf16/fp16 encoder outputs stay close to an fp32 reference. The
headline numerical finding: **fp16 is ~6× more faithful to fp32 than bf16 here**, while both
keep pooled-embedding cosine similarity ≥ 0.9996. 23 unit tests pass. Results in
`results/raw/correctness_20260613T102451Z.csv`. Nothing pushed.

## What changed

- `src/esm2_perf/pooling.py` — `build_pool_mask` + `masked_mean_pool` (vectorized),
  `masked_mean_pool_einsum`, `masked_mean_pool_reference` (per-sequence oracle).
- `src/esm2_perf/results.py` — `CORRECTNESS_COLUMNS` schema (one comparison per row).
- `scripts/02_check_correctness.py` — CLI harness with `--quick` and matrix/tolerance flags.
- `tests/test_pooling.py` — 9 fast CPU-only tests.
- `docs/milestones/03-correctness-and-pooling.md`, `docs/METHODOLOGY.md` (correctness
  protocol), `docs/commits/0004-correctness-and-pooling.md`.
- `results/raw/correctness_20260613T102451Z.csv` — 12-row result.

## What I should understand before continuing

- **What "correct" means here.** We do not have an external ground truth; we compare
  low-precision (bf16/fp16) against an fp32 *eager* reference on the same L4 and same token
  ids. The harness measures precision-induced divergence, not absolute truth.
- **Three pooling implementations.** Vectorized (production), `einsum`, and an explicit
  per-sequence loop. The loop is an independent oracle, so it can certify the fast paths now
  and the Triton kernel in Milestone 6. They agree to ~2e-6 in fp32 — pure summation-order
  round-off.
- **Pooling excludes special tokens by default.** The mean is over real amino-acid residues;
  `<cls>`/`<eos>` are dropped via the tokenizer `special_tokens_mask` (`--include-special`
  keeps them). All-padding rows pool to a zero vector, not NaN.
- **fp16 ≫ bf16 in accuracy.** fp16's 10-bit mantissa tracks fp32 far tighter than bf16's
  7-bit mantissa; bf16's edge is dynamic range, which this well-scaled encoder doesn't need.
  So for embedding *fidelity*, fp16 is the better low-precision choice (throughput was
  equal in Milestone 2).
- **Cosine vs `allclose`.** `passed` is a strict 1e-2 element-wise gate kept for reference.
  For an embedding, direction is what matters, so min cosine similarity is the metric to
  trust — bf16 "fails" the gate yet stays cosine ≥ 0.9996.

## Commands I should run manually

```bash
# Fast smoke (single cell, bf16 only), proves the path end to end
.venv/bin/python scripts/02_check_correctness.py --quick

# A small real slice (~1–2 min): bf16 + fp16, two lengths
.venv/bin/python scripts/02_check_correctness.py --dtypes bf16 fp16 --seq-lens 128 512 --batch-size 8 --output results/raw/_review.csv

# Inspect the committed results
column -s, -t results/raw/correctness_20260613T102451Z.csv | less -S

# Unit tests
.venv/bin/python -m pytest -q
```

## Questions I should be able to answer

- Why do the three pooling implementations differ at all (~2e-6)? (Floating-point summation
  order; the reductions are mathematically identical but accumulate in different orders.)
- Why is fp16 more accurate than bf16 here when Milestone 2 showed equal throughput?
  (10-bit vs 7-bit mantissa → ~6× lower absolute error; throughput is set by the 2-byte
  Tensor Core path, accuracy by mantissa width.)
- Why does bf16 show `passed=False` but I should not be alarmed? (The gate is a strict 1e-2
  element-wise `allclose`; the embedding's cosine similarity is ≥ 0.9996 — directionally
  near-identical. The gate is a reference, not the acceptance criterion.)
- Why is max relative error huge (1e4–1e5) even when absolute error is tiny? (Hidden states
  contain near-zero elements; dividing by them blows up relative error. That column is
  reported but deliberately not used as a gate.)
- How is the fp32 reference kept from inflating memory? (Computed first, stashed on CPU,
  released before bf16/fp16 load — one model resident at a time.)

## Possible bugs or misleading benchmark artifacts

- **Max relative error is not trustworthy.** Near-zero denominators inflate it; use absolute
  error, RMS error, and cosine similarity instead. Documented, not gated.
- The `passed` column depends entirely on the chosen `rtol/atol` (default 1e-2/1e-2). It is a
  coarse single-threshold flag across both hidden states and pooled embeddings — read it as a
  hint, not a verdict.
- Reference is fp32 *eager* on this same GPU, not a CPU or alternate device; this is a
  relative precision check, not validation against an independent ground truth.
- Synthetic uniform-random sequences are not biologically realistic (fine for the dtype-vs-
  dtype numerical relationship measured here).
- If a later change accidentally pools over `pooler_output` (randomly initialized, unused) or
  forgets to exclude special tokens, the embeddings would shift — the harness would catch the
  latter via the `exclude_special` column.

## Human notes

_Add review outcome / approval here before Milestone 4 begins._
