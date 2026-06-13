# Milestone 3: Correctness & pooling harness

## TL;DR

Before optimizing anything, we lock down what "correct" means. A new `esm2_perf.pooling`
module provides masked mean pooling over real amino-acid tokens in three equivalent forms
(vectorized, `einsum`, explicit reference) that agree to ~2e-6 in fp32. A correctness harness
(`scripts/02_check_correctness.py`) runs the ESM2 encoder in fp32/bf16/fp16 on identical
inputs and compares per-token hidden states and pooled embeddings against the fp32 reference.
Headline finding: **fp16 tracks fp32 far more tightly than bf16** (fp16 pooled max abs error
~0.01 and cosine ≥ 0.99999; bf16 pooled max abs error ~0.06–0.11 and cosine ≥ 0.9996).
Pooled embeddings are directionally near-identical in both low precisions. Results are in
`results/raw/correctness_20260613T102451Z.csv`.

## Short explanation

`src/esm2_perf/pooling.py` adds `build_pool_mask` (turn an attention mask + special-tokens
mask into a 0/1 pooling mask) and three masked-mean-pool implementations. The harness uses
them to (1) cross-check the implementations against each other on fp32 hidden states and (2)
measure how bf16/fp16 encoder outputs deviate from an fp32 reference, for both the full
`last_hidden_state` and the masked-mean-pooled sequence embedding. Errors are computed in
fp32 and written to a timestamped CSV with the new `CORRECTNESS_COLUMNS` schema. Nine new
CPU-only unit tests cover the pooling math and its edge cases.

## Longer explanation

### Files added / changed

- `src/esm2_perf/pooling.py` — `build_pool_mask`, `masked_mean_pool` (vectorized),
  `masked_mean_pool_einsum`, `masked_mean_pool_reference` (per-sequence). All-padding rows
  pool to a zero vector (count clamped to 1) rather than NaN.
- `src/esm2_perf/results.py` — new `CORRECTNESS_COLUMNS` schema for one comparison per row.
- `scripts/02_check_correctness.py` — CLI harness with `--quick`, `--model-name`,
  `--dtypes`, `--seq-lens`, `--batch-size`, `--include-special`, `--rtol`, `--atol`,
  `--output`.
- `tests/test_pooling.py` — 9 CPU-only tests (shape, known value, masking, all-pad edge
  case, three-way implementation agreement, special-token exclusion).
- `docs/METHODOLOGY.md` — new "Correctness protocol" section.

### Masked mean pooling

The sequence embedding is the mean of `last_hidden_state` over real amino-acid tokens. The
pool mask starts from the tokenizer `attention_mask` and, by default, removes `<cls>`/`<eos>`
using `special_tokens_mask` (so we average biological residues only; `--include-special`
keeps them). The three implementations exist so they can validate each other and, in
Milestone 6, the Triton kernel — the explicit per-sequence loop is the independent oracle,
not a re-spelling of the vectorized expression.

### Correctness protocol

The harness runs in two phases to keep memory honest (one model resident at a time, as in
the baseline):

1. **fp32 reference.** Load fp32, forward each cell, stash hidden + pooled on the CPU, and
   record the implementation-equivalence rows. Release the fp32 model.
2. **Low precision.** Load bf16, then fp16; forward the same token ids; compare against the
   stashed fp32 reference.

Per comparison we record max/mean absolute error, RMS error, max relative error, and (for
pooled embeddings) the minimum per-sequence cosine similarity. `passed` is
`torch.allclose(rtol=1e-2, atol=1e-2)` — a strict, somewhat arbitrary element-wise gate kept
for reference; the trustworthy signal for an embedding is cosine similarity.

### Results

Source: `results/raw/correctness_20260613T102451Z.csv` (12 rows), batch size 8, special
tokens excluded.

Implementation equivalence (fp32 hidden states):

| check | seq_len | max_abs_err | min_cosine |
|-------|--------:|------------:|-----------:|
| vectorized vs reference | 128 | 4.8e-07 | 0.99999994 |
| einsum vs reference | 128 | 1.9e-06 | 0.99999994 |
| vectorized vs reference | 512 | 4.8e-07 | 0.99999982 |
| einsum vs reference | 512 | 1.9e-06 | 0.99999982 |

Low precision vs fp32 (pooled embeddings):

| dtype | seq_len | max_abs_err | mean_abs_err | min_cosine | passed |
|-------|--------:|------------:|-------------:|-----------:|:------:|
| bf16 | 128 | 0.0604 | 0.00188 | 0.99979 | False |
| bf16 | 512 | 0.1059 | 0.00189 | 0.99965 | False |
| fp16 | 128 | 0.0107 | 0.00028 | 0.99999 | True |
| fp16 | 512 | 0.0116 | 0.00022 | 0.99999 | True |

For the full `last_hidden_state`, bf16 max abs error is ~0.32–0.46 and fp16 ~0.03–0.06.

Observations:

- **The three pooling implementations are interchangeable.** ~2e-6 max error in fp32 is pure
  summation-order round-off. The vectorized one is the production path; the reference is the
  oracle for the Triton kernel later.
- **fp16 ≫ bf16 in accuracy here.** fp16's 10-bit mantissa tracks fp32 about 6× tighter than
  bf16's 7-bit mantissa. bf16's advantage is dynamic range, which this well-scaled encoder
  does not need — so fp16 is the more faithful low-precision choice for embedding fidelity.
- **Pooled embeddings are directionally near-identical** in both precisions (cosine ≥ 0.9996),
  even where strict element-wise `allclose` fails. Averaging over tokens cancels much of the
  per-element noise.
- **bf16 "fails" the strict gate, and that is fine.** `passed=False` reflects a 1e-2
  element-wise threshold, not a meaningful embedding error. This is documented rather than
  hidden, and is exactly why cosine similarity is the headline metric.

## Validation

- `pytest -q` → 23 passed (14 prior + 9 new pooling tests).
- `python scripts/02_check_correctness.py --quick` → single-cell smoke run succeeded.
- Full run (bf16+fp16 × seq {128, 512}, batch 8) → 12 rows, no errors.
- Implementation equivalence re-confirmed at ~2e-6; fp16/bf16 deviations root-caused to
  mantissa width.

## Limitations

- Synthetic uniform-random amino-acid sequences; real proteomes differ in composition (does
  not affect the dtype-vs-dtype numerical relationship measured here).
- `passed` uses a single global `rtol/atol`; it is a coarse reference flag, not a per-tensor
  acceptance criterion. Max relative error is reported but not gated on (near-zero hidden-state
  elements inflate it). Cosine similarity and absolute/RMS error are the trustworthy columns.
- Comparison is against an fp32 *eager* reference on this same L4 — not against a CPU or a
  different GPU. We measure low-vs-high precision divergence, not absolute ground truth.
- Pooling is mean only; other poolings (`<cls>`, max, attention) are out of scope.

## Next steps

Milestone 4 — variable-length batching and padding efficiency: quantify padding waste with
`generate_variable_sequences`, compare batching strategies, and report real vs padded
tokens/sec. **Awaiting human approval before starting.**
