# Human review: Baseline ESM2 inference benchmark

## TL;DR

The second major milestone — *baseline ESM2 benchmark complete* — is done. A CUDA-event
timed, OOM-safe benchmark sweeps bf16/fp16 × {128,256,512,1022} × {1,2,4,8,16} on the L4
and writes `results/raw/baseline_20260610T101812Z.csv` (40 rows). 14 unit tests pass. A
real memory-measurement artifact was found and fixed before saving final numbers. Nothing
pushed.

## What changed

- `src/esm2_perf/sequences.py` — deterministic synthetic amino-acid sequences.
- `src/esm2_perf/timing.py` — `time_cuda` CUDA-event timing (warmup + per-iter sync).
- `src/esm2_perf/results.py` — CSV schema (`BASELINE_COLUMNS`) + schema-validated writer.
- `scripts/01_bench_baseline.py` — CLI baseline benchmark with `--quick` and a full set of
  matrix/output flags.
- `tests/test_sequences.py`, `tests/test_results.py` — 14 fast tests.
- `docs/milestones/02-baseline-esm2-inference.md`, `docs/METHODOLOGY.md` (timing protocol),
  `docs/commits/0003-baseline-benchmark.md`.
- `results/raw/baseline_20260610T101812Z.csv` — results (4.7 KB).

## What I should understand before continuing

- **Why CUDA events.** Kernel launches are async; `time.time()` around a launch measures
  almost nothing. CUDA events recorded on the stream, read after a `synchronize`, give true
  device time. Warmup absorbs lazy init / autotuning so we measure steady state.
- **Real vs padded tokens.** Here every sequence in a cell is the same length, so there is
  *no* padding and `actual_tokens` is the full token count. Padding waste is deliberately
  out of scope until Milestone 4 — these baseline latencies are the clean, no-waste
  reference.
- **What the numbers say.** Small workloads sit on a ~20 ms latency floor (launch/overhead
  bound). Throughput rises with batch/length and plateaus near 21–22k real tokens/sec, then
  dips at the largest cells as attention's quadratic length cost grows. bf16 ≈ fp16; memory
  is identical (both 2-byte weights) and tiny vs the 24 GB board.
- **The encoder vs LM head.** We load `AutoModel`/`EsmModel` and use `last_hidden_state`;
  the `pooler` reported MISSING at load is randomly initialized and intentionally unused.

## Commands I should run manually

```bash
# Fast smoke (single cell), proves the path end to end
.venv/bin/python scripts/01_bench_baseline.py --quick

# A small real slice you can eyeball in ~1 minute
.venv/bin/python scripts/01_bench_baseline.py --dtypes bf16 --seq-lens 128 512 --batch-sizes 1 8 --output results/raw/_review.csv

# Inspect the committed full results
column -s, -t results/raw/baseline_20260610T101812Z.csv | less -S

# Unit tests
.venv/bin/python -m pytest -q
```

## Questions I should be able to answer

- Why is latency ~constant (~20 ms) for the smallest cells regardless of token count?
  (Kernel-launch / fixed-overhead bound; not enough work to saturate the GPU.)
- Why does real-token throughput *decline* at the largest seq_len × batch cells?
  (Self-attention is O(L²); longer sequences cost disproportionately more per token.)
- Why are bf16 and fp16 essentially equal here, and when might they diverge?
  (Same 2-byte footprint and Tensor Core path on Ada; divergence shows up in numerical
  accuracy, not throughput — that's Milestone 3.)
- How do I know a cell didn't silently OOM? (The `oom` column; OOM is caught and recorded,
  not masked.)
- Why measure median latency rather than mean? (Robust to occasional scheduling outliers;
  mean and std are also recorded.)

## Possible bugs or misleading benchmark artifacts

- **Fixed (documented).** Cross-dtype peak-memory contamination: the first dtype's model
  was still resident when the second loaded, doubling the second's reported peak. Fixed by
  explicit release between dtypes; re-verified that fp16 == bf16 at equal shapes. If you re-
  run, confirm memory does not jump for the second dtype.
- Synthetic uniform-random sequences are not biologically realistic; fine for fixed-shape
  latency, but length/composition statistics will matter for batching (Milestone 4).
- Numbers are encoder-forward-only — no tokenization, pooling, or I/O — so they are an upper
  bound on a real embedding pipeline's throughput.
- The randomly-initialized `pooler` is unused; if a later change accidentally reads
  `pooler_output`, results would be meaningless. We only use `last_hidden_state`.

## Human notes

Reviewed Milestone 2. Results are plausible for L4: low-batch cases are overhead-bound, throughput improves with batching, and the largest 
  sequence-length/batch-size cases show the expected attention-related slowdown. It is a baseline for later correctness, batching, compile, profiling, and
  Triton experiments.
