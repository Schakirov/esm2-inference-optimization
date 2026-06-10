# Milestone 2: Baseline ESM2 inference benchmark

## TL;DR

A clean baseline benchmark loads `facebook/esm2_t33_650M_UR50D` and measures forward-pass
latency, throughput, and peak GPU memory across bf16/fp16, sequence lengths {128, 256, 512,
1022}, and batch sizes {1, 2, 4, 8, 16} on the NVIDIA L4. Timing uses CUDA events with
warmup and synchronization under `torch.inference_mode()`. The full 40-cell matrix runs
with no OOM; peak memory tops out at ~2.0 GB and real-token throughput saturates around
**21–22k tokens/sec**. Results are in `results/raw/baseline_20260610T101812Z.csv`.

## Short explanation

`scripts/01_bench_baseline.py` drives the benchmark using three small library modules:
`sequences.py` (deterministic synthetic amino-acid sequences), `timing.py` (CUDA-event
timing with warmup), and `results.py` (CSV schema + writer). For each (dtype, seq_len,
batch_size) cell it builds a same-length batch (so there is no intra-batch padding here),
times the forward pass, and records latency, real-token and sequence throughput, and peak
allocated memory. OOM is captured per cell instead of aborting the run.

## Longer explanation

### Files added

- `src/esm2_perf/sequences.py` — `generate_sequences` / `generate_variable_sequences` over
  the 20 standard amino acids, seeded for reproducibility.
- `src/esm2_perf/timing.py` — `time_cuda(fn, warmup, iters)` returning mean/median/std/min/
  max milliseconds via `torch.cuda.Event` pairs with per-iteration synchronization.
- `src/esm2_perf/results.py` — `BASELINE_COLUMNS`, `write_csv` (schema-validated), and
  timestamped-path helpers.
- `scripts/01_bench_baseline.py` — CLI benchmark with `--quick`, `--model-name`,
  `--dtypes`, `--seq-lens`, `--batch-sizes`, `--warmup`, `--iters`, `--output`.
- `tests/test_sequences.py`, `tests/test_results.py` — 14 fast unit tests.

### Method notes

- The model is loaded with `AutoModel` (the `EsmModel` encoder), not the masked-LM head.
  The load log reports `lm_head.*` as UNEXPECTED (we don't load the LM head) and
  `pooler.*` as MISSING/newly-initialized; this is harmless because we read only
  `last_hidden_state` and never use the randomly-initialized pooler.
- Each cell calls `reset_peak_memory_stats()` before timing and reads
  `max_memory_allocated()` after, so memory reflects that cell's forward pass.
- `actual_tokens` is the sum of the attention mask (real, non-pad tokens). Because every
  sequence in a cell shares one length, there is no padding here; padding efficiency is the
  subject of Milestone 4. Tokens include the two ESM2 special tokens (`<cls>`/`<eos>`),
  e.g. seq_len 128 → 130 tokens per sequence.

### A measurement bug found and fixed during this milestone

The first full run reported fp16 peak memory (~2.4 GB) as roughly double bf16 (~1.2 GB),
even though both dtypes store 2-byte weights. Running fp16 in isolation gave ~1.2 GB,
revealing the cause: the bf16 model was not being released before the fp16 model loaded, so
the second dtype's peak counted *both* resident weight sets. The fix explicitly releases the
previous model (`del model, tokenizer; gc.collect(); empty_cache(); synchronize()`) between
dtypes. After the fix, both dtypes report identical ~1.2 GB at equal shapes, as expected.
This is documented because it is exactly the kind of artifact that would silently inflate a
memory comparison.

## Results

Source: `results/raw/baseline_20260610T101812Z.csv` (40 rows, bf16 + fp16).

Representative bf16 cells (median latency / real tokens-per-sec / peak GB):

| seq_len | bs | latency_ms | tokens/sec | peak_GB |
|--------:|---:|-----------:|-----------:|--------:|
| 128 | 1 | 20.2 | 6.4k | 1.23 |
| 128 | 16 | 98.1 | 21.2k | 1.32 |
| 512 | 1 | 25.2 | 20.4k | 1.25 |
| 512 | 16 | 514.7 | 16.0k | 1.61 |
| 1022 | 1 | 46.5 | 22.0k | 1.27 |
| 1022 | 16 | 1066.5 | 15.4k | 2.00 |

Observations:

- **Overhead floor.** At small total work (e.g. seq 128, bs 1–2) latency is pinned near
  ~20 ms regardless of token count — the regime is launch/overhead-bound, not compute-bound.
- **Throughput saturates.** Real-token throughput climbs with batch/length and plateaus
  around 21–22k tokens/sec, then dips slightly at the largest cells as quadratic attention
  cost grows with sequence length.
- **bf16 ≈ fp16.** The two dtypes are within noise; bf16 is marginally faster at the largest
  shapes. Memory is identical (both 2-byte weights).
- **L4 is far from memory-bound here.** Peak ~2.0 GB of 24 GB, so no OOM in this matrix;
  larger batches/lengths are headroom for later milestones.

## Validation

- `pytest -q` → 14 passed.
- `python scripts/01_bench_baseline.py --quick` → single-cell smoke run succeeded.
- Full 40-cell matrix → completed with no OOM and no per-cell errors.
- Memory artifact identified, root-caused, fixed, and re-verified (fp16 == bf16 at equal
  shapes).

## Limitations

- Inputs are synthetic uniform-random amino-acid sequences; real proteomes have different
  length and composition statistics (affects later batching/padding results, not the fixed-
  shape latencies here).
- Same-length batches mean these numbers exclude padding overhead by construction.
- Throughput is for a single forward pass of the encoder only (no tokenization, no pooling,
  no data loading); end-to-end pipelines will be slower.
- fp32 was not included in the default matrix (slow/large and not the production dtype); the
  script supports `--dtypes fp32` for small cells if desired.

## Next steps

Milestone 3 — correctness harness and embedding pooling: implement masked mean pooling,
compare fp16/bf16 against an fp32 eager reference with documented tolerances, and validate
equivalent pooling implementations. **Awaiting human approval before starting.**
