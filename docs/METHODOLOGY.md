# Methodology

## TL;DR

How benchmarks in this repo are run so that results are trustworthy and reproducible:
CUDA-event GPU timing, explicit warmup, correct synchronization, deterministic synthetic
sequences, graceful OOM handling, and a clear split between *padded compute tokens/sec* and
*real amino-acid tokens/sec*.

## Short explanation

Each benchmark loads `facebook/esm2_t33_650M_UR50D`, builds synthetic inputs, runs warmup
iterations, then times steady-state inference with `torch.cuda.Event` pairs while holding
the GPU busy under `torch.inference_mode()`. Memory is read via
`torch.cuda.max_memory_allocated`. Results are written as timestamped CSVs under
`results/raw/`.

## Longer explanation

### Timing protocol (Milestone 2)

- **CUDA events, not wall clock.** Each timed iteration is bracketed by a fresh
  `torch.cuda.Event(enable_timing=True)` pair; after recording the end event we
  `torch.cuda.synchronize()` and read `start.elapsed_time(end)` (milliseconds). CUDA kernel
  launches are asynchronous, so host-side `time.time()` would measure launch overhead, not
  device execution.
- **Warmup.** A configurable number of untimed iterations (default 3) run first to absorb
  lazy CUDA init, cuDNN/cuBLAS autotuning, and allocator caching before measurement.
- **Steady state.** The default 10 timed iterations yield mean/median/std/min/max; the CSV
  reports median latency as the headline (robust to occasional outliers) plus mean and std.
- **Memory.** `reset_peak_memory_stats()` is called immediately before timing and
  `max_memory_allocated()` read after, so the figure reflects that cell's forward pass. The
  benchmark also fully releases a model (`del`, `gc.collect()`, `empty_cache()`,
  `synchronize()`) before loading the next dtype, so per-dtype peaks are not contaminated by
  previously-resident weights.
- **Inference mode.** All forwards run under `torch.inference_mode()` (no autograd state).
- **OOM handling.** `torch.cuda.OutOfMemoryError` is caught per cell, recorded as
  `oom=True`, and the matrix continues.

### Token accounting

`actual_tokens` is the sum of the attention mask — real, non-pad tokens, including the two
ESM2 special tokens (`<cls>`/`<eos>`). Throughput is reported as real tokens/sec and
sequences/sec. The split between real and padded tokens becomes central in the batching
milestone.

### Correctness protocol (Milestone 3)

Optimizations are only credible if they preserve the numbers. Before any speed work, the
correctness harness (`scripts/02_check_correctness.py`) pins down two things:

- **Pooling-implementation equivalence.** Three masked-mean-pool implementations — the
  vectorized production one, an `einsum` variant, and an explicit per-sequence reference —
  are compared on identical fp32 hidden states. They agree to ~2e-6 max absolute error
  (float32 summation-order noise), so any one can stand in for the others and the later
  Triton kernel has an unambiguous oracle.
- **Low-precision vs fp32 reference.** The encoder is run in fp32 (reference), bf16, and
  fp16 on the same token ids. We compare both the per-token `last_hidden_state` and the
  masked-mean-pooled embedding, reporting max/mean absolute error, RMS error, max relative
  error, and — for pooled embeddings — the minimum per-sequence cosine similarity.

**Masked mean pooling.** Pooling averages hidden vectors over real amino-acid tokens. The
pool mask is `attention_mask`, optionally with `<cls>`/`<eos>` removed via the tokenizer's
`special_tokens_mask` (the default; toggle with `--include-special`). A sequence with zero
selected tokens pools to a zero vector rather than NaN (the token count is clamped to 1).

**Why cosine similarity is the headline for pooled embeddings.** Element-wise
`torch.allclose` is reported (as `passed`) but is a deliberately strict, somewhat arbitrary
gate. For an *embedding*, what matters is direction, so the minimum cosine similarity across
the batch is the metric we trust. **Max relative error is intentionally not used as a gate:**
hidden states contain near-zero elements, so dividing by them inflates relative error to
meaningless magnitudes (1e4–1e5) even when absolute error is tiny. Absolute error, RMS error,
and cosine similarity are the trustworthy columns.

### Still to come

Padding-waste definition, batching strategies, `torch.compile` warmup vs steady-state
separation, and Triton kernel validation are documented as those milestones land.
