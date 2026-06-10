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

### Still to come

Padding-waste definition, batching strategies, `torch.compile` warmup vs steady-state
separation, and Triton kernel validation are documented as those milestones land.
