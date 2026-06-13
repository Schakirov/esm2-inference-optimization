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

### Padding waste & batching (Milestone 4)

Real proteomes have varied sequence lengths. Batching forces every sequence to be padded to
the longest in its batch, so the GPU computes over padding tokens that carry no signal.

- **Padding metrics.** For a batch, *padded tokens* = `batch_size × max_length_in_batch`;
  *real tokens* = sum of the per-sequence token counts. Over a whole pool,
  `padding_fraction = (padded − real) / padded` (the share of compute wasted on padding) and
  `padding_waste_ratio = padded / real` (≥ 1.0). Real tokens are independent of how
  sequences are grouped; padded tokens are not.
- **Strategies.** `naive` keeps the pool's arbitrary order; `sorted` sorts by length first so
  each batch holds similar-length sequences and pads less. Planning is pure and unit-tested on
  the CPU (`esm2_perf.batching`); only the timing touches the GPU.
- **Two throughput numbers.** Throughput is reported as both *real tokens/sec* (useful work)
  and *padded compute tokens/sec* (raw GPU rate). Length-sorting leaves the padded rate about
  unchanged but raises the real rate — that gap is the recovered waste, and the honest way to
  state the win.
- **Timing.** A fixed pool of variable-length sequences is shared across all cells so only the
  grouping changes. Each cell's batches are pre-tokenized onto the GPU, then one full pass
  over all batches is timed with CUDA events under `torch.inference_mode()` (warmup + median
  over iterations), mirroring the baseline timing protocol.

### torch.compile: cold start vs steady state (Milestone 5)

`torch.compile` trades a large one-time cost for a (hoped-for) faster steady state. Conflating
the two is the classic way to mis-measure it, so the harness (`scripts/04_bench_compile.py`)
keeps them separate:

- **Cold start.** The first call at a given input shape, measured on the **wall clock** — for
  compiled modes its cost is dominated by host-side compilation, which CUDA events (device
  timing) would not capture. With static shapes (`dynamic=False`, the default), each new
  `(batch, seq_len)` recompiles, so cold start also exposes recompilation cost.
- **Steady state.** CUDA-event-timed latency after warmup at the same shape — the rate you
  actually get once compiled. `speedup_vs_eager` is the eager steady latency divided by the
  compiled steady latency at the *same* shape; eager is always run first to supply that
  baseline.

**Modes.** `eager` (reference), `default` (Inductor), and optionally `reduce-overhead`
(CUDA graphs) and `max-autotune`. Dynamo state is reset between modes so each mode's cold
start reflects only its own compilation. **Static vs dynamic shapes** is a first-class axis:
`--dynamic` asks for shape-generic kernels (fewer recompiles, sometimes slower) instead of
per-shape specialization.

**Honest accounting.** The cold-start cost must be amortized over many calls to pay off, and
small/overhead-bound shapes often do not benefit at all. When compile does not beat eager for
a shape, the row records it (`speedup_vs_eager < 1`) rather than hiding it.

### Profiling: the kernel-level view (Milestone 7)

`torch.profiler` is run with CUDA+CPU activities over warm iterations, but the reported
breakdown deliberately uses a **kernel-only view**. `key_averages()` lists both the operator
events (`aten::addmm`, …) and the device kernels they launch; summing both double-counts GPU
time (we first measured ~449 ms/iter, almost exactly 2× the real forward). Filtering to leaf
CUDA kernels (keys not starting with `aten::`/`cuda`) gives self-times that **sum to the
measured forward latency** — the cross-check that the attribution is honest. Kernels are rolled
up into coarse categories (matmul / elementwise / copy / norm) by name; the split is approximate
but the matmul-vs-rest headline is robust. Only the small top-ops CSV is committed — the full
Chrome trace is large and stays out of the repo.

### Triton kernel validation (Milestone 6)

A custom kernel is only worth anything if it is correct, so the Triton masked-mean-pooling
kernel (`esm2_perf.triton_pooling`) is validated *before* it is benchmarked:

- **Against the oracle.** The kernel is tested against the Milestone 3 per-sequence reference
  (`masked_mean_pool_reference`) and the vectorized PyTorch pooling, to ≤1e-4 rtol / 1e-5 atol
  on fp32 inputs across several shapes — this is exactly why the independent reference was
  built in Milestone 3. Edge cases (all-padding rows → zeros, not NaN; `out_dtype`) are tested
  too. The tests require CUDA + Triton and skip on CPU-only machines.
- **Correctness reported alongside speed.** The benchmark records `max_abs_err_vs_ref` next to
  latency for every implementation, so a fast-but-wrong kernel cannot hide. The kernel
  accumulates in fp32, so its error vs the fp32 reference is round-off (~1e-8) while the bf16
  PyTorch paths carry ~1e-3.
- **Honest baselines and scope.** Speedup is reported vs the eager vectorized PyTorch pooling,
  but the analysis also compares against the *best* PyTorch option (`einsum`/`compiled`) so a
  weak baseline does not inflate the headline. Pooling is a tiny fraction of encoder time, so
  the milestone states plainly that the kernel does not move end-to-end throughput.
