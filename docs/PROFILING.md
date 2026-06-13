# Profiling the ESM2 encoder forward

## TL;DR

`torch.profiler` on the ESM2-650M encoder (bf16, seq 512, batch 8 on the L4) attributes the
forward to GPU kernels that sum to **225 ms/iter — matching the measured Milestone 2 latency**,
which validates the attribution. The time splits almost evenly between two halves:
**matmul ≈ 42%** (the FFN/projection linears dominate; flash-attention is only ~4.7%) and
**unfused elementwise ≈ 43%** (each GeLU, bias-add, residual, and scale is its own kernel doing
a full HBM round-trip). Tensor concatenations + dtype casts add ~14% (rotary-embedding `cat`,
bf16 copies); LayerNorm is ~1%. The large unfused-elementwise share is exactly the headroom
`torch.compile` exploited in [Milestone 5](milestones/05-torch-compile.md) — **the profile
explains the compile win.** Source: `scripts/06_profile_pytorch.py`,
`results/raw/profile_20260613T171924Z.csv`.

## Short explanation

The script warms up, then runs 10 forwards under `torch.profiler` with CUDA + CPU activities.
It reports a **kernel-level** view: leaf CUDA kernels only (no `aten::` dispatcher rows), so GPU
time is not double-counted, and the per-kernel self-times sum to the true forward time. Kernels
are rolled up into coarse categories by name. Only the small top-ops CSV is committed; the full
Chrome trace is optional via `--trace` and is not committed (it is large).

## Longer explanation

### Setup

- Model `facebook/esm2_t33_650M_UR50D`, bf16, **seq 512 × batch 8** (a compute-bound shape, so
  the breakdown reflects real kernel work rather than launch overhead).
- 5 warmup iters (outside the profiler) + 10 profiled iters, `torch.inference_mode()`, on the L4.
- PyTorch 2.12.0+cu130. ESM2 attention runs through PyTorch's fused **flash-attention** kernel.

### Why the kernel-level view

`prof.key_averages()` lists *both* the operator events (`aten::addmm`, …) and the device kernels
they launch. Summing both double-counts GPU time (the first run reported ~449 ms/iter — almost
exactly 2×). Filtering to leaf CUDA kernels (keys not starting with `aten::`/`cuda`) gives a sum
that equals the real forward time, so the percentages are trustworthy.

### Where the time goes

Per-iteration GPU time by category (total 225.1 ms/iter):

| category | ms/iter | share | what it is |
|----------|--------:|------:|------------|
| elementwise | 96.4 | 42.8% | GeLU (`erf`), bias-adds, residual adds, scaling/`mul` — each a separate kernel |
| matmul | 94.9 | 42.2% | linear-layer GEMMs (cutlass + ampere) + flash-attention |
| copy | 30.8 | 13.7% | dtype casts (`direct_copy`, bf16 copy) + `CatArrayBatchedCopy` (rotary embedding) |
| softmax/norm | 2.8 | 1.3% | LayerNorm (`vectorized_layer_norm_kernel`); softmax is fused inside flash-attn |
| other/reduction | ~0 | ~0% | — |

Top individual kernels (per iter; full names in the CSV):

| kernel | ms/iter | share | category |
|--------|--------:|------:|----------|
| `cutlass_80_tensorop_bf16_s16816gemm_relu_256x128` | 56.0 | 24.9% | matmul |
| `ampere_bf16_s1688gemm_128x128_relu` | 28.4 | 12.6% | matmul |
| `vectorized_elementwise … MulFunctor` | 18.2 | 8.1% | elementwise |
| `elementwise … add` | 13.3 | 5.9% | elementwise |
| `direct_copy_kernel` (cast) | 12.7 | 5.6% | copy |
| `vectorized_elementwise … erf` (GeLU) | 11.9 | 5.3% | elementwise |
| `pytorch_flash::flash_fwd_kernel` (attention) | 10.5 | 4.7% | matmul |
| `CatArrayBatchedCopy` (rotary cat) | 9.9 | 4.4% | copy |
| `vectorized_layer_norm_kernel` | 2.8 | 1.3% | softmax/norm |

### What this tells us

- **Attribution checks out.** Kernel self-times sum to 225 ms/iter, equal to the Milestone 2
  median for this shape. The forward runs essentially serially on one stream, so kernel time ≈
  wall time — no hidden overlap inflating or hiding the breakdown.
- **Attention is not the bottleneck here.** The fused flash-attention kernel is only ~4.7% at
  this shape. The matmul cost is dominated by the **FFN and projection linears** (the two GEMM
  families at 24.9% + 12.6%). Optimizing attention further would barely move this workload.
- **~43% is unfused elementwise — the fusion headroom.** GeLU, bias-adds, residual adds, and
  scalings each launch their own kernel and stream the activation tensor through HBM and back.
  This is precisely what `torch.compile`/Inductor fuses, and it explains the **1.63× compile
  speedup measured at this exact shape in Milestone 5**: the profile is the *why* behind that
  number.
- **~14% is copies/casts.** `CatArrayBatchedCopy` is the rotary-position-embedding concatenation;
  the `direct_copy`/`bf16_copy` kernels are dtype casts. Some of this is avoidable with fusion or
  layout changes, though it is secondary to the elementwise bucket.
- **Not memory-capacity bound.** Peak allocation is ~1.4 GB of 24 GB; the constraint is kernel
  time and HBM bandwidth (from the unfused elementwise traffic), not capacity.

## Limitations

- **The category rollup is heuristic** (kernel-name substrings). The matmul-vs-rest split is
  robust, but the finer buckets are approximate: "elementwise" bundles activation, bias, residual,
  and scaling; LayerNorm is split out; "copy" bundles casts and concatenations. Read the buckets
  as order-of-magnitude, not exact.
- **Eager only.** This is the eager breakdown; a compiled run would collapse most of the
  elementwise kernels into fused ones (that is the Milestone 5 result), so its profile looks very
  different.
- **One shape.** seq 512 × batch 8. Smaller/shorter shapes shift toward launch-overhead-bound
  (Milestone 2), and longer sequences raise attention's quadratic share.
- **This L4 + PyTorch 2.12.0+cu130 stack.** Kernel choices (which cutlass/ampere GEMM, flash-attn
  availability) and therefore the split are stack- and GPU-specific.
