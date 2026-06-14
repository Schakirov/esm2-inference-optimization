# Results

## TL;DR

Consolidated benchmark results for ESM2-650M embedding extraction on a single **NVIDIA L4**.
All tables below are **generated from the raw CSVs** in `results/raw/` by
[`scripts/07_summarize_results.py`](../scripts/07_summarize_results.py), so they cannot drift
from the data. Headline findings:

- **Baseline:** real-token throughput saturates around **~22k tokens/sec** (bf16); peak memory
  is **~2.0 GB of 24 GB** — the L4 is nowhere near capacity-bound for this model.
- **Biggest, cheapest win — batching:** length-**sorting** variable-length sequences cuts
  padding waste from ~41–44% to ~2–8% and lifts real-token throughput **~1.7×**, for free.
- **`torch.compile`:** a real **1.3×–2.2×** steady-state speedup, but **~28 s cold-start per
  shape** — only worth it for high-volume fixed-shape serving.
- **Triton pooling kernel:** **fastest in every shape** and **~1e-8** accurate, but pooling is
  <0.1% of runtime, so the end-to-end effect is **negligible** (a kernel-authoring demo).
- **Profiler:** the forward is ~42% matmul (FFN linears, *not* attention) and ~43% unfused
  elementwise — the elementwise share is exactly what `torch.compile` fuses.

What did **not** help is reported as honestly as what did — see
[What did not improve performance](#what-did-not-improve-performance).

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA L4, 24 GB GDDR6, 58 SMs, compute capability 8.9 (Ada) |
| Host | AWS `g6.2xlarge` |
| Driver / CUDA (smi) | 595.71.05 / 13.2 |
| PyTorch | 2.12.0+cu130 |
| transformers / triton | 5.10.2 / 3.7.0 |
| Model | `facebook/esm2_t33_650M_UR50D` (650M params, hidden 1280) |

Captured programmatically in [`results/raw/environment.txt`](../results/raw/environment.txt).
All numbers below are specific to this hardware + software stack and a synthetic
amino-acid sequence distribution.

## Baseline (Milestone 2)

<!-- BEGIN baseline -->
Real (non-pad) tokens/sec, **bf16**, same-length batches (no intra-batch padding):

| batch \ seq | 128 | 256 | 512 | 1022 |
| :-- | --: | --: | --: | --: |
| 1 | 6,446 | 12,923 | 20,384 | 22,040 |
| 2 | 13,265 | 20,249 | 20,952 | 20,203 |
| 4 | 20,797 | 20,995 | 20,721 | 17,992 |
| 8 | 21,126 | 21,083 | 18,052 | 15,795 |
| 16 | 21,203 | 18,263 | 15,977 | 15,363 |

Throughput saturates around **22,040 tok/s** (peak at batch 1, seq 1022); small batches underuse the GPU and long sequences at large batch fall back. Peak memory across the full bf16+fp16 matrix is **2.0 GB** of 24 GB — OOM events: **none**. fp16 throughput tracks bf16 within a few percent (slightly lower).

_Source: `results/raw/baseline_20260610T101812Z.csv`._
<!-- END baseline -->

## Correctness (Milestone 3)

<!-- BEGIN correctness -->
| check | approx | max abs err | min cosine | passed @1% |
| :-- | --: | --: | --: | --: |
| impl agreement (vectorized/einsum vs reference) | fp32 | 1.9e-06 | 1.000000 | ✓ |
| pooled vs fp32 | fp16 | 0.012 | 0.999991 | ✓ |
| pooled vs fp32 | bf16 | 0.11 | 0.999654 | ✗ |
| hidden vs fp32 | fp16 | 0.059 | — | ✗ |
| hidden vs fp32 | bf16 | 0.46 | — | ✗ |

The three pooling implementations are numerically identical in fp32 (≈2e-6). Against an fp32 encoder reference, **fp16 tracks far tighter than bf16**; pooled embeddings stay directionally near-identical in both low precisions (cosine ≥ 0.9996). The `✗` rows are honest: per-token hidden states in bf16/fp16 exceed a strict 1% tolerance, but the pooled vector — what downstream tasks use — does not, in fp16.

_Source: `results/raw/correctness_20260613T102451Z.csv`._
<!-- END correctness -->

## Variable-length batching & padding (Milestone 4)

<!-- BEGIN batching -->
| strategy | batch | padding waste | real tok/s | padded tok/s |
| :-- | --: | --: | --: | --: |
| naive | 8 | 41.0% | 10,476 | 17,749 |
| naive | 16 | 42.6% | 9,069 | 15,795 |
| naive | 32 | 44.0% | 8,463 | 15,109 |
| sorted | 8 | 2.2% | 18,382 | 18,794 |
| sorted | 16 | 4.5% | 15,996 | 16,757 |
| sorted | 32 | 8.3% | 14,147 | 15,432 |

Length-**sorting** the same 256-sequence pool cuts padding waste from **~41–44%** to **~2–8%** and lifts real-token throughput by **1.67×–1.76×** — at a roughly constant *padded*-token rate. That is the honest framing: the GPU does the same raw compute; sorting just wastes less of it on padding. This is the cheapest win in the project (a sort, no kernel or precision change).

_Source: `results/raw/batching_20260613T112856Z.csv`._
<!-- END batching -->

## torch.compile (Milestone 5)

<!-- BEGIN compile -->
bf16, static shapes (`dynamic=False`). Speedup is eager-steady ÷ compiled-steady at the same shape:

| mode | seq | batch | cold start (s) | steady (ms) | speedup |
| :-- | --: | --: | --: | --: | --: |
| default | 128 | 1 | 28 | 9.6 | 2.15× |
| default | 128 | 8 | 29 | 34.1 | 1.37× |
| default | 512 | 1 | 28 | 19.8 | 1.30× |
| default | 512 | 8 | 29 | 138.7 | 1.63× |
| reduce-overhead | 128 | 1 | 28 | 9.4 | 2.20× |
| reduce-overhead | 128 | 8 | 28 | 33.8 | 1.38× |
| reduce-overhead | 512 | 1 | 28 | 19.9 | 1.30× |
| reduce-overhead | 512 | 8 | 28 | 137.7 | 1.64× |

`torch.compile` delivers a real **1.30×–2.20×** steady-state speedup — largest on the small overhead-bound shape, ~1.6× on the large compute-bound one. The honest catch is the **~28 s cold-start compile per shape**: with static shapes every new `(batch, seq_len)` recompiles, so this only pays off after hundreds-to-thousands of calls at a fixed shape.

_Source: `results/raw/compile_20260613T150026Z.csv`._
<!-- END compile -->

## Triton masked-mean-pooling kernel (Milestone 6)

<!-- BEGIN triton -->
bf16. Speedup is vs the eager vectorized PyTorch pooling; error is max-abs vs the fp32 reference:

| shape (b×t) | PyTorch (ms) | einsum (ms) | Triton (ms) | Triton speedup | Triton err vs fp32 |
| :-- | --: | --: | --: | --: | --: |
| 8×128 | 0.097 | 0.132 | 0.076 | 1.29× | 3.0e-08 |
| 32×128 | 0.100 | 0.136 | 0.073 | 1.38× | 3.0e-08 |
| 8×512 | 0.099 | 0.135 | 0.076 | 1.31× | 7.5e-09 |
| 32×512 | 0.458 | 0.138 | 0.085 | 5.39× | 3.0e-08 |
| 8×1022 | 0.116 | 0.139 | 0.076 | 1.53× | 1.5e-08 |
| 32×1022 | 1.051 | 0.434 | 0.389 | 2.70× | 2.2e-08 |

The Triton kernel is **fastest in every shape** (**1.29×–5.39×** vs eager PyTorch) and **numerically near-exact** (~1e-8 vs ~1e-3 for the bf16 PyTorch paths), because it accumulates in fp32 and never materializes the `(B,T,H)` product. The essential caveat: pooling is only ~0.08 ms against a 25–225 ms encoder forward, so this is a clean micro-optimization with **negligible end-to-end effect** — included as a kernel-authoring demonstration, not a headline win.

_Source: `results/raw/triton_pooling_20260613T162844Z.csv`._
<!-- END triton -->

## Profiling (Milestone 7)

<!-- BEGIN profile -->
Leaf-CUDA-kernel self-time, bf16, seq 512 × batch 8 (top-15 kernels rolled into categories; they cover **99%** of the forward):

| category | share of forward | what it is |
| :-- | --: | --: |
| matmul | 42.2% | FFN/projection GEMMs + flash-attention (attention is only ~4.7%) |
| elementwise | 41.8% | GeLU, bias-adds, residual adds, scaling — each an unfused kernel |
| copy | 13.7% | dtype casts + rotary-embedding `cat` |
| softmax/norm | 1.3% | LayerNorm (softmax is fused inside flash-attn) |

The split is ~even between **matmul** (the FFN linears, not attention) and **unfused elementwise** — and that elementwise share is exactly the fusion headroom that explains the `torch.compile` win above. See [`docs/PROFILING.md`](PROFILING.md) for the full kernel list.

_Source: `results/raw/profile_20260613T171924Z.csv`._
<!-- END profile -->

## What improved performance

- **Length-sorted batching (~1.7× real-token throughput).** The single best return on effort:
  no kernel, no precision change, just grouping similar-length sequences so the GPU wastes far
  less compute on padding. Recommended unconditionally for batched embedding extraction.
- **bf16 (default).** Half the memory and the highest throughput of the precisions tested,
  with pooled-embedding cosine ≥ 0.9996 vs fp32.
- **`torch.compile` for fixed-shape, high-volume serving (1.3×–2.2×).** A genuine steady-state
  win *if* the ~28 s/shape cold start is amortized over many calls at a stable shape.
- **The Triton pooling kernel (1.3×–5.4× on the pooling step, ~1e-8 accurate).** Correct and
  fast in isolation — see the caveat below for why it doesn't move the end-to-end needle.

## What did not improve performance

Reported as honestly as the wins:

- **The Triton kernel does not change end-to-end latency.** Pooling is ~0.08 ms against a
  25–225 ms encoder forward (<0.1%). The kernel is a correctness/authoring demonstration, not
  an end-to-end speedup.
- **`torch.compile` cold start is brutal for dynamic workloads.** ~28 s to compile *each*
  shape with `dynamic=False`; a workload with many distinct `(batch, seq_len)` shapes would
  recompile constantly and could be net-slower than eager.
- **`einsum` / compiled pooling are *slower* than plain PyTorch** at small shapes (einsum
  ~0.74× at batch 8 / seq 128) — the "clever" pooling variants are not free.
- **Throughput saturates early.** Beyond ~batch 8 the L4 is compute-bound; larger batches at
  long sequence lengths *reduce* real-token throughput rather than raise it.
- **fp16 is more accurate than bf16 but slightly slower** — a precision/speed trade, not a
  free win; bf16 remains the throughput default.

## Reproduce

```bash
python scripts/01_bench_baseline.py --quick        # and 02..06 for the other experiments
python scripts/07_summarize_results.py             # regenerate every table above
python scripts/07_summarize_results.py --check     # CI: fail if tables are stale
```

See [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full protocol and
[`docs/LIMITATIONS.md`](LIMITATIONS.md) for scope and caveats.
