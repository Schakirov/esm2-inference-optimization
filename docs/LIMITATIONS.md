# Limitations

## TL;DR

What these benchmarks do **not** establish, so results are not over-read.

## Short explanation

- Numbers are specific to one NVIDIA L4 and the exact software stack captured in
  `results/raw/environment.txt`.
- Inputs are synthetic sequences over the 20 standard amino acids, not a real proteome;
  length distribution affects batching/padding results.
- No comparison against vendor-optimized stacks (TensorRT, custom FlashAttention builds)
  unless explicitly benchmarked.

## Longer explanation

(Expanded per milestone as concrete caveats are discovered — e.g. compile recompilation
behavior, Triton kernel applicability, measurement noise.)
