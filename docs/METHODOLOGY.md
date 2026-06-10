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

(Expanded as milestones land. Will cover: timing protocol, warmup counts, why CUDA events
rather than `time.time()`, dtype handling, padding-waste definition, batching strategies,
`torch.compile` warmup vs steady-state separation, and Triton kernel validation.)
