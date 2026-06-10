"""esm2_perf: utilities for benchmarking and optimizing ESM2 protein embedding extraction.

Modules are added milestone by milestone:
    timing    - CUDA-event timing helpers
    sequences - deterministic synthetic protein sequence generation
    results   - CSV result schema and writers
    pooling   - masked mean pooling over real amino-acid tokens
    batching  - variable-length batching / padding strategies
    triton_pooling - custom Triton masked-mean-pooling kernel
"""

__version__ = "0.1.0"
