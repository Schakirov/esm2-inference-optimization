"""CSV result schema and writers for benchmark outputs.

Keeping the column schema in one place makes results consistent across milestones and easy
to summarize later. Writers create parent directories and write a header row.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Baseline benchmark schema (Milestone 2). Required columns from the task plus a few
# timing detail columns that are cheap to record and useful for sanity-checking.
BASELINE_COLUMNS: List[str] = [
    "model_name",
    "gpu_name",
    "dtype",
    "batch_size",
    "seq_len",
    "actual_tokens",
    "latency_ms",            # median latency over timed iters
    "latency_mean_ms",
    "latency_std_ms",
    "tokens_per_sec",        # actual (non-pad) tokens / sec
    "sequences_per_sec",
    "max_memory_allocated_gb",
    "warmup",
    "iters",
    "oom",
    "notes",
]

# Correctness / pooling schema (Milestone 3). Each row records one comparison: either a
# low-precision (bf16/fp16) result against the fp32 eager reference, or two pooling
# implementations against each other at the same precision. Error metrics are computed in
# fp32. ``passed`` is a convenience flag from ``torch.allclose(rtol, atol)``; the raw error
# columns are the honest record regardless of the threshold chosen.
CORRECTNESS_COLUMNS: List[str] = [
    "model_name",
    "gpu_name",
    "check",                 # what is compared, e.g. pooled_vs_fp32 / hidden_vs_fp32 / impl_equiv
    "dtype",                 # precision of the approx side (fp32 for impl_equiv checks)
    "reference",             # precision/impl of the reference side
    "batch_size",
    "seq_len",
    "exclude_special",       # whether <cls>/<eos> were dropped before pooling
    "tensor",                # pooled | hidden
    "max_abs_err",
    "mean_abs_err",
    "rms_err",
    "max_rel_err",
    "min_cosine_sim",        # min over the batch (pooled embeddings only; blank for hidden)
    "rtol",
    "atol",
    "passed",
    "notes",
]

# Batching / padding schema (Milestone 4). Each row is one (strategy, batch_size) cell over a
# fixed pool of variable-length sequences. ``padding_fraction`` and ``padded_tokens`` capture
# wasted compute; throughput is reported both as real (non-pad) tokens/sec and as padded
# compute tokens/sec, so the win from length-sorted batching is visible as real-token gain at
# roughly constant padded-token rate.
BATCHING_COLUMNS: List[str] = [
    "model_name",
    "gpu_name",
    "dtype",
    "strategy",              # naive | sorted
    "num_seqs",
    "min_len",
    "max_len",
    "batch_size",
    "num_batches",
    "actual_tokens",         # real tokens summed over the whole pool (strategy-independent)
    "padded_tokens",         # tokens actually computed over (sum of batch_size * batch_max)
    "padding_fraction",      # (padded - actual) / padded, in [0, 1)
    "padding_waste_ratio",   # padded / actual, >= 1.0
    "latency_ms",            # median time for one full pass over all batches
    "real_tokens_per_sec",
    "padded_tokens_per_sec",
    "sequences_per_sec",
    "max_memory_allocated_gb",
    "warmup",
    "iters",
    "oom",
    "notes",
]

# torch.compile schema (Milestone 5). Each row is one (mode, dynamic, batch_size, seq_len)
# cell. ``cold_start_ms`` is the first call at that shape (includes compilation for compiled
# modes) measured on the wall clock; the steady-state columns are CUDA-event timed after
# warmup. ``speedup_vs_eager`` compares steady latency to the eager row at the same shape.
COMPILE_COLUMNS: List[str] = [
    "model_name",
    "gpu_name",
    "dtype",
    "mode",                  # eager | default | reduce-overhead | max-autotune
    "dynamic",               # whether torch.compile(dynamic=True) was used
    "batch_size",
    "seq_len",
    "actual_tokens",
    "cold_start_ms",         # first call at this shape (wall clock; includes compile)
    "steady_latency_ms",     # median steady-state latency over timed iters
    "steady_mean_ms",
    "steady_std_ms",
    "tokens_per_sec",        # real (non-pad) tokens / sec at steady state
    "sequences_per_sec",
    "speedup_vs_eager",      # eager_steady / this_steady at the same shape (blank for eager)
    "max_memory_allocated_gb",
    "warmup",
    "iters",
    "oom",
    "notes",
]

# Pooling-kernel schema (Milestone 6). Each row times one masked-mean-pool implementation
# (pytorch / einsum / compiled / triton) on a hidden-state tensor of a given shape, with the
# speedup vs the eager PyTorch pooling and the max abs error against the fp32 reference, so the
# kernel is reported as both fast (or not) and correct.
POOLING_COLUMNS: List[str] = [
    "gpu_name",
    "dtype",
    "impl",                  # pytorch | einsum | compiled | triton
    "batch_size",
    "seq_len",
    "hidden_size",
    "latency_ms",            # median over timed iters
    "latency_mean_ms",
    "latency_std_ms",
    "speedup_vs_pytorch",    # pytorch_latency / this_latency (1.0 for the pytorch row)
    "max_abs_err_vs_ref",    # vs fp32 per-sequence reference pooling
    "max_memory_allocated_gb",
    "warmup",
    "iters",
    "oom",
    "notes",
]


def timestamp() -> str:
    """UTC timestamp safe for filenames, e.g. 20260610T095837Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def timestamped_path(directory: str | Path, prefix: str, suffix: str = ".csv") -> Path:
    """Build ``<directory>/<prefix>_<timestamp><suffix>``."""
    return Path(directory) / f"{prefix}_{timestamp()}{suffix}"


def write_csv(
    path: str | Path,
    rows: Iterable[Dict[str, object]],
    columns: Sequence[str],
) -> Path:
    """Write ``rows`` to ``path`` as CSV with the given ``columns`` as header.

    Missing keys are written as empty cells; extra keys raise, so schema drift is caught
    early. Returns the written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)
    column_set = set(columns)
    with path.open("w", newline="") as f:
        # restval="" fills missing keys; extrasaction="raise" rejects unknown keys.
        writer = csv.DictWriter(
            f, fieldnames=columns, restval="", extrasaction="raise"
        )
        writer.writeheader()
        for row in rows:
            extras = set(row) - column_set
            if extras:
                raise ValueError(f"row has columns not in schema: {sorted(extras)}")
            writer.writerow(row)
    return path
