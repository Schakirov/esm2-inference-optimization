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
