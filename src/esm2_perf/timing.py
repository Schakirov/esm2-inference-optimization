"""CUDA-event-based timing helpers for GPU benchmarks.

Why CUDA events rather than ``time.time()``: CUDA kernel launches are asynchronous, so
wall-clock timing on the host measures launch overhead, not kernel execution. CUDA events
are recorded on the GPU stream and, after a synchronize, report true device-side elapsed
time.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List

import torch


@dataclass
class TimingResult:
    """Summary statistics (milliseconds) over timed iterations."""

    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iters: int
    per_iter_ms: List[float]

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.pop("per_iter_ms")
        return d


def time_cuda(
    fn: Callable[[], object],
    *,
    warmup: int = 3,
    iters: int = 10,
    device: str = "cuda",
) -> TimingResult:
    """Time a no-arg callable on the GPU using CUDA events.

    Runs ``warmup`` untimed iterations (to trigger lazy init, autotuning, and caching),
    then ``iters`` timed iterations, each bracketed by a fresh pair of CUDA events. The
    stream is synchronized after each iteration so the measured time reflects completed
    device work.

    Returns per-iteration and summary timings in milliseconds.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("time_cuda requires CUDA; no GPU is available")
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")

    torch.cuda.synchronize(device)
    for _ in range(max(warmup, 0)):
        fn()
    torch.cuda.synchronize(device)

    per_iter: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        per_iter.append(start.elapsed_time(end))  # milliseconds

    return TimingResult(
        mean_ms=statistics.fmean(per_iter),
        median_ms=statistics.median(per_iter),
        std_ms=statistics.pstdev(per_iter) if len(per_iter) > 1 else 0.0,
        min_ms=min(per_iter),
        max_ms=max(per_iter),
        iters=iters,
        per_iter_ms=per_iter,
    )
