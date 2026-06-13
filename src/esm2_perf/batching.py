"""Variable-length batching and padding-efficiency analysis.

When sequences of different lengths are batched together, every sequence is padded to the
longest one in its batch, so the GPU computes over padding tokens that carry no signal. This
module plans how sequences are grouped into batches under different strategies and quantifies
the resulting padding waste — independent of any GPU work, so it is unit-testable on the CPU.

Strategies:
    naive  - keep the given (arbitrary) order; pad each batch to its own max.
    sorted - sort by length first, so each batch holds similar-length sequences and pads less.

The benchmark script feeds ``lengths`` as ESM2 *token* counts (amino acids + 2 special
tokens), so a plan's ``actual_tokens`` / ``padded_tokens`` match the tokenizer exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

STRATEGIES = ("naive", "sorted")


def order_indices(lengths: Sequence[int], strategy: str) -> List[int]:
    """Return indices into ``lengths`` in the order the strategy would process them."""
    n = len(lengths)
    if strategy == "naive":
        return list(range(n))
    if strategy == "sorted":
        # Ascending by length; ties keep original order (stable sort).
        return sorted(range(n), key=lambda i: lengths[i])
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def chunk(indices: Sequence[int], batch_size: int) -> List[List[int]]:
    """Split ``indices`` into consecutive batches of at most ``batch_size``."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return [list(indices[i : i + batch_size]) for i in range(0, len(indices), batch_size)]


@dataclass
class BatchPlan:
    """A grouping of sequence indices into batches, with padding accounting.

    ``actual_tokens`` is the sum of real tokens (independent of strategy). ``padded_tokens``
    is what the GPU actually computes over: for each batch, ``len(batch) * max_length_in_batch``.
    """

    strategy: str
    batches: List[List[int]]
    actual_tokens: int
    padded_tokens: int

    @property
    def num_batches(self) -> int:
        return len(self.batches)

    @property
    def padding_fraction(self) -> float:
        """Fraction of computed tokens that are padding, in ``[0, 1)``."""
        if self.padded_tokens == 0:
            return 0.0
        return (self.padded_tokens - self.actual_tokens) / self.padded_tokens

    @property
    def padding_waste_ratio(self) -> float:
        """``padded_tokens / actual_tokens`` (>= 1.0); 1.0 means no padding waste."""
        if self.actual_tokens == 0:
            return 0.0
        return self.padded_tokens / self.actual_tokens


def plan_batches(lengths: Sequence[int], batch_size: int, strategy: str) -> BatchPlan:
    """Plan batches for ``lengths`` under ``strategy`` and tally padding waste."""
    if any(length < 1 for length in lengths):
        raise ValueError("all lengths must be >= 1")
    order = order_indices(lengths, strategy)
    batches = chunk(order, batch_size)
    actual_tokens = int(sum(lengths))
    padded_tokens = 0
    for batch in batches:
        max_len = max(lengths[i] for i in batch)
        padded_tokens += max_len * len(batch)
    return BatchPlan(
        strategy=strategy,
        batches=batches,
        actual_tokens=actual_tokens,
        padded_tokens=int(padded_tokens),
    )
