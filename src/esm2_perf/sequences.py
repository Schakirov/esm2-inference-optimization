"""Deterministic synthetic protein-sequence generation.

Sequences are drawn uniformly from the 20 standard amino acids. Generation is seeded so
that benchmark inputs are reproducible across runs and machines.
"""

from __future__ import annotations

import random
from typing import List

# The 20 standard (canonical) amino acids, one-letter codes.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
assert len(AMINO_ACIDS) == 20


def random_sequence(length: int, rng: random.Random) -> str:
    """Return a single random protein sequence of ``length`` amino acids."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def generate_sequences(n: int, length: int, seed: int = 0) -> List[str]:
    """Generate ``n`` fixed-length synthetic protein sequences.

    Deterministic for a given ``(n, length, seed)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = random.Random(seed)
    return [random_sequence(length, rng) for _ in range(n)]


def generate_variable_sequences(
    n: int,
    min_length: int,
    max_length: int,
    seed: int = 0,
) -> List[str]:
    """Generate ``n`` sequences with lengths sampled uniformly in ``[min, max]``.

    Used by the variable-length batching/padding milestone. Deterministic for a given
    ``(n, min_length, max_length, seed)``.
    """
    if min_length < 1 or max_length < min_length:
        raise ValueError(f"require 1 <= min <= max, got min={min_length}, max={max_length}")
    rng = random.Random(seed)
    lengths = [rng.randint(min_length, max_length) for _ in range(n)]
    return [random_sequence(length, rng) for length in lengths]
