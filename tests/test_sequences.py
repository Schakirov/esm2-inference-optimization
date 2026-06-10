"""Tests for synthetic protein-sequence generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.sequences import (  # noqa: E402
    AMINO_ACIDS,
    generate_sequences,
    generate_variable_sequences,
)


def test_amino_acid_alphabet():
    assert len(AMINO_ACIDS) == 20
    assert len(set(AMINO_ACIDS)) == 20
    assert "B" not in AMINO_ACIDS  # ambiguous codes excluded
    assert "X" not in AMINO_ACIDS


def test_fixed_length_and_alphabet():
    seqs = generate_sequences(5, 30, seed=0)
    assert len(seqs) == 5
    assert all(len(s) == 30 for s in seqs)
    allowed = set(AMINO_ACIDS)
    assert all(set(s) <= allowed for s in seqs)


def test_determinism_same_seed():
    assert generate_sequences(4, 50, seed=42) == generate_sequences(4, 50, seed=42)


def test_different_seed_differs():
    a = generate_sequences(4, 50, seed=1)
    b = generate_sequences(4, 50, seed=2)
    assert a != b


def test_variable_lengths_in_range():
    seqs = generate_variable_sequences(20, 10, 40, seed=7)
    assert len(seqs) == 20
    assert all(10 <= len(s) <= 40 for s in seqs)
    assert generate_variable_sequences(20, 10, 40, seed=7) == seqs  # deterministic


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_counts_raise(bad):
    with pytest.raises(ValueError):
        generate_sequences(bad, 10)


def test_invalid_length_raises():
    with pytest.raises(ValueError):
        generate_sequences(2, 0)
