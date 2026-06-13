"""Tests for variable-length batch planning and padding accounting.

These run on the CPU with plain integer length lists — no GPU or model — so they exercise the
batching/padding math and its edge cases independently of ESM2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.batching import (  # noqa: E402
    STRATEGIES,
    BatchPlan,
    chunk,
    order_indices,
    plan_batches,
)


def test_chunk_sizes():
    batches = chunk(list(range(10)), 4)
    assert [len(b) for b in batches] == [4, 4, 2]
    # Every index appears exactly once, order preserved.
    assert [i for b in batches for i in b] == list(range(10))


def test_chunk_invalid_batch_size():
    with pytest.raises(ValueError):
        chunk([1, 2, 3], 0)


def test_order_naive_preserves_order():
    lengths = [5, 1, 3]
    assert order_indices(lengths, "naive") == [0, 1, 2]


def test_order_sorted_ascending():
    lengths = [5, 1, 3]
    assert order_indices(lengths, "sorted") == [1, 2, 0]


def test_order_unknown_strategy_raises():
    with pytest.raises(ValueError):
        order_indices([1, 2], "bogus")


def test_known_padding_counts():
    # lengths 1,2,3,4; batch_size 2.
    # naive batches [1,2] and [3,4] -> padded 2*2 + 2*4 = 12; actual 10.
    naive = plan_batches([1, 2, 3, 4], 2, "naive")
    assert naive.actual_tokens == 10
    assert naive.padded_tokens == 12
    # sorted is identical here because the input is already ascending.
    sorted_plan = plan_batches([1, 2, 3, 4], 2, "sorted")
    assert sorted_plan.padded_tokens == 12


def test_sorting_reduces_padding():
    # An adversarial order where naive pairs the longest with the shortest.
    lengths = [1, 100, 2, 99]
    naive = plan_batches(lengths, 2, "naive")
    sorted_plan = plan_batches(lengths, 2, "sorted")
    # naive: [1,100]->200, [2,99]->198 = 398; sorted: [1,2]->4, [99,100]->200 = 204.
    assert naive.padded_tokens == 398
    assert sorted_plan.padded_tokens == 204
    assert sorted_plan.padded_tokens < naive.padded_tokens
    # Real tokens are the same regardless of grouping.
    assert naive.actual_tokens == sorted_plan.actual_tokens == 202


def test_equal_lengths_have_no_waste():
    plan = plan_batches([7, 7, 7, 7], 2, "naive")
    assert plan.padded_tokens == plan.actual_tokens
    assert plan.padding_fraction == 0.0
    assert plan.padding_waste_ratio == 1.0


def test_padding_fraction_and_ratio_math():
    plan = BatchPlan("naive", [[0, 1]], actual_tokens=10, padded_tokens=12)
    assert plan.num_batches == 1
    assert plan.padding_fraction == pytest.approx((12 - 10) / 12)
    assert plan.padding_waste_ratio == pytest.approx(12 / 10)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_plan_covers_all_indices_once(strategy):
    lengths = [3, 1, 4, 1, 5, 9, 2, 6]
    plan = plan_batches(lengths, 3, strategy)
    seen = sorted(i for b in plan.batches for i in b)
    assert seen == list(range(len(lengths)))


def test_invalid_length_raises():
    with pytest.raises(ValueError):
        plan_batches([1, 0, 3], 2, "naive")
