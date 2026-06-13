"""Tests for masked mean pooling and the pool-mask helper.

These run on the CPU with small hand-constructed tensors — no GPU or model download — so
they exercise the pooling math (and its edge cases) independently of ESM2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.pooling import (  # noqa: E402
    build_pool_mask,
    masked_mean_pool,
    masked_mean_pool_einsum,
    masked_mean_pool_reference,
)

_IMPLS = [masked_mean_pool, masked_mean_pool_einsum, masked_mean_pool_reference]


@pytest.mark.parametrize("fn", _IMPLS)
def test_shape(fn):
    hidden = torch.randn(4, 7, 16)
    mask = torch.ones(4, 7)
    out = fn(hidden, mask)
    assert out.shape == (4, 16)


@pytest.mark.parametrize("fn", _IMPLS)
def test_known_value(fn):
    # Two tokens kept (values 2 and 4), one masked out: mean should be 3.
    hidden = torch.tensor([[[2.0, 2.0], [4.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    out = fn(hidden, mask)
    assert torch.allclose(out, torch.tensor([[3.0, 3.0]]))


@pytest.mark.parametrize("fn", _IMPLS)
def test_masked_positions_are_ignored(fn):
    # Changing the hidden values under masked positions must not change the pooled result.
    hidden = torch.randn(3, 6, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]] * 3, dtype=torch.float32)
    out_a = fn(hidden, mask)
    hidden2 = hidden.clone()
    hidden2[:, 3:, :] = 123.0  # only masked-out positions
    out_b = fn(hidden2, mask)
    assert torch.allclose(out_a, out_b)


@pytest.mark.parametrize("fn", _IMPLS)
def test_all_padding_row_is_zero(fn):
    # A sequence with no kept tokens returns a zero vector rather than NaN.
    hidden = torch.randn(2, 5, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    out = fn(hidden, mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[1], torch.zeros(4))


def test_implementations_agree():
    # All three implementations agree to near round-off on random input.
    torch.manual_seed(0)
    hidden = torch.randn(5, 11, 32, dtype=torch.float64)
    mask = (torch.rand(5, 11) > 0.4).to(torch.float64)
    mask[0] = 0.0  # include an all-padding row
    ref = masked_mean_pool_reference(hidden, mask)
    assert torch.allclose(masked_mean_pool(hidden, mask), ref, rtol=1e-10, atol=1e-12)
    assert torch.allclose(masked_mean_pool_einsum(hidden, mask), ref, rtol=1e-10, atol=1e-12)


def test_build_pool_mask_excludes_special():
    # special_tokens_mask marks <cls> (pos 0), <eos> (pos 3); pos 4 is padding.
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 0]])
    mask = build_pool_mask(attention_mask, special, exclude_special=True)
    assert mask.tolist() == [[0, 1, 1, 0, 0]]


def test_build_pool_mask_include_special():
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 0]])
    mask = build_pool_mask(attention_mask, special, exclude_special=False)
    assert mask.tolist() == [[1, 1, 1, 1, 0]]


def test_exclude_special_changes_result():
    # With/without special tokens the pooled vector should differ when specials are nonzero.
    hidden = torch.randn(1, 4, 8)
    attention_mask = torch.ones(1, 4)
    special = torch.tensor([[1, 0, 0, 1]])
    pooled_excl = masked_mean_pool(hidden, build_pool_mask(attention_mask, special, exclude_special=True))
    pooled_incl = masked_mean_pool(hidden, build_pool_mask(attention_mask, special, exclude_special=False))
    assert not torch.allclose(pooled_excl, pooled_incl)
