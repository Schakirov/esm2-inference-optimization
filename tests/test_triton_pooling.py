"""Tests for the Triton masked-mean-pooling kernel.

These require CUDA + Triton, so they are skipped on CPU-only machines. They validate the
kernel against the Milestone 3 reference implementations (the per-sequence oracle), which is
the whole point of having that oracle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.pooling import masked_mean_pool, masked_mean_pool_reference  # noqa: E402
from esm2_perf import triton_pooling  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and triton_pooling.HAS_TRITON),
    reason="Triton pooling kernel requires CUDA + Triton",
)


def _random_mask(B, T, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lengths = torch.randint(1, T + 1, (B,), generator=g)
    idx = torch.arange(T).expand(B, T)
    return (idx < lengths[:, None]).to(device=device, dtype=torch.float32)


@pytest.mark.parametrize("B,T,H", [(4, 130, 1280), (8, 514, 1280), (1, 1, 256), (3, 200, 257)])
def test_matches_fp32_reference(B, T, H):
    torch.manual_seed(0)
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
    mask = _random_mask(B, T, "cuda")
    ref = masked_mean_pool_reference(hidden, mask)
    out = triton_pooling.triton_masked_mean_pool(hidden, mask)
    assert out.shape == (B, H)
    # fp32 reductions in different orders: agreement to tight float32 round-off.
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_matches_vectorized_pytorch():
    torch.manual_seed(1)
    hidden = torch.randn(6, 256, 1280, device="cuda", dtype=torch.float32)
    mask = _random_mask(6, 256, "cuda", seed=3)
    ref = masked_mean_pool(hidden, mask)
    out = triton_pooling.triton_masked_mean_pool(hidden, mask)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_all_padding_row_is_zero():
    hidden = torch.randn(2, 16, 1280, device="cuda", dtype=torch.float32)
    mask = torch.ones(2, 16, device="cuda")
    mask[1] = 0.0  # second row is all padding -> must be zeros, not NaN
    out = triton_pooling.triton_masked_mean_pool(hidden, mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[1], torch.zeros(1280, device="cuda"))


def test_bf16_input_upcast_matches_reference():
    torch.manual_seed(2)
    hidden_fp32 = torch.randn(4, 130, 1280, device="cuda", dtype=torch.float32)
    hidden_bf16 = hidden_fp32.to(torch.bfloat16)
    mask = _random_mask(4, 130, "cuda", seed=5)
    # Both read the same bf16 values and accumulate in fp32, so they should match closely.
    ref = masked_mean_pool_reference(hidden_bf16.to(torch.float32), mask)
    out = triton_pooling.triton_masked_mean_pool(hidden_bf16, mask)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3)


def test_out_dtype_option():
    hidden = torch.randn(2, 64, 1280, device="cuda", dtype=torch.float32)
    mask = _random_mask(2, 64, "cuda", seed=7)
    out = triton_pooling.triton_masked_mean_pool(hidden, mask, out_dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16
