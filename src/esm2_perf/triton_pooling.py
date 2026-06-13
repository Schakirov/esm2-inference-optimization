"""Custom Triton kernel for masked mean pooling over real amino-acid tokens.

This is a hand-written fused kernel that computes, for hidden states ``(B, T, H)`` and a 0/1
``mask`` ``(B, T)``:

    out[b, h] = sum_t mask[b, t] * hidden[b, t, h] / max(sum_t mask[b, t], 1)

in a single pass, accumulating in fp32. It is the functional twin of
``esm2_perf.pooling.masked_mean_pool`` (validated against that reference in the tests), and
exists to (a) demonstrate a real Triton reduction kernel and (b) be benchmarked honestly
against the PyTorch and ``torch.compile`` pooling paths — it is not assumed to be faster.

The reduction is over the sequence axis ``T``. The grid is ``(B, ceil(H / BLOCK_H))``: each
program owns one batch row and a block of hidden columns, and streams over ``T`` in chunks of
``BLOCK_T``, accumulating both the masked sum and the token count. An all-padding row (count 0)
pools to a zero vector rather than NaN, matching the reference.
"""

from __future__ import annotations

import torch

try:  # Triton is GPU-only; import lazily so CPU imports of the package don't fail.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on Triton-less machines
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _masked_mean_pool_kernel(
        hidden_ptr,
        mask_ptr,
        out_ptr,
        T,
        H,
        stride_hb,
        stride_ht,
        stride_hh,
        stride_mb,
        stride_mt,
        stride_ob,
        stride_oh,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        h_in = offs_h < H

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        count = tl.zeros((1,), dtype=tl.float32)

        for t0 in range(0, T, BLOCK_T):
            offs_t = t0 + tl.arange(0, BLOCK_T)
            t_in = offs_t < T

            m = tl.load(
                mask_ptr + pid_b * stride_mb + offs_t * stride_mt,
                mask=t_in,
                other=0.0,
            ).to(tl.float32)  # (BLOCK_T,)

            h_ptrs = (
                hidden_ptr
                + pid_b * stride_hb
                + offs_t[:, None] * stride_ht
                + offs_h[None, :] * stride_hh
            )
            tile = tl.load(
                h_ptrs, mask=t_in[:, None] & h_in[None, :], other=0.0
            ).to(tl.float32)  # (BLOCK_T, BLOCK_H)

            acc += tl.sum(tile * m[:, None], axis=0)  # (BLOCK_H,)
            count += tl.sum(m, axis=0)

        denom = tl.maximum(count, 1.0)  # (1,) -> avoids divide-by-zero on all-pad rows
        out = acc / denom
        tl.store(out_ptr + pid_b * stride_ob + offs_h * stride_oh, out, mask=h_in)


def triton_masked_mean_pool(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
    block_t: int = 128,
    block_h: int = 256,
) -> torch.Tensor:
    """Masked mean pool ``hidden_states`` (B, T, H) over ``mask`` (B, T) with a Triton kernel.

    Accumulates in fp32 and returns ``(B, H)`` in ``out_dtype`` (fp32 by default). Rows whose
    mask sums to zero return a zero vector. Requires CUDA + Triton.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton is not available; cannot run the Triton pooling kernel")
    if not hidden_states.is_cuda:
        raise ValueError("triton_masked_mean_pool requires CUDA tensors")
    if hidden_states.dim() != 3:
        raise ValueError(f"expected hidden_states (B, T, H), got {tuple(hidden_states.shape)}")

    B, T, H = hidden_states.shape
    hidden_states = hidden_states.contiguous()
    mask = mask.to(torch.float32).contiguous()

    out = torch.empty((B, H), device=hidden_states.device, dtype=torch.float32)
    grid = (B, triton.cdiv(H, block_h))
    _masked_mean_pool_kernel[grid](
        hidden_states,
        mask,
        out,
        T,
        H,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        mask.stride(0),
        mask.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_T=block_t,
        BLOCK_H=block_h,
    )
    return out.to(out_dtype)
