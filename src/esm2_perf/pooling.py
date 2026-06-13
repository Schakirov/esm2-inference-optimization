"""Masked mean pooling over real amino-acid tokens.

The ESM2 encoder returns a per-token ``last_hidden_state`` of shape ``(B, T, H)``. A common
sequence-level embedding is the *mean over real tokens* — i.e. averaging hidden vectors at
positions that are not padding, and (optionally) not the ``<cls>``/``<eos>`` special tokens.

This module provides:

- :func:`build_pool_mask` — turn a tokenizer ``attention_mask`` (and optional
  ``special_tokens_mask``) into a 0/1 *pooling mask* selecting the tokens to average.
- Three numerically-equivalent pooling implementations used to cross-check each other and,
  later, the Triton kernel:
    - :func:`masked_mean_pool` — the vectorized production implementation.
    - :func:`masked_mean_pool_einsum` — an ``einsum`` variant.
    - :func:`masked_mean_pool_reference` — an explicit per-sequence reference (slow, clear).

All three treat a sequence with *zero* selected tokens as producing a zero vector (rather
than NaN from a divide-by-zero), so an all-padding row is well defined.
"""

from __future__ import annotations

import torch


def build_pool_mask(
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor | None = None,
    *,
    exclude_special: bool = True,
) -> torch.Tensor:
    """Build a 0/1 pooling mask selecting the tokens to average.

    ``attention_mask`` is 1 for real (non-pad) tokens. When ``exclude_special`` is true and a
    ``special_tokens_mask`` is supplied (1 for special tokens such as ``<cls>``/``<eos>``/
    ``<pad>``), those positions are also dropped so pooling runs over amino-acid tokens only.

    Returns a tensor shaped like ``attention_mask`` with the same dtype.
    """
    mask = attention_mask.clone()
    if exclude_special and special_tokens_mask is not None:
        keep = (1 - special_tokens_mask.to(mask.dtype))
        mask = mask * keep
    return mask


def masked_mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``hidden_states`` (B, T, H) over positions where ``mask`` (B, T) is nonzero.

    Vectorized production implementation. Rows whose mask sums to zero return a zero vector
    (the count is clamped to 1 to avoid division by zero).
    """
    m = mask.to(hidden_states.dtype).unsqueeze(-1)  # (B, T, 1)
    summed = (hidden_states * m).sum(dim=1)          # (B, H)
    counts = m.sum(dim=1).clamp(min=1.0)             # (B, 1)
    return summed / counts


def masked_mean_pool_einsum(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``einsum`` variant of :func:`masked_mean_pool` (same result, different reduction)."""
    m = mask.to(hidden_states.dtype)                       # (B, T)
    summed = torch.einsum("bth,bt->bh", hidden_states, m)  # (B, H)
    counts = m.sum(dim=1).clamp(min=1.0).unsqueeze(-1)     # (B, 1)
    return summed / counts


def masked_mean_pool_reference(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Explicit per-sequence reference pooling (clear but slow).

    Loops over the batch and averages only the selected rows, so it is an independent check
    on the vectorized implementations rather than a reshuffling of the same expression.
    """
    bool_mask = mask.to(torch.bool)
    out = []
    for h, m in zip(hidden_states, bool_mask):
        if bool(m.any()):
            out.append(h[m].mean(dim=0))
        else:
            out.append(torch.zeros(h.shape[-1], dtype=h.dtype, device=h.device))
    return torch.stack(out, dim=0)
