#!/usr/bin/env python3
"""Milestone 3: correctness & pooling harness.

Validates two things and records them to a timestamped CSV under ``results/raw/``:

1. **Pooling-implementation equivalence.** The vectorized, ``einsum``, and explicit
   per-sequence masked-mean-pooling implementations in ``esm2_perf.pooling`` must agree to
   near floating-point round-off on the same fp32 hidden states.

2. **Low-precision vs fp32 reference.** For each cell we run the ESM2 encoder in fp32 (the
   reference), then in bf16 and fp16, and compare both the per-token ``last_hidden_state``
   and the masked-mean-pooled sequence embedding against the fp32 result. We report max/mean
   absolute error, RMS error, max relative error, and (for pooled embeddings) the minimum
   cosine similarity across the batch. A ``passed`` flag is derived from
   ``torch.allclose(rtol, atol)``, but the raw error columns are the honest record.

Pooling runs over real amino-acid tokens, excluding ``<cls>``/``<eos>`` via the tokenizer's
``special_tokens_mask`` (toggle with ``--include-special``).

The fp32 reference is computed once per cell and released before the low-precision models
load, so each model's weights are resident alone (mirroring the baseline benchmark's memory
hygiene). Reference outputs are stashed on the CPU between phases.

Examples:
    # Fast smoke test (single small cell, bf16 only)
    python scripts/02_check_correctness.py --quick

    # Custom cells
    python scripts/02_check_correctness.py \
        --dtypes bf16 fp16 --seq-lens 128 512 --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Make the in-repo package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.pooling import (  # noqa: E402
    build_pool_mask,
    masked_mean_pool,
    masked_mean_pool_einsum,
    masked_mean_pool_reference,
)
from esm2_perf.results import CORRECTNESS_COLUMNS, timestamped_path, write_csv  # noqa: E402
from esm2_perf.sequences import generate_sequences  # noqa: E402

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESM2 correctness & pooling harness")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16", "fp16"],
        choices=["bf16", "fp16"],
        help="low-precision dtypes to compare against the fp32 reference",
    )
    p.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include-special",
        action="store_true",
        help="pool over <cls>/<eos> too (default: exclude them)",
    )
    p.add_argument(
        "--rtol", type=float, default=1e-2, help="relative tolerance for the passed flag"
    )
    p.add_argument(
        "--atol", type=float, default=1e-2, help="absolute tolerance for the passed flag"
    )
    p.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: results/raw/correctness_<timestamp>.csv)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: bf16 only, seq_len 128, batch 2",
    )
    return p.parse_args()


def load_model(model_name: str, dtype: torch.dtype):
    """Load an ESM2 encoder + tokenizer onto the GPU in the requested dtype."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=dtype)
    model.eval()
    model.to("cuda")
    return tokenizer, model


def build_batch(tokenizer, batch_size: int, seq_len: int, seed: int):
    """Tokenize ``batch_size`` synthetic sequences, returning a CPU encoding.

    ``return_special_tokens_mask`` is requested so pooling can drop ``<cls>``/``<eos>``.
    Kept on the CPU so the same input ids can be reused across the fp32/bf16/fp16 models.
    """
    seqs = generate_sequences(batch_size, seq_len, seed=seed)
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len + 2,  # +2 for ESM2 <cls>/<eos> special tokens
        return_special_tokens_mask=True,
    )
    return enc


def encode(model, enc, exclude_special: bool):
    """Forward pass + pooling for one model. Returns (hidden, pooled) on the GPU."""
    model_inputs = {
        "input_ids": enc["input_ids"].to("cuda"),
        "attention_mask": enc["attention_mask"].to("cuda"),
    }
    with torch.inference_mode():
        out = model(**model_inputs)
    hidden = out.last_hidden_state
    pool_mask = build_pool_mask(
        model_inputs["attention_mask"],
        enc["special_tokens_mask"].to("cuda"),
        exclude_special=exclude_special,
    )
    pooled = masked_mean_pool(hidden, pool_mask)
    return hidden, pooled


def error_stats(approx: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """Absolute/relative error metrics, computed in fp32."""
    a = approx.float()
    r = ref.float()
    diff = (a - r).abs()
    denom = r.abs().clamp(min=1e-8)
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "rms_err": diff.pow(2).mean().sqrt().item(),
        "max_rel_err": (diff / denom).max().item(),
    }


def min_cosine(approx: torch.Tensor, ref: torch.Tensor) -> float:
    """Minimum per-row cosine similarity (worst sequence in the batch)."""
    cos = torch.nn.functional.cosine_similarity(approx.float(), ref.float(), dim=-1)
    return cos.min().item()


def _base_row(model_name: str, gpu_name: str, seq_len: int, batch_size: int, exclude_special: bool):
    row = {c: "" for c in CORRECTNESS_COLUMNS}
    row.update(
        model_name=model_name,
        gpu_name=gpu_name,
        seq_len=seq_len,
        batch_size=batch_size,
        exclude_special=exclude_special,
    )
    return row


def impl_equiv_rows(hidden_fp32, pool_mask, *, base) -> List[Dict[str, object]]:
    """Cross-check the three pooling implementations on the same fp32 hidden states."""
    ref = masked_mean_pool_reference(hidden_fp32, pool_mask)
    rows = []
    for name, fn in (("vectorized", masked_mean_pool), ("einsum", masked_mean_pool_einsum)):
        approx = fn(hidden_fp32, pool_mask)
        stats = error_stats(approx, ref)
        row = dict(base)
        row.update(
            check="impl_equiv",
            dtype="fp32",
            reference=f"fp32:{name}_vs_reference",
            tensor="pooled",
            min_cosine_sim=round(min_cosine(approx, ref), 8),
            rtol=1e-5,
            atol=1e-6,
            passed=bool(torch.allclose(approx.float(), ref.float(), rtol=1e-5, atol=1e-6)),
            **{k: round(v, 8) for k, v in stats.items()},
        )
        rows.append(row)
    return rows


def precision_rows(
    tensor_name, approx, ref, dtype_name, *, base, rtol, atol, with_cosine
) -> Dict[str, object]:
    """One comparison row for a low-precision tensor vs its fp32 reference."""
    stats = error_stats(approx, ref)
    row = dict(base)
    row.update(
        check=f"{tensor_name}_vs_fp32",
        dtype=dtype_name,
        reference="fp32",
        tensor=tensor_name,
        rtol=rtol,
        atol=atol,
        passed=bool(torch.allclose(approx.float(), ref.float(), rtol=rtol, atol=atol)),
        **{k: round(v, 8) for k, v in stats.items()},
    )
    if with_cosine:
        row["min_cosine_sim"] = round(min_cosine(approx, ref), 8)
    return row


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; this harness requires a GPU.", file=sys.stderr)
        return 1

    if args.quick:
        dtypes = ["bf16"]
        seq_lens = [128]
        batch_size = 2
    else:
        dtypes = args.dtypes
        seq_lens = args.seq_lens
        batch_size = args.batch_size

    exclude_special = not args.include_special
    gpu_name = torch.cuda.get_device_name(0)
    out_path = (
        Path(args.output)
        if args.output
        else timestamped_path("results/raw", "correctness")
    )

    print(f"model         : {args.model_name}")
    print(f"gpu           : {gpu_name}")
    print(f"dtypes        : {dtypes}  (vs fp32 reference)")
    print(f"seq_lens      : {seq_lens}")
    print(f"batch_size    : {batch_size}")
    print(f"exclude_special: {exclude_special}")
    print(f"rtol/atol     : {args.rtol}/{args.atol}")
    print(f"output        : {out_path}")
    print("-" * 70)

    # Phase 1: fp32 reference. Compute and stash hidden+pooled (and the pool mask) on the CPU
    # for every cell, plus the implementation-equivalence rows, then release the fp32 model.
    rows: List[Dict[str, object]] = []
    refs: Dict[int, Dict[str, torch.Tensor]] = {}
    encs: Dict[int, object] = {}
    print("[load] fp32 reference ...", flush=True)
    tokenizer, model = load_model(args.model_name, torch.float32)
    try:
        for seq_len in seq_lens:
            enc = build_batch(tokenizer, batch_size, seq_len, args.seed)
            encs[seq_len] = enc
            hidden, pooled = encode(model, enc, exclude_special)
            pool_mask = build_pool_mask(
                enc["attention_mask"].to("cuda"),
                enc["special_tokens_mask"].to("cuda"),
                exclude_special=exclude_special,
            )
            base = _base_row(args.model_name, gpu_name, seq_len, batch_size, exclude_special)
            equiv = impl_equiv_rows(hidden, pool_mask, base=base)
            rows.extend(equiv)
            worst_equiv = max(float(r["max_abs_err"]) for r in equiv)
            print(
                f"  fp32 seq={seq_len:>4}  impl_equiv max_abs_err={worst_equiv:.2e}",
                flush=True,
            )
            # Stash references on the CPU so the fp32 model can be freed.
            refs[seq_len] = {"hidden": hidden.float().cpu(), "pooled": pooled.float().cpu()}
            del hidden, pooled, pool_mask
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Phase 2: each low-precision model vs the stashed fp32 reference.
    for dtype_name in dtypes:
        dtype = _DTYPE_MAP[dtype_name]
        print(f"[load] {dtype_name} ...", flush=True)
        tokenizer, model = load_model(args.model_name, dtype)
        try:
            for seq_len in seq_lens:
                base = _base_row(
                    args.model_name, gpu_name, seq_len, batch_size, exclude_special
                )
                hidden, pooled = encode(model, encs[seq_len], exclude_special)
                ref = refs[seq_len]
                hid_row = precision_rows(
                    "hidden",
                    hidden,
                    ref["hidden"].to("cuda"),
                    dtype_name,
                    base=base,
                    rtol=args.rtol,
                    atol=args.atol,
                    with_cosine=False,
                )
                pool_row = precision_rows(
                    "pooled",
                    pooled,
                    ref["pooled"].to("cuda"),
                    dtype_name,
                    base=base,
                    rtol=args.rtol,
                    atol=args.atol,
                    with_cosine=True,
                )
                rows.append(hid_row)
                rows.append(pool_row)
                print(
                    f"  {dtype_name} seq={seq_len:>4}  "
                    f"hidden max_abs={hid_row['max_abs_err']:.3e}  "
                    f"pooled max_abs={pool_row['max_abs_err']:.3e}  "
                    f"cos>={pool_row['min_cosine_sim']:.5f}  "
                    f"pass={pool_row['passed']}",
                    flush=True,
                )
                del hidden, pooled
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    write_csv(out_path, rows, CORRECTNESS_COLUMNS)
    print("-" * 70)
    print(f"[wrote] {out_path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
