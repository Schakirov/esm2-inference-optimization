#!/usr/bin/env python3
"""Milestone 7: profile the ESM2 encoder forward with torch.profiler.

Runs the encoder under `torch.profiler` at a representative shape and reports where GPU time
actually goes: the top operators by self-CUDA time, their share of the total, and peak memory.
This turns the earlier latency numbers into a kernel-level explanation (matmuls vs attention vs
elementwise/normalization).

Only a small summary is written to disk (a top-ops CSV); the full Chrome trace is optional via
``--trace`` and is *not* committed (it is large). This keeps the repo light per the project's
"no huge profiler traces" rule.

Examples:
    # Default: bf16, seq 512, batch 8, top 15 ops
    python scripts/06_profile_pytorch.py

    # Export a Chrome trace too (large; do not commit)
    python scripts/06_profile_pytorch.py --trace /tmp/esm2_trace.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.profiler import ProfilerActivity, profile

# Make the in-repo package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.results import timestamped_path, write_csv  # noqa: E402
from esm2_perf.sequences import generate_sequences  # noqa: E402

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

PROFILE_COLUMNS: List[str] = [
    "rank",
    "kernel",
    "calls",
    "self_cuda_ms",        # GPU time in this kernel, summed over all calls in the profiled window
    "pct_self_cuda",       # share of total GPU kernel time
    "category",            # matmul | elementwise | softmax/norm | reduction | copy | other
]

# Coarse categories from kernel-name substrings, so we can roll the long kernel list up into a
# "where does GPU time go" summary. Order matters: first match wins.
_CATEGORY_RULES = [
    ("matmul", ("gemm", "cutlass", "ampere", "s1688", "wmma", "dot_kernel")),
    ("softmax/norm", ("softmax", "layer_norm", "layernorm", "norm")),
    ("reduction", ("reduce", "sum_kernel", "mean")),
    ("copy", ("memcpy", "memset", "copy", "cat")),
    ("elementwise", ("elementwise", "gelu", "erf", "add", "mul", "activation")),
]


def _categorize(name: str) -> str:
    low = name.lower()
    for cat, needles in _CATEGORY_RULES:
        if any(n in low for n in needles):
            return cat
    return "other"


def _is_kernel(evt) -> bool:
    """A leaf CUDA kernel, not an aten:: dispatcher op or a cuda runtime call.

    key_averages() lists both the operator events and the device kernels they launch; summing
    both double-counts GPU time. The kernel-only view sums to true GPU time and is unambiguous.
    """
    key = evt.key
    return not key.startswith("aten::") and not key.startswith("cuda")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile the ESM2 encoder forward")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace", default=None, help="optional Chrome trace path (large; not committed)")
    p.add_argument(
        "--output",
        default=None,
        help="top-ops CSV path (default: results/raw/profile_<timestamp>.csv)",
    )
    return p.parse_args()


def load_model(model_name: str, dtype: torch.dtype):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=dtype)
    model.eval()
    model.to("cuda")
    return tokenizer, model


def build_batch(tokenizer, batch_size: int, seq_len: int, seed: int):
    seqs = generate_sequences(batch_size, seq_len, seed=seed)
    enc = tokenizer(
        seqs, return_tensors="pt", padding=True, truncation=True, max_length=seq_len + 2
    )
    return {k: v.to("cuda") for k, v in enc.items()}


def _cuda_self_us(evt) -> float:
    """Self GPU time in microseconds, tolerant of torch's device/cuda attribute renames."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v:
            return float(v)
    return 0.0


def _cuda_total_us(evt) -> float:
    for attr in ("device_time_total", "cuda_time_total"):
        v = getattr(evt, attr, None)
        if v:
            return float(v)
    return 0.0


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; profiling requires a GPU.", file=sys.stderr)
        return 1

    dtype = _DTYPE_MAP[args.dtype]
    gpu_name = torch.cuda.get_device_name(0)
    out_path = Path(args.output) if args.output else timestamped_path("results/raw", "profile")

    print(f"model      : {args.model_name}")
    print(f"gpu        : {gpu_name}")
    print(f"dtype      : {args.dtype}   seq_len={args.seq_len}  batch_size={args.batch_size}")
    print(f"warmup/iters: {args.warmup}/{args.iters}")
    print(f"output     : {out_path}")
    print(f"torch      : {torch.__version__}")
    print("-" * 78)

    print("[load] model ...", flush=True)
    tokenizer, model = load_model(args.model_name, dtype)
    enc = build_batch(tokenizer, args.batch_size, args.seq_len, args.seed)

    # Warmup outside the profiler so lazy init / autotuning is not attributed to the kernels.
    for _ in range(args.warmup):
        with torch.inference_mode():
            model(**enc)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("[profile] running ...", flush=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(args.iters):
            with torch.inference_mode():
                model(**enc)
            torch.cuda.synchronize()

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Kernel-level view: leaf CUDA kernels only (no aten:: ops) so GPU time is not double-counted.
    kernels = [e for e in prof.key_averages() if _is_kernel(e) and _cuda_self_us(e) > 0]
    total_self_us = sum(_cuda_self_us(e) for e in kernels) or 1.0
    ranked = sorted(kernels, key=_cuda_self_us, reverse=True)

    rows: List[Dict[str, object]] = []
    print(f"\n{'rank':>4}  {'kernel':<46} {'calls':>6} {'self ms':>9} {'% ':>6}  category")
    for i, e in enumerate(ranked[: args.topk], start=1):
        self_ms = _cuda_self_us(e) / 1000.0
        pct = 100.0 * _cuda_self_us(e) / total_self_us
        cat = _categorize(e.key)
        rows.append(
            {
                "rank": i,
                "kernel": e.key,
                "calls": e.count,
                "self_cuda_ms": round(self_ms, 4),
                "pct_self_cuda": round(pct, 2),
                "category": cat,
            }
        )
        print(f"{i:>4}  {e.key[:46]:<46} {e.count:>6} {self_ms:>9.3f} {pct:>5.1f}%  {cat}")

    # Category rollup over ALL kernels (not just the top-k).
    cat_us: Dict[str, float] = {}
    for e in kernels:
        cat_us[_categorize(e.key)] = cat_us.get(_categorize(e.key), 0.0) + _cuda_self_us(e)

    total_self_ms = total_self_us / 1000.0
    print("-" * 78)
    print("by category (share of GPU kernel time):")
    for cat, us in sorted(cat_us.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {cat:<14} {us / 1000.0:>9.2f} ms   {100.0 * us / total_self_us:>5.1f}%")
    print("-" * 78)
    print(f"total GPU kernel time: {total_self_ms:.2f} ms over {args.iters} iters "
          f"({total_self_ms / args.iters:.2f} ms/iter)")
    print(f"peak memory allocated: {peak_gb:.3f} GB")

    write_csv(out_path, rows, PROFILE_COLUMNS)
    print(f"[wrote] {out_path}  ({len(rows)} rows)")

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"[trace] {args.trace}  (large; not for committing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
