#!/usr/bin/env python3
"""Milestone 6: Triton masked-mean-pooling kernel benchmark.

Benchmarks the hand-written Triton pooling kernel (`esm2_perf.triton_pooling`) against the
PyTorch pooling paths from Milestone 3 — vectorized, `einsum`, and `torch.compile`d — on
hidden-state tensors of shape ``(B, T, H=1280)`` with a realistic padding mask. For each
(impl, shape) cell it records steady-state latency (CUDA events), speedup vs the eager PyTorch
pooling, and the max absolute error against the fp32 per-sequence reference, so the kernel is
reported as both fast (or not) and correct.

Pooling is a small, memory-bound reduction and a tiny fraction of full encoder time, so this
isolates the op. The kernel is **not assumed** to beat PyTorch; whatever it does is recorded.

Examples:
    # Fast smoke test
    python scripts/05_bench_triton_pooling.py --quick

    # Custom run
    python scripts/05_bench_triton_pooling.py \
        --dtype bf16 --seq-lens 128 512 1022 --batch-sizes 8 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Make the in-repo package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.pooling import (  # noqa: E402
    masked_mean_pool,
    masked_mean_pool_einsum,
    masked_mean_pool_reference,
)
from esm2_perf.results import POOLING_COLUMNS, timestamped_path, write_csv  # noqa: E402
from esm2_perf.timing import time_cuda  # noqa: E402
from esm2_perf import triton_pooling  # noqa: E402

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
HIDDEN_SIZE = 1280  # ESM2-650M hidden size
_ALL_IMPLS = ["pytorch", "einsum", "compiled", "triton"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESM2 Triton pooling-kernel benchmark")
    p.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    p.add_argument("--impls", nargs="+", default=_ALL_IMPLS, choices=_ALL_IMPLS)
    p.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1022])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32])
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: results/raw/triton_pooling_<timestamp>.csv)",
    )
    p.add_argument("--quick", action="store_true", help="smoke test: one small shape, all impls")
    return p.parse_args()


def make_inputs(batch_size: int, seq_len: int, hidden: int, dtype, seed: int):
    """Random hidden states (B, T+2, H) and a realistic 0/1 padding mask, on the GPU."""
    T = seq_len + 2  # +2 for ESM2 <cls>/<eos>, matching the encoder's output length
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(batch_size, T, hidden, device="cuda", dtype=dtype, generator=gen)
    # Variable real lengths in [T//2, T] so there is genuine padding to skip.
    cpu_gen = torch.Generator().manual_seed(seed + 1)
    lengths = torch.randint(T // 2, T + 1, (batch_size,), generator=cpu_gen)
    idx = torch.arange(T).expand(batch_size, T)
    mask = (idx < lengths[:, None]).to(device="cuda", dtype=torch.float32)
    return hidden_states, mask


def get_impl(name: str):
    """Return a no-arg-friendly pooling callable ``fn(hidden, mask)`` for the named impl."""
    if name == "pytorch":
        return masked_mean_pool
    if name == "einsum":
        return masked_mean_pool_einsum
    if name == "triton":
        return triton_pooling.triton_masked_mean_pool
    if name == "compiled":
        return torch.compile(masked_mean_pool)
    raise ValueError(f"unknown impl {name!r}")


def bench_cell(
    impl_name: str,
    fn,
    hidden_states,
    mask,
    ref,
    *,
    base_row: Dict[str, object],
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    """Time one pooling impl at one shape; capture errors gracefully."""
    row = dict(base_row)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def forward() -> None:
            with torch.inference_mode():
                fn(hidden_states, mask)

        timing = time_cuda(forward, warmup=warmup, iters=iters)

        with torch.inference_mode():
            out = fn(hidden_states, mask).float()
        max_abs_err = (out - ref).abs().max().item()
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        row.update(
            latency_ms=round(timing.median_ms, 5),
            latency_mean_ms=round(timing.mean_ms, 5),
            latency_std_ms=round(timing.std_ms, 5),
            max_abs_err_vs_ref=float(f"{max_abs_err:.3e}"),
            max_memory_allocated_gb=round(peak_gb, 4),
        )
    except torch.cuda.OutOfMemoryError:
        row["oom"] = True
        row["notes"] = "CUDA OOM"
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - record and continue the matrix
        row["notes"] = f"error: {type(exc).__name__}: {exc}"[:200]
        torch.cuda.empty_cache()
    return row


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; this benchmark requires a GPU.", file=sys.stderr)
        return 1
    if "triton" in args.impls and not triton_pooling.HAS_TRITON:
        print("ERROR: triton is not installed; cannot benchmark the Triton kernel.", file=sys.stderr)
        return 1

    if args.quick:
        impls = list(args.impls)
        seq_lens = [128]
        batch_sizes = [8]
        warmup, iters = 3, 20
    else:
        impls = args.impls
        seq_lens = args.seq_lens
        batch_sizes = args.batch_sizes
        warmup, iters = args.warmup, args.iters

    # Keep pytorch first so its latency is the speedup baseline.
    impls = sorted(set(impls), key=lambda m: (m != "pytorch", _ALL_IMPLS.index(m)))

    dtype = _DTYPE_MAP[args.dtype]
    gpu_name = torch.cuda.get_device_name(0)
    out_path = (
        Path(args.output) if args.output else timestamped_path("results/raw", "triton_pooling")
    )

    print(f"gpu        : {gpu_name}")
    print(f"dtype      : {args.dtype}")
    print(f"impls      : {impls}")
    print(f"seq_lens   : {seq_lens}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"hidden     : {args.hidden_size}")
    print(f"warmup/iters: {warmup}/{iters}")
    print(f"output     : {out_path}")
    print("-" * 78)

    rows: List[Dict[str, object]] = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            hidden_states, mask = make_inputs(
                batch_size, seq_len, args.hidden_size, dtype, args.seed
            )
            # fp32 per-sequence reference (the correctness oracle) for this shape.
            ref = masked_mean_pool_reference(hidden_states.float(), mask)
            pytorch_latency = None
            for impl_name in impls:
                base_row: Dict[str, object] = {c: "" for c in POOLING_COLUMNS}
                base_row.update(
                    gpu_name=gpu_name,
                    dtype=args.dtype,
                    impl=impl_name,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_size=args.hidden_size,
                    warmup=warmup,
                    iters=iters,
                    oom=False,
                    notes="",
                )
                fn = get_impl(impl_name)
                row = bench_cell(
                    impl_name, fn, hidden_states, mask, ref,
                    base_row=base_row, warmup=warmup, iters=iters,
                )
                lat = row["latency_ms"]
                if impl_name == "pytorch" and lat != "":
                    pytorch_latency = float(lat)
                if lat != "" and pytorch_latency:
                    row["speedup_vs_pytorch"] = round(pytorch_latency / float(lat), 3)

                rows.append(row)
                if row["oom"]:
                    status = "OOM"
                elif row["notes"]:
                    status = str(row["notes"])
                else:
                    status = (
                        f"{row['latency_ms']:>9} ms  "
                        f"{row['speedup_vs_pytorch']:>6}x  "
                        f"err {row['max_abs_err_vs_ref']:>9}"
                    )
                print(
                    f"  seq={seq_len:>4} bs={batch_size:>3} {impl_name:>9}  {status}",
                    flush=True,
                )

    write_csv(out_path, rows, POOLING_COLUMNS)
    print("-" * 78)
    print(f"[wrote] {out_path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
