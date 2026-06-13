#!/usr/bin/env python3
"""Milestone 5: torch.compile & static-shape behavior.

Measures whether compiling the ESM2 encoder with ``torch.compile`` beats eager execution on
the L4, and what it costs to get there. For each (mode, shape) cell it records two distinct
numbers that are easy to conflate:

- **cold start** — the first call at a given input shape, measured on the wall clock. For
  compiled modes this includes the (large) one-time compilation; for a static-shape compile a
  *new* shape recompiles, so this also exposes recompilation cost.
- **steady state** — CUDA-event-timed latency after warmup, i.e. the rate you actually get
  once compiled. ``speedup_vs_eager`` compares this to the eager row at the same shape.

By default shapes are compiled statically (``dynamic=False``), so each new (batch, seq_len)
triggers a recompile — that is the point of the "static shapes" study. Pass ``--dynamic`` to
ask Dynamo for shape-generic kernels instead.

This milestone reports what it measures. If compile does not beat eager for a shape, that is
recorded, not hidden — small/overhead-bound shapes often do not benefit, and the cold-start
cost must be amortized over many calls to pay off.

Examples:
    # Fast smoke test (eager vs default compile, one small shape)
    python scripts/04_bench_compile.py --quick

    # Custom run
    python scripts/04_bench_compile.py \
        --modes eager default reduce-overhead --seq-lens 128 512 --batch-sizes 8
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Make the in-repo package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.results import COMPILE_COLUMNS, timestamped_path, write_csv  # noqa: E402
from esm2_perf.sequences import generate_sequences  # noqa: E402
from esm2_perf.timing import time_cuda  # noqa: E402

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
_MODES = ["eager", "default", "reduce-overhead", "max-autotune"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESM2 torch.compile benchmark")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    p.add_argument(
        "--modes",
        nargs="+",
        default=["eager", "default"],
        choices=_MODES,
        help="execution modes; 'eager' is the reference for speedup",
    )
    p.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8])
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="compile with dynamic=True (shape-generic) instead of static per-shape",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: results/raw/compile_<timestamp>.csv)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: eager + default compile, seq 128, batch 4, fewer iters",
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
    """Tokenize ``batch_size`` synthetic sequences of ``seq_len`` amino acids (moved to GPU)."""
    seqs = generate_sequences(batch_size, seq_len, seed=seed)
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len + 2,  # +2 for ESM2 <cls>/<eos>
    )
    enc = {k: v.to("cuda") for k, v in enc.items()}
    actual_tokens = int(enc["attention_mask"].sum().item())
    return enc, actual_tokens


def make_runner(model, mode: str, dynamic: bool):
    """Return the callable to time: the raw model (eager) or a compiled wrapper."""
    if mode == "eager":
        return model
    return torch.compile(model, mode=mode, dynamic=dynamic)


def bench_cell(
    runner,
    enc,
    actual_tokens: int,
    *,
    base_row: Dict[str, object],
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    """Measure cold-start (first call, wall clock) and steady-state (CUDA events) for a cell."""
    row = dict(base_row)
    row["actual_tokens"] = actual_tokens
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Cold start: first call at this shape. Wall clock, because for compiled modes the cost
        # is dominated by host-side compilation, which CUDA events would not capture.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            runner(**enc)
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1000.0

        # Steady state: warmup then CUDA-event timed iterations at the same shape.
        def forward() -> None:
            with torch.inference_mode():
                runner(**enc)

        timing = time_cuda(forward, warmup=warmup, iters=iters)
        latency_s = timing.median_ms / 1000.0
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        row.update(
            cold_start_ms=round(cold_ms, 2),
            steady_latency_ms=round(timing.median_ms, 4),
            steady_mean_ms=round(timing.mean_ms, 4),
            steady_std_ms=round(timing.std_ms, 4),
            tokens_per_sec=round(actual_tokens / latency_s, 2),
            sequences_per_sec=round(int(base_row["batch_size"]) / latency_s, 3),
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

    if args.quick:
        modes = ["eager", "default"]
        seq_lens = [128]
        batch_sizes = [4]
        warmup, iters = 2, 5
    else:
        modes = args.modes
        seq_lens = args.seq_lens
        batch_sizes = args.batch_sizes
        warmup, iters = args.warmup, args.iters

    # Keep eager first so its steady latencies are available as the speedup baseline.
    modes = sorted(set(modes), key=lambda m: (m != "eager", _MODES.index(m)))

    dtype = _DTYPE_MAP[args.dtype]
    gpu_name = torch.cuda.get_device_name(0)
    out_path = (
        Path(args.output) if args.output else timestamped_path("results/raw", "compile")
    )

    print(f"model      : {args.model_name}")
    print(f"gpu        : {gpu_name}")
    print(f"dtype      : {args.dtype}")
    print(f"modes      : {modes}")
    print(f"seq_lens   : {seq_lens}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"dynamic    : {args.dynamic}")
    print(f"warmup/iters: {warmup}/{iters}")
    print(f"output     : {out_path}")
    print(f"torch      : {torch.__version__}")
    print("-" * 78)

    import torch._dynamo as dynamo

    print(f"[load] {args.model_name} as {args.dtype} ...", flush=True)
    tokenizer, model = load_model(args.model_name, dtype)

    # Absorb lazy CUDA/cuDNN init with one untimed eager forward so eager cold-start is honest.
    enc0, _ = build_batch(tokenizer, batch_sizes[0], seq_lens[0], args.seed)
    with torch.inference_mode():
        model(**enc0)
    torch.cuda.synchronize()
    del enc0

    rows: List[Dict[str, object]] = []
    eager_steady: Dict[Tuple[int, int], float] = {}
    try:
        for mode in modes:
            # Fresh Dynamo state per mode so cold-start reflects this mode's compilation only.
            dynamo.reset()
            runner = make_runner(model, mode, args.dynamic)
            print(f"[mode] {mode} (dynamic={args.dynamic})", flush=True)
            for seq_len in seq_lens:
                for batch_size in batch_sizes:
                    enc, actual_tokens = build_batch(
                        tokenizer, batch_size, seq_len, args.seed
                    )
                    base_row: Dict[str, object] = {c: "" for c in COMPILE_COLUMNS}
                    base_row.update(
                        model_name=args.model_name,
                        gpu_name=gpu_name,
                        dtype=args.dtype,
                        mode=mode,
                        dynamic=args.dynamic,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        warmup=warmup,
                        iters=iters,
                        oom=False,
                        notes="",
                    )
                    row = bench_cell(
                        runner,
                        enc,
                        actual_tokens,
                        base_row=base_row,
                        warmup=warmup,
                        iters=iters,
                    )

                    key = (seq_len, batch_size)
                    if mode == "eager" and not row["oom"] and not row["notes"]:
                        eager_steady[key] = float(row["steady_latency_ms"])
                    elif mode != "eager" and key in eager_steady and row["steady_latency_ms"] != "":
                        row["speedup_vs_eager"] = round(
                            eager_steady[key] / float(row["steady_latency_ms"]), 3
                        )

                    rows.append(row)
                    if row["oom"]:
                        status = "OOM"
                    elif row["notes"]:
                        status = str(row["notes"])
                    else:
                        spd = row["speedup_vs_eager"]
                        spd_s = f"  {spd}x vs eager" if spd != "" else ""
                        status = (
                            f"cold {row['cold_start_ms']:>9} ms  "
                            f"steady {row['steady_latency_ms']:>8} ms  "
                            f"{row['tokens_per_sec']:>9} tok/s{spd_s}"
                        )
                    print(
                        f"  {mode:>14} seq={seq_len:>4} bs={batch_size:>2}  {status}",
                        flush=True,
                    )
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    write_csv(out_path, rows, COMPILE_COLUMNS)
    print("-" * 78)
    print(f"[wrote] {out_path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
