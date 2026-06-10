#!/usr/bin/env python3
"""Milestone 2: baseline ESM2 inference benchmark.

Loads ``facebook/esm2_t33_650M_UR50D`` via Hugging Face ``transformers`` and measures
forward-pass latency, throughput, and peak memory across dtypes, sequence lengths, and
batch sizes on a single GPU. Inputs are deterministic synthetic protein sequences (fixed
length per cell, so there is no intra-batch padding waste — that is studied separately in
the batching milestone).

Timing uses CUDA events with warmup and synchronization under ``torch.inference_mode()``.
OOM is captured per cell rather than aborting the run. Results are written to a timestamped
CSV under ``results/raw/``.

Examples:
    # Fast smoke test (single small cell)
    python scripts/01_bench_baseline.py --quick

    # Custom matrix
    python scripts/01_bench_baseline.py \
        --dtypes bf16 fp16 --seq-lens 128 512 --batch-sizes 1 4 16
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

from esm2_perf.results import BASELINE_COLUMNS, timestamped_path, write_csv  # noqa: E402
from esm2_perf.sequences import generate_sequences  # noqa: E402
from esm2_perf.timing import time_cuda  # noqa: E402

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline ESM2 inference benchmark")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16", "fp16"],
        choices=list(_DTYPE_MAP),
        help="dtypes to benchmark (fp32 is slow/large; use for small cells only)",
    )
    p.add_argument("--seq-lens", nargs="+", type=int, default=[128, 256, 512, 1022])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: results/raw/baseline_<timestamp>.csv)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: bf16, seq_len 128, batch 1, fewer iters",
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
    """Tokenize ``batch_size`` synthetic sequences of ``seq_len`` amino acids.

    Returns the moved-to-GPU encoding and the number of real (non-pad) tokens. Because all
    sequences share a length, padding to the longest is a no-op and ``actual_tokens``
    equals the full tensor token count.
    """
    seqs = generate_sequences(batch_size, seq_len, seed=seed)
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len + 2,  # +2 for ESM2 <cls>/<eos> special tokens
    )
    enc = {k: v.to("cuda") for k, v in enc.items()}
    actual_tokens = int(enc["attention_mask"].sum().item())
    return enc, actual_tokens


def bench_cell(
    tokenizer,
    model,
    *,
    model_name: str,
    gpu_name: str,
    dtype_name: str,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
    seed: int,
) -> Dict[str, object]:
    """Benchmark one (dtype, batch_size, seq_len) cell; capture OOM gracefully."""
    row: Dict[str, object] = {
        "model_name": model_name,
        "gpu_name": gpu_name,
        "dtype": dtype_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "actual_tokens": "",
        "latency_ms": "",
        "latency_mean_ms": "",
        "latency_std_ms": "",
        "tokens_per_sec": "",
        "sequences_per_sec": "",
        "max_memory_allocated_gb": "",
        "warmup": warmup,
        "iters": iters,
        "oom": False,
        "notes": "",
    }
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        enc, actual_tokens = build_batch(tokenizer, batch_size, seq_len, seed)
        row["actual_tokens"] = actual_tokens

        def forward() -> None:
            with torch.inference_mode():
                model(**enc)

        timing = time_cuda(forward, warmup=warmup, iters=iters)
        latency_s = timing.median_ms / 1000.0
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        row.update(
            {
                "latency_ms": round(timing.median_ms, 4),
                "latency_mean_ms": round(timing.mean_ms, 4),
                "latency_std_ms": round(timing.std_ms, 4),
                "tokens_per_sec": round(actual_tokens / latency_s, 2),
                "sequences_per_sec": round(batch_size / latency_s, 3),
                "max_memory_allocated_gb": round(peak_gb, 4),
            }
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
        dtypes = ["bf16"]
        seq_lens = [128]
        batch_sizes = [1]
        warmup, iters = 2, 5
    else:
        dtypes = args.dtypes
        seq_lens = args.seq_lens
        batch_sizes = args.batch_sizes
        warmup, iters = args.warmup, args.iters

    gpu_name = torch.cuda.get_device_name(0)
    out_path = (
        Path(args.output)
        if args.output
        else timestamped_path("results/raw", "baseline")
    )

    print(f"model      : {args.model_name}")
    print(f"gpu        : {gpu_name}")
    print(f"dtypes     : {dtypes}")
    print(f"seq_lens   : {seq_lens}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"warmup/iters: {warmup}/{iters}")
    print(f"output     : {out_path}")
    print("-" * 70)

    rows: List[Dict[str, object]] = []
    for dtype_name in dtypes:
        dtype = _DTYPE_MAP[dtype_name]
        print(f"[load] {args.model_name} as {dtype_name} ...", flush=True)
        tokenizer, model = load_model(args.model_name, dtype)
        try:
            for seq_len in seq_lens:
                for batch_size in batch_sizes:
                    row = bench_cell(
                        tokenizer,
                        model,
                        model_name=args.model_name,
                        gpu_name=gpu_name,
                        dtype_name=dtype_name,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        warmup=warmup,
                        iters=iters,
                        seed=args.seed,
                    )
                    rows.append(row)
                    if row["oom"]:
                        status = "OOM"
                    elif row["notes"]:
                        status = str(row["notes"])
                    else:
                        status = (
                            f"{row['latency_ms']:>8} ms  "
                            f"{row['tokens_per_sec']:>10} tok/s  "
                            f"{row['max_memory_allocated_gb']:>6} GB"
                        )
                    print(
                        f"  {dtype_name:>4} seq={seq_len:>4} bs={batch_size:>2}  {status}",
                        flush=True,
                    )
        finally:
            # Fully release this dtype's model before loading the next one, so the next
            # dtype's peak-memory measurement is not contaminated by resident weights.
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    write_csv(out_path, rows, BASELINE_COLUMNS)
    print("-" * 70)
    print(f"[wrote] {out_path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
