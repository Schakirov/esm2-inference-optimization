#!/usr/bin/env python3
"""Milestone 4: variable-length batching & padding-efficiency benchmark.

Real proteomes have sequences of many lengths. When such sequences are batched, each is
padded to the longest in its batch, so the GPU computes over padding tokens that carry no
signal. This benchmark quantifies that waste and how much a simple **length-sorting** batching
strategy recovers.

Over a fixed pool of variable-length synthetic sequences, for each (strategy, batch_size) cell
it plans the batches (``esm2_perf.batching``), pre-tokenizes them onto the GPU, then times one
full pass over all batches with CUDA events under ``torch.inference_mode()``. It records the
padding fraction and throughput as both *real* (non-pad) tokens/sec and *padded* compute
tokens/sec, plus peak memory. OOM is captured per cell.

The expected story: ``sorted`` does the same padded-compute work rate as ``naive`` but wastes
far fewer tokens on padding, so its real-token throughput is markedly higher.

Examples:
    # Fast smoke test
    python scripts/03_bench_batching.py --quick

    # Custom run
    python scripts/03_bench_batching.py \
        --num-seqs 256 --min-len 64 --max-len 512 --batch-sizes 8 16 32
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

from esm2_perf.batching import STRATEGIES, plan_batches  # noqa: E402
from esm2_perf.results import BATCHING_COLUMNS, timestamped_path, write_csv  # noqa: E402
from esm2_perf.sequences import generate_variable_sequences  # noqa: E402
from esm2_perf.timing import time_cuda  # noqa: E402

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESM2 variable-length batching benchmark")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    p.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES),
        choices=list(STRATEGIES),
    )
    p.add_argument("--num-seqs", type=int, default=128)
    p.add_argument("--min-len", type=int, default=64)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: results/raw/batching_<timestamp>.csv)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: small pool, batch 8, both strategies, fewer iters",
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


def build_batch_encodings(tokenizer, seqs, plan, max_len):
    """Tokenize each planned batch (padded to its own max) and move it to the GPU."""
    encs = []
    for batch in plan.batches:
        bseqs = [seqs[i] for i in batch]
        enc = tokenizer(
            bseqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len + 2,  # +2 for ESM2 <cls>/<eos>; no truncation since seqs <= max_len
        )
        encs.append({k: v.to("cuda") for k, v in enc.items()})
    return encs


def bench_cell(
    tokenizer,
    model,
    seqs,
    lengths,
    *,
    model_name: str,
    gpu_name: str,
    dtype_name: str,
    strategy: str,
    batch_size: int,
    min_len: int,
    max_len: int,
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    """Benchmark one (strategy, batch_size) cell; capture OOM gracefully."""
    plan = plan_batches(lengths, batch_size, strategy)
    row: Dict[str, object] = {c: "" for c in BATCHING_COLUMNS}
    row.update(
        model_name=model_name,
        gpu_name=gpu_name,
        dtype=dtype_name,
        strategy=strategy,
        num_seqs=len(seqs),
        min_len=min_len,
        max_len=max_len,
        batch_size=batch_size,
        num_batches=plan.num_batches,
        actual_tokens=plan.actual_tokens,
        padded_tokens=plan.padded_tokens,
        padding_fraction=round(plan.padding_fraction, 4),
        padding_waste_ratio=round(plan.padding_waste_ratio, 4),
        warmup=warmup,
        iters=iters,
        oom=False,
        notes="",
    )
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        encs = build_batch_encodings(tokenizer, seqs, plan, max_len)

        def forward_all() -> None:
            with torch.inference_mode():
                for enc in encs:
                    model(**enc)

        timing = time_cuda(forward_all, warmup=warmup, iters=iters)
        latency_s = timing.median_ms / 1000.0
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        row.update(
            latency_ms=round(timing.median_ms, 4),
            real_tokens_per_sec=round(plan.actual_tokens / latency_s, 2),
            padded_tokens_per_sec=round(plan.padded_tokens / latency_s, 2),
            sequences_per_sec=round(len(seqs) / latency_s, 3),
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
        num_seqs = 16
        min_len, max_len = 64, 256
        batch_sizes = [8]
        strategies = list(STRATEGIES)
        warmup, iters = 1, 2
    else:
        num_seqs = args.num_seqs
        min_len, max_len = args.min_len, args.max_len
        batch_sizes = args.batch_sizes
        strategies = args.strategies
        warmup, iters = args.warmup, args.iters

    if min_len < 1 or max_len < min_len:
        print(f"ERROR: require 1 <= min-len <= max-len, got {min_len}/{max_len}", file=sys.stderr)
        return 1

    # One fixed pool of variable-length sequences, shared across all strategies/batch sizes so
    # the only thing that changes is how they are grouped. Token length = amino acids + 2.
    seqs = generate_variable_sequences(num_seqs, min_len, max_len, seed=args.seed)
    lengths = [len(s) + 2 for s in seqs]

    gpu_name = torch.cuda.get_device_name(0)
    dtype = _DTYPE_MAP[args.dtype]
    out_path = (
        Path(args.output) if args.output else timestamped_path("results/raw", "batching")
    )

    print(f"model      : {args.model_name}")
    print(f"gpu        : {gpu_name}")
    print(f"dtype      : {args.dtype}")
    print(f"pool       : {num_seqs} seqs, lengths in [{min_len}, {max_len}] aa (seed {args.seed})")
    print(f"strategies : {strategies}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"warmup/iters: {warmup}/{iters}")
    print(f"output     : {out_path}")
    print("-" * 78)

    print(f"[load] {args.model_name} as {args.dtype} ...", flush=True)
    tokenizer, model = load_model(args.model_name, dtype)
    rows: List[Dict[str, object]] = []
    try:
        for strategy in strategies:
            for batch_size in batch_sizes:
                row = bench_cell(
                    tokenizer,
                    model,
                    seqs,
                    lengths,
                    model_name=args.model_name,
                    gpu_name=gpu_name,
                    dtype_name=args.dtype,
                    strategy=strategy,
                    batch_size=batch_size,
                    min_len=min_len,
                    max_len=max_len,
                    warmup=warmup,
                    iters=iters,
                )
                rows.append(row)
                if row["oom"]:
                    status = "OOM"
                elif row["notes"]:
                    status = str(row["notes"])
                else:
                    status = (
                        f"pad={row['padding_fraction']:<6} "
                        f"{row['latency_ms']:>9} ms  "
                        f"real {row['real_tokens_per_sec']:>10} tok/s  "
                        f"padded {row['padded_tokens_per_sec']:>10} tok/s  "
                        f"{row['max_memory_allocated_gb']:>6} GB"
                    )
                print(
                    f"  {strategy:>6} bs={batch_size:>3}  {status}",
                    flush=True,
                )
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    write_csv(out_path, rows, BATCHING_COLUMNS)
    print("-" * 78)
    print(f"[wrote] {out_path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
