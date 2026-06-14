#!/usr/bin/env python3
"""Milestone 8: summarize raw benchmark CSVs into portfolio tables.

This script is the bridge from ``results/raw/`` (one timestamped CSV per benchmark run) to
the human-facing report in ``docs/RESULTS.md``. It is deliberately read-only with respect to
the raw data: it never re-runs a benchmark, it only *condenses* the CSVs that earlier
milestones already produced.

For each experiment it:
  1. selects the **richest** raw CSV (most data rows, ties broken by latest timestamp) — so a
     one-row smoke-test file never shadows the full benchmark matrix;
  2. writes a condensed summary to ``results/processed/<key>_summary.csv``;
  3. renders a Markdown table and splices it into ``docs/RESULTS.md`` between
     ``<!-- BEGIN <key> -->`` / ``<!-- END <key> -->`` markers.

Re-running it regenerates every table from the raw CSVs, so the report cannot silently drift
from the data. ``--check`` verifies the committed tables are up to date without writing
anything (exit 1 on drift) — useful in the final cleanup sweep / CI.

Examples:
    python scripts/07_summarize_results.py            # regenerate tables + processed CSVs
    python scripts/07_summarize_results.py --check     # fail if docs/RESULTS.md is stale
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
Row = Dict[str, str]


# --------------------------------------------------------------------------- io helpers
def read_rows(path: Path) -> List[Row]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def pick_richest(raw_dir: Path, prefix: str) -> Optional[Path]:
    """Latest, richest CSV for a benchmark family.

    Sort key is (data-row count, filename): the file with the most rows wins, and the
    timestamped filename breaks ties toward the most recent run. This keeps a quick
    ``--quick`` smoke run (a 1-row CSV) from being chosen over the full matrix.
    """
    candidates = sorted(raw_dir.glob(f"{prefix}_*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: (len(read_rows(p)), p.name))


def write_processed(processed_dir: Path, key: str, columns: Sequence[str], rows: List[Row]) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / f"{key}_summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return out


# --------------------------------------------------------------------- formatting helpers
def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]], aligns: Optional[Sequence[str]] = None) -> str:
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    sep = {"left": ":--", "right": "--:", "center": ":-:"}
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    rule = "| " + " | ".join(sep[a] for a in aligns) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, rule, body])


def fnum(x: str, nd: int = 0) -> str:
    """Format a numeric string with thousands separators / fixed decimals."""
    v = float(x)
    return f"{v:,.{nd}f}"


def caption(path: Path) -> str:
    return f"\n\n_Source: `results/raw/{path.name}`._"


# ------------------------------------------------------------------------- summarizers
# Each returns (markdown, processed_columns, processed_rows). Markdown is spliced into the
# report; the processed rows are written verbatim to results/processed/<key>_summary.csv.

def summarize_baseline(path: Path) -> Tuple[str, List[str], List[Row]]:
    rows = read_rows(path)
    bf16 = [r for r in rows if r["dtype"] == "bf16"]
    seqs = sorted({int(r["seq_len"]) for r in bf16})
    bss = sorted({int(r["batch_size"]) for r in bf16})
    by = {(int(r["batch_size"]), int(r["seq_len"])): r for r in bf16}

    headers = ["batch \\ seq"] + [str(s) for s in seqs]
    table_rows = []
    for bs in bss:
        cells = [fnum(by[(bs, s)]["tokens_per_sec"]) if (bs, s) in by else "—" for s in seqs]
        table_rows.append([bs] + cells)

    peak = max(bf16, key=lambda r: float(r["tokens_per_sec"]))
    peak_mem = max(float(r["max_memory_allocated_gb"]) for r in rows)
    any_oom = any(r["oom"] == "True" for r in rows)

    md = (
        "Real (non-pad) tokens/sec, **bf16**, same-length batches (no intra-batch padding):\n\n"
        + md_table(headers, table_rows)
        + f"\n\nThroughput saturates around **{fnum(peak['tokens_per_sec'])} tok/s** "
        f"(peak at batch {peak['batch_size']}, seq {peak['seq_len']}); small batches underuse the GPU "
        f"and long sequences at large batch fall back. Peak memory across the full bf16+fp16 matrix is "
        f"**{peak_mem:.1f} GB** of 24 GB — OOM events: **{'yes' if any_oom else 'none'}**. fp16 throughput "
        "tracks bf16 within a few percent (slightly lower)."
        + caption(path)
    )
    cols = ["dtype", "batch_size", "seq_len", "actual_tokens", "latency_ms", "tokens_per_sec", "max_memory_allocated_gb"]
    return md, cols, bf16


def summarize_correctness(path: Path) -> Tuple[str, List[str], List[Row]]:
    rows = read_rows(path)
    headers = ["check", "approx", "max abs err", "min cosine", "passed @1%"]
    order = [
        ("impl_equiv", "fp32"),
        ("pooled_vs_fp32", "fp16"),
        ("pooled_vs_fp32", "bf16"),
        ("hidden_vs_fp32", "fp16"),
        ("hidden_vs_fp32", "bf16"),
    ]
    table_rows = []
    for check, dt in order:
        sel = [r for r in rows if r["check"] == check and r["dtype"] == dt]
        if not sel:
            continue
        maxerr = max(float(r["max_abs_err"]) for r in sel)
        cos = [float(r["min_cosine_sim"]) for r in sel if r["min_cosine_sim"]]
        passed = all(r["passed"] == "True" for r in sel)
        label = "impl agreement (vectorized/einsum vs reference)" if check == "impl_equiv" else check.replace("_", " ")
        table_rows.append([
            label, dt, f"{maxerr:.2g}",
            f"{min(cos):.6f}" if cos else "—",
            "✓" if passed else "✗",
        ])
    md = (
        md_table(headers, table_rows)
        + "\n\nThe three pooling implementations are numerically identical in fp32 (≈2e-6). Against an fp32 "
        "encoder reference, **fp16 tracks far tighter than bf16**; pooled embeddings stay directionally "
        "near-identical in both low precisions (cosine ≥ 0.9996). The `✗` rows are honest: per-token hidden "
        "states in bf16/fp16 exceed a strict 1% tolerance, but the pooled vector — what downstream tasks use "
        "— does not, in fp16."
        + caption(path)
    )
    cols = ["check", "dtype", "reference", "tensor", "max_abs_err", "min_cosine_sim", "passed"]
    return md, cols, rows


def summarize_batching(path: Path) -> Tuple[str, List[str], List[Row]]:
    rows = read_rows(path)
    bss = sorted({int(r["batch_size"]) for r in rows})
    by = {(r["strategy"], int(r["batch_size"])): r for r in rows}
    headers = ["strategy", "batch", "padding waste", "real tok/s", "padded tok/s"]
    table_rows = []
    for strat in ("naive", "sorted"):
        for bs in bss:
            r = by.get((strat, bs))
            if not r:
                continue
            table_rows.append([
                strat, bs, f"{float(r['padding_fraction']) * 100:.1f}%",
                fnum(r["real_tokens_per_sec"]), fnum(r["padded_tokens_per_sec"]),
            ])
    speedups = []
    for bs in bss:
        n, s = by.get(("naive", bs)), by.get(("sorted", bs))
        if n and s:
            speedups.append(float(s["real_tokens_per_sec"]) / float(n["real_tokens_per_sec"]))
    md = (
        md_table(headers, table_rows)
        + f"\n\nLength-**sorting** the same 256-sequence pool cuts padding waste from **~41–44%** to "
        f"**~2–8%** and lifts real-token throughput by **{min(speedups):.2f}×–{max(speedups):.2f}×** — at a "
        "roughly constant *padded*-token rate. That is the honest framing: the GPU does the same raw compute; "
        "sorting just wastes less of it on padding. This is the cheapest win in the project (a sort, no kernel "
        "or precision change)."
        + caption(path)
    )
    cols = ["strategy", "batch_size", "padding_fraction", "real_tokens_per_sec", "padded_tokens_per_sec"]
    return md, cols, rows


def summarize_compile(path: Path) -> Tuple[str, List[str], List[Row]]:
    rows = read_rows(path)
    compiled = [r for r in rows if r["mode"] != "eager"]
    headers = ["mode", "seq", "batch", "cold start (s)", "steady (ms)", "speedup"]
    table_rows = []
    for r in compiled:
        table_rows.append([
            r["mode"], r["seq_len"], r["batch_size"],
            f"{float(r['cold_start_ms']) / 1000:.0f}",
            fnum(r["steady_latency_ms"], 1), f"{float(r['speedup_vs_eager']):.2f}×",
        ])
    sp = [float(r["speedup_vs_eager"]) for r in compiled if r["speedup_vs_eager"]]
    md = (
        "bf16, static shapes (`dynamic=False`). Speedup is eager-steady ÷ compiled-steady at the same shape:\n\n"
        + md_table(headers, table_rows)
        + f"\n\n`torch.compile` delivers a real **{min(sp):.2f}×–{max(sp):.2f}×** steady-state speedup — largest "
        "on the small overhead-bound shape, ~1.6× on the large compute-bound one. The honest catch is the "
        "**~28 s cold-start compile per shape**: with static shapes every new `(batch, seq_len)` recompiles, so "
        "this only pays off after hundreds-to-thousands of calls at a fixed shape."
        + caption(path)
    )
    cols = ["mode", "seq_len", "batch_size", "cold_start_ms", "steady_latency_ms", "tokens_per_sec", "speedup_vs_eager"]
    return md, cols, compiled


def summarize_triton(path: Path) -> Tuple[str, List[str], List[Row]]:
    rows = read_rows(path)
    shapes = []
    for r in rows:
        key = (int(r["batch_size"]), int(r["seq_len"]))
        if key not in shapes:
            shapes.append(key)
    by = {(r["impl"], int(r["batch_size"]), int(r["seq_len"])): r for r in rows}
    headers = ["shape (b×t)", "PyTorch (ms)", "einsum (ms)", "Triton (ms)", "Triton speedup", "Triton err vs fp32"]
    table_rows = []
    for bs, sl in shapes:
        pt, ein, tr = by.get(("pytorch", bs, sl)), by.get(("einsum", bs, sl)), by.get(("triton", bs, sl))
        table_rows.append([
            f"{bs}×{sl}",
            fnum(pt["latency_ms"], 3), fnum(ein["latency_ms"], 3), fnum(tr["latency_ms"], 3),
            f"{float(tr['speedup_vs_pytorch']):.2f}×", f"{float(tr['max_abs_err_vs_ref']):.1e}",
        ])
    sp = [float(by[("triton", bs, sl)]["speedup_vs_pytorch"]) for bs, sl in shapes]
    md = (
        "bf16. Speedup is vs the eager vectorized PyTorch pooling; error is max-abs vs the fp32 reference:\n\n"
        + md_table(headers, table_rows)
        + f"\n\nThe Triton kernel is **fastest in every shape** (**{min(sp):.2f}×–{max(sp):.2f}×** vs eager "
        "PyTorch) and **numerically near-exact** (~1e-8 vs ~1e-3 for the bf16 PyTorch paths), because it "
        "accumulates in fp32 and never materializes the `(B,T,H)` product. The essential caveat: pooling is "
        "only ~0.08 ms against a 25–225 ms encoder forward, so this is a clean micro-optimization with "
        "**negligible end-to-end effect** — included as a kernel-authoring demonstration, not a headline win."
        + caption(path)
    )
    cols = ["impl", "batch_size", "seq_len", "latency_ms", "speedup_vs_pytorch", "max_abs_err_vs_ref"]
    keep = [r for r in rows if r["impl"] in ("pytorch", "einsum", "triton")]
    return md, cols, keep


def summarize_profile(path: Path) -> Tuple[str, List[str], List[Row]]:
    rows = read_rows(path)
    total_pct = sum(float(r["pct_self_cuda"]) for r in rows)
    cats: Dict[str, float] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0.0) + float(r["pct_self_cuda"])
    order = sorted(cats, key=cats.get, reverse=True)
    headers = ["category", "share of forward", "what it is"]
    blurb = {
        "matmul": "FFN/projection GEMMs + flash-attention (attention is only ~4.7%)",
        "elementwise": "GeLU, bias-adds, residual adds, scaling — each an unfused kernel",
        "copy": "dtype casts + rotary-embedding `cat`",
        "softmax/norm": "LayerNorm (softmax is fused inside flash-attn)",
    }
    table_rows = [[c, f"{cats[c]:.1f}%", blurb.get(c, "")] for c in order]
    md = (
        "Leaf-CUDA-kernel self-time, bf16, seq 512 × batch 8 (top-15 kernels rolled into categories; they "
        f"cover **{total_pct:.0f}%** of the forward):\n\n"
        + md_table(headers, table_rows)
        + "\n\nThe split is ~even between **matmul** (the FFN linears, not attention) and **unfused "
        "elementwise** — and that elementwise share is exactly the fusion headroom that explains the "
        "`torch.compile` win above. See [`docs/PROFILING.md`](PROFILING.md) for the full kernel list."
        + caption(path)
    )
    cols = ["rank", "kernel", "self_cuda_ms", "pct_self_cuda", "category"]
    return md, cols, rows


SUMMARIZERS: Dict[str, Tuple[str, Callable]] = {
    "baseline": ("baseline", summarize_baseline),
    "correctness": ("correctness", summarize_correctness),
    "batching": ("batching", summarize_batching),
    "compile": ("compile", summarize_compile),
    "triton": ("triton_pooling", summarize_triton),
    "profile": ("profile", summarize_profile),
}


# ------------------------------------------------------------------------- report splicing
def splice(report: str, key: str, markdown: str) -> str:
    begin, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    block = f"{begin}\n{markdown}\n{end}"
    if not pattern.search(report):
        raise SystemExit(f"marker pair for '{key}' not found in report; add:\n{begin}\n{end}")
    return pattern.sub(lambda _: block, report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "results" / "raw"))
    ap.add_argument("--processed-dir", default=str(REPO_ROOT / "results" / "processed"))
    ap.add_argument("--results-md", default=str(REPO_ROOT / "docs" / "RESULTS.md"))
    ap.add_argument("--check", action="store_true", help="verify report is up to date; do not write")
    ap.add_argument("--no-write-md", action="store_true", help="write processed CSVs but not the report")
    args = ap.parse_args()

    raw_dir, processed_dir, md_path = Path(args.raw_dir), Path(args.processed_dir), Path(args.results_md)
    report = md_path.read_text() if md_path.exists() else ""

    for key, (prefix, fn) in SUMMARIZERS.items():
        path = pick_richest(raw_dir, prefix)
        if path is None:
            print(f"  [skip] no {prefix}_*.csv in {raw_dir}", file=sys.stderr)
            continue
        markdown, cols, rows = fn(path)
        if not args.check:
            out = write_processed(processed_dir, key, cols, rows)
            print(f"  [ok]   {key:12s} <- {path.name}  ->  {out.relative_to(REPO_ROOT)} ({len(rows)} rows)")
        report = splice(report, key, markdown)

    if args.no_write_md:
        return 0

    current = md_path.read_text() if md_path.exists() else ""
    if args.check:
        if current != report:
            print("docs/RESULTS.md is STALE — re-run scripts/07_summarize_results.py", file=sys.stderr)
            return 1
        print("docs/RESULTS.md is up to date.")
        return 0
    md_path.write_text(report)
    print(f"  [ok]   wrote {md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
