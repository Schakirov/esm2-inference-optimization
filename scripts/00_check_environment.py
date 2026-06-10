#!/usr/bin/env python3
"""Milestone 1: reproducible environment / GPU check for the ESM2 L4 project.

Prints a human-readable summary of the Python, PyTorch, CUDA, and GPU environment and,
optionally, an ``nvidia-smi`` snapshot. Designed to fail gracefully when CUDA or a GPU is
unavailable so the repo is still inspectable on a CPU-only machine.

Usage:
    python scripts/00_check_environment.py                 # print to stdout
    python scripts/00_check_environment.py --output results/raw/environment.txt
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# Packages whose versions are worth recording for reproducibility.
_PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "numpy",
    "pandas",
    "triton",
    "biopython",
    "typer",
    "rich",
]


def _pkg_version(name: str) -> str:
    """Return an installed package version, tolerating absent packages.

    ``biopython`` is imported as ``Bio`` but distributed as ``biopython``; we try the
    distribution name first, then a couple of import-name fallbacks.
    """
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        pass
    import_name = {"biopython": "Bio"}.get(name, name)
    try:
        module = importlib.import_module(import_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "not installed"


def _human_bytes(num_bytes: float) -> str:
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.2f} GiB"


def collect_report() -> str:
    lines: list[str] = []

    def out(line: str = "") -> None:
        lines.append(line)

    out("=" * 70)
    out("ESM2 L4 environment check")
    out(f"generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out("=" * 70)

    out("")
    out("[ System ]")
    out(f"  python_version   : {platform.python_version()}")
    out(f"  python_executable: {sys.executable}")
    out(f"  platform         : {platform.platform()}")
    out(f"  processor        : {platform.processor() or 'unknown'}")
    out(f"  machine          : {platform.machine()}")

    out("")
    out("[ PyTorch / CUDA ]")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch should be installed
        out(f"  torch import FAILED: {exc!r}")
        torch = None  # type: ignore[assignment]

    if torch is not None:
        cuda_available = torch.cuda.is_available()
        out(f"  torch_version    : {torch.__version__}")
        out(f"  cuda_version     : {torch.version.cuda}")
        out(f"  cudnn_version    : {torch.backends.cudnn.version() if cuda_available else 'n/a'}")
        out(f"  cuda_available   : {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            out(f"  device_count     : {device_count}")
            for idx in range(device_count):
                props = torch.cuda.get_device_properties(idx)
                cap = f"{props.major}.{props.minor}"
                out(f"  --- device {idx} ---")
                out(f"    name           : {props.name}")
                out(f"    capability     : {cap}")
                out(f"    total_memory   : {_human_bytes(props.total_memory)}")
                out(f"    multiprocessors: {props.multi_processor_count}")
                out(f"    mem_allocated  : {_human_bytes(torch.cuda.memory_allocated(idx))}")
                out(f"    mem_reserved   : {_human_bytes(torch.cuda.memory_reserved(idx))}")
            try:
                out(f"  bf16_supported   : {torch.cuda.is_bf16_supported()}")
            except Exception as exc:
                out(f"  bf16_supported   : unknown ({exc!r})")
        else:
            out("  (no CUDA device visible to PyTorch — GPU benchmarks will not run)")

    out("")
    out("[ Package versions ]")
    for name in _PACKAGES:
        out(f"  {name:<14}: {_pkg_version(name)}")

    out("")
    out("[ nvidia-smi ]")
    smi = shutil.which("nvidia-smi")
    if smi is None:
        out("  nvidia-smi not found on PATH")
    else:
        try:
            query = (
                "name,driver_version,memory.total,memory.used,"
                "compute_cap,temperature.gpu,utilization.gpu"
            )
            result = subprocess.run(
                [smi, f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                out(f"  query: {query}")
                for row in result.stdout.strip().splitlines():
                    out(f"    {row.strip()}")
            else:
                out(f"  nvidia-smi exited {result.returncode}: {result.stderr.strip()}")
        except Exception as exc:
            out(f"  nvidia-smi failed: {exc!r}")

    out("")
    out("=" * 70)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="ESM2 L4 environment check")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to also write the report (e.g. results/raw/environment.txt)",
    )
    args = parser.parse_args()

    report = collect_report()
    sys.stdout.write(report)

    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)
        print(f"[wrote] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
