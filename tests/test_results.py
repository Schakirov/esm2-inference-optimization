"""Tests for the CSV result writer and schema helpers."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esm2_perf.results import (  # noqa: E402
    BASELINE_COLUMNS,
    BATCHING_COLUMNS,
    COMPILE_COLUMNS,
    timestamp,
    timestamped_path,
    write_csv,
)


def test_write_and_roundtrip(tmp_path):
    cols = ["a", "b", "c"]
    rows = [{"a": 1, "b": "x", "c": 2.5}, {"a": 3, "b": "y", "c": 4.0}]
    out = write_csv(tmp_path / "sub" / "out.csv", rows, cols)
    assert out.exists()  # parent dir created

    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == cols
        read = list(reader)
    assert read[0]["a"] == "1" and read[0]["b"] == "x"
    assert read[1]["c"] == "4.0"


def test_missing_keys_filled_blank(tmp_path):
    out = write_csv(tmp_path / "o.csv", [{"a": 1}], ["a", "b"])
    with out.open() as f:
        read = list(csv.DictReader(f))
    assert read[0]["b"] == ""


def test_extra_keys_raise(tmp_path):
    with pytest.raises(ValueError):
        write_csv(tmp_path / "o.csv", [{"a": 1, "z": 9}], ["a", "b"])


def test_baseline_columns_have_required_fields():
    required = {
        "model_name",
        "gpu_name",
        "dtype",
        "batch_size",
        "seq_len",
        "actual_tokens",
        "latency_ms",
        "tokens_per_sec",
        "sequences_per_sec",
        "max_memory_allocated_gb",
        "oom",
        "notes",
    }
    assert required <= set(BASELINE_COLUMNS)


def test_batching_columns_have_required_fields():
    required = {
        "model_name",
        "gpu_name",
        "strategy",
        "batch_size",
        "actual_tokens",
        "padded_tokens",
        "padding_fraction",
        "real_tokens_per_sec",
        "padded_tokens_per_sec",
        "oom",
        "notes",
    }
    assert required <= set(BATCHING_COLUMNS)


def test_compile_columns_have_required_fields():
    required = {
        "model_name",
        "gpu_name",
        "mode",
        "dynamic",
        "batch_size",
        "seq_len",
        "cold_start_ms",
        "steady_latency_ms",
        "tokens_per_sec",
        "speedup_vs_eager",
        "oom",
        "notes",
    }
    assert required <= set(COMPILE_COLUMNS)


def test_schemas_have_no_duplicate_columns():
    for cols in (BASELINE_COLUMNS, BATCHING_COLUMNS, COMPILE_COLUMNS):
        assert len(cols) == len(set(cols))


def test_timestamp_format():
    ts = timestamp()
    assert ts.endswith("Z") and "T" in ts and len(ts) == 16


def test_timestamped_path():
    p = timestamped_path("results/raw", "baseline")
    assert p.name.startswith("baseline_") and p.suffix == ".csv"
