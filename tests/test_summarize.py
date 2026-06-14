"""Tests for the Milestone 8 results summarizer (scripts/07_summarize_results.py).

The summarizer is a standalone script, not a package module, so it is loaded by path. Only
its pure helpers are exercised here (no GPU, no real benchmarks): table rendering, the
richest-file selection rule, and the marker-splice that keeps docs/RESULTS.md in sync.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_summarize_results.py"


def _load():
    spec = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


summ = _load()


def test_fnum_formats_thousands_and_decimals():
    assert summ.fnum("22040.27") == "22,040"
    assert summ.fnum("0.097", 3) == "0.097"


def test_md_table_shapes_header_rule_and_rows():
    md = summ.md_table(["a", "b"], [[1, 2], [3, 4]])
    lines = md.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| :-- | --: |"          # first col left, rest right
    assert lines[2] == "| 1 | 2 |"
    assert lines[3] == "| 3 | 4 |"


def test_pick_richest_prefers_more_rows_over_latest(tmp_path):
    # A later-timestamped 1-row smoke file must NOT shadow the full matrix.
    (tmp_path / "baseline_20260101T000000Z.csv").write_text(
        "dtype,tokens_per_sec\nbf16,1\nbf16,2\nbf16,3\n"
    )
    (tmp_path / "baseline_20260999T000000Z.csv").write_text("dtype,tokens_per_sec\nbf16,9\n")
    chosen = summ.pick_richest(tmp_path, "baseline")
    assert chosen.name == "baseline_20260101T000000Z.csv"


def test_pick_richest_returns_none_when_absent(tmp_path):
    assert summ.pick_richest(tmp_path, "nope") is None


def test_splice_replaces_only_between_markers():
    report = "head\n<!-- BEGIN x -->\nOLD\n<!-- END x -->\ntail"
    out = summ.splice(report, "x", "NEW")
    assert "NEW" in out and "OLD" not in out
    assert out.startswith("head") and out.endswith("tail")


def test_splice_raises_on_missing_markers():
    with pytest.raises(SystemExit):
        summ.splice("no markers here", "x", "NEW")
