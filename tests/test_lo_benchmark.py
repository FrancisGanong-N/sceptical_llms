"""Tests for the LO (linear optimization) Kaggle task wiring."""

from __future__ import annotations

from pathlib import Path

from benchmarks.lp_rate import BENCHMARK_CSV, load_benchmark
from scripts.build_lp_prompts import build_all


ROOT = Path(__file__).resolve().parent.parent


def test_lo_benchmark_has_six_json_prompts():
    prompts, items, benchmark = build_all()
    assert len(prompts) == 6
    assert len(benchmark) == 6
    assert all(row["variant"] == "json" for row in items)
    assert BENCHMARK_CSV.is_file()
    assert len(load_benchmark(BENCHMARK_CSV)) == 6


def test_lo_benchmark_notebook_exists():
    notebook = ROOT / "benchmark" / "lo-benchmark.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "lo_normative_accuracy" in text
    assert "evaluate_lp_rate_benchmark" in text


def test_lo_task_json_exists():
    path = ROOT / "lo_normative_accuracy.task.json"
    assert path.is_file()
    assert "lo_normative_accuracy" in path.read_text(encoding="utf-8")


def test_lo_tasks_module_defines_lo_tasks():
    source = (ROOT / "benchmarks" / "lp_rate_tasks.py").read_text(encoding="utf-8")
    assert 'name="lo_normative_accuracy"' in source
    assert 'name="lo_naive_confusion"' in source
    assert "evaluate_lp_rate_benchmark" in source
