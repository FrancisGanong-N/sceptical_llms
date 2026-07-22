"""Tests for the LO (linear optimization) Kaggle task wiring."""

from __future__ import annotations

from pathlib import Path

from benchmarks.lp_rate import BENCHMARK_CSV, load_benchmark
from scripts.build_lp_prompts import build_all


ROOT = Path(__file__).resolve().parent.parent


def test_lo_benchmark_has_eleven_json_prompts():
    prompts, items, benchmark = build_all()
    assert len(prompts) == 11
    assert len(benchmark) == 11
    assert all(row["variant"] == "json" for row in items)
    assert BENCHMARK_CSV.is_file()
    loaded = load_benchmark(BENCHMARK_CSV)
    assert len(loaded) == 11
    carpenter = loaded["carpenter_furniture__integrality__json"]
    assert carpenter.implicit_integer is True
    assert carpenter.implicit_nonnegative is False
    explicit = loaded["carpenter_furniture__integrality__explicit__json"]
    assert explicit.condition == "explicit"
    assert "whole numbers" in explicit.prompt
    assert "whole numbers" not in carpenter.prompt


def test_lo_benchmark_notebook_exists():
    notebook = ROOT / "benchmark" / "lo-benchmark.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "lo_normative_accuracy_2" in text
    assert "evaluate_lp_rate_benchmark" in text


def test_lo_task_json_exists():
    path = ROOT / "lo_normative_accuracy_2.task.json"
    assert path.is_file()
    assert "lo_normative_accuracy_2" in path.read_text(encoding="utf-8")


def test_lo_results_notebook_exists():
    notebook = ROOT / "benchmark" / "lo_results.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "score_table" in text
    assert "vignette_name" in text
    assert "merged_lo_results_from_kaggle_runs" in text


def test_merged_lo_helper_exported():
    from benchmarks.kaggle_runs import DEFAULT_LO_TASK_SLUG, merged_lo_results_from_kaggle_runs

    assert DEFAULT_LO_TASK_SLUG == "lo-normative-accuracy-2"
    assert callable(merged_lo_results_from_kaggle_runs)