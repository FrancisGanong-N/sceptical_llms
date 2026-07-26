"""Tests for the LO (linear optimization) Kaggle task wiring."""

from __future__ import annotations

from pathlib import Path

from benchmarks.lp_rate import BENCHMARK_CSV, load_benchmark
from scripts.build_lp_prompts import build_all


ROOT = Path(__file__).resolve().parent.parent


def test_lo_benchmark_has_fifteen_json_prompts():
    prompts, items, benchmark = build_all()
    assert len(prompts) == 15
    assert len(benchmark) == 15
    assert all(row["variant"] == "json" for row in items)
    assert BENCHMARK_CSV.is_file()
    loaded = load_benchmark(BENCHMARK_CSV)
    assert len(loaded) == 15
    carpenter = loaded["carpenter_furniture__integrality__json"]
    assert carpenter.implicit_integer is True
    assert carpenter.implicit_nonnegative is False
    print_shop = loaded["print_shop__integrality__json"]
    assert print_shop.implicit_integer is True
    assert print_shop.implicit_nonnegative is False
    pottery = loaded["pottery_studio__integrality__json"]
    assert pottery.implicit_integer is True
    assert pottery.implicit_nonnegative is False
    explicit = loaded["print_shop__integrality__explicit__json"]
    assert explicit.condition == "explicit"
    assert "whole numbers" in explicit.prompt
    assert "whole numbers" not in print_shop.prompt
    assert "whole numbers" not in carpenter.prompt
    assert "bookcases" in carpenter.prompt
    assert "posters" in print_shop.prompt
    assert "bowls" in pottery.prompt
    assert "vases" in pottery.prompt


def test_lo_benchmark_notebook_exists():
    notebook = ROOT / "benchmark" / "lo-benchmark.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "lo_normative_accuracy_3" in text
    assert "evaluate_lp_rate_benchmark" in text


def test_lo_task_json_exists():
    path = ROOT / "lo_normative_accuracy_3.task.json"
    assert path.is_file()
    assert "lo_normative_accuracy_3" in path.read_text(encoding="utf-8")


def test_lo_results_notebook_exists():
    notebook = ROOT / "benchmark" / "lo_results.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "score_table" in text
    assert "vignette_name" in text
    assert "merged_lo_results_from_kaggle_runs" in text


def test_lo_study_sheet_notebook_exists():
    notebook = ROOT / "benchmark" / "lo-study-sheet.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "build_lo_study_sheet" in text
    assert "lo-benchmark-study-sheet.txt" in text


def test_merged_lo_helper_exported():
    from benchmarks.kaggle_runs import DEFAULT_LO_TASK_SLUG, merged_lo_results_from_kaggle_runs

    assert DEFAULT_LO_TASK_SLUG == "lo-normative-accuracy-3"
    assert callable(merged_lo_results_from_kaggle_runs)