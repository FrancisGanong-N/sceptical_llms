"""Tests for the LO (linear optimization) Kaggle task wiring."""

from __future__ import annotations

from pathlib import Path

from benchmarks.lp_rate import BENCHMARK_CSV, load_benchmark
from scripts.build_lp_prompts import build_all


ROOT = Path(__file__).resolve().parent.parent


def test_lo_benchmark_has_thirty_six_prompts():
    prompts, items, benchmark = build_all()
    assert len(prompts) == 36
    assert len(benchmark) == 36
    assert {row["variant"] for row in items} == {
        "json",
        "needs_tacit_constraint",
        "detects_tacit_violation",
    }
    assert BENCHMARK_CSV.is_file()
    loaded = load_benchmark(BENCHMARK_CSV)
    assert len(loaded) == 36
    carpenter = loaded["carpenter_furniture__integrality__json"]
    assert carpenter.implicit_integer is True
    assert carpenter.implicit_nonnegative is False
    print_shop = loaded["print_shop__integrality__json"]
    assert print_shop.implicit_integer is True
    pottery = loaded["pottery_studio__integrality__json"]
    assert pottery.implicit_integer is True
    explicit = loaded["print_shop__integrality__explicit__json"]
    assert explicit.condition == "explicit"
    assert "whole numbers" in explicit.prompt
    assert "whole numbers" not in print_shop.prompt
    needs = loaded["print_shop__integrality__needs_tacit_constraint"]
    assert needs.normative_choice == "A"
    detects = loaded["print_shop__integrality__detects_tacit_violation"]
    assert detects.normative_choice == "B"
    coffee = loaded["coffee_roaster__fractional_ok__json"]
    assert coffee.implicit_integer is False
    coffee_needs = loaded["coffee_roaster__fractional_ok__needs_tacit_constraint"]
    assert coffee_needs.normative_choice == "A"
    chemicals = loaded["specialty_chemicals__signed_domain__json"]
    assert chemicals.implicit_nonnegative is False
    chemicals_needs = loaded[
        "specialty_chemicals__signed_domain__needs_tacit_constraint"
    ]
    assert chemicals_needs.normative_choice == "B"
    assert "gift_baskets" not in loaded
    assert "workshop_vehicles" not in loaded


def test_lo_benchmark_notebook_exists():
    notebook = ROOT / "benchmark" / "lo-benchmark.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "lo_normative_accuracy_5" in text
    assert "evaluate_lp_rate_benchmark" in text


def test_lo_task_json_exists():
    path = ROOT / "lo_normative_accuracy_5.task.json"
    assert path.is_file()
    assert "lo_normative_accuracy_5" in path.read_text(encoding="utf-8")
    assert not (ROOT / "lo_normative_accuracy_4.task.json").exists()


def test_lo_results_notebook_exists():
    notebook = ROOT / "benchmark" / "lo_results.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "score_table" in text
    assert "implicit_explicit_diff" in text
    assert "vignette_name" in text
    assert "merged_lo_results_from_kaggle_runs" in text
    assert "lo-normative-accuracy-5" in text


def test_lo_study_sheet_notebook_exists():
    notebook = ROOT / "benchmark" / "lo-study-sheet.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "build_lo_study_sheet" in text
    assert "lo-benchmark-study-sheet.txt" in text
    assert "lo-benchmark-study-sheet-all-prompts.txt" in text


def test_lo_study_sheet_includes_prompts_without_results(tmp_path):
    from scripts.build_lo_study_sheet import (
        _load_all_items,
        _load_non_explicit_items,
        build_study_sheet_text,
        write_all_prompts_study_sheet,
        write_study_sheet,
    )

    items = _load_non_explicit_items(ROOT)
    # 9 vignettes × (json implicit + needs + detects) = 27
    assert len(items) == 27
    assert all(r["condition"] != "explicit" for r in items)
    text = build_study_sheet_text(items, result_rows=[], title="test")
    assert "pottery studio" in text
    assert "coffee roaster" in text
    assert "workshop vehicles" not in text
    assert "gift baskets" not in text
    assert "(no model responses yet)" in text
    assert "__explicit__json" not in text
    out = write_study_sheet(root=ROOT, out_path=tmp_path / "sheet.txt")
    assert out.is_file()
    assert "PROMPT" in out.read_text(encoding="utf-8")

    all_items = _load_all_items(ROOT)
    assert len(all_items) == 36
    all_text = build_study_sheet_text(all_items, result_rows=[], title="all")
    assert "print_shop__integrality__explicit__json" in all_text
    assert "needs_tacit_constraint" in all_text
    assert "KEYED ANSWER" in all_text
    all_out = write_all_prompts_study_sheet(
        root=ROOT, out_path=tmp_path / "all.txt"
    )
    assert all_out.is_file()
    assert all_out.read_text(encoding="utf-8").count("PROMPT") == 36


def test_merged_lo_helper_exported():
    from benchmarks.kaggle_runs import DEFAULT_LO_TASK_SLUG, merged_lo_results_from_kaggle_runs

    assert DEFAULT_LO_TASK_SLUG == "lo-normative-accuracy-5"
    assert callable(merged_lo_results_from_kaggle_runs)


def test_benchmark_csv_has_constraint_columns():
    import csv

    with BENCHMARK_CSV.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    for col in (
        "optimization_criterion",
        "stated_constraints",
        "tacit_mistake",
    ):
        assert col in row
        assert row[col]
