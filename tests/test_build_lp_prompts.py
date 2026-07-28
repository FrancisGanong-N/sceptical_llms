"""Tests for the LP tacit-constraint prompt builder and scoring."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks import lp_rate
from scripts.build_lp_prompts import build_all, load_lp_vignettes, write_csvs


def _vignette(name: str):
    return next(v for v in load_lp_vignettes() if v.name == name)


def _item(items, example_id: str):
    return next(r for r in items if r["example_id"] == example_id)


class TestVignetteOptima:
    def test_carpenter_furniture(self):
        best = max(
            37.5 * x + 50 * y
            for x in range(0, 7)
            for y in range(0, 7)
            if 1.5 * x + 2.25 * y <= 9 and 2.25 * x + 1.5 * y <= 9
        )
        assert best == 200
        assert abs(37.5 * 2.4 + 50 * 2.4 - 210) < 1e-9
        v = _vignette("carpenter furniture")
        assert v.true_objective == "200"
        assert v.true_solution == {"bookcases": 0, "desks": 4}
        assert v.violating_solution == {"bookcases": 2.4, "desks": 2.4}

    def test_print_shop(self):
        best = max(
            37.5 * x + 50 * y
            for x in range(0, 7)
            for y in range(0, 7)
            if 1.5 * x + 2.25 * y <= 9 and 2.25 * x + 1.5 * y <= 9
        )
        assert best == 200
        v = _vignette("print shop")
        assert v.true_solution == {"posters": 0, "booklets": 4}

    def test_pottery_studio(self):
        best = max(
            41.25 * x + 45 * y
            for x in range(0, 8)
            for y in range(0, 8)
            if 1.65 * x + 2.025 * y <= 9 and 2.025 * x + 1.65 * y <= 9
        )
        assert best == 180
        v = _vignette("pottery studio")
        assert v.true_solution == {"bowls": 0, "vases": 4}
        assert abs(float(v.naive_objective) - 211.22) < 0.01

    def test_charter_buses(self):
        best = min(
            820.5 * x + 700.25 * y
            for x in range(0, 6)
            for y in range(0, 6)
            if 50 * x + 30 * y >= 130
        )
        assert best == 2341.25
        assert abs(820.5 * (130 / 50) - 2133.3) < 1e-9

    def test_fund_allocation(self):
        best = max(
            0.09 * a + 0.045 * b
            for a in range(0, 10001, 50)
            for b in range(0, 10001, 50)
            if a + b <= 10000
            and 1.5 * a + 0.5 * b <= 9000
            and a <= 8500
        )
        assert abs(best - 630) < 1e-6
        v = _vignette("fund allocation")
        assert v.true_solution == {"fund_a": 4000, "fund_b": 6000}
        assert v.violating_solution["fund_b"] < 0

    def test_warehouse_shipping(self):
        best = min(
            2.5 * n + 4.0 * s
            for n in range(0, 17)
            for s in range(0, 17)
            if n + s >= 16 and n + 0.5 * s <= 12
        )
        assert best == 52
        v = _vignette("warehouse shipping")
        assert v.true_solution == {"warehouse_1": 8, "warehouse_2": 8}
        assert v.violating_solution["warehouse_2"] < 0


class TestVignetteSet:
    def test_failure_mode_coverage(self):
        modes = [v.failure_mode for v in load_lp_vignettes()]
        assert modes.count("integrality") == 4
        assert modes.count("nonnegativity") == 2
        assert modes.count("both") == 0
        assert modes.count("none") == 0
        names = {v.name for v in load_lp_vignettes()}
        assert "gift baskets" not in names
        assert "workshop vehicles" not in names


class TestBuildAll:
    def test_prompt_count_v4(self):
        prompts, items, benchmark = build_all()
        # 6 vignettes × (json×2 + needs + detects) = 24
        assert len(prompts) == 24
        assert len(items) == 24
        assert len(benchmark) == 24
        assert {row["variant"] for row in benchmark} == {
            "json",
            "needs_tacit_constraint",
            "detects_tacit_violation",
        }
        conditions = {row["condition"] for row in items}
        assert conditions == {"implicit", "explicit"}
        assert sum(1 for row in items if row["condition"] == "implicit") == 18
        assert sum(1 for row in items if row["condition"] == "explicit") == 6
        assert sum(1 for row in items if row["variant"] == "json") == 12
        assert sum(1 for row in items if row["variant"] == "needs_tacit_constraint") == 6
        assert sum(1 for row in items if row["variant"] == "detects_tacit_violation") == 6

    def test_json_prompt_shape(self):
        prompts, items, _ = build_all()
        row = _item(items, "print_shop__integrality__json")
        prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == row["example_id"]
        )
        assert '"solution"' in prompt
        assert '"cost"' in prompt
        assert "whole numbers" not in prompt
        assert row["condition"] == "implicit"
        assert "Reply with only the letter" not in prompt

    def test_explicit_parallel_spells_out_constraints(self):
        prompts, items, _ = build_all()
        implicit = _item(items, "print_shop__integrality__json")
        explicit = _item(items, "print_shop__integrality__explicit__json")
        assert implicit["condition"] == "implicit"
        assert explicit["condition"] == "explicit"
        imp_prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == implicit["example_id"]
        )
        exp_prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == explicit["example_id"]
        )
        assert "whole numbers" not in imp_prompt
        assert "whole numbers" in exp_prompt

    def test_needs_tacit_constraint_prompt(self):
        prompts, items, _ = build_all()
        row = _item(items, "carpenter_furniture__integrality__needs_tacit_constraint")
        assert row["normative_choice"] == "A"
        assert row["condition"] == "implicit"
        prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == row["example_id"]
        )
        assert "unstated constraints" in prompt.lower() or "not stated" in prompt.lower()
        assert "Reply with only the letter (A or B)." in prompt
        assert row["option_a_label"].startswith("Yes")
        assert "Bookcases and desks must be whole numbers" not in prompt

    def test_detects_tacit_violation_prompt(self):
        prompts, items, _ = build_all()
        row = _item(
            items, "carpenter_furniture__integrality__detects_tacit_violation"
        )
        assert row["normative_choice"] == "B"
        prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == row["example_id"]
        )
        assert '"bookcases": 2.4' in prompt or '"bookcases": 2.4' in prompt.replace(
            " ", ""
        )
        assert "2.4" in prompt
        assert "Reply with only the letter (A or B)." in prompt
        assert "sensible" in prompt.lower()

    def test_constraint_type_flags(self):
        _, items, _ = build_all()
        by_id = {row["example_id"]: row for row in items}
        assert by_id["fund_allocation__nonnegativity__json"]["implicit_integer"] == "false"
        assert (
            by_id["fund_allocation__nonnegativity__json"]["implicit_nonnegative"]
            == "true"
        )
        assert "gift_baskets__both__json" not in by_id
        assert "workshop_vehicles__none__json" not in by_id

    def test_write_csvs(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("scripts.build_lp_prompts.OUT_DIR", tmp_path / "lp")
        count = write_csvs()
        assert count == 24
        with (tmp_path / "lp" / "benchmark.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 24
        assert "violating_solution" in rows[0]


class TestScoring:
    def test_parse_lp_json(self):
        parsed = lp_rate.parse_lp_json(
            '{"solution": {"bookcases": 0, "desks": 4}, "cost": 200}'
        )
        assert parsed.parseable
        assert parsed.cost == 200.0

    def test_within_one_percent(self):
        assert lp_rate.within_relative_tolerance(202, 200)
        assert not lp_rate.within_relative_tolerance(210, 200)

    def test_correct_and_naive_json_answers(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("scripts.build_lp_prompts.OUT_DIR", tmp_path / "lp")
        write_csvs()
        items = lp_rate.load_benchmark(tmp_path / "lp" / "benchmark.csv")
        assert len(items) == 24

        correct_rows = []
        for example_id, item in items.items():
            if item.is_json_solve:
                payload = {
                    "solution": item.true_solution,
                    "cost": item.true_objective,
                }
                correct_rows.append(
                    {"example_id": example_id, "response": json.dumps(payload)}
                )
            else:
                letter = item.normative_choice
                correct_rows.append({"example_id": example_id, "response": letter})
        score = lp_rate.score_run_rows(correct_rows, items=items)
        assert score.accuracy == 1.0
        assert lp_rate.naive_confusion_rate(correct_rows, items=items) == 0.0

        naive_rows = []
        for example_id, item in items.items():
            if not item.is_json_solve:
                continue
            if item.naive_objective is None:
                continue
            if item.true_objective is not None and lp_rate.within_relative_tolerance(
                item.naive_objective, item.true_objective
            ):
                continue
            naive_rows.append(
                {
                    "example_id": example_id,
                    "response": json.dumps(
                        {"solution": {}, "cost": item.naive_objective}
                    ),
                }
            )
        # carpenter, print, pottery, charter × 2 conditions = 8
        assert len(naive_rows) == 8
        assert lp_rate.score_run_rows(naive_rows, items=items).accuracy == 0.0
        assert lp_rate.naive_confusion_rate(naive_rows, items=items) == 1.0

    def test_audit_scoring(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("scripts.build_lp_prompts.OUT_DIR", tmp_path / "lp")
        write_csvs()
        items = lp_rate.load_benchmark(tmp_path / "lp" / "benchmark.csv")
        needs_id = "print_shop__integrality__needs_tacit_constraint"
        detects_id = "print_shop__integrality__detects_tacit_violation"
        assert (
            lp_rate.score_run_rows(
                [{"example_id": needs_id, "response": "A"}], items=items
            ).accuracy
            == 1.0
        )
        assert (
            lp_rate.score_run_rows(
                [{"example_id": needs_id, "response": "B"}], items=items
            ).accuracy
            == 0.0
        )
        assert (
            lp_rate.score_run_rows(
                [{"example_id": detects_id, "response": "B"}], items=items
            ).accuracy
            == 1.0
        )
        assert (
            lp_rate.score_run_rows(
                [{"example_id": detects_id, "response": "A"}], items=items
            ).accuracy
            == 0.0
        )
        assert (
            lp_rate.accuracy_for_variant(
                [{"example_id": needs_id, "response": "A"}],
                "needs_tacit_constraint",
                items=items,
            )
            == 1.0
        )

    def test_near_miss_within_tolerance(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("scripts.build_lp_prompts.OUT_DIR", tmp_path / "lp")
        write_csvs()
        items = lp_rate.load_benchmark(tmp_path / "lp" / "benchmark.csv")
        example_id = "print_shop__integrality__json"
        near = [
            {
                "example_id": example_id,
                "response": '{"solution": {"posters": 0, "booklets": 4}, "cost": 201.5}',
            }
        ]
        far = [
            {
                "example_id": example_id,
                "response": '{"solution": {"posters": 2.4, "booklets": 2.4}, "cost": 210}',
            }
        ]
        assert lp_rate.score_run_rows(near, items=items).accuracy == 1.0
        assert lp_rate.score_run_rows(far, items=items).accuracy == 0.0
