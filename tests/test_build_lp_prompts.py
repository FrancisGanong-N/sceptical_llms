"""Tests for the LP implicit-constraint prompt builder and scoring."""

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

    def test_print_shop(self):
        best = max(
            37.5 * x + 50 * y
            for x in range(0, 7)
            for y in range(0, 7)
            if 1.5 * x + 2.25 * y <= 9 and 2.25 * x + 1.5 * y <= 9
        )
        assert best == 200
        assert abs(37.5 * 2.4 + 50 * 2.4 - 210) < 1e-9
        v = _vignette("print shop")
        assert v.true_objective == "200"
        assert v.true_solution == {"posters": 0, "booklets": 4}

    def test_pottery_studio(self):
        # Mixed ±10% relative to carpenter / print-shop times and profits;
        # capacity restored to integer 9.
        best = max(
            41.25 * x + 45 * y
            for x in range(0, 8)
            for y in range(0, 8)
            if 1.65 * x + 2.025 * y <= 9 and 2.025 * x + 1.65 * y <= 9
        )
        assert best == 180
        cont = 9 / (1.65 + 2.025)
        assert abs(cont - 2.448979) < 1e-5
        assert abs(41.25 * cont + 45 * cont - 211.2245) < 1e-3
        v = _vignette("pottery studio")
        assert v.true_objective == "180"
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
        assert abs(2341.25 - 2133.3) / 2341.25 > 0.01

    def test_fund_allocation(self):
        # max 0.09 a + 0.045 b
        # a+b <= 10000, 1.5a+0.5b <= 9000, a <= 8500, a,b >= 0
        best = max(
            0.09 * a + 0.045 * b
            for a in range(0, 10001, 50)
            for b in range(0, 10001, 50)
            if a + b <= 10000
            and 1.5 * a + 0.5 * b <= 9000
            and a <= 8500
        )
        assert abs(best - 630) < 1e-6
        # Corner (4000, 6000) binds budget and risk.
        assert abs(0.09 * 4000 + 0.045 * 6000 - 630) < 1e-9
        assert abs(1.5 * 4000 + 0.5 * 6000 - 9000) < 1e-9
        v = _vignette("fund allocation")
        assert v.true_solution == {"fund_a": 4000, "fund_b": 6000}
        assert v.true_objective == "630"
        # Axis point (6000, 0) is feasible but worse.
        assert 0.09 * 6000 < 630

    def test_warehouse_shipping(self):
        # min 2.5 n + 4.0 s
        # n+s >= 16, n + 0.5 s <= 12, n,s >= 0
        best = min(
            2.5 * n + 4.0 * s
            for n in range(0, 17)
            for s in range(0, 17)
            if n + s >= 16 and n + 0.5 * s <= 12
        )
        assert best == 52
        assert abs(2.5 * 8 + 4.0 * 8 - 52) < 1e-9
        # All from warehouse 1 is infeasible on dock time: 16 > 12.
        assert 16 > 12
        v = _vignette("warehouse shipping")
        assert v.true_solution == {"warehouse_1": 8, "warehouse_2": 8}
        assert v.true_objective == "52"

    def test_gift_baskets(self):
        best = max(
            7.25 * x + 4.1 * y
            for x in range(0, 21)
            for y in [20 - x]
            if y >= 0 and 3 * x + 1 * y <= 61
        )
        assert best == 145
        # Without non-negativity / integrality: jam binds at 3d + s = 61 with
        # d + s = 20 → (d, s) = (20.5, -0.5), profit 146.575.
        assert abs(7.25 * 20.5 + 4.1 * (-0.5) - 146.575) < 1e-9
        v = _vignette("gift baskets")
        assert v.failure_mode == "both"
        assert v.true_solution == {"deluxe": 20, "standard": 0}
        assert v.true_objective == "145"
        assert v.naive_objective == "146.575"

    def test_workshop_vehicles_control(self):
        best = max(
            42.5 * x + 53.75 * y
            for x in range(0, 6)
            for y in range(0, 6)
            if 2.4 * x + 0.8 * y <= 12
            and 0.75 * x + 1.5 * y <= 7.5
        )
        assert best == 331.25
        v = _vignette("workshop vehicles")
        assert v.failure_mode == "none"
        assert v.true_objective == v.naive_objective == "331.25"


class TestVignetteSet:
    def test_failure_mode_coverage(self):
        modes = [v.failure_mode for v in load_lp_vignettes()]
        assert modes.count("integrality") == 4
        assert modes.count("nonnegativity") == 2
        assert modes.count("both") == 1
        assert modes.count("none") == 1


class TestBuildAll:
    def test_prompt_count_with_explicit_parallels(self):
        prompts, items, benchmark = build_all()
        # 7 trap vignettes x 2 conditions + 1 control = 15
        assert len(prompts) == 15
        assert len(items) == 15
        assert len(benchmark) == 15
        assert {row["variant"] for row in benchmark} == {"json"}
        assert all(row["response_type"] == "json" for row in benchmark)
        conditions = {row["condition"] for row in items}
        assert conditions == {"implicit", "explicit", "control"}
        assert sum(1 for row in items if row["condition"] == "implicit") == 7
        assert sum(1 for row in items if row["condition"] == "explicit") == 7
        assert sum(1 for row in items if row["condition"] == "control") == 1

    def test_json_prompt_shape(self):
        prompts, items, _ = build_all()
        row = _item(items, "print_shop__integrality__json")
        prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == row["example_id"]
        )
        assert '"solution"' in prompt
        assert '"cost"' in prompt
        assert "posters" in prompt
        assert "booklets" in prompt
        assert "Output the JSON object first on its own line." in prompt
        assert "any comments, qualifications, etc." in prompt
        assert "whole numbers" not in prompt
        assert row["true_solution"] == '{"booklets": 4, "posters": 0}'
        assert row["implicit_integer"] == "true"
        assert row["implicit_nonnegative"] == "false"
        assert row["condition"] == "implicit"
        assert "data_audit" not in prompt
        assert "Reply with only the letter" not in prompt

    def test_explicit_parallel_spells_out_constraints(self):
        prompts, items, _ = build_all()
        implicit = _item(items, "print_shop__integrality__json")
        explicit = _item(items, "print_shop__integrality__explicit__json")
        assert implicit["condition"] == "implicit"
        assert explicit["condition"] == "explicit"
        assert implicit["implicit_integer"] == explicit["implicit_integer"] == "true"
        assert (
            implicit["implicit_nonnegative"]
            == explicit["implicit_nonnegative"]
            == "false"
        )
        assert implicit["true_objective"] == explicit["true_objective"]
        imp_prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == implicit["example_id"]
        )
        exp_prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == explicit["example_id"]
        )
        assert "whole numbers" not in imp_prompt
        assert "whole numbers" in exp_prompt
        assert "no fractional print jobs" in exp_prompt

    def test_constraint_type_flags(self):
        _, items, _ = build_all()
        by_id = {row["example_id"]: row for row in items}
        assert by_id["fund_allocation__nonnegativity__json"]["implicit_integer"] == "false"
        assert (
            by_id["fund_allocation__nonnegativity__json"]["implicit_nonnegative"]
            == "true"
        )
        assert by_id["gift_baskets__both__json"]["implicit_integer"] == "true"
        assert by_id["gift_baskets__both__json"]["implicit_nonnegative"] == "true"
        assert by_id["workshop_vehicles__none__json"]["implicit_integer"] == "false"
        assert by_id["workshop_vehicles__none__json"]["implicit_nonnegative"] == "false"
        assert "workshop_vehicles__none__explicit__json" not in by_id

    def test_write_csvs(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("scripts.build_lp_prompts.OUT_DIR", tmp_path / "lp")
        count = write_csvs()
        assert count == 15
        with (tmp_path / "lp" / "benchmark.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 15
        assert "true_solution" in rows[0]
        assert "solution_keys" in rows[0]
        assert "implicit_integer" in rows[0]
        assert "implicit_nonnegative" in rows[0]


class TestScoring:
    def test_parse_lp_json(self):
        parsed = lp_rate.parse_lp_json(
            '{"solution": {"bookcases": 0, "desks": 4}, "cost": 200}'
        )
        assert parsed.parseable
        assert parsed.cost == 200.0
        assert parsed.solution == {"bookcases": 0, "desks": 4}

        fenced = lp_rate.parse_lp_json(
            'Here you go:\n```json\n{"solution": {"a": 1}, "cost": 16.8}\n```\n'
        )
        assert fenced.cost == 16.8

        with_comments = lp_rate.parse_lp_json(
            '{"solution": {"bookcases": 0, "desks": 4}, "cost": 200}\n'
            "Integer optimum; continuous would be 210."
        )
        assert with_comments.parseable
        assert with_comments.cost == 200.0
        assert with_comments.solution == {"bookcases": 0, "desks": 4}

        assert lp_rate.parse_lp_json("200").parseable is False

    def test_within_one_percent(self):
        assert lp_rate.within_relative_tolerance(202, 200)
        assert not lp_rate.within_relative_tolerance(210, 200)

    def test_correct_and_naive_json_answers(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr("scripts.build_lp_prompts.OUT_DIR", tmp_path / "lp")
        write_csvs()
        items = lp_rate.load_benchmark(tmp_path / "lp" / "benchmark.csv")
        assert len(items) == 15

        correct_rows = []
        for example_id, item in items.items():
            payload = {
                "solution": item.true_solution,
                "cost": item.true_objective,
            }
            correct_rows.append(
                {"example_id": example_id, "response": json.dumps(payload)}
            )
        score = lp_rate.score_run_rows(correct_rows, items=items)
        assert score.accuracy == 1.0
        assert lp_rate.naive_confusion_rate(correct_rows, items=items) == 0.0

        naive_rows = []
        for example_id, item in items.items():
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
        # Trap vignettes with a numeric naive optimum distinct from true:
        # carpenter, print shop, pottery, charter, gift (x2 conditions).
        # Fund/warehouse have matching naive/true under redesigned constraints.
        assert len(naive_rows) == 10
        assert lp_rate.score_run_rows(naive_rows, items=items).accuracy == 0.0
        assert lp_rate.naive_confusion_rate(naive_rows, items=items) == 1.0

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
