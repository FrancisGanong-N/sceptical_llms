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
        assert abs(0.084 * 10000 - 840) < 1e-9
        assert abs(0.084 * 12000 + 0.032 * -2000 - 944) < 1e-9

    def test_warehouse_shipping(self):
        best = min(
            3.25 * x + 2.1 * y
            for x in range(0, 9)
            for y in range(0, 9)
            if x + y >= 8
        )
        assert best == 16.8
        assert _vignette("warehouse shipping").naive_objective == "unbounded"

    def test_gift_baskets(self):
        best = max(
            7.25 * x + 4.1 * y
            for x in range(0, 21)
            for y in [20 - x]
            if y >= 0 and 2.5 * x + 0.75 * y <= 52.625
        )
        assert best == 145
        assert _vignette("gift baskets").failure_mode == "both"

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
        assert modes.count("integrality") == 2
        assert modes.count("nonnegativity") == 2
        assert modes.count("both") == 1
        assert modes.count("none") == 1


class TestBuildAll:
    def test_prompt_count_no_audits(self):
        prompts, items, benchmark = build_all()
        assert len(prompts) == 6
        assert len(items) == 6
        assert len(benchmark) == 6
        assert {row["variant"] for row in benchmark} == {"json"}
        assert all(row["response_type"] == "json" for row in benchmark)

    def test_json_prompt_shape(self):
        prompts, items, _ = build_all()
        row = _item(items, "carpenter_furniture__integrality__json")
        prompt = next(
            r["prompt"] for r in prompts if r["example_id"] == row["example_id"]
        )
        assert '"solution"' in prompt
        assert '"cost"' in prompt
        assert "bookcases" in prompt
        assert "desks" in prompt
        assert "Output the JSON object first on its own line." in prompt
        assert "any comments, qualifications, etc." in prompt
        assert row["true_solution"] == '{"bookcases": 0, "desks": 4}'
        assert "data_audit" not in prompt
        assert "Reply with only the letter" not in prompt

    def test_write_csvs(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("scripts.build_lp_prompts.OUT_DIR", tmp_path / "lp")
        count = write_csvs()
        assert count == 6
        with (tmp_path / "lp" / "benchmark.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 6
        assert "true_solution" in rows[0]
        assert "solution_keys" in rows[0]


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
        assert len(items) == 6

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
        assert len(naive_rows) == 4
        assert lp_rate.score_run_rows(naive_rows, items=items).accuracy == 0.0
        assert lp_rate.naive_confusion_rate(naive_rows, items=items) == 1.0

    def test_near_miss_within_tolerance(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("scripts.build_lp_prompts.OUT_DIR", tmp_path / "lp")
        write_csvs()
        items = lp_rate.load_benchmark(tmp_path / "lp" / "benchmark.csv")
        example_id = "carpenter_furniture__integrality__json"
        near = [
            {
                "example_id": example_id,
                "response": '{"solution": {"bookcases": 0, "desks": 4}, "cost": 201.5}',
            }
        ]
        far = [
            {
                "example_id": example_id,
                "response": '{"solution": {"bookcases": 2.4, "desks": 2.4}, "cost": 210}',
            }
        ]
        assert lp_rate.score_run_rows(near, items=items).accuracy == 1.0
        assert lp_rate.score_run_rows(far, items=items).accuracy == 0.0
