"""Tests for simple two-path prompt builder."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_simple_rate_prompts import (
    _from_vignette,
    build_all,
    load_simple_vignettes,
    write_csvs,
)
from scripts.build_base_rate_prompts import _load_two_cause


class TestSimpleParameters:
    def test_discharged_weapon_p_c_and_posterior(self):
        v = next(x for x in _load_two_cause() if x.name == "discharged weapon (last year)")
        s = _from_vignette(v)
        assert abs(s.p_c - 0.44 * 0.68) < 1e-9
        assert abs(s.p_d - 0.44 * 0.30) < 1e-9
        assert abs(s.s_c - 0.003) < 1e-9
        assert abs(s.s_d - 0.002) < 1e-9
        assert abs(s.posterior_c() - 0.7726) < 0.01

    def test_vignette_count(self):
        assert len(load_simple_vignettes()) == 9


class TestBuildAll:
    def test_prompt_count(self):
        prompts, items, benchmark = build_all()
        assert len(prompts) == 45
        assert len(items) == len(prompts)
        assert len(benchmark) == len(prompts)
        problem_types = {row["problem_type"] for row in benchmark}
        assert problem_types == {
            "well_posed",
            "implausible_c_d",
            "implausible_t",
        }

    def test_english_teacher_humanities_entity_labels(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        english = next(
            r for r in rows if r["example_id"] == "english_teacher_humanities__open_probs"
        )
        assert (
            "0.11% teach English as their primary assignment and "
            "0.13% have a bachelor's degree in English."
            in english["prompt"]
        )
        assert (
            "Among those who teach English as their primary assignment, 69% hold a master's degree or higher"
            in english["prompt"]
        )
        assert (
            "among those who have a bachelor's degree in English, 55% hold a master's degree or higher"
            in english["prompt"]
        )
        assert (
            "What is the probability that a high school teacher who holds a master's degree or higher is an English teacher?"
            in english["prompt"]
        )
        assert "public grades 9-12" not in english["prompt"]

    def test_ca_trump_voter_entity_labels(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        ca = next(r for r in rows if r["example_id"] == "ca_trump_voter__open_probs")
        assert (
            "7.8% are voters registered in Southern California and 4.9% are other California voters."
            in ca["prompt"]
        )
        assert (
            "Among voters registered in Southern California, 27% voted for Donald Trump"
            in ca["prompt"]
        )
        assert "among other California voters, 31% voted for Donald Trump" in ca["prompt"]
        assert (
            "What is the probability that a registered voter in California who voted for Donald Trump in the 2024 presidential election lives in Southern California?"
            in ca["prompt"]
        )

    def test_college_stem_work_entity_labels(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        college = next(
            r for r in rows if r["example_id"] == "college_stem_work__open_probs"
        )
        assert "NPSAS:20 universe" not in college["prompt"]
        assert (
            "4.6% study STEM and 17% are employed while enrolled."
            in college["prompt"]
        )
        assert "Among those who studied STEM, 85% returned for a second year" in college["prompt"]
        assert "among those employed while enrolled, 74% returned for a second year" in college["prompt"]
        assert (
            "What is the probability that a student who returned for a second year studied STEM?"
            in college["prompt"]
        )
        assert "first-generation" not in college["prompt"]

    def test_diabetes_insulin_obese_entity_labels(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        diabetes = next(
            r for r in rows if r["example_id"] == "diabetes_insulin_obese__open_probs"
        )
        assert (
            "3.2% use insulin and 5.3% are obese."
            in diabetes["prompt"]
        )
        assert "Among those who use insulin," in diabetes["prompt"]
        assert "among those who are obese," in diabetes["prompt"]
        assert (
            "What is the probability that an adult with diagnosed diabetes who had hemoglobin A1c above 9.0% uses insulin?"
            in diabetes["prompt"]
        )

    def test_healthcare_employment_entity_labels(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        health = next(
            r for r in rows if r["example_id"] == "healthcare_employment__open_probs"
        )
        assert "physician (MD/DO)" not in health["prompt"]
        assert "non-physician health care professional among A" not in health["prompt"]
        assert "1.1% are physicians and 9.9% are health care professionals who are not physicians" in health["prompt"]
        assert (
            "What is the probability that a health care professional who works in a hospital is a physician?"
            in health["prompt"]
        )

    def test_open_prompt_mentions_combined_entities(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        discharged = next(
            r for r in rows if r["example_id"] == "discharged_weapon_last_year__open_probs"
        )
        assert "urban police officers" in discharged["prompt"].lower()
        assert "rural/small-jurisdiction police officers" in discharged["prompt"].lower()
        assert "P(C|T)" not in discharged["prompt"]

    def test_items_include_source_probabilities(self):
        _, items, _ = build_all()
        row = next(r for r in items if r["example_id"] == "ca_trump_voter__mc_numeric_probs")
        assert float(row["p_c"]) == 0.13 * 0.38
        assert float(row["p_d"]) == 0.13 * 0.60
        assert float(row["p_t_given_c"]) == 0.31
        assert float(row["p_t_given_d"]) == 0.27

    def test_intersection_size_by_vignette(self):
        _, _, benchmark = build_all()
        by_name = {row["vignette_name"]: row["intersection_size"] for row in benchmark}
        assert by_name["discharged weapon (last year)"] == "0"
        assert by_name["diabetes insulin obese"] == "large"
        assert by_name["college STEM work"] == "medium"
        assert by_name["professional drivers speeding"] == "small"
        assert by_name["english teacher humanities"] == "large"

    def test_mc_full_probs_includes_meta_options(self):
        prompts, items, _ = build_all()
        row = next(r for r in items if r["example_id"] == "ca_trump_voter__mc_full_probs")
        prompt = next(r["prompt"] for r in prompts if r["example_id"] == row["example_id"])
        assert row["variant"] == "mc_full_probs"
        assert row["response_type"] == "mc_full"
        assert row["scepticism_required"] == "false"
        assert row["option_f_label"] == "Insufficient information"
        assert row["option_h_label"] == "Provided information is obviously incorrect"
        assert "F, G, or H" in prompt

    def test_implausible_c_d_uses_csv_stats(self):
        _, items, benchmark = build_all()
        row = next(
            r
            for r in items
            if r["example_id"] == "ca_trump_voter__implausible_c_d__mc_full_probs"
        )
        assert row["problem_type"] == "implausible_c_d"
        assert float(row["p_c"]) == 0.0494
        assert float(row["p_d"]) == 0.98
        assert row["scepticism_required"] == "true"
        assert row["scepticism_score_target"] == "H"
        assert row["normative"] == "implausible"

    def test_implausible_t_uses_csv_stats(self):
        _, items, _ = build_all()
        row = next(
            r
            for r in items
            if r["example_id"] == "healthcare_employment__implausible_t__mc_full_probs"
        )
        assert row["problem_type"] == "implausible_t"
        assert float(row["p_t_given_c"]) == 0.1
        assert float(row["p_t_given_d"]) == 0.7
        assert row["scepticism_score_target"] == "H"
