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
        assert len(load_simple_vignettes()) == 10


class TestBuildAll:
    def test_prompt_count(self):
        prompts, items, benchmark = build_all()
        assert len(prompts) == 20
        assert len(items) == len(prompts)
        assert len(benchmark) == len(prompts)

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
            "what is the probability they teach English as their primary assignment?"
            in english["prompt"]
        )
        assert "Given that a high school teacher holds a master's degree or higher" in english["prompt"]
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
        assert "what is the probability they were other California voters?" in ca["prompt"]

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
        assert "Given that a student returned for a second year" in college["prompt"]
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
        assert "what is the probability they use insulin?" in diabetes["prompt"]

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
            "what is the probability they were physicians?"
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
