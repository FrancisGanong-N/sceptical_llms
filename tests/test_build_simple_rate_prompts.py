"""Tests for simple two-path prompt builder."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_simple_rate_prompts import (
    META_SCEPTICISM,
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
        names = {v.name for v in load_simple_vignettes()}
        assert len(names) == 12
        assert "NAEP grade 4 reading (MA vs NM)" in names
        assert "HS graduation ACGR (WV vs AZ)" in names
        assert "NFL MLB watch attend" in names
        assert "fantasy sports (male vs female)" in names
        assert "fantasy sports (under 45 vs male)" in names
        assert "actor waiter overlap" not in names
        assert "college STEM work" not in names
        assert "professional drivers speeding" not in names

    def test_naep_ma_nm_posterior(self):
        v = next(
            x
            for x in _load_two_cause()
            if x.name == "NAEP grade 4 reading (MA vs NM)"
        )
        s = _from_vignette(v)
        assert abs(s.p_a - 1.0) < 1e-9
        assert abs(s.p_c - 0.748) < 0.001
        assert abs(s.p_d - 0.252) < 0.001
        assert abs(s.posterior_c() - 0.856) < 0.01

    def test_graduation_wv_az_posterior(self):
        v = next(
            x
            for x in _load_two_cause()
            if x.name == "HS graduation ACGR (WV vs AZ)"
        )
        s = _from_vignette(v)
        assert abs(s.p_a - 1.0) < 1e-9
        assert abs(s.p_c - 0.155) < 0.001
        assert abs(s.p_d - 0.845) < 0.001
        assert abs(s.posterior_c() - 0.180) < 0.01

    def test_graduation_wv_az_prompt_mentions_enrollment(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        grad = next(
            r
            for r in rows
            if r["example_id"] == "hs_graduation_acgr_wv_vs_az__open_probs"
        )
        assert "17,489 enrolled in West Virginia and 95,122 in Arizona" in grad["prompt"]
        assert (
            "An on-time graduate is a student who earns a regular high school diploma"
            in grad["prompt"]
        )
        assert (
            "a twelfth grader in West Virginia or Arizona who is an on-time graduate "
            "is from West Virginia"
            in grad["prompt"]
        )

    def test_naep_ma_nm_prompt_mentions_enrollment(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        naep = next(
            r
            for r in rows
            if r["example_id"] == "naep_grade_4_reading_ma_vs_nm__open_probs"
        )
        assert "66,751 enrolled in Massachusetts and 22,503 in New Mexico" in naep["prompt"]
        assert (
            "A proficient reader is a student who scores at or above NAEP Proficient"
            in naep["prompt"]
        )
        assert (
            "a fourth grader in Massachusetts or New Mexico who is a proficient reader "
            "is from Massachusetts"
            in naep["prompt"]
        )


class TestBuildAll:
    def test_prompt_count(self):
        prompts, items, benchmark = build_all()
        assert len(prompts) == 60
        assert len(items) == len(prompts)
        assert len(benchmark) == len(prompts)
        problem_types = {row["problem_type"] for row in benchmark}
        assert problem_types == {
            "well_posed",
            "implausible_c_d",
            "implausible_t",
        }
        overlap_well_posed = [
            row
            for row in benchmark
            if row["problem_type"] == "well_posed"
            and row["intersection_size"] not in ("", "0")
        ]
        assert len(overlap_well_posed) == 12
        assert all(row["scepticism_required"] == "false" for row in overlap_well_posed)
        assert {row["variant"] for row in overlap_well_posed} == {
            "open_probs",
            "mc_numeric_probs",
            "mc_full_probs",
        }

    def test_fantasy_sports_male_female_partition(self, tmp_path: Path, monkeypatch):
        v = next(
            x
            for x in _load_two_cause()
            if x.name == "fantasy sports (male vs female)"
        )
        s = _from_vignette(v)
        assert abs(s.p_c - 0.49) < 0.001
        assert abs(s.p_d - 0.51) < 0.001
        assert abs(s.posterior_c() - 0.70) < 0.01

        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        fantasy = next(
            r
            for r in rows
            if r["example_id"] == "fantasy_sports_male_vs_female__open_probs"
        )
        assert "49% are men and 51% are women" in fantasy["prompt"]
        assert "Among men, 34% played fantasy sports for money" in fantasy["prompt"]
        assert "among women, 14% played fantasy sports for money" in fantasy["prompt"]
        assert (
            "an adult who played fantasy sports for money is a man"
            in fantasy["prompt"]
        )

    def test_fantasy_sports_under_45_male_overlap(self, tmp_path: Path, monkeypatch):
        v = next(
            x for x in load_simple_vignettes() if x.name == "fantasy sports (under 45 vs male)"
        )
        assert abs(v.p_c - 0.45) < 0.001
        assert abs(v.p_d - 0.49) < 0.001
        assert abs(v.p_cd_given_a - 0.22) < 0.001
        assert abs(v.target_posterior() - 0.66) < 0.02
        assert abs(v.posterior_c() - 0.53) < 0.02

        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        fantasy = next(
            r
            for r in rows
            if r["example_id"] == "fantasy_sports_under_45_vs_male__open_probs"
        )
        assert "45% are under age 45 and 49% are men" in fantasy["prompt"]
        assert "An estimated 22% fall into both categories." in fantasy["prompt"]
        assert "Among those under age 45, 41% played fantasy sports for money" in fantasy["prompt"]
        assert "among men, 34% played fantasy sports for money" in fantasy["prompt"]
        assert (
            "an adult who played fantasy sports for money is under age 45"
            in fantasy["prompt"]
        )

    def test_nfl_mlb_watch_attend_overlap(self, tmp_path: Path, monkeypatch):
        v = next(x for x in load_simple_vignettes() if x.name == "NFL MLB watch attend")
        assert abs(v.p_c - 0.56) < 0.001
        assert abs(v.p_d - 0.33024) < 0.001
        assert abs(v.p_cd_given_a - 0.391) < 0.001
        assert abs(v.target_posterior() - 0.83) < 0.02
        assert abs(v.posterior_c() - 0.51) < 0.02

        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        sports = next(
            r for r in rows if r["example_id"] == "nfl_mlb_watch_attend__open_probs"
        )
        assert "56% watch NFL and 33% watch MLB" in sports["prompt"]
        assert "An estimated 25% fall into both categories." in sports["prompt"]
        assert "Among those who watch NFL, 13% attended an NFL or MLB game in person" in sports["prompt"]
        assert "among those who watch MLB, 21% attended an NFL or MLB game in person" in sports["prompt"]
        assert (
            "an adult who watched an NFL or MLB game in the past year and "
            "attended an NFL or MLB game in person watched an NFL game"
            in sports["prompt"]
        )

    def test_english_teacher_humanities_entity_labels(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        english = next(
            r for r in rows if r["example_id"] == "english_teacher_humanities__mc_full_probs"
        )
        assert (
            "0.11% teach English as their primary assignment and "
            "0.13% have a bachelor's degree in English."
            in english["prompt"]
        )
        assert "An estimated 0.09% fall into both categories." in english["prompt"]
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

    def test_diabetes_insulin_obese_entity_labels(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_simple_rate_prompts.OUT_DIR",
            tmp_path / "simple",
        )
        write_csvs()
        with (tmp_path / "simple" / "prompts.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        diabetes = next(
            r for r in rows if r["example_id"] == "diabetes_insulin_obese__mc_full_probs"
        )
        assert (
            "3.2% use insulin and 5.3% are obese."
            in diabetes["prompt"]
        )
        assert "An estimated 2.1% fall into both categories." in diabetes["prompt"]
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
        assert by_name["english teacher humanities"] == "large"
        assert by_name["NFL MLB watch attend"] == "large"
        assert by_name["fantasy sports (under 45 vs male)"] == "large"
        assert by_name["fantasy sports (male vs female)"] == "0"
        assert "college STEM work" not in by_name
        assert "professional drivers speeding" not in by_name
        assert "actor waiter overlap" not in by_name

    def test_overlap_scores_overlap_aware_normative(self):
        _, items, _ = build_all()
        row = next(
            r for r in items if r["example_id"] == "diabetes_insulin_obese__mc_full_probs"
        )
        assert row["scepticism_required"] == "false"
        assert row["scepticism_score_target"] != "F"
        assert row["normative"] == "underdetermined"
        assert row["well_posed"] == "false"
        assert float(row["p_c_and_d_given_a"]) > 0

    def test_mc_full_probs_single_meta_option(self):
        prompts, items, _ = build_all()
        row = next(r for r in items if r["example_id"] == "ca_trump_voter__mc_full_probs")
        prompt = next(r["prompt"] for r in prompts if r["example_id"] == row["example_id"])
        assert row["variant"] == "mc_full_probs"
        assert row["response_type"] == "mc_full"
        assert row["scepticism_required"] == "false"
        assert row["option_f_label"] == META_SCEPTICISM
        assert row["option_g_label"] == ""
        assert row["option_h_label"] == ""
        assert "or F" in prompt
        assert "G" not in prompt.split("Which answer is closest?")[-1]
        assert "Line 3 (optional)" in prompt

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
        assert row["scepticism_score_target"] == "F"
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
        assert row["scepticism_score_target"] == "F"
