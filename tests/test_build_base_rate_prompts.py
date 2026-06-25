"""Tests for base-rate vignette prompt builder."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_base_rate_prompts import (
    META_F,
    OUT_DIR,
    VARIANTS,
    _load_overlap,
    _load_two_cause,
    build_all,
    slug,
    write_csvs,
)


class TestVignetteLoad:
    def test_counts(self):
        assert len(_load_two_cause()) == 5
        assert len(_load_overlap()) == 4

    def test_well_posed_p_cd_zero(self):
        for v in _load_two_cause():
            assert v.p_cd == 0.0

    def test_overlap_p_cd_estimates(self):
        by_name = {v.name: v for v in _load_overlap()}
        assert abs(by_name["diabetes insulin obese"].p_cd - 0.188) < 1e-9
        assert abs(by_name["college STEM work"].p_cd - 0.09) < 1e-9
        assert abs(by_name["actor waiter overlap"].p_cd - 0.018) < 1e-9

    def test_intersection_size_labels(self):
        for v in _load_two_cause():
            assert v.intersection_size == "0"
        by_name = {v.name: v for v in _load_overlap()}
        assert by_name["diabetes insulin obese"].intersection_size == "large"
        assert by_name["college STEM work"].intersection_size == "medium"
        assert by_name["actor waiter overlap"].intersection_size == "small"
        assert by_name["professional drivers speeding"].intersection_size == "small"

    def test_ca_trump_posterior(self):
        v = next(v for v in _load_two_cause() if v.name == "CA Trump voter")
        assert abs(v.posterior_a() - 0.10) < 0.015

    def test_diabetes_overlap_posterior_below_partition(self):
        v = next(v for v in _load_overlap() if v.name == "diabetes insulin obese")
        assert v.posterior_a() < v.posterior_partition()
        assert abs(v.posterior_partition() - 0.63) < 0.02


class TestBuildAll:
    def test_variant_count(self):
        prompts, items, benchmark = build_all()
        vignette_count = len(_load_two_cause()) + len(_load_overlap())
        assert len(prompts) == vignette_count * len(VARIANTS)
        assert len(items) == len(prompts)
        assert len(benchmark) == len(prompts)

    def test_all_variants_present(self):
        _, items, _ = build_all()
        by_slug: dict[str, set[str]] = {}
        for row in items:
            by_slug.setdefault(row["vignette_name"], set()).add(row["variant"])
        for name, variants in by_slug.items():
            assert variants == set(VARIANTS), name

    def test_ca_trump_posterior(self):
        v = next(v for v in _load_two_cause() if v.name == "CA Trump voter")
        assert abs(v.posterior_a() - 0.10) < 0.015

    def test_all_mc_full_normative_is_numeric_letter(self):
        _, items, _ = build_all()
        full = [r for r in items if r["variant"] == "mc_full_probs"]
        assert len(full) == 9
        for row in full:
            assert row["normative_choice"] in "ABCDE"
            assert row["normative_open"] != META_F

    def test_probs_include_consultant_intro(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        discharged = pmap["discharged_weapon_last_year__open_probs"]
        assert "statistical consultant" in discharged
        assert "44%" in discharged
        assert "Among employed police officers" in discharged
        assert "In Employed police" not in discharged
        assert "group A" not in discharged
        assert "P(A)" not in discharged

    def test_no_probs_uses_estimate_question(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        no_probs = pmap["discharged_weapon_last_year__open_no_probs"]
        assert "Using knowledge of the world, please estimate the probability" in no_probs
        assert "a police officer or a security guard" in no_probs
        assert "discharged a weapon in the last year" in no_probs
        assert "No numerical probabilities" not in no_probs

    def test_probs_vs_no_probs(self):
        prompts, items, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        ids = {r["example_id"]: r for r in items}
        with_probs = ids[f"{slug('CA Trump voter')}__open_probs"]
        no_probs = ids[f"{slug('CA Trump voter')}__open_no_probs"]
        assert "P(A)" not in pmap[with_probs["example_id"]]
        assert "statistical consultant" in pmap[with_probs["example_id"]]
        assert "Using knowledge of the world, please estimate the probability" in pmap[no_probs["example_id"]]

    def test_ca_trump_geo_phrasing(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        ca = pmap[f"{slug('CA Trump voter')}__open_no_probs"]
        assert "registered in southern California" in ca
        assert "other parts of the state" in ca
        assert "other California registrant" not in ca.lower()

    def test_covid_vaccine_short_phrasing(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        covid = pmap[f"{slug('covid vaccine (blue/red)')}__open_no_probs"]
        assert "some received the 2024-25 COVID vaccine and the rest did not" in covid
        assert "people who received" not in covid
        assert "Among the vaccinated" in covid

    def test_professional_driver_short_phrasing(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        pro = pmap["professional_drivers_speeding__overlap__open_probs"]
        assert "a professional driver or other adult" in pro
        assert "primary occupation is not professional driving" not in pro

    def test_prose_quality_fixes(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        actor = pmap["actor_waiter_overlap__overlap__open_probs"]
        assert "some are actors and the rest are non-actors" in pmap[
            "actor_waiter_overlap__overlap__open_no_probs"
        ]
        assert "also work as waiters" in actor
        assert "people who hold" not in actor

        health = pmap[f"{slug('healthcare employment')}__open_probs"]
        assert "other employed adults" in health
        assert "works in a hospital" in health
        assert "Given that someone works in a hospital" in health
        assert "health care professional or other employed adult" not in health
        assert "not a health care professionals" not in health

        military = pmap[f"{slug('military overseas (federal pool)')}__open_probs"]
        assert "active-duty service member or a federal civilian" in military
        assert "has worked overseas" in military
        assert "had has worked" not in military
        assert "uS Army" not in military

        college = pmap["college_stem_work__overlap__open_probs"]
        assert "major in STEM" in college
        assert "STEM majors" in college
        assert "studys" not in college
        assert "were retained to year 2" not in college
        assert "returned for a second year" in college
        assert "among other students" in college
        assert "among continuing-generation students" not in college
        assert "Among undergraduate students" in college
        assert "In Undergraduate" not in college
        college_no = pmap["college_stem_work__overlap__open_no_probs"]
        college_no_body = college_no.split("\n\nUsing knowledge")[0]
        assert "some are first-generation students." in college_no_body
        assert "the rest are continuing-generation" not in college_no_body

        ca = pmap[f"{slug('CA Trump voter')}__open_probs"]
        assert "registered in California" in ca
        assert "Among US registered voters" in ca
        assert "In US registered voters" not in ca
        assert "registered elsewhere" in ca
        assert "home state is California" not in ca

        diabetes = pmap["diabetes_insulin_obese__overlap__open_probs"]
        assert "have diagnosed diabetes" in diabetes
        assert "those with diagnosed diabetes" in diabetes
        assert "adults with diagnosed diabetes and the remainder are adults without" not in diabetes

    def test_overlap_mentioned_in_prose(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        dia = pmap["diabetes_insulin_obese__overlap__open_probs"]
        assert "fall into both categories" in dia
        ca = pmap[f"{slug('CA Trump voter')}__open_probs"]
        assert "fall into both categories" not in ca

    def test_confidence_instruction(self):
        prompts, _, _ = build_all()
        for row in prompts:
            assert "confidence" in row["prompt"].lower()

    def test_mc_numeric_labels_are_unique(self):
        _, items, _ = build_all()
        for row in items:
            if not row["variant"].startswith("mc_"):
                continue
            labels = [row[f"option_{c}_label"] for c in "abcde"]
            assert len(labels) == len(set(labels)), row["example_id"]

    def test_numeric_mc_has_five_options_only(self):
        _, items, _ = build_all()
        for row in items:
            if not row["variant"].startswith("mc_numeric"):
                continue
            assert row["option_a_label"]
            assert row["option_e_label"]
            assert not row.get("option_f_label")

    def test_full_mc_has_meta_options(self):
        _, items, _ = build_all()
        full = [r for r in items if r["variant"] == "mc_full_probs"]
        assert len(full) == 9
        for row in full:
            assert row["option_f_label"] == META_F
            assert row["option_g_label"]
            assert row["option_h_label"]


class TestBenchmarkCsv:
    def test_condition_columns(self):
        _, _, benchmark = build_all()
        assert len(benchmark) == 54
        for row in benchmark:
            assert row["response_type"] in {"open", "mc_numeric", "mc_full"}
            assert row["has_statistics"] in {"true", "false"}
            assert row["problem_type"] in {"well_posed", "overlap"}
            assert row["intersection_size"] in {"0", "small", "medium", "large"}
            assert row["prompt"]
            assert row["variant"].endswith("_probs" if row["has_statistics"] == "true" else "_no_probs")

    def test_intersection_size_by_vignette(self):
        _, _, benchmark = build_all()
        by_key = {
            (row["vignette_name"], row["problem_type"]): row["intersection_size"]
            for row in benchmark
        }
        assert len(by_key) == 9
        assert by_key[("discharged weapon (last year)", "well_posed")] == "0"
        assert by_key[("diabetes insulin obese", "overlap")] == "large"
        assert by_key[("college STEM work", "overlap")] == "medium"
        assert by_key[("actor waiter overlap", "overlap")] == "small"
        assert by_key[("professional drivers speeding", "overlap")] == "small"

    def test_factorial_uniqueness(self):
        _, _, benchmark = build_all()
        keys = {
            (
                row["vignette_name"],
                row["problem_type"],
                row["response_type"],
                row["has_statistics"],
            )
            for row in benchmark
        }
        assert len(keys) == 54


class TestWrittenCsvs:
    def test_files_exist_after_build(self):
        write_csvs()
        assert (OUT_DIR / "prompts.csv").is_file()
        assert (OUT_DIR / "items.csv").is_file()
        assert (OUT_DIR / "benchmark.csv").is_file()

        with (OUT_DIR / "prompts.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 54

        with (OUT_DIR / "benchmark.csv").open(encoding="utf-8") as handle:
            bench = list(csv.DictReader(handle))
        assert len(bench) == 54
        assert "intersection_size" in bench[0]
        assert "problem_type" in bench[0]
        assert "response_type" in bench[0]
        assert "has_statistics" in bench[0]


def _prompt_for(example_id: str) -> str:
    with (OUT_DIR / "prompts.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["example_id"] == example_id:
                return row["prompt"]
    raise KeyError(example_id)
