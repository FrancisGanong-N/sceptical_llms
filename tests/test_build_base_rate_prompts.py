"""Tests for base-rate vignette prompt builder."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from scripts.build_base_rate_prompts import (
    META_F,
    OUT_DIR,
    CANONICAL_VARIANTS,
    SCEPTICISM_VARIANTS,
    _load_implausible,
    _load_overlap,
    _load_two_cause,
    build_all,
    load_vignettes,
    scepticism_required,
    slug,
    variants_for_vignette,
    write_csvs,
)


class TestVignetteLoad:
    def test_counts(self):
        assert len(_load_two_cause()) == 5
        assert len(_load_overlap()) == 5

    def test_well_posed_p_cd_zero(self):
        for v in _load_two_cause():
            assert v.p_cd == 0.0

    def test_overlap_p_cd_estimates(self):
        by_name = {v.name: v for v in _load_overlap()}
        assert abs(by_name["diabetes insulin obese"].p_cd - 0.188) < 1e-9
        assert abs(by_name["college STEM work"].p_cd - 0.09) < 1e-9
        assert abs(by_name["actor waiter overlap"].p_cd - 0.018) < 1e-9
        assert abs(by_name["english teacher humanities"].p_cd - 0.114) < 1e-9

    def test_intersection_size_labels(self):
        for v in _load_two_cause():
            assert v.intersection_size == "0"
        by_name = {v.name: v for v in _load_overlap()}
        assert by_name["diabetes insulin obese"].intersection_size == "large"
        assert by_name["college STEM work"].intersection_size == "medium"
        assert by_name["actor waiter overlap"].intersection_size == "small"
        assert by_name["professional drivers speeding"].intersection_size == "small"
        assert by_name["english teacher humanities"].intersection_size == "large"

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
        vignettes = load_vignettes()
        assert len(_load_implausible()) == 10
        assert len(vignettes) == 20
        expected = sum(len(variants_for_vignette(v)) for v in vignettes)
        assert expected == 34
        assert len(prompts) == expected
        assert len(items) == len(prompts)
        assert len(benchmark) == len(prompts)

    def test_variants_by_scepticism(self):
        vignettes = load_vignettes()
        for v in vignettes:
            variants = set(variants_for_vignette(v))
            if scepticism_required(v):
                assert variants == set(SCEPTICISM_VARIANTS), v.name
            else:
                assert variants == set(CANONICAL_VARIANTS), v.name

    def test_all_variants_present(self):
        _, items, _ = build_all()
        by_prefix: dict[str, set[str]] = {}
        for row in items:
            prefix = row["example_id"].rsplit("__", 1)[0]
            by_prefix.setdefault(prefix, set()).add(row["variant"])
        assert len(by_prefix) == 20
        vignettes = {v.example_prefix(): v for v in load_vignettes()}
        for prefix, variants in by_prefix.items():
            expected = set(variants_for_vignette(vignettes[prefix]))
            assert variants == expected, prefix

    def test_ca_trump_posterior(self):
        v = next(v for v in _load_two_cause() if v.name == "CA Trump voter")
        assert abs(v.posterior_a() - 0.10) < 0.015

    def test_all_mc_full_normative_is_numeric_letter(self):
        _, items, _ = build_all()
        full = [r for r in items if r["variant"] == "mc_full_probs"]
        assert len(full) == 20
        for row in full:
            assert row["normative_choice"] in "ABCDE"
            assert row["normative_open"] != META_F

    def test_scepticism_vignettes_have_mc_full_only(self):
        _, items, _ = build_all()
        scepticism_prefixes = {
            v.example_prefix() for v in load_vignettes() if scepticism_required(v)
        }
        assert len(scepticism_prefixes) == 13
        for row in items:
            prefix = row["example_id"].rsplit("__", 1)[0]
            if prefix in scepticism_prefixes:
                assert row["variant"] == "mc_full_probs", row["example_id"]
            else:
                assert row["variant"] in CANONICAL_VARIANTS, row["example_id"]

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

    def test_professional_driver_short_phrasing(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        pro = pmap["professional_drivers_speeding__overlap__open_probs"]
        assert "a professional driver or other adult" in pro
        assert "primary occupation is not professional driving" not in pro

    def test_english_teacher_short_phrasing(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        eng = pmap["english_teacher_humanities__overlap__mc_full_probs"]
        assert "a public grades 9-12 teacher or other employed adult" in eng
        assert "Among those who teach English/language arts" in eng
        assert "Among teach English" not in eng

    def test_ca_trump_geo_phrasing(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        ca = pmap[f"{slug('CA Trump voter')}__open_probs"]
        assert "registered in southern California" in ca
        assert "other parts of the state" in ca
        assert "other California registrant" not in ca.lower()

    def test_covid_vaccine_short_phrasing(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        covid = pmap[f"{slug('covid vaccine (blue/red)')}__open_probs"]
        assert "27% received the 2024-25 COVID vaccine" in covid
        assert "people who received" not in covid
        assert "Among the vaccinated" in covid

    def test_prose_quality_fixes(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        actor = pmap["actor_waiter_overlap__overlap__open_probs"]
        assert "0.03% are actors and the remainder are non-actors" in actor
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

        college = pmap["college_stem_work__overlap__mc_full_probs"]
        assert "major in STEM" in college
        assert "STEM majors" in college
        assert "studys" not in college
        assert "were retained to year 2" not in college
        assert "returned for a second year" in college
        assert "among other students" in college
        assert "among continuing-generation students" not in college
        assert "Among undergraduate students" in college
        assert "In Undergraduate" not in college

        ca = pmap[f"{slug('CA Trump voter')}__open_probs"]
        assert "registered in California" in ca
        assert "Among US registered voters" in ca
        assert "In US registered voters" not in ca
        assert "registered elsewhere" in ca
        assert "home state is California" not in ca

        diabetes = pmap["diabetes_insulin_obese__overlap__mc_full_probs"]
        assert "have diagnosed diabetes" in diabetes
        assert "those with diagnosed diabetes" in diabetes
        assert "adults with diagnosed diabetes and the remainder are adults without" not in diabetes

    def test_overlap_mentioned_in_prose(self):
        prompts, _, _ = build_all()
        pmap = {r["example_id"]: r["prompt"] for r in prompts}
        dia = pmap["diabetes_insulin_obese__overlap__mc_full_probs"]
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
            if not row["variant"].startswith("mc_numeric"):
                continue
            labels = [
                row[f"option_{letter}_label"]
                for letter in "abcde"
                if row.get(f"option_{letter}_label")
            ]
            assert len(labels) == len(set(labels)), row["example_id"]
            assert labels, row["example_id"]
            for label in labels:
                assert re.fullmatch(r"About \d+%", label), (row["example_id"], label)

    def test_mc_numeric_labels_round_to_whole_percents(self):
        _, items, _ = build_all()
        discharged = next(
            row
            for row in items
            if row["example_id"] == "discharged_weapon_last_year__mc_numeric_probs"
        )
        labels = [
            discharged[f"option_{letter}_label"]
            for letter in "abcde"
            if discharged.get(f"option_{letter}_label")
        ]
        assert "About 91%" in labels
        assert not any("." in label for label in labels)

    def test_actor_waiter_mc_numeric_dedupes_to_two_options(self):
        _, items, _ = build_all()
        actor = next(
            row
            for row in items
            if row["example_id"] == "actor_waiter_overlap__overlap__mc_numeric_probs"
        )
        labels = [actor[f"option_{letter}_label"] for letter in "abcde" if actor.get(f"option_{letter}_label")]
        assert labels == ["About 0%", "About 19%"]
        assert not actor["option_c_label"]

    def test_numeric_mc_has_one_to_five_options(self):
        _, items, _ = build_all()
        for row in items:
            if not row["variant"].startswith("mc_numeric"):
                continue
            labels = [
                row[f"option_{letter}_label"]
                for letter in "abcde"
                if row.get(f"option_{letter}_label")
            ]
            assert 1 <= len(labels) <= 5, row["example_id"]
            assert not row.get("option_f_label")

    def test_full_mc_has_meta_options(self):
        _, items, _ = build_all()
        full = [r for r in items if r["variant"] == "mc_full_probs"]
        assert len(full) == 20
        for row in full:
            assert row["option_f_label"] == META_F
            assert row["option_g_label"]
            assert row["option_h_label"]


class TestBenchmarkCsv:
    def test_condition_columns(self):
        _, _, benchmark = build_all()
        assert len(benchmark) == 34
        for row in benchmark:
            assert row["response_type"] in {"open", "mc_numeric", "mc_full"}
            assert row["has_statistics"] == "true"
            assert row["problem_type"] in {"well_posed", "overlap"}
            assert row["intersection_size"] in {"0", "small", "medium", "large"}
            assert row["prompt"]
            assert row["variant"].endswith("_probs")

    def test_intersection_size_by_vignette(self):
        _, _, benchmark = build_all()
        by_key = {
            (row["vignette_name"], row["normative"], row["problem_type"]): row["intersection_size"]
            for row in benchmark
        }
        assert len(by_key) == 20
        assert by_key[("discharged weapon (last year)", "well_posed", "well_posed")] == "0"
        assert by_key[("discharged weapon (last year)", "implausible", "well_posed")] == "0"
        assert by_key[("diabetes insulin obese", "underdetermined", "overlap")] == "large"
        assert by_key[("diabetes insulin obese", "implausible", "overlap")] == "large"

    def test_factorial_uniqueness(self):
        _, _, benchmark = build_all()
        keys = {
            (
                row["vignette_name"],
                row["normative"],
                row["response_type"],
            )
            for row in benchmark
        }
        assert len(keys) == 34


class TestScoringMeasures:
    def test_scepticism_required_by_intersection_size(self):
        _, _, benchmark = build_all()
        by_vignette = {
            row["vignette_name"]: row["scepticism_required"]
            for row in benchmark
            if row["variant"] == "mc_full_probs" and row["normative"] != "implausible"
        }
        assert by_vignette["discharged weapon (last year)"] == "false"
        assert by_vignette["actor waiter overlap"] == "false"
        assert by_vignette["college STEM work"] == "true"
        assert by_vignette["diabetes insulin obese"] == "true"

    def test_numeric_score_is_partition_shortcut(self):
        _, items, _ = build_all()
        discharged = next(
            row
            for row in items
            if row["example_id"] == "discharged_weapon_last_year__mc_numeric_probs"
        )
        assert discharged["numeric_score_percent"]
        choice = discharged["numeric_score_choice"]
        assert choice
        label = discharged[f"option_{choice.lower()}_label"]
        rounded_partition = str(round(float(discharged["numeric_score_percent"])))
        assert rounded_partition in label

    def test_scepticism_score_target_when_required(self):
        _, _, benchmark = build_all()
        by_id = {row["example_id"]: row for row in benchmark}
        diabetes_full = by_id["diabetes_insulin_obese__overlap__mc_full_probs"]
        assert diabetes_full["scepticism_required"] == "true"
        assert diabetes_full["scepticism_score_target"] == "F|G|H"

    def test_scepticism_score_target_when_not_required(self):
        _, _, benchmark = build_all()
        by_id = {row["example_id"]: row for row in benchmark}
        discharged = by_id["discharged_weapon_last_year__mc_full_probs"]
        assert discharged["scepticism_required"] == "false"
        assert discharged["scepticism_score_target"] == discharged["normative_choice"]

        discharged_mc = by_id["discharged_weapon_last_year__mc_numeric_probs"]
        assert discharged_mc["scepticism_score_target"] == "n/a"

        discharged_open = by_id["discharged_weapon_last_year__open_probs"]
        assert discharged_open["scepticism_score_target"] == discharged_open["normative_percent"]

    def test_implausible_vignette_requires_scepticism(self):
        vignettes = _load_two_cause() + _load_overlap()
        assert not scepticism_required(
            next(v for v in vignettes if v.name == "discharged weapon (last year)")
        )
        assert not scepticism_required(
            next(v for v in vignettes if v.name == "actor waiter overlap")
        )
        assert scepticism_required(
            next(v for v in vignettes if v.name == "diabetes insulin obese")
        )
        for v in _load_implausible():
            assert v.normative == "implausible"
            assert scepticism_required(v)


class TestImplausibleVignettes:
    def test_implausible_items_are_mc_full_only(self):
        _, items, _ = build_all()
        implausible = [row for row in items if row["normative"] == "implausible"]
        assert len(implausible) == 10
        assert all(row["variant"] == "mc_full_probs" for row in implausible)
        assert all(row["scepticism_required"] == "true" for row in implausible)
        assert all(row["scepticism_score_target"] == "H" for row in implausible)

    def test_ca_trump_implausible_changes_p_a_in_prompt(self):
        prompts, _, _ = build_all()
        pmap = {row["example_id"]: row["prompt"] for row in prompts}
        base = pmap[f"{slug('CA Trump voter')}__open_probs"]
        implausible = pmap[f"{slug('CA Trump voter')}__implausible__mc_full_probs"]
        assert "13%" in base
        assert "80%" in implausible
        assert "80%" not in base


class TestWrittenCsvs:
    def test_files_exist_after_build(self):
        write_csvs()
        assert (OUT_DIR / "prompts.csv").is_file()
        assert (OUT_DIR / "items.csv").is_file()
        assert (OUT_DIR / "benchmark.csv").is_file()

        with (OUT_DIR / "prompts.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 34

        with (OUT_DIR / "benchmark.csv").open(encoding="utf-8") as handle:
            bench = list(csv.DictReader(handle))
        assert len(bench) == 34
        assert "intersection_size" in bench[0]
        assert "problem_type" in bench[0]
        assert "response_type" in bench[0]
        assert "has_statistics" in bench[0]
        assert "numeric_score_percent" in bench[0]
        assert "scepticism_required" in bench[0]
        assert "scepticism_score_target" in bench[0]


def _prompt_for(example_id: str) -> str:
    with (OUT_DIR / "prompts.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["example_id"] == example_id:
                return row["prompt"]
    raise KeyError(example_id)
