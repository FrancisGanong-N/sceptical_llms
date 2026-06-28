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
