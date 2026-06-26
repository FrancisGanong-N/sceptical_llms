"""Tests for the sceptical base-rate benchmark."""

import csv
from pathlib import Path

from benchmarks.base_rate import (
    BENCHMARK_CSV,
    load_benchmark,
    matches_scepticism_target,
    merge_run_results,
    parse_mc_choice,
    parse_response,
    prompts_to_dataframe,
    score_base_rate_responses,
    score_example,
    score_mc_numeric_example,
    score_pivot_dataframe,
    split_response_lines,
    write_merged_results_csv,
)
from benchmarks.base_rate_tasks import (
    base_rate_bias_index,
    base_rate_normative_accuracy,
    base_rate_prompt_response,
)


class TestBenchmarkData:
    def test_benchmark_csv_exists(self):
        assert BENCHMARK_CSV.is_file()

    def test_load_benchmark(self):
        items = load_benchmark()
        assert len(items) == 120
        response_types = {item.response_type for item in items.values()}
        assert response_types == {"open", "mc_numeric", "mc_full"}

    def test_prompts_dataframe(self):
        df = prompts_to_dataframe()
        assert list(df.columns) == ["example_id", "prompt"]
        assert len(df) == 120
        assert "statistical consultant" in df.iloc[0]["prompt"]

    def test_prompts_dataframe_max_prompts(self):
        df = prompts_to_dataframe(max_prompts=2)
        assert len(df) == 2
        full = prompts_to_dataframe()
        assert list(df["example_id"]) == list(full["example_id"][:2])


class TestParsing:
    def test_split_response_lines(self):
        assert split_response_lines("91%\n4") == ("91%", "4")

    def test_parse_open_probability(self):
        parsed = parse_response("91%\n4", scoring_type="open")
        assert parsed.answer_type == "probability"
        assert parsed.percent == 91.0
        assert parsed.confidence == 4
        assert parsed.answer_line == "91%"

    def test_parse_open_meta(self):
        parsed = parse_response(
            "Insufficient information\n2",
            scoring_type="open",
        )
        assert parsed.answer_type == "meta_insufficient"
        assert parsed.confidence_line == "2"

    def test_parse_mc_choice(self):
        assert parse_mc_choice("B\n5") == "B"
        assert parse_mc_choice("I choose option H.") == "H"


class TestScoring:
    def test_open_well_posed_percent_matches_target(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        score = score_base_rate_responses(
            {example_id: "91.2%\n4"},
            items=items,
        )
        assert score.accuracy == 1.0
        assert score.examples[0].score is True

    def test_open_well_posed_meta_is_not_target_when_numeric_agrees(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        score = score_base_rate_responses(
            {example_id: "Insufficient information\n3"},
            items=items,
        )
        assert score.accuracy == 0.0
        assert score.examples[0].score is False

    def test_open_off_target_is_not_scored(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        item = items[example_id]
        parsed = parse_response("50%\n3", scoring_type="open")
        scored = score_example(item, parsed)
        assert scored.parseable is True
        assert scored.score is False

    def test_open_overlap_partition_percent_within_half_percent(self):
        items = load_benchmark()
        example_id = "diabetes_insulin_obese__overlap__open_probs"
        item = items[example_id]
        target = float(item.scepticism_score_target)
        score = score_base_rate_responses(
            {example_id: f"{target + 0.4}%\n2"},
            items=items,
        )
        assert score.examples[0].score is True
        miss = score_base_rate_responses(
            {example_id: f"{target + 0.6}%\n2"},
            items=items,
        )
        assert miss.examples[0].score is False

    def test_open_small_posterior_off_target_is_not_scored(self):
        items = load_benchmark()
        example_id = "actor_waiter_overlap__overlap__open_no_probs"
        parsed = parse_response("2%\n2", scoring_type="open")
        scored = score_example(items[example_id], parsed)
        assert scored.score is False

    def test_llm_extra_api_params(self):
        from benchmarks.base_rate_tasks import _llm_extra_api_params

        params = _llm_extra_api_params(128)
        assert params["max_tokens"] == 128
        assert "max_output_tokens" not in params
        assert params["extra_body"]["max_output_tokens"] == 128

    def test_mc_numeric_normative_choice_when_target_na(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__mc_numeric_probs"
        normative = items[example_id].normative_choice
        score = score_base_rate_responses(
            {example_id: f"{normative}\n5"},
            items=items,
        )
        assert score.accuracy == 1.0

    def test_mc_lure_choice(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__mc_numeric_probs"
        item = items[example_id]
        lure = sorted(item.lure_choices)[0]
        score = score_base_rate_responses(
            {example_id: f"{lure}\n5"},
            items=items,
        )
        assert score.accuracy == 0.0

    def test_mc_numeric_meta_letter_does_not_match_na_target(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__mc_numeric_probs"
        item = items[example_id]
        parsed = parse_response("F\n4", scoring_type="mc_numeric")
        scored = score_mc_numeric_example(item, parsed)
        assert scored.parseable is True
        assert scored.score is False

    def test_mc_full_normative_matches_target_when_numeric_agrees(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__mc_full_probs"
        normative = items[example_id].normative_choice
        score = score_base_rate_responses(
            {example_id: f"{normative}\n4"},
            items=items,
        )
        assert score.accuracy == 1.0

    def test_mc_full_numeric_not_accepted_when_scepticism_required(self):
        items = load_benchmark()
        example_id = "diabetes_insulin_obese__overlap__mc_full_probs"
        item = items[example_id]
        partition_letter = next(
            option.letter
            for option in item.options
            if option.lure == "partition shortcut (assumes P(C∩D|A)=0)"
        )
        score = score_base_rate_responses(
            {example_id: f"{partition_letter}\n4"},
            items=items,
        )
        assert score.accuracy == 0.0
        meta_score = score_base_rate_responses({example_id: "F\n4"}, items=items)
        assert meta_score.accuracy == 1.0

    def test_matches_scepticism_target_meta(self):
        items = load_benchmark()
        item = items["discharged_weapon_last_year__open_probs"]
        parsed = parse_response("Insufficient information\n1", scoring_type="open")
        assert matches_scepticism_target(item, parsed) is False


class TestMergeResults:
    def test_merge_includes_benchmark_and_run_fields(self, tmp_path: Path):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        run_rows = [
            {
                "example_id": example_id,
                "response": "91.2%\n4",
                "reasoning": "Used Bayes.",
                "model": "test-model",
            }
        ]
        merged = merge_run_results(run_rows, items=items)
        row = next(r for r in merged if r["example_id"] == example_id)
        assert row["vignette_name"] == "discharged weapon (last year)"
        assert row["response_type"] == "open"
        assert row["llm_response"] == "91.2%\n4"
        assert row["answer_line"] == "91.2%"
        assert row["confidence_line"] == "4"
        assert row["reasoning"] == "Used Bayes."
        assert row["score"] == "true"
        assert row["scoring_type"] == "open"
        assert row["model"] == "test-model"
        assert row["scepticism_score_target"] == "91.21"

    def test_write_merged_results_csv(self, tmp_path: Path):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        out = tmp_path / "merged.csv"
        merged_path, pivot_path = write_merged_results_csv(
            [
                {
                    "example_id": example_id,
                    "response": "91.2%\n4",
                    "reasoning": "",
                    "model": "test-model",
                }
            ],
            out,
            items=items,
        )
        assert pivot_path.is_file()
        with out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 120
        assert "llm_response" in rows[0]
        assert "score" in rows[0]
        assert "model" in rows[0]


class TestScorePivot:
    def test_score_pivot_shape(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        merged = merge_run_results(
            [
                {
                    "example_id": example_id,
                    "response": "91.2%\n4",
                    "reasoning": "",
                    "model": "model-a",
                }
            ],
            items=items,
        )
        pivot = score_pivot_dataframe(merged)
        assert list(pivot.columns) == [
            "open_probs",
            "open_no_probs",
            "mc_numeric_probs",
            "mc_numeric_no_probs",
            "mc_full_probs",
            "mc_full_no_probs",
        ]
        assert list(pivot.index) == ["model-a"]
        assert pivot.loc["model-a", "open_probs"] == round(1 / 20, 3)
        assert pivot.loc["model-a", "open_no_probs"] == 0.0

    def test_score_pivot_multiple_models(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        merged = merge_run_results(
            [
                {
                    "example_id": example_id,
                    "response": "91.2%\n4",
                    "reasoning": "",
                    "model": "model-a",
                },
                {
                    "example_id": example_id,
                    "response": "91.2%\n3",
                    "reasoning": "",
                    "model": "model-b",
                },
            ],
            items=items,
        )
        pivot = score_pivot_dataframe(merged)
        assert set(pivot.index) == {"model-a", "model-b"}
        assert pivot.loc["model-a", "open_probs"] == round(1 / 20, 3)
        assert pivot.loc["model-b", "open_probs"] == round(1 / 20, 3)
        assert len(merged) == 120 * 2


class TestTaskRegistration:
    def test_tasks_registered(self):
        assert base_rate_normative_accuracy.name == "base_rate_normative_accuracy"
        assert base_rate_bias_index.name == "base_rate_bias_index"
        assert base_rate_prompt_response.store_task is False
