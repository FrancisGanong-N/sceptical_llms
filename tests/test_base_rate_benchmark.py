"""Tests for the sceptical base-rate benchmark."""

from benchmarks.base_rate import (
    BENCHMARK_CSV,
    load_benchmark,
    parse_mc_choice,
    parse_response,
    prompts_to_dataframe,
    score_base_rate_responses,
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
        assert len(items) == 54
        response_types = {item.response_type for item in items.values()}
        assert response_types == {"open", "mc_numeric", "mc_full"}

    def test_prompts_dataframe(self):
        df = prompts_to_dataframe()
        assert list(df.columns) == ["example_id", "prompt"]
        assert len(df) == 54
        assert "statistical consultant" in df.iloc[0]["prompt"]


class TestParsing:
    def test_parse_open_probability(self):
        parsed = parse_response("91%\n4", response_type="open")
        assert parsed.response_type == "probability"
        assert parsed.percent == 91.0
        assert parsed.confidence == 4

    def test_parse_open_meta(self):
        parsed = parse_response(
            "Insufficient information\n2",
            response_type="open",
        )
        assert parsed.response_type == "meta_insufficient"

    def test_parse_mc_choice(self):
        assert parse_mc_choice("B\n5") == "B"
        assert parse_mc_choice("I choose option H.") == "H"


class TestScoring:
    def test_open_normative_percent(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        score = score_base_rate_responses(
            {example_id: "91.2%\n4"},
            items=items,
        )
        assert score.normative_accuracy == 1.0
        assert score.bias_index == 0.0

    def test_open_meta_counts_as_bias(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        score = score_base_rate_responses(
            {example_id: "Insufficient information\n3"},
            items=items,
        )
        assert score.normative_accuracy == 0.0
        assert score.bias_index == 1.0

    def test_mc_normative_choice(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__mc_numeric_probs"
        normative = items[example_id].normative_choice
        score = score_base_rate_responses(
            {example_id: f"{normative}\n5"},
            items=items,
        )
        assert score.normative_accuracy == 1.0
        assert score.bias_index == 0.0

    def test_mc_lure_choice(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__mc_numeric_probs"
        item = items[example_id]
        lure = sorted(item.lure_choices)[0]
        score = score_base_rate_responses(
            {example_id: f"{lure}\n5"},
            items=items,
        )
        assert score.normative_accuracy == 0.0
        assert score.bias_index == 1.0

    def test_mc_full_meta_is_bias(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__mc_full_probs"
        score = score_base_rate_responses(
            {example_id: "F\n4"},
            items=items,
        )
        assert score.normative_accuracy == 0.0
        assert score.bias_index == 1.0


class TestTaskRegistration:
    def test_tasks_registered(self):
        assert base_rate_normative_accuracy.name == "base_rate_normative_accuracy"
        assert base_rate_bias_index.name == "base_rate_bias_index"
        assert base_rate_prompt_response.store_task is False
