"""Tests for the simple two-path base-rate benchmark."""

from benchmarks.simple_rate import (
    BENCHMARK_CSV,
    PATH_C_LURE_NAME,
    load_benchmark,
    matches_path_c_confusion,
    parse_response,
    score_run_rows,
)


def _score_responses(responses: dict[str, str]):
    items = load_benchmark()
    run_rows = [
        {"example_id": example_id, "response": response, "model": "test-model"}
        for example_id, response in responses.items()
    ]
    return score_run_rows(run_rows, items=items)


class TestSimpleBenchmarkData:
    def test_benchmark_csv_exists(self):
        assert BENCHMARK_CSV.is_file()

    def test_load_benchmark(self):
        items = load_benchmark()
        assert len(items) == 20
        assert {item.response_type for item in items.values()} == {"open", "mc_numeric"}
        discharged = items["discharged_weapon_last_year__open_probs"]
        assert discharged.p_t_given_c == 0.003


class TestPathCScoring:
    def test_open_normative_scores_true(self):
        score = _score_responses({"discharged_weapon_last_year__open_probs": "77%"})
        assert score.examples[0].score is True
        assert score.examples[0].parseable is True

    def test_open_path_c_confusion_detected(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__open_probs"
        item = items[example_id]
        parsed = parse_response("0.3%", scoring_type="open")
        assert matches_path_c_confusion(item, parsed) is True

    def test_mc_path_c_lure_detected(self):
        items = load_benchmark()
        example_id = "ca_trump_voter__mc_numeric_probs"
        item = items[example_id]
        path_c_letter = next(
            option.letter
            for option in item.options
            if option.lure == PATH_C_LURE_NAME
        )
        assert path_c_letter == "C"
        parsed = parse_response(path_c_letter, scoring_type="mc_numeric")
        assert matches_path_c_confusion(item, parsed) is True
        scored = score_run_rows(
            [
                {
                    "example_id": example_id,
                    "response": path_c_letter,
                    "model": "test-model",
                }
            ],
            items=items,
        )
        assert scored.examples[0].score is False
