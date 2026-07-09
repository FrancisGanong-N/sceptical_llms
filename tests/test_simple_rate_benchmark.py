"""Tests for the simple two-path base-rate benchmark."""

from benchmarks.simple_rate import (
    BENCHMARK_CSV,
    PATH_C_LURE_NAME,
    load_benchmark,
    matches_path_c_confusion,
    merge_run_results,
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
        assert len(items) == 99
        assert {item.response_type for item in items.values()} == {
            "mc_full",
            "data_audit",
            "response_audit",
        }
        discharged = items["discharged_weapon_last_year__natural__mc_full"]
        assert discharged.p_t_given_c == 0.003


class TestPathCScoring:
    def test_mc_full_normative_scores_true(self):
        items = load_benchmark()
        example_id = "discharged_weapon_last_year__natural__mc_full"
        item = items[example_id]
        score = _score_responses({example_id: item.normative_choice})
        assert score.examples[0].score is True
        assert score.examples[0].parseable is True

    def test_mc_full_path_c_confusion_detected(self):
        items = load_benchmark()
        example_id = "ca_republican_voter__natural__mc_full"
        item = items[example_id]
        path_c_letter = next(
            option.letter
            for option in item.options
            if option.lure == PATH_C_LURE_NAME
        )
        parsed = parse_response(path_c_letter, scoring_type=item.scoring_type)
        assert matches_path_c_confusion(item, parsed) is True

    def test_mc_full_path_c_lure_scored_false(self):
        items = load_benchmark()
        example_id = "ca_republican_voter__natural__mc_full"
        item = items[example_id]
        path_c_letter = next(
            option.letter
            for option in item.options
            if option.lure == PATH_C_LURE_NAME
        )
        parsed = parse_response(path_c_letter, scoring_type="mc_full")
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


class TestMergeResults:
    def test_merge_fill_missing_false_only_includes_actual_runs(self):
        items = load_benchmark()
        example_id = "ca_republican_voter__natural__mc_full"
        item = items[example_id]
        run_rows = [
            {
                "example_id": example_id,
                "response": item.normative_choice,
                "reasoning": "",
                "model": "test-model",
            }
        ]
        merged = merge_run_results(run_rows, items=items, fill_missing=False)
        assert len(merged) == 1
        assert merged[0]["example_id"] == example_id
        assert merged[0]["llm_response"] == item.normative_choice


class TestAlteredScoring:
    def test_altered_mc_full_scores_f(self):
        items = load_benchmark()
        example_id = "ca_republican_voter__altered__mc_full"
        item = items[example_id]
        assert item.scepticism_required is True
        assert item.scepticism_score_target == "F"
        scored = score_run_rows(
            [{"example_id": example_id, "response": "F\n4", "model": "test-model"}],
            items=items,
        )
        assert scored.examples[0].score is True

    def test_altered_data_audit_scores_b(self):
        items = load_benchmark()
        example_id = "ca_republican_voter__altered__data_audit"
        item = items[example_id]
        assert item.scepticism_score_target == "B"
        scored = score_run_rows(
            [{"example_id": example_id, "response": "B", "model": "test-model"}],
            items=items,
        )
        assert scored.examples[0].score is True


class TestNaturalScoring:
    def test_partition_mc_full_scepticism_not_required(self):
        items = load_benchmark()
        example_id = "ca_republican_voter__natural__mc_full"
        item = items[example_id]
        assert item.scepticism_required is False
        scored = score_run_rows(
            [{"example_id": example_id, "response": "F", "model": "test-model"}],
            items=items,
        )
        assert scored.examples[0].score is False

    def test_well_posed_partition_data_audit_scores_a(self):
        items = load_benchmark()
        example_id = "ca_republican_voter__natural__data_audit"
        item = items[example_id]
        assert item.scepticism_required is False
        assert item.normative_choice == "A"
        scored = score_run_rows(
            [{"example_id": example_id, "response": "A", "model": "test-model"}],
            items=items,
        )
        assert scored.examples[0].score is True

    def test_well_posed_partition_response_audit_scores_a(self):
        items = load_benchmark()
        example_id = "ca_republican_voter__natural__response_audit"
        item = items[example_id]
        assert item.scepticism_required is False
        assert item.normative_choice == "A"
        scored = score_run_rows(
            [{"example_id": example_id, "response": "A", "model": "test-model"}],
            items=items,
        )
        assert scored.examples[0].score is True

    def test_overlap_mc_full_scores_f(self):
        items = load_benchmark()
        example_id = "diabetes_insulin_obese__natural__mc_full"
        item = items[example_id]
        assert item.scepticism_required is True
        assert item.scepticism_score_target == "F"
        scored_f = score_run_rows(
            [{"example_id": example_id, "response": "F", "model": "test-model"}],
            items=items,
        )
        assert scored_f.examples[0].score is True
        scored_bayes = score_run_rows(
            [
                {
                    "example_id": example_id,
                    "response": item.normative_choice,
                    "model": "test-model",
                }
            ],
            items=items,
        )
        assert scored_bayes.examples[0].score is False

    def test_overlap_data_audit_scores_b(self):
        items = load_benchmark()
        example_id = "diabetes_insulin_obese__natural__data_audit"
        item = items[example_id]
        assert item.scepticism_score_target == "B"
        scored = score_run_rows(
            [{"example_id": example_id, "response": "B", "model": "test-model"}],
            items=items,
        )
        assert scored.examples[0].score is True
