"""Tests for Kaggle run.json export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.kaggle_runs import (
    BaseRatePromptRecord,
    dedupe_base_rate_prompt_records,
    filter_run_rows_to_benchmark,
    load_base_rate_prompt_records_from_run_file,
    load_base_rate_run_rows_from_tree,
    merged_results_from_kaggle_runs,
)


def _sample_simple_run() -> dict:
    return {
        "taskVersion": {"name": "simple_rate_normative_accuracy"},
        "modelVersion": {"slug": "google/gemini-2.5-flash"},
        "results": [{"numericResult": {"value": 0.45}, "type": "AGGREGATED"}],
        "subruns": [
            {
                "taskVersion": {"name": "Simple Rate Prompt Response"},
                "modelVersion": {"slug": "google/gemini-2.5-flash"},
                "results": [
                    {
                        "dictResult": {
                            "example_id": "ca_republican_voter__natural__mc_full",
                            "response": "A",
                            "reasoning": [],
                            "answer_line": "A",
                            "confidence_line": "",
                            "parsed_answer_type": "mc_choice",
                            "parsed_percent": None,
                            "parsed_choice": "A",
                            "parsed_confidence": None,
                            "scoring_type": "mc_numeric",
                        },
                        "type": "AGGREGATED",
                    }
                ],
            },
            {
                "taskVersion": {"name": "Simple Rate Prompt Response"},
                "modelVersion": {"slug": "google/gemini-2.5-flash"},
                "results": [
                    {
                        "dictResult": {
                            "example_id": "ca_republican_voter__natural__data_audit",
                            "response": "A",
                            "reasoning": None,
                            "answer_line": "A",
                            "confidence_line": "",
                            "parsed_answer_type": "mc_choice",
                            "parsed_percent": None,
                            "parsed_choice": "A",
                            "parsed_confidence": None,
                            "scoring_type": "data_audit",
                        },
                        "type": "AGGREGATED",
                    }
                ],
            },
        ],
    }


def _sample_aggregate_run() -> dict:
    return {
        "taskVersion": {"name": "base_rate_normative_accuracy"},
        "modelVersion": {"slug": "google/gemini-2.5-flash"},
        "results": [{"numericResult": {"value": 0.45}, "type": "AGGREGATED"}],
        "subruns": [
            {
                "taskVersion": {"name": "Base Rate Prompt Response"},
                "modelVersion": {"slug": "google/gemini-2.5-flash"},
                "results": [
                    {
                        "dictResult": {
                            "example_id": "ca_republican_voter__open_probs",
                            "response": "About 10%\n3",
                            "reasoning": [],
                            "answer_line": "About 10%",
                            "confidence_line": "3",
                            "parsed_answer_type": "probability",
                            "parsed_percent": 10.0,
                            "parsed_choice": "",
                            "parsed_confidence": 3,
                            "scoring_type": "open",
                        },
                        "type": "AGGREGATED",
                    }
                ],
            },
            {
                "taskVersion": {"name": "Base Rate Prompt Response"},
                "modelVersion": {"slug": "google/gemini-2.5-flash"},
                "results": [
                    {
                        "dictResult": {
                            "example_id": "ca_republican_voter__mc_numeric_probs",
                            "response": "D\n4",
                            "reasoning": None,
                            "answer_line": "D",
                            "confidence_line": "4",
                            "parsed_answer_type": "mc_choice",
                            "parsed_percent": None,
                            "parsed_choice": "D",
                            "parsed_confidence": 4,
                            "scoring_type": "mc_numeric",
                        },
                        "type": "AGGREGATED",
                    }
                ],
            },
        ],
    }


class TestKaggleRunsExport:
    def test_extract_prompt_records_from_subruns(self, tmp_path: Path):
        run_file = tmp_path / "base_rate_normative_accuracy.run.json"
        run_file.write_text(json.dumps(_sample_aggregate_run()), encoding="utf-8")

        records = load_base_rate_prompt_records_from_run_file(run_file)
        assert len(records) == 2
        by_id = {record.example_id: record for record in records}
        assert by_id["ca_republican_voter__open_probs"].model == "google/gemini-2.5-flash"
        assert "About 10%" in by_id["ca_republican_voter__open_probs"].response
        assert by_id["ca_republican_voter__mc_numeric_probs"].response.startswith("D")

    def test_load_run_rows_from_tree(self, tmp_path: Path):
        run_file = tmp_path / "nested" / "aggregate.run.json"
        run_file.parent.mkdir(parents=True)
        run_file.write_text(json.dumps(_sample_aggregate_run()), encoding="utf-8")

        rows = load_base_rate_run_rows_from_tree(tmp_path)
        assert len(rows) == 2
        assert {row["example_id"] for row in rows} == {
            "ca_republican_voter__open_probs",
            "ca_republican_voter__mc_numeric_probs",
        }

    def test_dedupe_keeps_latest(self):
        first = BaseRatePromptRecord(
            model="google/gemini-2.5-flash",
            example_id="ca_republican_voter__open_probs",
            response="old",
            reasoning=None,
            run_file=Path("a.run.json"),
        )
        second = BaseRatePromptRecord(
            model="google/gemini-2.5-flash",
            example_id="ca_republican_voter__open_probs",
            response="new",
            reasoning=None,
            run_file=Path("b.run.json"),
        )
        out = dedupe_base_rate_prompt_records([first, second])
        assert len(out) == 1
        assert out[0].response == "new"

    def test_filter_run_rows_to_benchmark_drops_stale_example_ids(self):
        from benchmarks.simple_rate import BENCHMARK_CSV, load_benchmark_rows

        _, benchmark_rows = load_benchmark_rows(BENCHMARK_CSV)
        benchmark_ids = {row["example_id"] for row in benchmark_rows}
        run_rows = [
            {
                "example_id": "ca_trump_voter__natural__mc_full",
                "response": "10%",
                "model": "test-model",
            },
            {
                "example_id": "ca_republican_voter__natural__mc_full",
                "response": "A",
                "model": "test-model",
            },
        ]
        filtered = filter_run_rows_to_benchmark(
            run_rows,
            benchmark_example_ids=benchmark_ids,
        )
        assert len(filtered) == 1
        assert filtered[0]["example_id"] == "ca_republican_voter__natural__mc_full"

    def test_merged_results_from_kaggle_runs(self, tmp_path: Path):
        from benchmarks.base_rate import BENCHMARK_CSV

        run_file = tmp_path / "aggregate.run.json"
        run_file.write_text(json.dumps(_sample_aggregate_run()), encoding="utf-8")

        merged = merged_results_from_kaggle_runs(tmp_path, benchmark_path=BENCHMARK_CSV)
        models = {row["model"] for row in merged}
        assert models == {"google/gemini-2.5-flash"}
        assert len(merged) == 2
        scored = [
            row
            for row in merged
            if row["example_id"] == "ca_republican_voter__open_probs"
            and row["llm_response"]
        ]
        assert len(scored) == 1
        assert scored[0]["parseable"] in {"true", "false"}

    def test_merged_simple_results_from_kaggle_runs(self, tmp_path: Path):
        from benchmarks.simple_rate import BENCHMARK_CSV
        from benchmarks.kaggle_runs import merged_simple_results_from_kaggle_runs

        run_file = tmp_path / "simple_rate_normative_accuracy.run.json"
        run_file.write_text(json.dumps(_sample_simple_run()), encoding="utf-8")

        merged = merged_simple_results_from_kaggle_runs(tmp_path, benchmark_path=BENCHMARK_CSV)
        models = {row["model"] for row in merged}
        assert models == {"google/gemini-2.5-flash"}
        assert len(merged) == 2
        assert "path_c_confusion" in merged[0]
