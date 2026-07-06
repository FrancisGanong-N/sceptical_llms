"""Tests for kbench resume evaluation helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd

from benchmarks.kbench_resume import (
    count_complete_prompt_slots,
    evaluate_prompt_task_with_resume,
    is_prompt_cache_complete,
    missing_prompt_ids,
    prompt_cache_path,
)


class FakeLLM:
    model = "test-model"


def _write_completed_cache(task: object, param_id: object, model: str) -> None:
    path = prompt_cache_path(task, param_id, model)
    path.write_text(
        json.dumps(
            {
                "state": "BENCHMARK_TASK_RUN_STATE_COMPLETED",
                "results": [
                    {
                        "dictResult": {
                            "example_id": f"ex_{param_id}",
                            "response": "cached",
                        },
                        "type": "AGGREGATED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _make_fake_task():
    calls: list[list[object]] = []

    class FakeTask:
        name = "fake prompt response"

        def evaluate(self, llm, evaluation_data, **kwargs):
            calls.append(list(evaluation_data.index))
            from kaggle_benchmarks import runs

            batch = []
            llm_list = llm if isinstance(llm, list) else [llm]
            subject = llm_list[0]
            for param_id in evaluation_data.index:
                run = MagicMock()
                run.param_id = param_id
                run.params = {"_id": param_id}
                run.result = {
                    "example_id": evaluation_data.loc[param_id, "example_id"],
                    "response": "fresh",
                }
                run.evaluated_subject = subject
                batch.append(run)
                _write_completed_cache(self, param_id, FakeLLM.model)
            return runs.Runs(batch)

        def run(self, llm, _id, **kwargs):
            run = MagicMock()
            run.param_id = _id
            run.params = {"_id": _id, **kwargs}
            run.cached = True
            run.result = {"example_id": kwargs["example_id"], "response": "cached"}
            run.evaluated_subject = llm
            return run

    task = FakeTask()
    task.calls = calls  # type: ignore[attr-defined]
    return task


def _sample_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            [
                {"example_id": "ex_0", "prompt": "p0"},
                {"example_id": "ex_1", "prompt": "p1"},
                {"example_id": "ex_2", "prompt": "p2"},
            ]
        )
        .reset_index(names=["_id"])
        .set_index("_id")
    )


class TestKbenchResume:
    def test_is_prompt_cache_complete(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        task = _make_fake_task()
        assert not is_prompt_cache_complete(task, 0, FakeLLM.model)
        _write_completed_cache(task, 0, FakeLLM.model)
        assert is_prompt_cache_complete(task, 0, FakeLLM.model)

    def test_missing_prompt_ids(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        task = _make_fake_task()
        frame = _sample_frame()
        _write_completed_cache(task, 0, FakeLLM.model)
        _write_completed_cache(task, 2, FakeLLM.model)
        assert missing_prompt_ids(task, frame, [FakeLLM()]) == [1]

    def test_count_complete_prompt_slots(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        task = _make_fake_task()
        frame = _sample_frame()
        _write_completed_cache(task, 1, FakeLLM.model)
        assert count_complete_prompt_slots(task, frame, [FakeLLM()]) == 1

    def test_evaluate_runs_only_missing_prompts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        task = _make_fake_task()
        frame = _sample_frame()
        _write_completed_cache(task, 0, FakeLLM.model)

        result = evaluate_prompt_task_with_resume(
            task,
            llm=FakeLLM(),
            evaluation_data=frame,
            max_attempts=1,
            cleanup_after_complete=False,
        )

        assert task.calls == [[1, 2]]
        assert len(result.runs) == 3
        by_id = {run.result["example_id"]: run.result["response"] for run in result.runs}
        assert by_id["ex_0"] == "cached"
        assert by_id["ex_1"] == "fresh"
        assert by_id["ex_2"] == "fresh"

    def test_cleanup_removes_cache_after_full_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        task = _make_fake_task()
        frame = _sample_frame().iloc[:1]

        evaluate_prompt_task_with_resume(
            task,
            llm=FakeLLM(),
            evaluation_data=frame,
            max_attempts=1,
            cleanup_after_complete=True,
        )

        assert not is_prompt_cache_complete(task, 0, FakeLLM.model)
