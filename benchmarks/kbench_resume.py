"""Resume incomplete kbench per-prompt evaluations using on-disk run caches."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import kaggle_benchmarks as kbench

from kaggle_benchmarks.kaggle import serialization

logger = logging.getLogger(__name__)

COMPLETED_RUN_STATE = "BENCHMARK_TASK_RUN_STATE_COMPLETED"
DEFAULT_RESUME_MAX_ATTEMPTS = 3
DEFAULT_RESUME_DELAY_SECONDS = 5


def model_slug_from_llm(llm: Any) -> str:
    return getattr(llm, "model", None) or getattr(llm, "name", None) or "unknown"


def model_slug_from_run(run: Any) -> str:
    subject = getattr(run, "evaluated_subject", None)
    if subject is None:
        return "unknown"
    return getattr(subject, "model", None) or getattr(subject, "name", None) or "unknown"


def prompt_cache_id(param_id: object, model_slug: str) -> str:
    suffix = f"_{model_slug}" if model_slug else ""
    return f"run_param_id_{param_id}{suffix}"


def prompt_cache_path(task: Any, param_id: object, model_slug: str) -> Path:
    return Path(
        serialization.generate_run_filename(task.name, prompt_cache_id(param_id, model_slug))
    )


def is_prompt_cache_complete(task: Any, param_id: object, model_slug: str) -> bool:
    path = prompt_cache_path(task, param_id, model_slug)
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            run_data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return run_data.get("state") == COMPLETED_RUN_STATE


def prepare_evaluation_frame(evaluation_data: Any):
    """Use benchmark row index as stable kbench ``param_id`` (cache key)."""
    import pandas as pd

    frame = evaluation_data
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    if frame.index.name == "_id":
        return frame
    if "_id" in frame.columns:
        return frame.set_index("_id", drop=True)
    return frame.reset_index(names=["_id"]).set_index("_id")


def missing_prompt_ids(
    task: Any,
    evaluation_data: Any,
    llms: list[Any],
) -> list[object]:
    """Return index values with no COMPLETED on-disk cache for any ``llm``."""
    frame = prepare_evaluation_frame(evaluation_data)
    missing: list[object] = []
    for param_id in frame.index:
        for llm in llms:
            if not is_prompt_cache_complete(task, param_id, model_slug_from_llm(llm)):
                missing.append(param_id)
                break
    return missing


def count_complete_prompt_slots(
    task: Any,
    evaluation_data: Any,
    llms: list[Any],
) -> int:
    frame = prepare_evaluation_frame(evaluation_data)
    complete = 0
    for param_id in frame.index:
        for llm in llms:
            if is_prompt_cache_complete(task, param_id, model_slug_from_llm(llm)):
                complete += 1
    return complete


def _scalar_grid_kwargs(evaluate_kwargs: dict[str, Any]) -> dict[str, Any]:
    reserved = frozenset({"remove_run_files", "n_jobs", "max_attempts", "retry_delay"})
    grid: dict[str, Any] = {}
    for key, value in evaluate_kwargs.items():
        if key in reserved:
            continue
        if isinstance(value, list) and len(value) == 1:
            grid[key] = value[0]
        else:
            grid[key] = value
    return grid


def _run_key(run: Any) -> tuple[object, str]:
    param_id = run.param_id
    if param_id is None and isinstance(run.params, dict):
        param_id = run.params.get("_id")
    return (param_id, model_slug_from_run(run))


def _collect_cached_runs(
    task: Any,
    frame: Any,
    llms: list[Any],
    evaluate_kwargs: dict[str, Any],
    merged: dict[tuple[object, str], Any],
) -> None:
    grid_scalar = _scalar_grid_kwargs(evaluate_kwargs)
    for param_id in frame.index:
        row_kwargs = frame.loc[param_id].to_dict()
        for llm_item in llms:
            model = model_slug_from_llm(llm_item)
            key = (param_id, model)
            if key in merged:
                continue
            if not is_prompt_cache_complete(task, param_id, model):
                continue
            run = task.run(llm=llm_item, _id=param_id, **row_kwargs, **grid_scalar)
            merged[key] = run


def evaluate_prompt_task_with_resume(
    task: Any,
    *,
    llm: Any | list[Any],
    evaluation_data: Any,
    max_attempts: int = DEFAULT_RESUME_MAX_ATTEMPTS,
    retry_delay: int = DEFAULT_RESUME_DELAY_SECONDS,
    cleanup_after_complete: bool = True,
    **evaluate_kwargs: Any,
) -> kbench.Runs:
    """Evaluate only prompts missing a COMPLETED cache; retry until full or exhausted.

    Completed per-prompt ``*.run.json`` files are left on disk between attempts so an
    interrupted session or dashboard rerun can fill gaps without redoing finished items.
    """
    llms = llm if isinstance(llm, list) else [llm]
    frame = prepare_evaluation_frame(evaluation_data)
    expected = len(frame) * len(llms)
    evaluate_kwargs.setdefault("remove_run_files", False)

    merged: dict[tuple[object, str], Any] = {}
    attempt = 0

    with kbench.client.enable_cache():
        while attempt < max_attempts:
            attempt += 1
            missing_ids = missing_prompt_ids(task, frame, llms)
            n_complete = count_complete_prompt_slots(task, frame, llms)

            if not missing_ids:
                break

            if attempt == 1 and n_complete:
                print(
                    f"Resuming {task.name}: {n_complete}/{expected} cached; "
                    f"running {len(missing_ids)} missing prompt(s).",
                    flush=True,
                )
            elif attempt > 1:
                print(
                    f"Retry {attempt}/{max_attempts} for {task.name}: "
                    f"{len(missing_ids)} prompt(s) still missing.",
                    flush=True,
                )
                time.sleep(retry_delay)

            batch = task.evaluate(
                llm=llms,
                evaluation_data=frame.loc[missing_ids],
                **evaluate_kwargs,
            )
            for run in batch.runs:
                merged[_run_key(run)] = run

            if count_complete_prompt_slots(task, frame, llms) >= expected:
                break

        _collect_cached_runs(task, frame, llms, evaluate_kwargs, merged)

    final_runs = kbench.Runs(list(merged.values()))
    if cleanup_after_complete and count_complete_prompt_slots(task, frame, llms) >= expected:
        for param_id in frame.index:
            for llm_item in llms:
                prompt_cache_path(task, param_id, model_slug_from_llm(llm_item)).unlink(
                    missing_ok=True
                )

    return final_runs
