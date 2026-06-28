"""Kaggle Benchmarks tasks for the simple two-path base-rate benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import kaggle_benchmarks as kbench

from benchmarks.simple_rate import (
    BENCHMARK_CSV,
    DEFAULT_MERGED_RESULTS_CSV,
    load_benchmark,
    parse_response,
    path_c_confusion_rate,
    prompts_to_dataframe,
    score_pivot_dataframe,
    score_run_rows,
    write_merged_results_csv,
)

BASE_RATE_MAX_OUTPUT_TOKENS = 128
BASE_RATE_N_JOBS = 1
_active_max_output_tokens = BASE_RATE_MAX_OUTPUT_TOKENS


def _kaggle_merged_results_path() -> Path:
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working") / "simple_merged_results.csv"
    return DEFAULT_MERGED_RESULTS_CSV


def model_slug_from_run(run) -> str:
    subject = getattr(run, "evaluated_subject", None)
    if subject is None:
        return "unknown"
    return getattr(subject, "model", None) or getattr(subject, "name", None) or "unknown"


def model_slug_from_llm(llm) -> str:
    return getattr(llm, "model", None) or getattr(llm, "name", None) or "unknown"


def _llm_extra_api_params(token_cap: int) -> dict[str, int | dict[str, int]]:
    return {
        "max_tokens": token_cap,
        "extra_body": {"max_output_tokens": token_cap},
    }


def _prompt_llm(llm, prompt: str) -> str:
    token_cap = _active_max_output_tokens
    return llm.prompt(
        prompt,
        reasoning="none",
        extra_api_params=_llm_extra_api_params(token_cap),
    )


@kbench.task(
    store_task=False,
    description="One simple base-rate item: prompt, parsed lines, and reasoning.",
)
def simple_rate_prompt_response(llm, example_id: str, prompt: str) -> dict:
    response = _prompt_llm(llm, prompt)
    item = load_benchmark()[example_id]
    parsed = parse_response(response, scoring_type=item.scoring_type)
    return {
        "example_id": example_id,
        "response": response,
        "reasoning": kbench.last_reasoning_traces(),
        "answer_line": parsed.answer_line,
        "confidence_line": parsed.confidence_line,
        "parsed_answer_type": parsed.answer_type,
        "parsed_percent": parsed.percent,
        "parsed_choice": parsed.choice,
        "parsed_confidence": parsed.confidence,
        "scoring_type": item.scoring_type,
    }


def evaluate_simple_rate_benchmark(
    llm,
    *,
    benchmark_path: str | Path | None = None,
    merged_results_path: str | Path | None = None,
    max_prompts: int | None = None,
    max_output_tokens: int = BASE_RATE_MAX_OUTPUT_TOKENS,
    n_jobs: int = BASE_RATE_N_JOBS,
):
    """Run simple benchmark prompts, score normative P(C|T), track P(T|C) confusion."""
    global _active_max_output_tokens

    benchmark_path = Path(benchmark_path or BENCHMARK_CSV)
    llms = llm if isinstance(llm, list) else [llm]
    evaluation_data = prompts_to_dataframe(benchmark_path, max_prompts=max_prompts)
    example_ids = (
        None
        if max_prompts is None
        else list(evaluation_data["example_id"])
    )
    previous_max_output_tokens = _active_max_output_tokens
    _active_max_output_tokens = max_output_tokens
    try:
        with kbench.client.enable_cache():
            runs = simple_rate_prompt_response.evaluate(
                llm=llms,
                evaluation_data=evaluation_data,
                n_jobs=n_jobs,
                remove_run_files=True,
            )
    finally:
        _active_max_output_tokens = previous_max_output_tokens

    run_rows = []
    for run in runs.runs:
        row = dict(run.result)
        row["model"] = model_slug_from_run(run)
        if row["model"] == "unknown" and len(llms) == 1:
            row["model"] = model_slug_from_llm(llms[0])
        run_rows.append(row)

    score = score_run_rows(run_rows)
    merged_path, pivot_path = write_merged_results_csv(
        run_rows,
        merged_results_path or _kaggle_merged_results_path(),
        benchmark_path=benchmark_path,
        score=score,
        example_ids=example_ids,
    )
    with merged_path.open(newline="", encoding="utf-8") as handle:
        merged_rows = list(csv.DictReader(handle))
    pivot = score_pivot_dataframe(merged_rows)
    path_c_rate = path_c_confusion_rate(run_rows)
    return runs, score, merged_path, pivot_path, pivot, path_c_rate


@kbench.task(
    name="simple_rate_normative_accuracy",
    description=(
        "Simple base-rate benchmark (20 conditions): fraction of parseable answers "
        "matching normative P(C|T). Higher is better."
    ),
)
def simple_rate_normative_accuracy(llm) -> float:
    _, score, _, _, _, _ = evaluate_simple_rate_benchmark(llm)
    return float(score.accuracy)


@kbench.task(
    name="simple_rate_path_c_confusion",
    description=(
        "Simple base-rate benchmark: fraction of parseable answers matching the "
        "P(T|C) lure (inverse conditional). Lower is better."
    ),
)
def simple_rate_path_c_confusion(llm) -> float:
    _, _, _, _, _, path_c_rate = evaluate_simple_rate_benchmark(llm)
    return float(path_c_rate)
