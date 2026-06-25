"""Kaggle Benchmarks tasks for the sceptical base-rate benchmark."""

from pathlib import Path

import kaggle_benchmarks as kbench

from benchmarks.base_rate import (
    BENCHMARK_CSV,
    DEFAULT_MERGED_RESULTS_CSV,
    load_benchmark,
    parse_response,
    prompts_to_dataframe,
    score_pivot_dataframe,
    score_run_rows,
    split_response_lines,
    write_merged_results_csv,
)


def _kaggle_merged_results_path() -> Path:
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working") / "base_rate_merged_results.csv"
    return DEFAULT_MERGED_RESULTS_CSV


def model_slug_from_run(run) -> str:
    subject = getattr(run, "evaluated_subject", None)
    if subject is None:
        return "unknown"
    return getattr(subject, "model", None) or getattr(subject, "name", None) or "unknown"


def model_slug_from_llm(llm) -> str:
    return getattr(llm, "model", None) or getattr(llm, "name", None) or "unknown"


# Kaggle Model Proxy reserves quota from max_output_tokens; keep this low for
# two-line answers (percent/letter + confidence). Reasoning is disabled to
# avoid large thinking-token reservations on google/* models.
BASE_RATE_MAX_OUTPUT_TOKENS = 128
BASE_RATE_N_JOBS = 1
_active_max_output_tokens = BASE_RATE_MAX_OUTPUT_TOKENS


def _llm_extra_api_params(token_cap: int) -> dict[str, int | dict[str, int]]:
    """Cap generation length for Kaggle Model Proxy.

    The OpenAI SDK accepts ``max_tokens`` on chat.completions.create.
    Model Proxy uses ``max_output_tokens`` in the request body for quota
    reservation; passing it as a top-level kwarg raises TypeError.
    """
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
    description="One base-rate benchmark item: prompt, parsed lines, and reasoning.",
)
def base_rate_prompt_response(llm, example_id: str, prompt: str) -> dict:
    response = _prompt_llm(llm, prompt)
    item = load_benchmark()[example_id]
    parsed = parse_response(response, scoring_type=item.scoring_type)
    answer_line, confidence_line = split_response_lines(response)
    return {
        "example_id": example_id,
        "response": response,
        "reasoning": kbench.last_reasoning_traces(),
        "answer_line": answer_line,
        "confidence_line": confidence_line,
        "parsed_answer_type": parsed.answer_type,
        "parsed_percent": parsed.percent,
        "parsed_choice": parsed.choice,
        "parsed_confidence": parsed.confidence,
        "scoring_type": item.scoring_type,
    }


def evaluate_base_rate_benchmark(
    llm,
    *,
    benchmark_path: str | Path | None = None,
    merged_results_path: str | Path | None = None,
    max_prompts: int | None = None,
    max_output_tokens: int = BASE_RATE_MAX_OUTPUT_TOKENS,
    n_jobs: int = BASE_RATE_N_JOBS,
):
    """Run benchmark prompts, score by response type, and write merged results CSV."""
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
            runs = base_rate_prompt_response.evaluate(
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
    import csv

    with merged_path.open(newline="", encoding="utf-8") as handle:
        merged_rows = list(csv.DictReader(handle))
    pivot = score_pivot_dataframe(merged_rows)
    return runs, score, merged_path, pivot_path, pivot


def _score_from_runs(runs: kbench.Runs):
    rows = []
    for run in runs.runs:
        row = dict(run.result)
        row["model"] = model_slug_from_run(run)
        rows.append(row)
    return score_run_rows(rows)


@kbench.task(
    name="base_rate_normative_accuracy",
    description=(
        "Sceptical base-rate benchmark (54 conditions): fraction of parseable answers "
        "matching the overlap-aware Bayesian posterior. Higher is better."
    ),
)
def base_rate_normative_accuracy(llm) -> float:
    _, score, _, _, _ = evaluate_base_rate_benchmark(llm)
    return float(score.normative_accuracy)


@kbench.task(
    name="base_rate_bias_index",
    description=(
        "Sceptical base-rate benchmark: fraction of parseable answers choosing a lure "
        "(numeric shortcut, meta scepticism, or wrong MC option). Higher indicates "
        "more base-rate neglect or unwarranted scepticism."
    ),
)
def base_rate_bias_index(llm) -> float:
    _, score, _, _, _ = evaluate_base_rate_benchmark(llm)
    return float(score.bias_index)
