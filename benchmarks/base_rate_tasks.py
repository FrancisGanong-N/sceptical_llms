"""Kaggle Benchmarks tasks for the sceptical base-rate benchmark."""

import kaggle_benchmarks as kbench

from benchmarks.base_rate import (
    BENCHMARK_CSV,
    load_benchmark,
    parse_response,
    prompts_to_dataframe,
    score_base_rate_responses,
)


@kbench.task(
    store_task=False,
    description="One base-rate benchmark item: prompt and parsed response fields.",
)
def base_rate_prompt_response(llm, example_id: str, prompt: str) -> dict:
    response = llm.prompt(prompt)
    item = load_benchmark()[example_id]
    parsed = parse_response(response, response_type=item.response_type)
    return {
        "example_id": example_id,
        "response": response,
        "reasoning": kbench.last_reasoning_traces(),
        "response_type": parsed.response_type,
        "percent": parsed.percent,
        "choice": parsed.choice,
        "confidence": parsed.confidence,
    }


def _evaluate_base_rate_benchmark(llm, benchmark_path: str | None = None):
    evaluation_data = prompts_to_dataframe(benchmark_path or BENCHMARK_CSV)
    with kbench.client.enable_cache():
        return base_rate_prompt_response.evaluate(
            llm=[llm],
            evaluation_data=evaluation_data,
            n_jobs=2,
            remove_run_files=True,
        )


def _score_from_runs(runs: kbench.Runs):
    rows = [run.result for run in runs.runs]
    return score_base_rate_responses(rows)


@kbench.task(
    name="base_rate_normative_accuracy",
    description=(
        "Sceptical base-rate benchmark (54 conditions): fraction of parseable answers "
        "matching the overlap-aware Bayesian posterior. Higher is better."
    ),
)
def base_rate_normative_accuracy(llm) -> float:
    runs = _evaluate_base_rate_benchmark(llm)
    score = _score_from_runs(runs)
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
    runs = _evaluate_base_rate_benchmark(llm)
    score = _score_from_runs(runs)
    return float(score.bias_index)
