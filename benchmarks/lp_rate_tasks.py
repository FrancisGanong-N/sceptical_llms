"""Kaggle Benchmarks tasks for the LP tacit-constraint benchmark."""

import csv
from pathlib import Path

import kaggle_benchmarks as kbench

from benchmarks.kbench_openai_patch import apply_null_message_patch
from benchmarks.kbench_resume import evaluate_prompt_task_with_resume

apply_null_message_patch()
from benchmarks.lp_rate import (
    BENCHMARK_CSV,
    DEFAULT_MERGED_RESULTS_CSV,
    VARIANT_DETECTS_VIOLATION,
    VARIANT_JSON,
    VARIANT_NEEDS_TACIT,
    accuracy_for_variant,
    load_benchmark,
    naive_confusion_rate,
    parse_lp_response,
    prompts_to_dataframe,
    score_pivot_dataframe,
    score_run_rows,
    write_merged_results_csv,
)
from benchmarks.simple_rate_tasks import (
    _llm_extra_api_params,
    model_slug_from_llm,
    model_slug_from_run,
)

LP_MAX_OUTPUT_TOKENS = 1024
LP_N_JOBS = 1


def _kaggle_merged_results_path() -> Path:
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working") / "lp_merged_results.csv"
    return DEFAULT_MERGED_RESULTS_CSV


def _prompt_llm(llm, prompt: str, *, max_output_tokens: int) -> str:
    return llm.prompt(
        prompt,
        reasoning="none",
        extra_api_params=_llm_extra_api_params(max_output_tokens),
    )


@kbench.task(
    store_task=False,
    description="One LP tacit-constraint item: prompt, parsed answer, and reasoning.",
)
def lp_rate_prompt_response(
    llm,
    example_id: str,
    prompt: str,
    max_output_tokens: int = LP_MAX_OUTPUT_TOKENS,
) -> dict:
    response = _prompt_llm(llm, prompt, max_output_tokens=max_output_tokens)
    item = load_benchmark()[example_id]
    parsed = parse_lp_response(response, scoring_type=item.scoring_type)
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


def evaluate_lp_rate_benchmark(
    llm,
    *,
    benchmark_path: str | Path | None = None,
    merged_results_path: str | Path | None = None,
    max_prompts: int | None = None,
    max_output_tokens: int | None = None,
    n_jobs: int = LP_N_JOBS,
):
    """Run LP benchmark prompts and score JSON + A/B audit variants."""
    benchmark_path = Path(benchmark_path or BENCHMARK_CSV)
    llms = llm if isinstance(llm, list) else [llm]
    evaluation_data = prompts_to_dataframe(benchmark_path, max_prompts=max_prompts)
    example_ids = (
        None
        if max_prompts is None
        else list(evaluation_data["example_id"])
    )
    token_cap = (
        LP_MAX_OUTPUT_TOKENS
        if max_output_tokens is None
        else max_output_tokens
    )
    runs = evaluate_prompt_task_with_resume(
        lp_rate_prompt_response,
        llm=llms,
        evaluation_data=evaluation_data,
        max_output_tokens=[token_cap],
        n_jobs=n_jobs,
    )

    run_rows = []
    for run in runs.runs:
        row = dict(run.result)
        row["model"] = model_slug_from_run(run)
        if row["model"] == "unknown" and len(llms) == 1:
            row["model"] = model_slug_from_llm(llms[0])
        run_rows.append(row)

    items = load_benchmark(benchmark_path)
    score = score_run_rows(run_rows, items=items)
    merged_path, pivot_path = write_merged_results_csv(
        run_rows,
        merged_results_path or _kaggle_merged_results_path(),
        benchmark_path=benchmark_path,
        items=items,
        score=score,
        example_ids=example_ids,
    )
    with merged_path.open(newline="", encoding="utf-8") as handle:
        merged_rows = list(csv.DictReader(handle))
    pivot = score_pivot_dataframe(merged_rows)
    naive_rate = naive_confusion_rate(run_rows, items=items)
    variant_scores = {
        VARIANT_JSON: accuracy_for_variant(run_rows, VARIANT_JSON, items=items),
        VARIANT_NEEDS_TACIT: accuracy_for_variant(
            run_rows, VARIANT_NEEDS_TACIT, items=items
        ),
        VARIANT_DETECTS_VIOLATION: accuracy_for_variant(
            run_rows, VARIANT_DETECTS_VIOLATION, items=items
        ),
    }
    return runs, score, merged_path, pivot_path, pivot, naive_rate, variant_scores


@kbench.task(
    name="lo_normative_accuracy_5",
    description=(
        "LO (linear optimization) benchmark: overall fraction of parsable "
        "answers that are keyed-correct across JSON solve prompts and tacit-"
        "constraint audits. Higher is better."
    ),
)
def lo_normative_accuracy_5(llm) -> float:
    _, score, _, _, _, _, _ = evaluate_lp_rate_benchmark(llm)
    return float(score.accuracy)


@kbench.task(
    name="lo_normative_accuracy_4",
    description=(
        "Deprecated alias of lo_normative_accuracy_5 (re-run Build task with v5)."
    ),
)
def lo_normative_accuracy_4(llm) -> float:
    return lo_normative_accuracy_5(llm)


@kbench.task(
    name="lp_needs_tacit_constraint",
    description=(
        "LO audit: fraction of parsable A/B answers that correctly say solving "
        "the problem requires unstated constraints (integrality / "
        "non-negativity). Higher is better."
    ),
)
def lp_needs_tacit_constraint(llm) -> float:
    _, _, _, _, _, _, variant_scores = evaluate_lp_rate_benchmark(llm)
    return float(variant_scores[VARIANT_NEEDS_TACIT])


@kbench.task(
    name="lp_detects_tacit_violation",
    description=(
        "LO audit: fraction of parsable A/B answers that correctly reject a "
        "proposed plan that violates an unstated constraint. Higher is better."
    ),
)
def lp_detects_tacit_violation(llm) -> float:
    _, _, _, _, _, _, variant_scores = evaluate_lp_rate_benchmark(llm)
    return float(variant_scores[VARIANT_DETECTS_VIOLATION])


@kbench.task(
    name="lo_naive_confusion",
    description=(
        "LO (linear optimization) benchmark: fraction of parsable JSON answers "
        "whose cost is within 1% of the naive stated-constraints-only optimum. "
        "Lower is better."
    ),
)
def lo_naive_confusion(llm) -> float:
    _, _, _, _, _, naive_rate, _ = evaluate_lp_rate_benchmark(llm)
    return float(naive_rate)


@kbench.task(
    name="lp_rate_normative_accuracy",
    description=(
        "Alias of lo_normative_accuracy_5 (LP / LO tacit-constraint benchmark)."
    ),
)
def lp_rate_normative_accuracy(llm) -> float:
    _, score, _, _, _, _, _ = evaluate_lp_rate_benchmark(llm)
    return float(score.accuracy)


@kbench.task(
    name="lp_rate_naive_confusion",
    description=(
        "Alias of lo_naive_confusion (LP / LO tacit-constraint benchmark)."
    ),
)
def lp_rate_naive_confusion(llm) -> float:
    _, _, _, _, _, naive_rate, _ = evaluate_lp_rate_benchmark(llm)
    return float(naive_rate)
