"""Simple two-path base-rate benchmark (C, D -> T; estimate P(C|T))."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks import base_rate
from benchmarks.base_rate import (
    MERGE_RESULT_COLUMNS,
    NUMERIC_SCORE_TOLERANCE_PERCENT,
    BaseRateBenchmarkItem,
    BaseRateOption,
    BaseRateScore,
    ExampleScore,
    ParsedResponse,
    matches_percent_target,
    matches_scepticism_target,
    open_scored_percent,
    parse_response,
    score_example,
)

__all__ = [
    "BENCHMARK_CSV",
    "PATH_C_LURE_NAME",
    "parse_response",
    "load_benchmark",
    "matches_path_c_confusion",
    "path_c_confusion_rate",
    "score_run_rows",
    "write_merged_results_csv",
]

if TYPE_CHECKING:
    import pandas as pd

BENCHMARK_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "simple" / "benchmark.csv"
)
DEFAULT_MERGED_RESULTS_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "kaggle_runs"
    / "simple"
    / "merged_results.csv"
)

PATH_C_LURE_NAME = "P(T|C) confusion"
CONDITION_COLUMN_ORDER = ("open_probs", "mc_numeric_probs", "mc_full_probs")
SIMPLE_MERGE_EXTRA_COLUMNS = ("path_c_confusion",)


@dataclass(frozen=True)
class SimpleRateBenchmarkItem(BaseRateBenchmarkItem):
    p_t_given_c: float
    p_t_given_d: float


def path_c_percent(item: BaseRateBenchmarkItem) -> float:
    if isinstance(item, SimpleRateBenchmarkItem):
        return item.p_t_given_c * 100.0
    return float(getattr(item, "p_t_given_c", 0.0)) * 100.0


def path_c_letter(item: BaseRateBenchmarkItem) -> str | None:
    for option in item.options:
        if option.lure == PATH_C_LURE_NAME:
            return option.letter
    return None


def matches_path_c_confusion(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
) -> bool:
    """True when the final answer matches the P(T|C) lure (inverse conditional)."""
    if item.scoring_type == "open":
        final = open_scored_percent(parsed)
        if final is None:
            return False
        return base_rate.matches_percent_target(final, path_c_percent(item))

    letter = path_c_letter(item)
    if letter is not None and parsed.choice == letter:
        return True
    chosen = item.option_by_letter.get(parsed.choice or "")
    if chosen is not None and chosen.percent is not None:
        return matches_percent_target(chosen.percent, path_c_percent(item))
    return False


def _item_from_row(row: dict[str, str]) -> SimpleRateBenchmarkItem:
    base = base_rate._item_from_row(row)
    response_type = base.response_type
    if response_type == "mc":
        response_type = "mc_numeric"
    return SimpleRateBenchmarkItem(
        example_id=base.example_id,
        vignette_name=base.vignette_name,
        prompt=base.prompt,
        problem_type=base.problem_type,
        intersection_size=base.intersection_size,
        response_type=response_type,
        has_statistics=base.has_statistics,
        variant=base.variant,
        normative_percent=base.normative_percent,
        normative_choice=base.normative_choice,
        scepticism_required=base.scepticism_required,
        scepticism_score_target=base.scepticism_score_target,
        normative_type=base.normative_type,
        options=base.options,
        p_t_given_c=float(row["p_t_given_c"]),
        p_t_given_d=float(row["p_t_given_d"]),
    )


def load_benchmark(
    path: Path | str = BENCHMARK_CSV,
) -> dict[str, SimpleRateBenchmarkItem]:
    items: dict[str, SimpleRateBenchmarkItem] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = _item_from_row(row)
            items[item.example_id] = item
    return items


def load_benchmark_rows(
    path: Path | str = BENCHMARK_CSV,
) -> tuple[list[str], list[dict[str, str]]]:
    return base_rate.load_benchmark_rows(path)


def prompts_to_dataframe(
    path: Path | str = BENCHMARK_CSV,
    *,
    max_prompts: int | None = None,
) -> "pd.DataFrame":
    import pandas as pd

    items = load_benchmark(path)
    rows = [
        {"example_id": item.example_id, "prompt": item.prompt}
        for item in sorted(items.values(), key=lambda item: item.example_id)
    ]
    if max_prompts is not None:
        rows = rows[:max_prompts]
    return pd.DataFrame(rows)


def example_score_to_merge_fields(scored: ExampleScore, *, path_c: bool) -> dict[str, str]:
    fields = base_rate.example_score_to_merge_fields(scored)
    fields["path_c_confusion"] = str(path_c).lower()
    return fields


def score_run_rows(
    run_rows: list[dict[str, object]],
    items: dict[str, SimpleRateBenchmarkItem] | None = None,
) -> BaseRateScore:
    items = items or load_benchmark()
    examples: list[ExampleScore] = []
    n_scored = 0
    n_parseable = 0

    for row in sorted(
        run_rows,
        key=lambda entry: (str(entry["example_id"]), str(entry.get("model") or "")),
    ):
        example_id = str(row["example_id"]).strip()
        model = str(row.get("model") or "unknown").strip()
        item = items[example_id]
        response = str(row.get("response") or "")
        parsed = parse_response(response, scoring_type=item.scoring_type)
        scored = score_example(item, parsed)
        scored.model = model
        examples.append(scored)
        if scored.parseable:
            n_parseable += 1
            if scored.score:
                n_scored += 1

    n_items = len(run_rows)
    return BaseRateScore(
        accuracy=n_scored / n_parseable if n_parseable else 0.0,
        parse_rate=n_parseable / n_items if n_items else 0.0,
        n_items=n_items,
        n_scored=n_scored,
        n_parseable=n_parseable,
        examples=examples,
    )


def path_c_confusion_rate(
    run_rows: list[dict[str, object]],
    items: dict[str, SimpleRateBenchmarkItem] | None = None,
) -> float:
    """Fraction of parseable answers matching the P(T|C) lure."""
    items = items or load_benchmark()
    n_parseable = 0
    n_path_c = 0
    for row in run_rows:
        example_id = str(row["example_id"]).strip()
        item = items[example_id]
        parsed = parse_response(
            str(row.get("response") or ""),
            scoring_type=item.scoring_type,
        )
        if parsed.answer_type == "unparseable" and parsed.choice is None:
            continue
        if item.scoring_type == "open" and parsed.answer_type == "unparseable":
            continue
        if item.scoring_type != "open" and parsed.choice is None:
            continue
        n_parseable += 1
        if matches_path_c_confusion(item, parsed):
            n_path_c += 1
    return n_path_c / n_parseable if n_parseable else 0.0


def merge_run_results(
    run_rows: list[dict[str, object]],
    *,
    benchmark_path: Path | str = BENCHMARK_CSV,
    items: dict[str, SimpleRateBenchmarkItem] | None = None,
    score: BaseRateScore | None = None,
    example_ids: list[str] | None = None,
    fill_missing: bool = True,
) -> list[dict[str, str]]:
    items = items or load_benchmark(benchmark_path)
    _, benchmark_rows = load_benchmark_rows(benchmark_path)
    benchmark_by_id = {row["example_id"]: row for row in benchmark_rows}
    if example_ids is not None:
        benchmark_by_id = {
            example_id: benchmark_by_id[example_id]
            for example_id in example_ids
            if example_id in benchmark_by_id
        }
    models = sorted(
        {str(row.get("model")).strip() for row in run_rows if row.get("model")}
    )
    if not models:
        models = ["unknown"]

    run_by_key = {
        (str(row["example_id"]).strip(), str(row.get("model") or "unknown").strip()): row
        for row in run_rows
    }

    if fill_missing:
        pair_keys = [
            (example_id, model)
            for example_id in sorted(benchmark_by_id)
            for model in models
        ]
    else:
        pair_keys = sorted(
            (example_id, model)
            for example_id, model in run_by_key
            if example_id in benchmark_by_id
        )

    full_run_rows: list[dict[str, object]] = []
    for example_id, model in pair_keys:
        run_row = run_by_key.get((example_id, model), {})
        full_run_rows.append(
            {
                "example_id": example_id,
                "response": run_row.get("response", ""),
                "reasoning": run_row.get("reasoning", ""),
                "model": model,
            }
        )

    score = score or score_run_rows(full_run_rows, items=items)
    score_by_key = {(example.example_id, example.model): example for example in score.examples}

    merged: list[dict[str, str]] = []
    for example_id, model in pair_keys:
        benchmark_row = dict(benchmark_by_id[example_id])
        run_row = run_by_key.get((example_id, model), {})
        scored = score_by_key[(example_id, model)]
        item = items[example_id]
        response = str(run_row.get("response") or "")
        parsed = parse_response(response, scoring_type=item.scoring_type)
        path_c = matches_path_c_confusion(item, parsed)
        merge_fields = {
            "model": model,
            "llm_response": response,
            "reasoning": base_rate._normalize_reasoning(run_row.get("reasoning")),
            "answer_line": scored.answer_line,
            "confidence_line": scored.confidence_line,
            **example_score_to_merge_fields(scored, path_c=path_c),
        }
        merged.append({**benchmark_row, **merge_fields})

    return merged


def write_merged_results_csv(
    run_rows: list[dict[str, object]],
    path: Path | str | None = None,
    *,
    benchmark_path: Path | str = BENCHMARK_CSV,
    items: dict[str, SimpleRateBenchmarkItem] | None = None,
    score: BaseRateScore | None = None,
    example_ids: list[str] | None = None,
) -> tuple[Path, Path]:
    benchmark_fields, _ = load_benchmark_rows(benchmark_path)
    fieldnames = benchmark_fields + list(MERGE_RESULT_COLUMNS) + list(
        SIMPLE_MERGE_EXTRA_COLUMNS
    )
    merged = merge_run_results(
        run_rows,
        benchmark_path=benchmark_path,
        items=items,
        score=score,
        example_ids=example_ids,
    )
    out = Path(path or DEFAULT_MERGED_RESULTS_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print_score_pivots(merged)
    pivot_path = out.with_name("simple_rate_score_pivot.csv")
    write_score_pivot_csv(merged, pivot_path)
    return out, pivot_path


def print_score_pivots(merged_rows: list[dict[str, str]]) -> None:
    """Print mean normative score and P(T|C) confusion rate by model and condition."""
    pivot = score_pivot_dataframe(merged_rows)
    if pivot.empty:
        print("No merged results to pivot.", flush=True)
        return

    from collections import Counter

    scores = Counter(str(row.get("score", "")).lower() for row in merged_rows)
    path_c = Counter(str(row.get("path_c_confusion", "")).lower() for row in merged_rows)
    print(
        "\nScore pivot (rows=model, columns=condition; "
        "cell = mean of 0/1 normative P(C|T) scores):",
        flush=True,
    )
    print(
        "Score counts:",
        ", ".join(f"{key}={value}" for key, value in sorted(scores.items()) if key),
        flush=True,
    )
    print(
        "P(T|C) confusion counts:",
        ", ".join(f"{key}={value}" for key, value in sorted(path_c.items()) if key),
        flush=True,
    )
    print(pivot.to_string(na_rep=""), flush=True)
    overall = pivot.mean(axis=1).round(3)
    print("\nOverall mean normative score by model:", flush=True)
    for model, value in overall.items():
        print(f"  {model}: {value:.3f}", flush=True)


def write_score_pivot_csv(
    merged_rows: list[dict[str, str]],
    path: Path | str,
) -> Path:
    pivot = score_pivot_dataframe(merged_rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out)
    return out


def condition_column(
    response_type: str,
    has_statistics: str | None = None,
    *,
    variant: str | None = None,
) -> str:
    del has_statistics
    if variant:
        return variant
    if response_type == "mc":
        return "mc_numeric_probs"
    return f"{response_type}_probs"


def score_pivot_dataframe(merged_rows: list[dict[str, str]]) -> "pd.DataFrame":
    """Pivot merged rows: rows=models, columns=variant, values=mean normative score."""
    import pandas as pd

    if not merged_rows:
        return pd.DataFrame(columns=list(CONDITION_COLUMN_ORDER))

    frame = pd.DataFrame(merged_rows)
    frame["condition"] = frame.apply(
        lambda row: condition_column(
            row["response_type"],
            row.get("has_statistics"),
            variant=(row.get("variant") or "").strip() or None,
        ),
        axis=1,
    )
    frame["score_value"] = frame["score"].map(
        lambda value: 1 if str(value).lower() == "true" else 0
    )

    pivot = (
        frame.groupby(["model", "condition"], as_index=False)["score_value"]
        .mean()
        .pivot(index="model", columns="condition", values="score_value")
    )
    pivot = pivot.reindex(columns=list(CONDITION_COLUMN_ORDER))
    pivot.index.name = "model"
    return pivot.sort_index().round(3)
