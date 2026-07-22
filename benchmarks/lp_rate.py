"""LP implicit-constraint benchmark: JSON solution+cost scoring.

Each item asks for a JSON object ``{"solution": {...}, "cost": <number>}``.
A parseable ``cost`` scores correct when it is within 1% relative of
``true_objective``. Naive-LP confusion is the fraction of parseable answers
within 1% of ``naive_objective``.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks import base_rate
from benchmarks.base_rate import (
    MERGE_RESULT_COLUMNS,
    BaseRateScore,
    ExampleScore,
    ParsedResponse,
)

__all__ = [
    "BENCHMARK_CSV",
    "OBJECTIVE_TOLERANCE_RELATIVE",
    "parse_lp_json",
    "parse_lp_response",
    "load_benchmark",
    "matches_true_objective",
    "matches_naive_lp",
    "naive_confusion_rate",
    "score_run_rows",
    "write_merged_results_csv",
]

if TYPE_CHECKING:
    import pandas as pd

BENCHMARK_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "lp" / "benchmark.csv"
)
DEFAULT_MERGED_RESULTS_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "kaggle_runs"
    / "lp"
    / "merged_results.csv"
)

OBJECTIVE_TOLERANCE_RELATIVE = 0.01
CONDITION_COLUMN_ORDER = ("json",)
LP_MERGE_EXTRA_COLUMNS = (
    "naive_lp_confusion",
    "parsed_objective",
    "parsed_solution",
)

JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class LpBenchmarkItem:
    example_id: str
    vignette_name: str
    prompt: str
    problem_type: str
    intersection_size: str
    response_type: str
    has_statistics: bool
    variant: str
    failure_mode: str
    true_objective: float | None
    naive_objective: float | None
    true_solution: dict[str, float]
    solution_keys: tuple[str, ...]
    objective_name: str
    normative_percent: float
    normative_choice: str
    scepticism_required: bool
    scepticism_score_target: str
    normative_type: str

    @property
    def scoring_type(self) -> str:
        return "json"


@dataclass
class ParsedLpResponse:
    answer_line: str = ""
    cost: float | None = None
    solution: dict[str, Any] | None = None
    raw_json: dict[str, Any] | None = None
    parseable: bool = False

    @property
    def percent(self) -> float | None:
        """Alias used by shared merge helpers that expect a numeric parse."""
        return self.cost


def within_relative_tolerance(
    value: float,
    target: float,
    *,
    relative: float = OBJECTIVE_TOLERANCE_RELATIVE,
) -> bool:
    if target == 0:
        return abs(value) <= relative
    return abs(value - target) <= relative * abs(target)


def _parse_optional_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text or text.lower() in {"unbounded", "n/a", "na"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _first_balanced_json_object(text: str) -> str | None:
    """Return the first top-level ``{...}`` span, ignoring trailing prose."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None
    candidates: list[str] = []
    fenced = JSON_BLOCK_PATTERN.findall(text)
    candidates.extend(fenced)
    balanced = _first_balanced_json_object(text)
    if balanced is not None:
        candidates.append(balanced)
    stripped = text.strip()
    if stripped.startswith("{"):
        first_line = stripped.splitlines()[0].strip()
        candidates.append(first_line)
        candidates.append(stripped)
    match = JSON_OBJECT_PATTERN.search(text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _cost_from_payload(payload: dict[str, Any]) -> float | None:
    for key in ("cost", "total_cost", "objective", "total"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace("$", "").replace(",", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                continue
    return None


def _solution_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    solution = payload.get("solution")
    if isinstance(solution, dict):
        return solution
    return None


def parse_lp_json(text: str) -> ParsedLpResponse:
    payload = _extract_json_object(text)
    if payload is None:
        return ParsedLpResponse(answer_line=text.strip(), parseable=False)
    cost = _cost_from_payload(payload)
    solution = _solution_from_payload(payload)
    return ParsedLpResponse(
        answer_line=text.strip(),
        cost=cost,
        solution=solution,
        raw_json=payload,
        parseable=cost is not None,
    )


def parse_lp_response(response: str, *, scoring_type: str = "json") -> ParsedResponse:
    """Adapter returning a base_rate ParsedResponse for task plumbing."""
    del scoring_type
    parsed = parse_lp_json(response)
    return ParsedResponse(
        answer_type="probability" if parsed.parseable else "unparseable",
        answer_line=parsed.answer_line,
        percent=parsed.cost,
        percent_candidates=() if parsed.cost is None else (parsed.cost,),
    )


def _item_from_row(row: dict[str, str]) -> LpBenchmarkItem:
    true_objective = _parse_optional_float(row.get("true_objective", ""))
    normative_percent = true_objective if true_objective is not None else 0.0
    if row.get("normative_percent", "").strip():
        try:
            normative_percent = float(row["normative_percent"])
        except ValueError:
            pass
    true_solution_raw = (row.get("true_solution") or "").strip()
    true_solution: dict[str, float] = {}
    if true_solution_raw:
        loaded = json.loads(true_solution_raw)
        true_solution = {str(k): float(v) for k, v in loaded.items()}
    keys_raw = (row.get("solution_keys") or "").strip()
    solution_keys = tuple(part for part in keys_raw.split(",") if part)
    return LpBenchmarkItem(
        example_id=row["example_id"].strip(),
        vignette_name=row["vignette_name"].strip(),
        prompt=row["prompt"].strip(),
        problem_type=row["problem_type"].strip(),
        intersection_size=(row.get("intersection_size") or "").strip(),
        response_type=row["response_type"].strip(),
        has_statistics=_as_bool(row["has_statistics"]),
        variant=row["variant"].strip(),
        failure_mode=(row.get("failure_mode") or "").strip(),
        true_objective=true_objective,
        naive_objective=_parse_optional_float(row.get("naive_objective", "")),
        true_solution=true_solution,
        solution_keys=solution_keys,
        objective_name=(row.get("objective_name") or "").strip(),
        normative_percent=normative_percent,
        normative_choice=(row.get("normative_choice") or "").strip().upper(),
        scepticism_required=_as_bool(row.get("scepticism_required", "")),
        scepticism_score_target=(row.get("scepticism_score_target") or "").strip(),
        normative_type=(row.get("normative") or "").strip(),
    )


def load_benchmark(
    path: Path | str = BENCHMARK_CSV,
) -> dict[str, LpBenchmarkItem]:
    items: dict[str, LpBenchmarkItem] = {}
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


def matches_true_objective(
    item: LpBenchmarkItem,
    parsed: ParsedLpResponse,
) -> bool:
    if item.true_objective is None or parsed.cost is None:
        return False
    return within_relative_tolerance(parsed.cost, item.true_objective)


def matches_naive_lp(
    item: LpBenchmarkItem,
    parsed: ParsedLpResponse,
) -> bool:
    if item.naive_objective is None or parsed.cost is None:
        return False
    if item.true_objective is not None and within_relative_tolerance(
        item.naive_objective, item.true_objective
    ):
        return False
    return within_relative_tolerance(parsed.cost, item.naive_objective)


def score_lp_example(item: LpBenchmarkItem, parsed: ParsedLpResponse) -> ExampleScore:
    return ExampleScore(
        example_id=item.example_id,
        scoring_type="open",
        answer_type="probability" if parsed.parseable else "unparseable",
        answer_line=parsed.answer_line,
        confidence_line="",
        comment_line="",
        parsed_percent=parsed.cost,
        parsed_choice=None,
        parsed_confidence=None,
        parseable=parsed.parseable,
        score=parsed.parseable and matches_true_objective(item, parsed),
    )


def score_run_rows(
    run_rows: list[dict[str, object]],
    items: dict[str, LpBenchmarkItem] | None = None,
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
        parsed = parse_lp_json(response)
        scored = score_lp_example(item, parsed)
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


def naive_confusion_rate(
    run_rows: list[dict[str, object]],
    items: dict[str, LpBenchmarkItem] | None = None,
) -> float:
    items = items or load_benchmark()
    n_parseable = 0
    n_naive = 0
    for row in run_rows:
        example_id = str(row["example_id"]).strip()
        item = items[example_id]
        parsed = parse_lp_json(str(row.get("response") or ""))
        if not parsed.parseable:
            continue
        n_parseable += 1
        if matches_naive_lp(item, parsed):
            n_naive += 1
    return n_naive / n_parseable if n_parseable else 0.0


def example_score_to_merge_fields(
    scored: ExampleScore,
    *,
    naive: bool,
    parsed: ParsedLpResponse,
) -> dict[str, str]:
    fields = base_rate.example_score_to_merge_fields(scored)
    fields["naive_lp_confusion"] = str(naive).lower()
    fields["parsed_objective"] = (
        "" if scored.parsed_percent is None else f"{scored.parsed_percent:g}"
    )
    fields["parsed_solution"] = (
        "" if parsed.solution is None else json.dumps(parsed.solution, sort_keys=True)
    )
    return fields


def merge_run_results(
    run_rows: list[dict[str, object]],
    *,
    benchmark_path: Path | str = BENCHMARK_CSV,
    items: dict[str, LpBenchmarkItem] | None = None,
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

    if not pair_keys:
        raise ValueError(
            "No run rows match the current benchmark example_ids. "
            "Download fresh Kaggle runs for the updated benchmark "
            f"({benchmark_path})."
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
    score_by_key = {
        (example.example_id, example.model): example for example in score.examples
    }

    merged: list[dict[str, str]] = []
    for example_id, model in pair_keys:
        benchmark_row = dict(benchmark_by_id[example_id])
        run_row = run_by_key.get((example_id, model), {})
        scored = score_by_key[(example_id, model)]
        item = items[example_id]
        response = str(run_row.get("response") or "")
        parsed = parse_lp_json(response)
        naive = matches_naive_lp(item, parsed)
        merge_fields = {
            "model": model,
            "llm_response": response,
            "reasoning": base_rate._normalize_reasoning(run_row.get("reasoning")),
            "answer_line": scored.answer_line,
            "confidence_line": scored.confidence_line,
            **example_score_to_merge_fields(scored, naive=naive, parsed=parsed),
        }
        merged.append({**benchmark_row, **merge_fields})

    return merged


def write_merged_results_csv(
    run_rows: list[dict[str, object]],
    path: Path | str | None = None,
    *,
    benchmark_path: Path | str = BENCHMARK_CSV,
    items: dict[str, LpBenchmarkItem] | None = None,
    score: BaseRateScore | None = None,
    example_ids: list[str] | None = None,
) -> tuple[Path, Path]:
    benchmark_fields, _ = load_benchmark_rows(benchmark_path)
    fieldnames = benchmark_fields + list(MERGE_RESULT_COLUMNS) + list(
        LP_MERGE_EXTRA_COLUMNS
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
    pivot_path = out.with_name("lp_rate_score_pivot.csv")
    write_score_pivot_csv(merged, pivot_path)
    return out, pivot_path


def condition_column(
    response_type: str,
    has_statistics: str | None = None,
    *,
    variant: str | None = None,
) -> str:
    del has_statistics
    if variant:
        return variant
    return response_type


def score_pivot_dataframe(merged_rows: list[dict[str, str]]) -> "pd.DataFrame":
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


def print_score_pivots(merged_rows: list[dict[str, str]]) -> None:
    pivot = score_pivot_dataframe(merged_rows)
    if pivot.empty:
        print("No merged results to pivot.", flush=True)
        return

    from collections import Counter

    scores = Counter(str(row.get("score", "")).lower() for row in merged_rows)
    naive = Counter(
        str(row.get("naive_lp_confusion", "")).lower() for row in merged_rows
    )
    print(
        "\nScore pivot (rows=model, columns=variant; "
        "cell = mean of 0/1 keyed scores):",
        flush=True,
    )
    print(
        "Score counts:",
        ", ".join(f"{key}={value}" for key, value in sorted(scores.items()) if key),
        flush=True,
    )
    print(
        "Naive LP confusion counts:",
        ", ".join(f"{key}={value}" for key, value in sorted(naive.items()) if key),
        flush=True,
    )
    print(pivot.to_string(na_rep=""), flush=True)
    overall = pivot.mean(axis=1).round(3)
    print("\nOverall mean keyed score by model:", flush=True)
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
