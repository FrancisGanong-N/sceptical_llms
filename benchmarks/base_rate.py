"""Base-rate benchmark loading, response parsing, and scoring."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

BENCHMARK_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "base_rate" / "benchmark.csv"
)
DEFAULT_MERGED_RESULTS_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "kaggle_runs"
    / "base_rate"
    / "merged_results.csv"
)

NORMATIVE_TOLERANCE_PERCENT = 1.0
LURE_TOLERANCE_PERCENT = 0.05
MC_NUMERIC_LETTERS = frozenset("ABCDE")
MC_FULL_LETTERS = frozenset("ABCDEFGH")
META_LETTERS = frozenset("FGH")

ScoringType = Literal["open", "mc_numeric", "mc_full"]
ParsedAnswerType = Literal[
    "probability",
    "meta_insufficient",
    "meta_inconsistent",
    "meta_obviously_incorrect",
    "mc_choice",
    "unparseable",
]
STANDALONE_CHOICE_PATTERN = re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])")
LABELED_CHOICE_PATTERN = re.compile(
    r"\b(?:option|choice|answer)\s*([A-H])\b",
    re.IGNORECASE,
)
PERCENT_TOKEN_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)(?:\.{2,})?\s*(?:\\%|%(?![A-Za-z])|(?:percent|percentage)(?=\s|[,.)\]]|$))",
    re.IGNORECASE,
)
DECIMAL_PROBABILITY_PATTERN = re.compile(r"(?<![\d.])(0\.\d+)(?![\d.])")
META_INSUFFICIENT_PATTERN = re.compile(r"insufficient\s+information", re.IGNORECASE)
META_INCONSISTENT_PATTERN = re.compile(r"inconsistent\s+information", re.IGNORECASE)
META_OBVIOUSLY_INCORRECT_PATTERN = re.compile(
    r"obviously\s+incorrect|provided\s+information\s+is\s+obviously\s+incorrect",
    re.IGNORECASE,
)

MERGE_RESULT_COLUMNS = (
    "model",
    "llm_response",
    "reasoning",
    "answer_line",
    "confidence_line",
    "parsed_answer_type",
    "parsed_percent",
    "parsed_choice",
    "parsed_confidence",
    "scoring_type",
    "parseable",
    "score",
)

RESPONSE_TYPE_ORDER = ("open", "mc_numeric", "mc_full")
HAS_STATISTICS_ORDER = ("probs", "no_probs")
CONDITION_COLUMN_ORDER = tuple(
    f"{response_type}_{statistics}"
    for response_type in RESPONSE_TYPE_ORDER
    for statistics in HAS_STATISTICS_ORDER
)
@dataclass(frozen=True)
class BaseRateOption:
    letter: str
    label: str
    lure: str
    percent: float | None


@dataclass(frozen=True)
class BaseRateBenchmarkItem:
    example_id: str
    vignette_name: str
    prompt: str
    problem_type: str
    intersection_size: str
    response_type: str
    has_statistics: bool
    variant: str
    normative_percent: float
    normative_choice: str
    scepticism_score_target: str
    options: tuple[BaseRateOption, ...]

    @property
    def scoring_type(self) -> ScoringType:
        return self.response_type  # type: ignore[return-value]

    @property
    def is_open(self) -> bool:
        return self.response_type == "open"

    @property
    def option_by_letter(self) -> dict[str, BaseRateOption]:
        return {option.letter: option for option in self.options}

    @property
    def lure_choices(self) -> frozenset[str]:
        if not self.normative_choice:
            return frozenset()
        return frozenset(
            option.letter
            for option in self.options
            if option.letter != self.normative_choice
        )

    @property
    def lure_percents(self) -> frozenset[float]:
        return frozenset(
            option.percent
            for option in self.options
            if option.percent is not None
            and option.letter != self.normative_choice
        )


@dataclass
class ParsedResponse:
    answer_type: ParsedAnswerType
    answer_line: str = ""
    confidence_line: str = ""
    percent: float | None = None
    choice: str | None = None
    confidence: int | None = None


@dataclass
class ExampleScore:
    example_id: str
    scoring_type: ScoringType
    answer_type: ParsedAnswerType
    answer_line: str
    confidence_line: str
    parsed_percent: float | None
    parsed_choice: str | None
    parsed_confidence: int | None
    parseable: bool
    score: bool
    model: str = "unknown"


@dataclass
class BaseRateScore:
    accuracy: float
    parse_rate: float
    n_items: int
    n_scored: int
    n_parseable: int
    examples: list[ExampleScore] = field(default_factory=list)

    @property
    def normative_accuracy(self) -> float:
        """Backwards-compatible alias for mean scepticism-target score."""
        return self.accuracy

    @property
    def bias_index(self) -> float:
        """Backwards-compatible alias for 1 - accuracy (non-matching parseable answers)."""
        return 1.0 - self.accuracy


def _parse_percent_label(label: str) -> float | None:
    if not label:
        return None
    match = re.search(r"([\d.]+)\s*%", label)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _option_columns() -> list[tuple[str, str, str]]:
    return [
        (f"option_{letter}_label", f"option_{letter}_lure", letter.upper())
        for letter in "abcdefgh"
    ]


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def split_response_lines(response: str) -> tuple[str, str]:
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    answer_line = lines[0] if lines else ""
    confidence_line = lines[1] if len(lines) > 1 else ""
    return answer_line, confidence_line


def parse_confidence_line(confidence_line: str) -> int | None:
    if not confidence_line:
        return None
    match = re.search(r"\b([1-5])\b", confidence_line)
    if not match:
        return None
    return int(match.group(1))


def _extract_percent_values(text: str) -> list[float]:
    values: list[float] = []
    for match in PERCENT_TOKEN_PATTERN.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def parse_probability_from_line(answer_line: str) -> float | None:
    if not answer_line:
        return None

    percents = _extract_percent_values(answer_line)
    if percents:
        return percents[-1]

    decimals = DECIMAL_PROBABILITY_PATTERN.findall(answer_line.lower())
    if decimals:
        value = float(decimals[-1])
        if 0 < value <= 1:
            return value * 100.0
    return None


def _parse_meta_type(answer_line: str) -> ParsedAnswerType | None:
    if META_INSUFFICIENT_PATTERN.search(answer_line):
        return "meta_insufficient"
    if META_INCONSISTENT_PATTERN.search(answer_line):
        return "meta_inconsistent"
    if META_OBVIOUSLY_INCORRECT_PATTERN.search(answer_line):
        return "meta_obviously_incorrect"
    return None


def parse_mc_choice_from_line(answer_line: str) -> str | None:
    if not answer_line:
        return None
    labeled = LABELED_CHOICE_PATTERN.findall(answer_line)
    if labeled:
        return labeled[-1].upper()
    standalone = STANDALONE_CHOICE_PATTERN.findall(answer_line)
    if standalone:
        return standalone[-1].upper()
    return None


def parse_response(response: str, *, scoring_type: ScoringType) -> ParsedResponse:
    answer_line, confidence_line = split_response_lines(response)
    confidence = parse_confidence_line(confidence_line)

    if scoring_type == "open":
        meta = _parse_meta_type(answer_line) if answer_line else None
        if meta is not None:
            return ParsedResponse(
                answer_type=meta,
                answer_line=answer_line,
                confidence_line=confidence_line,
                confidence=confidence,
            )
        percent = parse_probability_from_line(answer_line)
        if percent is not None:
            return ParsedResponse(
                answer_type="probability",
                answer_line=answer_line,
                confidence_line=confidence_line,
                percent=percent,
                confidence=confidence,
            )
        return ParsedResponse(
            answer_type="unparseable",
            answer_line=answer_line,
            confidence_line=confidence_line,
            confidence=confidence,
        )

    choice = parse_mc_choice_from_line(answer_line)
    if choice is None:
        return ParsedResponse(
            answer_type="unparseable",
            answer_line=answer_line,
            confidence_line=confidence_line,
            confidence=confidence,
        )
    return ParsedResponse(
        answer_type="mc_choice",
        answer_line=answer_line,
        confidence_line=confidence_line,
        choice=choice,
        confidence=confidence,
    )


def matches_percent_target(
    value: float,
    target: float,
    tolerance: float = NORMATIVE_TOLERANCE_PERCENT,
) -> bool:
    return abs(value - target) <= tolerance


def normative_tolerance_percent(target: float) -> float:
    """Tighter band for tiny posteriors; ±1 pp for larger ones."""
    return min(
        NORMATIVE_TOLERANCE_PERCENT,
        max(LURE_TOLERANCE_PERCENT, abs(target) * 0.5),
    )


def _item_from_row(row: dict[str, str]) -> BaseRateBenchmarkItem:
    options: list[BaseRateOption] = []
    for label_key, lure_key, letter in _option_columns():
        label = (row.get(label_key) or "").strip()
        if not label:
            continue
        options.append(
            BaseRateOption(
                letter=letter,
                label=label,
                lure=(row.get(lure_key) or "").strip(),
                percent=_parse_percent_label(label),
            )
        )
    return BaseRateBenchmarkItem(
        example_id=row["example_id"].strip(),
        vignette_name=row["vignette_name"].strip(),
        prompt=row["prompt"].strip(),
        problem_type=row["problem_type"].strip(),
        intersection_size=row["intersection_size"].strip(),
        response_type=row["response_type"].strip(),
        has_statistics=_as_bool(row["has_statistics"]),
        variant=row["variant"].strip(),
        normative_percent=float(row["normative_percent"]),
        normative_choice=(row.get("normative_choice") or "").strip().upper(),
        scepticism_score_target=(row.get("scepticism_score_target") or "").strip(),
        options=tuple(options),
    )


def load_benchmark(path: Path | str = BENCHMARK_CSV) -> dict[str, BaseRateBenchmarkItem]:
    items: dict[str, BaseRateBenchmarkItem] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = _item_from_row(row)
            items[item.example_id] = item
    return items


def load_benchmark_rows(path: Path | str = BENCHMARK_CSV) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{key: (row.get(key) or "") for key in fieldnames} for row in reader]
    return fieldnames, rows


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


def matches_scepticism_target(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
) -> bool:
    """True when the parsed answer matches ``scepticism_score_target`` for this item."""
    target = item.scepticism_score_target.strip()
    if not target:
        return False

    lowered = target.lower()
    if lowered == "n/a":
        if item.scoring_type == "mc_numeric":
            return (
                parsed.choice is not None
                and parsed.choice in MC_NUMERIC_LETTERS
                and parsed.choice == item.normative_choice
            )
        return parsed.answer_type != "unparseable"

    if lowered == "meta":
        return parsed.answer_type.startswith("meta_")

    if "|" in target:
        letters = frozenset(part.strip().upper() for part in target.split("|") if part.strip())
        return parsed.choice is not None and parsed.choice in letters

    if len(target) == 1 and target.upper() in MC_FULL_LETTERS:
        return parsed.choice is not None and parsed.choice == target.upper()

    try:
        target_percent = float(target)
    except ValueError:
        return False

    if parsed.percent is None:
        return False
    return matches_percent_target(
        parsed.percent,
        target_percent,
        tolerance=normative_tolerance_percent(target_percent),
    )


def _example_score_base(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
) -> ExampleScore:
    return ExampleScore(
        example_id=item.example_id,
        scoring_type=item.scoring_type,
        answer_type=parsed.answer_type,
        answer_line=parsed.answer_line,
        confidence_line=parsed.confidence_line,
        parsed_percent=parsed.percent,
        parsed_choice=parsed.choice,
        parsed_confidence=parsed.confidence,
        parseable=False,
        score=False,
    )


def _is_parseable(item: BaseRateBenchmarkItem, parsed: ParsedResponse) -> bool:
    if item.scoring_type == "open":
        return parsed.answer_type != "unparseable"
    return parsed.choice is not None


def score_open_example(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
    *,
    lure_percents: frozenset[float] | None = None,
) -> ExampleScore:
    del lure_percents  # kept for test/backwards-compatible call signature
    scored = _example_score_base(item, parsed)
    if not _is_parseable(item, parsed):
        return scored

    scored.parseable = True
    scored.score = matches_scepticism_target(item, parsed)
    return scored


def score_mc_numeric_example(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
) -> ExampleScore:
    scored = _example_score_base(item, parsed)
    if parsed.choice is None:
        return scored

    scored.parseable = True
    scored.score = matches_scepticism_target(item, parsed)
    return scored


def score_mc_full_example(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
) -> ExampleScore:
    scored = _example_score_base(item, parsed)
    if parsed.choice is None:
        return scored

    scored.parseable = True
    scored.score = matches_scepticism_target(item, parsed)
    return scored


def score_example(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
    *,
    items: dict[str, BaseRateBenchmarkItem] | None = None,
) -> ExampleScore:
    del items
    if item.scoring_type == "open":
        return score_open_example(item, parsed)
    if item.scoring_type == "mc_numeric":
        return score_mc_numeric_example(item, parsed)
    return score_mc_full_example(item, parsed)


def _normalize_reasoning(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return "\n---\n".join(parts)
    return str(value).strip()


def score_run_rows(
    run_rows: list[dict[str, object]],
    items: dict[str, BaseRateBenchmarkItem] | None = None,
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


def score_base_rate_responses(
    responses: dict[str, str] | list[dict[str, str]],
    items: dict[str, BaseRateBenchmarkItem] | None = None,
) -> BaseRateScore:
    if isinstance(responses, dict):
        run_rows = [
            {"example_id": example_id, "response": response}
            for example_id, response in responses.items()
        ]
    else:
        run_rows = [
            {
                "example_id": row["example_id"],
                "response": row.get("response", ""),
                "reasoning": row.get("reasoning", ""),
            }
            for row in responses
        ]
    return score_run_rows(run_rows, items=items)


def item_label(vignette_name: str, problem_type: str) -> str:
    if problem_type == "well_posed":
        return vignette_name
    return f"{vignette_name} ({problem_type})"


def condition_column(response_type: str, has_statistics: str) -> str:
    suffix = "probs" if str(has_statistics).strip().lower() == "true" else "no_probs"
    return f"{response_type}_{suffix}"


def score_pivot_dataframe(merged_rows: list[dict[str, str]]) -> "pd.DataFrame":
    """Pivot merged rows: rows=models, columns=condition, values=mean score (0/1)."""
    import pandas as pd

    if not merged_rows:
        return pd.DataFrame(columns=list(CONDITION_COLUMN_ORDER))

    frame = pd.DataFrame(merged_rows)
    frame["condition"] = frame.apply(
        lambda row: condition_column(row["response_type"], row["has_statistics"]),
        axis=1,
    )
    frame["score_value"] = frame["score"].map(lambda value: 1 if str(value).lower() == "true" else 0)

    pivot = (
        frame.groupby(["model", "condition"], as_index=False)["score_value"]
        .mean()
        .pivot(index="model", columns="condition", values="score_value")
    )
    pivot = pivot.reindex(columns=list(CONDITION_COLUMN_ORDER))
    pivot.index.name = "model"
    return pivot.sort_index().round(3)


def print_score_pivots(merged_rows: list[dict[str, str]]) -> None:
    """Print mean score by model and condition (0/1 scores averaged)."""
    pivot = score_pivot_dataframe(merged_rows)
    if pivot.empty:
        print("No merged results to pivot.", flush=True)
        return

    from collections import Counter

    scores = Counter(str(row.get("score", "")).lower() for row in merged_rows)
    print(
        "\nScore pivot (rows=model, columns=condition; "
        "cell = mean of 0/1 scepticism-target scores across vignettes):",
        flush=True,
    )
    print(
        "Score counts in merged rows:",
        ", ".join(f"{key}={value}" for key, value in sorted(scores.items()) if key),
        flush=True,
    )
    print(pivot.to_string(na_rep=""), flush=True)
    overall = pivot.mean(axis=1).round(3)
    print("\nOverall mean score by model:", flush=True)
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


def example_score_to_merge_fields(scored: ExampleScore) -> dict[str, str]:
    return {
        "parsed_answer_type": scored.answer_type,
        "parsed_percent": "" if scored.parsed_percent is None else f"{scored.parsed_percent:g}",
        "parsed_choice": scored.parsed_choice or "",
        "parsed_confidence": ""
        if scored.parsed_confidence is None
        else str(scored.parsed_confidence),
        "scoring_type": scored.scoring_type,
        "parseable": str(scored.parseable).lower(),
        "score": str(scored.score).lower(),
    }


def merge_run_results(
    run_rows: list[dict[str, object]],
    *,
    benchmark_path: Path | str = BENCHMARK_CSV,
    items: dict[str, BaseRateBenchmarkItem] | None = None,
    score: BaseRateScore | None = None,
    example_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    items = items or load_benchmark(benchmark_path)
    benchmark_fields, benchmark_rows = load_benchmark_rows(benchmark_path)
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

    full_run_rows: list[dict[str, object]] = []
    for example_id in sorted(benchmark_by_id):
        for model in models:
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
    for example_id in sorted(benchmark_by_id):
        for model in models:
            benchmark_row = dict(benchmark_by_id[example_id])
            run_row = run_by_key.get((example_id, model), {})
            scored = score_by_key[(example_id, model)]
            response = str(run_row.get("response") or "")
            answer_line, confidence_line = split_response_lines(response)
            merge_fields = {
                "model": model,
                "llm_response": response,
                "reasoning": _normalize_reasoning(run_row.get("reasoning")),
                "answer_line": answer_line,
                "confidence_line": confidence_line,
                **example_score_to_merge_fields(scored),
            }
            merged.append({**benchmark_row, **merge_fields})

    return merged


def write_merged_results_csv(
    run_rows: list[dict[str, object]],
    path: Path | str | None = None,
    *,
    benchmark_path: Path | str = BENCHMARK_CSV,
    items: dict[str, BaseRateBenchmarkItem] | None = None,
    score: BaseRateScore | None = None,
    example_ids: list[str] | None = None,
) -> Path:
    benchmark_fields, _ = load_benchmark_rows(benchmark_path)
    fieldnames = benchmark_fields + list(MERGE_RESULT_COLUMNS)
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
    pivot_path = out.with_name("base_rate_score_pivot.csv")
    write_score_pivot_csv(merged, pivot_path)
    return out, pivot_path


# Backwards-compatible aliases used in tests.
parse_mc_choice = parse_mc_choice_from_line
