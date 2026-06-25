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
ScoreOutcome = Literal["normative", "biased", "off_target", "unparseable"]

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
    "score_outcome",
    "normative",
    "biased",
    "lure_matched",
)

RESPONSE_TYPE_ORDER = ("open", "mc_numeric", "mc_full")
HAS_STATISTICS_ORDER = ("probs", "no_probs")
CONDITION_COLUMN_ORDER = tuple(
    f"{response_type}_{statistics}"
    for response_type in RESPONSE_TYPE_ORDER
    for statistics in HAS_STATISTICS_ORDER
)
SCORE_OUTCOME_LABELS = {
    "normative": "N",
    "biased": "B",
    "off_target": "O",
    "unparseable": "?",
}


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
            and not matches_percent_target(option.percent, self.normative_percent)
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
    normative_percent: float
    normative_choice: str
    parseable: bool
    score_outcome: ScoreOutcome
    normative: bool
    biased: bool
    lure_matched: str = ""
    model: str = "unknown"


@dataclass
class BaseRateScore:
    normative_accuracy: float
    bias_index: float
    parse_rate: float
    n_items: int
    n_normative: int
    n_biased: int
    n_parseable: int
    examples: list[ExampleScore] = field(default_factory=list)


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


def _mc_numeric_sibling_id(example_id: str) -> str | None:
    if "__open_" not in example_id:
        return None
    return example_id.replace("__open_", "__mc_numeric_", 1)


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


def prompts_to_dataframe(path: Path | str = BENCHMARK_CSV) -> "pd.DataFrame":
    import pandas as pd

    items = load_benchmark(path)
    rows = [
        {"example_id": item.example_id, "prompt": item.prompt}
        for item in sorted(items.values(), key=lambda item: item.example_id)
    ]
    return pd.DataFrame(rows)


def _open_lure_percents(
    item: BaseRateBenchmarkItem, items: dict[str, BaseRateBenchmarkItem]
) -> frozenset[float]:
    sibling_id = _mc_numeric_sibling_id(item.example_id)
    if sibling_id and sibling_id in items:
        return items[sibling_id].lure_percents
    return frozenset()


def _closest_lure_percent(
    percent: float, lure_percents: frozenset[float]
) -> tuple[float | None, float | None]:
    if not lure_percents:
        return None, None
    closest = min(lure_percents, key=lambda lure: abs(percent - lure))
    return closest, abs(percent - closest)


def _lure_name_for_percent(
    percent: float,
    item: BaseRateBenchmarkItem,
    lure_percents: frozenset[float],
) -> str:
    closest, _ = _closest_lure_percent(percent, lure_percents)
    if closest is None:
        return ""
    for option in item.options:
        if option.percent is not None and matches_percent_target(option.percent, closest):
            return option.lure
    return f"lure_percent:{closest:g}"


def score_open_example(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
    *,
    lure_percents: frozenset[float],
) -> ExampleScore:
    base = ExampleScore(
        example_id=item.example_id,
        scoring_type="open",
        answer_type=parsed.answer_type,
        answer_line=parsed.answer_line,
        confidence_line=parsed.confidence_line,
        parsed_percent=parsed.percent,
        parsed_choice=None,
        parsed_confidence=parsed.confidence,
        normative_percent=item.normative_percent,
        normative_choice="",
        parseable=False,
        score_outcome="unparseable",
        normative=False,
        biased=False,
    )

    if parsed.answer_type == "unparseable":
        return base

    base.parseable = True
    if parsed.answer_type.startswith("meta_"):
        base.score_outcome = "biased"
        base.biased = True
        base.lure_matched = parsed.answer_type.removeprefix("meta_")
        return base

    if parsed.percent is None:
        return base

    normative = matches_percent_target(parsed.percent, item.normative_percent)
    closest_lure, lure_distance = _closest_lure_percent(parsed.percent, lure_percents)
    biased = (
        closest_lure is not None
        and lure_distance is not None
        and matches_percent_target(parsed.percent, closest_lure)
    )
    if normative and biased:
        dist_normative = abs(parsed.percent - item.normative_percent)
        normative = dist_normative <= lure_distance
        biased = lure_distance < dist_normative

    if normative:
        base.score_outcome = "normative"
        base.normative = True
        return base
    if biased:
        base.score_outcome = "biased"
        base.biased = True
        base.lure_matched = _lure_name_for_percent(parsed.percent, item, lure_percents)
        return base

    base.score_outcome = "off_target"
    base.biased = False
    return base


def score_mc_numeric_example(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
) -> ExampleScore:
    base = ExampleScore(
        example_id=item.example_id,
        scoring_type="mc_numeric",
        answer_type=parsed.answer_type,
        answer_line=parsed.answer_line,
        confidence_line=parsed.confidence_line,
        parsed_percent=None,
        parsed_choice=parsed.choice,
        parsed_confidence=parsed.confidence,
        normative_percent=item.normative_percent,
        normative_choice=item.normative_choice,
        parseable=False,
        score_outcome="unparseable",
        normative=False,
        biased=False,
    )

    if parsed.choice is None:
        return base

    base.parseable = True
    if parsed.choice not in MC_NUMERIC_LETTERS:
        base.score_outcome = "biased"
        base.biased = True
        base.lure_matched = "out_of_range_letter"
        return base

    if parsed.choice == item.normative_choice:
        base.score_outcome = "normative"
        base.normative = True
        return base

    option = item.option_by_letter.get(parsed.choice)
    base.score_outcome = "biased"
    base.biased = True
    base.lure_matched = option.lure if option else parsed.choice
    return base


def score_mc_full_example(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
) -> ExampleScore:
    base = ExampleScore(
        example_id=item.example_id,
        scoring_type="mc_full",
        answer_type=parsed.answer_type,
        answer_line=parsed.answer_line,
        confidence_line=parsed.confidence_line,
        parsed_percent=None,
        parsed_choice=parsed.choice,
        parsed_confidence=parsed.confidence,
        normative_percent=item.normative_percent,
        normative_choice=item.normative_choice,
        parseable=False,
        score_outcome="unparseable",
        normative=False,
        biased=False,
    )

    if parsed.choice is None:
        return base

    base.parseable = True
    if parsed.choice not in MC_FULL_LETTERS:
        base.score_outcome = "biased"
        base.biased = True
        base.lure_matched = "out_of_range_letter"
        return base

    if parsed.choice == item.normative_choice:
        base.score_outcome = "normative"
        base.normative = True
        return base

    option = item.option_by_letter.get(parsed.choice)
    base.score_outcome = "biased"
    base.biased = True
    if parsed.choice in META_LETTERS:
        base.lure_matched = option.lure if option else f"meta_{parsed.choice.lower()}"
    else:
        base.lure_matched = option.lure if option else parsed.choice
    return base


def score_example(
    item: BaseRateBenchmarkItem,
    parsed: ParsedResponse,
    *,
    items: dict[str, BaseRateBenchmarkItem],
) -> ExampleScore:
    if item.scoring_type == "open":
        return score_open_example(
            item,
            parsed,
            lure_percents=_open_lure_percents(item, items),
        )
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
    n_normative = 0
    n_biased = 0
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
        scored = score_example(item, parsed, items=items)
        scored.model = model
        examples.append(scored)
        if scored.parseable:
            n_parseable += 1
            if scored.normative:
                n_normative += 1
            if scored.biased:
                n_biased += 1

    n_items = len(run_rows)
    return BaseRateScore(
        normative_accuracy=n_normative / n_parseable if n_parseable else 0.0,
        bias_index=n_biased / n_parseable if n_parseable else 0.0,
        parse_rate=n_parseable / n_items if n_items else 0.0,
        n_items=n_items,
        n_normative=n_normative,
        n_biased=n_biased,
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


def score_outcome_label(score_outcome: str) -> str:
    return SCORE_OUTCOME_LABELS.get(score_outcome, score_outcome)


def _condition_cell_scores(frame: "pd.DataFrame") -> str:
    ordered = frame.sort_values("item")
    return "".join(ordered["score"].tolist())


def score_pivot_dataframe(merged_rows: list[dict[str, str]]) -> "pd.DataFrame":
    """Pivot merged rows: rows=models, columns=condition, values=per-item score codes."""
    import pandas as pd

    if not merged_rows:
        return pd.DataFrame(columns=list(CONDITION_COLUMN_ORDER))

    frame = pd.DataFrame(merged_rows)
    frame["item"] = frame.apply(
        lambda row: item_label(row["vignette_name"], row["problem_type"]),
        axis=1,
    )
    frame["condition"] = frame.apply(
        lambda row: condition_column(row["response_type"], row["has_statistics"]),
        axis=1,
    )
    frame["score"] = frame["score_outcome"].map(score_outcome_label)

    pivot = (
        frame.groupby(["model", "condition"], group_keys=False)
        .apply(_condition_cell_scores, include_groups=False)
        .unstack("condition")
    )
    pivot = pivot.reindex(columns=list(CONDITION_COLUMN_ORDER))
    pivot = pivot.fillna("")
    pivot.index.name = "model"
    return pivot.sort_index()


def print_score_pivots(merged_rows: list[dict[str, str]]) -> None:
    """Print a pivoted score table: rows=models, columns=condition variants."""
    pivot = score_pivot_dataframe(merged_rows)
    if pivot.empty:
        print("No merged results to pivot.")
        return

    print("\nScore pivot (rows=model, columns=condition; each cell = item scores in order, N/B/O/?):")
    print(pivot.to_string())


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
        "score_outcome": scored.score_outcome,
        "normative": str(scored.normative).lower(),
        "biased": str(scored.biased).lower(),
        "lure_matched": scored.lure_matched,
    }


def merge_run_results(
    run_rows: list[dict[str, object]],
    *,
    benchmark_path: Path | str = BENCHMARK_CSV,
    items: dict[str, BaseRateBenchmarkItem] | None = None,
    score: BaseRateScore | None = None,
) -> list[dict[str, str]]:
    items = items or load_benchmark(benchmark_path)
    benchmark_fields, benchmark_rows = load_benchmark_rows(benchmark_path)
    benchmark_by_id = {row["example_id"]: row for row in benchmark_rows}
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
) -> Path:
    benchmark_fields, _ = load_benchmark_rows(benchmark_path)
    fieldnames = benchmark_fields + list(MERGE_RESULT_COLUMNS)
    merged = merge_run_results(
        run_rows,
        benchmark_path=benchmark_path,
        items=items,
        score=score,
    )
    out = Path(path or DEFAULT_MERGED_RESULTS_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print_score_pivots(merged)
    return out


# Backwards-compatible aliases used in tests.
parse_mc_choice = parse_mc_choice_from_line
