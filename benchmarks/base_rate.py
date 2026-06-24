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

NORMATIVE_TOLERANCE_PERCENT = 1.0

ResponseType = Literal[
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
    def is_open(self) -> bool:
        return self.response_type == "open"

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
    response_type: ResponseType
    percent: float | None = None
    choice: str | None = None
    confidence: int | None = None


@dataclass
class ExampleScore:
    example_id: str
    response_type: ResponseType
    parsed_percent: float | None
    parsed_choice: str | None
    normative_percent: float
    normative_choice: str
    normative: bool
    biased: bool
    parseable: bool


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
    return [(f"option_{letter}_label", f"option_{letter}_lure", letter.upper()) for letter in "abcdefgh"]


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _answer_lines(response: str) -> list[str]:
    return [line.strip() for line in response.strip().splitlines() if line.strip()]


def _answer_line(response: str) -> str:
    lines = _answer_lines(response)
    return lines[0] if lines else ""


def parse_confidence(response: str) -> int | None:
    lines = _answer_lines(response)
    if len(lines) < 2:
        return None
    match = re.search(r"\b([1-5])\b", lines[1])
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


def parse_probability(response: str) -> float | None:
    answer = _answer_line(response)
    if not answer:
        return None

    percents = _extract_percent_values(answer)
    if percents:
        return percents[-1]

    decimals = DECIMAL_PROBABILITY_PATTERN.findall(answer.lower())
    if decimals:
        value = float(decimals[-1])
        if 0 < value <= 1:
            return value * 100.0
    return None


def _parse_meta_type(answer: str) -> ResponseType | None:
    if META_INSUFFICIENT_PATTERN.search(answer):
        return "meta_insufficient"
    if META_INCONSISTENT_PATTERN.search(answer):
        return "meta_inconsistent"
    if META_OBVIOUSLY_INCORRECT_PATTERN.search(answer):
        return "meta_obviously_incorrect"
    return None


def parse_mc_choice(response: str) -> str | None:
    answer = _answer_line(response) or response.strip()
    if not answer:
        return None
    labeled = LABELED_CHOICE_PATTERN.findall(answer)
    if labeled:
        return labeled[-1].upper()
    standalone = STANDALONE_CHOICE_PATTERN.findall(answer)
    if standalone:
        return standalone[-1].upper()
    return None


def parse_response(response: str, *, response_type: str) -> ParsedResponse:
    confidence = parse_confidence(response)
    if response_type == "open":
        answer = _answer_line(response)
        meta = _parse_meta_type(answer) if answer else None
        if meta is not None:
            return ParsedResponse(response_type=meta, confidence=confidence)
        percent = parse_probability(response)
        if percent is not None:
            return ParsedResponse(
                response_type="probability",
                percent=percent,
                confidence=confidence,
            )
        return ParsedResponse(response_type="unparseable", confidence=confidence)

    choice = parse_mc_choice(response)
    if choice is None:
        return ParsedResponse(response_type="unparseable", confidence=confidence)
    return ParsedResponse(response_type="mc_choice", choice=choice, confidence=confidence)


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


def prompts_to_dataframe(path: Path | str = BENCHMARK_CSV) -> "pd.DataFrame":
    import pandas as pd

    items = load_benchmark(path)
    rows = [
        {"example_id": item.example_id, "prompt": item.prompt}
        for item in sorted(items.values(), key=lambda item: item.example_id)
    ]
    return pd.DataFrame(rows)


def _open_lure_percents(item: BaseRateBenchmarkItem, items: dict[str, BaseRateBenchmarkItem]) -> frozenset[float]:
    sibling_id = _mc_numeric_sibling_id(item.example_id)
    if sibling_id and sibling_id in items:
        return items[sibling_id].lure_percents
    return frozenset()


def _score_open_response(
    parsed: ParsedResponse,
    item: BaseRateBenchmarkItem,
    lure_percents: frozenset[float],
) -> tuple[bool, bool, bool]:
    if parsed.response_type == "unparseable":
        return False, False, False

    parseable = True
    if parsed.response_type.startswith("meta_"):
        return False, True, parseable

    if parsed.percent is None:
        return False, False, False

    normative = matches_percent_target(parsed.percent, item.normative_percent)
    biased = any(matches_percent_target(parsed.percent, lure) for lure in lure_percents)
    if normative and biased:
        dist_normative = abs(parsed.percent - item.normative_percent)
        closest_lure = min(lure_percents, key=lambda lure: abs(parsed.percent - lure))
        dist_lure = abs(parsed.percent - closest_lure)
        normative = dist_normative <= dist_lure
        biased = dist_lure < dist_normative
    return normative, biased, parseable


def _score_mc_response(parsed: ParsedResponse, item: BaseRateBenchmarkItem) -> tuple[bool, bool, bool]:
    if parsed.choice is None:
        return False, False, False
    parseable = True
    normative = parsed.choice == item.normative_choice
    biased = parsed.choice in item.lure_choices
    return normative, biased, parseable


def score_base_rate_responses(
    responses: dict[str, str] | list[dict[str, str]],
    items: dict[str, BaseRateBenchmarkItem] | None = None,
) -> BaseRateScore:
    items = items or load_benchmark()

    if isinstance(responses, list):
        normalized: dict[str, str] = {}
        for row in responses:
            example_id = row["example_id"].strip()
            normalized[example_id] = row.get("response", "")
        responses = normalized

    examples: list[ExampleScore] = []
    n_normative = 0
    n_biased = 0
    n_parseable = 0

    for example_id in sorted(responses):
        raw = responses[example_id]
        item = items[example_id]
        parsed = parse_response(raw, response_type=item.response_type)

        if item.is_open:
            lure_percents = _open_lure_percents(item, items)
            normative, biased, parseable = _score_open_response(parsed, item, lure_percents)
        else:
            normative, biased, parseable = _score_mc_response(parsed, item)

        if parseable:
            n_parseable += 1
            if normative:
                n_normative += 1
            if biased:
                n_biased += 1

        examples.append(
            ExampleScore(
                example_id=example_id,
                response_type=parsed.response_type,
                parsed_percent=parsed.percent,
                parsed_choice=parsed.choice,
                normative_percent=item.normative_percent,
                normative_choice=item.normative_choice,
                normative=normative,
                biased=biased,
                parseable=parseable,
            )
        )

    n_items = len(responses)
    normative_accuracy = n_normative / n_parseable if n_parseable else 0.0
    bias_index = n_biased / n_parseable if n_parseable else 0.0
    parse_rate = n_parseable / n_items if n_items else 0.0

    return BaseRateScore(
        normative_accuracy=normative_accuracy,
        bias_index=bias_index,
        parse_rate=parse_rate,
        n_items=n_items,
        n_normative=n_normative,
        n_biased=n_biased,
        n_parseable=n_parseable,
        examples=examples,
    )
