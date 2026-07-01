#!/usr/bin/env python3
"""Build simple two-path base-rate prompts (C, D -> T; estimate P(C|T))."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_base_rate_prompts import (  # noqa: E402
    INTRO,
    Vignette,
    _is_california_voter,
    _is_covid_vaccinated,
    _is_diabetic_adult,
    _is_first_generation_student,
    _is_public_secondary_teacher,
    _load_overlap,
    _load_two_cause,
    _mc_letter_list,
    _mc_rounded_label,
    _pct,
    _role_phrase,
    _rounded_mc_percent,
    _setting_phrase,
    _t_event,
    _t_event_in_question,
    slug,
)


def _simple_pct(value: float) -> str:
    if value < 0.01:
        return f"{value * 100:.2f}%"
    return _pct(value)

OUT_DIR = ROOT / "data" / "simple"
VARIANTS = ("open_probs", "mc_numeric_probs")

SIMPLE_LURE_KEYS = ("product", "path_d", "path_c", "normative", "p_d")
SIMPLE_RESERVE_LURE_KEYS = ("p_c",)
SIMPLE_LURE_NAMES = {
    "normative": "Bayes P(C|T)",
    "path_c": "P(T|C) confusion",
    "path_d": "P(T|D) confusion",
    "product": "P(T|C)*P(T|D) confusion",
    "p_c": "P(C) confusion",
    "p_d": "P(D) confusion",
}

BENCHMARK_CONDITION_FIELDS = (
    "example_id",
    "vignette_name",
    "problem_type",
    "intersection_size",
    "response_type",
    "has_statistics",
    "variant",
    "prompt",
)

BENCHMARK_SCORING_FIELDS = (
    "well_posed",
    "normative",
    "p_c_and_d_given_a",
    "p_c",
    "p_d",
    "p_t_given_c",
    "p_t_given_d",
    "normative_choice",
    "normative_percent",
    "normative_open",
    "confidence_required",
    "numeric_score_percent",
    "numeric_score_choice",
    "scepticism_required",
    "scepticism_score_target",
    *[f"option_{c}_label" for c in "abcde"],
    *[f"option_{c}_lure" for c in "abcde"],
    *[f"option_{c}_label" for c in "fgh"],
    *[f"option_{c}_lure" for c in "fgh"],
)

BENCHMARK_FIELDS = BENCHMARK_CONDITION_FIELDS + BENCHMARK_SCORING_FIELDS

ITEM_FIELDS = [
    "example_id",
    "vignette_name",
    "variant",
    "well_posed",
    "normative",
    "p_c_and_d_given_a",
    "p_c",
    "p_d",
    "p_t_given_c",
    "p_t_given_d",
    "response_type",
    "normative_choice",
    "normative_percent",
    "normative_open",
    "confidence_required",
    "numeric_score_percent",
    "numeric_score_choice",
    "scepticism_required",
    "scepticism_score_target",
    *[f"option_{c}_label" for c in "abcde"],
    *[f"option_{c}_lure" for c in "abcde"],
    *[f"option_{c}_label" for c in "fgh"],
    *[f"option_{c}_lure" for c in "fgh"],
]


@dataclass(frozen=True)
class SimpleVignette:
    """Two-path model: C = old A∧old C, D = old A∧old D."""

    name: str
    universe: str
    old_a: str
    old_c: str
    old_d: str
    t: str
    p_c: float
    p_d: float
    s_c: float
    s_d: float
    intersection_size: str

    @property
    def c_label(self) -> str:
        return _entity_label(self.old_c)

    @property
    def d_label(self) -> str:
        return _entity_label(self.old_d)

    @property
    def c_group(self) -> str:
        return _entity_group(self.old_c)

    @property
    def d_group(self) -> str:
        return _entity_group(self.old_d)

    def posterior_c(self) -> float:
        num = self.s_c * self.p_c
        den = num + self.s_d * self.p_d
        return num / den if den else 0.0

    def posterior_d(self) -> float:
        num = self.s_d * self.p_d
        den = self.s_c * self.p_c + num
        return num / den if den else 0.0

    def question_target_subtype(self) -> str:
        if self.name == "CA Trump voter":
            return self.old_d
        return self.old_c

    def target_posterior(self) -> float:
        if self.name == "CA Trump voter":
            return self.posterior_d()
        return self.posterior_c()

    def lure_percents(self) -> dict[str, float]:
        norm = self.target_posterior() * 100
        return {
            "normative": norm,
            "path_c": self.s_c * 100,
            "path_d": self.s_d * 100,
            "product": self.s_c * self.s_d * 100,
            "p_c": self.p_c * 100,
            "p_d": self.p_d * 100,
        }

    def example_prefix(self) -> str:
        return slug(self.name)


def _entity_phrase(subtype: str) -> str:
    """Short prose phrase for a subtype-under-A description."""
    head = subtype.split(":")[0].strip()
    if not head:
        return subtype.strip()
    if re.search(r"\s+among\s+", head, re.I):
        head = re.split(r"\s+among\s+", head, maxsplit=1, flags=re.I)[0]
    head = re.sub(r"\s*\(.*", "", head).strip()
    if re.search(r"^physician", head, re.I):
        return "physicians"
    if re.search(r"non-physician health care professional", head, re.I):
        return "health care professionals who are not physicians"
    s = head[0].lower() + head[1:]
    s = re.sub(r"\bus\b", "US", s)
    if re.search(r"^uses insulin", s, re.I):
        return "use insulin"
    if re.search(r"^has obesity", s, re.I):
        return "are obese"
    if re.search(r"STEM field of study", head, re.I):
        return "study STEM"
    if re.search(r"employed while enrolled", head, re.I):
        return "employed while enrolled"
    if re.search(r"southern california registrant", head, re.I):
        return "voters registered in Southern California"
    if re.search(r"other california registrant", head, re.I):
        return "other California voters"
    if re.search(r"main teaching assignment is English", head, re.I):
        return "teach English as their primary assignment"
    if re.search(r"bachelor.*major field is English", head, re.I):
        return "have a bachelor's degree in English"
    if s.startswith("uses "):
        return s.replace("uses ", "use ", 1)
    if s.startswith("has "):
        return s.replace("has ", "have ", 1)
    return s


def _entity_label(subtype: str) -> str:
    return _entity_phrase(subtype)


def _entity_group(subtype: str) -> str:
    phrase = _entity_phrase(subtype)
    if phrase in {"voters registered in Southern California", "other California voters"}:
        return phrase
    if phrase.startswith(("use ", "have ", "are ")):
        return phrase
    if phrase.endswith("s"):
        return phrase
    if phrase.endswith("y"):
        return phrase[:-1] + "ies"
    return phrase + "s"


def _entity_share_clause(subtype: str, pct: float) -> str:
    phrase = _entity_phrase(subtype)
    if phrase == "study STEM":
        return f"{_simple_pct(pct)} study STEM"
    if phrase == "employed while enrolled":
        return f"{_simple_pct(pct)} are employed while enrolled"
    if phrase == "teach English as their primary assignment":
        return f"{_simple_pct(pct)} teach English as their primary assignment"
    if phrase == "have a bachelor's degree in English":
        return f"{_simple_pct(pct)} have a bachelor's degree in English"
    if phrase.startswith(("use ", "are ")):
        return f"{_simple_pct(pct)} {phrase}"
    if phrase.endswith("members"):
        return f"{_simple_pct(pct)} are {phrase}"
    return f"{_simple_pct(pct)} are {_entity_group(subtype)}"


def _entity_among_group(subtype: str) -> str:
    phrase = _entity_phrase(subtype)
    if phrase == "study STEM":
        return "those who studied STEM"
    if phrase == "employed while enrolled":
        return "those employed while enrolled"
    if phrase == "teach English as their primary assignment":
        return "those who teach English as their primary assignment"
    if phrase == "have a bachelor's degree in English":
        return "those who have a bachelor's degree in English"
    if phrase.startswith(("use ", "have ", "are ")):
        return f"those who {phrase}"
    return _entity_group(subtype)


def _entity_answer_clause(subtype: str) -> str:
    phrase = _entity_phrase(subtype)
    if phrase == "study STEM":
        return "studied STEM"
    if phrase == "teach English as their primary assignment":
        return "were an English teacher"
    if phrase.startswith(("use ", "are ")):
        return phrase
    return f"were {_entity_group(subtype)}"


def _singularize_entity_group(group: str) -> str:
    if group.endswith(" officers"):
        return group.removesuffix(" officers") + " officer"
    if group.endswith(" members"):
        return group.removesuffix(" members") + " member"
    if group.endswith(" drivers"):
        return group.removesuffix(" drivers") + " driver"
    if group.endswith(" residents"):
        return group.removesuffix(" residents") + " resident"
    if group.endswith(" voters"):
        return group.removesuffix(" voters") + " voter"
    if group == "physicians":
        return "physician"
    if group.endswith("s") and not group.endswith("ss"):
        return group[:-1]
    return group


def _article_for(phrase: str) -> str:
    if phrase.upper().startswith("US "):
        return "a"
    first = phrase[:1].lower()
    if first in "aeiou":
        return "an"
    return "a"


def _entity_question_predicate(subtype: str) -> str:
    phrase = _entity_phrase(subtype)
    if phrase == "voters registered in Southern California":
        return "lives in Southern California"
    if phrase == "study STEM":
        return "studied STEM"
    if phrase == "teach English as their primary assignment":
        return "is an English teacher"
    if phrase.startswith("are "):
        return f"is {phrase[4:]}"
    if phrase.startswith("use "):
        return phrase.replace("use ", "uses ", 1)
    group = _entity_group(subtype)
    singular = _singularize_entity_group(group)
    if singular.startswith("uS "):
        singular = "US " + singular[3:]
    return f"is {_article_for(singular)} {singular}"


def _t_event_relative_clause(t: str) -> str:
    event = _t_event_in_question(t)
    if event == "discharged a weapon in the last year":
        return "discharged his weapon in the last year"
    return event


def _simple_open_suffix() -> str:
    return (
        "\n\nReply with a percentage (for example, 16% or 16 percent)."
    )


def _simple_mc_suffix(*, letters: str) -> str:
    return (
        f"\n\nWhich answer is closest? Reply with only the letter ({letters})."
    )


def _simple_given_t_subject(old_a: str) -> str:
    """Subject for simple-model P(C|T) questions (pool = old A = C union D)."""
    if _is_covid_vaccinated(old_a):
        return "an adult living in the US who is vaccinated for 2024-25 COVID"
    if _is_diabetic_adult(old_a):
        return "an adult with diagnosed diabetes"
    if _is_california_voter(old_a):
        return "a registered voter in California"
    if _is_first_generation_student(old_a):
        return "a student"
    if _is_public_secondary_teacher(old_a):
        return "a high school teacher"
    return _role_phrase(old_a)


def _simple_setting_phrase(universe: str) -> str:
    setting = _setting_phrase(universe)
    return re.sub(r"\s*\([^)]*\)", "", setting)


def _question_given_t(v: SimpleVignette) -> str:
    subject = _simple_given_t_subject(v.old_a)
    relative_event = _t_event_relative_clause(v.t)
    predicate = _entity_question_predicate(v.question_target_subtype())
    if " who " in subject:
        clause = f"{subject} and {relative_event}"
    else:
        clause = f"{subject} who {relative_event}"
    return f"What is the probability that {clause} {predicate}?"


def _from_vignette(v: Vignette) -> SimpleVignette:
    return SimpleVignette(
        name=v.name,
        universe=v.universe,
        old_a=v.a,
        old_c=v.c,
        old_d=v.d,
        t=v.t,
        p_c=v.p_a * v.q_c,
        p_d=v.p_a * v.q_d,
        s_c=v.s_c,
        s_d=v.s_d,
        intersection_size=v.intersection_size,
    )


def load_simple_vignettes() -> list[SimpleVignette]:
    vignettes = _load_two_cause() + _load_overlap()
    return [_from_vignette(v) for v in vignettes]


def _shuffle_keys(example_id: str, keys: tuple[str, ...]) -> tuple[str, ...]:
    digest = hashlib.sha256(f"{example_id}:{':'.join(keys)}".encode()).hexdigest()
    ordered = list(keys)
    seed = int(digest[:8], 16)
    for i in range(len(ordered) - 1, 0, -1):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        j = seed % (i + 1)
        ordered[i], ordered[j] = ordered[j], ordered[i]
    return tuple(ordered)


def _select_mc_lure_keys(
    percents: dict[str, float], example_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    primary = list(_shuffle_keys(example_id, SIMPLE_LURE_KEYS))
    reserve = [
        key
        for key in _shuffle_keys(example_id, SIMPLE_RESERVE_LURE_KEYS)
        if key not in primary
    ]
    ordered: list[str] = []
    if "normative" in percents:
        ordered.append("normative")
    for key in primary + reserve:
        if key not in ordered:
            ordered.append(key)

    seen_rounded: set[int] = set()
    keys: list[str] = []
    labels: list[str] = []
    for key in ordered:
        rounded = _rounded_mc_percent(percents[key])
        if rounded in seen_rounded:
            continue
        seen_rounded.add(rounded)
        keys.append(key)
        labels.append(_mc_rounded_label(percents[key]))
        if len(keys) == 5:
            break

    if "normative" not in keys:
        raise ValueError(f"normative missing from MC options for {example_id}")

    return tuple(keys), tuple(labels)


def build_mc_options(
    v: SimpleVignette, example_id: str
) -> tuple[dict[str, str], dict[str, str], str, tuple[str, ...]]:
    percents = v.lure_percents()
    keys, display_labels = _select_mc_lure_keys(percents, example_id)
    option_letters = tuple("ABCDE"[: len(keys)])
    labels: dict[str, str] = {}
    lures: dict[str, str] = {}
    normative_letter = "A"
    for letter, key, label in zip(option_letters, keys, display_labels):
        labels[letter] = label
        lures[letter] = SIMPLE_LURE_NAMES[key]
        if key == "normative":
            normative_letter = letter
    return labels, lures, normative_letter, option_letters


def _narrative_entity_order(
    v: SimpleVignette,
) -> tuple[tuple[str, float, float], tuple[str, float, float]]:
    """Return ((subtype, p, s), (subtype, p, s)) in narrative order."""
    first = (v.old_c, v.p_c, v.s_c)
    second = (v.old_d, v.p_d, v.s_d)
    if re.search(r"other california registrant", v.old_c, re.I) and re.search(
        r"southern california registrant", v.old_d, re.I
    ):
        first, second = second, first
    return first, second


def narrative_with_probs(v: SimpleVignette) -> str:
    setting = _simple_setting_phrase(v.universe)
    (c_subtype, c_p, c_s), (d_subtype, d_p, d_s) = _narrative_entity_order(v)
    parts = [
        INTRO,
        "",
        (
            f"{setting}, {_entity_share_clause(c_subtype, c_p)} and "
            f"{_entity_share_clause(d_subtype, d_p)}."
        ),
        (
            f"Among {_entity_among_group(c_subtype)}, {_simple_pct(c_s)} {_t_event(v.t)}; "
            f"among {_entity_among_group(d_subtype)}, {_simple_pct(d_s)} {_t_event(v.t)}."
        ),
        "",
        _question_given_t(v),
    ]
    return "\n".join(parts)


def _shared_item_fields(v: SimpleVignette) -> dict[str, str]:
    normative = v.target_posterior()
    return {
        "vignette_name": v.name,
        "intersection_size": v.intersection_size,
        "well_posed": "true",
        "normative": "well_posed",
        "p_c_and_d_given_a": "0",
        "p_c": f"{v.p_c:.6g}",
        "p_d": f"{v.p_d:.6g}",
        "p_t_given_c": f"{v.s_c:.6g}",
        "p_t_given_d": f"{v.s_d:.6g}",
        "normative_percent": f"{normative * 100:.4g}",
        "normative_open": _pct(normative),
        "confidence_required": "false",
        "scepticism_required": "false",
    }


def build_prompt(v: SimpleVignette, variant: str) -> tuple[str, dict[str, str]]:
    example_id = f"{v.example_prefix()}__{variant}"
    body = narrative_with_probs(v)
    item: dict[str, str] = {
        "example_id": example_id,
        "variant": variant,
        **_shared_item_fields(v),
    }

    if variant == "open_probs":
        item.update(
            {
                "response_type": "open",
                "normative_choice": "",
                "numeric_score_percent": "",
                "numeric_score_choice": "",
                "scepticism_score_target": item["normative_percent"],
            }
        )
        for letter in "abcde":
            item[f"option_{letter}_label"] = ""
            item[f"option_{letter}_lure"] = ""
        for letter in "fgh":
            item[f"option_{letter}_label"] = ""
            item[f"option_{letter}_lure"] = ""
        prompt = body + _simple_open_suffix()
        return prompt, item

    labels, lures, normative_letter, option_letters = build_mc_options(v, example_id)
    item.update(
        {
            "response_type": "mc",
            "normative_choice": normative_letter,
            "numeric_score_percent": item["normative_percent"],
            "numeric_score_choice": normative_letter,
            "scepticism_score_target": "n/a",
        }
    )
    lines = [body, ""]
    for letter in "abcde":
        item[f"option_{letter}_label"] = ""
        item[f"option_{letter}_lure"] = ""
    for letter in option_letters:
        item[f"option_{letter.lower()}_label"] = labels[letter]
        item[f"option_{letter.lower()}_lure"] = lures[letter]
        lines.append(f"{letter}. {labels[letter]}")
    for letter in "fgh":
        item[f"option_{letter}_label"] = ""
        item[f"option_{letter}_lure"] = ""
    prompt = "\n".join(lines) + _simple_mc_suffix(
        letters=_mc_letter_list(option_letters)
    )
    return prompt, item


def build_all() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    prompts: list[dict[str, str]] = []
    items: list[dict[str, str]] = []
    for vignette in load_simple_vignettes():
        for variant in VARIANTS:
            prompt, item = build_prompt(vignette, variant)
            prompts.append({"example_id": item["example_id"], "prompt": prompt})
            items.append(item)

    pmap = {row["example_id"]: row["prompt"] for row in prompts}
    benchmark: list[dict[str, str]] = []
    for item in items:
        row = {k: item.get(k, "") for k in BENCHMARK_SCORING_FIELDS}
        row.update(
            {
                "example_id": item["example_id"],
                "vignette_name": item["vignette_name"],
                "problem_type": "well_posed",
                "intersection_size": item["intersection_size"],
                "response_type": item["response_type"],
                "has_statistics": "true",
                "variant": item["variant"],
                "prompt": pmap[item["example_id"]],
            }
        )
        benchmark.append(row)
    return prompts, items, benchmark


def write_csvs() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts, items, benchmark = build_all()

    with (OUT_DIR / "prompts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "prompt"])
        writer.writeheader()
        writer.writerows(prompts)

    with (OUT_DIR / "items.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITEM_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in items:
            writer.writerow({k: row.get(k, "") for k in ITEM_FIELDS})

    with (OUT_DIR / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmark)

    return len(prompts)


def main() -> int:
    count = write_csvs()
    vignette_count = len(load_simple_vignettes())
    print(
        f"Wrote {count} prompts ({vignette_count} vignettes × {len(VARIANTS)} variants)"
    )
    print(f"Output: {OUT_DIR} (prompts.csv, items.csv, benchmark.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
