#!/usr/bin/env python3
"""Build simple two-path base-rate prompts (C, D -> T; estimate P(C|T))."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_base_rate_prompts import (  # noqa: E402
    COVID_VACCINE_SHORT,
    INTRO,
    OVERLAP_CSV,
    Vignette,
    _f,
    _intersection_size_label,
    _is_california_voter,
    _is_covid_vaccinated,
    _is_diabetic_adult,
    _is_first_generation_student,
    _is_public_secondary_teacher,
    _load_two_cause,
    _mc_letter_list,
    _mc_rounded_label,
    _overlap_p_cd,
    _pct,
    _role_phrase,
    _rounded_mc_percent,
    _setting_phrase,
    _t_event,
    _t_event_in_question,
    slug,
)

META_SCEPTICISM = (
    "The problem as stated is not sufficiently described, or is inconsistent, "
    "or is obviously incorrect"
)
DATA_AUDIT_INTRO = (
    "You are a data analyst. Your customer has asked you to make a prediction "
    "for a particular problem."
)
DATA_AUDIT_QUESTION = (
    "Does this problem description give enough sensible information for a "
    "well-founded conclusion?"
)
DATA_AUDIT_YES = (
    "Yes — the problem description gives enough sensible information for a "
    "well-founded conclusion."
)
DATA_AUDIT_NO = f"No — {META_SCEPTICISM[0].lower()}{META_SCEPTICISM[1:]}"
RESPONSE_AUDIT_GOOD = "Yes — this is a sound, well-founded answer."
RESPONSE_AUDIT_BAD = "No — this answer is not appropriate for the customer."

SIMPLE_EXCLUDED_OVERLAP_NAMES = frozenset(
    {
        "college STEM work",
        "professional drivers speeding",
        "actor waiter overlap",
    }
)


def _simple_pct(value: float) -> str:
    if value < 0.01:
        return f"{value * 100:.2f}%"
    return _pct(value)

OUT_DIR = ROOT / "data" / "simple"
IMPLAUSIBLE_P_C_D_CSV = OUT_DIR / "implausible_p_c_d.csv"
IMPLAUSIBLE_P_T_GIVEN_CSV = OUT_DIR / "implausible_p_t_given.csv"
PROBLEM_TYPE_WELL_POSED = "well_posed"
PROBLEM_TYPE_ALTERED = "altered"
CONDITION_NATURAL = "natural"
CONDITION_ALTERED = "altered"
CANONICAL_VARIANTS = ("mc_prob", "mc_w_meta", "data_audit", "response_audit")
VARIANTS = CANONICAL_VARIANTS

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
    "condition",
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
    "condition",
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
    p_a: float
    p_c: float
    p_d: float
    p_cd_given_a: float
    s_c: float
    s_d: float
    intersection_size: str
    condition: str = CONDITION_NATURAL
    problem_type: str = PROBLEM_TYPE_WELL_POSED
    s_n: float = 0.0

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

    def p_cd_universe(self) -> float:
        return self.p_a * self.p_cd_given_a

    def s_cd(self) -> float:
        return (self.s_c + self.s_d) / 2.0

    def posterior_c(self) -> float:
        num = self.s_c * self.p_c
        den = num + self.s_d * self.p_d
        return num / den if den else 0.0

    def posterior_c_overlap_aware(self) -> float:
        p_cd = self.p_cd_universe()
        p_c_only = max(0.0, self.p_c - p_cd)
        p_ct = p_c_only * self.s_c + p_cd * self.s_cd()
        p_t = (
            p_c_only * self.s_c
            + max(0.0, self.p_d - p_cd) * self.s_d
            + p_cd * self.s_cd()
        )
        return p_ct / p_t if p_t else 0.0

    def posterior_d(self) -> float:
        num = self.s_d * self.p_d
        den = self.s_c * self.p_c + num
        return num / den if den else 0.0

    def p_t_universe(self) -> float:
        return (
            self.s_c * self.p_c
            + self.s_d * self.p_d
            + self.s_n * max(0.0, 1.0 - self.p_a)
        )

    def posterior_c_with_n(self) -> float:
        p_t = self.p_t_universe()
        if p_t <= 0:
            return 0.0
        return self.s_c * self.p_c / p_t

    def question_target_subtype(self) -> str:
        if self.name == "CA Trump voter":
            return self.old_d
        return self.old_c

    def target_posterior(self) -> float:
        if self.name == "CA Trump voter":
            return self.posterior_d()
        if self.p_cd_given_a > 0:
            return self.posterior_c_overlap_aware()
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
        return f"{slug(self.name)}__{self.condition}"


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
    if re.search(r"massachusetts fourth grader", head, re.I):
        return "Massachusetts fourth graders"
    if re.search(r"massachusetts public school fourth grader", head, re.I):
        return "Massachusetts fourth graders"
    if re.search(r"new mexico fourth grader", head, re.I):
        return "New Mexico fourth graders"
    if re.search(r"new mexico public school fourth grader", head, re.I):
        return "New Mexico fourth graders"
    if re.search(r"west virginia twelfth grader", head, re.I):
        return "West Virginia twelfth graders"
    if re.search(r"west virginia public school twelfth grader", head, re.I):
        return "West Virginia twelfth graders"
    if re.search(r"arizona twelfth grader", head, re.I):
        return "Arizona twelfth graders"
    if re.search(r"arizona public school twelfth grader", head, re.I):
        return "Arizona twelfth graders"
    if re.search(r"watched an entire NFL game", head, re.I):
        return "watch NFL"
    if re.search(r"watched an entire MLB game", head, re.I):
        return "watch MLB"
    if re.search(r"^male$", head, re.I):
        return "men"
    if re.search(r"^female$", head, re.I):
        return "women"
    if re.search(r"^under age 45$", head, re.I):
        return "under age 45"
    if re.search(r"research doctorate holder without an md", head, re.I):
        return "research doctorate holders without an MD"
    if re.search(r"main teaching assignment is English", head, re.I):
        return "teach English as their primary assignment"
    if re.search(r"bachelor.*major field is English", head, re.I):
        return "have a bachelor's degree in English"
    if re.search(r"also holds a waiter", head, re.I):
        return "also work as waiters"
    if re.search(r"actor occupation is the secondary", head, re.I):
        return "have a secondary acting job"
    if s.startswith("uses "):
        return s.replace("uses ", "use ", 1)
    if s.startswith("has "):
        return s.replace("has ", "have ", 1)
    return s


def _entity_label(subtype: str) -> str:
    return _entity_phrase(subtype)


def _entity_group(subtype: str) -> str:
    phrase = _entity_phrase(subtype)
    if phrase in {
        "voters registered in Southern California",
        "other California voters",
        "Massachusetts fourth graders",
        "New Mexico fourth graders",
        "West Virginia twelfth graders",
        "Arizona twelfth graders",
        "men",
        "women",
        "under age 45",
    }:
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
    if phrase == "watch NFL":
        return f"{_simple_pct(pct)} watch NFL"
    if phrase == "watch MLB":
        return f"{_simple_pct(pct)} watch MLB"
    if phrase == "under age 45":
        return f"{_simple_pct(pct)} are under age 45"
    if phrase == "men":
        return f"{_simple_pct(pct)} are men"
    if phrase == "women":
        return f"{_simple_pct(pct)} are women"
    if phrase == "employed while enrolled":
        return f"{_simple_pct(pct)} are employed while enrolled"
    if phrase == "teach English as their primary assignment":
        return f"{_simple_pct(pct)} teach English as their primary assignment"
    if phrase == "have a bachelor's degree in English":
        return f"{_simple_pct(pct)} have a bachelor's degree in English"
    if phrase == "also work as waiters":
        return f"{_simple_pct(pct)} also work as waiters"
    if phrase == "have a secondary acting job":
        return f"{_simple_pct(pct)} have a secondary acting job"
    if phrase.startswith("also "):
        return f"{_simple_pct(pct)} {phrase}"
    if phrase.startswith(("use ", "have ", "are ")):
        return f"{_simple_pct(pct)} {phrase}"
    if phrase.endswith("members"):
        return f"{_simple_pct(pct)} are {phrase}"
    return f"{_simple_pct(pct)} are {_entity_group(subtype)}"


def _entity_among_group(subtype: str) -> str:
    phrase = _entity_phrase(subtype)
    if phrase in {"Massachusetts fourth graders", "New Mexico fourth graders"}:
        return phrase
    if phrase in {"West Virginia twelfth graders", "Arizona twelfth graders"}:
        return phrase
    if phrase == "study STEM":
        return "those who studied STEM"
    if phrase == "watch NFL":
        return "those who watch NFL"
    if phrase == "watch MLB":
        return "those who watch MLB"
    if phrase == "under age 45":
        return "those under age 45"
    if phrase == "men":
        return "men"
    if phrase == "women":
        return "women"
    if phrase == "employed while enrolled":
        return "those employed while enrolled"
    if phrase == "teach English as their primary assignment":
        return "those who teach English as their primary assignment"
    if phrase == "have a bachelor's degree in English":
        return "those who have a bachelor's degree in English"
    if phrase == "also work as waiters":
        return "those who also work as waiters"
    if phrase == "have a secondary acting job":
        return "those who have a secondary acting job"
    if phrase.startswith("also "):
        return f"those who {phrase[5:]}"
    if phrase.startswith(("use ", "have ", "are ")):
        return f"those who {phrase}"
    return _entity_group(subtype)


def _entity_answer_clause(subtype: str) -> str:
    phrase = _entity_phrase(subtype)
    if phrase == "study STEM":
        return "studied STEM"
    if phrase == "also work as waiters":
        return "also worked as a waiter"
    if phrase == "have a secondary acting job":
        return "had a secondary acting job"
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
    if phrase == "Massachusetts fourth graders":
        return "is from Massachusetts"
    if phrase == "New Mexico fourth graders":
        return "is a New Mexico fourth grader"
    if phrase == "West Virginia twelfth graders":
        return "is from West Virginia"
    if phrase == "Arizona twelfth graders":
        return "is an Arizona twelfth grader"
    if phrase == "watch NFL":
        return "watched an NFL game"
    if phrase == "watch MLB":
        return "watched an MLB game"
    if phrase == "men":
        return "is a man"
    if phrase == "women":
        return "is a woman"
    if phrase == "under age 45":
        return "is under age 45"
    if phrase == "also work as waiters":
        return "also works as a waiter"
    if phrase == "have a secondary acting job":
        return "has a secondary acting job"
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
    if event == "is a proficient reader":
        return event
    if event == "is an on-time graduate":
        return event
    return event


def _simple_open_suffix() -> str:
    return (
        "\n\nReply with a percentage (for example, 16% or 16 percent)."
    )


def _simple_mc_suffix(*, letters: str) -> str:
    return (
        f"\n\nWhich answer is closest? Reply with only the letter ({letters})."
    )


def _simple_mc_full_suffix(*, letters: str) -> str:
    return (
        f"\n\nWhich answer is closest? Reply on two or three lines.\n"
        f"Line 1: only the letter ({letters}).\n"
        "Line 2: your confidence from 1 (not confident) to 5 (very confident).\n"
        "Line 3 (optional): a brief comment explaining your choice."
    )


def _is_naep_ma_nm_pool(text: str) -> bool:
    return bool(
        re.search(r"massachusetts or new mexico", text, re.I)
        and re.search(r"fourth grader", text, re.I)
    )


def _is_graduation_wv_az_pool(text: str) -> bool:
    return bool(
        re.search(r"west virginia or arizona", text, re.I)
        and re.search(r"twelfth grader", text, re.I)
    )


def _is_nfl_mlb_watcher(text: str) -> bool:
    return bool(
        re.search(r"nfl or mlb", text, re.I)
        or re.search(r"nfl and mlb", text, re.I)
    )


def _is_us_adult_universe(text: str) -> bool:
    return bool(re.search(r"adult 18 years or older", text, re.I))


def _simple_given_t_subject(old_a: str) -> str:
    """Subject for simple-model P(C|T) questions (pool = old A = C union D)."""
    if _is_graduation_wv_az_pool(old_a):
        return "a twelfth grader in West Virginia or Arizona"
    if _is_naep_ma_nm_pool(old_a):
        return "a fourth grader in Massachusetts or New Mexico"
    if _is_nfl_mlb_watcher(old_a):
        return "an adult who watched an NFL or MLB game in the past year"
    if _is_us_adult_universe(old_a):
        return "an adult"
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
    if _is_actor_occupation(old_a):
        return "an actor in the US labor force"
    return _role_phrase(old_a)


def _is_actor_occupation(old_a: str) -> bool:
    return bool(re.search(r"actor occupation", old_a, re.I))


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
        p_a=v.p_a,
        p_c=v.p_a * v.q_c,
        p_d=v.p_a * v.q_d,
        p_cd_given_a=v.p_cd,
        s_c=v.s_c,
        s_d=v.s_d,
        s_n=v.f_n,
        intersection_size=v.intersection_size,
    )


def _load_overlap_for_simple() -> list[Vignette]:
    rows: list[Vignette] = []
    with OVERLAP_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["name"] in SIMPLE_EXCLUDED_OVERLAP_NAMES:
                continue
            rows.append(
                Vignette(
                    name=row["name"],
                    universe=row["universe"],
                    a=row["A"],
                    n=row["N"],
                    c=row["C"],
                    d=row["D"],
                    t=row["T"],
                    p_a=_f(row["P_A"]),
                    q_c=_f(row["P_C_given_A"]),
                    q_d=_f(row["P_D_given_A"]),
                    s_c=_f(row["P_T_given_C"]),
                    s_d=_f(row["P_T_given_D"]),
                    f_n=_f(row["P_T_given_N"]),
                    p_cd=_overlap_p_cd(row),
                    well_posed=False,
                    intersection_size=_intersection_size_label(row, well_posed=False),
                    normative=row.get("normative", "underdetermined"),
                )
            )
    return rows


def load_simple_vignettes() -> list[SimpleVignette]:
    vignettes = _load_two_cause() + _load_overlap_for_simple()
    return [_from_vignette(v) for v in vignettes]


def _implausible_value_is_set(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and text.upper() not in {"N/A", "NA"}


def _parse_implausible_float(value: str, *, vignette_name: str, column: str) -> float:
    text = (value or "").strip()
    if not _implausible_value_is_set(text):
        raise ValueError(
            f"Missing implausible value for {vignette_name!r} column {column!r}"
        )
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float {text!r} for {vignette_name!r} column {column!r}"
        ) from exc


def _load_implausible_p_c_d() -> dict[str, tuple[float, float]]:
    if not IMPLAUSIBLE_P_C_D_CSV.is_file():
        return {}
    stats: dict[str, tuple[float, float]] = {}
    with IMPLAUSIBLE_P_C_D_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["vignette_name"].strip()
            stats[name] = (
                _parse_implausible_float(row["C"], vignette_name=name, column="C"),
                _parse_implausible_float(row["D"], vignette_name=name, column="D"),
            )
    return stats


def _load_implausible_p_t_given() -> dict[str, tuple[float, float]]:
    if not IMPLAUSIBLE_P_T_GIVEN_CSV.is_file():
        return {}
    stats: dict[str, tuple[float, float]] = {}
    with IMPLAUSIBLE_P_T_GIVEN_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["vignette_name"].strip()
            stats[name] = (
                _parse_implausible_float(
                    row["P(T|C)"], vignette_name=name, column="P(T|C)"
                ),
                _parse_implausible_float(
                    row["P(T|D)"], vignette_name=name, column="P(T|D)"
                ),
            )
    return stats


def has_overlap(v: SimpleVignette) -> bool:
    """True when pathways C and D may co-occur (non-zero intersection under A)."""
    return (v.intersection_size or "").strip() not in ("", "0")


def is_well_posed_vignette(v: SimpleVignette) -> bool:
    """Natural condition with disjoint C and D (no intersection under A)."""
    return v.condition == CONDITION_NATURAL and not has_overlap(v)


def scepticism_required_for_variant(v: SimpleVignette, variant: str) -> bool:
    """Whether this variant scores against scepticism_score_target vs Bayes numeric."""
    if is_well_posed_vignette(v):
        return False
    return variant in ("mc_w_meta", "data_audit", "response_audit")


def variants_for_vignette(v: SimpleVignette) -> tuple[str, ...]:
    return CANONICAL_VARIANTS


def _uses_absolute_c_d_shares(v: SimpleVignette) -> bool:
    """Altered P(C)/P(D) are stated as universe shares, not conditional on A."""
    return v.condition == CONDITION_ALTERED


def _stub_ai_response(v: SimpleVignette) -> str:
    return f"About {_rounded_mc_percent(v.target_posterior() * 100)}%"


def _data_audit_normative_choice(v: SimpleVignette) -> str:
    return "A" if is_well_posed_vignette(v) else "B"


def _response_audit_normative_choice(v: SimpleVignette) -> str:
    return "A" if is_well_posed_vignette(v) else "B"


def _mc_w_meta_scepticism_target(v: SimpleVignette) -> str:
    return "n/a" if is_well_posed_vignette(v) else "F"


def _clear_mc_options(item: dict[str, str]) -> None:
    for letter in "abcdefgh":
        item[f"option_{letter}_label"] = ""
        item[f"option_{letter}_lure"] = ""


def _build_mc_prob_sections(
    v: SimpleVignette, example_id: str
) -> tuple[list[str], dict[str, str], dict[str, str], str, tuple[str, ...]]:
    labels, lures, normative_letter, option_letters = build_mc_options(v, example_id)
    lines: list[str] = []
    for letter in option_letters:
        lines.append(f"{letter}. {labels[letter]}")
    return lines, labels, lures, normative_letter, option_letters


def _prediction_problem_body(v: SimpleVignette) -> str:
    """Scenario statistics and question, without the consultant role intro."""
    text = narrative_with_probs(v)
    if text.startswith(INTRO):
        text = text[len(INTRO) :].lstrip("\n")
    return text.strip()


def _format_mc_prob_prompt(
    v: SimpleVignette, example_id: str, *, include_intro: bool = True
) -> str:
    body = narrative_with_probs(v)
    lines, _, _, _, option_letters = _build_mc_prob_sections(v, example_id)
    parts = []
    if include_intro:
        parts.append(body)
        parts.append("")
    parts.extend(lines)
    suffix_letters = _mc_letter_list(option_letters)
    parts.append("")
    parts.append(
        f"Which answer is closest? Reply with only the letter ({suffix_letters})."
    )
    return "\n".join(parts)


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


def _naep_ma_nm_share_line(v: SimpleVignette) -> str | None:
    if v.name != "NAEP grade 4 reading (MA vs NM)":
        return None
    (c_subtype, c_p, _), (d_subtype, d_p, _) = _narrative_entity_order(v)
    return (
        "Among public school fourth graders in Massachusetts or New Mexico "
        "(66,751 enrolled in Massachusetts and 22,503 in New Mexico, fall 2022 CCD), "
        f"{_simple_pct(c_p)} are from Massachusetts and "
        f"{_simple_pct(d_p)} are from New Mexico."
    )


NAEP_PROFICIENT_READER_DEFINITION = (
    "A proficient reader is a student who scores at or above NAEP Proficient "
    "on the grade-4 reading assessment (the same national standard in both states)."
)


def _naep_ma_nm_narrative_with_probs(v: SimpleVignette) -> str:
    (c_subtype, c_p, c_s), (d_subtype, d_p, d_s) = _narrative_entity_order(v)
    parts = [
        INTRO,
        "",
        _naep_ma_nm_share_line(v),
        NAEP_PROFICIENT_READER_DEFINITION,
        (
            f"Among fourth graders from Massachusetts, {_simple_pct(c_s)} are proficient readers; "
            f"among fourth graders from New Mexico, {_simple_pct(d_s)} are proficient readers."
        ),
        "",
        _question_given_t(v),
    ]
    return "\n".join(parts)


def _graduation_wv_az_share_line(v: SimpleVignette) -> str | None:
    if v.name != "HS graduation ACGR (WV vs AZ)":
        return None
    (c_subtype, c_p, _), (d_subtype, d_p, _) = _narrative_entity_order(v)
    return (
        "Among public school twelfth graders in West Virginia or Arizona "
        "(17,489 enrolled in West Virginia and 95,122 in Arizona, fall 2022 CCD), "
        f"{_simple_pct(c_p)} are from West Virginia and "
        f"{_simple_pct(d_p)} are from Arizona."
    )


HS_GRADUATION_ACGR_DEFINITION = (
    "An on-time graduate is a student who earns a regular high school diploma "
    "within four years of starting 9th grade, counted under the federal "
    "adjusted cohort graduation rate (ACGR—the same national methodology in both states)."
)


def _graduation_wv_az_narrative_with_probs(v: SimpleVignette) -> str:
    (c_subtype, c_p, c_s), (d_subtype, d_p, d_s) = _narrative_entity_order(v)
    parts = [
        INTRO,
        "",
        _graduation_wv_az_share_line(v),
        HS_GRADUATION_ACGR_DEFINITION,
        (
            f"Among twelfth graders from West Virginia, {_simple_pct(c_s)} are on-time graduates; "
            f"among twelfth graders from Arizona, {_simple_pct(d_s)} are on-time graduates."
        ),
        "",
        _question_given_t(v),
    ]
    return "\n".join(parts)


def _book_drama_streaming_share_line(v: SimpleVignette) -> str | None:
    if v.name != "book drama streaming":
        return None
    return (
        f"Among US adults 18 years and older, {_simple_pct(v.p_c)} read at least "
        f"one book in the past 12 months and {_simple_pct(v.p_d)} watch streaming "
        f"services."
    )


def _book_drama_streaming_question(v: SimpleVignette) -> str:
    return (
        "What is the probability that a US adult who read drama or watches dramas "
        "online read at least one book in the past 12 months?"
    )


def _book_drama_streaming_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        _book_drama_streaming_share_line(v),
        f"An estimated {_simple_pct(v.p_cd_universe())} fall into both categories.",
        (
            f"Among adults who read books, {_simple_pct(v.s_c)} read at least one "
            f"drama book in the past year. Among adults who watch streaming services, "
            f"{_simple_pct(v.s_d)} watch drama films or shows as online content."
        ),
        "",
        _book_drama_streaming_question(v),
    ]
    return "\n".join(parts)


def _dog_cat_household_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        (
            f"Among US households, {_simple_pct(v.p_c)} own at least one dog and "
            f"{_simple_pct(v.p_d)} own at least one cat."
        ),
        f"An estimated {_simple_pct(v.p_cd_universe())} fall into both categories.",
        (
            f"Among dog-owning households, {_simple_pct(v.s_c)} took at least one "
            f"pet to a veterinarian in the past 12 months. Among cat-owning "
            f"households, {_simple_pct(v.s_d)} did."
        ),
        "",
        (
            "What is the probability that a household that took at least one pet "
            "to a veterinarian in the past 12 months owns at least one dog?"
        ),
    ]
    return "\n".join(parts)


def _youtube_facebook_news_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        (
            f"Among US adults 18 years and older, {_simple_pct(v.p_c)} use YouTube "
            f"and {_simple_pct(v.p_d)} use Facebook."
        ),
        f"An estimated {_simple_pct(v.p_cd_universe())} fall into both categories.",
        (
            f"Among YouTube users, {_simple_pct(v.s_c)} got news on YouTube in the "
            f"past month. Among Facebook users, {_simple_pct(v.s_d)} got news on "
            f"Facebook in the past month."
        ),
        "",
        (
            "What is the probability that a US adult who got news on YouTube or "
            "Facebook in the past month uses YouTube?"
        ),
    ]
    return "\n".join(parts)


def _college_grad_professional_job_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        (
            f"Among US adults aged 25 years and older, {_simple_pct(v.p_c)} hold a "
            f"bachelor's degree or higher and {_simple_pct(v.p_d)} are employed in "
            f"a management, business, science, or arts occupation."
        ),
        f"An estimated {_simple_pct(v.p_cd_universe())} fall into both categories.",
        (
            f"Among adults with a bachelor's degree or higher, {_simple_pct(v.s_c)} "
            f"earned $100,000 or more in the past 12 months. Among adults employed "
            f"in management, business, science, or arts occupations, "
            f"{_simple_pct(v.s_d)} did."
        ),
        "",
        (
            "What is the probability that a US adult aged 25 or older who earned "
            "$100,000 or more in the past 12 months holds a bachelor's degree or "
            "higher?"
        ),
    ]
    return "\n".join(parts)


def _homeowner_suburban_mortgage_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        (
            f"Among US households, {_simple_pct(v.p_c)} own their home and "
            f"{_simple_pct(v.p_d)} are in suburban areas."
        ),
        f"An estimated {_simple_pct(v.p_cd_universe())} fall into both categories.",
        (
            f"Among households that own their home, {_simple_pct(v.s_c)} have a "
            f"mortgage. Among suburban households, {_simple_pct(v.s_d)} have a "
            f"mortgage."
        ),
        "",
        (
            "What is the probability that a US household with a mortgage "
            "owns its home?"
        ),
    ]
    return "\n".join(parts)


def _parent_married_dual_income_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        (
            f"Among US adults 18 years and older, {_simple_pct(v.p_c)} live with "
            f"an own child under age 18 and {_simple_pct(v.p_d)} are currently "
            f"married."
        ),
        f"An estimated {_simple_pct(v.p_cd_universe())} fall into both categories.",
        (
            f"Among adults living with an own child under 18, {_simple_pct(v.s_c)} "
            f"live in a dual-income household. Among married adults, "
            f"{_simple_pct(v.s_d)} live in a dual-income household."
        ),
        "",
        (
            "What is the probability that a US adult who lives in a dual-income "
            "household lives with an own child under age 18?"
        ),
    ]
    return "\n".join(parts)


def _republican_gun_owner_ban_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        (
            f"Among US adults 18 years and older, {_simple_pct(v.p_c)} identify as "
            f"Republican or lean Republican and {_simple_pct(v.p_d)} personally own "
            f"a firearm."
        ),
        f"An estimated {_simple_pct(v.p_cd_universe())} fall into both categories.",
        (
            f"Among Republicans, {_simple_pct(v.s_c)} favor banning assault-style "
            f"weapons. Among firearm owners, {_simple_pct(v.s_d)} favor banning "
            f"assault-style weapons."
        ),
        "",
        (
            "What is the probability that a US adult who favors banning "
            "assault-style weapons identifies as Republican or leans Republican?"
        ),
    ]
    return "\n".join(parts)


def _physician_phd_research_share_line(v: SimpleVignette) -> str | None:
    if v.name != "physician vs PhD research":
        return None
    return (
        f"Among US-residing active physicians and employed research doctorate "
        f"holders, {_simple_pct(v.p_c)} are physicians (MD/DO) and "
        f"{_simple_pct(v.p_d)} hold a research doctorate without an MD."
    )


def _physician_phd_research_question(v: SimpleVignette) -> str:
    return (
        "What is the probability that a member of this workforce who engages "
        "in research is a physician (MD/DO)?"
    )


def _physician_phd_research_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        _physician_phd_research_share_line(v),
        (
            f"Among physicians, {_simple_pct(v.s_c)} engage in research as part "
            f"of professional work. Among research doctorate holders without an MD, "
            f"{_simple_pct(v.s_d)} report research and development as their primary "
            f"work activity."
        ),
        "",
        _physician_phd_research_question(v),
    ]
    return "\n".join(parts)


def _tax_mfj_single_ctc_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        (
            "Among US individual income tax returns filed as married filing "
            f"jointly or single, {_simple_pct(v.p_c)} are married filing jointly "
            f"and {_simple_pct(v.p_d)} are single."
        ),
        (
            f"Among married-filing-jointly returns, {_simple_pct(v.s_c)} claimed "
            f"at least one qualifying child for the Child Tax Credit. Among single "
            f"returns, {_simple_pct(v.s_d)} did."
        ),
        "",
        (
            "What is the probability that a return in this group that claimed a "
            "qualifying child for the Child Tax Credit was married filing jointly?"
        ),
    ]
    return "\n".join(parts)


def _homeownership_under_35_vs_65_narrative_with_probs(v: SimpleVignette) -> str:
    parts = [
        INTRO,
        "",
        (
            "Among US households headed by someone under 35 or age 65 or older, "
            f"{_simple_pct(v.p_c)} are younger households (head under 35) and "
            f"{_simple_pct(v.p_d)} are senior households (head age 65 or older)."
        ),
        (
            f"Among younger households, {_simple_pct(v.s_c)} own their home. "
            f"Among senior households, {_simple_pct(v.s_d)} do."
        ),
        "",
        (
            "What is the probability that a household in this group that owns "
            "its home is a younger household?"
        ),
    ]
    return "\n".join(parts)


def _q_c_given_a(v: SimpleVignette) -> float:
    return v.p_c / v.p_a if v.p_a else 0.0


def _q_d_given_a(v: SimpleVignette) -> float:
    return v.p_d / v.p_a if v.p_a else 0.0


def _covid_vaccine_blue_red_narrative_with_probs(v: SimpleVignette) -> str:
    if _uses_absolute_c_d_shares(v):
        parts = [
            INTRO,
            "",
            (
                f"Among US adults 18 years and older, {_simple_pct(v.p_c)} live in "
                f"blue states and {_simple_pct(v.p_d)} live in red states."
            ),
            (
                f"Among adults in blue states, {_simple_pct(v.s_c)} had COVID-19 in "
                f"the past 12 months; among adults in red states, "
                f"{_simple_pct(v.s_d)} did."
            ),
            "",
            (
                "What is the probability that a vaccinated adult who had COVID-19 "
                "in the past 12 months lives in a blue state?"
            ),
        ]
        return "\n".join(parts)

    unvaccinated = max(0.0, 1.0 - v.p_a)
    parts = [
        INTRO,
        "",
        (
            f"Among US adults 18 years and older, {_simple_pct(v.p_a)} received "
            f"{COVID_VACCINE_SHORT} and {_simple_pct(unvaccinated)} did not."
        ),
        (
            f"Among the vaccinated, {_simple_pct(_q_c_given_a(v))} live in blue "
            f"states and {_simple_pct(_q_d_given_a(v))} live in red states."
        ),
        (
            f"Among vaccinated adults in blue states, {_simple_pct(v.s_c)} had "
            f"COVID-19 in the past 12 months; among vaccinated adults in red "
            f"states, {_simple_pct(v.s_d)} did."
        ),
        "",
        (
            "What is the probability that a vaccinated adult who had COVID-19 "
            "in the past 12 months lives in a blue state?"
        ),
    ]
    return "\n".join(parts)


def _military_overseas_narrative_with_probs(v: SimpleVignette) -> str:
    if _uses_absolute_c_d_shares(v):
        parts = [
            INTRO,
            "",
            (
                "Among US active-duty military and federal civilian employees, "
                f"{_simple_pct(v.p_c)} serve in the Army and "
                f"{_simple_pct(v.p_d)} serve in the Navy or Air Force."
            ),
            (
                f"Among soldiers in the Army, {_simple_pct(v.s_c)} have served "
                f"overseas at least once since joining; among members of the Navy "
                f"or Air Force, {_simple_pct(v.s_d)} have."
            ),
            "",
            (
                "What is the probability that an active-duty service member who "
                "has served overseas serves in the Army?"
            ),
        ]
        return "\n".join(parts)

    parts = [
        INTRO,
        "",
        (
            "Among active-duty US military service members, "
            f"{_simple_pct(_q_c_given_a(v))} serve in the Army and "
            f"{_simple_pct(_q_d_given_a(v))} serve in the Navy or Air Force."
        ),
        (
            f"Among soldiers in the Army, {_simple_pct(v.s_c)} have served "
            f"overseas at least once since joining; among members of the Navy "
            f"or Air Force, {_simple_pct(v.s_d)} have."
        ),
        "",
        (
            "What is the probability that an active-duty service member who "
            "has served overseas serves in the Army?"
        ),
    ]
    return "\n".join(parts)


def _simple_overlap_clause(v: SimpleVignette) -> str:
    if not has_overlap(v) or v.p_cd_given_a <= 0:
        return ""
    return (
        f" An estimated {_simple_pct(v.p_cd_universe())} fall into both categories."
    )


def narrative_with_probs(v: SimpleVignette) -> str:
    if v.name == "NAEP grade 4 reading (MA vs NM)":
        return _naep_ma_nm_narrative_with_probs(v)
    if v.name == "HS graduation ACGR (WV vs AZ)":
        return _graduation_wv_az_narrative_with_probs(v)
    if v.name == "book drama streaming":
        return _book_drama_streaming_narrative_with_probs(v)
    if v.name == "dog cat household":
        return _dog_cat_household_narrative_with_probs(v)
    if v.name == "youtube facebook news":
        return _youtube_facebook_news_narrative_with_probs(v)
    if v.name == "college grad professional job":
        return _college_grad_professional_job_narrative_with_probs(v)
    if v.name == "homeowner suburban mortgage":
        return _homeowner_suburban_mortgage_narrative_with_probs(v)
    if v.name == "parent married dual income":
        return _parent_married_dual_income_narrative_with_probs(v)
    if v.name == "republican gun owner ban":
        return _republican_gun_owner_ban_narrative_with_probs(v)
    if v.name == "physician vs PhD research":
        return _physician_phd_research_narrative_with_probs(v)
    if v.name == "tax MFJ single CTC":
        return _tax_mfj_single_ctc_narrative_with_probs(v)
    if v.name == "homeownership under 35 vs 65":
        return _homeownership_under_35_vs_65_narrative_with_probs(v)
    if v.name == "covid vaccine (blue/red)":
        return _covid_vaccine_blue_red_narrative_with_probs(v)
    if v.name == "military overseas (federal pool)":
        return _military_overseas_narrative_with_probs(v)
    setting = _simple_setting_phrase(v.universe)
    (c_subtype, c_p, c_s), (d_subtype, d_p, d_s) = _narrative_entity_order(v)
    share_line = _naep_ma_nm_share_line(v) or _graduation_wv_az_share_line(v)
    if share_line is None:
        share_line = (
            f"{setting}, {_entity_share_clause(c_subtype, c_p)} and "
            f"{_entity_share_clause(d_subtype, d_p)}."
            f"{_simple_overlap_clause(v)}"
        )
    parts = [
        INTRO,
        "",
        share_line,
        (
            f"Among {_entity_among_group(c_subtype)}, {_simple_pct(c_s)} {_t_event(v.t)}; "
            f"among {_entity_among_group(d_subtype)}, {_simple_pct(d_s)} {_t_event(v.t)}."
        ),
        "",
        _question_given_t(v),
    ]
    return "\n".join(parts)


def _shared_item_fields(v: SimpleVignette) -> dict[str, str]:
    is_altered = v.condition == CONDITION_ALTERED
    well_posed = is_well_posed_vignette(v)
    normative = v.target_posterior()
    if is_altered:
        normative_label = "implausible"
    elif has_overlap(v):
        normative_label = "underdetermined"
    else:
        normative_label = "well_posed"
    return {
        "vignette_name": v.name,
        "condition": v.condition,
        "problem_type": v.problem_type,
        "intersection_size": v.intersection_size,
        "well_posed": str(well_posed).lower(),
        "normative": normative_label,
        "p_c_and_d_given_a": (
            f"{v.p_cd_given_a:.6g}" if v.p_cd_given_a > 0 else "0"
        ),
        "p_c": f"{v.p_c:.6g}",
        "p_d": f"{v.p_d:.6g}",
        "p_t_given_c": f"{v.s_c:.6g}",
        "p_t_given_d": f"{v.s_d:.6g}",
        "normative_percent": f"{normative * 100:.4g}",
        "normative_open": _pct(normative),
        "confidence_required": "false",
    }


def build_prompt(v: SimpleVignette, variant: str) -> tuple[str, dict[str, str]]:
    example_id = f"{v.example_prefix()}__{variant}"
    body = narrative_with_probs(v)
    item: dict[str, str] = {
        "example_id": example_id,
        "variant": variant,
        **_shared_item_fields(v),
    }
    _clear_mc_options(item)
    item["confidence_required"] = "false"

    if variant == "mc_prob":
        lines, labels, lures, normative_letter, option_letters = _build_mc_prob_sections(
            v, example_id
        )
        item.update(
            {
                "response_type": "mc",
                "normative_choice": normative_letter,
                "numeric_score_percent": item["normative_percent"],
                "numeric_score_choice": normative_letter,
                "scepticism_required": "false",
                "scepticism_score_target": "n/a",
            }
        )
        for letter in option_letters:
            item[f"option_{letter.lower()}_label"] = labels[letter]
            item[f"option_{letter.lower()}_lure"] = lures[letter]
        prompt = _format_mc_prob_prompt(v, example_id)
        return prompt, item

    if variant == "mc_w_meta":
        lines, labels, lures, normative_letter, option_letters = _build_mc_prob_sections(
            v, example_id
        )
        scepticism_target = _mc_w_meta_scepticism_target(v)
        item.update(
            {
                "response_type": "mc_full",
                "normative_choice": normative_letter,
                "numeric_score_percent": item["normative_percent"],
                "numeric_score_choice": normative_letter,
                "scepticism_required": str(
                    scepticism_required_for_variant(v, "mc_w_meta")
                ).lower(),
                "scepticism_score_target": scepticism_target,
            }
        )
        for letter in option_letters:
            item[f"option_{letter.lower()}_label"] = labels[letter]
            item[f"option_{letter.lower()}_lure"] = lures[letter]
        item["option_f_label"] = META_SCEPTICISM
        item["option_f_lure"] = "bad_data (insufficient, inconsistent, or incorrect)"
        prompt_lines = [body, ""] + lines + [f"F. {META_SCEPTICISM}", ""]
        suffix_letters = f"{_mc_letter_list(option_letters)}, or F"
        prompt = "\n".join(prompt_lines) + _simple_mc_suffix(letters=suffix_letters)
        return prompt, item

    if variant == "data_audit":
        normative_letter = _data_audit_normative_choice(v)
        problem_body = _prediction_problem_body(v)
        item.update(
            {
                "response_type": "data_audit",
                "normative_choice": normative_letter,
                "numeric_score_percent": "",
                "numeric_score_choice": "",
                "scepticism_required": str(
                    scepticism_required_for_variant(v, "data_audit")
                ).lower(),
                "scepticism_score_target": normative_letter,
            }
        )
        item["option_a_label"] = DATA_AUDIT_YES
        item["option_a_lure"] = "well_founded"
        item["option_b_label"] = DATA_AUDIT_NO
        item["option_b_lure"] = "bad_data"
        prompt = "\n".join(
            [
                DATA_AUDIT_INTRO,
                "",
                problem_body,
                "",
                DATA_AUDIT_QUESTION,
                "",
                f"A. {DATA_AUDIT_YES}",
                f"B. {DATA_AUDIT_NO}",
                "",
                "Which answer is closest? Reply with only the letter (A or B).",
            ]
        )
        return prompt, item

    if variant == "response_audit":
        embedded = _format_mc_prob_prompt(v, example_id)
        stub = _stub_ai_response(v)
        normative_letter = _response_audit_normative_choice(v)
        item.update(
            {
                "response_type": "response_audit",
                "normative_choice": normative_letter,
                "numeric_score_percent": item["normative_percent"],
                "numeric_score_choice": "",
                "scepticism_required": str(
                    scepticism_required_for_variant(v, "response_audit")
                ).lower(),
                "scepticism_score_target": normative_letter,
            }
        )
        item["option_a_label"] = RESPONSE_AUDIT_GOOD
        item["option_a_lure"] = "good_response"
        item["option_b_label"] = RESPONSE_AUDIT_BAD
        item["option_b_lure"] = "bad_response"
        prompt = "\n".join(
            [
                "You are a data auditor, checking the work of another AI.",
                "",
                "The AI was given the following prompt:",
                "",
                embedded,
                "",
                "And it gave the following response:",
                "",
                stub,
                "",
                "Is this a good answer to give your customer?",
                "",
                f"A. {RESPONSE_AUDIT_GOOD}",
                f"B. {RESPONSE_AUDIT_BAD}",
                "",
                "Which answer is closest? Reply with only the letter (A or B).",
            ]
        )
        return prompt, item

    raise ValueError(f"unknown variant: {variant}")


def _validate_implausible_csv_coverage(
    base_vignettes: list[SimpleVignette],
    p_c_d_stats: dict[str, tuple[float, float]],
    p_t_stats: dict[str, tuple[float, float]],
) -> None:
    names = {v.name for v in base_vignettes}
    missing_c_d = sorted(names - set(p_c_d_stats))
    missing_t = sorted(names - set(p_t_stats))
    if missing_c_d or missing_t:
        parts: list[str] = []
        if missing_c_d:
            parts.append(
                f"missing from {IMPLAUSIBLE_P_C_D_CSV.name}: {', '.join(missing_c_d)}"
            )
        if missing_t:
            parts.append(
                f"missing from {IMPLAUSIBLE_P_T_GIVEN_CSV.name}: {', '.join(missing_t)}"
            )
        raise ValueError(
            "Every simple vignette needs altered stats in both implausible CSVs. "
            + "; ".join(parts)
        )


def build_condition_vignettes(
    base_vignettes: list[SimpleVignette],
    p_c_d_stats: dict[str, tuple[float, float]],
    p_t_stats: dict[str, tuple[float, float]],
) -> list[SimpleVignette]:
    _validate_implausible_csv_coverage(base_vignettes, p_c_d_stats, p_t_stats)
    vignettes: list[SimpleVignette] = []
    for vignette in base_vignettes:
        vignettes.append(
            replace(
                vignette,
                condition=CONDITION_NATURAL,
                problem_type=PROBLEM_TYPE_WELL_POSED,
            )
        )
        p_c, p_d = p_c_d_stats[vignette.name]
        s_c, s_d = p_t_stats[vignette.name]
        vignettes.append(
            replace(
                vignette,
                p_c=p_c,
                p_d=p_d,
                s_c=s_c,
                s_d=s_d,
                condition=CONDITION_ALTERED,
                problem_type=PROBLEM_TYPE_ALTERED,
            )
        )
    return vignettes


def build_all() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    prompts: list[dict[str, str]] = []
    items: list[dict[str, str]] = []
    base_vignettes = load_simple_vignettes()
    p_c_d_stats = _load_implausible_p_c_d()
    p_t_stats = _load_implausible_p_t_given()
    vignettes = build_condition_vignettes(base_vignettes, p_c_d_stats, p_t_stats)

    for vignette in vignettes:
        for variant in variants_for_vignette(vignette):
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
                "condition": item["condition"],
                "problem_type": item["problem_type"],
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
    _, items, _ = build_all()
    natural_count = sum(1 for row in items if row["condition"] == CONDITION_NATURAL)
    altered_count = sum(1 for row in items if row["condition"] == CONDITION_ALTERED)
    print(
        f"Wrote {count} prompts "
        f"({natural_count} natural + {altered_count} altered, "
        f"× {len(CANONICAL_VARIANTS)} variants each)"
    )
    print(f"Output: {OUT_DIR} (prompts.csv, items.csv, benchmark.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
