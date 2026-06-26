#!/usr/bin/env python3
"""Build base-rate vignette prompts (with/without probs; open and MC variants)."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "base_rate"
TWO_CAUSE_CSV = ROOT / "docs" / "base-rate-two-cause-vignettes.csv"
OVERLAP_CSV = ROOT / "docs" / "base-rate-overlap-vignettes.csv"
IMPLAUSIBLE_CSV = OUT_DIR / "implausible_statistics.csv"

IMPLAUSIBLE_PARAM_FIELD = {
    "P_A": "p_a",
    "P_C_given_A": "q_c",
    "P_D_given_A": "q_d",
    "P_T_given_C": "s_c",
    "P_T_given_D": "s_d",
}

META_F = "Insufficient information"
META_G = "Inconsistent information"
META_H = "Provided information is obviously incorrect"

OPEN_META = f'"{META_F}" | "{META_G}" | "{META_H}"'

LURE_KEYS = ("product", "path_d", "path_c", "normative", "partition")
RESERVE_LURE_KEYS = ("p_t_a", "p_a")
ALL_LURE_KEYS = LURE_KEYS + RESERVE_LURE_KEYS
LURE_NAMES = {
    "product": "q_c*s_c × q_d*s_d (product; assumes independence)",
    "path_d": "q_d*s_d only (second pathway; assumes partition)",
    "path_c": "q_c*s_c only (first pathway; assumes partition)",
    "normative": "overlap-aware Bayes (uses P(C∩D|A) estimate)",
    "partition": "partition shortcut (assumes P(C∩D|A)=0)",
    "p_t_a": "P(T | A) confusion",
    "p_a": "P(A) confusion",
}

VARIANTS = (
    "open_probs",
    "open_no_probs",
    "mc_numeric_probs",
    "mc_numeric_no_probs",
    "mc_full_probs",
    "mc_full_no_probs",
)

INTRO = (
    "You are a statistical consultant. Your task is to estimate a "
    "conditional probability from the information below."
)


COVID_VACCINE_SHORT = "the 2024-25 COVID vaccine"


def _head_raw(text: str) -> str:
    head = text.split(":")[0].strip()
    return re.sub(r"\s*\(.*", "", head).strip()


def _head_match(text: str) -> str:
    """Lowercase head text for role/subtype detection."""
    head = text.split(":")[0].strip()
    head = re.sub(r"\s*\(.*", "", head).strip()
    return head.lower() if head else ""


def _head_display(text: str) -> str:
    """Head text for prose, preserving US."""
    head = text.split(":")[0].strip()
    head = re.sub(r"\s*\(.*", "", head).strip()
    if not head:
        return ""
    s = head[0].lower() + head[1:]
    return re.sub(r"\bus\b", "US", s)


def _is_covid_vaccinated(text: str) -> bool:
    head = _head_match(text)
    return head.startswith("received the updated") and "covid" in head


def _is_covid_unvaccinated(text: str) -> bool:
    head = _head_match(text)
    return head.startswith("did not receive the updated") and "covid" in head


def _is_professional_driver(text: str) -> bool:
    return _head_match(text).startswith("professional driver")


def _is_other_employed_adult(text: str) -> bool:
    return "primary occupation is not professional driving" in _head_match(text)


def _is_actor_holder(text: str) -> bool:
    return "holds an actor occupation" in _head_match(text)


def _is_non_actor_labor(text: str) -> bool:
    return "no actor occupation" in _head_match(text)


def _is_health_care_professional(text: str) -> bool:
    return _head_match(text).startswith("health care professional")


def _is_non_health_care_employed(text: str) -> bool:
    return "not a health care professional" in _head_match(text)


def _is_military_service_member(text: str) -> bool:
    return "active-duty us military service member" in _head_match(text)


def _is_federal_civilian(text: str) -> bool:
    return _head_match(text).startswith("us federal civilian employee")


def _is_california_voter(text: str) -> bool:
    head = _head_match(text)
    return "home state is california" in head and "not california" not in head


def _is_non_california_voter(text: str) -> bool:
    return "home state is not california" in _head_match(text)


def _is_diabetic_adult(text: str) -> bool:
    return _head_match(text).startswith("adult with diagnosed diabetes")


def _is_non_diabetic_adult(text: str) -> bool:
    return _head_match(text).startswith("adult without diagnosed diabetes")


def _is_first_generation_student(text: str) -> bool:
    return _head_match(text).startswith("first-generation student")


def _is_continuing_generation_student(text: str) -> bool:
    return _head_match(text).startswith("continuing-generation student")


def _is_public_secondary_teacher(text: str) -> bool:
    return _head_match(text).startswith("public school teacher with main assignment in grades 9-12")


def _is_non_secondary_teacher_employed(text: str) -> bool:
    return "not a public school teacher in grades 9-12" in _head_match(text)


def _asks_given_t_only(v: Vignette) -> bool:
    """Vignettes where the natural question is P(A|T), conditioning on T alone."""
    return (
        (_is_covid_vaccinated(v.a) and _is_covid_unvaccinated(v.n))
        or (_is_diabetic_adult(v.a) and _is_non_diabetic_adult(v.n))
        or (_is_california_voter(v.a) and _is_non_california_voter(v.n))
        or (_is_health_care_professional(v.a) and _is_non_health_care_employed(v.n))
    )


def _target_outcome_phrase(v: Vignette) -> str:
    if _is_covid_vaccinated(v.a):
        return "they were vaccinated for 2024-25 COVID"
    if _is_diabetic_adult(v.a):
        return "they have diagnosed diabetes"
    if _is_california_voter(v.a):
        return "they were registered in California"
    if _is_health_care_professional(v.a):
        return "they were a health care professional"
    return f"the person was {_role_answer_phrase(v.a)}"


def _estimate_target_phrase(v: Vignette) -> str:
    if _is_covid_vaccinated(v.a):
        return "was vaccinated for 2024-25 COVID"
    if _is_diabetic_adult(v.a):
        return "have diagnosed diabetes"
    if _is_california_voter(v.a):
        return "was registered in California"
    if _is_health_care_professional(v.a):
        return "was a health care professional"
    return f"was {_role_answer_phrase(v.a)}"


def _role_core(text: str) -> str:
    """Strip qualifiers; normalize verb-led descriptions to short prose."""
    if _is_covid_vaccinated(text):
        return "vaccinated for 2024-25 COVID"
    if _is_covid_unvaccinated(text):
        return "not vaccinated for 2024-25 COVID"
    if _is_other_employed_adult(text):
        return "other adult"
    if _is_public_secondary_teacher(text):
        return "public grades 9-12 teacher"
    if _is_non_secondary_teacher_employed(text):
        return "other employed adult"
    if _is_actor_holder(text):
        return "actor"
    if _is_non_actor_labor(text):
        return "non-actor"
    if _is_health_care_professional(text):
        return "health care professional"
    if _is_non_health_care_employed(text):
        return "other employed adult"
    if _is_military_service_member(text):
        return "active-duty US military service member"
    if _is_federal_civilian(text):
        return "US federal civilian employee"
    if _is_california_voter(text):
        return "registered in California"
    if _is_non_california_voter(text):
        return "registered elsewhere"
    if _is_diabetic_adult(text):
        return "have diagnosed diabetes"
    if _is_non_diabetic_adult(text):
        return "do not have diagnosed diabetes"
    s = _head_display(text)
    if s.startswith("received "):
        return f"people who {s}"
    if s.startswith("did not receive "):
        return f"people who {s}"
    return s


def _role_phrase(text: str) -> str:
    """Natural phrase for a person-type, e.g. 'a police officer'."""
    s = _role_core(text)
    if s.startswith("people who "):
        return s
    if s.startswith(("a ", "an ")):
        return s
    if s.upper().startswith("US "):
        return f"a {s}"
    article = "an" if s[:1] in "aeiou" else "a"
    return f"{article} {s}"


def _role_plural(text: str) -> str:
    """Plural role for rate sentences, e.g. 'police officers'."""
    if _is_actor_holder(text):
        return "actors"
    if _is_non_actor_labor(text):
        return "non-actors"
    if _is_california_voter(text):
        return "registered in California"
    if _is_non_california_voter(text):
        return "registered elsewhere"
    s = _role_core(text)
    if s.startswith("people who "):
        return s
    rules: tuple[tuple[str, str], ...] = (
        (r"^adult ", "adults "),
        (r"registered voter", "registered voters"),
        (r"health care professional", "health care professionals"),
        (r"police officer", "police officers"),
        (r"security guard", "security guards"),
        (r"active-duty us military service member", "active-duty US military service members"),
        (r"us federal civilian employee", "US federal civilian employees"),
        (r"professional driver", "professional drivers"),
        (r"first-generation student", "first-generation students"),
        (r"continuing-generation student", "continuing-generation students"),
        (r"other employed adult", "other employed adults"),
        (r"employed adult", "employed adults"),
    )
    for pattern, replacement in rules:
        if re.search(pattern, s, re.I):
            return re.sub(pattern, replacement, s, count=1, flags=re.I)
    words = s.split()
    if words and not words[-1].endswith("s"):
        words[-1] = f"{words[-1]}s"
    return " ".join(words)


def _subtype_phrase(text: str) -> str:
    """Short label for a subtype under the first group."""
    head = text.split(":")[0].strip()
    if re.search(r"\s+among\s+", head, re.I):
        head = re.split(r"\s+among\s+", head, maxsplit=1, flags=re.I)[0]
    head = re.sub(r"\s*\(.*", "", head).strip()
    if not head:
        return text.strip().lower()
    if re.search(r"also holds a waiter", head, re.I):
        return "also work as waiters"
    if re.search(r"actor occupation is the secondary", head, re.I):
        return "have a secondary acting job"
    if re.search(r"stem field of study", head, re.I):
        return "major in STEM"
    if re.search(r"employed while enrolled", head, re.I):
        return "work while enrolled"
    if re.search(r"blue-state resident", head, re.I):
        return "live in blue states"
    if re.search(r"red-state resident", head, re.I):
        return "live in red states"
    if re.search(r"southern california registrant", head, re.I):
        return "registered in southern California"
    if re.search(r"other california registrant", head, re.I):
        return "registered in other parts of the state"
    if re.search(r"^bus driver", head, re.I):
        return "bus drivers"
    if re.search(r"heavy and tractor-trailer truck driver", head, re.I):
        return "heavy truck drivers"
    if re.search(r"main teaching assignment is English", head, re.I):
        return "teach English/language arts"
    if re.search(r"bachelor.*major field is English", head, re.I):
        return "majored in English/language arts"
    if re.search(r"^physician", head, re.I):
        return "physicians"
    if re.search(r"non-physician health care professional", head, re.I):
        return "non-physician health care professionals"
    if re.search(r"^US Army active-duty", head, re.I):
        return "US Army active-duty service members"
    if re.search(r"^US Navy or US Air Force active-duty", head, re.I):
        return "US Navy or US Air Force active-duty service members"
    s = head[0].lower() + head[1:]
    s = re.sub(r"\bus\b", "US", s)
    if s.startswith("uses "):
        return s.replace("uses ", "use ", 1)
    if s.startswith("has "):
        return s.replace("has ", "have ", 1)
    return s


def _subtype_plural(text: str) -> str:
    label = _subtype_phrase(text)
    if label == "major in STEM":
        return "STEM majors"
    if label == "work while enrolled":
        return "students who work while enrolled"
    if label in {"teach English/language arts", "majored in English/language arts"}:
        return f"those who {label}"
    if label == "also work as waiters":
        return "actors who also work as waiters"
    if label == "have a secondary acting job":
        return "actors whose acting job is secondary"
    if label.startswith("live in "):
        return f"those who {label}"
    if label.startswith("registered in "):
        return f"those {label}"
    if label.startswith(("use ", "have ", "also work")):
        return f"those who {label}"
    if label.endswith("members") or label.endswith("s"):
        return label
    return f"{label}s"


def _subtype_some_phrase(text: str) -> str:
    label = _subtype_phrase(text)
    if label.startswith(
        (
            "live in ",
            "registered in ",
            "also work",
            "have a secondary",
            "major in",
            "work while",
            "teach ",
            "majored in ",
        )
    ):
        return label
    if label.startswith(("use ", "have ")):
        return label
    return f"are {_subtype_plural(text)}"


def _subtype_share_phrase(text: str, pct: float) -> str:
    label = _subtype_phrase(text)
    if label.startswith(
        (
            "live in ",
            "registered in ",
            "also work",
            "have a secondary",
            "major in",
            "work while",
            "teach ",
            "majored in ",
        )
    ):
        return f"{_pct(pct)} {label}"
    if label.startswith(("use ", "have ")):
        return f"{_pct(pct)} {label}"
    if label.endswith("members"):
        return f"{_pct(pct)} are {label}"
    return f"{_pct(pct)} are {_subtype_plural(text)}"


def _t_event(t: str) -> str:
    """Outcome event for rate sentences (plural-friendly verbs)."""
    head = _head_raw(t)
    if not head:
        return t.strip().rstrip(".").lower()
    s = _head_display(t)
    if s.startswith("had covid"):
        return s
    if s.startswith("had "):
        return s
    if s.startswith("has worked"):
        return "have worked overseas"
    if s.startswith("has "):
        return s.replace("has ", "have ", 1)
    if s.startswith("holds "):
        return s.replace("holds ", "hold ", 1)
    if s.startswith("works in"):
        return "work in a hospital"
    if s.startswith("worked part-time at primary job"):
        return "worked part-time at their primary job in the reference week"
    if s.startswith("retained to year 2"):
        return "returned for a second year"
    if s.startswith(("voted ", "received ", "discharged ")):
        return s
    if s.startswith("hemoglobin"):
        return f"had {s}"
    return s


def _t_event_in_question(t: str) -> str:
    """Outcome event phrased for singular subject in a question."""
    event = _t_event(t)
    singular = {
        "work in a hospital": "works in a hospital",
        "have worked overseas": "has worked overseas",
        "hold a master's degree or higher": "holds a master's degree or higher",
    }
    return singular.get(event, event)


def _role_answer_phrase(text: str) -> str:
    """Phrase after 'the person was' in the target question."""
    if _is_covid_vaccinated(text):
        return "vaccinated for 2024-25 COVID"
    if _is_actor_holder(text):
        return "an actor"
    if _is_military_service_member(text):
        return "an active-duty service member"
    return _role_phrase(text)


def _pool_phrase(a: str, n: str) -> str:
    """Combined pool for questions."""
    if _is_covid_vaccinated(a) and _is_covid_unvaccinated(n):
        return "vaccinated for 2024-25 COVID or not"
    if _is_professional_driver(a) and _is_other_employed_adult(n):
        return "a professional driver or other adult"
    if _is_public_secondary_teacher(a) and _is_non_secondary_teacher_employed(n):
        return "a public grades 9-12 teacher or other employed adult"
    if _is_actor_holder(a) and _is_non_actor_labor(n):
        return "an actor or a non-actor"
    if _is_health_care_professional(a) and _is_non_health_care_employed(n):
        return "a health care professional or other employed adult"
    if _is_military_service_member(a) and _is_federal_civilian(n):
        return "an active-duty service member or a federal civilian employee"
    if _is_california_voter(a) and _is_non_california_voter(n):
        return "registered in California or elsewhere"
    if _is_diabetic_adult(a) and _is_non_diabetic_adult(n):
        return "with or without diagnosed diabetes"
    return f"{_role_phrase(a)} or {_role_phrase(n)}"


def _setting_phrase(universe: str) -> str:
    """Opening phrase for the population line, e.g. 'Among US registered voters…'."""
    if universe.startswith("Employed "):
        tail = universe.removeprefix("Employed ")
        return f"Among employed {tail[0].lower()}{tail[1:]}"
    if universe.startswith("US "):
        return f"Among {universe}"
    if universe.startswith("Undergraduate "):
        tail = universe.removeprefix("Undergraduate ")
        return f"Among undergraduate {tail[0].lower()}{tail[1:]}"
    return f"In {universe}"


def _population_split_line(v: Vignette, *, with_probs: bool) -> str:
    setting = _setting_phrase(v.universe)
    if _is_covid_vaccinated(v.a) and _is_covid_unvaccinated(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} received {COVID_VACCINE_SHORT} "
                f"and the remainder did not."
            )
        return (
            f"{setting}, some received {COVID_VACCINE_SHORT} and the rest did not."
        )
    if _is_professional_driver(v.a) and _is_other_employed_adult(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} are professional drivers "
                f"and the remainder are other adults."
            )
        return (
            f"{setting}, some are professional drivers and the rest are other adults."
        )
    if _is_public_secondary_teacher(v.a) and _is_non_secondary_teacher_employed(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} are public grades 9-12 teachers "
                f"and the remainder are other employed adults."
            )
        return (
            f"{setting}, some are public grades 9-12 teachers "
            f"and the rest are other employed adults."
        )
    if _is_actor_holder(v.a) and _is_non_actor_labor(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} are actors "
                f"and the remainder are non-actors."
            )
        return f"{setting}, some are actors and the rest are non-actors."
    if _is_health_care_professional(v.a) and _is_non_health_care_employed(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} are health care professionals "
                f"and the remainder are other employed adults."
            )
        return (
            f"{setting}, some are health care professionals "
            f"and the rest are other employed adults."
        )
    if _is_military_service_member(v.a) and _is_federal_civilian(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} are active-duty service members "
                f"and the remainder are federal civilian employees."
            )
        return (
            f"{setting}, some are active-duty service members "
            f"and the rest are federal civilian employees."
        )
    if _is_california_voter(v.a) and _is_non_california_voter(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} are registered in California "
                f"and the remainder are registered elsewhere."
            )
        return (
            f"{setting}, some are registered in California "
            f"and the rest are registered elsewhere."
        )
    if _is_diabetic_adult(v.a) and _is_non_diabetic_adult(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} have diagnosed diabetes "
                f"and the remainder do not."
            )
        return (
            f"{setting}, some have diagnosed diabetes and the rest do not."
        )
    if _is_first_generation_student(v.a) and _is_continuing_generation_student(v.n):
        if with_probs:
            return (
                f"{setting}, {_pct(v.p_a)} are first-generation students "
                f"and the remainder are continuing-generation students."
            )
        return f"{setting}, some are first-generation students."
    a_plural = _role_plural(v.a)
    n_plural = _role_plural(v.n)
    if with_probs:
        return (
            f"{setting}, {_pct(v.p_a)} are {a_plural} "
            f"and the remainder are {n_plural}."
        )
    return (
        f"{setting}, some are {a_plural} "
        f"and the rest are {n_plural}."
    )


def _among_a_phrase(v: Vignette) -> str:
    if _is_covid_vaccinated(v.a):
        return "the vaccinated"
    if _is_actor_holder(v.a):
        return "actors"
    if _is_military_service_member(v.a):
        return "active-duty service members"
    if _is_california_voter(v.a):
        return "those registered in California"
    if _is_diabetic_adult(v.a):
        return "those with diagnosed diabetes"
    if _is_public_secondary_teacher(v.a):
        return "public grades 9-12 teachers"
    return _role_plural(v.a)


def _among_n_phrase(v: Vignette) -> str:
    if _is_covid_unvaccinated(v.n):
        return "the unvaccinated"
    if _is_other_employed_adult(v.n):
        return "other adults"
    if _is_non_secondary_teacher_employed(v.n):
        return "other employed adults"
    if _is_non_actor_labor(v.n):
        return "non-actors"
    if _is_non_health_care_employed(v.n):
        return "other employed adults"
    if _is_federal_civilian(v.n):
        return "federal civilian employees"
    if _is_non_california_voter(v.n):
        return "those registered elsewhere"
    if _is_non_diabetic_adult(v.n):
        return "those without diagnosed diabetes"
    if _is_continuing_generation_student(v.n):
        return "other students"
    return _role_plural(v.n)


def _question_given_t(v: Vignette) -> str:
    if _asks_given_t_only(v):
        return (
            f"Given that someone {_t_event_in_question(v.t)}, "
            f"what is the probability {_target_outcome_phrase(v)}?"
        )
    pool = _pool_phrase(v.a, v.n)
    event = _t_event_in_question(v.t)
    return (
        f"Given that {pool} {event}, "
        f"what is the probability the person was {_role_answer_phrase(v.a)}?"
    )


def _estimate_question(v: Vignette) -> str:
    if _asks_given_t_only(v):
        return (
            f"Using knowledge of the world, please estimate the probability that someone "
            f"who {_t_event(v.t)} {_estimate_target_phrase(v)}."
        )
    pool = _pool_phrase(v.a, v.n)
    event = _t_event_in_question(v.t)
    return (
        f"Using knowledge of the world, please estimate the probability that a person who was {pool} "
        f"and who {event} was {_role_answer_phrase(v.a)}."
    )


def _overlap_clause(v: Vignette) -> str:
    if v.p_cd <= 0:
        return ""
    return f" An estimated {_pct(v.p_cd)} fall into both categories."


def _ca_geo_split(c: str, d: str) -> bool:
    return (
        _subtype_phrase(c) == "registered in other parts of the state"
        and _subtype_phrase(d) == "registered in southern California"
    )


def _subtype_order(v: Vignette) -> tuple[str, str, float, float]:
    """Return (first_subtype, second_subtype, first_q, second_q); southern CA first when applicable."""
    if _ca_geo_split(v.c, v.d):
        return v.d, v.c, v.q_d, v.q_c
    return v.c, v.d, v.q_c, v.q_d


def _s_for_subtype(v: Vignette, subtype: str) -> float:
    return v.s_c if subtype == v.c else v.s_d


def _subtype_partition_line(v: Vignette, a_plural: str, *, with_probs: bool) -> str:
    if _ca_geo_split(v.c, v.d):
        if with_probs:
            first, second, q_first, q_second = _subtype_order(v)
            return (
                f"Among {a_plural}, {_subtype_share_phrase(first, q_first)} and "
                f"{_subtype_share_phrase(second, q_second)}."
            )
        return (
            f"Among {a_plural}, some are registered in southern California "
            f"and some in other parts of the state."
        )
    if with_probs:
        return (
            f"Among {a_plural}, {_subtype_share_phrase(v.c, v.q_c)} and "
            f"{_subtype_share_phrase(v.d, v.q_d)}."
        )
    return (
        f"Among {a_plural}, some {_subtype_some_phrase(v.c)} and "
        f"some {_subtype_some_phrase(v.d)}."
    )


def narrative_with_probs(v: Vignette) -> str:
    """Prose vignette with probabilities woven into definitions."""
    among_a = _among_a_phrase(v)
    among_n = _among_n_phrase(v)
    first, second, _, _ = _subtype_order(v)

    parts = [
        INTRO,
        "",
        _population_split_line(v, with_probs=True),
        _subtype_partition_line(v, among_a, with_probs=True) + _overlap_clause(v),
        (
            f"Among {_subtype_plural(first)}, {_pct(_s_for_subtype(v, first))} {_t_event(v.t)}; "
            f"among {_subtype_plural(second)}, {_pct(_s_for_subtype(v, second))} {_t_event(v.t)}; "
            f"among {among_n}, {_pct(v.f_n)} {_t_event(v.t)}."
        ),
        "",
        _question_given_t(v),
    ]
    return "\n".join(parts)


def narrative_no_probs(v: Vignette) -> str:
    """Scenario description without numeric rates."""
    parts = [
        INTRO,
        "",
        _population_split_line(v, with_probs=False),
        _subtype_partition_line(v, _among_a_phrase(v), with_probs=False)
        + (f"{_overlap_clause(v)}" if v.p_cd > 0 else ""),
        "",
        _estimate_question(v),
    ]
    return "\n".join(parts)


def slug(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _f(value: str) -> float:
    return float(value)


def _pct(value: float) -> str:
    if value < 0.001:
        return f"{value * 100:.2f}%"
    if value < 0.01:
        return f"{value * 100:.1f}%"
    if value < 0.1 and abs(value * 100 - round(value * 100)) > 0.05:
        return f"{value * 100:.1f}%"
    return f"{round(value * 100)}%"


def _rounded_mc_percent(percent: float) -> int:
    """Nearest whole percent for MC numeric option labels (percent is 0–100 scale)."""
    return int(round(percent))


def _mc_rounded_label(percent: float) -> str:
    return f"About {_rounded_mc_percent(percent)}%"


MC_SINGLE_ROUNDED_PERCENT_NOTICES: list[str] = []


def _shuffle_keys(example_id: str, keys: tuple[str, ...]) -> tuple[str, ...]:
    digest = hashlib.sha256(f"{example_id}:{':'.join(keys)}".encode()).hexdigest()
    ordered = list(keys)
    seed = int(digest[:8], 16)
    for i in range(len(ordered) - 1, 0, -1):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        j = seed % (i + 1)
        ordered[i], ordered[j] = ordered[j], ordered[i]
    return tuple(ordered)


def _overlap_p_cd(row: dict[str, str]) -> float:
    point = (row.get("P_C_and_D_given_A") or "").strip()
    if point:
        return _f(point)
    lo = (row.get("P_C_and_D_given_A_min") or "").strip()
    hi = (row.get("P_C_and_D_given_A_max") or "").strip()
    if lo and hi:
        return (_f(lo) + _f(hi)) / 2.0
    return 0.0


@dataclass(frozen=True)
class Vignette:
    name: str
    universe: str
    a: str
    n: str
    c: str
    d: str
    t: str
    p_a: float
    q_c: float
    q_d: float
    s_c: float
    s_d: float
    f_n: float
    p_cd: float
    well_posed: bool
    intersection_size: str
    normative: str = "well_posed"
    p_t_given_a_target: float | None = None

    @property
    def p_n(self) -> float:
        return 1.0 - self.p_a

    def _cells(self) -> dict[str, float]:
        return {
            "c_only": max(0.0, self.q_c - self.p_cd),
            "d_only": max(0.0, self.q_d - self.p_cd),
            "cd": self.p_cd,
            "neither": max(0.0, 1.0 - self.q_c - self.q_d + self.p_cd),
        }

    def s_cd(self) -> float:
        """P(T | C∩D, A) when not stated separately."""
        return (self.s_c + self.s_d) / 2.0

    def p_t_given_a_conditional(self) -> float:
        """P(T|A) using four-way partition and overlap estimate."""
        cells = self._cells()
        base = (
            cells["c_only"] * self.s_c
            + cells["d_only"] * self.s_d
            + cells["cd"] * self.s_cd()
        )
        neither = cells["neither"]
        if neither <= 1e-12:
            return base
        if self.p_t_given_a_target is not None:
            s_neither = max(0.0, (self.p_t_given_a_target - base) / neither)
        else:
            s_neither = 0.0
        return base + neither * s_neither

    def p_t_given_a_partition(self) -> float:
        """P(T|A) if C and D are treated as disjoint (P(C∩D|A)=0)."""
        return self.q_c * self.s_c + self.q_d * self.s_d

    def p_t(self) -> float:
        return self.p_t_given_a_conditional() * self.p_a + self.f_n * self.p_n

    def posterior_with_p_t_a(self, p_t_a: float) -> float:
        den = p_t_a * self.p_a + self.f_n * self.p_n
        return (p_t_a * self.p_a / den) if den else 0.0

    def posterior_a(self) -> float:
        return self.posterior_with_p_t_a(self.p_t_given_a_conditional())

    def posterior_partition(self) -> float:
        return self.posterior_with_p_t_a(self.p_t_given_a_partition())

    def lure_percents(self) -> dict[str, float]:
        a_path = self.q_c * self.s_c
        b_path = self.q_d * self.s_d
        return {
            "product": self.posterior_with_p_t_a(a_path * b_path) * 100,
            "path_d": self.posterior_with_p_t_a(b_path) * 100,
            "path_c": self.posterior_with_p_t_a(a_path) * 100,
            "normative": self.posterior_a() * 100,
            "partition": self.posterior_partition() * 100,
            "p_t_a": self.p_t_given_a_partition() * 100,
            "p_a": self.p_a * 100,
        }

    def example_prefix(self) -> str:
        base = slug(self.name)
        if not self.well_posed:
            base = f"{base}__overlap"
        if self.normative == "implausible":
            base = f"{base}__implausible"
        return base


def body_with_probs(v: Vignette) -> str:
    return narrative_with_probs(v)


def body_no_probs(v: Vignette) -> str:
    return narrative_no_probs(v)


def _load_two_cause() -> list[Vignette]:
    rows: list[Vignette] = []
    with TWO_CAUSE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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
                    p_cd=0.0,
                    well_posed=True,
                    intersection_size=_intersection_size_label(row, well_posed=True),
                    normative="well_posed",
                    p_t_given_a_target=_f(row["P_T_given_A"]),
                )
            )
    return rows


def _load_overlap() -> list[Vignette]:
    rows: list[Vignette] = []
    with OVERLAP_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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


def _implausible_value_is_set(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and text.upper() not in {"N/A", "NA"}


def _apply_implausible_parameter(v: Vignette, parameter: str, value: float) -> Vignette:
    field = IMPLAUSIBLE_PARAM_FIELD[parameter]
    return replace(v, **{field: value, "normative": "implausible"})


def _load_implausible() -> list[Vignette]:
    if not IMPLAUSIBLE_CSV.is_file():
        return []

    edits: dict[str, tuple[str, float]] = {}
    with IMPLAUSIBLE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            implausible_value = (row.get("implausible_value") or "").strip()
            if not _implausible_value_is_set(implausible_value):
                continue
            name = row["vignette_name"].strip()
            if name in edits:
                raise ValueError(f"Multiple implausible values for vignette {name!r}")
            edits[name] = (row["parameter"].strip(), float(implausible_value))

    base_by_name = {v.name: v for v in _load_two_cause() + _load_overlap()}
    missing = sorted(set(edits) - set(base_by_name))
    if missing:
        raise ValueError(f"Unknown vignette_name(s) in implausible_statistics.csv: {missing}")

    vignettes: list[Vignette] = []
    for name in sorted(edits):
        parameter, value = edits[name]
        if parameter not in IMPLAUSIBLE_PARAM_FIELD:
            raise ValueError(f"Unknown parameter {parameter!r} for vignette {name!r}")
        vignettes.append(_apply_implausible_parameter(base_by_name[name], parameter, value))
    return vignettes


def load_vignettes() -> list[Vignette]:
    return _load_two_cause() + _load_overlap() + _load_implausible()


def _shuffle_lure_keys(example_id: str) -> tuple[str, ...]:
    return _shuffle_keys(example_id, LURE_KEYS)


def _select_mc_lure_keys(
    percents: dict[str, float], example_id: str
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Pick up to five lure keys with unique nearest-integer percent labels."""
    primary = list(_shuffle_lure_keys(example_id))
    reserve = [
        key
        for key in _shuffle_keys(example_id, RESERVE_LURE_KEYS)
        if key not in primary
    ]
    ordered: list[str] = []
    for key in ("normative", "partition"):
        if key in percents and key not in ordered:
            ordered.append(key)
    for key in primary + reserve:
        if key not in ordered:
            ordered.append(key)

    all_rounded = {_rounded_mc_percent(percents[key]) for key in ordered}
    single_rounded_percent = len(all_rounded) == 1

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

    return tuple(keys), tuple(labels), single_rounded_percent


def build_mc_options(
    v: Vignette, example_id: str
) -> tuple[dict[str, str], dict[str, str], str, tuple[str, ...]]:
    """Return labels, lures, normative letter, and option letters for numeric MC choices."""
    percents = v.lure_percents()
    keys, display_labels, single_rounded_percent = _select_mc_lure_keys(percents, example_id)
    if single_rounded_percent:
        notice = (
            f"{example_id} ({v.name}): only one distinct rounded percent "
            f"({display_labels[0]}) among numeric MC lure values"
        )
        MC_SINGLE_ROUNDED_PERCENT_NOTICES.append(notice)

    option_letters = tuple("ABCDE"[: len(keys)])
    labels: dict[str, str] = {}
    lures: dict[str, str] = {}
    normative_letter = "A"
    for letter, key, label in zip(option_letters, keys, display_labels):
        labels[letter] = label
        lures[letter] = LURE_NAMES[key]
        if key == "normative":
            normative_letter = letter
    partition_letter = _resolve_partition_letter(keys, option_letters, percents)
    return labels, lures, normative_letter, option_letters, partition_letter


def _open_suffix(*, with_meta: bool) -> str:
    meta_line = (
        "Line 1: a percentage (for example, 16% or 16 percent), or exactly one of: "
        f"{OPEN_META}.\n"
        if with_meta
        else "Line 1: a percentage (for example, 16% or 16 percent).\n"
    )
    return (
        "\n\nReply on two lines.\n"
        f"{meta_line}"
        "Line 2: your confidence from 1 (not confident) to 5 (very confident)."
    )


def _mc_letter_list(letters: tuple[str, ...]) -> str:
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def _mc_suffix(*, letters: str) -> str:
    return (
        f"\n\nWhich answer is closest? Reply on two lines.\n"
        f"Line 1: only the letter ({letters}).\n"
        "Line 2: your confidence from 1 (not confident) to 5 (very confident)."
    )


def _variant_has_probs(variant: str) -> bool:
    return not variant.endswith("_no_probs")


def _intersection_size_label(row: dict, *, well_posed: bool) -> str:
    if well_posed:
        return "0"
    stored = (row.get("intersection_size") or "").strip()
    return stored or "?"


def _response_format(variant: str) -> str:
    if variant.startswith("open_"):
        return "open"
    if variant.startswith("mc_numeric_"):
        return "mc_numeric"
    if variant.startswith("mc_full_"):
        return "mc_full"
    raise ValueError(f"unknown variant: {variant}")


def _problem_type(well_posed: bool) -> str:
    return "well_posed" if well_posed else "overlap"


def scepticism_required(v: Vignette) -> bool:
    """Whether the item expects scepticism (overlap-aware refusal or meta), not blind maths."""
    if v.normative == "implausible":
        return True
    if v.intersection_size in {"medium", "large"}:
        return True
    return False


def _resolve_partition_letter(
    keys: tuple[str, ...],
    option_letters: tuple[str, ...],
    percents: dict[str, float],
) -> str:
    for letter, key in zip(option_letters, keys):
        if key == "partition":
            return letter
    partition_rounded = _rounded_mc_percent(percents["partition"])
    for letter, key in zip(option_letters, keys):
        if _rounded_mc_percent(percents[key]) == partition_rounded:
            return letter
    raise ValueError("partition lure missing from MC options")


def _numeric_targets_round_same(v: Vignette) -> bool:
    """True when overlap-aware and partition posteriors round to the same MC percent."""
    normative_percent = v.posterior_a() * 100
    partition_percent = v.posterior_partition() * 100
    return _rounded_mc_percent(normative_percent) == _rounded_mc_percent(partition_percent)


def _scoring_measure_fields(
    v: Vignette,
    variant: str,
    *,
    partition_letter: str = "",
    normative_letter: str = "",
) -> dict[str, str]:
    """Numeric (uncritical partition) and scepticism scoring metadata for one variant."""
    partition_percent = f"{v.posterior_partition() * 100:.4g}"
    normative_percent = f"{v.posterior_a() * 100:.4g}"
    required = scepticism_required(v)

    if v.normative == "implausible":
        if variant.startswith("open"):
            scepticism_target = "meta"
        elif variant.startswith("mc_full"):
            scepticism_target = "H"
        else:
            scepticism_target = "n/a"
    elif required:
        if variant.startswith("open"):
            scepticism_target = partition_percent
        elif variant.startswith("mc_full"):
            scepticism_target = "F|G|H"
        else:
            scepticism_target = partition_letter
    elif _numeric_targets_round_same(v):
        if variant.startswith("open"):
            scepticism_target = normative_percent
        elif variant.startswith("mc_numeric"):
            scepticism_target = "n/a"
        elif variant.startswith("mc_full"):
            scepticism_target = normative_letter
        else:
            scepticism_target = normative_percent
    elif variant.startswith("mc_numeric"):
        scepticism_target = "n/a"
    elif variant.startswith("mc_full"):
        scepticism_target = "F|G|H"
    else:
        scepticism_target = "meta"

    return {
        "numeric_score_percent": partition_percent,
        "numeric_score_choice": partition_letter if variant.startswith("mc") else "",
        "scepticism_required": str(required).lower(),
        "scepticism_score_target": scepticism_target,
    }


def build_prompt(v: Vignette, variant: str) -> tuple[str, dict[str, str]]:
    """Return prompt text and item metadata fields for one variant."""
    example_id = f"{v.example_prefix()}__{variant}"
    body = body_with_probs(v) if _variant_has_probs(variant) else body_no_probs(v)
    item: dict[str, str] = {
        "example_id": example_id,
        "vignette_name": v.name,
        "variant": variant,
        "well_posed": str(v.well_posed).lower(),
        "problem_type": _problem_type(v.well_posed),
        "intersection_size": v.intersection_size,
        "response_format": _response_format(variant),
        "has_statistics": str(_variant_has_probs(variant)).lower(),
        "normative": v.normative,
        "p_c_and_d_given_a": str(v.p_cd),
        "confidence_required": "true",
    }

    if variant.startswith("open"):
        item["response_type"] = "open"
        item["normative_open"] = _pct(v.posterior_a())
        item["normative_percent"] = f"{v.posterior_a() * 100:.4g}"
        item["normative_choice"] = ""
        item.update(_scoring_measure_fields(v, variant))
        prompt = body + _open_suffix(with_meta=True)
        return prompt, item

    labels, lures, numeric_letter, option_letters, partition_letter = build_mc_options(
        v, example_id
    )
    item["response_type"] = "mc"
    item["normative_choice"] = numeric_letter
    item["normative_percent"] = f"{v.posterior_a() * 100:.4g}"
    item["normative_open"] = _pct(v.posterior_a())
    item.update(
        _scoring_measure_fields(
            v,
            variant,
            partition_letter=partition_letter,
            normative_letter=numeric_letter,
        )
    )

    lines = [body, ""]
    for letter in "abcde":
        item[f"option_{letter}_label"] = ""
        item[f"option_{letter}_lure"] = ""
    for letter in option_letters:
        item[f"option_{letter.lower()}_label"] = labels[letter]
        item[f"option_{letter.lower()}_lure"] = lures[letter]
        lines.append(f"{letter}. {labels[letter]}")

    if variant.startswith("mc_full"):
        for letter, label, lure in (
            ("F", META_F, "insufficient information"),
            ("G", META_G, "inconsistent information"),
            ("H", META_H, "obviously incorrect premises"),
        ):
            item[f"option_{letter.lower()}_label"] = label
            item[f"option_{letter.lower()}_lure"] = lure
            lines.append(f"{letter}. {label}")
        numeric_letters = _mc_letter_list(option_letters)
        prompt = (
            "\n".join(lines)
            + _mc_suffix(letters=f"{numeric_letters}, F, G, or H")
        )
    else:
        prompt = "\n".join(lines) + _mc_suffix(letters=_mc_letter_list(option_letters))

    return prompt, item


def build_all() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    MC_SINGLE_ROUNDED_PERCENT_NOTICES.clear()
    vignettes = load_vignettes()
    prompts: list[dict[str, str]] = []
    items: list[dict[str, str]] = []

    for v in vignettes:
        for variant in VARIANTS:
            prompt, item = build_prompt(v, variant)
            prompts.append({"example_id": item["example_id"], "prompt": prompt})
            items.append(item)

    benchmark = build_benchmark(prompts, items)
    return prompts, items, benchmark


def build_benchmark(
    prompts: list[dict[str, str]], items: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Merge prompts, condition columns, and scoring metadata into one row per item."""
    pmap = {row["example_id"]: row["prompt"] for row in prompts}
    rows: list[dict[str, str]] = []
    for item in items:
        row = {k: item.get(k, "") for k in BENCHMARK_SCORING_FIELDS}
        row.update(
            {
                "example_id": item["example_id"],
                "vignette_name": item["vignette_name"],
                "problem_type": item["problem_type"],
                "intersection_size": item["intersection_size"],
                "response_type": item["response_format"],
                "has_statistics": item["has_statistics"],
                "variant": item["variant"],
                "prompt": pmap[item["example_id"]],
            }
        )
        rows.append(row)
    return rows


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


def write_csvs() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts, items, benchmark = build_all()

    if MC_SINGLE_ROUNDED_PERCENT_NOTICES:
        print(
            "\nNumeric MC prompts with only one distinct rounded percent "
            f"({len(MC_SINGLE_ROUNDED_PERCENT_NOTICES)}):",
            flush=True,
        )
        for notice in MC_SINGLE_ROUNDED_PERCENT_NOTICES:
            print(f"  - {notice}", flush=True)

    with (OUT_DIR / "prompts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "prompt"])
        writer.writeheader()
        writer.writerows(prompts)

    item_fields = [
        "example_id",
        "vignette_name",
        "variant",
        "well_posed",
        "normative",
        "p_c_and_d_given_a",
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

    with (OUT_DIR / "items.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=item_fields, extrasaction="ignore")
        writer.writeheader()
        for row in items:
            writer.writerow({k: row.get(k, "") for k in item_fields})

    with (OUT_DIR / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmark)

    return len(prompts)


def main() -> int:
    count = write_csvs()
    vignette_count = len(load_vignettes())
    print(
        f"Wrote {count} prompts ({vignette_count} vignettes × {len(VARIANTS)} variants)"
    )
    print(f"Output: {OUT_DIR} (prompts.csv, items.csv, benchmark.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
