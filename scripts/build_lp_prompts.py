#!/usr/bin/env python3
"""Build LP prompts: linear programs with unstated (tacit) constraints.

Each vignette is a small optimization word problem. Some constraints are
obvious from the domain (solutions must be whole units, quantities cannot
be negative) but are left unstated in the implicit condition. A solver that
optimizes only the stated constraints may reach a different — wrong —
optimum (the "naive LP" answer).

For every trap vignette the builder emits:

  - ``variant=json``, ``condition=implicit``: solve; constraints unstated
  - ``variant=json``, ``condition=explicit``: solve; constraints spelled out
  - ``variant=needs_tacit_constraint``: A/B — does solving require unstated
    constraints? (keyed **A** = yes)
  - ``variant=detects_tacit_violation``: A/B — is a tacit-violating stub plan
    sensible? (keyed **B** = no)

The response format for ``json`` is a JSON object with the optimal
``solution`` and the final ``cost`` (objective value). Scoring accepts a
parsable ``cost`` within 1% of ``true_objective``. Audit variants ask for
a letter (A or B).

failure_mode column values:
  - integrality    naive optimum is fractional (e.g. 2.6 buses)
  - nonnegativity  non-negativity is an unstated domain constraint
  - both           (reserved; not used in v4 vignette set)

Constraint-type flags (true/false strings):
  - implicit_integer      integrality is part of the trap
  - implicit_nonnegative  non-negativity is part of the trap
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "lp"

INTRO = (
    "You are an operations consultant. Your task is to recommend the best "
    "plan from the information below."
)

NEEDS_TACIT_INTRO = (
    "You are reviewing an operations-planning problem. Decide whether a "
    "sensible optimal plan requires applying constraints that are not "
    "stated in the problem text."
)
NEEDS_TACIT_QUESTION = (
    "Does finding a sensible optimal plan for this problem require applying "
    "any constraints that are not stated above (for example, that counts "
    "must be whole numbers, or that quantities cannot be negative)?"
)
NEEDS_TACIT_YES = (
    "Yes — unstated constraints (such as whole numbers or non-negativity) "
    "are needed"
)
NEEDS_TACIT_NO = (
    "No — the stated information is enough as written; no unstated "
    "constraints are required"
)

DETECTS_VIOLATION_INTRO = (
    "You are an operations auditor, checking a proposed plan for another "
    "consultant."
)
DETECTS_VIOLATION_QUESTION = (
    "Is this a sensible plan to recommend to the client?"
)
DETECTS_VIOLATION_YES = "Yes — this plan is sensible"
DETECTS_VIOLATION_NO = "No — this plan is not appropriate"

FAILURE_MODE_INTEGRALITY = "integrality"
FAILURE_MODE_NONNEGATIVITY = "nonnegativity"
FAILURE_MODE_BOTH = "both"

CONDITION_IMPLICIT = "implicit"
CONDITION_EXPLICIT = "explicit"

VARIANT_JSON = "json"
VARIANT_NEEDS_TACIT = "needs_tacit_constraint"
VARIANT_DETECTS_VIOLATION = "detects_tacit_violation"
CANONICAL_VARIANTS = (
    VARIANT_JSON,
    VARIANT_NEEDS_TACIT,
    VARIANT_DETECTS_VIOLATION,
)

BENCHMARK_FIELDS = (
    "example_id",
    "vignette_name",
    "failure_mode",
    "condition",
    "problem_type",
    "intersection_size",
    "response_type",
    "has_statistics",
    "variant",
    "prompt",
    "well_posed",
    "normative",
    "true_objective",
    "naive_objective",
    "true_solution",
    "solution_keys",
    "objective_name",
    "implicit_integer",
    "implicit_nonnegative",
    "normative_percent",
    "normative_choice",
    "confidence_required",
    "scepticism_required",
    "scepticism_score_target",
    "option_a_label",
    "option_b_label",
    "violating_solution",
    "violating_objective",
)

ITEM_FIELDS = tuple(field for field in BENCHMARK_FIELDS if field != "prompt")


def slug(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return text.strip("_")


@dataclass(frozen=True)
class LpVignette:
    """One optimization word problem with an unstated-constraint trap."""

    name: str
    failure_mode: str
    narrative: str
    objective_name: str
    question: str
    solution_keys: tuple[str, ...]
    true_solution: dict[str, float]
    true_objective: str
    naive_objective: str
    # Spelled-out constraint sentence(s) for the explicit parallel prompt.
    explicit_addendum: str
    # Stub plan that violates a tacit constraint (for detects_tacit_violation).
    violating_solution: dict[str, float]
    violating_objective: str

    @property
    def well_posed(self) -> bool:
        return False

    @property
    def implicit_integer(self) -> bool:
        return self.failure_mode in {
            FAILURE_MODE_INTEGRALITY,
            FAILURE_MODE_BOTH,
        }

    @property
    def implicit_nonnegative(self) -> bool:
        return self.failure_mode in {
            FAILURE_MODE_NONNEGATIVITY,
            FAILURE_MODE_BOTH,
        }

    def example_prefix(self) -> str:
        return f"{slug(self.name)}__{self.failure_mode}"

    def true_solution_json(self) -> str:
        return json.dumps(self.true_solution, sort_keys=True)

    def violating_solution_json(self) -> str:
        return json.dumps(self.violating_solution, sort_keys=True)


# Optima verified by enumeration in tests/test_build_lp_prompts.py.
LP_VIGNETTES: tuple[LpVignette, ...] = (
    LpVignette(
        name="carpenter furniture",
        failure_mode=FAILURE_MODE_INTEGRALITY,
        narrative=(
            "A carpenter builds bookcases and desks for sale. Each bookcase "
            "takes 1.5 hours of cutting and 2.25 hours of assembly, and each desk "
            "takes 2.25 hours of cutting and 1.5 hours of assembly. This week the "
            "carpenter has at most 9 hours of cutting time and at most 9 "
            "hours of assembly time. The profit is $37.50 per bookcase and $50 "
            "per desk."
        ),
        explicit_addendum=(
            "Bookcases and desks must be whole numbers (no fractional furniture)."
        ),
        objective_name="total profit",
        question="What production plan gives the highest total profit for the week?",
        solution_keys=("bookcases", "desks"),
        true_solution={"bookcases": 0, "desks": 4},
        true_objective="200",
        naive_objective="210",
        violating_solution={"bookcases": 2.4, "desks": 2.4},
        violating_objective="210",
    ),
    LpVignette(
        name="print shop",
        failure_mode=FAILURE_MODE_INTEGRALITY,
        narrative=(
            "A print shop produces posters and booklets for sale. Each poster "
            "takes 1.5 hours of printing and 2.25 hours of binding, and each "
            "booklet takes 2.25 hours of printing and 1.5 hours of binding. This "
            "week the shop has at most 9 hours of printing time and at most 9 "
            "hours of binding time. The profit is $37.50 per poster and $50 per "
            "booklet."
        ),
        explicit_addendum=(
            "Posters and booklets must be whole numbers (no fractional print jobs)."
        ),
        objective_name="total profit",
        question="What production plan gives the highest total profit for the week?",
        solution_keys=("posters", "booklets"),
        true_solution={"posters": 0, "booklets": 4},
        true_objective="200",
        naive_objective="210",
        violating_solution={"posters": 2.4, "booklets": 2.4},
        violating_objective="210",
    ),
    LpVignette(
        name="pottery studio",
        failure_mode=FAILURE_MODE_INTEGRALITY,
        narrative=(
            "A pottery studio makes bowls and vases for sale. Each bowl takes "
            "1.65 hours of throwing and 2.025 hours of glazing, and each vase "
            "takes 2.025 hours of throwing and 1.65 hours of glazing. This week "
            "the studio has at most 9 hours of throwing time and at most 9 "
            "hours of glazing time. The profit is $41.25 per bowl and $45 per "
            "vase."
        ),
        explicit_addendum=(
            "Bowls and vases must be whole numbers (no fractional pieces)."
        ),
        objective_name="total profit",
        question="What production plan gives the highest total profit for the week?",
        solution_keys=("bowls", "vases"),
        true_solution={"bowls": 0, "vases": 4},
        true_objective="180",
        naive_objective="211.22",
        violating_solution={"bowls": 2.44898, "vases": 2.44898},
        violating_objective="211.22",
    ),
    LpVignette(
        name="charter buses",
        failure_mode=FAILURE_MODE_INTEGRALITY,
        narrative=(
            "A school must transport at least 130 students to a regional "
            "competition. A large bus seats 50 students and costs $820.50 to "
            "charter, and a small bus seats 30 students and costs $700.25 to "
            "charter."
        ),
        explicit_addendum=(
            "The numbers of large and small buses must be whole numbers "
            "(no fractional buses)."
        ),
        objective_name="total cost",
        question="What charter plan meets the requirement at the lowest total cost?",
        solution_keys=("large_buses", "small_buses"),
        true_solution={"large_buses": 2, "small_buses": 1},
        true_objective="2341.25",
        naive_objective="2133.3",
        violating_solution={"large_buses": 2.6, "small_buses": 0},
        violating_objective="2133.3",
    ),
    LpVignette(
        name="fund allocation",
        failure_mode=FAILURE_MODE_NONNEGATIVITY,
        narrative=(
            "An investor may invest up to $10,000 split between Fund A and "
            "Fund B. Fund A returns 9.0% per year and Fund B returns 4.5% per "
            "year. Each dollar in Fund A uses 1.5 risk units and each dollar in "
            "Fund B uses 0.5 risk units; the investor's risk budget is 9,000 "
            "units. Fund A accepts at most $8,500 from any one client."
        ),
        explicit_addendum=(
            "The amount invested in each fund must be non-negative "
            "(no shorting / borrowing against a fund)."
        ),
        objective_name="total return",
        question="What allocation gives the highest total return in the first year?",
        solution_keys=("fund_a", "fund_b"),
        true_solution={"fund_a": 4000, "fund_b": 6000},
        true_objective="630",
        # Without non-negativity the same corner remains optimal for this LP;
        # the trap is still unstated non-negativity under a non-trivial mix.
        naive_objective="630",
        # Hand-crafted violating stub (negative Fund B).
        violating_solution={"fund_a": 8500, "fund_b": -7500},
        violating_objective="427.5",
    ),
    LpVignette(
        name="warehouse shipping",
        failure_mode=FAILURE_MODE_NONNEGATIVITY,
        narrative=(
            "A retailer needs at least 16 pallets delivered to a store. "
            "Warehouse 1 ships for $2.50 per pallet and uses 1.0 hour of dock "
            "time per pallet; Warehouse 2 ships for $4.00 per pallet and uses "
            "0.5 hour of dock time per pallet. The store has at most 12 hours "
            "of dock time available."
        ),
        explicit_addendum=(
            "Shipments from each warehouse must be non-negative "
            "(a warehouse cannot ship a negative number of pallets)."
        ),
        objective_name="total cost",
        question=(
            "What shipping plan meets the store's requirement at the lowest "
            "total cost?"
        ),
        solution_keys=("warehouse_1", "warehouse_2"),
        true_solution={"warehouse_1": 8, "warehouse_2": 8},
        true_objective="52",
        naive_objective="52",
        # Hand-crafted violating stub (negative Warehouse 2 shipment).
        violating_solution={"warehouse_1": 20, "warehouse_2": -4},
        violating_objective="34",
    ),
)


def load_lp_vignettes() -> tuple[LpVignette, ...]:
    return LP_VIGNETTES


def _json_reply_instruction(v: LpVignette) -> str:
    solution_fields = ", ".join(f'"{key}": <number>' for key in v.solution_keys)
    schematic = {
        "solution": {key: 0 for key in v.solution_keys},
        "cost": 0,
    }
    return (
        "Reply in this format:\n"
        "1. Output the JSON object first on its own line.\n"
        "2. Then on the next line any comments, qualifications, etc.\n"
        "\n"
        "JSON object of this form:\n"
        f'{{"solution": {{{solution_fields}}}, "cost": <number>}}\n'
        "The `solution` object is the optimal plan. The `cost` field is the "
        f"final {v.objective_name} under that plan.\n"
        f"Example shape (values are placeholders): {json.dumps(schematic)}\n"
        "Do not include a dollar sign or markdown fences around the JSON."
    )


def _narrative_for_condition(v: LpVignette, condition: str) -> str:
    if condition != CONDITION_EXPLICIT:
        return v.narrative
    addendum = (v.explicit_addendum or "").strip()
    if not addendum:
        raise ValueError(
            f"vignette {v.name!r} has no explicit_addendum for explicit condition"
        )
    return f"{v.narrative} {addendum}"


def _format_json_prompt(v: LpVignette, *, condition: str) -> str:
    return "\n".join(
        [
            INTRO,
            "",
            _narrative_for_condition(v, condition),
            "",
            v.question,
            "",
            _json_reply_instruction(v),
        ]
    )


def _format_needs_tacit_prompt(v: LpVignette) -> str:
    return "\n".join(
        [
            NEEDS_TACIT_INTRO,
            "",
            v.narrative,
            "",
            NEEDS_TACIT_QUESTION,
            "",
            f"A. {NEEDS_TACIT_YES}",
            f"B. {NEEDS_TACIT_NO}",
            "",
            "Which answer is closest? Reply with only the letter (A or B).",
        ]
    )


def _format_detects_violation_prompt(v: LpVignette) -> str:
    stub = {
        "solution": v.violating_solution,
        "cost": float(v.violating_objective),
    }
    embedded = "\n".join(
        [
            INTRO,
            "",
            v.narrative,
            "",
            v.question,
            "",
            _json_reply_instruction(v),
        ]
    )
    return "\n".join(
        [
            DETECTS_VIOLATION_INTRO,
            "",
            "The consultant was given the following prompt:",
            "",
            embedded,
            "",
            "And proposed the following plan:",
            "",
            json.dumps(stub, sort_keys=True),
            "",
            DETECTS_VIOLATION_QUESTION,
            "",
            f"A. {DETECTS_VIOLATION_YES}",
            f"B. {DETECTS_VIOLATION_NO}",
            "",
            "Which answer is closest? Reply with only the letter (A or B).",
        ]
    )


def _shared_item_fields(v: LpVignette, *, condition: str) -> dict[str, str]:
    if condition == CONDITION_EXPLICIT:
        problem_type = "explicit_constraints"
        normative = "explicit_constraints"
        well_posed = "true"
    else:
        problem_type = "implicit_constraints"
        normative = "implicit_constraints"
        well_posed = "false"
    return {
        "vignette_name": v.name,
        "failure_mode": v.failure_mode,
        "condition": condition,
        "problem_type": problem_type,
        "intersection_size": "",
        "has_statistics": "true",
        "well_posed": well_posed,
        "normative": normative,
        "true_objective": v.true_objective,
        "naive_objective": v.naive_objective,
        "true_solution": v.true_solution_json(),
        "solution_keys": ",".join(v.solution_keys),
        "objective_name": v.objective_name,
        "implicit_integer": str(v.implicit_integer).lower(),
        "implicit_nonnegative": str(v.implicit_nonnegative).lower(),
        "normative_percent": v.true_objective,
        "normative_choice": "",
        "confidence_required": "false",
        "scepticism_required": "false",
        "scepticism_score_target": "n/a",
        "option_a_label": "",
        "option_b_label": "",
        "violating_solution": v.violating_solution_json(),
        "violating_objective": v.violating_objective,
    }


def _example_id(v: LpVignette, variant: str, *, condition: str) -> str:
    if variant == VARIANT_JSON and condition == CONDITION_EXPLICIT:
        return f"{v.example_prefix()}__explicit__{variant}"
    if variant == VARIANT_JSON:
        return f"{v.example_prefix()}__{variant}"
    return f"{v.example_prefix()}__{variant}"


def build_prompt(
    v: LpVignette,
    variant: str = VARIANT_JSON,
    *,
    condition: str | None = None,
) -> tuple[str, dict[str, str]]:
    if variant not in CANONICAL_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    if variant == VARIANT_JSON:
        resolved = condition or CONDITION_IMPLICIT
        if resolved not in {CONDITION_IMPLICIT, CONDITION_EXPLICIT}:
            raise ValueError(f"unknown condition for json variant: {resolved}")
        item: dict[str, str] = {
            "example_id": _example_id(v, variant, condition=resolved),
            "variant": variant,
            "response_type": "json",
            **_shared_item_fields(v, condition=resolved),
        }
        return _format_json_prompt(v, condition=resolved), item

    # Audit variants always use the implicit (unstated) narrative.
    resolved = CONDITION_IMPLICIT
    shared = _shared_item_fields(v, condition=resolved)
    if variant == VARIANT_NEEDS_TACIT:
        shared.update(
            {
                "problem_type": "needs_tacit_constraint",
                "normative": "needs_tacit_constraint",
                "normative_choice": "A",
                "scepticism_required": "true",
                "scepticism_score_target": "A",
                "option_a_label": NEEDS_TACIT_YES,
                "option_b_label": NEEDS_TACIT_NO,
            }
        )
        prompt = _format_needs_tacit_prompt(v)
        response_type = VARIANT_NEEDS_TACIT
    else:
        shared.update(
            {
                "problem_type": "detects_tacit_violation",
                "normative": "detects_tacit_violation",
                "normative_choice": "B",
                "scepticism_required": "true",
                "scepticism_score_target": "B",
                "option_a_label": DETECTS_VIOLATION_YES,
                "option_b_label": DETECTS_VIOLATION_NO,
            }
        )
        prompt = _format_detects_violation_prompt(v)
        response_type = VARIANT_DETECTS_VIOLATION

    item = {
        "example_id": _example_id(v, variant, condition=resolved),
        "variant": variant,
        "response_type": response_type,
        **shared,
    }
    return prompt, item


def build_all() -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]
]:
    prompts: list[dict[str, str]] = []
    items: list[dict[str, str]] = []
    for vignette in load_lp_vignettes():
        # Solve prompts: implicit + explicit.
        for condition in (CONDITION_IMPLICIT, CONDITION_EXPLICIT):
            prompt, item = build_prompt(
                vignette, VARIANT_JSON, condition=condition
            )
            prompts.append({"example_id": item["example_id"], "prompt": prompt})
            items.append(item)
        # Audit prompts: one each on the implicit narrative.
        for variant in (VARIANT_NEEDS_TACIT, VARIANT_DETECTS_VIOLATION):
            prompt, item = build_prompt(vignette, variant)
            prompts.append({"example_id": item["example_id"], "prompt": prompt})
            items.append(item)

    pmap = {row["example_id"]: row["prompt"] for row in prompts}
    benchmark: list[dict[str, str]] = []
    for item in items:
        row = {key: item.get(key, "") for key in BENCHMARK_FIELDS}
        row["prompt"] = pmap[item["example_id"]]
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
        writer = csv.DictWriter(
            handle, fieldnames=list(ITEM_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in items:
            writer.writerow({key: row.get(key, "") for key in ITEM_FIELDS})

    with (OUT_DIR / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(BENCHMARK_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(benchmark)

    return len(prompts)


def main() -> int:
    count = write_csvs()
    by_mode: dict[str, int] = {}
    for vignette in load_lp_vignettes():
        by_mode[vignette.failure_mode] = by_mode.get(vignette.failure_mode, 0) + 1
    modes = ", ".join(f"{mode}={n}" for mode, n in sorted(by_mode.items()))
    n_v = len(load_lp_vignettes())
    print(
        f"Wrote {count} prompts "
        f"({n_v} vignettes × "
        f"(json implicit+explicit + needs_tacit + detects_violation); "
        f"failure modes: {modes})"
    )
    print(f"Output: {OUT_DIR} (prompts.csv, items.csv, benchmark.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
