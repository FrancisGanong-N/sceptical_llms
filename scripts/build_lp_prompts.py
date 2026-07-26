#!/usr/bin/env python3
"""Build LP prompts: linear programs with implicit (unstated) constraints.

Each vignette is a small optimization word problem. Some constraints are
"obvious" from the domain (solutions must be whole units, quantities cannot
be negative) but are never stated. A solver that optimizes only the stated
constraints reaches a different — wrong — optimum (the "naive LP" answer).

For every vignette with an implicit-constraint trap, the builder emits two
prompts that share the same keyed optimum:
  - ``condition=implicit``: constraints left unstated (the trap)
  - ``condition=explicit``: the same constraints spelled out in the prompt

The well-posed control vignette (``failure_mode=none``) is emitted once.

The response format is a JSON object with the optimal ``solution`` and the
final ``cost`` (objective value). Scoring accepts a parseable ``cost`` within
1% of ``true_objective``.

failure_mode column values:
  - none           well-posed control; the naive LP optimum is the true answer
  - integrality    naive optimum is fractional (e.g. 2.6 buses)
  - nonnegativity  naive optimum drives a quantity negative (or is unbounded)
  - both           naive optimum is fractional and negative

Constraint-type flags (true/false strings):
  - implicit_integer      integrality is part of the trap (integrality / both)
  - implicit_nonnegative  non-negativity is part of the trap (nonnegativity / both)
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

FAILURE_MODE_NONE = "none"
FAILURE_MODE_INTEGRALITY = "integrality"
FAILURE_MODE_NONNEGATIVITY = "nonnegativity"
FAILURE_MODE_BOTH = "both"

CONDITION_IMPLICIT = "implicit"
CONDITION_EXPLICIT = "explicit"
CONDITION_CONTROL = "control"

VARIANT = "json"
CANONICAL_VARIANTS = (VARIANT,)

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
)

ITEM_FIELDS = tuple(field for field in BENCHMARK_FIELDS if field != "prompt")


def slug(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return text.strip("_")


@dataclass(frozen=True)
class LpVignette:
    """One optimization word problem with an optional implicit-constraint trap."""

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
    # Empty for the well-posed control.
    explicit_addendum: str = ""

    @property
    def well_posed(self) -> bool:
        return self.failure_mode == FAILURE_MODE_NONE

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


# Optima verified by enumeration in tests/test_build_lp_prompts.py.
LP_VIGNETTES: tuple[LpVignette, ...] = (
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
    ),
    LpVignette(
        name="gift baskets",
        failure_mode=FAILURE_MODE_BOTH,
        narrative=(
            "A bakery has a standing order for exactly 20 gift baskets. A "
            "deluxe basket uses 2.5 jars of jam and earns a $7.25 profit, and a "
            "standard basket uses 0.75 jars of jam and earns a $4.10 profit. The "
            "bakery has 52.625 jars of jam in stock."
        ),
        explicit_addendum=(
            "The numbers of deluxe and standard baskets must be whole numbers "
            "and non-negative (no fractional baskets, and no negative counts)."
        ),
        objective_name="total profit",
        question="What mix of baskets fills the order with the highest total profit?",
        solution_keys=("deluxe", "standard"),
        true_solution={"deluxe": 20, "standard": 0},
        true_objective="145",
        naive_objective="149.725",
    ),
    LpVignette(
        name="workshop vehicles",
        failure_mode=FAILURE_MODE_NONE,
        narrative=(
            "A workshop builds go-karts and mini-bikes. Each go-kart needs 2.4 "
            "hours of welding and 0.75 hour of painting, and each mini-bike "
            "needs 0.8 hour of welding and 1.5 hours of painting. The workshop "
            "has at most 12 hours of welding and at most 7.5 hours of painting "
            "available. The profit is $42.50 per go-kart and $53.75 per mini-bike."
        ),
        explicit_addendum="",
        objective_name="total profit",
        question="What production plan gives the highest total profit?",
        solution_keys=("go_karts", "mini_bikes"),
        true_solution={"go_karts": 4, "mini_bikes": 3},
        true_objective="331.25",
        naive_objective="331.25",
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


def _conditions_for_vignette(v: LpVignette) -> tuple[str, ...]:
    if v.well_posed:
        return (CONDITION_CONTROL,)
    return (CONDITION_IMPLICIT, CONDITION_EXPLICIT)


def _shared_item_fields(v: LpVignette, *, condition: str) -> dict[str, str]:
    if condition == CONDITION_CONTROL:
        problem_type = "well_posed"
        normative = "well_posed"
        well_posed = "true"
    elif condition == CONDITION_EXPLICIT:
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
    }


def build_prompt(
    v: LpVignette,
    variant: str = VARIANT,
    *,
    condition: str | None = None,
) -> tuple[str, dict[str, str]]:
    if variant != VARIANT:
        raise ValueError(f"unknown variant: {variant}")
    resolved = condition or (
        CONDITION_CONTROL if v.well_posed else CONDITION_IMPLICIT
    )
    if resolved == CONDITION_EXPLICIT:
        example_id = f"{v.example_prefix()}__explicit__{variant}"
    else:
        example_id = f"{v.example_prefix()}__{variant}"
    item: dict[str, str] = {
        "example_id": example_id,
        "variant": variant,
        "response_type": "json",
        **_shared_item_fields(v, condition=resolved),
    }
    return _format_json_prompt(v, condition=resolved), item


def build_all() -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]
]:
    prompts: list[dict[str, str]] = []
    items: list[dict[str, str]] = []
    for vignette in load_lp_vignettes():
        for condition in _conditions_for_vignette(vignette):
            for variant in CANONICAL_VARIANTS:
                prompt, item = build_prompt(
                    vignette, variant, condition=condition
                )
                prompts.append(
                    {"example_id": item["example_id"], "prompt": prompt}
                )
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
    n_implicit = sum(1 for v in load_lp_vignettes() if not v.well_posed)
    print(
        f"Wrote {count} prompts "
        f"({len(load_lp_vignettes())} vignettes; "
        f"{n_implicit} implicit+explicit pairs + "
        f"{len(load_lp_vignettes()) - n_implicit} control; "
        f"failure modes: {modes})"
    )
    print(f"Output: {OUT_DIR} (prompts.csv, items.csv, benchmark.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
