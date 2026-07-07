#!/usr/bin/env python3
"""Build a plain-text study sheet of simple benchmark scenario language."""

from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_CSV = ROOT / "data" / "simple" / "benchmark.csv"
OUT_PATH = ROOT / "docs" / "simple-benchmark-scenario-study-sheet.txt"

PARTITION_ORDER = [
    "discharged weapon (last year)",
    "CA Trump voter",
    "healthcare employment",
    "military overseas (federal pool)",
    "covid vaccine (blue/red)",
    "NAEP grade 4 reading (MA vs NM)",
    "HS graduation ACGR (WV vs AZ)",
    "fantasy sports (male vs female)",
    "physician vs PhD research",
    "tax MFJ single CTC",
    "homeownership under 35 vs 65",
]

OVERLAP_ORDER = [
    "diabetes insulin obese",
    "english teacher humanities",
    "NFL MLB watch attend",
    "fantasy sports (under 45 vs male)",
    "book drama streaming",
    "dog cat household",
    "youtube facebook news",
    "college grad professional job",
    "homeowner suburban mortgage",
    "parent married dual income",
    "republican gun owner ban",
]

PROBLEM_TYPE_LABEL = {
    "well_posed": "natural (well-posed)",
    "altered": "altered",
}


def _pct(value: str) -> str:
    try:
        x = float(value)
    except ValueError:
        return value
    if x < 0.01:
        return f"{x * 100:.2f}%"
    if x < 1:
        return f"{x * 100:.1f}%".replace(".0%", "%")
    return f"{x:.4g}"


def _scenario_body(prompt: str) -> str:
    text = prompt.strip()
    for marker in (
        "\n\nReply with a percentage",
        "\n\nWhich answer is closest?",
        "\n\nA. About",
    ):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
    return text


def _prob_line(row: dict[str, str]) -> str:
    parts = [
        f"P(C)={_pct(row['p_c'])}",
        f"P(D)={_pct(row['p_d'])}",
    ]
    if row["intersection_size"] not in ("", "0"):
        parts.append(f"P(C∩D)={_pct(row['p_c_and_d_given_a'])}")
    parts.extend(
        [
            f"P(T|C)={_pct(row['p_t_given_c'])}",
            f"P(T|D)={_pct(row['p_t_given_d'])}",
            f"answer={row['normative_open']}",
        ]
    )
    return "  ".join(parts)


def _load_rows() -> list[dict[str, str]]:
    with BENCHMARK_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_section(title: str, names: list[str]) -> list[str]:
    lines = [title, ""]
    for i, name in enumerate(names, 1):
        lines.append(f"  {i:2}. {name}")
    lines.append("")
    return lines


def _vignette_block(row: dict[str, str]) -> list[str]:
    name = row["vignette_name"]
    kind = (
        "overlap"
        if row["intersection_size"] not in ("", "0")
        else "partition"
    )
    problem = PROBLEM_TYPE_LABEL.get(row["problem_type"], row["condition"])
    header = f"{name.upper()}  [{kind}; {problem}]"
    sep = "=" * max(80, len(header) + 4)
    lines = [
        sep,
        header,
        sep,
        f"example_id: {row['example_id']}",
        _prob_line(row),
        "",
        _scenario_body(row["prompt"]),
        "",
    ]
    return lines


def build_study_sheet() -> str:
    rows = _load_rows()
    by_name_variant: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["vignette_name"], row["condition"], row["variant"])
        by_name_variant[key] = row

    lines: list[str] = [
        "=" * 80,
        "SIMPLE BENCHMARK — SCENARIO LANGUAGE STUDY SHEET",
        "=" * 80,
        "",
        f"Generated: {date.today().isoformat()}",
        f"Source: {BENCHMARK_CSV.relative_to(ROOT)}",
        "Format: mc_prob scenario text (consultant intro + statistics + question).",
        "22 vignettes × natural condition only (altered/implausible excluded).",
        "",
        "QUICK INDEX",
        "-" * 80,
    ]
    lines.extend(_index_section("Partition (disjoint C and D):", PARTITION_ORDER))
    lines.extend(_index_section("Overlap (explicit P(C∩D) in prompt):", OVERLAP_ORDER))

    lines.extend(["=" * 80, "PART I — PARTITION VIGNETTES (natural)", "=" * 80, ""])
    for name in PARTITION_ORDER:
        row = by_name_variant.get((name, "natural", "mc_prob"))
        if row:
            lines.extend(_vignette_block(row))

    lines.extend(["=" * 80, "PART II — OVERLAP VIGNETTES (natural)", "=" * 80, ""])
    for name in OVERLAP_ORDER:
        row = by_name_variant.get((name, "natural", "mc_prob"))
        if row:
            lines.extend(_vignette_block(row))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    text = build_study_sheet()
    OUT_PATH.write_text(text, encoding="utf-8")
    vignette_count = len(PARTITION_ORDER) + len(OVERLAP_ORDER)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} ({vignette_count} vignettes, natural only)")


if __name__ == "__main__":
    main()
