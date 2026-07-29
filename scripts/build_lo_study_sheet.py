#!/usr/bin/env python3
"""Build study sheets for LO benchmark prompts.

Writes:

- ``docs/lo-benchmark-study-sheet.txt`` — non-explicit prompts only (implicit JSON
  + tacit audits; no explicit JSON parallels).
- ``docs/lo-benchmark-study-sheet-all-prompts.txt`` — every benchmark prompt
  (36 in v5).

Each entry includes the prompt, keyed answer, and model responses from
downloaded Kaggle runs when available.
"""

from __future__ import annotations

import csv
import sys
import textwrap
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.kaggle_runs import (  # noqa: E402
    DEFAULT_LO_TASK_SLUG,
    merged_lo_results_from_kaggle_runs,
)

WIDTH = 80
SEP = "=" * WIDTH
SUB = "-" * WIDTH
OUT_PATH = ROOT / "docs" / "lo-benchmark-study-sheet.txt"
OUT_PATH_ALL = ROOT / "docs" / "lo-benchmark-study-sheet-all-prompts.txt"
BENCHMARK_CSV = ROOT / "data" / "lp" / "benchmark.csv"
NON_EXPLICIT_CONDITIONS = frozenset({"implicit", "control"})
AUDIT_VARIANTS = frozenset({"needs_tacit_constraint", "detects_tacit_violation"})


def _short_model(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def _score_label(row: dict[str, str]) -> str:
    score = str(row.get("score", "")).strip().lower()
    if score == "true":
        return "CORRECT"
    if score == "false":
        return "WRONG"
    return score or "n/a"


def _wrap_line(text: str, *, subsequent_indent: str = "") -> list[str]:
    text = text.rstrip()
    if not text:
        return [""]
    return textwrap.wrap(
        text,
        width=WIDTH,
        subsequent_indent=subsequent_indent,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


def _wrap_block(text: str) -> list[str]:
    """Word-wrap a multi-paragraph block to WIDTH, preserving blank lines."""
    out: list[str] = []
    paragraphs = (text or "").replace("\r\n", "\n").split("\n")
    for para in paragraphs:
        if not para.strip():
            out.append("")
            continue
        stripped = para.lstrip(" ")
        indent = para[: len(para) - len(stripped)]
        available = max(20, WIDTH - len(indent))
        wrapped = textwrap.wrap(
            stripped,
            width=available,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not wrapped:
            out.append(indent.rstrip() or "")
            continue
        out.extend(indent + line for line in wrapped)
    return out


def _item_sort_key(row: dict[str, str]) -> tuple[str, int, str]:
    variant = (row.get("variant") or "").strip()
    condition = (row.get("condition") or "").strip()
    if variant == "json" and condition == "implicit":
        order = 0
    elif variant == "json" and condition == "explicit":
        order = 1
    elif variant == "needs_tacit_constraint":
        order = 2
    elif variant == "detects_tacit_violation":
        order = 3
    else:
        order = 99
    return (row.get("vignette_name", ""), order, row.get("example_id", ""))


def _load_benchmark_items(
    root: Path,
    *,
    include: Callable[[dict[str, str]], bool] | None = None,
) -> list[dict[str, str]]:
    path = root / "data" / "lp" / "benchmark.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if include is not None:
        rows = [row for row in rows if include(row)]
    return sorted(rows, key=_item_sort_key)


def _non_explicit_item(row: dict[str, str]) -> bool:
    condition = (row.get("condition") or "").strip()
    variant = (row.get("variant") or "").strip()
    if variant in AUDIT_VARIANTS:
        return True
    return condition in NON_EXPLICIT_CONDITIONS


def _load_non_explicit_items(root: Path) -> list[dict[str, str]]:
    return _load_benchmark_items(root, include=_non_explicit_item)


def _load_all_items(root: Path) -> list[dict[str, str]]:
    return _load_benchmark_items(root)


def _load_result_rows(
    root: Path,
    *,
    example_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    kaggle_dir = root / "data" / "kaggle_runs" / DEFAULT_LO_TASK_SLUG
    if not kaggle_dir.is_dir():
        return []
    try:
        merged = merged_lo_results_from_kaggle_runs(
            kaggle_dir,
            benchmark_path=root / "data" / "lp" / "benchmark.csv",
            fill_missing=False,
        )
    except (FileNotFoundError, ValueError):
        return []
    if example_ids is None:
        return merged
    return [row for row in merged if row.get("example_id") in example_ids]


def _header_line(item: dict[str, str]) -> str:
    vignette = item.get("vignette_name", "")
    failure = item.get("failure_mode", "")
    condition = item.get("condition", "")
    variant = item.get("variant", "")
    flags = []
    if str(item.get("implicit_integer", "")).lower() == "true":
        flags.append("integer")
    if str(item.get("implicit_nonnegative", "")).lower() == "true":
        flags.append("nonnegative")
    flag_text = "+".join(flags) if flags else "none"
    return (
        f"{vignette}  [{failure}]  variant={variant}  condition={condition}  "
        f"implicit constraints: {flag_text}"
    )


def _append_keyed_answer(lines: list[str], item: dict[str, str]) -> None:
    variant = (item.get("variant") or "").strip()
    if variant in AUDIT_VARIANTS:
        lines.append("KEYED ANSWER")
        lines.append(SUB)
        choice = (item.get("normative_choice") or "").strip()
        lines.extend(_wrap_line(f"normative_choice: {choice or 'n/a'}"))
        a_label = (item.get("option_a_label") or "").strip()
        b_label = (item.get("option_b_label") or "").strip()
        if a_label:
            lines.extend(_wrap_line(f"A. {a_label}"))
        if b_label:
            lines.extend(_wrap_line(f"B. {b_label}"))
        if variant == "detects_tacit_violation":
            lines.extend(
                _wrap_line(
                    f"violating stub solution: {item.get('violating_solution', '')}"
                )
            )
            lines.extend(
                _wrap_line(
                    f"violating stub objective: {item.get('violating_objective', '')}"
                )
            )
        lines.append("")
        return

    lines.append("CORRECT SOLUTION")
    lines.append(SUB)
    lines.extend(_wrap_line(f"solution: {item.get('true_solution', '')}"))
    lines.extend(
        _wrap_line(
            f"objective ({item.get('objective_name', 'cost')}): "
            f"{item.get('true_objective', '')}"
        )
    )
    lines.extend(
        _wrap_line(
            f"naive stated-only objective: {item.get('naive_objective', '')}"
        )
    )
    lines.append("")


def _append_responses(
    lines: list[str],
    item: dict[str, str],
    group: list[dict[str, str]],
) -> None:
    lines.append("RESPONSES")
    lines.append(SUB)
    variant = (item.get("variant") or "").strip()
    is_audit = variant in AUDIT_VARIANTS

    if not group:
        lines.append("(no model responses yet)")
        lines.append("")
        lines.append(SUB)
        lines.append("")
        return

    for row in group:
        model = row.get("model", "")
        response = (row.get("llm_response") or "").rstrip()
        lines.extend(_wrap_line(f"{_short_model(model)}  [{_score_label(row)}]"))
        lines.extend(_wrap_line(f"  model: {model}", subsequent_indent="  "))
        if is_audit:
            lines.extend(
                _wrap_line(
                    f"  parsed_choice: {row.get('parsed_choice', '')}",
                    subsequent_indent="  ",
                )
            )
        else:
            parsed_sol = row.get("parsed_solution") or ""
            parsed_obj = row.get("parsed_objective") or row.get("parsed_percent") or ""
            lines.extend(
                _wrap_line(
                    f"  parsed_solution: {parsed_sol}",
                    subsequent_indent="  ",
                )
            )
            lines.extend(
                _wrap_line(
                    f"  parsed_objective: {parsed_obj}",
                    subsequent_indent="  ",
                )
            )
            lines.extend(
                _wrap_line(
                    f"  naive_lp_confusion: {row.get('naive_lp_confusion', '')}",
                    subsequent_indent="  ",
                )
            )
        lines.append("")
        if response:
            lines.extend(_wrap_block(response))
        else:
            lines.append("(empty response)")
        lines.append("")
        lines.append(SUB)
    lines.append("")


def build_study_sheet_text(
    items: list[dict[str, str]],
    result_rows: list[dict[str, str]],
    *,
    title: str,
    task_slug: str = DEFAULT_LO_TASK_SLUG,
) -> str:
    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        by_example[row["example_id"]].append(row)

    lines: list[str] = []
    lines.extend(_wrap_line(title))
    lines.extend(_wrap_line(f"Source runs: data/kaggle_runs/{task_slug}/"))
    lines.extend(
        _wrap_line(
            "For each prompt: full text, keyed answer, then model responses "
            "(prompts are included even when no runs exist yet)."
        )
    )
    lines.append("")

    for item in items:
        example_id = item.get("example_id", "")
        lines.append(SEP)
        lines.extend(_wrap_line(_header_line(item)))
        lines.extend(_wrap_line(f"example_id: {example_id}"))
        lines.append(SEP)
        lines.append("")
        lines.append("PROMPT")
        lines.append(SUB)
        lines.extend(_wrap_block((item.get("prompt") or "").rstrip()))
        lines.append("")
        _append_keyed_answer(lines, item)
        group = sorted(
            by_example.get(example_id, []),
            key=lambda r: r.get("model", ""),
        )
        _append_responses(lines, item, group)

    return "\n".join(lines).rstrip() + "\n"


def write_study_sheet(
    *,
    root: Path = ROOT,
    out_path: Path = OUT_PATH,
) -> Path:
    items = _load_non_explicit_items(root)
    if not items:
        raise ValueError("No non-explicit LO benchmark rows found.")
    example_ids = {row["example_id"] for row in items}
    result_rows = _load_result_rows(root, example_ids=example_ids)
    text = build_study_sheet_text(
        items,
        result_rows,
        title=(
            "LO benchmark study sheet - non-explicit prompts only "
            "(implicit JSON + tacit audits)"
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def write_all_prompts_study_sheet(
    *,
    root: Path = ROOT,
    out_path: Path = OUT_PATH_ALL,
) -> Path:
    items = _load_all_items(root)
    if not items:
        raise ValueError("No LO benchmark rows found.")
    result_rows = _load_result_rows(root)
    text = build_study_sheet_text(
        items,
        result_rows,
        title="LO benchmark study sheet - all prompts",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main() -> int:
    path = write_study_sheet()
    path_all = write_all_prompts_study_sheet()
    items = _load_non_explicit_items(ROOT)
    all_items = _load_all_items(ROOT)
    result_rows = _load_result_rows(ROOT)
    n_with = len({r["example_id"] for r in result_rows})
    n_m = len({r["model"] for r in result_rows}) if result_rows else 0
    print(f"Wrote {path}")
    print(f"Wrote {path_all}")
    print(
        f"Non-explicit prompts: {len(items)}; all prompts: {len(all_items)}; "
        f"with responses: {n_with}; models: {n_m}; result rows: {len(result_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
