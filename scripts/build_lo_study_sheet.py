#!/usr/bin/env python3
"""Build a study sheet for LO implicit-condition prompts.

Writes ``docs/lo-benchmark-study-sheet.txt`` with, for each implicit vignette:
prompt, keyed solution, and every model response from downloaded Kaggle runs.
"""

from __future__ import annotations

import sys
import textwrap
from collections import defaultdict
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
BENCHMARK_CSV = ROOT / "data" / "lp" / "benchmark.csv"


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
        # Keep indentation of the source line as a prefix when wrapping.
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


def _load_implicit_rows(root: Path) -> list[dict[str, str]]:
    kaggle_dir = root / "data" / "kaggle_runs" / DEFAULT_LO_TASK_SLUG
    if not kaggle_dir.is_dir():
        raise FileNotFoundError(
            f"No runs under {kaggle_dir}. Download with lo_results.ipynb "
            f"or: python -m kaggle benchmarks tasks download "
            f"{DEFAULT_LO_TASK_SLUG} -o {kaggle_dir}"
        )
    merged = merged_lo_results_from_kaggle_runs(
        kaggle_dir,
        benchmark_path=root / "data" / "lp" / "benchmark.csv",
        fill_missing=False,
    )
    return [row for row in merged if (row.get("condition") or "").strip() == "implicit"]


def build_study_sheet_text(rows: list[dict[str, str]]) -> str:
    by_vignette: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_vignette[row["vignette_name"]].append(row)

    lines: list[str] = []
    lines.extend(
        _wrap_line("LO benchmark study sheet - implicit conditions only")
    )
    lines.extend(
        _wrap_line(f"Source runs: data/kaggle_runs/{DEFAULT_LO_TASK_SLUG}/")
    )
    lines.extend(
        _wrap_line(
            "For each vignette: prompt, keyed solution, then model responses."
        )
    )
    lines.append("")

    for vignette in sorted(by_vignette):
        group = sorted(by_vignette[vignette], key=lambda r: r.get("model", ""))
        sample = group[0]
        failure = sample.get("failure_mode", "")
        flags = []
        if str(sample.get("implicit_integer", "")).lower() == "true":
            flags.append("integer")
        if str(sample.get("implicit_nonnegative", "")).lower() == "true":
            flags.append("nonnegative")
        flag_text = "+".join(flags) if flags else "none"

        lines.append(SEP)
        lines.extend(
            _wrap_line(
                f"{vignette}  [{failure}]  implicit constraints: {flag_text}"
            )
        )
        lines.extend(
            _wrap_line(f"example_id: {sample.get('example_id', '')}")
        )
        lines.append(SEP)
        lines.append("")
        lines.append("PROMPT")
        lines.append(SUB)
        lines.extend(_wrap_block((sample.get("prompt") or "").rstrip()))
        lines.append("")
        lines.append("CORRECT SOLUTION")
        lines.append(SUB)
        lines.extend(
            _wrap_line(f"solution: {sample.get('true_solution', '')}")
        )
        lines.extend(
            _wrap_line(
                f"objective ({sample.get('objective_name', 'cost')}): "
                f"{sample.get('true_objective', '')}"
            )
        )
        lines.extend(
            _wrap_line(
                f"naive stated-only objective: {sample.get('naive_objective', '')}"
            )
        )
        lines.append("")
        lines.append("RESPONSES")
        lines.append(SUB)

        for row in group:
            model = row.get("model", "")
            parsed_sol = row.get("parsed_solution") or ""
            parsed_obj = row.get("parsed_objective") or row.get("parsed_percent") or ""
            response = (row.get("llm_response") or "").rstrip()
            lines.extend(
                _wrap_line(f"{_short_model(model)}  [{_score_label(row)}]")
            )
            lines.extend(_wrap_line(f"  model: {model}", subsequent_indent="  "))
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

    return "\n".join(lines).rstrip() + "\n"


def write_study_sheet(
    *,
    root: Path = ROOT,
    out_path: Path = OUT_PATH,
) -> Path:
    rows = _load_implicit_rows(root)
    if not rows:
        raise ValueError("No implicit-condition rows found in merged LO results.")
    text = build_study_sheet_text(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main() -> int:
    path = write_study_sheet()
    rows = _load_implicit_rows(ROOT)
    n_v = len({r["vignette_name"] for r in rows})
    n_m = len({r["model"] for r in rows})
    print(f"Wrote {path}")
    print(f"Implicit vignettes: {n_v}; models: {n_m}; rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
