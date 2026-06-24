import argparse
import csv
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WIDTH = 80
SEP = "=" * WIDTH
SUB = "-" * WIDTH

INTERSECTION_SMALL = 0.10
INTERSECTION_LARGE = 0.25


def _fmt_prob(value: str) -> str:
    if not value:
        return "—"
    try:
        x = float(value)
    except ValueError:
        return value
    if x < 0.01 and x > 0:
        return f"{x:.4f}"
    return f"{x:.3f}"


def _short_label(text: str, max_len: int = 20) -> str:
    if not text:
        return ""
    s = text.split(":")[0].split("(")[0].strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _wrap(label: str, text: str) -> list[str]:
    prefix = f"  {label}: "
    return textwrap.wrap(
        text,
        width=WIDTH,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wrap_plain(text: str, *, indent: str = "  ") -> list[str]:
    return textwrap.wrap(
        text,
        width=WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _overlap_short(row: dict, *, well_posed: bool) -> str:
    if well_posed:
        return "partition"
    return row.get("normative", "overlap") or "overlap"


def _as_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _classify_intersection_ratio(ratio: float) -> str:
    if ratio <= 0:
        return "0"
    if ratio < INTERSECTION_SMALL:
        return "small"
    if ratio > INTERSECTION_LARGE:
        return "large"
    return "medium"


def _intersection_ratio(row: dict, *, well_posed: bool) -> float | None:
    qc = _as_float(row.get("P_C_given_A", ""))
    qd = _as_float(row.get("P_D_given_A", ""))
    if qc is None or qd is None:
        return None
    larger = max(qc, qd)
    if larger <= 0:
        return None
    if well_posed:
        return 0.0
    p_cd = _as_float(row.get("P_C_and_D_given_A", ""))
    if p_cd is not None:
        return p_cd / larger
    cd_lo = _as_float(row.get("P_C_and_D_given_A_min", ""))
    cd_hi = _as_float(row.get("P_C_and_D_given_A_max", ""))
    if cd_lo is None and cd_hi is None:
        cd_lo, cd_hi, _, _ = _frechet_bounds(qc, qd)
    if cd_lo is not None and cd_hi is not None:
        return ((cd_lo + cd_hi) / 2) / larger
    return None


def _intersection_size(row: dict, *, well_posed: bool) -> str:
    stored = (row.get("intersection_size") or "").strip()
    if stored:
        return stored
    ratio = _intersection_ratio(row, well_posed=well_posed)
    if ratio is None:
        return "?"
    return _classify_intersection_ratio(ratio)


def _intersection_size_line(row: dict, *, well_posed: bool) -> str:
    size = _intersection_size(row, well_posed=well_posed)
    ratio = _intersection_ratio(row, well_posed=well_posed)
    if well_posed or ratio is None:
        return f"intersection size: {size} (C and D exclusive under A)"
    return (
        f"intersection size: {size} "
        f"(P(C∩D|A) / max(P(C|A),P(D|A)) ≈ {ratio:.0%}; "
        f"small <{INTERSECTION_SMALL:.0%}, medium {INTERSECTION_SMALL:.0%}–"
        f"{INTERSECTION_LARGE:.0%}, large >{INTERSECTION_LARGE:.0%})"
    )


def _overlap_label(row: dict, *, well_posed: bool) -> str:
    qc = _as_float(row.get("P_C_given_A", ""))
    qd = _as_float(row.get("P_D_given_A", ""))
    if qc is None or qd is None:
        return "(invalid P(C|A) or P(D|A) in CSV — check quoting)"
    total = qc + qd
    if well_posed:
        return (
            f"partition (P(C|A)+P(D|A)={total:.3f}; "
            f"C and D exclusive under A)"
        )
    norm = row.get("normative", "")
    note = f"; P(C|A)+P(D|A)={total:.3f}"
    if norm == "inconsistent":
        return f"{norm}{note} (>1 if read as exclusive)"
    return f"{norm}{note} (C and D may overlap)"


def _p_a_given_t(row: dict, *, well_posed: bool) -> str:
    key = "P_A_given_T" if well_posed else "P_A_given_T_partition"
    val = row.get(key, "")
    if not well_posed:
        return _fmt_prob(val) + " *"
    return _fmt_prob(val)


def _frechet_bounds(qc: float, qd: float) -> tuple[float, float, float, float]:
    """Return (P(C∩D|A) lo/hi, P(C∪D|A) lo/hi) from marginals."""
    cd_lo = max(0.0, qc + qd - 1.0)
    cd_hi = min(qc, qd)
    or_lo = max(qc, qd)
    or_hi = min(1.0, qc + qd)
    return cd_lo, cd_hi, or_lo, or_hi


def _fmt_joint_cell(lo: float | None, hi: float | None, point: str) -> str:
    pt = _as_float(point)
    if pt is not None:
        return _fmt_prob(point)
    if lo is not None and hi is not None:
        if abs(lo - hi) < 1e-9:
            return _fmt_prob(str(lo))
        return f"[{_fmt_prob(str(lo))},{_fmt_prob(str(hi))}]"
    return "—"


def _overlap_joint_cells(row: dict) -> tuple[str, str]:
    qc = _as_float(row.get("P_C_given_A", ""))
    qd = _as_float(row.get("P_D_given_A", ""))
    cd_lo = _as_float(row.get("P_C_and_D_given_A_min", ""))
    cd_hi = _as_float(row.get("P_C_and_D_given_A_max", ""))
    or_lo = _as_float(row.get("P_C_or_D_given_A_min", ""))
    or_hi = _as_float(row.get("P_C_or_D_given_A_max", ""))
    if cd_lo is None and cd_hi is None and qc is not None and qd is not None:
        cd_lo, cd_hi, or_lo, or_hi = _frechet_bounds(qc, qd)
    return (
        _fmt_joint_cell(cd_lo, cd_hi, row.get("P_C_and_D_given_A", "")),
        _fmt_joint_cell(or_lo, or_hi, row.get("P_C_or_D_given_A", "")),
    )


def _numeric_table(row: dict, *, well_posed: bool) -> list[str]:
    headers = ("overlap", "P(A)", "P(C|A)", "P(D|A)", "P(A|T)")
    values = (
        _overlap_short(row, well_posed=well_posed),
        _fmt_prob(row.get("P_A", "")),
        _fmt_prob(row.get("P_C_given_A", "")),
        _fmt_prob(row.get("P_D_given_A", "")),
        _p_a_given_t(row, well_posed=well_posed),
    )
    widths = (12, 7, 7, 7, 7)
    lines = [
        "  " + " ".join(f"{h:>{w}}" for h, w in zip(headers, widths)),
        "  " + " ".join(f"{v:>{w}}" for v, w in zip(values, widths)),
    ]
    cause_parts = [
        f"A: {_short_label(row.get('A', ''))}",
        f"C: {_short_label(row.get('C', ''))}",
        f"D: {_short_label(row.get('D', ''))}",
    ]
    lines.extend(_wrap_plain("  ".join(cause_parts), indent="  "))
    if not well_posed:
        cd_cell, or_cell = _overlap_joint_cells(row)
        j_headers = ("P(C∩D|A)", "P(C∨D|A)")
        j_values = (cd_cell, or_cell)
        j_widths = (12, 12)
        lines.append(
            "  "
            + " ".join(f"{h:>{w}}" for h, w in zip(j_headers, j_widths))
        )
        lines.append(
            "  " + " ".join(f"{v:>{w}}" for v, w in zip(j_values, j_widths))
        )
        status = row.get("joint_status", "")
        if status:
            lines.extend(_wrap_plain(f"joint ({status})"))
        note = row.get("joint_note", "")
        if note:
            lines.extend(_wrap_plain(note))
    return lines


def format_row(row: dict, index: int, *, well_posed: bool) -> list[str]:
    lines = [SEP, f"{index}. {row['name']}"]
    lines.extend(_wrap_plain(_intersection_size_line(row, well_posed=well_posed)))
    lines.extend(_wrap_plain(_overlap_label(row, well_posed=well_posed)))
    if not well_posed:
        lines.extend(
            _wrap_plain(
                "(* P(A|T) is the partition shortcut if C and D are "
                "wrongly treated as disjoint)"
            )
        )
    lines.append(SUB)
    lines.extend(_numeric_table(row, well_posed=well_posed))
    lines.append("")
    lines.extend(_wrap("universe", row.get("universe", "")))
    lines.extend(_wrap("T", row.get("T", "")))
    lines.append("")
    for label in ("A", "C", "D", "N"):
        lines.extend(_wrap(label, row.get(label, "")))
    lines.append("")
    return lines


def format_table(path: Path, *, well_posed: bool, title: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = [title, ""]
    for i, row in enumerate(rows, 1):
        lines.extend(format_row(row, i, well_posed=well_posed))
    return lines


def _load_vignettes(path: Path, *, well_posed: bool) -> list[tuple[str, str, bool]]:
    """Return (name, intersection_size, well_posed) sorted by name."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        (
            row["name"],
            _intersection_size(row, well_posed=well_posed),
            well_posed,
        )
        for row in rows
    ]


def format_intersection_summary() -> list[str]:
    """Category × intersection-size summary across all vignette CSVs."""
    entries: list[tuple[str, str, bool]] = []
    entries.extend(
        _load_vignettes(
            ROOT / "docs" / "base-rate-two-cause-vignettes.csv",
            well_posed=True,
        )
    )
    entries.extend(
        _load_vignettes(
            ROOT / "docs" / "base-rate-overlap-vignettes.csv",
            well_posed=False,
        )
    )

    size_order = ("0", "small", "medium", "large", "?")
    by_size: dict[str, list[str]] = {s: [] for s in size_order}
    for name, size, well_posed in entries:
        label = name if well_posed else f"{name} (overlap)"
        if well_posed and any(
            n == name and not wp for n, _, wp in entries
        ):
            label = f"{name} (well-posed)"
        bucket = size if size in by_size else "?"
        by_size[bucket].append(label)

    lines = [
        "INTERSECTION SIZE BY CATEGORY",
        "",
        "Ratio = P(C∩D|A) / max(P(C|A), P(D|A)) under A.",
        f"Thresholds: 0 = exclusive partition; small <{INTERSECTION_SMALL:.0%}; "
        f"medium {INTERSECTION_SMALL:.0%}–{INTERSECTION_LARGE:.0%}; "
        f"large >{INTERSECTION_LARGE:.0%}.",
        "",
        "  size     vignette",
        "  ----     --------",
    ]
    for size in size_order:
        names = sorted(by_size[size])
        if not names:
            continue
        for i, label in enumerate(names):
            prefix = size if i == 0 else ""
            lines.append(f"  {prefix:<7}  {label}")

    lines.extend(["", "  vignette                              size", "  " + "-" * 44])
    for name, size, well_posed in sorted(entries, key=lambda e: (e[1], e[0])):
        display = name if well_posed else f"{name} (overlap)"
        if well_posed and any(n == name and not wp for n, _, wp in entries):
            display = f"{name} (well-posed)"
        lines.append(f"  {display:<38} {size}")

    lines.append("")
    return lines


def default_output_path(which: str) -> Path:
    if which in ("well-posed", "two-cause"):
        return ROOT / "docs" / "vignette-table-well-posed.txt"
    if which == "overlap":
        return ROOT / "docs" / "vignette-table-overlap.txt"
    return ROOT / "docs" / "vignette-table.txt"


def build_output(which: str) -> list[str]:
    lines: list[str] = []
    if which == "all":
        lines.extend(format_intersection_summary())
    if which in ("all", "well-posed", "two-cause"):
        if lines:
            lines.append("")
        lines.extend(
            format_table(
                ROOT / "docs" / "base-rate-two-cause-vignettes.csv",
                well_posed=True,
                title="WELL-POSED TWO-CAUSE VIGNETTES",
            )
        )
    if which in ("all", "overlap"):
        if lines:
            lines.append("")
        lines.extend(
            format_table(
                ROOT / "docs" / "base-rate-overlap-vignettes.csv",
                well_posed=False,
                title="OVERLAP VIGNETTES (C and D may co-occur under A)",
            )
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-print base-rate vignette tables to a text file."
    )
    parser.add_argument(
        "which",
        nargs="?",
        default="all",
        choices=("all", "well-posed", "two-cause", "overlap"),
        help="which table to print (default: all)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (default: docs/vignette-table*.txt)",
    )
    args = parser.parse_args()

    lines = build_output(args.which)
    out_path = args.output or default_output_path(args.which)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path, file=sys.stdout)


if __name__ == "__main__":
    main()
