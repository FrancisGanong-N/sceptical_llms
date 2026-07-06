#!/usr/bin/env python3
"""Ensure implausible CSVs list every simple vignette (preserving existing values)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_simple_rate_prompts import (  # noqa: E402
    IMPLAUSIBLE_P_C_D_CSV,
    IMPLAUSIBLE_P_T_GIVEN_CSV,
    load_simple_vignettes,
)

# Altered P(T|C), P(T|D) for vignettes added after the original 15-row CSVs.
_NEW_P_T_GIVEN: dict[str, tuple[float, float]] = {
    "book drama streaming": (0.10, 0.50),
    "dog cat household": (0.50, 0.05),
    "youtube facebook news": (0.10, 0.50),
    "college grad professional job": (0.10, 0.50),
    "homeowner suburban mortgage": (0.10, 0.50),
    "parent married dual income": (0.50, 0.05),
    "republican gun owner ban": (0.10, 0.50),
}

_NEW_P_C_D: dict[str, tuple[float, float]] = {
    "book drama streaming": (0.90, 0.15),
    "dog cat household": (0.20, 0.90),
    "youtube facebook news": (0.90, 0.15),
    "college grad professional job": (0.90, 0.15),
    "homeowner suburban mortgage": (0.90, 0.15),
    "parent married dual income": (0.20, 0.90),
    "republican gun owner ban": (0.90, 0.15),
}


def _read_p_c_d() -> dict[str, tuple[float, float]]:
    if not IMPLAUSIBLE_P_C_D_CSV.is_file():
        return {}
    out: dict[str, tuple[float, float]] = {}
    with IMPLAUSIBLE_P_C_D_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["vignette_name"].strip()
            out[name] = (float(row["C"]), float(row["D"]))
    return out


def _read_p_t_given() -> dict[str, tuple[float, float]]:
    if not IMPLAUSIBLE_P_T_GIVEN_CSV.is_file():
        return {}
    out: dict[str, tuple[float, float]] = {}
    with IMPLAUSIBLE_P_T_GIVEN_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["vignette_name"].strip()
            out[name] = (float(row["P(T|C)"]), float(row["P(T|D)"]))
    return out


def _write_p_c_d(rows: dict[str, tuple[float, float]], names: list[str]) -> None:
    IMPLAUSIBLE_P_C_D_CSV.parent.mkdir(parents=True, exist_ok=True)
    with IMPLAUSIBLE_P_C_D_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["vignette_name", "C", "D"])
        writer.writeheader()
        for name in names:
            p_c, p_d = rows[name]
            writer.writerow({"vignette_name": name, "C": p_c, "D": p_d})


def _write_p_t_given(rows: dict[str, tuple[float, float]], names: list[str]) -> None:
    IMPLAUSIBLE_P_T_GIVEN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with IMPLAUSIBLE_P_T_GIVEN_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["vignette_name", "P(T|C)", "P(T|D)"])
        writer.writeheader()
        for name in names:
            s_c, s_d = rows[name]
            writer.writerow({"vignette_name": name, "P(T|C)": s_c, "P(T|D)": s_d})


def sync_implausible_csvs() -> tuple[list[str], list[str]]:
    names = [v.name for v in load_simple_vignettes()]
    p_c_d = _read_p_c_d()
    p_t = _read_p_t_given()

    added_c_d: list[str] = []
    added_t: list[str] = []

    for name in names:
        if name not in p_c_d:
            if name not in _NEW_P_C_D:
                raise ValueError(f"No altered P(C)/P(D) for vignette {name!r}")
            p_c_d[name] = _NEW_P_C_D[name]
            added_c_d.append(name)
        if name not in p_t:
            if name not in _NEW_P_T_GIVEN:
                raise ValueError(f"No altered P(T|·) for vignette {name!r}")
            p_t[name] = _NEW_P_T_GIVEN[name]
            added_t.append(name)

    extra_c_d = sorted(set(p_c_d) - set(names))
    extra_t = sorted(set(p_t) - set(names))
    if extra_c_d or extra_t:
        raise ValueError(
            f"Unknown vignette_name(s) in implausible CSVs: "
            f"p_c_d={extra_c_d}, p_t={extra_t}"
        )

    _write_p_c_d(p_c_d, names)
    _write_p_t_given(p_t, names)
    return added_c_d, added_t


def main() -> int:
    added_c_d, added_t = sync_implausible_csvs()
    names = [v.name for v in load_simple_vignettes()]
    print(f"Wrote {len(names)} rows to {IMPLAUSIBLE_P_C_D_CSV.relative_to(ROOT)}")
    print(f"Wrote {len(names)} rows to {IMPLAUSIBLE_P_T_GIVEN_CSV.relative_to(ROOT)}")
    if added_c_d:
        print(f"  added P(C)/P(D): {', '.join(added_c_d)}")
    if added_t:
        print(f"  added P(T|·): {', '.join(added_t)}")
    if not added_c_d and not added_t:
        print("  (all vignettes already present; order normalized)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
