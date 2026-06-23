#!/usr/bin/env python3
"""Build the multi-cause base-rate causality diagram."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures"
OUT_PNG = OUT_DIR / "base-rate-multi-cause.png"
OUT_SVG = OUT_DIR / "base-rate-multi-cause.svg"

ARROW_KW = dict(
    arrowstyle="-|>",
    mutation_scale=14,
    linewidth=1.5,
    color="#333333",
    shrinkA=8,
    shrinkB=8,
)
LABEL_KW = dict(
    fontsize=9,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.95),
)

POSTERIOR_EQ = (
    r"$P(A \mid T)"
    r"=\frac{[P(C \mid A)\,P(T \mid C)+P(D \mid A)\,P(T \mid D)]\,P(A)}"
    r"{[P(C \mid A)\,P(T \mid C)+P(D \mid A)\,P(T \mid D)]\,P(A)"
    r"+P(E \mid B)\,P(T \mid E)\,P(B)+P(T \mid N)\,P(N)}$"
)


def draw_node(ax, x, y, title, subtitle=None, *, width=1.6, height=0.85, facecolor="#e8f4fc"):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor="#2c5f8a",
        linewidth=1.6,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x, y + 0.12, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x, y - 0.18, subtitle, ha="center", va="center", fontsize=8.5, color="#444444")
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=13, fontweight="bold")
    return x, y


def _bezier_point(p0, p1, p2, t: float) -> tuple[float, float]:
    u = 1.0 - t
    return (
        u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
    )


def _arc_control(x1, y1, x2, y2, rad: float) -> tuple[float, float]:
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return mx, my
    perp_x, perp_y = -dy / length, dx / length
    return mx + perp_x * rad * length, my + perp_y * rad * length


def _label_on_curve(ax, p0, p1, p2, label: str, *, t: float = 0.5) -> None:
    lx, ly = _bezier_point(p0, p1, p2, t)
    ax.text(lx, ly, label, **LABEL_KW)


def _label_on_line(ax, p0, p1, label: str) -> None:
    lx, ly = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ax.text(lx, ly, label, **LABEL_KW)


def draw_straight_arrow(ax, x1, y1, x2, y2, label: str) -> None:
    p0, p1 = (x1, y1), (x2, y2)
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=0", **ARROW_KW))
    _label_on_line(ax, p0, p1, label)


def draw_arc_arrow(ax, x1, y1, x2, y2, label: str, *, rad: float) -> None:
    p0, p2 = (x1, y1), (x2, y2)
    p1 = _arc_control(x1, y1, x2, y2, rad)
    path = MplPath([p0, p1, p2], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(FancyArrowPatch(path=path, **ARROW_KW))
    _label_on_curve(ax, p0, p1, p2, label)


def draw_downward_arc_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    label: str,
    *,
    bulge: float,
    shift_x: float = 0.0,
    label_t: float = 0.5,
) -> None:
    p0, p2 = (x1, y1), (x2, y2)
    p1 = ((x1 + x2) / 2 + shift_x, (y1 + y2) / 2 - bulge)
    path = MplPath([p0, p1, p2], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(FancyArrowPatch(path=path, **ARROW_KW))
    _label_on_curve(ax, p0, p1, p2, label, t=label_t)


def build_figure():
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.4, 9.2)
    ax.axis("off")

    causes_y, conditions_y, test_y = 7.85, 5.5, 3.15
    node_height = 0.85
    label_gap = 0.15
    arrow_pad = node_height / 2 + 0.05
    eq_y = 1.35
    sep_y = 2.3

    ax.text(
        5,
        8.95,
        "Multi-cause base-rate model",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        5,
        8.62,
        "Arrows flow downward: causes -> conditions -> test",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    for layer_y, name in (
        (causes_y, "Causes (mutually exclusive)"),
        (conditions_y, "Conditions"),
        (test_y, "Test"),
    ):
        ax.text(
            0.25,
            layer_y + node_height / 2 + label_gap,
            name,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#666666",
        )

    c_x, d_x = 2.4, 3.4
    a_x = (c_x + d_x) / 2
    b_x = 5.45
    n_x = 8.0
    t_x = 5.0

    a = draw_node(ax, a_x, causes_y, "A", "prior P(A)")
    b = draw_node(ax, b_x, causes_y, "B", "prior P(B)")
    n = draw_node(ax, n_x, causes_y, "N", "prior 1 - P(A) - P(B)", facecolor="#f4f4f4")

    c = draw_node(ax, c_x, conditions_y, "C", width=1.0)
    d = draw_node(ax, d_x, conditions_y, "D", width=1.0)
    e = draw_node(ax, b_x, conditions_y, "E")

    t = draw_node(ax, t_x, test_y, "T", "positive test", facecolor="#fff3e0")

    draw_straight_arrow(ax, a[0], a[1] - arrow_pad, c[0], c[1] + arrow_pad, "P(C|A)")
    draw_straight_arrow(ax, a[0], a[1] - arrow_pad, d[0], d[1] + arrow_pad, "P(D|A)")
    draw_straight_arrow(ax, b[0], b[1] - arrow_pad, e[0], e[1] + arrow_pad, "P(E|B)")

    draw_downward_arc_arrow(
        ax,
        n[0],
        n[1] - arrow_pad,
        t[0],
        t[1] + arrow_pad,
        "P(T|N)",
        bulge=0.18,
        shift_x=1.4,
        label_t=0.38,
    )

    draw_downward_arc_arrow(ax, c[0], c[1] - arrow_pad, t[0], t[1] + arrow_pad, "P(T|C)", bulge=0.38)
    draw_downward_arc_arrow(ax, d[0], d[1] - arrow_pad, t[0], t[1] + arrow_pad, "P(T|D)", bulge=0.30)
    draw_downward_arc_arrow(ax, e[0], e[1] - arrow_pad, t[0], t[1] + arrow_pad, "P(T|E)", bulge=0.22)

    ax.plot([0.4, 9.6], [sep_y, sep_y], color="#cccccc", linewidth=1.0, zorder=0)
    ax.text(5, eq_y, POSTERIOR_EQ, ha="center", va="center", fontsize=13)

    fig.tight_layout()
    return fig


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
