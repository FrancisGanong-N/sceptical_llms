#!/usr/bin/env python3
"""Build the simple two-path base-rate causality diagram (C, D -> T)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures"
OUT_PNG = OUT_DIR / "base-rate-simple.png"
OUT_SVG = OUT_DIR / "base-rate-simple.svg"

ARROW_KW = dict(
    arrowstyle="-|>",
    mutation_scale=14,
    linewidth=1.5,
    color="#333333",
    shrinkA=8,
    shrinkB=8,
)
LABEL_KW = dict(
    fontsize=10,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.95),
)

POSTERIOR_EQ = (
    r"$P(C \mid T)"
    r"=\frac{P(T \mid C)\,P(C)}{P(T \mid C)\,P(C)+P(T \mid D)\,P(D)}$"
)


def draw_node(
    ax,
    x,
    y,
    title,
    subtitle=None,
    *,
    width=1.55,
    height=0.85,
    facecolor="#e8f4fc",
):
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
        ax.text(x, y - 0.18, subtitle, ha="center", va="center", fontsize=9, color="#444444")
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=13, fontweight="bold")
    return x, y


def draw_straight_arrow(ax, x1, y1, x2, y2, label: str) -> None:
    p0, p1 = (x1, y1), (x2, y2)
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=0", **ARROW_KW))
    lx, ly = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ax.text(lx, ly, label, **LABEL_KW)


def build_figure():
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.8, 8.2)
    ax.axis("off")

    causes_y, test_y = 6.2, 3.4
    node_height = 0.85
    label_gap = 0.15
    arrow_pad = node_height / 2 + 0.05
    eq_y = 1.35
    sep_y = 2.15

    ax.text(
        5,
        7.55,
        "Simple base-rate model",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        5,
        7.18,
        "Two mutually exclusive cases C and D converge on test T",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    for layer_y, name in ((causes_y, "Cases (mutually exclusive)"), (test_y, "Test")):
        ax.text(
            0.35,
            layer_y + node_height / 2 + label_gap,
            name,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#666666",
        )

    c_x, d_x, t_x = 3.25, 6.75, 5.0

    c = draw_node(ax, c_x, causes_y, "C", "prior P(C)")
    d = draw_node(ax, d_x, causes_y, "D", "prior P(D)")
    t = draw_node(ax, t_x, test_y, "T", "positive test", facecolor="#fff3e0")

    draw_straight_arrow(ax, c[0], c[1] - arrow_pad, t[0], t[1] + arrow_pad, r"$P(T \mid C)$")
    draw_straight_arrow(ax, d[0], d[1] - arrow_pad, t[0], t[1] + arrow_pad, r"$P(T \mid D)$")

    ax.plot([0.5, 9.5], [sep_y, sep_y], color="#cccccc", linewidth=1.0, zorder=0)
    ax.text(5, eq_y, POSTERIOR_EQ, ha="center", va="center", fontsize=14)
    ax.text(
        5,
        eq_y - 0.55,
        "Question: what is P(C | T)?",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )

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
