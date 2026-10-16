"""
Fig 1 — DRIADA Architecture Scheme
Generates the architecture triangle diagram for the eLife paper.

Usage:
    python fig1_architecture.py          # saves fig1_architecture.pdf + .png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elife_paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Layout ──────────────────────────────────────────────────────────
# Wider figure, give more horizontal room so Populations isn't clipped.
# Triangle vertices spread wider; center hub lower to give vertical room.
POS = {
    "Neurons":     (0.50, 0.90),
    "Networks":    (0.10, 0.13),
    "Populations": (0.90, 0.13),
    "Center":      (0.50, 0.44),
}

# ── Colors ──────────────────────────────────────────────────────────
C_PURPLE = "#6A1B9A"
C_GREEN  = "#558B2F"
C_ORANGE = "#E65100"
C_RED    = "#C62828"

# ── Figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.set_xlim(-0.12, 1.12)
ax.set_ylim(-0.10, 1.05)
ax.set_aspect("equal")
ax.axis("off")


# ── Helper: draw a rounded-rect node ────────────────────────────────
def draw_node(xy, label, subtitle=None, color=C_PURPLE,
              width=0.20, height=0.10, fontsize=14, dashed=False):
    x, y = xy
    box = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.02",
        facecolor="white", edgecolor=color,
        linewidth=2.5,
        linestyle="--" if dashed else "-",
        zorder=3,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x, y + 0.018, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color, zorder=4)
        ax.text(x, y - 0.022, subtitle, ha="center", va="center",
                fontsize=8.5, color=color, zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color, zorder=4)


# ── Helper: draw a curved arrow with label ──────────────────────────
def draw_arrow(xy_from, xy_to, label, color=C_GREEN, rad=0.15,
               fontsize=9, label_pos=(None, None), ha="center", va="center",
               bold=False, shrink_a=20, shrink_b=20):
    arrow = mpatches.FancyArrowPatch(
        xy_from, xy_to,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="->,head_width=6,head_length=6",
        color=color, linewidth=1.8, zorder=2,
        shrinkA=shrink_a, shrinkB=shrink_b,
    )
    ax.add_patch(arrow)

    if not label:
        return

    # Use explicit label position if given, else midpoint
    lx, ly = label_pos
    if lx is None:
        lx = (xy_from[0] + xy_to[0]) / 2
    if ly is None:
        ly = (xy_from[1] + xy_to[1]) / 2

    ax.text(lx, ly, label, ha=ha, va=va,
            fontsize=fontsize, color=color, zorder=5,
            fontweight="bold" if bold else "normal",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.9, pad=1.5))


# ── Draw nodes ──────────────────────────────────────────────────────
draw_node(POS["Neurons"], "Neurons", color=C_PURPLE,
          width=0.22, height=0.10, fontsize=16)
draw_node(POS["Networks"], "Networks", color=C_GREEN,
          width=0.22, height=0.10, fontsize=16)
draw_node(POS["Populations"], "Populations", color=C_ORANGE,
          width=0.26, height=0.10, fontsize=16)
draw_node(POS["Center"], "Behavior", subtitle="TimeSeries / MultiTimeSeries",
          color=C_RED, width=0.30, height=0.10, fontsize=12, dashed=True)

# ══════════════════════════════════════════════════════════════════════
# GREEN ARROWS — vertex-to-vertex, curve OUTWARD to avoid center hub
# ══════════════════════════════════════════════════════════════════════

# ── Right edge: Neurons <-> Populations ──
# Outer arrow (curves right): Neurons -> Populations  "neurons to pop code"
draw_arrow(POS["Neurons"], POS["Populations"],
           "neurons to\npop code", color=C_GREEN, rad=-0.30,
           label_pos=(0.86, 0.58), fontsize=8.5, ha="center")
# Inner arrow (curves right): Populations -> Neurons  "dim. reduction"
draw_arrow(POS["Populations"], POS["Neurons"],
           "dim.\nreduction", color=C_GREEN, rad=-0.30,
           label_pos=(0.72, 0.58), fontsize=8.5, ha="center")

# ── Left edge: Neurons -> Networks (two parallel, both curve left) ──
draw_arrow(POS["Neurons"], POS["Networks"],
           "func.\nnetworks", color=C_GREEN, rad=0.30,
           label_pos=(0.14, 0.58), fontsize=8.5, ha="center")
draw_arrow(POS["Neurons"], POS["Networks"],
           "rec.\ngraphs", color=C_GREEN, rad=0.45,
           label_pos=(-0.02, 0.48), fontsize=8.5, ha="center")

# ── Bottom edge: Populations -> Networks (two, curve downward) ──
draw_arrow(POS["Populations"], POS["Networks"],
           "RSA", color=C_GREEN, rad=-0.25,
           label_pos=(0.50, 0.02), fontsize=8.5)
draw_arrow(POS["Populations"], POS["Networks"],
           "graph analysis", color=C_GREEN, rad=-0.45,
           label_pos=(0.50, -0.06), fontsize=8.5)

# ══════════════════════════════════════════════════════════════════════
# ORANGE ARROWS — center hub to/from vertices
# ══════════════════════════════════════════════════════════════════════

# Center <-> Neurons: INTENSE (two arrows forming bidirectional pair)
draw_arrow(POS["Center"], POS["Neurons"],
           "INTENSE", color=C_ORANGE, rad=-0.18,
           label_pos=(0.36, 0.68), fontsize=11, bold=True)
draw_arrow(POS["Neurons"], POS["Center"],
           "", color=C_ORANGE, rad=-0.18)

# Center -> Networks: func. clusters
draw_arrow(POS["Center"], POS["Networks"],
           "func.\nclusters", color=C_ORANGE, rad=0.0,
           label_pos=(0.24, 0.33), fontsize=8.5, ha="right")

# Center -> Populations: pop. coding
draw_arrow(POS["Center"], POS["Populations"],
           "pop.\ncoding", color=C_ORANGE, rad=0.0,
           label_pos=(0.76, 0.33), fontsize=8.5, ha="left")

# ── Save ────────────────────────────────────────────────────────────
fig.savefig(elife_paths.out_path("fig1_architecture.pdf"),
            bbox_inches="tight", dpi=300)
fig.savefig(elife_paths.out_path("fig1_architecture.png"),
            bbox_inches="tight", dpi=300)
print("Saved fig1_architecture.pdf and .png")
