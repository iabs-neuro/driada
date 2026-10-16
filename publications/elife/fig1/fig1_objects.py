"""
Fig 1B — DRIADA Object Relationship Graph
Right panel of the architecture figure for the eLife paper.

4-row compact waterfall layout:
  Row 1: MVData, Experiment
  Row 2: MultiTimeSeries, Embedding, Neuron, TimeSeries
  Row 3: ProximityGraph, RecurrenceGraph, VisibilityGraph, OPN
  Row 4: Network

Usage:
    python fig1_objects.py          # saves fig1_objects.pdf + .png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elife_paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colors by domain ────────────────────────────────────────────────
C_EXP  = "#C62828"   # experiment domain (red)
C_TS   = "#6A1B9A"   # time series domain (purple)
C_DR   = "#E65100"   # dim reduction domain (orange)
C_NET  = "#558B2F"   # network domain (green)
C_INH  = "#37474F"   # inheritance arrows (dark blue-grey)

# ── Row Y-coordinates ───────────────────────────────────────────────
Y1 = 0.88   # MVData, Experiment
Y2 = 0.62   # MultiTimeSeries, Neuron, TimeSeries
Y3 = 0.36   # Embedding, ProximityGraph, RecurrenceGraph, VG, OPN
Y4 = 0.10   # Network (base class)

POS = {
    "MVData":                  (0.15, Y1),
    "Experiment":              (0.72, Y1),
    "Embedding":               (0.15, Y2),
    "MultiTimeSeries":         (0.38, Y2),
    "Neuron":                  (0.58, Y2),
    "TimeSeries":              (0.82, Y2),
    "ProximityGraph":          (0.13, Y3),
    "RecurrenceGraph":         (0.38, Y3),
    "VisibilityGraph":         (0.63, Y3),
    "OrdinalPartition\nNetwork": (0.88, Y3),
    "Network":                 (0.50, Y4),
}

# ── Dash pattern — fine dots ───────────────────────────────────────
DASH = (0, (0.5, 1.5))

# ── Figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(-0.12, 1.15)
ax.set_ylim(-0.02, 1.00)
ax.set_aspect("equal")
ax.axis("off")


def draw_node(name, color, width=0.18, height=0.07, fontsize=10):
    x, y = POS[name]
    box = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.015",
        facecolor="white", edgecolor=color, linewidth=2.2, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, name, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=4,
            linespacing=0.85)


def draw_arrow(name_from, name_to, label="", color=C_INH, rad=0.0,
               dashed=False, fontsize=7.5, label_pos=None, ha="center",
               shrink_a=20, shrink_b=20, lw=2.2):
    xy_from = POS[name_from]
    xy_to = POS[name_to]
    arrow = mpatches.FancyArrowPatch(
        xy_from, xy_to,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="->,head_width=7,head_length=7",
        color=color, linewidth=lw, zorder=2.5,
        linestyle=DASH if dashed else "-",
        shrinkA=shrink_a, shrinkB=shrink_b,
    )
    ax.add_patch(arrow)
    if not label:
        return
    if label_pos is None:
        lx = (xy_from[0] + xy_to[0]) / 2
        ly = (xy_from[1] + xy_to[1]) / 2
    else:
        lx, ly = label_pos
    ax.text(lx, ly, label, ha=ha, va="center",
            fontsize=fontsize, color=color, zorder=6, style="italic",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))


# ══════════════════════════════════════════════════════════════════════
# DRAW NODES
# ══════════════════════════════════════════════════════════════════════
draw_node("MVData",            C_DR,  width=0.15, fontsize=11)
draw_node("Experiment",        C_EXP, width=0.20, fontsize=12)
draw_node("MultiTimeSeries",   C_TS,  width=0.24, fontsize=10)
draw_node("Neuron",            C_EXP, width=0.14, fontsize=10)
draw_node("TimeSeries",        C_TS,  width=0.18, fontsize=10)
draw_node("Embedding",         C_DR,  width=0.17, fontsize=10)
draw_node("ProximityGraph",    C_NET, width=0.19, fontsize=9)
draw_node("RecurrenceGraph",   C_NET, width=0.19, fontsize=9)
draw_node("Network",           C_NET, width=0.15, fontsize=10)
draw_node("VisibilityGraph",   C_NET, width=0.19, fontsize=9)
draw_node("OrdinalPartition\nNetwork", C_NET, width=0.19, fontsize=8.5)

# ══════════════════════════════════════════════════════════════════════
# INHERITANCE (solid, dark)
# ══════════════════════════════════════════════════════════════════════

# MVData → MultiTimeSeries
draw_arrow("MVData", "MultiTimeSeries", "inherits", color=C_INH,
           rad=-0.15, label_pos=(0.22, 0.76), fontsize=7, ha="left")

# ProximityGraph → Network  (row 3 → row 4)
draw_arrow("Network", "ProximityGraph", "inherits", color=C_INH,
           rad=0.20, label_pos=(0.24, 0.22), fontsize=7, ha="left")

# RecurrenceGraph → ProximityGraph  (same row, arc above)
draw_arrow("ProximityGraph", "RecurrenceGraph", "inherits", color=C_INH,
           rad=-0.35, label_pos=(0.25, 0.44), fontsize=7, ha="center")

# VisibilityGraph → Network  (row 3 → row 4)
draw_arrow("Network", "VisibilityGraph", "inherits", color=C_INH,
           rad=-0.15, label_pos=(0.63, 0.22), fontsize=7, ha="left")

# OrdinalPartitionNetwork → Network  (row 3 → row 4)
draw_arrow("Network", "OrdinalPartition\nNetwork", "inherits", color=C_INH,
           rad=-0.30, label_pos=(0.78, 0.18), fontsize=7, ha="left")

# ══════════════════════════════════════════════════════════════════════
# COMPOSITION / PRODUCES (dashed)
# ══════════════════════════════════════════════════════════════════════

# --- Experiment domain ---

# Experiment → MultiTimeSeries
draw_arrow("Experiment", "MultiTimeSeries",
           ".calcium  .spikes", color=C_EXP, dashed=True,
           rad=0.15, label_pos=(0.50, 0.80), fontsize=7, ha="center")

# Experiment → Neuron
draw_arrow("Experiment", "Neuron", ".neurons", color=C_EXP, dashed=True,
           rad=0.0, label_pos=(0.70, 0.78), fontsize=7, ha="left")

# Experiment → TimeSeries
draw_arrow("Experiment", "TimeSeries",
           ".dynamic_features", color=C_EXP, dashed=True,
           rad=-0.10, label_pos=(0.84, 0.78), fontsize=7, ha="left")

# Neuron → TimeSeries
draw_arrow("Neuron", "TimeSeries", ".ca_ts  .sp_ts", color=C_EXP, dashed=True,
           rad=0.0, label_pos=(0.73, 0.56), fontsize=7, ha="center")

# --- MTS / Embedding produces (orange dashed) ---

# MultiTimeSeries → Embedding  (same row, MTS is right of Embedding)
draw_arrow("MultiTimeSeries", "Embedding",
           ".get_embedding()", color=C_DR, dashed=True,
           rad=0.30, label_pos=(0.26, 0.68), fontsize=6.5, ha="center")

# MultiTimeSeries → ProximityGraph  (row 2 → row 3)
draw_arrow("MultiTimeSeries", "ProximityGraph",
           ".get_proximity_graph()", color=C_DR, dashed=True,
           rad=0.15, label_pos=(0.18, 0.50), fontsize=6.5, ha="left")

# Embedding → ProximityGraph  (row 2 → row 3)
draw_arrow("Embedding", "ProximityGraph", ".graph", color=C_DR, dashed=True,
           rad=0.0, label_pos=(0.06, 0.50), fontsize=7, ha="left")

# --- TimeSeries produces (purple dashed) ---

# TimeSeries → RecurrenceGraph
draw_arrow("TimeSeries", "RecurrenceGraph",
           ".recurrence_graph()", color=C_TS, dashed=True,
           rad=0.15, label_pos=(0.52, 0.50), fontsize=6.5, ha="left")

# TimeSeries → VisibilityGraph
draw_arrow("TimeSeries", "VisibilityGraph",
           ".visibility_graph()", color=C_TS, dashed=True,
           rad=0.0, label_pos=(0.78, 0.50), fontsize=6.5, ha="left")

# TimeSeries → OrdinalPartitionNetwork
draw_arrow("TimeSeries", "OrdinalPartition\nNetwork",
           ".ordinal_partition_network()", color=C_TS, dashed=True,
           rad=-0.15, label_pos=(0.94, 0.50), fontsize=6.5, ha="left")

# ══════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════
ly = 0.03
lx = 0.02

ax.annotate("", xy=(lx + 0.08, ly), xytext=(lx, ly),
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.3",
                            color=C_INH, lw=2.2))
ax.text(lx + 0.09, ly, "inheritance", fontsize=9, va="center", color=C_INH)

ax.annotate("", xy=(lx + 0.34, ly), xytext=(lx + 0.26, ly),
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.3",
                            color="#616161", lw=2.2, linestyle=DASH))
ax.text(lx + 0.35, ly, "produces / contains", fontsize=9, va="center",
        color="#616161")

# ── Save ────────────────────────────────────────────────────────────
fig.savefig(elife_paths.out_path("fig1_objects.pdf"),
            bbox_inches="tight", dpi=300)
fig.savefig(elife_paths.out_path("fig1_objects.png"),
            bbox_inches="tight", dpi=300)
print("Saved fig1_objects.pdf and .png")
