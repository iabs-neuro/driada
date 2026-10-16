"""
Fig 3 — Composite figure for eLife paper.

Symmetric GridSpec with computed square cells, B bottom 70:30 via SubplotSpec.
The functional network panels belong to Fig 4.

Layout (7 columns, symmetric halves):
  col 0: A_left (1)   col 1: gap (g)   col 2: A_right (1)
  col 3: MAIN GAP (G)
  col 4: B_left (1)   col 5: gap (g)   col 6: B_right (1)

Layout (5 rows):
  row 0:  Top panels (traj, occ, place, HD) — SQUARE cells
  row 1:  AB internal gap
  row 2:  Bottom panels (umap_r, umap_s, loco+density) — SQUARE cells
  row 3:  BIG gap (AB to C)
  row 4:  C violin (full width)

Run:  python publications/elife/fig3/fig3_composite.py
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from collections import defaultdict

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

from driada.utils.publication.style import StylePreset

# ---------------------------------------------------------------------------
# panel imports
# ---------------------------------------------------------------------------
from panel_a import plot_on_axes as plot_panel_A
from panel_b import plot_on_axes as plot_panel_B
from panel_e import create_panel_E, create_nsel_bars

# ---------------------------------------------------------------------------
# shared style
# ---------------------------------------------------------------------------
style = StylePreset.publication_default().copy(
    base_spine_width=1.2,
    base_tick_width=1.2,
    base_tick_length=3,
    base_tick_pad=1.5,
    base_tick_labelsize=7,
    base_label_size=8,
    base_title_size=9,
    base_legend_fontsize=6,
    base_annotation_fontsize=7,
    tight_layout=False,
)

# ---------------------------------------------------------------------------
# figure dimensions
# ---------------------------------------------------------------------------
W_CM, H_CM = 18.0, 18.0
DPI = 600
LEFT, RIGHT, TOP, BOTTOM = 0.10, 0.96, 0.96, 0.10

# ---------------------------------------------------------------------------
# GridSpec: symmetric 7-column, square-cell rows
# ---------------------------------------------------------------------------
print("=" * 60)
print("Fig 3 Composite -- Symmetric GridSpec, Square Cells")
print("=" * 60)
t0 = time.perf_counter()

# Column ratios (symmetric halves)
g = 0.30   # inner gap within each half
G = 1.00   # main gap (white block) between A and B
col_ratios = [1, g, 1, G, 1, g, 1]
sum_col = sum(col_ratios)

# Physical content area
W_content = W_CM * (RIGHT - LEFT)
H_content = H_CM * (TOP - BOTTOM)

# Physical width of one content column (ratio=1)
col_w_cm = W_content / sum_col

# Row ratios: compute R so content cells are physically square
# R / (2R + non_R_sum) * H_content = col_w_cm
other_rows = {
    'gap_AB': 1.4,
    'gap_AB_C': 1.25,
    'violin': 2.5,          # C: selectivity violin
}
non_R_sum = sum(other_rows.values())
R = col_w_cm * non_R_sum / (H_content - 2 * col_w_cm)

row_ratios = [
    R,                          # row 0: top panels (SQUARE)
    other_rows['gap_AB'],       # row 1: gap
    R,                          # row 2: bottom panels (SQUARE)
    other_rows['gap_AB_C'],     # row 3: gap
    other_rows['violin'],       # row 4: C violin (selectivity prevalence)
]
sum_row = sum(row_ratios)

# Print computed sizes
sq_h = R / sum_row * H_content
print(f"Square cell: {col_w_cm:.2f} x {sq_h:.2f} cm (aspect {sq_h/col_w_cm:.4f})")

fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54), dpi=DPI)

gs = GridSpec(5, 7, figure=fig,
    height_ratios=row_ratios,
    width_ratios=col_ratios,
    left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
    wspace=0, hspace=0,
)

# ---------------------------------------------------------------------------
# compute panel sizes for StylePreset scaling
# ---------------------------------------------------------------------------
def col_w(ratio):
    return W_content * (ratio / sum_col)

def row_h(ratio):
    return H_content * (ratio / sum_row)

sq_size = (col_w_cm, sq_h)  # square panel size

# C panel sizes (violin + n_sel bars, split 75:25)
c_h = row_h(other_rows['violin'])
c_violin_w = W_content * 0.75
c_bars_w = W_content * 0.25

# B bottom sizes (70:30 of half-width)
half_w = col_w(1 + g + 1)  # full half-width including inner gap
b_loco_w = half_w * 0.70
b_dens_w = half_w * 0.30
b_bot_h = sq_h

# ---------------------------------------------------------------------------
# Row 0: Top panels -- all in master grid, all SQUARE
# ---------------------------------------------------------------------------
print("\nCreating top row panels...")

ax_traj  = fig.add_subplot(gs[0, 0], label="A_traj")
ax_occ   = fig.add_subplot(gs[0, 2], label="A_occ")
ax_place = fig.add_subplot(gs[0, 4], label="B_place")
ax_hd    = fig.add_subplot(gs[0, 6], label="B_hd", projection="polar")

# ---------------------------------------------------------------------------
# Row 2: Bottom panels -- A panels in master grid, B via SubplotSpec (70:30)
# ---------------------------------------------------------------------------
print("Creating bottom row panels...")

ax_umap  = fig.add_subplot(gs[2, 0], label="A_umap")
ax_umaps = fig.add_subplot(gs[2, 2], label="A_umaps")

# B bottom: 70:30 split via SubplotSpec (row alignment from master grid)
inner_b_bot = GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[2, 4:7],
    width_ratios=[0.70, 0.30], wspace=0.35,
)
ax_disc    = fig.add_subplot(inner_b_bot[0, 0], label="B_disc")
ax_density = fig.add_subplot(inner_b_bot[0, 1], label="B_density")

# ---------------------------------------------------------------------------
# Plot Panel A content
# ---------------------------------------------------------------------------
print("\nPlotting Panel A...")
plot_panel_A(ax_traj, ax_occ, ax_umap, ax_umaps, fig, style, {
    "A_traj":  sq_size,
    "A_occ":   sq_size,
    "A_umap":  sq_size,
    "A_umaps": sq_size,
})

# ---------------------------------------------------------------------------
# Plot Panel B content
# ---------------------------------------------------------------------------
print("\nPlotting Panel B...")
plot_panel_B(ax_place, ax_hd, ax_disc, ax_density, fig, style, {
    "B_place":   sq_size,
    "B_disc":    (b_loco_w, b_bot_h),
    "B_density": (b_dens_w, b_bot_h),
})

# ---------------------------------------------------------------------------
# Panel C: Selectivity violin + n_sel bar chart (split row)
# ---------------------------------------------------------------------------
print("\nCreating Panel C...")
inner_c = GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[4, 0:7],
    width_ratios=[0.75, 0.25], wspace=0.35,
)
ax_C = fig.add_subplot(inner_c[0, 0], label="C")
create_panel_E(ax_C, style, (c_violin_w, c_h))

ax_C_bars = fig.add_subplot(inner_c[0, 1], label="C_bars")
create_nsel_bars(ax_C_bars, style, (c_bars_w, c_h))

# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------
print("\nAdding panel labels...")


def add_panel_labels(fig, label_map, fontsize=12, x_offset=-0.03, y_offset=0.01):
    """Add panel labels aligned by grid row and column."""
    fig.canvas.draw()

    info = {}
    for label, ax_ref in label_map.items():
        if isinstance(ax_ref, str):
            ax = next(a for a in fig.axes if a.get_label() == ax_ref)
        else:
            ax = ax_ref
        pos = ax.get_position()
        info[label] = {
            "x": pos.x0 + x_offset,
            "y": pos.y0 + pos.height + y_offset,
            "row_top": round(pos.y0 + pos.height, 2),
            "col_left": round(pos.x0, 2),
        }

    # Enforce row alignment
    row_groups = defaultdict(list)
    for label, d in info.items():
        row_groups[d["row_top"]].append(label)
    for members in row_groups.values():
        if len(members) > 1:
            shared_y = max(info[l]["y"] for l in members)
            for l in members:
                info[l]["y"] = shared_y

    # Enforce column alignment
    col_groups = defaultdict(list)
    for label, d in info.items():
        col_groups[d["col_left"]].append(label)
    for members in col_groups.values():
        if len(members) > 1:
            shared_x = min(info[l]["x"] for l in members)
            for l in members:
                info[l]["x"] = shared_x

    for label, d in info.items():
        fig.text(d["x"], d["y"], label,
                 fontsize=fontsize, fontweight="bold",
                 va="bottom", ha="left", transform=fig.transFigure)


add_panel_labels(fig, {
    "A": "A_traj",
    "B": "B_place",
    "C": "C",
}, fontsize=12, x_offset=-0.02, y_offset=0.012)

# ---------------------------------------------------------------------------
# Programmatic verification
# ---------------------------------------------------------------------------
print("\nVerification:")


def verify_figure(fig, expected=None, tolerance=0.005):
    """Verify axis alignment, squareness, and overlap."""
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    issues = []

    axes_info = {}
    for ax in fig.axes:
        name = ax.get_label()
        if not name or name.startswith("_"):
            continue
        pos = ax.get_position()
        w_inch = pos.width * fig_w
        h_inch = pos.height * fig_h
        axes_info[name] = {
            "x0": pos.x0, "y0": pos.y0,
            "x1": pos.x0 + pos.width, "y1": pos.y0 + pos.height,
            "w_cm": w_inch * 2.54, "h_cm": h_inch * 2.54,
            "aspect": h_inch / w_inch if w_inch > 0 else 0,
        }

    for name, info in sorted(axes_info.items()):
        sq = " [SQUARE]" if abs(info["aspect"] - 1.0) < 0.02 else ""
        print(f"  {name:15s}: ({info['w_cm']:.1f} x {info['h_cm']:.1f} cm) "
              f"aspect={info['aspect']:.3f}{sq}")

    if expected is None:
        return axes_info

    # Check squareness
    for name in expected.get("square", []):
        if name in axes_info:
            a = axes_info[name]["aspect"]
            if abs(a - 1.0) > 0.02:
                issues.append(f"SQUARE FAIL: {name} aspect={a:.3f}")

    # Check row alignment
    for row in expected.get("rows", []):
        present = [n for n in row if n in axes_info]
        if len(present) < 2:
            continue
        bottoms = [axes_info[n]["y0"] for n in present]
        tops = [axes_info[n]["y1"] for n in present]
        spread_b = max(bottoms) - min(bottoms)
        spread_t = max(tops) - min(tops)
        if spread_b > tolerance:
            issues.append(f"ROW BASELINE MISALIGNED: {present} spread={spread_b:.4f}")
        if spread_t > tolerance:
            issues.append(f"ROW TOP MISALIGNED: {present} spread={spread_t:.4f}")

    # Check column alignment
    for col in expected.get("cols", []):
        present = [n for n in col if n in axes_info]
        if len(present) < 2:
            continue
        lefts = [axes_info[n]["x0"] for n in present]
        spread = max(lefts) - min(lefts)
        if spread > tolerance:
            issues.append(f"COL LEFT MISALIGNED: {present} spread={spread:.4f}")

    # Check overlap between non-internal axes
    names = [n for n in axes_info if "_" not in n]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = axes_info[names[i]], axes_info[names[j]]
            overlap_x = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            overlap_y = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if overlap_x > tolerance and overlap_y > tolerance:
                issues.append(f"OVERLAP: {names[i]} and {names[j]}")

    if issues:
        print("VERIFICATION FAILED:")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("VERIFICATION PASSED")

    return axes_info


verify_figure(fig, expected={
    "square": ["A_traj", "A_occ", "A_umap", "A_umaps", "B_place", "B_hd"],
    "rows": [
        ["A_traj", "A_occ", "B_place"],  # top row (skip B_hd - polar)
        ["A_umap", "A_umaps", "B_disc", "B_density"],  # bottom row
    ],
    "cols": [
        ["A_traj", "A_umap"],  # left-A column
        ["A_occ", "A_umaps"],  # right-A column
    ],
})

# A and C must share the same y-axis (left edge)
fig.canvas.draw()
_ax_a = next(a for a in fig.axes if a.get_label() == "A_traj")
_ax_c = next(a for a in fig.axes if a.get_label() == "C")
_dx = abs(_ax_a.get_position().x0 - _ax_c.get_position().x0)
print(f"\nA/C left-edge offset: {_dx:.5f} "
      f"({'ALIGNED' if _dx < 0.005 else 'MISALIGNED'})")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_png = elife_paths.out_path("fig3.png")
out_pdf = elife_paths.out_path("fig3.pdf")
fig.savefig(out_png, dpi=DPI, facecolor="white", edgecolor="none")
fig.savefig(out_pdf, dpi=DPI, facecolor="white", edgecolor="none")
plt.close(fig)

total_time = time.perf_counter() - t0
print(f"\nSaved {out_png}")
print(f"Saved {out_pdf}")
print(f"Total time: {total_time:.1f}s")
