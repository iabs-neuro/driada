"""Figure 2: the composer of the panels, with the output folder and name as arguments and the panel
letters in lower case when lowercase=True.
The colorbar label of panel B is set in fig2b_trajectory_delegated.CBAR_LABEL.

Figure 2 (v8, DRIADA 1.2.2 recompute): INTENSE vs classic place cell validation.

Copy of science/intense_figures/fig2/fig2_delegated.py; panels C, E, F come from
fig2_panels_v8.py (data: fig2_data_v8.py), A, B, D from the original caches.

This script uses the publication framework for layout/dimensions while
delegating actual plotting to individual panel scripts. Each panel script
uses the publication framework's StylePreset for consistent styling.

Layout (left-to-right, top-to-bottom):
    Row 1 (40%):  A wavelets          |  B trajectory (square)
    Row 2 (28%):  C circles (50%)     |  D activity maps (50%)
    Row 3 (32%):  E metrics (52%)     |  gap  |  F agreement, two axes (48%)
    Bottom (2%):  controllable blank space
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pathlib import Path

import _paths  # noqa: F401

from driada.utils.publication import StylePreset

# Import from _delegated panel scripts (single code path, StylePreset-based)
from fig2a_wavelets_delegated import create_fig2a
from fig2b_trajectory_delegated import create_fig2b
from fig2d_activity_maps_delegated import create_fig2d
from fig2_panels_v8 import create_fig2c, create_fig2e, create_fig2f

import _paths_fig as _F  # noqa: E402  (figure_chain/ is put on sys.path by _paths.py)
OUT_DIR = _F.out_dir(Path(__file__).resolve().parent.parent) / "fig2_out"


# ==========================================================================
# LAYOUT CONSTANTS (cm)
# ==========================================================================
CANVAS_WIDTH_CM = 18.0
CANVAS_HEIGHT_CM = 22.0
DPI = 600

# Gaps
H_GAP_CM = 0.3    # horizontal gap (rows 1-2)
V_GAP1_CM = 0.15  # vertical gap between rows 1–2 (tight)
V_GAP2_CM = 1.0   # vertical gap between rows 2–3 (more breathing room)
E_F_GAP_CM = 0.6  # explicit spacer between E and F

# Bottom blank space (controllable)
BOTTOM_MARGIN_FRAC = 0.05  # 5% of total height

# Available height for content rows
_BOTTOM_CM = CANVAS_HEIGHT_CM * BOTTOM_MARGIN_FRAC
_ROW_TOTAL = CANVAS_HEIGHT_CM - V_GAP1_CM - V_GAP2_CM - _BOTTOM_CM

# Row heights — ratios 44:24:32
ROW1_H = _ROW_TOTAL * 0.44
ROW2_H = _ROW_TOTAL * 0.24
ROW3_H = _ROW_TOTAL * 0.32

# Row 1: B is square (subfigure width = row height), A gets the rest
COL_B_W = ROW1_H
COL_A_W = CANVAS_WIDTH_CM - COL_B_W - H_GAP_CM

# Row 2: 50/50 split
_COL_HALF = (CANVAS_WIDTH_CM - H_GAP_CM) / 2

# Row 3: 65/35 split with explicit gap between E and F
_ROW3_AVAIL = CANVAS_WIDTH_CM - E_F_GAP_CM
# F holds two axes (conditional confirmation and top-N Jaccard)
COL_E_W = _ROW3_AVAIL * 0.52
COL_F_W = _ROW3_AVAIL * 0.48

# Panel sizes (width, height) in cm — passed to StylePreset for scaling
PANEL_SIZE_A = (COL_A_W, ROW1_H)
PANEL_SIZE_B = (COL_B_W, ROW1_H)
PANEL_SIZE_C = (_COL_HALF, ROW2_H)
PANEL_SIZE_D = (_COL_HALF, ROW2_H)
PANEL_SIZE_E = (COL_E_W, ROW3_H)
PANEL_SIZE_F = (COL_F_W, ROW3_H)


def create_fig2_delegated(output_dir=None, save_formats=('png',), output_name='fig2',
                          lowercase=True, panels='ABCDEF'):
    """Create Figure 2 by delegating to individual panel scripts.

    Each panel script uses the publication framework's StylePreset
    for auto-scaled styling based on panel size.
    """
    if output_dir is None:
        output_dir = OUT_DIR
    output_dir = Path(output_dir)

    # ==========================================================================
    # STYLE — shared across all panels
    # ==========================================================================
    style = StylePreset.publication_default().copy(
        base_tick_length=3,         # match fig1
        base_tick_pad=4,            # §7.1: reduce tick-label gap (default 8)
        base_legend_fontsize=7,     # match fig1 legend sizing
        base_annotation_fontsize=7,
        base_label_size=9,          # slightly smaller axis labels for ~8cm panels
    )

    # ==========================================================================
    # FIGURE AND SUBFIGURE LAYOUT
    # ==========================================================================
    fig_w = CANVAS_WIDTH_CM / 2.54
    fig_h = CANVAS_HEIGHT_CM / 2.54
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    # Height ratios: 3 content rows + 2 gaps + bottom margin
    row_fracs = [ROW1_H / CANVAS_HEIGHT_CM,
                 V_GAP1_CM / CANVAS_HEIGHT_CM,
                 ROW2_H / CANVAS_HEIGHT_CM,
                 V_GAP2_CM / CANVAS_HEIGHT_CM,
                 ROW3_H / CANVAS_HEIGHT_CM,
                 BOTTOM_MARGIN_FRAC]

    # 6 rows: content, gap, content, gap, content, bottom_margin
    row_subfigs = fig.subfigures(6, 1, height_ratios=row_fracs, hspace=0)
    row1_sf = row_subfigs[0]
    row2_sf = row_subfigs[2]
    row3_sf = row_subfigs[4]
    # rows 1, 3, 5 are spacers (empty)

    # --- Row 1: A (wider) | B (square) ---
    row1_cols = row1_sf.subfigures(1, 2, width_ratios=[COL_A_W, COL_B_W],
                                    wspace=H_GAP_CM / CANVAS_WIDTH_CM)
    sf_a = row1_cols[0]
    sf_b = row1_cols[1]

    # --- Row 2: C | D (50/50) ---
    row2_cols = row2_sf.subfigures(1, 2, width_ratios=[1, 1],
                                    wspace=H_GAP_CM / CANVAS_WIDTH_CM)
    sf_c = row2_cols[0]
    sf_d = row2_cols[1]

    # --- Row 3: E | spacer | F ---
    row3_cols = row3_sf.subfigures(1, 3,
                                    width_ratios=[COL_E_W, E_F_GAP_CM, COL_F_W],
                                    wspace=0)
    sf_e = row3_cols[0]
    # row3_cols[1] is blank spacer
    sf_f = row3_cols[2]

    # ==========================================================================
    # DELEGATE TO PANEL SCRIPTS
    # ==========================================================================
    print("Creating Figure 2...")

    print("  Panel A: Wavelets...")
    if 'A' in panels:
        create_fig2a(fig=sf_a, style=style, panel_size=PANEL_SIZE_A)

    print("  Panel B: Trajectory...")
    if 'B' in panels:
        create_fig2b(fig=sf_b, style=style, panel_size=PANEL_SIZE_B)

    print("  Panel C: Venn diagrams...")
    if 'C' in panels:
        create_fig2c(fig=sf_c, style=style, panel_size=PANEL_SIZE_C)

    print("  Panel D: Activity maps...")
    if 'D' in panels:
        create_fig2d(fig=sf_d, style=style, panel_size=PANEL_SIZE_D)

    print("  Panel E: Metrics...")
    if 'E' in panels:
        create_fig2e(fig=sf_e, style=style, panel_size=PANEL_SIZE_E)

    print("  Panel F: Fraction plot...")
    if 'F' in panels:
        create_fig2f(fig=sf_f, style=style, panel_size=PANEL_SIZE_F)

    # ==========================================================================
    # PANEL LABELS — placed on the main figure
    # ==========================================================================
    print("  Adding panel labels...")
    label_fontsize = 12
    label_fontweight = 'bold'

    # Compute label positions in figure coordinates (per-row column starts)
    row2_top = (ROW1_H + V_GAP1_CM) / CANVAS_HEIGHT_CM
    row3_top = (ROW1_H + V_GAP1_CM + ROW2_H + V_GAP2_CM) / CANVAS_HEIGHT_CM

    label_x_left = 0.005
    # Align B and D vertically — shifted left for tighter placement
    label_x_right_col = (COL_A_W + H_GAP_CM) / CANVAS_WIDTH_CM - 0.025
    # F label shifted further left
    label_x_f = (COL_E_W + E_F_GAP_CM) / CANVAS_WIDTH_CM - 0.01

    label_positions = {
        'A': (label_x_left,      1.0 - 0.005),
        'B': (label_x_right_col, 1.0 - 0.005),
        'C': (label_x_left,      1.0 - row2_top + 0.005),
        'D': (label_x_right_col, 1.0 - row2_top + 0.005),
        'E': (label_x_left,      1.0 - row3_top + 0.005),
        'F': (label_x_f,         1.0 - row3_top + 0.005),
    }

    for name, (x, y) in label_positions.items():
        fig.text(x, y, name.lower() if lowercase else name, fontsize=label_fontsize,
                 fontweight=label_fontweight, va='top', ha='left')
    fig._label_positions = label_positions
    fig._sf_a = sf_a

    # ==========================================================================
    # SAVE
    # ==========================================================================
    for fmt in save_formats:
        output_path = output_dir / f'{output_name}.{fmt}'
        if output_path.exists():
            raise FileExistsError(output_path)
        fig.savefig(output_path, dpi=DPI, facecolor='white', edgecolor='none')
        print(f"  Saved: {output_path}")

    return fig


if __name__ == '__main__':
    raise SystemExit("run ../make_fig2.py")
