"""Figure 4: INTENSE synthetic validation — Delegation approach.

Layout (from reference):
    Top row (~60%):
        LEFT:  Panel A — generation pipeline (4 stacked subplots)
        RIGHT: Panel B — continuous F1 heatmaps (2×2)
               Panel C — discrete F1 heatmaps (2×2)

    Bottom row (~37%):
        LEFT:  Panel D — continuous PR curves (1×2)
        RIGHT: Panel E — discrete PR curves (1×2)

Structure:
    3 row-subfigures (top_content, gap, bottom_content) × 1
    Each content row splits into 2 column-subfigures.
    Top-right column further splits into B + gap + C.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from driada.utils.publication import StylePreset

from .fig4a_generation_pipeline import create_fig4a
from .fig4bc_heatmaps import create_heatmap_panel
from .fig4de_pr_curves import create_pr_panel


# ==========================================================================
# LAYOUT CONSTANTS (cm)
# ==========================================================================
CANVAS_WIDTH_CM = 18.0
CANVAS_HEIGHT_CM = 24.0
DPI = 300

# Column widths
LEFT_COL_CM = 8.5
RIGHT_COL_CM = CANVAS_WIDTH_CM - LEFT_COL_CM   # 8.6
H_GAP_CM = 0.3
TOP_H_GAP_CM = 1.0   # wider gap between A and B/C in top row

# Row heights (fraction of canvas)
TOP_ROW_FRAC = 0.64
V_GAP_FRAC = 0.04
BOTTOM_ROW_FRAC = 1.0 - TOP_ROW_FRAC - V_GAP_FRAC  # 0.32

# Within right column of top row: B, gap, C
BC_GAP_FRAC = 0.01   # fraction of top-right subfigure height
B_FRAC = 0.495
C_FRAC = 1.0 - B_FRAC - BC_GAP_FRAC  # 0.495

# SNR and skip_prob values
SNRS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
SKPROBS = np.linspace(0.0, 0.8, 9)


def create_fig4_delegated(output_dir=None, save_formats=('png', 'pdf'),
                          c_data=None, d_data=None, output_name='fig4_delegated',
                          lowercase=False):
    """Create Figure 4 by delegating to individual panel scripts.

    Parameters
    ----------
    output_dir : Path, optional
    save_formats : tuple of str
    c_data, d_data : ndarray, optional
        Pre-loaded NPZ data. If None, loads old NPZ files.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir = Path(output_dir)

    # ======================================================================
    # LOAD DATA
    # ======================================================================
    if c_data is None or d_data is None:
        data_dir = Path(__file__).parent.parent
        c_data = np.load(data_dir / 'continuous INTENSE test mult v2.npz')['arr_0']
        d_data = np.load(data_dir / 'discrete INTENSE test mult v2.npz')['arr_0']

    # ======================================================================
    # FIGURE AND SUBFIGURE LAYOUT
    # ======================================================================
    fig_w = CANVAS_WIDTH_CM / 2.54
    fig_h = CANVAS_HEIGHT_CM / 2.54
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    # 3 rows: top content, gap, bottom content
    row_subfigs = fig.subfigures(
        3, 1,
        height_ratios=[TOP_ROW_FRAC, V_GAP_FRAC, BOTTOM_ROW_FRAC],
        hspace=0,
    )
    top_sf = row_subfigs[0]
    # row_subfigs[1] is gap (empty)
    bot_sf = row_subfigs[2]

    # --- Top row: A (left) | B+C (right) ---
    top_cols = top_sf.subfigures(
        1, 2,
        width_ratios=[LEFT_COL_CM, RIGHT_COL_CM],
        wspace=TOP_H_GAP_CM / CANVAS_WIDTH_CM,
    )
    sf_a = top_cols[0]
    sf_bc = top_cols[1]

    # --- Top-right: B (top) + gap + C (bottom) ---
    bc_rows = sf_bc.subfigures(
        3, 1,
        height_ratios=[B_FRAC, BC_GAP_FRAC, C_FRAC],
        hspace=0,
    )
    sf_b = bc_rows[0]
    # bc_rows[1] is gap
    sf_c = bc_rows[2]

    # --- Bottom row: D (left) | E (right) ---
    bot_cols = bot_sf.subfigures(
        1, 2,
        width_ratios=[1, 1],
        wspace=H_GAP_CM / CANVAS_WIDTH_CM,
    )
    sf_d = bot_cols[0]
    sf_e = bot_cols[1]

    # ======================================================================
    # DELEGATE TO PANEL SCRIPTS
    # ======================================================================
    # Shared StylePreset — publication-quality sizes for composite panels
    style = StylePreset.publication_default().copy(
        base_tick_length=3,
        base_tick_pad=4,
        base_label_size=9,
        base_legend_fontsize=7,
        base_annotation_fontsize=7,
        base_line_width=1.0,
    )

    # Panel sizes in cm (matching subfigure allocations)
    panel_size_a = (LEFT_COL_CM, CANVAS_HEIGHT_CM * TOP_ROW_FRAC)
    panel_size_b = (RIGHT_COL_CM, CANVAS_HEIGHT_CM * TOP_ROW_FRAC * B_FRAC)
    panel_size_c = (RIGHT_COL_CM, CANVAS_HEIGHT_CM * TOP_ROW_FRAC * C_FRAC)
    bot_col_cm = (CANVAS_WIDTH_CM - H_GAP_CM) / 2  # equal split
    panel_size_d = (bot_col_cm, CANVAS_HEIGHT_CM * BOTTOM_ROW_FRAC)
    panel_size_e = (bot_col_cm, CANVAS_HEIGHT_CM * BOTTOM_ROW_FRAC)

    print("Creating Figure 4...")

    print("  Panel A: Generation pipeline...")
    create_fig4a(fig=sf_a, style=style, panel_size=panel_size_a)

    print("  Panel B: Continuous heatmaps...")
    create_heatmap_panel(c_data, 'c', fig=sf_b, snrs=SNRS, skprobs=SKPROBS,
                         style=style, panel_size=panel_size_b,
                         vmax=1.0, show_colorbar=False)

    print("  Panel C: Discrete heatmaps...")
    create_heatmap_panel(d_data, 'd', fig=sf_c, snrs=SNRS, skprobs=SKPROBS,
                         style=style, panel_size=panel_size_c,
                         vmax=1.0, show_colorbar=True)

    # PR curves skip SNR=2 (noisy, uninformative)
    pr_snrs = SNRS[1:]
    c_data_pr = c_data[:, :, :, :, 1:, :]
    d_data_pr = d_data[:, :, :, :, 1:, :]

    print("  Panel D: Continuous PR curves...")
    create_pr_panel(c_data_pr, 'c', fig=sf_d, snrs=pr_snrs, skprobs=SKPROBS,
                    style=style, panel_size=panel_size_d)

    print("  Panel E: Discrete PR curves...")
    create_pr_panel(d_data_pr, 'd', fig=sf_e, snrs=pr_snrs, skprobs=SKPROBS,
                    style=style, panel_size=panel_size_e)

    # ======================================================================
    # PANEL LABELS — placed on the main figure
    # ======================================================================
    print("  Adding panel labels...")
    label_fontsize = 12
    label_fontweight = 'bold'

    left_x = 0.005
    right_x = LEFT_COL_CM / CANVAS_WIDTH_CM + 0.005  # aligned for B, C, E

    # Y positions: top of each row
    top_y = 0.995
    bc_split_y = TOP_ROW_FRAC * (1.0 - B_FRAC - BC_GAP_FRAC / 2)
    c_label_y = 1.0 - TOP_ROW_FRAC * (B_FRAC + BC_GAP_FRAC) + 0.005
    bot_y = 1.0 - TOP_ROW_FRAC - V_GAP_FRAC + 0.005

    label_positions = {
        'A': (left_x,      top_y),
        'B': (right_x, top_y),
        'C': (right_x, c_label_y),
        'D': (left_x,  bot_y),
        'E': (right_x, bot_y),
    }

    for name, (x, y) in label_positions.items():
        fig.text(x, y, name.lower() if lowercase else name, fontsize=label_fontsize,
                 fontweight=label_fontweight, va='top', ha='left')

    # ======================================================================
    # SAVE
    # ======================================================================
    for fmt in save_formats:
        output_path = output_dir / f'{output_name}.{fmt}'
        if output_path.exists():
            raise FileExistsError(output_path)
        fig.savefig(output_path, dpi=DPI, facecolor='white', edgecolor='none')
        print(f"  Saved: {output_path}")

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate combined INTENSE Figure 4 (delegated)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figure')
    args = parser.parse_args()

    print("=" * 60)
    print("Creating INTENSE Figure 4 (Delegated Approach)")
    print("=" * 60)

    fig = create_fig4_delegated()

    if not args.no_show:
        plt.show()

    print("Done!")
