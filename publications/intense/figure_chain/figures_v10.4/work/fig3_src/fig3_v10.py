"""Figure 3: the composer of the panels, with the output folder and name as arguments and the panel
letters in lower case when lowercase=True. Panel C reads the per-session table
<[figures] cache_dir>/fig3c_per_session.csv.

Figure 3: Combined INTENSE real-data results — Delegation approach.

Layout (left-to-right, top-to-bottom):
    Row 1 (25%):  A — object selectivity: [mouse A.png | fig3a trace+density]
    Row 2 (25%):  B — rear selectivity:   [mouse B.png | fig3b trace+density]
    Row 3 (20%):  C bar chart (30%)  |  D polar plot (30%)  |  E place heatmap (40%)
    Row 4 (30%):  F feature-feature MI heatmap (50%)  |  G Venn diagram (50%)

Panel sources:
    A left:  mouse A.png — mouse sniffing object
    A right: fig3a_object_delegated.py — NOF H02 3D neuron 312, object 3
    B left:  mouse B.png — mouse rearing against wall
    B right: fig3b_rear_delegated.py — NOF H31 2D neuron 35, rear
    C:       fig3c_bars_delegated.py — n_sel + delay sign bars (with significance)
    D:       fig3d_polar_delegated.py — head direction tuning, NOF H36 3D neuron 158
    E:       fig3e_heatmap_place_delegated.py — spatial map of the same neuron
    F:       fig3f_heatmap_delegated.py — feature-feature MI, mean over 64 NOF sessions

All data: seeded DRIADA 1.2.2 NOF recompute; run fig3_numbers.py first.
    G:       fig3g_venn_delegated.py — Venn diagram MI decomposition
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import _v122  # noqa: E402,F401  DRIADA 1.2.2 snapshot first on sys.path

from driada.utils.publication import StylePreset

# Import individual panel creation functions (support both package and script mode)
try:
    from .fig3a_object_delegated import create_fig3a
    from .fig3b_rear_delegated import create_fig3b
    from .fig3c_bars_delegated import create_fig3c
    from .fig3d_polar_delegated import create_fig3d
    from .fig3e_heatmap_place_delegated import create_fig3e
    from .fig3f_heatmap_delegated import create_fig3f
    from .fig3g_venn_delegated import create_fig3g
except ImportError:
    from fig3a_object_delegated import create_fig3a
    from fig3b_rear_delegated import create_fig3b
    from fig3c_bars_delegated import create_fig3c
    from fig3d_polar_delegated import create_fig3d
    from fig3e_heatmap_place_delegated import create_fig3e
    from fig3f_heatmap_delegated import create_fig3f
    from fig3g_venn_delegated import create_fig3g


# ==========================================================================
# LAYOUT CONSTANTS (cm)
# ==========================================================================
CANVAS_WIDTH_CM = 18.0
CANVAS_HEIGHT_CM = 24.0
DPI = 600

# Vertical gaps between rows (cm)
V_GAP_12_CM = 0.3   # between rows 1-2 (tight, same panel type)
V_GAP_23_CM = 0.6   # between rows 2-3
V_GAP_34_CM = 0.8   # between rows 3-4

# Horizontal gaps (cm)
H_GAP_CM = 0.3

# Bottom margin
BOTTOM_MARGIN_FRAC = 0.03

# Available height for content
_gaps_total = V_GAP_12_CM + V_GAP_23_CM + V_GAP_34_CM
_bottom_cm = CANVAS_HEIGHT_CM * BOTTOM_MARGIN_FRAC
_row_total = CANVAS_HEIGHT_CM - _gaps_total - _bottom_cm

# Row height ratios: 25:25:22:28
ROW1_H = _row_total * 0.25  # A: object selectivity
ROW2_H = _row_total * 0.25  # B: rear selectivity
ROW3_H = _row_total * 0.22  # C, D, E (×1.1)
ROW4_H = _row_total * 0.28  # F, G

# Row 1 & 2 internal splits
ICON_W_FRAC = 0.20
PLOT_W_FRAC = 1.0 - ICON_W_FRAC

# Row 3 column width ratios + asymmetric gaps (D closer to E than to C)
COL_C_RATIO = 0.28
GAP_CD_RATIO = 0.05   # larger gap: C ←→ D
COL_D_RATIO = 0.28
GAP_DE_RATIO = 0.02   # smaller gap: D ←→ E
COL_E_RATIO = 0.37

# Row 4 column width ratios
COL_F_RATIO = 0.50
COL_G_RATIO = 0.50

# Physical column widths (cm) — for panel_size arguments
_row3_avail = CANVAS_WIDTH_CM  # gaps handled by wspace
COL_C_W = _row3_avail * COL_C_RATIO
COL_D_W = _row3_avail * COL_D_RATIO
COL_E_W = _row3_avail * COL_E_RATIO
COL_F_W = CANVAS_WIDTH_CM * COL_F_RATIO
COL_G_W = CANVAS_WIDTH_CM * COL_G_RATIO


def _load_mouse_images():
    """Load separate mouse A.png and mouse B.png images."""
    import _paths_fig as _F  # figure_chain/ is on sys.path (_v122.py)
    # 'mouse A.png' and 'mouse B.png' of this folder are the drawings of [figures] fig3_mouse_a and
    # fig3_mouse_b (assets/ of the paper folder) byte for byte
    img_a = plt.imread(_F.P.FIG3_MOUSE_A)
    img_b = plt.imread(_F.P.FIG3_MOUSE_B)
    return img_a, img_b


def _place_image(subfig, img):
    """Place an image into a subfigure, preserving original aspect ratio.

    Image is vertically aligned with the adjacent plot area
    (axes span bottom=0.55 to top=0.92, legend extends to ~0.43).
    """
    ax = subfig.subplots()
    ax.imshow(img, aspect='equal')
    ax.set_anchor('C')
    ax.axis('off')
    # Axes positioned to center image at ~0.67 (midpoint of visible plot content)
    subfig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.35)


def create_fig3_delegated(output_dir=None, save_formats=('png',), output_name='fig3',
                          lowercase=True, panels='ABCDEFG', create_c=None):
    """Create Figure 3 by delegating to individual panel scripts."""
    if output_dir is None:
        import _paths_fig as _F
        output_dir = _F.out_dir(Path(__file__).resolve().parent.parent) / 'fig3_out'
    output_dir = Path(output_dir)

    # ==========================================================================
    # FIGURE CANVAS
    # ==========================================================================
    fig_w = CANVAS_WIDTH_CM / 2.54
    fig_h = CANVAS_HEIGHT_CM / 2.54
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, facecolor='white')

    # ==========================================================================
    # MASTER GRID: 4 content rows + 3 gap rows + 1 bottom margin
    # ==========================================================================
    gs_main = fig.add_gridspec(
        8, 1,
        height_ratios=[ROW1_H, V_GAP_12_CM, ROW2_H, V_GAP_23_CM,
                        ROW3_H, V_GAP_34_CM, ROW4_H, _bottom_cm],
        hspace=0,
    )

    # Create row-level subfigures (skip gap rows 1, 3, 5, 7)
    subfig_row1 = fig.add_subfigure(gs_main[0, 0])
    subfig_row2 = fig.add_subfigure(gs_main[2, 0])
    subfig_row3 = fig.add_subfigure(gs_main[4, 0])
    subfig_row4 = fig.add_subfigure(gs_main[6, 0])

    # ==========================================================================
    # STYLE — single preset for all panels (protocol §7.1)
    # ==========================================================================
    style = StylePreset.publication_default().copy(
        base_tick_length=3,
        base_tick_pad=4,
        base_tick_labelsize=8,
        base_label_size=9,
        base_legend_fontsize=7,
        base_annotation_fontsize=7,
        base_line_width=1.0,
        base_spine_width=1.0,
    )

    # ==========================================================================
    # MICE IMAGE — load and split
    # ==========================================================================
    print("Loading mouse images...")
    img_a, img_b = _load_mouse_images()

    # ==========================================================================
    # ROW 1: Panel A — icon (15%) + object selectivity (85%)
    # ==========================================================================
    print("Creating Panel A: Object selectivity...")
    gs_r1 = subfig_row1.add_gridspec(1, 2, width_ratios=[ICON_W_FRAC, PLOT_W_FRAC],
                                      wspace=0)
    subfig_a_icon = subfig_row1.add_subfigure(gs_r1[0, 0])
    subfig_a = subfig_row1.add_subfigure(gs_r1[0, 1])

    panel_size_a = (CANVAS_WIDTH_CM * PLOT_W_FRAC, ROW1_H)
    if 'A' in panels:
        _place_image(subfig_a_icon, img_a)
        create_fig3a(fig=subfig_a, style=style, panel_size=panel_size_a)

    # ==========================================================================
    # ROW 2: Panel B — icon (15%) + rear selectivity (85%)
    # ==========================================================================
    print("Creating Panel B: Rear selectivity...")
    gs_r2 = subfig_row2.add_gridspec(1, 2, width_ratios=[ICON_W_FRAC, PLOT_W_FRAC],
                                      wspace=0)
    subfig_b_icon = subfig_row2.add_subfigure(gs_r2[0, 0])
    subfig_b = subfig_row2.add_subfigure(gs_r2[0, 1])

    panel_size_b = (CANVAS_WIDTH_CM * PLOT_W_FRAC, ROW2_H)
    if 'B' in panels:
        _place_image(subfig_b_icon, img_b)
        create_fig3b(fig=subfig_b, style=style, panel_size=panel_size_b)

    # ==========================================================================
    # ROW 3: C | gap | D | gap | E  (asymmetric gaps: D closer to E)
    # ==========================================================================
    gs_r3 = subfig_row3.add_gridspec(
        1, 5,
        width_ratios=[COL_C_RATIO, GAP_CD_RATIO, COL_D_RATIO,
                       GAP_DE_RATIO, COL_E_RATIO],
        wspace=0)
    subfig_c = subfig_row3.add_subfigure(gs_r3[0, 0])
    subfig_d = subfig_row3.add_subfigure(gs_r3[0, 2])
    subfig_e = subfig_row3.add_subfigure(gs_r3[0, 4])

    print("Creating Panel C: Bar charts...")
    panel_size_c = (COL_C_W, ROW3_H)
    if 'C' in panels:
        _, axes_c = (create_c or create_fig3c)(fig=subfig_c, style=style, panel_size=panel_size_c)
        fig._panel_c = (subfig_c, axes_c)

    print("Creating Panel D: Polar plot...")
    panel_size_d = (COL_D_W, ROW3_H)
    if 'D' in panels:
        create_fig3d(fig=subfig_d, style=style, panel_size=panel_size_d)

    print("Creating Panel E: Place heatmap...")
    panel_size_e = (COL_E_W, ROW3_H)
    if 'E' in panels:
        create_fig3e(fig=subfig_e, style=style, panel_size=panel_size_e)

    # ==========================================================================
    # ROW 4: F (50%) | G (50%)
    # ==========================================================================
    _avg_col4 = CANVAS_WIDTH_CM / 2
    _wspace4 = H_GAP_CM / _avg_col4

    gs_r4 = subfig_row4.add_gridspec(
        1, 2, width_ratios=[COL_F_RATIO, COL_G_RATIO],
        wspace=_wspace4)
    subfig_f = subfig_row4.add_subfigure(gs_r4[0, 0])
    subfig_g = subfig_row4.add_subfigure(gs_r4[0, 1])

    print("Creating Panel F: MI heatmap...")
    panel_size_f = (COL_F_W, ROW4_H)
    if 'F' in panels:
        create_fig3f(fig=subfig_f, style=style, panel_size=panel_size_f)

    print("Creating Panel G: Venn diagram...")
    panel_size_g = (COL_G_W, ROW4_H)
    if 'G' in panels:
        create_fig3g(fig=subfig_g, style=style, panel_size=panel_size_g)

    # ==========================================================================
    # PANEL LABELS — placed on the main figure in figure coordinates
    # ==========================================================================
    print("Adding panel labels...")
    label_fs = 11
    label_fw = 'bold'

    # Compute y-positions of each row top in figure coordinates
    _total_h = (ROW1_H + V_GAP_12_CM + ROW2_H + V_GAP_23_CM
                + ROW3_H + V_GAP_34_CM + ROW4_H + _bottom_cm)
    def _y_top(cum_from_top):
        """Convert cumulative cm from top to figure y-fraction."""
        return 1.0 - cum_from_top / _total_h

    y_r1 = _y_top(0)
    y_r2 = _y_top(ROW1_H + V_GAP_12_CM)
    y_r3 = _y_top(ROW1_H + V_GAP_12_CM + ROW2_H + V_GAP_23_CM)
    y_r4 = _y_top(ROW1_H + V_GAP_12_CM + ROW2_H + V_GAP_23_CM
                   + ROW3_H + V_GAP_34_CM)

    # x-positions: all labels at left edge of their column
    lx = 0.005  # small offset from column left
    ty = 0.008
    # Shift C,D,E up by ~1 font size (11pt / canvas_height_inches)
    _fs_frac = (label_fs / 72) / fig_h  # ~0.016
    # Row 3 x-positions (exact from 5-column layout)
    x_c_frac = 0.0
    x_d_frac = COL_C_RATIO + GAP_CD_RATIO
    x_e_frac = COL_C_RATIO + GAP_CD_RATIO + COL_D_RATIO + GAP_DE_RATIO

    # Row 4 x-positions
    x_f_frac = 0.0
    x_g_frac = COL_F_RATIO + _wspace4 * COL_F_RATIO

    label_positions = {
        'A': (lx,                    y_r1 - ty),          # above mice image
        'B': (lx,                    y_r2 - ty),          # above mice image
        'C': (x_c_frac + lx,        y_r3 - ty + _fs_frac),  # up by font size
        'D': (x_d_frac + lx,         y_r3 - ty + _fs_frac),
        'E': (x_e_frac + lx,         y_r3 - ty + _fs_frac),
        'F': (x_f_frac + lx,        y_r4 - ty),
        'G': (x_g_frac + lx - 0.02, y_r4 - ty),  # shift left
    }

    for name, (lbl_x, lbl_y) in label_positions.items():
        fig.text(lbl_x, lbl_y, name.lower() if lowercase else name, fontsize=label_fs,
                 fontweight=label_fw, va='top', ha='left')
    fig._label_positions = label_positions
    fig._subfigs = dict(A=subfig_a, B=subfig_b, C=subfig_c, D=subfig_d, E=subfig_e, F=subfig_f, G=subfig_g,
                        A_icon=subfig_a_icon, B_icon=subfig_b_icon)

    # ==========================================================================
    # SAVE
    # ==========================================================================
    for fmt in save_formats:
        output_path = output_dir / f'{output_name}.{fmt}'
        if output_path.exists():
            raise FileExistsError(output_path)
        fig.savefig(output_path, dpi=DPI, facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


if __name__ == '__main__':
    raise SystemExit("run ../make_fig3.py")
