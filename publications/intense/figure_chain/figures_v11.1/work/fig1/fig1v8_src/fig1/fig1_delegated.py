"""Figure 1: the composer of the panels, with the output name as an argument, the panel letters in lower
case when lowercase=True and the refusal to overwrite. Label switches live in the panel modules:
fig1a_calcium_traces.NEURON_LABEL_IDX, fig1c_neuron_feature_pair.NEURON_LABEL,
fig1e_scatter_marginals.LEGEND_COUNTS.

Figure 1: Combined INTENSE method overview - Delegation approach.

This script uses the publication framework for layout/dimensions while
delegating actual plotting to individual panel scripts. Each panel script
uses the publication framework's StylePreset for consistent styling.

Key principle: Framework controls SIZE, StylePreset controls STYLE.
StylePreset provides auto-scaling based on panel size.

Panel dimensions (measured from reference at 600 DPI):
  Left column (same width=11.85cm for aligned x-axes):
    - Panel A: Calcium traces (11.85 × 7.0 cm)
    - Panel B: Behavioral traces (11.85 × 8.0 cm)
    - Panel D: Neuron-behavior pair (11.85 × 6.5 cm)
  Right column:
    - Panel C: Scheme/workflow (6.5 × 15.2 cm)
    - Panel E: Scatter with marginals (6.5 × 6.5 cm)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pathlib import Path

from driada.utils.publication import PanelLayout, PanelLabeler, StylePreset

# Import individual panel creation functions
from .fig1a_calcium_traces import create_fig1a, RU_LABELS
from .fig1b_behavioral_traces import create_fig1b
from .fig1c_neuron_feature_pair import create_fig1c
from .fig1e_scatter_marginals import create_fig1e


import _paths_fig as _F  # noqa: E402  (figure_chain/ is on sys.path, see data_loader.py)

# SCHEME_PATH: raster scheme of panel C ([figures] fig1_scheme).
SCHEME_PATH = _F.P.FIG1_SCHEME


def create_fig1_delegated(output_dir=None, save_formats=('png',),
                          fig1a_labels=None, fig1a_neuron_fontscale=1.0,
                          output_name='fig1_real', lowercase=True):
    """Create Figure 1 by delegating to individual panel scripts.

    Each panel script uses the publication framework's StylePreset
    for auto-scaled styling. StylePreset scales line widths, fonts,
    and markers based on the panel size.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir = Path(output_dir)

    # ==========================================================================
    # CANVAS AND DPI - Hardcoded to match reference exactly
    # ==========================================================================
    # Add margins from top and left to prevent label truncation
    LEFT_MARGIN = 0.1
    TOP_MARGIN = 0.6  # Increased for panel C label
    # Canvas width expanded to fit Panel E (1.25 * Panel D height)
    CANVAS_WIDTH_CM = 18.88 + LEFT_MARGIN  # Expanded to fit larger Panel E
    CANVAS_HEIGHT_CM = 21.52 + TOP_MARGIN  # 22.12 cm with margin
    DPI = 600

    # ==========================================================================
    # STYLE - Single source of truth using publication framework
    # ==========================================================================
    style = StylePreset.publication_default().copy(
        base_tick_length=3,
        base_tick_pad=4,  # reduced from 8 to decrease tick-label spacing
        base_legend_fontsize=7,
        base_annotation_fontsize=7,
        base_line_width=1.0,
        base_label_size=9,  # reduced by 1 from default 10
    )

    # ==========================================================================
    # PANEL POSITIONS - User-provided measurements from reference (in cm)
    # EXACT values - do NOT change without user approval (1% tolerance max)
    # All positions shifted by margins
    # ==========================================================================
    # Vertical positions (top, bottom) in cm from top of canvas:
    # Each of A, B, D increased by 1 cm in height (+3 cm total)
    PANEL_A_TOP, PANEL_A_BOTTOM = 0.20 + TOP_MARGIN, 7.66 + TOP_MARGIN      # +1 cm height
    PANEL_B_TOP, PANEL_B_BOTTOM = 7.91 + TOP_MARGIN, 16.00 + TOP_MARGIN     # shifted +1, +1 cm height
    PANEL_D_TOP, PANEL_D_BOTTOM = 16.34 + TOP_MARGIN, 21.52 + TOP_MARGIN    # shifted +2, +1 cm height

    # Column widths - left column starts at LEFT_MARGIN
    LEFT_COL_WIDTH = 12.11  # Increased by 1 cm

    # Calculate Panel D height (increased by 2 cm total)
    PANEL_D_HEIGHT = (21.52 + TOP_MARGIN) - (16.34 + TOP_MARGIN) + 2.0  # 7.18 cm

    # Panel E: reduced by 1.5 cm from previous 1.25 * original D height
    PANEL_E_SIZE = (5.18 * 1.25) - 1.5  # ~4.975 cm square

    # Right column width - slightly wider than Panel E for larger Panel C image
    RIGHT_COL_WIDTH = PANEL_E_SIZE + 0.5  # 0.5 cm wider for Panel C

    # Panel C: increased slightly (take space from spacer to keep E in place)
    # Net 0.03 cm reduction to pull Panel E up slightly
    PANEL_C_HEIGHT = CANVAS_HEIGHT_CM - PANEL_E_SIZE - 1.5 - TOP_MARGIN - 0.03

    # Calculate heights (PANEL_D_HEIGHT already calculated above for Panel E sizing)
    PANEL_A_HEIGHT = PANEL_A_BOTTOM - PANEL_A_TOP  # 7.46 cm
    PANEL_B_HEIGHT = PANEL_B_BOTTOM - PANEL_B_TOP  # 8.09 cm
    # PANEL_D_HEIGHT already set above = 5.18 cm
    # PANEL_C_HEIGHT already set above
    PANEL_E_HEIGHT = PANEL_E_SIZE  # square

    # Gaps between panels
    GAP_A_B = PANEL_B_TOP - PANEL_A_BOTTOM  # 0.25 cm
    GAP_B_D = PANEL_D_TOP - PANEL_B_BOTTOM  # 0.34 cm

    panel_size_a = (LEFT_COL_WIDTH, PANEL_A_HEIGHT)
    panel_size_b = (LEFT_COL_WIDTH, PANEL_B_HEIGHT)
    panel_size_d = (LEFT_COL_WIDTH, PANEL_D_HEIGHT)
    panel_size_c = (RIGHT_COL_WIDTH, PANEL_C_HEIGHT)
    panel_size_e = (PANEL_E_SIZE, PANEL_E_SIZE)  # Square!

    # ==========================================================================
    # LAYOUT - Two independent columns using nested subfigures
    # Left column: A, B, D (stacked)
    # Right column: C, E (stacked)
    # No dependency between D and E heights
    # ==========================================================================
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Calculate figure size in inches
    fig_width_inches = CANVAS_WIDTH_CM / 2.54
    fig_height_inches = CANVAS_HEIGHT_CM / 2.54

    # Create main figure
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=DPI)

    # Create two column subfigures
    # Left column width ratio vs right column
    left_width_ratio = LEFT_COL_WIDTH / (LEFT_COL_WIDTH + RIGHT_COL_WIDTH)
    right_width_ratio = RIGHT_COL_WIDTH / (LEFT_COL_WIDTH + RIGHT_COL_WIDTH)

    # Calculate top margin fraction (content starts below margin)
    top_margin_frac = TOP_MARGIN / CANVAS_HEIGHT_CM
    content_top = 1.0 - top_margin_frac

    # Use subfigures() to create two columns with top margin
    # subfigures() accepts height_ratios and width_ratios
    subfigs_row = fig.subfigures(2, 2, height_ratios=[top_margin_frac, content_top],
                                  width_ratios=[left_width_ratio, right_width_ratio],
                                  wspace=0, hspace=0)
    # Row 0 is the margin (empty), row 1 contains the content
    subfig_left = subfigs_row[1, 0]
    subfig_right = subfigs_row[1, 1]

    # --- Left column: A, B, D ---
    left_total_height = PANEL_A_HEIGHT + PANEL_B_HEIGHT + PANEL_D_HEIGHT
    gs_left = subfig_left.add_gridspec(
        3, 1,
        height_ratios=[PANEL_A_HEIGHT, PANEL_B_HEIGHT, PANEL_D_HEIGHT],
        hspace=0
    )
    subfig_a = subfig_left.add_subfigure(gs_left[0, 0])
    subfig_b = subfig_left.add_subfigure(gs_left[1, 0])
    subfig_d = subfig_left.add_subfigure(gs_left[2, 0])

    # --- Right column: C, E, spacer ---
    # Add spacer below E to push it up (align E with Panel D region)
    BOTTOM_SPACER = 3.28  # cm below E (net increase of 0.03 cm from original 3.25)
    gs_right = subfig_right.add_gridspec(
        3, 1,
        height_ratios=[PANEL_C_HEIGHT, PANEL_E_SIZE, BOTTOM_SPACER],
        hspace=0
    )
    subfig_c = subfig_right.add_subfigure(gs_right[0, 0])
    subfig_e = subfig_right.add_subfigure(gs_right[1, 0])
    # Row 2 is empty spacer

    # Build subfigs dict for compatibility
    subfigs = {
        'A': subfig_a,
        'B': subfig_b,
        'C': subfig_c,
        'D': subfig_d,
        'E': subfig_e,
    }

    print("Creating figure layout...")

    # ==========================================================================
    # PANEL A - Calcium traces (delegate to fig1a)
    # ==========================================================================
    print("Creating Panel A: Calcium traces...")
    # Pass subfigure - script creates its own axes and handles layout
    create_fig1a(fig=subfigs['A'], style=style, panel_size=panel_size_a,
                 labels=fig1a_labels, neuron_fontscale=fig1a_neuron_fontscale)

    # ==========================================================================
    # PANEL B - Behavioral traces (delegate to fig1b)
    # ==========================================================================
    print("Creating Panel B: Behavioral traces...")
    # Pass subfigure - script creates its own axes and handles layout
    create_fig1b(fig=subfigs['B'], style=style, panel_size=panel_size_b)

    # ==========================================================================
    # PANEL C - Scheme/workflow diagram (image)
    # ==========================================================================
    print("Creating Panel C: Scheme...")
    subfig_c = subfigs['C']

    # Panel C now has its own correctly-sized subfigure
    ax_c = subfig_c.subplots(1, 1)

    scheme_path = SCHEME_PATH
    if scheme_path.exists():
        img = plt.imread(scheme_path)
        # Use fixed aspect ratio - image will be smaller if panel can't support it
        ax_c.imshow(img, aspect='equal')
        # Anchor to top so empty space is at bottom, not distributed
        ax_c.set_anchor('N')
    else:
        ax_c.text(0.5, 0.5, 'Workflow\nDiagram\n(image not found)',
                  ha='center', va='center', fontsize=style.base_label_size,
                  transform=ax_c.transAxes, color='gray')
    ax_c.axis('off')

    # Use full subfigure space, let aspect ratio handle sizing
    subfig_c.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    # ==========================================================================
    # PANEL D - Neuron-feature pair with density (delegate to fig1c)
    # ==========================================================================
    print("Creating Panel D: Neuron-feature pair...")
    # Pass subfigure - script creates its own axes and handles layout
    create_fig1c(fig=subfigs['D'], style=style, panel_size=panel_size_d)

    # ==========================================================================
    # PANEL E - Scatter with marginals (delegate to fig1e)
    # ==========================================================================
    print("Creating Panel E: Scatter with marginals...")
    subfig_e = subfigs['E']

    # Pass subfigure, style, and panel size - full delegation!
    create_fig1e(fig=subfig_e, style=style, panel_size=panel_size_e)

    # ==========================================================================
    # PANEL LABELS - Aligned vertically within columns on main figure
    # ==========================================================================
    print("Adding panel labels...")
    label_fontsize = 12
    label_fontweight = 'bold'

    # Calculate column x-positions in figure coordinates
    total_width = LEFT_COL_WIDTH + RIGHT_COL_WIDTH
    left_col_frac = LEFT_COL_WIDTH / total_width

    # 0.3 cm right shift: 0.3 / CANVAS_WIDTH_CM ≈ 0.016
    left_label_x = 0.005 + 0.3 / CANVAS_WIDTH_CM  # Shifted 0.3 cm right
    right_label_x = left_col_frac + 0.005  # Small offset from right column start

    # Calculate y-positions for left column (A, B, D)
    # Content starts at y = 1.0 - top_margin_frac
    top_margin_frac = TOP_MARGIN / CANVAS_HEIGHT_CM
    content_top = 1.0 - top_margin_frac
    content_height_frac = 1.0 - top_margin_frac  # Content spans from 0 to content_top

    left_total = PANEL_A_HEIGHT + PANEL_B_HEIGHT + PANEL_D_HEIGHT
    y_offset = 0.005  # Small offset from top

    # Top labels (A, C) go in the margin area
    top_label_y = 1.0 - 0.005  # Near very top of figure (in margin)

    # Calculate E's y-position first (to align D with it)
    right_total = PANEL_C_HEIGHT + PANEL_E_SIZE + BOTTOM_SPACER
    e_label_y = content_top - (PANEL_C_HEIGHT / right_total) * content_height_frac + y_offset

    # Calculate original D position
    d_original_y = content_top - ((PANEL_A_HEIGHT + PANEL_B_HEIGHT) / left_total) * content_height_frac + y_offset

    # Calculate shift needed to align D with E
    d_shift = e_label_y - d_original_y

    label_positions = {
        'A': (left_label_x, top_label_y),
        'B': (left_label_x, content_top - (PANEL_A_HEIGHT / left_total) * content_height_frac + y_offset + d_shift),
        'D': (left_label_x, e_label_y),  # Aligned with E
    }

    # Right column labels
    label_positions['C'] = (right_label_x, top_label_y)  # Same height as A (in margin)
    label_positions['E'] = (right_label_x, e_label_y)

    # Add labels to main figure
    for name, (x, y) in label_positions.items():
        fig.text(x, y, name.lower() if lowercase else name, fontsize=label_fontsize,
                 fontweight=label_fontweight, va='top', ha='left', transform=fig.transFigure)
    fig._label_positions = label_positions
    fig._subfigs = subfigs

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
    raise SystemExit('run ../../make_fig1_real.py')
