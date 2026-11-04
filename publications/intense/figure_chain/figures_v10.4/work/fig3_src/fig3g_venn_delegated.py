"""Figure 3G: Venn diagram — MI decomposition (3-way: behavior 1, behavior 2, neural activity).

Delegation-ready: works standalone or embedded in fig3_delegated.py.
Uses StylePreset for consistent sizing across standalone and embedded modes.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Tuple

from driada.utils.publication import StylePreset


# Region colors (no blending — each region has a distinct color)
COLORS = {
    'X': "#FF6B6B",      # Light red
    'Y': "#4ECDC4",      # Teal
    'A': "#9B59B6",      # Purple
    'XY': "#FFD93D",     # Yellow (X∩Y)
    'XA': "#E67E22",     # Orange (X∩A)
    'YA': "#3498DB",     # Blue (Y∩A)
    'XYA': "#2CA02C",    # Green (X∩Y∩A)
}


def create_fig3g(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create panel G: Venn diagram of MI decomposition.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or SubFigure, optional
        If None, creates standalone figure. Otherwise embeds into given subfigure.
    style : StylePreset, optional
        Styling preset. Standalone default: from_make_beautiful() with overrides.
    panel_size : tuple of (width_cm, height_cm), optional
        Physical panel size in cm. Default: 25.40 x 20.32 (10x8 inches).
    """
    if style is None:
        style = StylePreset.from_make_beautiful().copy(
            base_label_size=22,           # set labels ("behavior 1 (X)" etc.)
            base_annotation_fontsize=18,  # region labels, annotation text
            base_legend_fontsize=16,      # legend items
            base_line_width=2.0,          # circle outlines, arrows
            base_spine_width=1.0,         # annotation box outline
        )
    if panel_size is None:
        panel_size = (25.40, 20.32)  # 10x8 inches in cm

    standalone = (fig is None)

    if standalone:
        fig, ax = plt.subplots(
            figsize=(panel_size[0] / 2.54, panel_size[1] / 2.54), dpi=300)
    else:
        ax = fig.subplots()

    # Derive all sizes from StylePreset
    set_label_fs = style.base_label_size
    region_fs = style.base_annotation_fontsize
    center_fs = style.base_annotation_fontsize
    legend_fs = style.base_legend_fontsize
    annot_fs = style.base_annotation_fontsize
    circle_lw = style.get_line_width(panel_size, 'cm')
    arrow_lw = style.get_line_width(panel_size, 'cm')
    annot_box_lw = style.base_spine_width

    # Create Venn diagram
    venn = venn3(
        subsets=(1, 1, 1, 1, 1, 1, 1),
        set_labels=(r'behavior 1 ($X$)', r'behavior 2 ($Y$)',
                    r'neural activity ($A$)'),
        ax=ax,
    )

    # --- Color and label each region ---

    # Single sets (clear labels)
    for region_id, color_key in [('100', 'X'), ('010', 'Y'), ('001', 'A')]:
        patch = venn.get_patch_by_id(region_id)
        if patch:
            venn.get_label_by_id(region_id).set_text('')
            patch.set_color(COLORS[color_key])
            patch.set_alpha(0.8)

    # Pairwise intersections
    pairwise = [
        ('110', 'XY', r'$I(X,Y)$', None),
        ('101', 'XA', r'$I(X,A)$', (-0.02, -0.02)),
        ('011', 'YA', r'$I(Y,A)$', (+0.02, -0.02)),
    ]
    for region_id, color_key, label_text, offset in pairwise:
        patch = venn.get_patch_by_id(region_id)
        if patch:
            lbl = venn.get_label_by_id(region_id)
            lbl.set_text(label_text)
            lbl.set_fontsize(region_fs)
            patch.set_color(COLORS[color_key])
            patch.set_alpha(0.8)
            if offset:
                x, y = lbl.get_position()
                lbl.set_position((x + offset[0], y + offset[1]))

    # Triple intersection
    patch_111 = venn.get_patch_by_id('111')
    if patch_111:
        lbl = venn.get_label_by_id('111')
        lbl.set_text(r'$I(X,Y,A)$')
        lbl.set_fontsize(center_fs)
        lbl.set_fontweight('bold')
        patch_111.set_color(COLORS['XYA'])
        patch_111.set_alpha(0.9)

    # Circle outlines
    circles = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), ax=ax,
                            linewidth=circle_lw, linestyle='-')
    for circle in circles:
        circle.set_edgecolor('black')
        circle.set_alpha(1.0)

    # Set label styling
    for text in venn.set_labels:
        text.set_fontsize(set_label_fs)

    # Region label styling
    for text in venn.subset_labels:
        if text and text.get_text():
            if text.get_fontweight() != 'bold':
                text.set_fontsize(region_fs)

    # Legend
    labels = [
        r'$I(X,Y)$ - behavior 1 & behavior 2 MI',
        r'$I(X,A)$ - behavior 1 & activity MI',
        r'$I(Y,A)$ - behavior 2 & activity MI',
        r'$I(X,Y,A)$ - interaction information',
    ]
    legend_elements = [
        mpatches.Patch(color=COLORS['XY'], alpha=0.8, label=labels[0]),
        mpatches.Patch(color=COLORS['XA'], alpha=0.8, label=labels[1]),
        mpatches.Patch(color=COLORS['YA'], alpha=0.8, label=labels[2]),
        mpatches.Patch(color=COLORS['XYA'], alpha=0.9, label=labels[3]),
    ]
    ax.legend(
        handles=legend_elements, loc='lower center',
        bbox_to_anchor=(0.5, -0.42),
        fontsize=legend_fs, frameon=False, fancybox=False, shadow=False,
    )

    # Annotation arrow
    ax.annotate(
        'synergistic/redundant\ninformation',
        xy=(0, -0.05), xytext=(0.78, -0.60),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.4',
                        color='black', lw=arrow_lw),
        fontsize=annot_fs, ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                  edgecolor='black', linewidth=annot_box_lw),
    )

    ax.grid(False)
    ax.set_aspect('equal')
    fig.patch.set_facecolor('white')

    if standalone:
        fig.tight_layout(rect=[0, 0.10, 1, 1])
    else:
        fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.20)

    return fig, ax


if __name__ == '__main__':
    output_dir = Path(__file__).parent
    fig, ax = create_fig3g()
    out_path = output_dir / 'fig3g_venn_standalone.png'
    fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    print(f"Saved: {out_path}")
