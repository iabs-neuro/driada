"""Figure 2C (delegated): Venn diagrams of INTENSE vs classic place cell detection.

Single code path using StylePreset, following the fig1 delegation pattern.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from typing import Optional, Tuple
from scipy.optimize import brentq

from driada.utils.publication import StylePreset

try:
    from .data_loader import compute_fig2c_data, load_fig2c_cache, save_fig2c_cache
except ImportError:
    from data_loader import compute_fig2c_data, load_fig2c_cache, save_fig2c_cache

# Colors matching reference notebook
IN1 = np.array([255, 128, 128]) / 255.0
OUT1 = np.array([255, 0, 0]) / 255.0
IN2 = np.array([85, 160, 251]) / 255.0
OUT2 = np.array([0, 0, 192]) / 255.0
IN_MID = np.array([184, 86, 215]) / 255.0
OUT_MID = np.array([118, 6, 154]) / 255.0


def _intersection_area(d, R, r):
    if d <= abs(R - r):
        return np.pi * min(R, r) ** 2
    if d >= r + R:
        return 0
    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2 * d * r))
    beta = np.arccos((d2 + R2 - r2) / (2 * d * R))
    return r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2 * alpha)
                                             + R2 * np.sin(2 * beta))


def _find_d(A, R, r):
    if A >= np.pi * min(r, R) ** 2:
        return abs(R - r) * 0.99
    if A <= 0:
        return R + r
    return brentq(lambda d: _intersection_area(d, R, r) - A,
                  abs(R - r), R + r)


def _intersection_boundary(x1, r1, x2, r2, n=200):
    d = x2 - x1
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - a**2)
    ix = x1 + a

    angle1_bot = np.arctan2(-h, ix - x1)
    angle1_top = np.arctan2(h, ix - x1)
    theta1 = np.linspace(angle1_bot, angle1_top, n)
    arc1 = np.column_stack([x1 + r1 * np.cos(theta1), r1 * np.sin(theta1)])

    angle2_top = np.arctan2(h, ix - x2)
    angle2_bot = np.arctan2(-h, ix - x2)
    theta2 = np.linspace(angle2_top, angle2_bot + 2 * np.pi, n)
    arc2 = np.column_stack([x2 + r2 * np.cos(theta2), r2 * np.sin(theta2)])

    return np.vstack([arc1, arc2])


def _draw_venn(ax, intense_only, classic_only, both, label,
               fontsize=24, annotation_fontsize=24, edge_lw=5):
    """Draw a single Venn diagram on the given axes."""
    a1 = intense_only + both
    a2 = classic_only + both
    scale = 55
    r1 = np.sqrt(a1) / scale
    r2 = np.sqrt(a2) / scale

    target_overlap = both / a1 * (np.pi * r1**2)
    d = _find_d(target_overlap, r1, r2)
    total_width = r1 + d + r2
    x1 = -total_width / 2 + r1
    x2 = total_width / 2 - r2

    ax.add_patch(plt.Circle((x1, 0), r1, fc=IN1, ec=OUT1, lw=edge_lw, zorder=1))
    ax.add_patch(plt.Circle((x2, 0), r2, fc=IN2, ec=OUT2, lw=edge_lw, zorder=2))

    intersection_patch = plt.Circle((x1, 0), r1, fc=IN_MID, ec='none', zorder=3)
    theta = np.linspace(0, 2 * np.pi, 200)
    clip_verts = np.column_stack([x2 + r2 * np.cos(theta), r2 * np.sin(theta)])
    clip_patch = mpatches.PathPatch(MplPath(clip_verts), transform=ax.transData)
    ax.add_patch(intersection_patch)
    intersection_patch.set_clip_path(clip_patch)

    ax.add_patch(plt.Circle((x1, 0), r1, fc='none', ec=OUT1, lw=edge_lw, zorder=4))
    ax.add_patch(plt.Circle((x2, 0), r2, fc='none', ec=OUT2, lw=edge_lw, zorder=4))

    if both > 0 and d < r1 + r2:
        lens_verts = _intersection_boundary(x1, r1, x2, r2)
        lens_path = MplPath(np.vstack([lens_verts, lens_verts[0]]))
        ax.add_patch(mpatches.PathPatch(
            lens_path, fc='none', ec=OUT_MID, lw=edge_lw,
            zorder=5, transform=ax.transData))

    x_ovl_left = x2 - r2
    x_ovl_right = x1 + r1
    ax.text((x1 - r1 + x_ovl_left) / 2, 0, str(intense_only),
            fontsize=fontsize, ha='center', va='center', fontweight='bold', zorder=6)
    ax.text((x_ovl_right + x2 + r2) / 2, 0, str(classic_only),
            fontsize=fontsize, ha='center', va='center', fontweight='bold', zorder=6)
    ax.text((x_ovl_left + x_ovl_right) / 2, 0, str(both),
            fontsize=fontsize, ha='center', va='center', fontweight='bold', zorder=6)

    max_r = max(r1, r2)
    ax.text(0, -max_r - 0.12, label, fontsize=annotation_fontsize,
            ha='center', va='top', zorder=6)

    margin = 0.12
    ax.set_xlim(-total_width / 2 - margin, total_width / 2 + margin)
    ax.set_ylim(-max_r - 0.22, max_r + margin)
    ax.set_aspect('equal')
    ax.axis('off')


def create_fig2c(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[plt.Figure], list]:
    """Create Figure 2C: Venn diagrams of INTENSE vs classic PCs."""
    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (40.64, 17.78)  # 16x7 inches in cm

    try:
        data = load_fig2c_cache()
    except FileNotFoundError:
        data = compute_fig2c_data()
        save_fig2c_cache(data)

    # Figure/axes creation — only branch
    standalone = (fig is None)
    if standalone:
        fig, axs = plt.subplots(1, 2, figsize=(panel_size[0] / 2.54,
                                                 panel_size[1] / 2.54), dpi=300)
    else:
        axs = fig.subplots(1, 2)

    # Shared plotting logic
    annotation_fontsize = style.get_annotation_fontsize(panel_size, 'cm')
    label_fontsize = style.base_label_size
    line_width = style.get_line_width(panel_size, 'cm')

    number_fontsize = style.base_legend_fontsize  # match legend size
    _draw_venn(axs[0], data['p05_intense_only'], data['p05_classic_only'],
               data['p05_both'], r'$p < 0.05$',
               fontsize=number_fontsize, annotation_fontsize=label_fontsize,
               edge_lw=line_width)
    _draw_venn(axs[1], data['any_pc_intense_only'], data['any_pc_classic_only'],
               data['any_pc_both'], 'any PC',
               fontsize=number_fontsize, annotation_fontsize=label_fontsize,
               edge_lw=line_width)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=IN1, edgecolor=OUT1, linewidth=line_width, label='INTENSE'),
        Patch(facecolor=IN2, edgecolor=OUT2, linewidth=line_width, label='Classic'),
        Patch(facecolor=IN_MID, edgecolor=OUT_MID, linewidth=line_width, label='Intersection'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=style.base_legend_fontsize, frameon=False,
               bbox_to_anchor=(0.5, 0.01))

    if standalone:
        fig.tight_layout(rect=[0, 0.10, 1, 1])
    else:
        # Add left/right whitespace to make circles visually smaller
        fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.15,
                            wspace=0.05)

    return fig, list(axs)
