"""Figure 2B (delegated): Trajectory colored by calcium activity.

Single code path using StylePreset, following the fig1 delegation pattern.

Copy of figures_v10.4/work/fig2_src/fig2b_trajectory_delegated.py for v11.1: the colour bar can carry
ticks at calcium values (CBAR_TICKS).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from typing import Optional, Tuple

from driada.utils.publication import StylePreset

try:
    from .data_loader import compute_fig2b_data, load_fig2b_cache, save_fig2b_cache
except ImportError:
    from data_loader import compute_fig2b_data, load_fig2b_cache, save_fig2b_cache


# Colorbar label of the panel inside the composed figure ('calcium activity' in v7/v8).
CBAR_LABEL = 'calcium activity'
# CACHE_FILE: None reads cache/fig2b_data.npz of v7-v10.2, else the file of panel_bd_v104.py.
CACHE_FILE = None
# CBAR_TICKS: None keeps the automatic ticks, which give the colour value itself (log calcium rescaled
# to 0-1, linear steps); else (positions, labels): calcium values placed where their colour lies on
# that log scale (figures_v11.1/work/fig2/panel_b_ticks.py).
CBAR_TICKS = None


def _plot_multi_segment(x, y, values, cmap_name='jet', ax=None,
                        linewidths=4, alpha=0.7):
    """Plot trajectory as colored line segments using LineCollection."""
    x, y, values = np.array(x), np.array(y), np.array(values)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    colors = cmap(norm(values))
    lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=alpha)
    ax.add_collection(lc)
    return ax, lc, norm


def create_fig2b(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """Create Figure 2B: Trajectory colored by calcium activity."""
    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (25.4, 20.32)  # 10x8 inches in cm

    # Load data
    try:
        cache = load_fig2b_cache() if CACHE_FILE is None else dict(np.load(CACHE_FILE))
        x_ds, y_ds, neur_ds = cache['x_ds'], cache['y_ds'], cache['neur_ds']
        spike_x, spike_y = cache['spike_x'], cache['spike_y']
    except FileNotFoundError:
        x_ds, y_ds, neur_ds, spike_x, spike_y = compute_fig2b_data()

    # Figure/axes creation — only branch
    standalone = (fig is None)
    if standalone:
        fig, ax = plt.subplots(figsize=(panel_size[0] / 2.54,
                                         panel_size[1] / 2.54), dpi=300)
    else:
        ax = fig.subplots(1, 1)

    # Shared plotting logic
    line_width = style.get_line_width(panel_size, 'cm')
    marker_size = style.get_marker_size(panel_size, 'cm')

    ax, lc, norm = _plot_multi_segment(
        x_ds, y_ds, neur_ds, cmap_name='jet', ax=ax,
        linewidths=line_width, alpha=0.7)

    ax.scatter(spike_x, spike_y, c='violet', marker='*',
               s=marker_size * 30, alpha=1.0, edgecolors='brown',
               linewidths=line_width * 0.3, zorder=2, label='events')

    ax.set_xlim(x_ds.min() - 0.02, x_ds.max() + 0.02)
    ax.set_ylim(y_ds.min() - 0.02, y_ds.max() + 0.02)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_aspect('equal')
    ax.set_xlabel('animal position: x')
    ax.set_ylabel('animal position: y')
    style.apply_to_axes(ax, panel_size, 'cm')

    # Colorbar — both modes, sized appropriately
    sm = cm.ScalarMappable(cmap=cm.get_cmap('jet'), norm=norm)
    if standalone:
        cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('calcium activity (rescaled)', fontsize=20, labelpad=10)
        ax.legend(bbox_to_anchor=(1.15, -0.08), loc='upper center',
                  ncol=1, fontsize=style.base_legend_fontsize)
    else:
        # bottom slightly lower than A's 0.25 to compensate for colorbar
        # stealing width and aspect='equal' centering the plot upward.
        fig.subplots_adjust(left=0.08, right=0.82, bottom=0.18, top=0.99)
        cbar = fig.colorbar(sm, ax=ax, pad=0.05, fraction=0.05,
                            shrink=0.75, anchor=(0.0, 0.5))
        if CBAR_TICKS is not None:
            cbar.set_ticks(CBAR_TICKS[0], labels=CBAR_TICKS[1])
        cbar.ax.tick_params(labelsize=style.base_label_size * 0.7)
        cbar.set_label(CBAR_LABEL, fontsize=style.base_label_size * 0.7,
                        labelpad=2)
        global LAST_CBAR_LABEL
        LAST_CBAR_LABEL = cbar.ax.yaxis.label  # kept for the pixel check of make_fig2.py
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
                  ncol=1, frameon=False,
                  fontsize=style.base_legend_fontsize)

    return fig, ax
