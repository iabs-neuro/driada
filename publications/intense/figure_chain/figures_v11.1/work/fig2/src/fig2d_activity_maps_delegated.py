"""Figure 2D (delegated): Activity maps (calcium vs spikes).

Single code path using StylePreset, following the fig1 delegation pattern.

Copy of figures_v10.4/work/fig2_src/fig2d_activity_maps_delegated.py for v11.1 with one more switch,
NORMALIZE; at its default the panel is that of v11.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from driada.utils.publication import StylePreset

# CACHE_FILE: None reads cache/fig2d_data.npz of v7-v10.2 (maps smoothed with 0.5 bin), else the
# file of panel_bd_v104.py.
CACHE_FILE = None
# NORMALIZE: False draws both maps on one scale from 0 to the larger maximum (v7-v11); True draws each
# map normalized to [0, 1] as Methods describe it and as the similarity metrics of 2e use it.
NORMALIZE = False

try:
    from .data_loader import compute_fig2d_data, load_fig2d_cache, save_fig2d_cache
except ImportError:
    from data_loader import compute_fig2d_data, load_fig2d_cache, save_fig2d_cache


def create_fig2d(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[plt.Figure], list]:
    """Create Figure 2D: Activity maps (calcium vs spikes)."""
    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (20.32, 12.7)  # 8x5 inches in cm

    try:
        cache = load_fig2d_cache() if CACHE_FILE is None else dict(np.load(CACHE_FILE))
        ca_map, sp_map = cache['ca_map'], cache['sp_map']
    except FileNotFoundError:
        raise FileNotFoundError(
            "Fig2d cache not found. "
            "Run: python -m science.intense_figures.fig2.fig2d_activity_maps --cache")

    # Methods (comparison of spatial selectivity): both maps are normalized to [0, 1] before they are
    # compared; this is normalize_image of data_loader.calculate_act_maps_sim, map by map
    def _normalize(img):
        img = np.array(img, dtype=float)
        rng = np.max(img) - np.min(img)
        return np.zeros_like(img) if rng == 0 else (img - np.min(img)) / rng

    if NORMALIZE:
        ca_map, sp_map = _normalize(ca_map), _normalize(sp_map)
        vmin, vmax = 0, 1
    else:
        vmin = 0
        vmax = max(ca_map.max(), sp_map.max())

    # Figure/axes creation — only branch
    standalone = (fig is None)
    if standalone:
        fig, axs = plt.subplots(1, 2, figsize=(panel_size[0] / 2.54,
                                                 panel_size[1] / 2.54),
                                dpi=300, constrained_layout=True)
    else:
        axs = fig.subplots(1, 2)

    # Shared plotting logic
    titles = ['Calcium', 'Events']
    for i, (ax, data, title) in enumerate(zip(axs, [ca_map, sp_map], titles)):
        im = ax.imshow(data, cmap='jet', interpolation='nearest',
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=style.base_label_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_anchor('N')  # §1.4: anchor image to top, whitespace goes to bottom

    if not standalone:
        fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.12)

    cbar = fig.colorbar(im, ax=list(axs), orientation='horizontal',
                        fraction=0.06, pad=0.04, shrink=0.75,
                        label='Normalized activity')
    cbar.set_label('Normalized activity', fontsize=style.base_legend_fontsize)
    cbar.ax.tick_params(labelsize=style.base_legend_fontsize)

    return fig, list(axs)
