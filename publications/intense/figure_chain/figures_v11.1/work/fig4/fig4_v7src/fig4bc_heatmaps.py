"""Figure 4B,C: F1 score heatmaps for continuous and discrete benchmarks.

Panel B: Continuous features — 2x2 grid (corr., MI, corr.+shuffles, INTENSE)
Panel C: Discrete features — 2x2 grid (average, MI, avg+shuffles, INTENSE)

Data source: validation benchmark NPZ files.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List

from driada.utils.publication import StylePreset


# =============================================================================
# Configuration
# =============================================================================

SKPROBS = np.linspace(0.0, 0.8, 9)
SNRS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
METRICS = ['precision', 'recall']
STAGES = ['stage1', 'stage2']

METHOD_NAMES = {
    'c': ['corr.', 'MI', 'corr. + shuffles', 'INTENSE'],
    'd': ['average', 'MI', 'average + shuffles', 'INTENSE'],
}

# Which stage to show per method index
# Dummies (0,1) only have stage1; INTENSE (2,3) use stage1 for comparable view
STAGE_IDX = 0  # stage1 for all methods


# =============================================================================
# Heatmap panel
# =============================================================================

def create_heatmap_panel(
    data_npz,
    test_type,
    fig=None,
    panel_size: Optional[Tuple[float, float]] = None,
    snrs=None,
    skprobs=None,
    style: Optional[StylePreset] = None,
    font_scale: float = 1.0,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create 2x2 F1 heatmap grid for one feature type.

    Parameters
    ----------
    data_npz : ndarray of shape (2, 4, 2, n_skip, n_snr, n_rand)
        Benchmark results array. Dim 0: [precision, recall].
    test_type : 'c' or 'd'
        Continuous or discrete features.
    fig : Figure, optional
        If provided, draw on this figure/subfigure.
    panel_size : tuple (w, h) in cm, optional
    snrs : list of float, optional
    skprobs : array, optional
    style : StylePreset, optional
    font_scale : float
        Deprecated — kept for backward compatibility.
    vmax : float, optional
        Max value for colormap. Default: 0.6 for continuous, 1.0 for discrete.
    show_colorbar : bool
        Whether to show colorbar below heatmaps. Default True.

    Returns
    -------
    fig, axs
    """
    standalone = (fig is None)

    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (10 * 2.54, 8 * 2.54)  # standalone: 10x8 inches in cm
    if snrs is None:
        snrs = SNRS
    if skprobs is None:
        skprobs = SKPROBS
    if vmax is None:
        vmax = 0.6 if test_type == 'c' else 1.0

    methods = METHOD_NAMES[test_type]

    sf = style.calculate_scale_factor(panel_size, 'cm')
    title_fs = style.base_label_size * sf  # title same size as labels
    label_fs = style.base_label_size * sf
    tick_fs = style.base_tick_labelsize * sf
    legend_fs = style.base_legend_fontsize * sf
    cbar_label_fs = style.base_legend_fontsize * sf
    cbar_tick_fs = style.base_legend_fontsize * sf

    # Average over realizations
    data = np.nanmean(data_npz, axis=-1)

    # F1 = 2 * precision * recall / (precision + recall)
    prec = data[0]  # (4, 2, n_skip, n_snr)
    rec = data[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * prec * rec / (prec + rec)
    f1 = np.nan_to_num(f1, nan=0.0)

    # Create figure
    if standalone:
        fig_size_inches = (panel_size[0] / 2.54, panel_size[1] / 2.54)
        fig, axs = plt.subplots(2, 2, figsize=fig_size_inches, dpi=300)
    else:
        axs = fig.subplots(2, 2)

    im = None
    for j, method in enumerate(methods):
        row, col = j // 2, j % 2
        ax = axs[row, col]
        plot_data = f1[j, STAGE_IDX].T  # (n_snr, n_skip)

        im = ax.imshow(plot_data, cmap='viridis', vmin=0, vmax=vmax,
                       aspect='auto', origin='lower')

        ax.set_xticks(np.arange(len(skprobs)))
        ax.set_yticks(np.arange(len(snrs)))

        ax.set_xticklabels([f'{sp:.1f}'.lstrip('0') for sp in skprobs], fontsize=tick_fs)
        ax.set_yticklabels([int(s) for s in snrs], fontsize=tick_fs)

        # Only show axis labels on bottom row (x) and left column (y)
        if row == 1:
            ax.set_xlabel(r'$p_{skip}$', fontsize=label_fs)
        else:
            ax.set_xlabel('')

        if col == 0:
            ax.set_ylabel('SNR', fontsize=label_fs)
        else:
            ax.set_ylabel('')

        # Apply spine/tick styling from StylePreset
        style.apply_to_axes(ax, panel_size, 'cm')

        # Set title AFTER apply_to_axes to preserve original case (INTENSE, MI)
        ax.set_title(method, fontsize=title_fs)

    if show_colorbar:
        fig.subplots_adjust(bottom=0.203, top=0.970, left=0.1, right=0.97,
                            hspace=0.7, wspace=0.3)
        cbar_ax = fig.add_axes([0.1, 0.03, 0.87, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=cbar_tick_fs)
        cbar.set_label(r'$F_1$ score', fontsize=cbar_label_fs)
    else:
        fig.subplots_adjust(bottom=0.149, top=0.916, left=0.1, right=0.97,
                            hspace=0.7, wspace=0.3)

    return fig, list(np.ravel(axs)), im


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent

    # Load old NPZ files
    c_data = np.load(data_dir / 'continuous INTENSE test mult v2.npz')['arr_0']
    d_data = np.load(data_dir / 'discrete INTENSE test mult v2.npz')['arr_0']

    # Panel B: Continuous
    fig_b, axs_b, _ = create_heatmap_panel(c_data, 'c')
    fig_b.savefig(output_dir / 'fig4b_heatmaps_continuous.png', dpi=300)
    fig_b.savefig(output_dir / 'fig4b_heatmaps_continuous.pdf', dpi=300)
    plt.close(fig_b)
    print("Panel B (continuous) saved")

    # Panel C: Discrete
    fig_c, axs_c, _ = create_heatmap_panel(d_data, 'd')
    fig_c.savefig(output_dir / 'fig4c_heatmaps_discrete.png', dpi=300)
    fig_c.savefig(output_dir / 'fig4c_heatmaps_discrete.pdf', dpi=300)
    plt.close(fig_c)
    print("Panel C (discrete) saved")
