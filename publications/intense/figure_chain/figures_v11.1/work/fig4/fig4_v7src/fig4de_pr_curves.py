"""Figure 4D,E: Precision-recall curves for continuous and discrete benchmarks.

Panel D: Continuous features — 2 scatter plots (by SNR, by p_skip)
Panel E: Discrete features — 2 scatter plots (by SNR, by p_skip)

Data source: validation benchmark NPZ files.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List

from driada.utils.plot import make_beautiful
from driada.utils.publication import StylePreset


# =============================================================================
# Configuration
# =============================================================================

SKPROBS = np.linspace(0.0, 0.8, 9)
SNRS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

METHOD_NAMES = {
    'c': ['corr.', 'MI', 'corr. + shuffles', 'INTENSE'],
    'd': ['average', 'MI', 'average + shuffles', 'INTENSE'],
}

COLORS = ['c', 'b', 'g', 'r']
MARKERS = ['s', 'o', '^', '*']

# Which points of the other parameter each PR coordinate stands for. 'mean': averaged over it
# (the v7 drawing); 'slice': taken at p_skip = 0 (left) and at the largest SNR (right), as the
# caption of the figure states.
PR_SELECT = 'mean'


# =============================================================================
# PR curve panel
# =============================================================================

def create_pr_panel(
    data_npz,
    test_type,
    fig=None,
    panel_size: Optional[Tuple[float, float]] = None,
    snrs=None,
    skprobs=None,
    style: Optional[StylePreset] = None,
    font_scale: float = 1.0,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create 1x2 precision-recall scatter panel for one feature type.

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

    Returns
    -------
    fig, axs
    """
    standalone = (fig is None)

    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (11 * 2.54, 7 * 2.54)  # standalone: 11x7 inches in cm
    if snrs is None:
        snrs = SNRS
    if skprobs is None:
        skprobs = SKPROBS

    methods = METHOD_NAMES[test_type]

    sf = style.calculate_scale_factor(panel_size, 'cm')
    lw = style.get_line_width(panel_size, 'cm')
    label_fs = style.base_label_size * sf
    tick_fs = style.base_tick_labelsize * sf
    legend_fs = style.base_legend_fontsize * sf
    cbar_label_fs = style.base_legend_fontsize * sf
    cbar_tick_fs = style.base_legend_fontsize * sf

    # Marker sizes scale with panel width
    if standalone:
        star_size = 200
        other_size = 100
    else:
        star_size = 25
        other_size = 12

    # Average over realizations
    data = np.nanmean(data_npz, axis=-1)

    # Create figure
    if standalone:
        fig_size_inches = (panel_size[0] / 2.54, panel_size[1] / 2.54)
        fig, axs = plt.subplots(1, 2, figsize=fig_size_inches, dpi=300)
    else:
        axs = fig.subplots(1, 2)

    scatter = None
    for i, prop in enumerate(['SNR', r'$p_{skip}$']):
        ax = axs[i]

        if i == 0:  # SNR
            props = np.linspace(0.0, 1.0, len(snrs))
            cbarprop = snrs
        else:  # p_skip
            props = np.linspace(0.0, 1.0, len(skprobs))
            cbarprop = [f'{v:.1f}'.lstrip('0') for v in skprobs]

        for j, method in enumerate(methods):
            # Simple methods use stage1, shuffled methods use stage2
            si = 0 if j in [0, 1] else 1

            two_param_data = data[:, j, si]  # (2, n_skip, n_snr-1)

            # Average over the other parameter
            # i=0 (SNR): average over skip (axis=1) -> (2, n_snr-1)
            # i=1 (p_skip): average over SNR (axis=2) -> (2, n_skip)
            if PR_SELECT == 'slice':
                plot_data = two_param_data[:, 0, :] if i == 0 else two_param_data[:, :, -1]
            else:
                plot_data = np.mean(two_param_data, axis=i + 1)
            pr, rec = plot_data
            pr = np.nan_to_num(pr, nan=0.0)
            rec = np.nan_to_num(rec, nan=0.0)

            # Grey connecting line (skip corr+shuffles for continuous — noisy)
            if not (j == 2 and test_type == 'c'):
                ax.plot(pr, rec, c='grey', alpha=0.5, linewidth=lw,
                        linestyle='solid', zorder=-1)

            # Scatter plot
            marker_size = star_size if MARKERS[j] == '*' else other_size
            scatter = ax.scatter(
                pr, rec, s=marker_size,
                marker=MARKERS[j], c=props, cmap='jet',
                label=method if i == 0 else None,
                alpha=1.0,
            )

        # Apply styling via StylePreset
        make_beautiful(ax, style=style, panel_size=panel_size,
                       panel_units='cm', tight_layout=False)

        ax.set_aspect('equal', adjustable='box')
        if not standalone:
            ax.set_anchor('N')
        ax.set_xlim(-0.0, 1.02)
        ax.set_ylim(-0.0, 1.02)

        xtick_locations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ytick_locations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(xtick_locations)
        ax.set_yticks(ytick_locations)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)

        ax.set_xlabel('precision', fontsize=label_fs)
        if i == 0:
            ax.set_ylabel('recall', fontsize=label_fs)
        else:
            ax.set_ylabel('')

        # Reference line at precision=0.05
        ax.axvline(0.05, c='k', linestyle='solid', alpha=0.5, linewidth=lw * 0.7)

        # Colorbar below each subplot
        cbar_pad = 0.28 if not standalone else 0.25
        cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal',
                            pad=cbar_pad, ticks=props, fraction=0.046)
        cbar.set_label(prop, fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)
        cbar.ax.set_xticklabels(cbarprop, fontsize=cbar_tick_fs)

    # Layout first, then legend
    if standalone:
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, wspace=0.2)
    else:
        fig.subplots_adjust(top=0.90, bottom=0.28, left=0.12,
                            right=0.99, wspace=0.16)

    # Single legend above the plots
    handles, labels = axs[0].get_legend_handles_labels()
    legend_y = 0.99 if standalone else 1.0
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, legend_y),
               ncol=4, fontsize=legend_fs,
               frameon=False,
               columnspacing=0.5, handletextpad=0.3)

    return fig, list(axs)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent

    # Load old NPZ files
    c_data = np.load(data_dir / 'continuous INTENSE test mult v2.npz')['arr_0']
    d_data = np.load(data_dir / 'discrete INTENSE test mult v2.npz')['arr_0']

    # Panel D: Continuous
    fig_d, axs_d = create_pr_panel(c_data, 'c')
    fig_d.savefig(output_dir / 'fig4d_pr_continuous.png', dpi=300)
    fig_d.savefig(output_dir / 'fig4d_pr_continuous.pdf', dpi=300)
    plt.close(fig_d)
    print("Panel D (continuous PR) saved")

    # Panel E: Discrete
    fig_e, axs_e = create_pr_panel(d_data, 'd')
    fig_e.savefig(output_dir / 'fig4e_pr_discrete.png', dpi=300)
    fig_e.savefig(output_dir / 'fig4e_pr_discrete.pdf', dpi=300)
    plt.close(fig_e)
    print("Panel E (discrete PR) saved")
