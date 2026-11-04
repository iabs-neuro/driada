"""Figure 4A: Synthetic data generation pipeline for continuous features.

Shows the 4-step pipeline:
  1. FBM time series with ROI selection
  2. Randomized binary time series (before/after skip filtering)
  3. Heterogeneous Poisson process (smoothed event rate)
  4. Pseudo calcium signal
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple, List

from driada.experiment.synthetic import (
    generate_fbm_time_series,
    select_signal_roi,
    delete_one_islands,
    apply_poisson_to_binary_series,
    generate_pseudo_calcium_signal,
)
from driada.intense.visual import plot_shadowed_groups
from driada.utils.plot import make_beautiful
from driada.utils.publication import StylePreset


# =============================================================================
# Parameters (matching old notebook)
# =============================================================================

HURST = 0.7
SEED = 13
RATE_0 = 0.1
RATE_1 = 1.0
DURATION = 600       # seconds
SAMPLING_RATE = 20   # Hz
MAX_GAP = 0.5        # seconds — gaps shorter than this are filled
LENGTH = int(DURATION * SAMPLING_RATE) - 3000

EVENT_RATE = 0.2
AMPLITUDE_RANGE = (0.5, 2)
DECAY_TIME = 2
NOISE_STD = 0.1
SKIP_PROB = 0.6

# Smoothing for Poisson rate visualization
SMOOTH_SIGMA_SEC = 3.0

# EXACT_SHADING: a shaded period spans exactly its samples, [t_first, t_last + 1 / SAMPLING_RATE].
# plot_shadowed_groups of DRIADA extends every period by one unit of the axis, here 1 s.
EXACT_SHADING = False


# =============================================================================
# Helper
# =============================================================================

def _shade(ax, xvals, binary, **kwargs):
    if EXACT_SHADING:
        from nfp_fps import shade_runs
        return shade_runs(ax, xvals, binary, 1, SAMPLING_RATE, **kwargs)
    return plot_shadowed_groups(ax, xvals, binary, **kwargs)


def fill_binary_gaps(binary_series, max_gap):
    """Fill short zero-gaps between islands of 1s.

    If consecutive 0s between two runs of 1s are shorter than max_gap samples,
    fill them with 1s (morphological closing).
    """
    result = binary_series.copy()
    n = len(result)
    i = 0
    while i < n:
        if result[i] == 0:
            gap_start = i
            while i < n and result[i] == 0:
                i += 1
            gap_len = i - gap_start
            # Fill if gap is short AND bounded by 1s on both sides
            if gap_len <= max_gap and gap_start > 0 and i < n:
                result[gap_start:i] = 1
        else:
            i += 1
    return result


# =============================================================================
# Panel creation
# =============================================================================

def create_fig4a(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    font_scale: float = 1.0,
) -> Tuple[Optional[plt.Figure], List[plt.Axes]]:
    """Create Figure 4A: synthetic data generation pipeline.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or SubFigure, optional
        If provided, create axes on this figure/subfigure.
        Otherwise create new standalone figure.
    style : StylePreset, optional
        If provided, use for consistent styling. If None, uses
        from_make_beautiful() defaults for standalone mode.
    panel_size : tuple of (width, height) in cm, optional
        Panel size for scaling. Default: standalone size.
    font_scale : float
        Deprecated — kept for backward compatibility.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes (4 axes)
    """
    standalone = (fig is None)

    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (12 * 2.54, 20 * 2.54)  # standalone: 12×20 inches in cm

    lw = style.get_line_width(panel_size, 'cm')
    sf = style.calculate_scale_factor(panel_size, 'cm')
    title_fs = style.base_label_size * sf  # title same size as labels
    label_fs = style.base_label_size * sf
    legend_fs = style.base_legend_fontsize * sf
    tick_fs = style.base_tick_labelsize * sf

    # -----------------------------------------------------------------
    # Generate data
    # -----------------------------------------------------------------
    fbm_series = generate_fbm_time_series(LENGTH, HURST, SEED)
    loc, lower_border, upper_border = select_signal_roi(fbm_series, seed=SEED)

    binary_series = np.zeros(LENGTH)
    binary_series[(fbm_series >= lower_border) & (fbm_series <= upper_border)] = 1
    binary_series = fill_binary_gaps(binary_series, MAX_GAP * SAMPLING_RATE)

    mod_binary_series = delete_one_islands(binary_series, SKIP_PROB, seed=SEED)

    poisson_series = apply_poisson_to_binary_series(
        mod_binary_series, RATE_0, RATE_1, seed=SEED,
    )

    pseudo_calcium = generate_pseudo_calcium_signal(
        duration=DURATION,
        events=poisson_series,
        sampling_rate=SAMPLING_RATE,
        event_rate=EVENT_RATE,
        amplitude_range=AMPLITUDE_RANGE,
        decay_time=DECAY_TIME,
        noise_std=NOISE_STD,
        seed=SEED,
    )

    xvals = np.arange(LENGTH) / SAMPLING_RATE

    # -----------------------------------------------------------------
    # Create figure
    # -----------------------------------------------------------------
    if standalone:
        fig_size_inches = (panel_size[0] / 2.54, panel_size[1] / 2.54)
        fig, axs = plt.subplots(
            4, 1,
            figsize=fig_size_inches,
            dpi=300,
            sharex=True,
        )
    else:
        axs = fig.subplots(4, 1, sharex=True)

    ax0, ax2, ax3, ax4 = axs

    title_pad = 4 * sf

    # -----------------------------------------------------------------
    # Panel 1: Continuous time series with ROI
    # -----------------------------------------------------------------
    offset = 2
    ax0.set_title('1. continuous time series', fontsize=title_fs, pad=title_pad)
    ax0.plot(xvals, fbm_series + offset, c='b', linewidth=lw,
             label='fBM time series')
    _shade(ax0, xvals, binary_series, color='grey', alpha=0.5,
           label='sel. region')
    ax0.fill_between(xvals, lower_border + offset, upper_border + offset,
                     color='g', alpha=0.5, label='range of interest')
    ax0.set_ylabel('signal, a.u.', fontsize=label_fs)

    # -----------------------------------------------------------------
    # Panel 2: Randomized binary time series
    # -----------------------------------------------------------------
    ax2.plot(xvals, binary_series, c='g', linewidth=lw,
             linestyle='dashed', alpha=1.0, label='before filtering')
    ax2.plot(xvals, mod_binary_series, c='b', linewidth=lw,
             label='after filtering')
    ax2.set_title('2. randomized binary time series', fontsize=title_fs, pad=title_pad)
    ax2.set_ylabel('state', fontsize=label_fs)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['off', 'on'])

    # -----------------------------------------------------------------
    # Panel 3: Heterogeneous Poisson process
    # -----------------------------------------------------------------
    sigma_samples = SMOOTH_SIGMA_SEC * SAMPLING_RATE
    smoothed_rates = gaussian_filter1d(poisson_series.astype(float),
                                       sigma=sigma_samples)
    ax3.plot(xvals, smoothed_rates, c='r', linewidth=lw,
             label='event rate (smoothed)')
    _shade(ax3, xvals, mod_binary_series, color='dimgray',
           alpha=0.8, label='rand. sel. region')
    ax3.set_title('3. inhomogeneous Poisson process', fontsize=title_fs, pad=title_pad)
    ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax3.set_ylabel('events', fontsize=label_fs)

    # -----------------------------------------------------------------
    # Panel 4: Pseudo calcium signal
    # -----------------------------------------------------------------
    _shade(ax4, xvals, mod_binary_series, color='dimgray',
           alpha=0.8, label='rand. sel. region')
    ax4.plot(xvals, pseudo_calcium, c='m', linewidth=lw,
             label='generated signal')
    ax4.set_title('4. pseudo calcium signal', fontsize=title_fs, pad=title_pad)
    ax4.set_xlabel('time, s', fontsize=label_fs)
    ax4.set_ylabel('signal, a.u.', fontsize=label_fs, labelpad=5 * sf)

    # -----------------------------------------------------------------
    # Apply styling to all axes (BEFORE legends, so apply_to_axes
    # doesn't re-create them with default positioning)
    # -----------------------------------------------------------------
    for ax in axs:
        make_beautiful(ax, style=style, panel_size=panel_size,
                       panel_units='cm', lowercase_labels=True)

    # -----------------------------------------------------------------
    # Legends — placed AFTER make_beautiful to preserve positioning
    # -----------------------------------------------------------------
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
               ncol=3, fontsize=legend_fs, frameon=False)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
               ncol=2, fontsize=legend_fs, frameon=False)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
               ncol=2, fontsize=legend_fs, frameon=False)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.36),
               ncol=2, fontsize=legend_fs, frameon=False)

    if standalone:
        fig.subplots_adjust(left=0.15, hspace=0.5)
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.10,
                            top=0.97, hspace=0.55)

    return fig, list(axs)


if __name__ == '__main__':
    from pathlib import Path

    output_dir = Path(__file__).parent
    fig, axs = create_fig4a()
    fig.savefig(output_dir / 'fig4a_generation_pipeline.pdf', dpi=300)
    fig.savefig(output_dir / 'fig4a_generation_pipeline.png', dpi=300)
    print(f"Figure 4A saved to {output_dir}")
    plt.show()
