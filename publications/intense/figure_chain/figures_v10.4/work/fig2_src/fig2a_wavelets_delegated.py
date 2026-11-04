"""Figure 2A (delegated): Wavelet scalogram and calcium signal.

Single code path using StylePreset, following the fig1 delegation pattern.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from driada.utils.publication import StylePreset
from science.intense_figures.fig1.plot_utils import plot_shadowed_groups

try:
    from .data_loader import compute_fig2a_data, load_fig2a_cache, save_fig2a_cache
except ImportError:
    from data_loader import compute_fig2a_data, load_fig2a_cache, save_fig2a_cache

FPS = 20
# DATA: None reads the cache of v7-v10.2 (events = periods above a threshold of the smoothed
# trace, fs = 20); otherwise dict(sig, sp, Wx, wvt_scales) computed by fig2a_data_v103.py.
DATA = None
# FREQ_TICKS: None keeps the v7 labels (five scale values put on the automatic ticks, so they do
# not mark the rows they sit on); otherwise frequencies in Hz, each on the row of the scalogram
# with that frequency. For the Morse wavelet with centered_scale=True the peak lies at 1/scale
# radians per sample, so a row of scale s has the frequency FPS / (2 pi s) (ssqueezepy
# scale_to_freq gives the same within its frequency grid).
FREQ_TICKS = None
# EVENT_TICKS: each event as a grey tick above the trace, at the middle of the event, drawn as the
# event ticks of Fig. 1a; False draws grey periods. EVENTS_LABEL: legend entry of the events.
EVENT_TICKS = False
EVENTS_LABEL = 'Reconstructed events'
# YLABEL_X: horizontal position of both y-axis titles, in axes fraction (-0.12 in v7). Further left
# keeps 'signal, a.u.' clear of its tick labels; the two titles stay aligned.
YLABEL_X = -0.12


def freq_rows(wvt_scales, freqs, fps):
    """Row of the scalogram (fractional) at which each frequency in Hz lies."""
    s = fps / (2 * np.pi * np.asarray(freqs, float))
    return np.interp(np.log(s), np.log(wvt_scales), np.arange(len(wvt_scales)))


def create_fig2a(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[plt.Figure], list]:
    """Create Figure 2A: Wavelet scalogram and calcium signal."""
    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (30.48, 35.56)  # 12x14 inches in cm

    # Load data
    if DATA is not None:
        sig, sp, Wx, wvt_scales = DATA['sig'], DATA['sp'], DATA['Wx'], DATA['wvt_scales']
    else:
      try:
        cache = load_fig2a_cache()
        sig, sp, Wx, wvt_scales = cache['sig'], cache['sp'], cache['Wx'], cache['wvt_scales']
      except FileNotFoundError:
        sig, sp, Wx, wvt_scales = compute_fig2a_data()

    # Figure/axes creation — only branch
    standalone = (fig is None)
    if standalone:
        fig, axs = plt.subplots(2, 1, figsize=(panel_size[0] / 2.54,
                                                 panel_size[1] / 2.54),
                                sharex=True, dpi=300)
    else:
        axs = fig.subplots(2, 1, sharex=True)
    ax1, ax2 = axs

    # Shared plotting logic
    line_width = style.get_line_width(panel_size, 'cm')
    n_frames = len(sig)
    duration_s = n_frames / FPS
    xvals = np.arange(n_frames)

    # Top: scalogram
    im = ax1.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    ax1.set_ylabel('frequency')
    if FREQ_TICKS is None:
        ax1.set_yticklabels(np.round(wvt_scales[::80][::-1], 1))
    else:
        ax1.set_yticks(freq_rows(wvt_scales, FREQ_TICKS, FPS))
        ax1.set_yticklabels([f"{f:g}" for f in FREQ_TICKS])
    style.apply_to_axes(ax1, panel_size, 'cm')
    ax1.yaxis.set_label_coords(YLABEL_X, 0.5)

    # Bottom: signal + events
    ax2.plot(xvals, sig, linewidth=max(line_width - 0.5, 0.5), c='b',
             label='Fluorescence signal')
    if EVENT_TICKS:
        b = (np.asarray(sp) == 1).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.r_[0, b, 0]) != 0)
        mid = (xvals[edges[0::2]] + xvals[edges[1::2] - 1] + 1) / 2
        # y in axes fraction. The gap between the axes is hspace = 0.12 of an axes height
        # (subplots_adjust below), 79 px at 600 dpi, and the time-axis ticks of the scalogram
        # reach 25 px into it from above; a tick of 5 pt (42 px) centred at 1.032 stands on the
        # top edge of the trace axes and stays 12 px clear of them.
        ax2.plot(mid, np.full(len(mid), 1.032), transform=ax2.get_xaxis_transform(),
                 linestyle='none', marker='|', color='0.45', markersize=5,
                 markeredgewidth=1.4, clip_on=False, label=EVENTS_LABEL)
    else:
        plot_shadowed_groups(ax2, xvals, sp, color='grey', alpha=0.5,
                             ymin=0.04, ymax=1.0,
                             label=EVENTS_LABEL)
    ax2.set_xlim(0, n_frames)
    step = 2000 if duration_s > 5000 else 100
    tick_seconds = np.arange(0, duration_s + 1, step).astype(int)
    ax2.set_xticks(tick_seconds * FPS)
    ax2.set_xticklabels(tick_seconds)
    ax2.set_ylabel('signal, a.u.')
    ax2.set_xlabel('time, s')
    style.apply_to_axes(ax2, panel_size, 'cm')
    ax2.yaxis.set_label_coords(YLABEL_X, 0.5)

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.40),
               ncol=2, frameon=style.legend_frameon,
               fontsize=style.base_legend_fontsize)

    # Re-apply after styling
    ax2.set_xlim(0, n_frames)
    ax2.set_xticks(tick_seconds * FPS)
    ax2.set_xticklabels(tick_seconds)

    # Embedded mode: fill the subfigure, reserve bottom for xlabel + legend
    if not standalone:
        fig.subplots_adjust(left=0.16, bottom=0.25, top=0.93, hspace=0.12)

    # Colorbar — standalone only (needs figure-level positioning)
    if standalone:
        ax1_pos = ax1.get_position()
        cbar_ax = fig.add_axes([ax1_pos.x0, ax1_pos.y0 - 0.02,
                                ax1_pos.width, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                            label='Magnitude')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label='Magnitude', fontsize=20)
        fig.subplots_adjust(hspace=0.35)

    return fig, [ax1, ax2]
