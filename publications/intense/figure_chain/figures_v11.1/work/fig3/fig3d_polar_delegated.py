"""Figure 3D: Polar plot — head direction tuning of a mixed-selective neuron.

Delegation-ready: works standalone or embedded in fig3_delegated.py.
Data: aligned NOF session H36 3D, neuron 158 (row of Calcium = neuron_idx of the DRIADA 1.2.2
database). After disentanglement the neuron is selective for head direction (excess MI 0.072
bits, p 1.5e-4, delay +25 frames) and position (0.192 bits, p 2.6e-6, delay -10 frames); the
same neuron is shown in panel E.
Uses StylePreset for consistent sizing across standalone and embedded modes.

Copy of figures_v10.4/work/fig3_src/fig3d_polar_delegated.py for Fig. 3 of v11.1. The polar axis
keeps the matplotlib default: 0 deg to the right, angles counter-clockwise. The direction arrays of
the data are measured from the x axis counter-clockwise, in the coordinates of the place map of
panel E, so the panel is drawn in the axes of that map. In Fig. 3 of v11.1 the example is a
body-direction neuron; make_fig3_v111.py hands the body direction to this module.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import Optional, Tuple

import _v122  # noqa: E402  DRIADA 1.2.2 snapshot first on sys.path
from driada.utils.publication import StylePreset


_MOUSE, _SESSION = 'H36', '3D'
_DEFAULT_ALIGNED_NPZ = _v122.aligned(_MOUSE, _SESSION)
_DEFAULT_NEURON_IDX = 158
_DEFAULT_OPTIMAL_DELAY = 25  # frames at 20 fps, opt_delay of the head-direction pair


def load_polar_data(aligned_npz=None, neuron_idx=None, optimal_delay=None):
    """Load head direction angles and calcium signal for one neuron.

    Parameters
    ----------
    aligned_npz : str or Path, optional
        Path to aligned experiment NPZ. Default: LNOF_J54_4D.
    neuron_idx : int, optional
        Neuron index in the calcium matrix. Default: 34.
    optimal_delay : int, optional
        Time shift in frames: calcium lags behind direction by this many frames.
        Default: 30 frames.

    Returns
    -------
    angles_rad : ndarray
        Head direction in radians.
    signals : ndarray
        MinMaxScaled, clipped-to-zero calcium signal (delay-aligned).
    """
    if aligned_npz is None:
        aligned_npz = _DEFAULT_ALIGNED_NPZ
    if neuron_idx is None:
        neuron_idx = _DEFAULT_NEURON_IDX
    if optimal_delay is None:
        optimal_delay = _DEFAULT_OPTIMAL_DELAY

    data = np.load(aligned_npz, allow_pickle=True)
    angles_rad = data['headdirection']
    calcium = data['Calcium'][neuron_idx]

    scaler = MinMaxScaler()
    signals = scaler.fit_transform(calcium.reshape(-1, 1)).ravel()
    signals = np.clip(signals, 0, None)

    # Apply optimal delay: calcium responds after the direction change
    if optimal_delay > 0:
        signals = signals[optimal_delay:]
        angles_rad = angles_rad[:-optimal_delay]
    elif optimal_delay < 0:
        signals = signals[:optimal_delay]
        angles_rad = angles_rad[-optimal_delay:]

    return angles_rad, signals


def create_fig3d(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    angles_rad=None,
    signals=None,
    bins=40,
    kde_bw=0.04,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create panel D: polar body direction tuning plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or SubFigure, optional
        If None, creates standalone figure. Otherwise embeds into given subfigure.
    style : StylePreset, optional
        Styling preset. Standalone default: from_make_beautiful() with overrides.
    panel_size : tuple of (width_cm, height_cm), optional
        Physical panel size in cm. Default: 20.32 x 20.32 (8x8 inches).
    angles_rad : ndarray, optional
        Body direction angles in radians. Loaded from default source if None.
    signals : ndarray, optional
        Calcium signal values. Loaded from default source if None.
    bins : int
        Number of angular bins for the histogram.
    kde_bw : float
        Bandwidth for KDE smoothing.
    """
    if style is None:
        style = StylePreset.from_make_beautiful().copy(
            base_tick_labelsize=16,
            base_label_size=18,
            base_legend_fontsize=14,
            base_line_width=2.0,
            base_title_size=20,
        )
    if panel_size is None:
        panel_size = (20.32, 20.32)  # 8x8 inches in cm

    # Load data if not provided
    if angles_rad is None or signals is None:
        angles_rad, signals = load_polar_data()

    standalone = (fig is None)

    if standalone:
        fig = plt.figure(
            figsize=(panel_size[0] / 2.54, panel_size[1] / 2.54), dpi=300)
        ax = fig.add_subplot(111, projection='polar')
    else:
        ax = fig.add_subplot(111, projection='polar')

    # Derive sizes from StylePreset
    tick_fs = style.base_tick_labelsize
    legend_fs = style.base_legend_fontsize
    kde_lw = style.get_line_width(panel_size, 'cm')

    # Occupancy-normalised tuning: mean signal in each direction bin, so that directions the
    # animal faced more often are not mistaken for preferred ones.
    ang = np.mod(angles_rad + np.pi, 2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    idx = np.clip(np.digitize(ang, edges) - 1, 0, bins - 1)
    occ = np.bincount(idx, minlength=bins)
    hist = np.bincount(idx, weights=signals, minlength=bins) / np.maximum(occ, 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(
        bin_centers, hist, width=2 * np.pi / bins,
        alpha=0.5, color='blue', edgecolor='blue',
        linewidth=0.3, label='mean signal per direction bin',
    )
    # Circular Gaussian smoothing of the binned tuning curve
    k = np.exp(-0.5 * (np.arange(-bins // 2, bins // 2) / (kde_bw * bins)) ** 2)
    k /= k.sum()
    smooth = np.real(np.fft.ifft(np.fft.fft(hist) * np.fft.fft(np.fft.ifftshift(k))))
    theta = np.append(bin_centers, bin_centers[0] + 2 * np.pi)
    ax.plot(
        theta, np.append(smooth, smooth[0]), '#2ca02c',
        linewidth=kde_lw, label='smoothed tuning curve',
    )

    # Polar axis configuration
    ax.grid(True, alpha=0.6, color='#aaaaaa', linewidth=style.base_spine_width * 0.6, zorder=2.5)
    ax.spines['polar'].set_linewidth(style.base_spine_width)
    ax.spines['polar'].set_color('#333333')
    ax.tick_params(axis='both', labelsize=legend_fs, pad=1)
    ax.set_yticklabels([])

    # Legend below the plot, matching reference layout
    legend = ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.22),
        fontsize=legend_fs, frameon=False, ncol=1,
        handlelength=2.0, handletextpad=0.6,
    )

    fig.patch.set_facecolor('white')

    if standalone:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.25)

    return fig, ax


if __name__ == '__main__':
    output_dir = Path(__file__).parent
    fig, ax = create_fig3d()
    out_path = output_dir / 'fig3d_polar_standalone.png'
    fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    print(f"Saved: {out_path}")
