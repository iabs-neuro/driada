"""Figure 3E: Spatial activity map of the mixed-selective neuron of panel D.

Delegation-ready: works standalone or embedded in fig3_delegated.py.
Data: aligned NOF session H36 3D, neuron 158 (position pair: excess MI 0.192 bits, p 2.6e-6,
delay -10 frames in the DRIADA 1.2.2 database).
Uses StylePreset for consistent sizing across standalone and embedded modes.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import Optional, Tuple

import _v122  # noqa: E402  DRIADA 1.2.2 snapshot first on sys.path
from driada.utils.publication import StylePreset


_MOUSE, _SESSION = 'H36', '3D'
_DEFAULT_ALIGNED_NPZ = _v122.aligned(_MOUSE, _SESSION)
_DEFAULT_NEURON_IDX = 158
_DEFAULT_OPTIMAL_DELAY = -10  # frames at 20 fps, opt_delay of the position pair


def _build_heatmap(x, y, intensity, grid_size=100, sigma=5):
    """Bin mean intensity onto a 2D grid and smooth with Gaussian kernel."""
    x_grid = np.linspace(x.min(), x.max(), grid_size)
    y_grid = np.linspace(y.min(), y.max(), grid_size)
    Z_sum = np.zeros((grid_size, grid_size))
    Z_count = np.zeros((grid_size, grid_size))
    for xi, yi, val in zip(x, y, intensity):
        x_idx = np.searchsorted(x_grid, xi) - 1
        y_idx = np.searchsorted(y_grid, yi) - 1
        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            Z_sum[y_idx, x_idx] += val
            Z_count[y_idx, x_idx] += 1
    # Mean per bin; unvisited bins get 0
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = np.where(Z_count > 0, Z_sum / Z_count, 0)
    X, Y = np.meshgrid(x_grid, y_grid)
    return X, Y, gaussian_filter(Z, sigma=sigma)


def load_place_data(aligned_npz=None, neuron_idx=None, optimal_delay=None):
    """Load x, y positions and calcium signal for one neuron.

    Parameters
    ----------
    aligned_npz : str or Path, optional
        Path to aligned experiment NPZ. Default: LNOF_J30_2D.
    neuron_idx : int, optional
        Neuron index in the calcium matrix. Default: 258.
    optimal_delay : int, optional
        Time shift in frames. Default: 20 frames.

    Returns
    -------
    x, y : ndarray
        Animal position coordinates.
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
    x_raw = data['x']
    y_raw = data['y']
    calcium = data['Calcium'][neuron_idx]

    # Normalize positions to [0, 1]
    x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    scaler = MinMaxScaler()
    signals = scaler.fit_transform(calcium.reshape(-1, 1)).ravel()
    signals = np.clip(signals, 0, None)

    if optimal_delay > 0:
        signals = signals[optimal_delay:]
        x = x[:-optimal_delay]
        y = y[:-optimal_delay]
    elif optimal_delay < 0:
        signals = signals[:optimal_delay]
        x = x[-optimal_delay:]
        y = y[-optimal_delay:]

    return x, y, signals


def create_fig3e(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    x=None,
    y=None,
    signals=None,
    grid_size=100,
    sigma=5,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create panel E: place cell spatial heatmap.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or SubFigure, optional
        If None, creates standalone figure. Otherwise embeds into given subfigure.
    style : StylePreset, optional
        Styling preset. Standalone default: from_make_beautiful() with overrides.
    panel_size : tuple of (width_cm, height_cm), optional
        Physical panel size in cm. Default: 25.40 x 20.32 (10x8 inches).
    x, y : ndarray, optional
        Animal position coordinates. Loaded from default source if None.
    signals : ndarray, optional
        Calcium signal values. Loaded from default source if None.
    grid_size : int
        Spatial grid resolution.
    sigma : float
        Gaussian smoothing kernel width in grid units.
    """
    if style is None:
        style = StylePreset.from_make_beautiful().copy(
            base_tick_labelsize=16,
            base_label_size=18,
            base_line_width=1.5,
        )
    if panel_size is None:
        panel_size = (25.40, 20.32)  # 10x8 inches in cm

    # Load data if not provided
    if x is None or y is None or signals is None:
        x, y, signals = load_place_data()

    standalone = (fig is None)

    if standalone:
        fig, ax = plt.subplots(
            figsize=(panel_size[0] / 2.54, panel_size[1] / 2.54), dpi=300)
    else:
        ax = fig.subplots()

    # Derive sizes from StylePreset
    tick_fs = style.base_tick_labelsize
    label_fs = style.base_label_size

    # Build and plot heatmap
    X, Y, Z = _build_heatmap(x, y, signals, grid_size=grid_size, sigma=sigma)

    cf = ax.contourf(X, Y, Z, 20, cmap='jet')

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, pad=0.08, format='%.2f')
    cbar.ax.tick_params(labelsize=label_fs)
    cbar.set_label('calcium activity', size=style.base_legend_fontsize, labelpad=style.base_tick_pad)

    # Axis labels and styling
    ax.set_xlabel('animal position: x', fontsize=label_fs, labelpad=2)
    ax.set_ylabel('animal position: y', fontsize=label_fs, labelpad=2)
    ax.tick_params(axis='both', labelsize=tick_fs,
                   length=style.base_tick_length, pad=style.base_tick_pad)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_aspect('equal')

    fig.patch.set_facecolor('white')

    if standalone:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.25)

    return fig, ax


if __name__ == '__main__':
    output_dir = Path(__file__).parent
    fig, ax = create_fig3e()
    out_path = output_dir / 'fig3e_place_standalone.png'
    fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    print(f"Saved: {out_path}")
