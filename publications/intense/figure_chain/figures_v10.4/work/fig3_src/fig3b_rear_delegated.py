"""Figure 3B: Rear-selective neuron — calcium trace + density during rearing.

Delegation-ready: works standalone or embedded in fig3_delegated.py.
Data: aligned NOF session H31 2D, neuron 35 (row of Calcium = neuron_idx of the
DRIADA 1.2.2 database), rear: me_excess 0.064 bits, p 3.8e-11, signal ratio 6.8; only selectivity after disentanglement.
Styling matches fig1c via plot_neuron_feature_pair + StylePreset.from_make_beautiful().
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt

import _v122  # DRIADA 1.2.2 snapshot first on sys.path
from driada.intense.visual import plot_neuron_feature_pair
from driada.utils.publication import StylePreset

# Time axis: None draws it as the library does (frames / 20); a number is the frame rate of
# the session, drawn with nfp_fps.plot_neuron_feature_pair_fps. EXACT_SHADING: behaviour
# periods span exactly their frames (the library adds one unit of the axis, i.e. 1 s).
FPS = None
EXACT_SHADING = False


_MOUSE, _SESSION = 'H31', '2D'
_DEFAULT_NEURON_IDX = 35
_DEFAULT_FEATURE = 'rear'
_DEFAULT_OPTIMAL_DELAY = 30  # frames at 20 fps, opt_delay of the pair in the 1.2.2 database
_STATIC_FEATURES = {'fps': 20, 't_rise_sec': 0.05, 't_off_sec': 1.5}
_FORCE_CONTINUOUS = ['x', 'y', 'speed', 'bodydirection', 'headdirection']


def _load_experiment(data_npz=None):
    """Load the session as the analysis tool does."""
    return _v122.load_session(_MOUSE, _SESSION)


def create_fig3b(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Create panel B: rear-selective neuron time series + density.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or SubFigure, optional
        If None, creates standalone figure. Otherwise embeds into given subfigure.
    style : StylePreset, optional
        Styling preset. Default: StylePreset.from_make_beautiful().
    panel_size : tuple of (width, height), optional
        Panel size in cm for scaling. Default: (20, 6) matching fig1c.
    """
    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (20, 8)

    exp = _load_experiment()

    # Binarize the feature (NPZ stores smoothed continuous values)
    feat = getattr(exp, _DEFAULT_FEATURE)
    feat.data = (feat.data > 0).astype(float)
    feat.scdata = feat.data.copy()

    # Apply optimal delay: shift calcium LEFT so it aligns with the feature
    delay = _DEFAULT_OPTIMAL_DELAY
    if delay != 0:
        exp.neurons[_DEFAULT_NEURON_IDX].ca.scdata = np.roll(
            exp.neurons[_DEFAULT_NEURON_IDX].ca.scdata, -delay)

    n_frames = exp.calcium.data.shape[1]
    ind2 = n_frames - max(delay, 0)
    ds = max(1, ind2 // 12000)

    standalone = (fig is None)
    if standalone:
        fig = plt.figure(
            figsize=(panel_size[0], panel_size[1]), dpi=300)

    gs = fig.add_gridspec(1, 2, width_ratios=[0.75, 0.25], wspace=0.20)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    axs_tuple = (ax0, ax1)

    plot_fn, extra = plot_neuron_feature_pair, {}
    if FPS is not None:
        from nfp_fps import plot_neuron_feature_pair_fps as plot_fn
        extra = dict(fps=FPS, exact_shading=EXACT_SHADING)
    result_fig = plot_fn(
        exp,
        cell_id=_DEFAULT_NEURON_IDX,
        featname=_DEFAULT_FEATURE,
        ind1=0,
        ind2=ind2,
        ds=ds,
        add_density_plot=True,
        axs=axs_tuple,
        style=style,
        panel_size=panel_size,
        skip_tight_layout=True,
        bcolor='darkorange',
        neuron_label='neuron 35 fluorescence',
        feature_label='rearing',
        non_feature_label='no rearing',
        **extra,
    )

    result_fig = fig

    # Match fig1c post-processing
    axes = result_fig.get_axes()
    if len(axes) >= 2:
        axes[0].set_ylim(None, 1.01)
        axes[1].set_ylim(0, None)

    # Reposition legends
    legend_y_offset = -0.37 if panel_size[1] < 10 else -0.252
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_bbox_to_anchor((0.5, legend_y_offset))

    # Match fig1c axis labels
    if len(axes) >= 1:
        axes[0].set_ylabel('signal, a.u.')
    if len(axes) >= 2:
        axes[1].set_xlabel('log(signal)')

    bottom_margin = 0.42 if panel_size[1] < 10 else 0.35
    result_fig.subplots_adjust(left=0.08, bottom=bottom_margin, right=0.98, top=0.95)

    return result_fig


if __name__ == '__main__':
    output_dir = Path(__file__).parent
    fig = create_fig3b()
    out_path = output_dir / 'fig3b_rear_standalone.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_path}')
