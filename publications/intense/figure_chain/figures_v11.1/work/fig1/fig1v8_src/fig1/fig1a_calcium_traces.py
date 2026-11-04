"""Figure 1A: Calcium imaging traces with event detection.

Shows calcium activity traces for 3 neurons with shadowed event regions.
Size: 17.78 x 24.6 cm
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from driada.utils.plot import make_beautiful
from driada.utils.publication import StylePreset

# Handle both module and standalone imports
try:
    from .data_loader import load_nof_experiment, load_fig1a_cache, cache_exists
    from .plot_utils import plot_shadowed_groups
except ImportError:
    from data_loader import load_nof_experiment, load_fig1a_cache, cache_exists
    from plot_utils import plot_shadowed_groups


DEFAULT_LABELS = {
    "ylabel": "signal, a.u.",
    "neuron": "neuron {idx}",
    "xlabel": "time, s",
    "neural_activity": "neural activity",
    "events": "reconstructed events",
}

# Number printed for each plotted neuron. The cache keys (36-38) are rows of the older export,
# whose calcium matrix has one extra leading row; set {36: 35, 37: 36, 38: 37} to print the rows
# of the synchronized files (empty: print the cache keys, as in v7/v8).
NEURON_LABEL_IDX = {}

# FPS: frame rate of the session for the time axis (20 in v7-v10.2). EVENTS: None shades the
# periods above event_threshold (v7-v10.2); otherwise {cache key: binary event train of the
# plotted frames} from the wavelet detector. EXACT_SHADING: a period spans exactly its frames
# (plot_shadowed_groups adds one unit of the axis, i.e. 1 s, to each).
FPS = 20.0
EVENTS = None
EXACT_SHADING = False
# EVENT_TICKS: instead of grey periods, mark each event by a grey tick above the trace, at the
# middle of the event (an event of 0.5 s is narrower than the line of a large transient).
EVENT_TICKS = False
# YLIM_TOP: upper limit of the three trace axes; at 1.2 the event ticks (drawn at 1.12) stay inside
# the axes, clear of the trace above.
YLIM_TOP = 1.01

# Russian labels for the localized version of the figure.
RU_LABELS = {
    "ylabel": "сигнал, отн. ед.",
    "neuron": "нейрон {idx}",
    "xlabel": "время, с",
    "neural_activity": "нейронная активность",
    "events": "события",
}


def create_fig1a(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    labels: Optional[dict] = None,
    neuron_fontscale: float = 1.0,
) -> Tuple[Optional[plt.Figure], List[plt.Axes]]:
    """Create Figure 1A: Calcium traces with events.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or SubFigure, optional
        If provided, create axes on this figure/subfigure.
        Otherwise create new standalone figure.
    style : StylePreset, optional
        Style configuration from publication framework.
        If None, uses StylePreset.from_make_beautiful() for standalone figures.
    panel_size : tuple of (width, height), optional
        Panel size in cm for scaling. If None, uses default standalone size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axs : list of matplotlib.axes.Axes
        List of 3 axes objects
    """
    # Default to standalone preset if not provided
    if style is None:
        style = StylePreset.from_make_beautiful()

    if labels is None:
        labels = DEFAULT_LABELS

    # Default panel size for standalone
    if panel_size is None:
        panel_size = (17.78, 24.6)

    # Parameters
    ds = 1  # Downsampling factor
    neuron_indices = [36, 37, 38]  # Neurons to plot
    event_threshold = 0.25  # Threshold for event detection

    # Load data from cache (fast) or full experiment (slow)
    try:
        cache = load_fig1a_cache()
        ca_data = {idx: cache[f'neuron_{idx}_ca'] for idx in neuron_indices}
        use_cache = True
    except FileNotFoundError:
        Exp = load_nof_experiment(animal_id='H01', day='1D', verbose=False)
        ca_data = {idx: Exp.neurons[idx].ca.scdata[:5000] for idx in neuron_indices}
        use_cache = False

    # Create figure if not provided
    standalone = (fig is None)
    if standalone:
        fig, axs = plt.subplots(3, 1, figsize=(panel_size[0]/2.54, panel_size[1]/2.54),
                               dpi=300, sharex=True)
    else:
        # Create axes on provided figure/subfigure
        axs = fig.subplots(3, 1, sharex=True)

    # Get scaled style values
    line_width = style.get_line_width(panel_size, 'cm')
    annotation_fontsize = style.get_annotation_fontsize(panel_size, 'cm')

    # Plot each neuron
    for i, ax_i in enumerate(axs):
        # Get calcium data from cache or dict
        cell_ind = neuron_indices[i]
        ca = ca_data[cell_ind]

        # Convert frame indices to time in seconds (assuming 20 fps)
        time_vals = np.arange(len(ca))[::ds] / FPS

        # Plot calcium trace - only add legend labels on LAST subplot
        ax_i.plot(time_vals, ca[::ds],
                  c='b', linewidth=line_width,
                  label=labels['neural_activity'] if i == 2 else None)

        # Detect and shadow events
        if EVENTS is None:
            pseudo_events = np.zeros(len(ca))
            pseudo_events[np.where(ca > event_threshold)[0]] = 1
        else:
            pseudo_events = np.asarray(EVENTS[cell_ind][:len(ca)], dtype=float)
        if EVENT_TICKS:
            b = (np.asarray(pseudo_events[::ds]) == 1).astype(np.int8)
            edges = np.flatnonzero(np.diff(np.r_[0, b, 0]) != 0)
            mid = (time_vals[edges[0::2]] + time_vals[edges[1::2] - 1] + ds / FPS) / 2
            ax_i.plot(mid, np.full(len(mid), 1.12), linestyle='none', marker='|', color='0.45',
                      markersize=5, markeredgewidth=1.4, clip_on=False,
                      label=labels['events'] if i == 2 else None)
        elif EXACT_SHADING:
            from nfp_fps import shade_runs
            shade_runs(ax_i, time_vals, pseudo_events[::ds], ds, FPS, alpha=0.5, color='grey',
                       label=labels['events'] if i == 2 else None)
        else:
            plot_shadowed_groups(ax_i, time_vals, pseudo_events[::ds],
                                 alpha=0.5, color='grey',
                                 label=labels['events'] if i == 2 else None)

        # Y-axis label only on MIDDLE subplot (shared label)
        if i == 1:
            ax_i.set_ylabel(labels['ylabel'])

        # Add neuron label inside the frame (upper right)
        ax_i.text(0.98, 0.95, labels['neuron'].format(idx=NEURON_LABEL_IDX.get(cell_ind, cell_ind)),
                  transform=ax_i.transAxes,
                  fontsize=annotation_fontsize * neuron_fontscale,
                  va='top', ha='right')

        # X-label only on bottom subplot
        if i == 2:
            ax_i.set_xlabel(labels['xlabel'])

        # Apply framework styling to axis
        style.apply_to_axes(ax_i, panel_size, 'cm')

        # Handle legend on bottom subplot - centered below xlabel (20% reduced spacing)
        if i == 2:
            ax_i.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.583),  # Moved up 20% total
                ncol=2,
                frameon=style.legend_frameon,
                fontsize=style.base_legend_fontsize,
            )

        # Set y-ticks and upper limit slightly above 1 to avoid cropping (1.01 invisible)
        ax_i.set_yticks([0.0, 0.5, 1.0])
        ax_i.set_ylim(None, YLIM_TOP)

    # Increased hspace between subplots like in reference, reserve bottom for xlabel + legend
    # top=0.98 leaves room for the "1.0" y-tick label at the top
    fig.subplots_adjust(left=0.12, bottom=0.26, right=0.95, top=0.98, hspace=0.2)

    return fig, list(axs)


if __name__ == '__main__':
    """Run standalone to test."""
    from pathlib import Path

    # Output to same directory as script
    output_dir = Path(__file__).parent

    # Uses StylePreset.from_make_beautiful() by default (standalone size)
    fig, axs = create_fig1a()
    fig.savefig(output_dir / 'fig1a_calcium_traces.pdf', dpi=300)
    fig.savefig(output_dir / 'fig1a_calcium_traces.png', dpi=300)
    print(f"Figure 1A saved to {output_dir}")

    # Russian-labelled version (does not affect the English paper composite,
    # which calls create_fig1a() with the default English labels).
    fig_ru, _ = create_fig1a(labels=RU_LABELS, neuron_fontscale=0.7)
    fig_ru.savefig(output_dir / 'fig1a_calcium_traces_ru.pdf', dpi=300,
                   bbox_inches='tight')
    fig_ru.savefig(output_dir / 'fig1a_calcium_traces_ru.png', dpi=300,
                   bbox_inches='tight')
    print(f"Figure 1A (RU) saved to {output_dir}")
