"""Figure 1C: Neuron-feature pair visualization with density plot.

Shows calcium activity of a neuron alongside a discrete behavioral feature,
with a density comparison plot showing signal distributions during feature presence/absence.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from pathlib import Path
from typing import Optional, Tuple

from driada.utils.publication import StylePreset
from driada.intense.visual import plot_neuron_feature_pair

# Handle both module and standalone imports
try:
    from .data_loader import load_nof_experiment
except ImportError:
    from data_loader import load_nof_experiment


# cell_id 102 is a row of the older export (one extra leading row); the same neuron is row 101
# of the synchronized file. The printed label is set here ('neuron 102 fluorescence' in v7/v8).
NEURON_LABEL = 'neuron 102 fluorescence'

# FPS: None draws the time axis as the library does (frames / 20); a number is the frame rate
# of the session, drawn with nfp_fps.plot_neuron_feature_pair_fps. EXACT_SHADING: interaction
# periods span exactly their frames (the library adds 1 s to each).
FPS = None
EXACT_SHADING = False


def create_fig1c(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
):
    """Create Figure 1C: Neuron-feature pair with density plot.

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
    """
    # Default to standalone preset if not provided
    if style is None:
        style = StylePreset.from_make_beautiful()

    # Default panel size for standalone
    if panel_size is None:
        panel_size = (20, 6)  # Default figsize for plot_neuron_feature_pair

    # Load experiment - H01 2D has great object4 selectivity
    Exp = load_nof_experiment(animal_id='H01', day='2D', verbose=False)

    # If figure provided, create our own axes on it with proper layout
    standalone = (fig is None)
    if not standalone:
        # Create axes with proper width_ratios for time series (75%) and density (25%)
        # Matches reference where density is narrower, reduced spacing
        gs = fig.add_gridspec(1, 2, width_ratios=[0.75, 0.25], wspace=0.20)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        axs_tuple = (ax0, ax1)
    else:
        axs_tuple = None

    # Create the neuron-feature pair plot using driada's function
    plot_fn, extra = plot_neuron_feature_pair, {}
    if FPS is not None:
        from nfp_fps import plot_neuron_feature_pair_fps as plot_fn
        extra = dict(fps=FPS, exact_shading=EXACT_SHADING)
    result_fig = plot_fn(
        Exp,
        cell_id=102,
        featname='object4',
        ind1=0,
        ind2=6000,
        ds=1,
        add_density_plot=True,
        axs=axs_tuple,
        style=style,
        panel_size=panel_size,
        skip_tight_layout=True,  # We handle layout ourselves
        bcolor='g',
        neuron_label=NEURON_LABEL,
        feature_label='interaction with object 4',
        non_feature_label='no interaction with object 4',
        **extra,
    )

    # Use provided fig or the one created by plot_neuron_feature_pair
    if not standalone:
        result_fig = fig

    # Adjust axes
    axes = result_fig.get_axes()
    if len(axes) >= 2:
        # Fix cropping issue - set upper limit slightly above 1 (1.01 invisible)
        time_series_ax = axes[0]
        time_series_ax.set_ylim(None, 1.01)

        # Density plot y-axis - truncate to 0.85 to avoid cropping at top
        density_ax = axes[1]
        density_ax.set_ylim(None, 0.85)

    # Reposition legends (reduced spacing)
    legend_y_offset = -0.315 if panel_size[1] < 10 else -0.252  # Moved up 10%
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_bbox_to_anchor((0.5, legend_y_offset))

    # Shorten legend labels on both subplots
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                label = text.get_text()
                text.set_text(label.replace('interaction', 'inter.'))

    # Change y-label to match Panel A
    if len(axes) >= 1:
        axes[0].set_ylabel('signal, a.u.')

    # Adjust bottom margin based on panel height (smaller panels need larger relative margin)
    # left=0.12 to align y-axis with Panels A and B
    bottom_margin = 0.55 if panel_size[1] < 10 else 0.35
    result_fig.subplots_adjust(left=0.12, bottom=bottom_margin, right=0.98, top=1.0)

    return result_fig


if __name__ == '__main__':
    """Run standalone to generate figure."""
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Generate INTENSE Figure 1C')
    parser.add_argument('--no-show', action='store_true', help='Do not display figure')
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    print("Loading experiment and creating Figure 1C...")
    # Uses StylePreset.from_make_beautiful() by default (standalone size)
    fig = create_fig1c()

    fig.savefig(output_dir / 'fig1c_neuron_feature_pair.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1c_neuron_feature_pair.png', dpi=300, bbox_inches='tight')
    print(f"Figure 1C saved to {output_dir}")

    if not args.no_show:
        plt.show()
