"""Figure 1B: Behavioral traces with head direction, object presence, and position.

Shows behavioral data traces: head direction, object 4 presence, and x/y coordinates.
Size: 17.78 x 24.6 cm (same as Fig 1A)
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
    from .data_loader import load_nof_experiment, load_fig1b_cache
    from .plot_utils import plot_shadowed_groups
except ImportError:
    from data_loader import load_nof_experiment, load_fig1b_cache
    from plot_utils import plot_shadowed_groups


# FPS: frame rate of the session for the time axis (20 in v7-v10.2). EXACT_SHADING: the
# interaction periods span exactly their frames (plot_shadowed_groups adds 1 s to each).
FPS = 20.0
EXACT_SHADING = False


def create_fig1b(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[plt.Figure], List[plt.Axes]]:
    """Create Figure 1B: Behavioral traces.

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

    # Default panel size for standalone
    if panel_size is None:
        panel_size = (17.78, 24.6)

    # Parameters
    ds = 1  # Downsampling factor
    lb = 0  # Left bound
    rb = 5000  # Right bound

    # Load data from cache (fast) or full experiment (slow)
    try:
        cache = load_fig1b_cache()
        headdirection = cache['headdirection']
        object4 = cache['object4']
        x_data = cache['x']
        y_data = cache['y']
    except FileNotFoundError:
        Exp = load_nof_experiment(animal_id='H01', day='3D', verbose=False)
        headdirection = Exp.headdirection.data[lb:rb]
        object4 = Exp.object4.data[lb:rb]
        x_data = Exp.x.data[lb:rb]
        y_data = Exp.y.data[lb:rb]

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

    # Convert frame indices to time in seconds (assuming 20 fps)
    time_vals = np.arange(lb, rb)[::ds] / FPS

    # Plot each behavioral trace
    for i, ax_i in enumerate(axs):
        if i == 0:
            # Head direction (RED to match reference)
            ax_i.plot(time_vals, headdirection[::ds],
                      c='r', linewidth=line_width)
        elif i == 1:
            # Object interaction (GRAY to match reference)
            if EXACT_SHADING:
                from nfp_fps import shade_runs
                shade_runs(ax_i, time_vals, object4[::ds], ds, FPS, alpha=0.7, color='gray', label=None)
            else:
                plot_shadowed_groups(ax_i, time_vals, object4[::ds],
                                     alpha=0.7, color='gray', label=None)
        elif i == 2:
            # X and Y coordinates (MAGENTA and CYAN to match reference)
            # Create legend entries
            ax_i.plot([], [], c='r', linewidth=line_width,
                      label='head direction angle')
            ax_i.fill_between([], [], color='gray', alpha=0.7,
                              label='inter. with object 4')
            ax_i.plot([], [], c='m', linewidth=line_width,
                      label='animal position: x')
            ax_i.plot([], [], c='c', linewidth=line_width,
                      label='animal position: y')

            # Now plot actual data
            ax_i.plot(time_vals, x_data[::ds],
                      c='m', linewidth=line_width)
            ax_i.plot(time_vals, y_data[::ds],
                      c='c', linewidth=line_width)

        # Remove y-axis ticks and labels completely (reference has none)
        ax_i.set_yticks([])
        ax_i.set_yticklabels([])

        # X-label only on bottom subplot
        if i == 2:
            ax_i.set_xlabel('time, s')

        # Apply framework styling to axis
        style.apply_to_axes(ax_i, panel_size, 'cm')

        # Handle legend on bottom subplot - centered below xlabel (20% reduced spacing)
        if i == 2:
            ax_i.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.648),  # Moved up 20% total
                ncol=2,
                frameon=style.legend_frameon,
                fontsize=style.base_legend_fontsize,
            )

    # Increased hspace between subplots like in reference, reserve bottom for xlabel + legend (20% reduced)
    # left=0.12 to align y-axis with Panel A
    fig.subplots_adjust(left=0.12, bottom=0.32, right=0.95, top=1.0, hspace=0.2)

    return fig, list(axs)


if __name__ == '__main__':
    """Run standalone to test."""
    from pathlib import Path

    # Output to same directory as script
    output_dir = Path(__file__).parent

    # Uses StylePreset.from_make_beautiful() by default (standalone size)
    fig, axs = create_fig1b()
    fig.savefig(output_dir / 'fig1b_behavioral_traces.pdf', dpi=300)
    fig.savefig(output_dir / 'fig1b_behavioral_traces.png', dpi=300)
    print(f"Figure 1B saved to {output_dir}")
    plt.show()
