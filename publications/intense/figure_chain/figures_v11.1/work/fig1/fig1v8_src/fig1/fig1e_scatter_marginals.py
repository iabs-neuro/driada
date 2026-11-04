"""Figure 1E: INTENSE results scatter plot with marginal distributions.

Shows MI vs log(p-value) for neuron-feature pairs with acceptance thresholds.
Accepted pairs shown in green, rejected in blue.
Size: 17.78 x 17.78 cm (square for scatter aspect ratio)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from driada.utils.plot import make_beautiful
from driada.utils.publication import StylePreset

# Handle both module and standalone imports
try:
    from .data_loader import load_nof_experiment
except ImportError:
    from data_loader import load_nof_experiment


def plot_scatter_with_marginals(
    x, y,
    x2=None, y2=None,
    mi_thr=None,
    pval_hardthr=None,
    marginal_type='hist_kde',
    bins=30,
    kde_smooth=1.0,
    scatter_color='b',
    scatter_color2='#4daf4a',
    marginal_color='gray',
    figsize=(8, 8),
    dpi=300,
    title='Scatter with Marginals',
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    skip_tight_layout: bool = False,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Plot scatter with marginal distributions for INTENSE results.

    Parameters
    ----------
    x : array-like
        Log10(p-values) for first group (rejected)
    y : array-like
        MI values for first group (rejected)
    x2 : array-like, optional
        Log10(p-values) for second group (accepted)
    y2 : array-like, optional
        MI values for second group (accepted)
    mi_thr : float, optional
        MI threshold line (horizontal)
    pval_hardthr : float, optional
        P-value threshold (will be log10 transformed for vertical line)
    marginal_type : str, default 'hist_kde'
        Type of marginal: 'histogram', 'kde', 'hist_kde', or 'boxplot'
    bins : int, default 30
        Number of histogram bins
    kde_smooth : float, default 1.0
        KDE bandwidth adjustment
    scatter_color : str, default 'b'
        Color for first group (rejected)
    scatter_color2 : str, default '#4daf4a'
        Color for second group (accepted)
    marginal_color : str, default 'gray'
        Color for marginal distributions (when single group)
    figsize : tuple, default (8, 8)
        Figure size in inches (ignored if fig provided)
    dpi : int, default 300
        Figure DPI (ignored if fig provided)
    title : str, default 'Scatter with Marginals'
        Plot title
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on (for framework integration)
    style : StylePreset, optional
        Style preset from publication framework.
        If None, uses StylePreset.from_make_beautiful() for standalone figures.
    panel_size : tuple of float, optional
        Panel size (width, height) in cm for style scaling.
        If None, uses default standalone size.
    skip_tight_layout : bool, optional
        If True, skip plt.tight_layout() call. Useful for subfigure layouts.
        Default: False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : dict
        Dictionary with 'scatter', 'top', 'right' axes
    """
    # Default to standalone preset if not provided
    if style is None:
        style = StylePreset.from_make_beautiful()

    # Default panel size for standalone
    if panel_size is None:
        panel_size = (17.78, 17.78)

    # Get scaled style values
    line_width = style.get_line_width(panel_size, 'cm')
    marker_size = style.get_marker_size(panel_size, 'cm')

    # Create figure or use provided
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)

    # Use width_ratios and height_ratios for better proportions
    # Main scatter should be larger, marginals less squeezed
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.24], height_ratios=[0.24, 1],
                          wspace=0.02, hspace=0.02)

    # Main scatter plot (bottom-left, largest area)
    ax = fig.add_subplot(gs[1, 0])

    # Scatter points
    ax.scatter(x, y, c=scatter_color, label='rejected', alpha=0.7, s=marker_size)
    if x2 is not None and y2 is not None:
        ax.scatter(x2, y2, c=scatter_color2, label='accepted', alpha=0.7, s=marker_size)

    # Y-axis ticks (matching reference)
    y_ticks = [0, 0.02, 0.04, 0.06, 0.08, 0.10]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{val:.2f}' for val in y_ticks])

    # X-axis ticks (matching reference: log(p) from -8 to 0)
    x_ticks = [-8, -6, -4, -2, 0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks])

    # Labels
    ax.set_xlabel(r'$log(p)$')
    ax.set_ylabel('mutual information')

    # Apply framework styling
    style.apply_to_axes(ax, panel_size, 'cm')

    # Add legend below the scatter plot (two rows, reduced spacing)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.27),  # Moved up 10%
        ncol=1,
        frameon=False,
        fontsize=style.base_legend_fontsize if hasattr(style, 'base_legend_fontsize') else 7,
    )

    # Threshold lines
    if mi_thr is not None:
        ax.axhline(mi_thr, c='m', linewidth=line_width, linestyle='dashed')
    if pval_hardthr is not None:
        ax.axvline(np.log10(pval_hardthr), c='m', linewidth=line_width, linestyle='dashed')

    # Square aspect
    ax.set_box_aspect(1)

    # Top marginal (x-axis) - top-left
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_top.tick_params(labelbottom=False)

    # Right marginal (y-axis) - bottom-right, narrow
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_right.tick_params(labelleft=False)

    # Plot marginals based on type
    if marginal_type == 'histogram':
        ax_top.hist(x, bins=bins, color=marginal_color, alpha=0.7, density=True)
        ax_right.hist(y, bins=bins, color=marginal_color, alpha=0.7, density=True, orientation='horizontal')

    elif marginal_type == 'kde':
        sns.kdeplot(x=x, ax=ax_top, color=marginal_color, bw_adjust=kde_smooth, linewidth=line_width)
        sns.kdeplot(y=y, ax=ax_right, color=marginal_color, bw_adjust=kde_smooth, linewidth=line_width)

    elif marginal_type == 'hist_kde':
        ax_top.hist(x, bins=bins, color=scatter_color, alpha=0.7, density=True)
        ax_right.hist(y, bins=bins, color=scatter_color, alpha=0.7, density=True, orientation='horizontal')
        sns.kdeplot(x=x, ax=ax_top, color=scatter_color, bw_adjust=kde_smooth, linewidth=line_width)
        sns.kdeplot(y=y, ax=ax_right, color=scatter_color, bw_adjust=kde_smooth, linewidth=line_width)

        if x2 is not None and y2 is not None:
            ax_top.hist(x2, bins=bins//2, color=scatter_color2, alpha=0.7, density=True)
            ax_right.hist(y2, bins=bins//2, color=scatter_color2, alpha=0.7, density=True, orientation='horizontal')
            sns.kdeplot(x=x2, ax=ax_top, color=scatter_color2, bw_adjust=kde_smooth, linewidth=line_width)
            sns.kdeplot(y=y2, ax=ax_right, color=scatter_color2, bw_adjust=kde_smooth, linewidth=line_width)

    elif marginal_type == 'boxplot':
        ax_top.boxplot(x, vert=False, patch_artist=True,
                       boxprops=dict(facecolor=marginal_color, alpha=0.7))
        ax_right.boxplot(y, vert=True, patch_artist=True,
                         boxprops=dict(facecolor=marginal_color, alpha=0.7))

    ax_top.set_axis_off()
    ax_right.set_axis_off()

    if not skip_tight_layout:
        plt.tight_layout()
    else:
        # When used as subfigure, set margins for x-axis alignment with Panel D
        # Minimize margins to maximize scatter plot area
        fig.subplots_adjust(left=0.18, right=1.0, bottom=0.15, top=1.0)

    return fig, {'scatter': ax, 'top': ax_top, 'right': ax_right}


import _paths_fig as _F  # noqa: E402  (figure_chain/ is on sys.path, see data_loader.py)
REAL_PAIRS = _F.data_dir(Path(__file__).resolve().parent) / 'cache' / 'fig1e_real_pairs.npz'
X_CLIP = 20.0
# Legend entries with the number of pairs, e.g. 'accepted (9,009)', as in v8; False: names only.
LEGEND_COUNTS = True
# HIST_OUTLINE_LW: outline of the histogram bars in points (None: no outline, as in v8-v10.2).
# RIGHT_HIST_HEADROOM: room beyond the longest bar of the right histogram, as a fraction of it.
# MARKER_SCALE: point size as a fraction of the marker size of the style (0.15 in v8-v10.2).
HIST_OUTLINE_LW = None
RIGHT_HIST_HEADROOM = 0.0
MARKER_SCALE = 0.15


def create_fig1e(
    data_path=REAL_PAIRS,
    mi_threshold=0.04,
    pval_threshold=0.001,
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    seed=42,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create Figure 1E: real NOF neuron-feature pairs in the significance-information plane.

    Pairs are those of the 64 NOF sessions (DRIADA 1.2.2, seed 42, before disentanglement)
    that passed stage 1 and so have a stage-2 p-value (built by extract_fig1e_data.py).
    x: -log10 p (clipped at X_CLIP); y: excess MI above chance, log scale.
    Green: accepted (stage-2 significant, excess MI > threshold, activation);
    grey: p below threshold but anti-selective (removed by the filter); blue: the rest.

    Parameters
    ----------
    data_path : str or Path
        npz written by extract_fig1e_data.py.
    mi_threshold : float, default 0.04
        Excess-MI threshold (bits).
    pval_threshold : float, default 0.001
        Per-pair p-value threshold.
    fig : matplotlib.figure.Figure, optional
        Existing figure or subfigure to draw into.
    style : StylePreset, optional
        Style preset; standalone preset if None.
    panel_size : tuple of float, optional
        Panel size in cm for style scaling.
    seed : int, default 42
        Seed of the drawing order shuffle (points are drawn in random order).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict
        'scatter', 'top', 'right' axes.
    """
    if style is None:
        style = StylePreset.from_make_beautiful()
    if panel_size is None:
        panel_size = (17.78, 17.78)
    if fig is None:
        fig = plt.figure(figsize=(8, 8), dpi=300)

    d = np.load(data_path)
    x = np.minimum(-np.log10(d['pval']), X_CLIP)
    y = np.log10(d['me_excess'])
    acc, anti = d['accepted'], d['anti_selective']
    rej = ~acc & ~anti
    groups = [('rejected', rej, '#3b6fb6'), ('anti-selective', anti, '#9e9e9e'),
              ('accepted', acc, '#4daf4a')]

    line_width = style.get_line_width(panel_size, 'cm')
    marker_size = style.get_marker_size(panel_size, 'cm') * MARKER_SCALE

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.24], height_ratios=[0.24, 1],
                          wspace=0.02, hspace=0.02)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # one shuffled draw order so that no group hides the others
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    colors = np.empty(len(x), dtype=object)
    for _, m, c in groups:
        colors[m] = c
    ax.scatter(x[order], y[order], c=list(colors[order]), s=marker_size, alpha=0.5,
               linewidths=0, rasterized=True)
    for name, m, c in groups:
        ax.scatter([], [], c=c, s=12, label=f'{name} ({m.sum():,})' if LEGEND_COUNTS else name)

    xb = np.linspace(0, X_CLIP, 41)
    yb = np.linspace(y.min(), y.max(), 41)
    for name, m, c in groups:
        ax_top.hist(x[m], bins=xb, color=c, alpha=0.6, density=True)
        ax_right.hist(y[m], bins=yb, color=c, alpha=0.6, density=True,
                      orientation='horizontal')
    if HIST_OUTLINE_LW:
        for name, m, c in groups:
            ax_top.hist(x[m], bins=xb, color=c, density=True, histtype='step',
                        linewidth=HIST_OUTLINE_LW, zorder=3)
            ax_right.hist(y[m], bins=yb, color=c, density=True, histtype='step',
                          linewidth=HIST_OUTLINE_LW, orientation='horizontal', zorder=3)
    if RIGHT_HIST_HEADROOM:
        ax_right.set_xlim(0, ax_right.get_xlim()[1] * (1 + RIGHT_HIST_HEADROOM))

    ax.axvline(-np.log10(pval_threshold), c='m', linewidth=line_width, linestyle='dashed')
    ax.axhline(np.log10(mi_threshold), c='m', linewidth=line_width, linestyle='dashed')

    y_ticks = [0.003, 0.01, 0.04, 0.1, 0.3]
    ax.set_yticks(np.log10(y_ticks))
    ax.set_yticklabels([f'{v:g}' for v in y_ticks])
    ax.set_xticks([0, 3, 10, 20])
    ax.set_xticklabels(['0', '3', '10', r'$\geq$20'])
    ax.set_xlim(0, X_CLIP + 0.5)
    ax.set_xlabel(r'$-\log_{10} p$')
    ax.set_ylabel('excess MI (bits)')
    style.apply_to_axes(ax, panel_size, 'cm')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.36), ncol=1, frameon=False,
              fontsize=getattr(style, 'base_legend_fontsize', 7), handletextpad=0.2,
              borderaxespad=0, labelspacing=0.2)
    ax.set_box_aspect(1)
    ax_top.set_axis_off()
    ax_right.set_axis_off()
    fig.subplots_adjust(left=0.18, right=1.0, bottom=0.15, top=1.0)
    return fig, {'scatter': ax, 'top': ax_top, 'right': ax_right}


def create_fig1e_with_framework(data_path=None, mi_threshold=0.04, pval_threshold=0.001,
                                 panel_size=(12, 12)):
    """Create Figure 1E using publication framework for consistent styling.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to .npz file with data
    mi_threshold : float, default 0.04
        MI threshold for acceptance
    pval_threshold : float, default 0.001
        P-value threshold for acceptance
    panel_size : tuple, default (12, 12)
        Panel size in cm

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict
    """
    from driada.utils.publication import PanelLayout, StylePreset

    # Setup layout
    layout = PanelLayout(units='cm', dpi=300)
    layout.add_panel('scatter', size=panel_size)
    layout.set_grid(rows=1, cols=1)

    style = StylePreset.publication_default()

    # Create figure from layout
    fig, _ = layout.create_figure(style=None)  # Don't apply style yet

    # Load or generate data (same as create_fig1e)
    if data_path is not None:
        data = np.load(data_path)
        pvals_rejected = data.get('pvals', data.get('pval', np.array([])))
        mi_rejected = data.get('powers', data.get('power', np.array([])))
        pvals_accepted = data.get('pvals_accepted', data.get('good_pval', np.array([])))
        mi_accepted = data.get('powers_accepted', data.get('good_power', np.array([])))
    else:
        np.random.seed(42)
        n_rejected, n_accepted = 1500, 150
        pvals_rejected = 10 ** np.random.normal(-2.5, 1.2, n_rejected)
        pvals_rejected = np.clip(pvals_rejected, 1e-8, 1)
        mi_rejected = np.abs(np.random.exponential(0.015, n_rejected))
        mi_rejected = np.clip(mi_rejected, 0, 0.1)
        pvals_accepted = 10 ** np.random.normal(-5, 1.5, n_accepted)
        pvals_accepted = np.clip(pvals_accepted, 1e-8, 0.01)
        mi_accepted = np.random.normal(0.055, 0.015, n_accepted)
        mi_accepted = np.clip(mi_accepted, 0.03, 0.1)

    # Plot with framework integration
    fig, axes = plot_scatter_with_marginals(
        np.log10(pvals_rejected), mi_rejected,
        x2=np.log10(pvals_accepted),
        y2=mi_accepted,
        mi_thr=mi_threshold,
        pval_hardthr=pval_threshold,
        marginal_type='hist_kde',
        kde_smooth=0.5,
        bins=20,
        fig=fig,
        style=style,
        panel_size=panel_size,
    )

    return fig, axes


if __name__ == '__main__':
    """Run standalone to test."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Generate INTENSE Figure 1E')
    parser.add_argument('--no-show', action='store_true', help='Do not display figure')
    parser.add_argument('--with-framework', action='store_true', help='Also generate framework version')
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    # Generate standard version (uses StylePreset.from_make_beautiful() by default)
    print("Generating Figure 1E...")
    fig, axes = create_fig1e()
    fig.savefig(output_dir / 'fig1e_scatter_marginals.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1e_scatter_marginals.png', dpi=300, bbox_inches='tight')
    print(f"Figure 1E saved to {output_dir}")

    # Optionally generate framework version
    if args.with_framework:
        print("Generating Figure 1E with framework...")
        fig2, axes2 = create_fig1e_with_framework()
        fig2.savefig(output_dir / 'fig1e_with_framework.pdf', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / 'fig1e_with_framework.png', dpi=300, bbox_inches='tight')
        print(f"Figure 1E (framework) saved to {output_dir}")

    if not args.no_show:
        plt.show()
