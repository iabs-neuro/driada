"""MI vs p-value scatter plots for NeuronDatabase."""

import numpy as np
import matplotlib.pyplot as plt

from driada.utils.plot import make_beautiful
from .configs import MI_THRESHOLD, PVAL_THRESHOLD
from .tables import apply_significance_filters


def plot_mi_pval_scatter(db, feature, session, ax,
                         mi_threshold=MI_THRESHOLD,
                         pval_threshold=PVAL_THRESHOLD,
                         filter_delay=None):
    """Scatter plot of log10(p-value) vs MI for one feature and session.

    Parameters
    ----------
    db : NeuronDatabase
    feature : str
    session : str
    ax : matplotlib.axes.Axes
    mi_threshold : float
    pval_threshold : float
    filter_delay : bool or None
        None uses db.filter_delay.
    """
    if filter_delay is None:
        filter_delay = db.filter_delay

    df = db.data
    df = df[(df['feature'] == feature) & (df['session'] == session)]

    # All neurons with a real MI value
    valid = df[df['me'] > 0].copy()

    # Significant subset
    sig = apply_significance_filters(
        valid, mi_threshold=mi_threshold, pval_threshold=pval_threshold,
        filter_delay=filter_delay)

    pval_floor = 1e-30
    log_pval = np.log10(np.clip(valid['pval'].values.astype(float),
                                pval_floor, None))
    mi_vals = valid['me'].values.astype(float)

    log_pval_sig = np.log10(np.clip(sig['pval'].values.astype(float),
                                    pval_floor, None))
    mi_sig = sig['me'].values.astype(float)

    ax.scatter(log_pval, mi_vals, c='b', alpha=0.5,
               label=f'all ({len(valid)})', s=10)
    ax.scatter(log_pval_sig, mi_sig, c='g', alpha=0.7,
               label=f'significant ({len(sig)})', s=10)

    ax.axhline(mi_threshold, c='k', lw=0.8)
    ax.axvline(np.log10(pval_threshold), c='k', lw=0.8)

    make_beautiful(ax)
    ax.set_xlabel('log10(p-value)')
    ax.set_ylabel('MI')
    ax.set_title(session)
    ax.set_xlim(-30, 0)
    ax.legend(fontsize=8)


def _grid_shape(n):
    """Return (rows, cols) for n panels."""
    if n <= 3:
        return 1, n
    if n == 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    side = int(np.ceil(np.sqrt(n)))
    return side, side


def plot_mi_pval_grid(db, feature, sessions=None, figsize=(18, 12),
                      mi_threshold=MI_THRESHOLD,
                      pval_threshold=PVAL_THRESHOLD,
                      filter_delay=None):
    """Multi-panel MI vs p-value scatter, one subplot per session.

    Parameters
    ----------
    db : NeuronDatabase
    feature : str
    sessions : list[str] or None
        Defaults to db.sessions.
    figsize : tuple
    mi_threshold : float
    pval_threshold : float
    filter_delay : bool or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if sessions is None:
        sessions = db.sessions

    nrows, ncols = _grid_shape(len(sessions))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.atleast_1d(axs).ravel()

    for i, session in enumerate(sessions):
        plot_mi_pval_scatter(db, feature, session, axs[i],
                             mi_threshold=mi_threshold,
                             pval_threshold=pval_threshold,
                             filter_delay=filter_delay)

    # Hide unused axes
    for j in range(len(sessions), len(axs)):
        axs[j].set_visible(False)

    fig.suptitle(feature, fontsize=16)
    fig.tight_layout()
    return fig
