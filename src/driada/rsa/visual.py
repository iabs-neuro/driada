"""
Visualization functions for RSA.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage

from ..utils.plot import make_beautiful, create_default_figure


def plot_rdm(
    rdm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (8, 7),
    show_values: bool = False,
    dendrogram_ratio: float = 0.2,
    cbar_label: str = "Dissimilarity",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a representational dissimilarity matrix with optional dendrogram.

    Parameters
    ----------
    rdm : np.ndarray
        Square RDM matrix
    labels : list of str, optional
        Labels for each condition/item
    title : str, optional
        Plot title
    cmap : str, default 'RdBu_r'
        Colormap for the heatmap
    figsize : tuple, default (8, 7)
        Figure size if creating new figure
    show_values : bool, default False
        Whether to show numerical values in cells
    dendrogram_ratio : float, default 0.2
        Proportion of figure height for dendrogram (0 to disable)
    cbar_label : str, default 'Dissimilarity'
        Label for colorbar
    ax : matplotlib.Axes, optional
        Existing axes to plot on (disables dendrogram)

    Returns
    -------
    fig : matplotlib.Figure
        The figure object"""
    n_items = rdm.shape[0]

    if labels is None:
        labels = [f"Item {i+1}" for i in range(n_items)]

    if ax is None:
        # Create figure with optional dendrogram
        if dendrogram_ratio > 0:
            # Create figure
            fig = plt.figure(figsize=figsize)

            # Create grid for dendrogram and heatmap
            gs = fig.add_gridspec(
                2,
                2,
                height_ratios=[dendrogram_ratio, 1 - dendrogram_ratio],
                width_ratios=[dendrogram_ratio, 1 - dendrogram_ratio],
                hspace=0.02,
                wspace=0.02,
            )

            # Dendrogram axes
            ax_dendro_top = fig.add_subplot(gs[0, 1])
            ax_dendro_left = fig.add_subplot(gs[1, 0])
            ax_main = fig.add_subplot(gs[1, 1])

            # Hide the unused corner
            ax_corner = fig.add_subplot(gs[0, 0])
            ax_corner.axis('off')

            # Compute linkage
            linkage_matrix = linkage(rdm, method="average")

            # Plot dendrograms
            dendro_top = dendrogram(
                linkage_matrix, ax=ax_dendro_top, orientation="top", no_labels=True
            )
            dendro_left = dendrogram(
                linkage_matrix, ax=ax_dendro_left, orientation="left", no_labels=True
            )

            # Completely hide dendrogram axes (not just invisible)
            ax_dendro_top.axis('off')
            ax_dendro_left.axis('off')

            # Reorder RDM according to dendrogram
            order = dendro_top["leaves"]
            rdm_ordered = rdm[order][:, order]
            labels_ordered = [labels[i] for i in order]

            ax = ax_main
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            rdm_ordered = rdm
            labels_ordered = labels
    else:
        fig = ax.figure
        rdm_ordered = rdm
        labels_ordered = labels

    # Plot heatmap
    im = ax.imshow(rdm_ordered, cmap=cmap, aspect="auto", interpolation='nearest')

    # Set ticks and labels with proper styling
    ax.set_xticks(np.arange(n_items))
    ax.set_yticks(np.arange(n_items))
    ax.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels_ordered, fontsize=11)

    # Style tick parameters
    ax.tick_params(axis='both', which='major', width=2, length=6, labelsize=11)

    # Add values if requested
    if show_values and n_items <= 20:  # Only show values for small RDMs
        for i in range(n_items):
            for j in range(n_items):
                text_color = "white" if rdm_ordered[i, j] > np.median(rdm_ordered) else "black"
                ax.text(
                    j,
                    i,
                    f"{rdm_ordered[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

    # Add colorbar with proper positioning
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10, width=2, length=4)

    # Add title
    if title:
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(False)

    # Style spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Adjust layout to prevent overlap (use rect to leave space for colorbar)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_rdm_comparison(
    rdms: List[np.ndarray],
    labels: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Plot multiple RDMs side by side for comparison.

    Parameters
    ----------
    rdms : list of np.ndarray
        List of RDM matrices to compare
    labels : list of str, optional
        Labels for conditions/items (same for all RDMs)
    titles : list of str, optional
        Title for each RDM
    figsize : tuple, optional
        Figure size (default based on number of RDMs)
    cmap : str, default 'RdBu_r'
        Colormap for the heatmaps

    Returns
    -------
    fig : matplotlib.Figure
        The figure object"""
    n_rdms = len(rdms)

    # Validate that all RDMs have the same shape
    if n_rdms > 0:
        first_shape = rdms[0].shape
        for i, rdm in enumerate(rdms[1:], 1):
            if rdm.shape != first_shape:
                raise ValueError(
                    f"All RDMs must have the same shape. RDM 0 has shape {first_shape}, "
                    f"but RDM {i} has shape {rdm.shape}"
                )

    if figsize is None:
        figsize = (6 * n_rdms, 5)

    if titles is None:
        titles = [f"RDM {i+1}" for i in range(n_rdms)]

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_rdms, figsize=figsize)
    if n_rdms == 1:
        axes = [axes]

    # Find global min/max for consistent color scale
    vmin = min(rdm.min() for rdm in rdms)
    vmax = max(rdm.max() for rdm in rdms)

    for i, (rdm, ax, title) in enumerate(zip(rdms, axes, titles)):
        im = ax.imshow(rdm, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        # Set labels with proper styling
        if labels is not None:
            n_items = rdm.shape[0]
            ax.set_xticks(np.arange(n_items))
            ax.set_yticks(np.arange(n_items))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
            if i == 0:  # Only show y labels on first plot
                ax.set_yticklabels(labels, fontsize=12)
            else:
                ax.set_yticklabels([])

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.tick_params(width=2, length=6)
        ax.grid(False)

        # Style spines
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Add single colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Dissimilarity", fontsize=12)
    cbar.ax.tick_params(labelsize=10, width=2, length=4)

    fig.tight_layout()
    return fig
