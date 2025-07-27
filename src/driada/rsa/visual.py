"""
Visualization functions for RSA.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Union, Dict, Any
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.colors import LinearSegmentedColormap

from ..utils.plot import make_beautiful, create_default_figure


def plot_rdm(
    rdm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    cmap: str = 'RdBu_r',
    figsize: Tuple[float, float] = (8, 7),
    show_values: bool = False,
    dendrogram_ratio: float = 0.2,
    cbar_label: str = 'Dissimilarity',
    ax: Optional[plt.Axes] = None
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
        The figure object
    """
    n_items = rdm.shape[0]
    
    if labels is None:
        labels = [f"Item {i+1}" for i in range(n_items)]
    
    if ax is None:
        # Create figure with optional dendrogram
        if dendrogram_ratio > 0:
            # Use create_default_figure for consistent styling
            fig, _ = create_default_figure(figsize=figsize, squeeze=False)
            
            # Create grid for dendrogram and heatmap
            gs = fig.add_gridspec(2, 2, height_ratios=[dendrogram_ratio, 1-dendrogram_ratio],
                                  width_ratios=[dendrogram_ratio, 1-dendrogram_ratio],
                                  hspace=0.02, wspace=0.02)
            
            # Dendrogram axes
            ax_dendro_top = fig.add_subplot(gs[0, 1])
            ax_dendro_left = fig.add_subplot(gs[1, 0])
            ax_main = fig.add_subplot(gs[1, 1])
            
            # Compute linkage
            linkage_matrix = linkage(rdm, method='average')
            
            # Plot dendrograms
            dendro_top = dendrogram(linkage_matrix, ax=ax_dendro_top, 
                                    orientation='top', no_labels=True)
            dendro_left = dendrogram(linkage_matrix, ax=ax_dendro_left,
                                     orientation='left', no_labels=True)
            
            # Hide dendrogram axes
            ax_dendro_top.set_visible(False)
            ax_dendro_left.set_visible(False)
            
            # Reorder RDM according to dendrogram
            order = dendro_top['leaves']
            rdm_ordered = rdm[order][:, order]
            labels_ordered = [labels[i] for i in order]
            
            ax = ax_main
            make_beautiful(ax)  # Apply DRIADA styling
        else:
            fig, ax = create_default_figure(figsize=figsize)
            rdm_ordered = rdm
            labels_ordered = labels
    else:
        fig = ax.figure
        rdm_ordered = rdm
        labels_ordered = labels
    
    # Plot heatmap
    im = ax.imshow(rdm_ordered, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_items))
    ax.set_yticks(np.arange(n_items))
    ax.set_xticklabels(labels_ordered, rotation=45, ha='right')
    ax.set_yticklabels(labels_ordered)
    
    # Add values if requested
    if show_values and n_items <= 20:  # Only show values for small RDMs
        for i in range(n_items):
            for j in range(n_items):
                text_color = 'white' if rdm_ordered[i, j] > np.median(rdm_ordered) else 'black'
                ax.text(j, i, f'{rdm_ordered[i, j]:.2f}',
                        ha='center', va='center', color=text_color, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    
    # Add title
    if title:
        ax.set_title(title, pad=20)
    
    # Add grid
    ax.grid(False)
    
    # Apply DRIADA styling if not already applied
    if not hasattr(ax, '_driada_styled'):
        make_beautiful(ax)
        ax._driada_styled = True
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_rdm_comparison(
    rdms: List[np.ndarray],
    labels: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'RdBu_r'
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
        The figure object
    """
    n_rdms = len(rdms)
    
    if figsize is None:
        figsize = (6 * n_rdms, 5)
    
    if titles is None:
        titles = [f"RDM {i+1}" for i in range(n_rdms)]
    
    # Use create_default_figure for consistent styling
    fig, axes = create_default_figure(figsize=figsize, ncols=n_rdms)
    if n_rdms == 1:
        axes = [axes]
    
    # Find global min/max for consistent color scale
    vmin = min(rdm.min() for rdm in rdms)
    vmax = max(rdm.max() for rdm in rdms)
    
    for i, (rdm, ax, title) in enumerate(zip(rdms, axes, titles)):
        im = ax.imshow(rdm, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set labels
        if labels is not None:
            n_items = rdm.shape[0]
            ax.set_xticks(np.arange(n_items))
            ax.set_yticks(np.arange(n_items))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            if i == 0:  # Only show y labels on first plot
                ax.set_yticklabels(labels)
            else:
                ax.set_yticklabels([])
        
        ax.set_title(title)
    
    # Add single colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Dissimilarity')
    
    plt.tight_layout()
    return fig


def plot_mds_from_rdm(
    rdm: np.ndarray,
    labels: Optional[List[str]] = None,
    n_dims: int = 2,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    colors: Optional[List] = None,
    marker_size: int = 100
) -> plt.Figure:
    """
    Plot MDS visualization of RDM.
    
    Parameters
    ----------
    rdm : np.ndarray
        Representational dissimilarity matrix
    labels : list of str, optional
        Labels for each item
    n_dims : int, default 2
        Number of MDS dimensions (2 or 3)
    title : str, optional
        Plot title
    figsize : tuple, default (8, 8)
        Figure size
    colors : list, optional
        Colors for each item
    marker_size : int, default 100
        Size of scatter points
        
    Returns
    -------
    fig : matplotlib.Figure
        The figure object
    """
    from sklearn.manifold import MDS
    
    # Compute MDS
    mds = MDS(n_components=n_dims, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(rdm)
    
    # Create plot
    if n_dims == 2:
        fig, ax = create_default_figure(figsize=figsize)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                            c=colors, s=marker_size, alpha=0.7)
        
        # Add labels
        if labels is not None:
            for i, label in enumerate(labels):
                ax.annotate(label, (coords[i, 0], coords[i, 1]),
                           xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('MDS Dimension 1')
        ax.set_ylabel('MDS Dimension 2')
        
    elif n_dims == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                            c=colors, s=marker_size, alpha=0.7)
        
        # Add labels
        if labels is not None:
            for i, label in enumerate(labels):
                ax.text(coords[i, 0], coords[i, 1], coords[i, 2], label)
        
        ax.set_xlabel('MDS Dimension 1')
        ax.set_ylabel('MDS Dimension 2')
        ax.set_zlabel('MDS Dimension 3')
    
    if title:
        ax.set_title(title)
    
    return fig


def plot_rdm_similarity_matrix(
    similarity_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "RDM Similarity Matrix",
    figsize: Tuple[float, float] = (8, 7),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Plot similarity matrix between multiple RDMs.
    
    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square matrix of RDM similarities
    labels : list of str, optional
        Labels for each RDM
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
        
    Returns
    -------
    fig : matplotlib.Figure
        The figure object
    """
    n_rdms = similarity_matrix.shape[0]
    
    if labels is None:
        labels = [f"RDM {i+1}" for i in range(n_rdms)]
    
    fig, ax = create_default_figure(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(similarity_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_rdms))
    ax.set_yticks(np.arange(n_rdms))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add values
    for i in range(n_rdms):
        for j in range(n_rdms):
            text_color = 'white' if similarity_matrix[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                    ha='center', va='center', color=text_color)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity')
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig