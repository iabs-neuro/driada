"""
Visualization utilities for DRIADA
==================================

This module provides reusable visualization functions for embedding comparisons,
trajectory plots, and component interpretation in dimensionality reduction analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import gaussian_kde

# Import validation utilities
from driada.utils.data import check_positive, check_nonnegative

# Default DPI for all plots
DEFAULT_DPI = 150


def plot_embedding_comparison(
    embeddings: Dict[str, np.ndarray],
    features: Optional[Dict[str, np.ndarray]] = None,
    feature_names: Optional[Dict[str, str]] = None,
    methods: Optional[List[str]] = None,
    with_trajectory: bool = True,
    compute_metrics: bool = True,
    trajectory_kwargs: Optional[Dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> plt.Figure:
    """
    Create comprehensive embedding comparison figure with behavioral features and trajectories.

    Parameters
    ----------
    embeddings : dict
        Dictionary mapping method names to embedding arrays (n_samples, n_components).
        Arrays must be 2D with at least 2 components.
    features : dict, optional
        Dictionary mapping feature names to feature arrays. Arrays must have same
        length as embeddings. Default features used: 'angle' (circular position 
        in radians [-π, π]) and 'speed'
    feature_names : dict, optional
        Dictionary mapping feature keys to display names
    methods : list of str, optional
        List of methods to plot (if None, uses all keys in embeddings)
    with_trajectory : bool, default True
        Whether to include trajectory visualization as a third row
    compute_metrics : bool, default True
        Whether to compute and display metrics (density contours, percentiles)
    trajectory_kwargs : dict, optional
        Additional keyword arguments for trajectory plotting
    figsize : tuple, optional
        Figure size (width, height). If None, computed based on number of methods
    save_path : str, optional
        Path to save the figure
    dpi : int, default DEFAULT_DPI
        DPI resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    
    Raises
    ------
    ValueError
        If embeddings are not 2D arrays with at least 2 components, or if
        feature arrays have mismatched lengths
    
    Notes
    -----
    Methods not found in embeddings dict are silently skipped.
    KDE computation failures are caught and contours are omitted.    """
    # Validate embeddings
    for method, embedding in embeddings.items():
        if embedding.ndim != 2:
            raise ValueError(f"Embedding for {method} must be 2D, got shape {embedding.shape}")
        if embedding.shape[1] < 2:
            raise ValueError(f"Embedding for {method} must have at least 2 components, got {embedding.shape[1]}")
    
    if methods is None:
        methods = list(embeddings.keys())

    n_methods = len(methods)
    n_rows = 3 if with_trajectory else 2

    # Set figure size
    if figsize is None:
        figsize = (6 * n_methods, 5 * n_rows)

    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_methods, hspace=0.3, wspace=0.3)

    # Default feature names
    if feature_names is None:
        feature_names = {"angle": "Head Direction", "speed": "Speed"}

    # Default trajectory kwargs
    if trajectory_kwargs is None:
        trajectory_kwargs = {}

    default_traj_kwargs = {
        "linewidth": 0.8,
        "alpha": 0.3,
        "color": "k",
        "arrow_spacing": 20,
        "arrow_scale": 0.3,
        "start_marker": "o",
        "end_marker": "s",
        "marker_size": 100,
    }
    default_traj_kwargs.update(trajectory_kwargs)

    for i, method in enumerate(methods):
        if method not in embeddings:
            continue

        embedding = embeddings[method]
        
        # Validate features match embedding length
        if features is not None:
            for feat_name, feat_array in features.items():
                if len(feat_array) != len(embedding):
                    raise ValueError(f"Feature '{feat_name}' length ({len(feat_array)}) "
                                   f"doesn't match embedding length ({len(embedding)})")

        # First row: colored by angle/position
        ax1 = fig.add_subplot(gs[0, i])

        if features is not None and "angle" in features:
            angle = features["angle"]
            # Normalize angle to [0, 1] for color mapping
            angle_norm = (angle + np.pi) / (2 * np.pi)

            scatter = ax1.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=angle_norm,
                cmap="hsv",
                alpha=0.7,
                s=2,
                vmin=0,
                vmax=1,
                edgecolors="none",
            )

            cbar = plt.colorbar(
                scatter, ax=ax1, label=feature_names.get("angle", "Angle")
            )
            # Set colorbar ticks to show actual angles
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

            # Add density contours if requested
            if compute_metrics:
                try:
                    kde = gaussian_kde(embedding[:, :2].T)
                    x_min, x_max = ax1.get_xlim()
                    y_min, y_max = ax1.get_ylim()
                    X, Y = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
                    positions_grid = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(kde(positions_grid).T, X.shape)
                    ax1.contour(X, Y, Z, colors="gray", alpha=0.3, linewidths=0.5)
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    pass  # Skip contours if KDE fails (singular matrix, etc.)
        else:
            ax1.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=1)

        ax1.set_xlabel("Component 0")
        ax1.set_ylabel("Component 1")
        ax1.set_title(f'{method.upper()} - {feature_names.get("angle", "Position")}')
        ax1.grid(True, alpha=0.3)

        # Second row: colored by speed or second feature
        ax2 = fig.add_subplot(gs[1, i])

        if features is not None and "speed" in features:
            speed = features["speed"]

            scatter = ax2.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=speed,
                cmap="viridis",
                alpha=0.7,
                s=2,
                edgecolors="none",
            )

            cbar = plt.colorbar(
                scatter, ax=ax2, label=feature_names.get("speed", "Speed")
            )

            # Add percentile markers if requested
            if compute_metrics:
                speed_percentiles = np.percentile(speed, [25, 50, 75])
                for p, val in zip([25, 50, 75], speed_percentiles):
                    cbar.ax.axhline(y=val, color="red", alpha=0.3, linewidth=0.5)
                    cbar.ax.text(
                        1.05,
                        val,
                        f"{p}%",
                        transform=cbar.ax.get_yaxis_transform(),
                        fontsize=8,
                        va="center",
                    )
        else:
            ax2.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=1)

        ax2.set_xlabel("Component 0")
        ax2.set_ylabel("Component 1")
        ax2.set_title(f'{method.upper()} - {feature_names.get("speed", "Feature 2")}')
        ax2.grid(True, alpha=0.3)

        # Third row: trajectory visualization
        if with_trajectory:
            ax3 = fig.add_subplot(gs[2, i])

            # Plot trajectory
            ax3.plot(
                embedding[:, 0],
                embedding[:, 1],
                color=default_traj_kwargs["color"],
                alpha=default_traj_kwargs["alpha"],
                linewidth=default_traj_kwargs["linewidth"],
            )

            # Add arrow markers to show direction
            trajectory_samples = len(embedding)
            arrow_spacing = max(
                1, trajectory_samples // default_traj_kwargs["arrow_spacing"]
            )

            for j in range(0, trajectory_samples - arrow_spacing, arrow_spacing):
                if j + 1 < trajectory_samples:
                    dx = embedding[j + 1, 0] - embedding[j, 0]
                    dy = embedding[j + 1, 1] - embedding[j, 1]

                    # Only plot arrow if movement is significant
                    if np.sqrt(dx**2 + dy**2) > 0.001:
                        ax3.arrow(
                            embedding[j, 0],
                            embedding[j, 1],
                            dx * default_traj_kwargs["arrow_scale"],
                            dy * default_traj_kwargs["arrow_scale"],
                            head_width=0.02,
                            head_length=0.02,
                            fc="red",
                            ec="red",
                            alpha=0.6,
                        )

            # Mark start and end points
            ax3.scatter(
                embedding[0, 0],
                embedding[0, 1],
                c="green",
                s=default_traj_kwargs["marker_size"],
                marker=default_traj_kwargs["start_marker"],
                edgecolors="black",
                linewidth=2,
                label="Start",
                zorder=5,
            )
            ax3.scatter(
                embedding[-1, 0],
                embedding[-1, 1],
                c="red",
                s=default_traj_kwargs["marker_size"],
                marker=default_traj_kwargs["end_marker"],
                edgecolors="black",
                linewidth=2,
                label="End",
                zorder=5,
            )

            ax3.set_xlabel("Component 0")
            ax3.set_ylabel("Component 1")
            ax3.set_title(f"{method.upper()} - Trajectory")
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc="best", fontsize=8)
            ax3.set_aspect("equal", adjustable="datalim")

    # Set main title
    title = "Population Embeddings: Behavioral Features"
    if with_trajectory:
        title += " and Trajectories"
    plt.suptitle(title, fontsize=16)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_trajectories(
    embeddings: Dict[str, np.ndarray],
    methods: Optional[List[str]] = None,
    trajectory_kwargs: Optional[Dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> plt.Figure:
    """
    Create figure showing trajectories in embedding space for multiple methods.

    Parameters
    ----------
    embeddings : dict
        Dictionary mapping method names to embedding arrays. Arrays must be 2D
        with at least 2 components.
    methods : list of str, optional
        List of methods to plot (if None, uses all keys in embeddings)
    trajectory_kwargs : dict, optional
        Keyword arguments for trajectory plotting
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save the figure
    dpi : int, default DEFAULT_DPI
        DPI resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    
    Raises
    ------
    ValueError
        If embeddings are not 2D arrays with at least 2 components    """
    # Validate embeddings
    for method, embedding in embeddings.items():
        if embedding.ndim != 2:
            raise ValueError(f"Embedding for {method} must be 2D, got shape {embedding.shape}")
        if embedding.shape[1] < 2:
            raise ValueError(f"Embedding for {method} must have at least 2 components, got {embedding.shape[1]}")
        if len(embedding) == 0:
            raise ValueError(f"Embedding for {method} is empty")
    
    if methods is None:
        methods = list(embeddings.keys())

    n_methods = len(methods)

    if figsize is None:
        figsize = (6 * n_methods, 5)

    fig = plt.figure(figsize=figsize)

    # Default trajectory kwargs
    if trajectory_kwargs is None:
        trajectory_kwargs = {}

    default_kwargs = {
        "linewidth": 0.8,
        "alpha": 0.3,
        "color": "k",
        "arrow_spacing": 20,
        "arrow_scale": 0.3,
        "start_marker": "o",
        "end_marker": "s",
        "marker_size": 100,
    }
    default_kwargs.update(trajectory_kwargs)

    for i, method in enumerate(methods):
        if method not in embeddings:
            continue

        embedding = embeddings[method]
        ax = fig.add_subplot(1, n_methods, i + 1)

        # Plot trajectory
        ax.plot(
            embedding[:, 0],
            embedding[:, 1],
            color=default_kwargs["color"],
            alpha=default_kwargs["alpha"],
            linewidth=default_kwargs["linewidth"],
        )

        # Add direction arrows
        trajectory_samples = len(embedding)
        arrow_spacing = max(1, trajectory_samples // default_kwargs["arrow_spacing"])

        for j in range(0, trajectory_samples - arrow_spacing, arrow_spacing):
            if j + 1 < trajectory_samples:
                dx = embedding[j + 1, 0] - embedding[j, 0]
                dy = embedding[j + 1, 1] - embedding[j, 1]

                if np.sqrt(dx**2 + dy**2) > 0.001:
                    # Scale arrow size with data range
                    x_range = embedding[:, 0].max() - embedding[:, 0].min()
                    y_range = embedding[:, 1].max() - embedding[:, 1].min()
                    arrow_scale = min(x_range, y_range) * 0.01
                    
                    ax.arrow(
                        embedding[j, 0],
                        embedding[j, 1],
                        dx * default_kwargs["arrow_scale"],
                        dy * default_kwargs["arrow_scale"],
                        head_width=arrow_scale,
                        head_length=arrow_scale,
                        fc="red",
                        ec="red",
                        alpha=0.6,
                    )

        # Mark start and end
        ax.scatter(
            embedding[0, 0],
            embedding[0, 1],
            c="green",
            s=default_kwargs["marker_size"],
            marker=default_kwargs["start_marker"],
            edgecolors="black",
            linewidth=2,
            label="Start",
            zorder=5,
        )
        ax.scatter(
            embedding[-1, 0],
            embedding[-1, 1],
            c="red",
            s=default_kwargs["marker_size"],
            marker=default_kwargs["end_marker"],
            edgecolors="black",
            linewidth=2,
            label="End",
            zorder=5,
        )

        ax.set_xlabel("Component 0")
        ax.set_ylabel("Component 1")
        ax.set_title(f"{method.upper()} - Trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

    plt.suptitle("Temporal Trajectories in Embedding Space", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_component_interpretation(
    mi_matrices: Dict[str, np.ndarray],
    feature_names: List[str],
    methods: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    metadata: Optional[Dict[str, Dict]] = None,
    compute_metrics: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> plt.Figure:
    """
    Create figure showing mutual information between embedding components and features.

    Parameters
    ----------
    mi_matrices : dict
        Dictionary mapping method names to MI matrices (n_features, n_components).
        MI values should be non-negative.
    feature_names : list of str
        Names of features for y-axis labels
    methods : list of str, optional
        List of methods to plot (if None, uses all keys in mi_matrices)
    n_components : int, optional
        Number of components to show (default: min 5 or available)
    metadata : dict, optional
        Dictionary of metadata for each method (e.g., explained variance for PCA)
    compute_metrics : bool, default True
        Whether to show additional metrics (e.g., explained variance)
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save the figure
    dpi : int, default DEFAULT_DPI
        DPI resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    
    Raises
    ------
    ValueError
        If MI matrices are not 2D or contain negative values    """
    # Validate MI matrices
    for method, mi_matrix in mi_matrices.items():
        if mi_matrix.ndim != 2:
            raise ValueError(f"MI matrix for {method} must be 2D, got shape {mi_matrix.shape}")
        if len(feature_names) != mi_matrix.shape[0]:
            raise ValueError(f"MI matrix for {method} has {mi_matrix.shape[0]} features but "
                           f"feature_names has {len(feature_names)} names")
        if np.any(mi_matrix < 0):
            raise ValueError(f"MI matrix for {method} contains negative values")
    
    if methods is None:
        methods = list(mi_matrices.keys())

    n_methods = len(methods)

    if figsize is None:
        figsize = (8 * n_methods, 6)

    fig = plt.figure(figsize=figsize)

    for idx, method in enumerate(methods):
        if method not in mi_matrices:
            continue

        mi_matrix = mi_matrices[method]

        # Determine number of components to show
        n_comp_available = mi_matrix.shape[1]
        if n_components is None:
            n_comp_show = min(5, n_comp_available)
        else:
            n_comp_show = min(n_components, n_comp_available)

        # Create subplot
        ax = plt.subplot(1, n_methods, idx + 1)

        # Plot MI heatmap
        mi_subset = mi_matrix[:, :n_comp_show]
        max_mi = np.max(mi_subset) if np.max(mi_subset) > 0 else 1

        im = ax.imshow(mi_subset, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max_mi)

        # Set labels
        ax.set_xticks(range(n_comp_show))

        # Create component labels based on method
        if method.lower() == "pca":
            comp_labels = [f"PC{i}" for i in range(n_comp_show)]
        elif method.lower() == "umap":
            comp_labels = [f"UMAP{i}" for i in range(n_comp_show)]
        elif method.lower() == "le":
            comp_labels = [f"LE{i}" for i in range(n_comp_show)]
        else:
            comp_labels = [f"{method.upper()}{i}" for i in range(n_comp_show)]

        ax.set_xticklabels(comp_labels)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel(f"{method.upper()} Components")
        ax.set_title(f"{method.upper()} Component-Feature MI")

        # Add MI values on cells
        for i in range(len(feature_names)):
            for j in range(n_comp_show):
                text_color = "black" if mi_subset[i, j] < max_mi * 0.5 else "white"
                ax.text(
                    j,
                    i,
                    f"{mi_subset[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label="Mean MI (bits)")

        # Add method-specific metrics if available
        if compute_metrics and metadata is not None and method in metadata:
            method_meta = metadata[method]

            # For PCA, show explained variance
            if method.lower() == "pca" and "explained_variance_ratio" in method_meta:
                var_exp = method_meta["explained_variance_ratio"][:n_comp_show]
                var_text = "Var explained: " + ", ".join(
                    [f"{v*100:.1f}%" for v in var_exp]
                )
                ax.text(
                    0.5,
                    -0.15,
                    var_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                    style="italic",
                )

    plt.suptitle(
        "Component Interpretation: Mutual Information between Components and Features",
        fontsize=16,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_embeddings_grid(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    methods: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    colormap: str = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    n_cols: int = 4,
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> Optional[plt.Figure]:
    """
    Create grid of embeddings for multiple methods and scenarios.

    Parameters
    ----------
    embeddings : dict of dict
        Nested dictionary: {method: {scenario: embedding_array}}. Arrays must be
        2D with at least 2 components.
    labels : array or dict, optional
        Color labels for points. Can be array (same for all) or dict matching structure
    methods : list, optional
        Methods to plot (default: all in embeddings)
    scenarios : list, optional
        Scenarios to plot (default: all available)
    metrics : dict, optional
        Nested dict of metrics: {method: {scenario: {metric_name: value}}}.
        At most 2 metrics shown per subplot.
    colormap : str
        Colormap for scatter plots
    figsize : tuple, optional
        Figure size
    n_cols : int
        Number of columns in grid
    save_path : str, optional
        Path to save figure
    dpi : int, default DEFAULT_DPI
        DPI resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The generated figure, or None if no valid embeddings to plot
    
    Raises
    ------
    ValueError
        If embeddings are not 2D or label lengths mismatch    """
    if methods is None:
        methods = list(embeddings.keys())

    # Collect all scenario-method pairs
    all_plots = []
    for method in methods:
        if method not in embeddings:
            continue
        if scenarios is None:
            method_scenarios = list(embeddings[method].keys())
        else:
            method_scenarios = [s for s in scenarios if s in embeddings[method]]

        for scenario in method_scenarios:
            if embeddings[method][scenario] is not None:
                embedding = embeddings[method][scenario]
                if embedding.ndim != 2 or embedding.shape[1] < 2:
                    raise ValueError(f"Embedding for {method}/{scenario} must be 2D with "
                                   f"at least 2 components, got shape {embedding.shape}")
                all_plots.append((method, scenario))

    if not all_plots:
        return None

    # Calculate grid dimensions
    n_plots = len(all_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each embedding
    for idx, (method, scenario) in enumerate(all_plots):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        embedding = embeddings[method][scenario]

        # Get labels for coloring
        if labels is None:
            color_labels = np.arange(len(embedding))
        elif isinstance(labels, dict):
            if method in labels and scenario in labels[method]:
                color_labels = labels[method][scenario]
                if len(color_labels) != len(embedding):
                    raise ValueError(f"Labels for {method}/{scenario} have length {len(color_labels)} "
                                   f"but embedding has {len(embedding)} samples")
            else:
                color_labels = np.arange(len(embedding))
        else:
            color_labels = labels
            if len(color_labels) != len(embedding):
                raise ValueError(f"Labels have length {len(color_labels)} but embedding "
                               f"for {method}/{scenario} has {len(embedding)} samples")

        # Create scatter plot
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=color_labels,
            cmap=colormap,
            s=10,
            alpha=0.7,
            edgecolors="none",
        )

        # Add title with metrics if available
        title = f"{method} - {scenario}"
        if metrics and method in metrics and scenario in metrics[method]:
            metric_strs = []
            for metric_name, value in metrics[method][scenario].items():
                if isinstance(value, float):
                    metric_strs.append(f"{metric_name}: {value:.3f}")
            if metric_strs:
                title += "\n" + ", ".join(metric_strs[:2])  # Show max 2 metrics

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Component 0")
        ax.set_ylabel("Component 1")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


# Note: plot_quality_metrics_comparison, plot_quality_vs_speed_tradeoff
# are removed as they are too specific to certain examples and not reused elsewhere


def plot_neuron_selectivity_summary(
    selectivity_counts: Dict[str, int],
    total_neurons: int,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> plt.Figure:
    """
    Create bar plot summarizing neuron selectivity categories.

    Parameters
    ----------
    selectivity_counts : dict
        Dictionary mapping category names to counts. Counts should be non-negative
        integers with sum <= total_neurons.
    total_neurons : int
        Total number of neurons. Must be positive.
    colors : dict, optional
        Dictionary mapping category names to colors
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    dpi : int, default DEFAULT_DPI
        DPI resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    
    Raises
    ------
    ValueError
        If total_neurons <= 0 or counts are invalid    """
    # Validate inputs using utility functions
    check_positive(total_neurons=total_neurons)
    
    # Validate counts
    for category, count in selectivity_counts.items():
        if not isinstance(count, (int, np.integer)):
            raise ValueError(f"Count for category '{category}' must be an integer, got {type(count).__name__}")
    
    # Check all counts are non-negative
    check_nonnegative(**selectivity_counts)
    
    # Check sum doesn't exceed total
    total_count = sum(selectivity_counts.values())
    if total_count > total_neurons:
        raise ValueError(f"Sum of counts ({total_count}) exceeds total_neurons ({total_neurons})")
    
    if colors is None:
        # Default colors for common categories
        colors = {
            "Spatial": "darkgreen",
            "spatial": "darkgreen",
            "position_2d": "darkgreen",
            "x_position": "green",
            "y_position": "lightgreen",
            "head_direction": "blue",
            "speed": "orange",
            "task_type": "red",
            "reward": "purple",
            "Non-spatial": "gray",
            "non_spatial": "gray",
            "Non-selective": "lightgray",
        }

    fig, ax = plt.subplots(figsize=figsize)

    categories = list(selectivity_counts.keys())
    counts = list(selectivity_counts.values())

    # Get colors for each category
    bar_colors = [colors.get(cat, "steelblue") for cat in categories]

    # Create bars
    bars = ax.bar(categories, counts, color=bar_colors, alpha=0.7)

    # Add percentage labels
    for bar, count in zip(bars, counts):
        percentage = count / total_neurons * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Number of neurons")
    ax.set_title("Neuron Selectivity Categories")
    ax.set_ylim(0, max(counts) * 1.15)

    # Add total count as text
    ax.text(
        0.02,
        0.98,
        f"Total neurons: {total_neurons}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_component_selectivity_heatmap(
    selectivity_matrix: np.ndarray,
    methods: List[str],
    n_components_per_method: Optional[Dict[str, int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> plt.Figure:
    """
    Create heatmap showing neuron selectivity to embedding components.

    Parameters
    ----------
    selectivity_matrix : ndarray
        Matrix of shape (n_neurons, total_components) with MI values.
        Must be 2D with non-negative values.
    methods : list of str
        List of DR method names. Cannot be empty.
    n_components_per_method : dict, optional
        Number of components for each method. If None, assumes equal
        distribution across methods.
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    dpi : int, default DEFAULT_DPI
        DPI resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    
    Raises
    ------
    ValueError
        If selectivity_matrix is not 2D, contains negative values,
        methods list is empty, or component counts don't match matrix    """
    # Validate inputs
    if selectivity_matrix.ndim != 2:
        raise ValueError(f"selectivity_matrix must be 2D, got shape {selectivity_matrix.shape}")
    
    if len(methods) == 0:
        raise ValueError("methods list cannot be empty")
    
    # Check for non-negative values
    if np.any(selectivity_matrix < 0):
        raise ValueError("selectivity_matrix cannot contain negative values")
    
    # Check for NaN or inf
    if np.any(~np.isfinite(selectivity_matrix)):
        raise ValueError("selectivity_matrix contains NaN or infinite values")
    n_neurons, total_components = selectivity_matrix.shape

    if n_components_per_method is None:
        # Assume equal components per method
        n_methods = len(methods)
        if total_components % n_methods != 0:
            raise ValueError(f"Total components ({total_components}) not evenly divisible by number of methods ({n_methods})")
        n_comp_each = total_components // n_methods
        n_components_per_method = {m: n_comp_each for m in methods}
    else:
        # Validate component counts match matrix
        total_specified = sum(n_components_per_method.values())
        if total_specified != total_components:
            raise ValueError(f"Sum of components ({total_specified}) doesn't match matrix columns ({total_components})")
        
        # Ensure all methods have component counts
        for method in methods:
            if method not in n_components_per_method:
                raise ValueError(f"Missing component count for method '{method}'")
            check_positive(**{f"{method}_components": n_components_per_method[method]})

    if figsize is None:
        figsize = (5 * len(methods), 8)

    fig, axes = plt.subplots(1, len(methods), figsize=figsize)
    if len(methods) == 1:
        axes = [axes]

    comp_start = 0
    for ax, method in zip(axes, methods):
        n_comp = n_components_per_method[method]

        # Extract subset for this method
        method_matrix = selectivity_matrix[:, comp_start : comp_start + n_comp]

        # Plot heatmap
        im = ax.imshow(
            method_matrix.T, aspect="auto", cmap="hot", interpolation="nearest"
        )

        ax.set_xlabel("Neuron ID")
        ax.set_ylabel("Component")
        ax.set_title(f"{method.upper()} Component Selectivity")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mutual Information (bits)")

        # Set component labels
        ax.set_yticks(range(n_comp))
        ax.set_yticklabels([f"C{i}" for i in range(n_comp)])

        comp_start += n_comp

    plt.suptitle("Neuron Selectivity to Embedding Components", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
