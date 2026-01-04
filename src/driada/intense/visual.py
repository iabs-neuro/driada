import numpy as np

import matplotlib.pyplot as plt
from ..utils.plot import create_default_figure, make_beautiful
from ..utils.data import rescale
from scipy.stats import rankdata, gaussian_kde, wasserstein_distance
import seaborn as sns


def plot_pc_activity(exp, cell_ind, place_key=("x", "y"), ds=5, ax=None, 
                    show_trajectory=False, show_spikes=True, cmap="plasma",
                    marker_size=100, marker_style="*", marker_color="k",
                    scatter_alpha=0.8, trajectory_alpha=0.3, trajectory_color="gray",
                    figsize_base=6, title_format="Cell {cell_ind}, Rel MI={rel_mi:.4f}, pval={pval:.2e}",
                    xlabel=None, ylabel=None, show_stats=True):
    """
    Plot place cell activity overlaid on spatial trajectory.

    Parameters
    ----------
    exp : Experiment
        Experiment object with spatial data and neurons.
    cell_ind : int
        Index of the neuron to plot.
    place_key : tuple or str, optional
        Feature key for spatial data. Default: ("x", "y").
        Can be tuple like ("x", "y") or string like "position".
    ds : int, optional
        Downsampling factor. Default: 5.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_trajectory : bool, optional
        Whether to show trajectory line. Default: False.
    show_spikes : bool, optional
        Whether to show spike markers. Default: True.
    cmap : str, optional
        Colormap for activity. Default: "plasma".
    marker_size : int, optional
        Size of spike markers. Default: 100.
    marker_style : str, optional
        Marker style for spikes. Default: "*".
    marker_color : str, optional
        Color for spike markers. Default: "k".
    scatter_alpha : float, optional
        Alpha for activity scatter. Default: 0.8.
    trajectory_alpha : float, optional
        Alpha for trajectory line. Default: 0.3.
    trajectory_color : str, optional
        Color for trajectory. Default: "gray".
    figsize_base : float, optional
        Base figure size (adjusted by aspect ratio). Default: 6.
    title_format : str, optional
        Format string for title. Available keys: cell_ind, rel_mi, pval.
        Default: "Cell {cell_ind}, Rel MI={rel_mi:.4f}, pval={pval:.2e}"
    xlabel : str, optional
        X-axis label. Default: first element of place_key or "x".
    ylabel : str, optional
        Y-axis label. Default: second element of place_key or "y".
    show_stats : bool, optional
        Whether to show statistics in title. Default: True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    
    Raises
    ------
    KeyError
        If cell_ind or place_key not found in stats_table
    IndexError
        If cell_ind >= number of neurons
    ValueError
        If place data is not 2D
    AttributeError
        If required attributes missing from experiment
    
    Notes
    -----
    - Uses log-transformed calcium data for color mapping
    - Figure aspect ratio automatically adjusted based on spatial extent
    - Stats (MI and p-value) retrieved from experiment's stats_table
    - Supports both tuple place keys like ("x", "y") and single feature names
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from driada.experiment import load_demo_experiment
    >>> 
    >>> # Basic place cell plot with position data
    >>> exp = load_demo_experiment(verbose=False)
    >>> # Use x_pos and y_pos which are available in demo data
    >>> # Note: Demo data doesn't have spike data, so we disable spike display
    >>> ax = plot_pc_activity(exp, cell_ind=5, place_key=("x_pos", "y_pos"),
    ...                      show_spikes=False, show_stats=False)
    >>> plt.close()  # Suppress display
    >>> 
    >>> # Using separate x,y features with custom styling
    >>> ax = plot_pc_activity(exp, cell_ind=10, place_key=("x_pos", "y_pos"),
    ...                      show_trajectory=True, cmap='viridis', 
    ...                      marker_color='red', show_stats=False, 
    ...                      show_spikes=False, ds=20)
    >>> plt.close()
    >>> 
    >>> # Custom styling example
    >>> ax = plot_pc_activity(exp, cell_ind=3, place_key=("x_pos", "y_pos"),
    ...                      ds=10, show_stats=False, show_spikes=False)
    >>> plt.close()
    """
    # Validate inputs
    if cell_ind < 0 or cell_ind >= exp.n_cells:
        raise IndexError(f"cell_ind {cell_ind} out of range [0, {exp.n_cells})")
    
    # Get spatial data
    if isinstance(place_key, tuple) and len(place_key) == 2:
        x_key, y_key = place_key
        x_data = getattr(exp, x_key).data
        y_data = getattr(exp, y_key).data
        default_xlabel = x_key
        default_ylabel = y_key
    else:
        # Single feature - try to get x,y components
        place_feat = getattr(exp, place_key)
        if hasattr(place_feat, 'data') and place_feat.data.ndim == 2:
            x_data = place_feat.data[:, 0]
            y_data = place_feat.data[:, 1]
            default_xlabel = f"{place_key}_x"
            default_ylabel = f"{place_key}_y"
        else:
            raise ValueError(f"Place feature {place_key} must be 2D or provide tuple of features")
    
    # Get statistics if available
    if show_stats and hasattr(exp, 'stats_table') and place_key in exp.stats_table:
        try:
            pc_stats = exp.stats_table[place_key][cell_ind]
            pval = pc_stats.get("pval", None)
            rel_mi = pc_stats.get("rel_me_beh", pc_stats.get("rel_mi_beh", None))
        except (KeyError, IndexError):
            pval = None
            rel_mi = None
            show_stats = False
    else:
        show_stats = False

    # Create figure if needed
    if ax is None:
        lenx = max(x_data) - min(x_data)
        leny = max(y_data) - min(y_data)
        xyratio = max(lenx / leny, leny / lenx)
        fig, ax = create_default_figure(figsize=(figsize_base * xyratio, figsize_base))

    # Get neural activity
    neur = rescale(np.log(exp.neurons[cell_ind].ca.data + 1e-10))
    
    # Plot trajectory if requested
    if show_trajectory:
        ax.plot(x_data[::ds], y_data[::ds], c=trajectory_color, 
                alpha=trajectory_alpha, zorder=1)
    
    # Plot activity
    ax.scatter(x_data[::ds], y_data[::ds], c=neur[::ds], 
               cmap=cmap, alpha=scatter_alpha, zorder=2)
    
    # Plot spikes if requested
    if show_spikes and hasattr(exp.neurons[cell_ind], 'sp'):
        spinds = np.where(exp.neurons[cell_ind].sp.data != 0)[0]
        if len(spinds) > 0:
            ax.scatter(x_data[spinds], y_data[spinds], 
                      c=marker_color, alpha=1, marker=marker_style,
                      linewidth=2, s=marker_size, zorder=3)
    
    # Labels
    ax.set_xlabel(xlabel or default_xlabel)
    ax.set_ylabel(ylabel or default_ylabel)
    
    # Title
    if show_stats and pval is not None and rel_mi is not None:
        title = title_format.format(cell_ind=cell_ind, rel_mi=rel_mi, pval=pval)
    else:
        title = f"Cell {cell_ind}"
    ax.set_title(title)

    return ax


def plot_neuron_feature_density(
    exp,
    data_type,
    cell_id,
    featname,
    ind1=0,
    ind2=100000,
    ds=1,
    shift=None,
    ax=None,
    compute_wsd=False,
):
    """
    Plot density distribution of neural activity conditioned on feature values.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing neurons and features.
    data_type : str
        Type of neural data: 'calcium' or 'spikes'.
    cell_id : int
        Index of the neuron.
    featname : str
        Name of the behavioral feature.
    ind1 : int, optional
        Start frame index. Default: 0.
    ind2 : int, optional
        End frame index. Default: 100000.
    ds : int, optional
        Downsampling factor. Default: 1.
    shift : int, optional
        Temporal shift in frames. Currently not implemented. Default: None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    compute_wsd : bool, optional
        Whether to compute Wasserstein distance for binary features. Default: False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    
    Raises
    ------
    NotImplementedError
        If data_type='spikes' with binary feature
    IndexError
        If cell_id >= number of neurons
    AttributeError
        If feature not found in experiment
    
    Notes
    -----
    - Binary features: Uses KDE with bw_adjust=0.5, log10 transform
    - Continuous features: Adds 1e-8 noise, uses 100x100 grid
    - Uses .scdata attribute for scaled data access
    - shift parameter is accepted but not used
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from driada.experiment import load_demo_experiment
    >>> 
    >>> # Plot calcium vs feature density
    >>> exp = load_demo_experiment(verbose=False)
    >>> # Using 'speed' feature which is available in demo data
    >>> ax = plot_neuron_feature_density(exp, 'calcium', 5, 'speed')
    >>> plt.close()  # Suppress display
    >>> 
    >>> # For binary features (if available in your data)
    >>> # ax = plot_neuron_feature_density(exp, 'calcium', 10, 'licking', 
    >>> #                                 compute_wsd=True)
    """
    ind2 = min(exp.n_frames, ind2)

    if data_type == "calcium":
        sig = exp.neurons[cell_id].ca.scdata[ind1:ind2][::ds]
    if data_type == "spikes":
        sig = exp.neurons[cell_id].sp.scdata[ind1:ind2][::ds]

    feature = getattr(exp, featname)
    bdata = feature.scdata[ind1:ind2][::ds]
    rbdata = rescale(rankdata(bdata))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if feature.is_binary:
        if data_type == "calcium":
            vals0 = np.log10(sig[np.where((rbdata == min(rbdata)) & (sig > 0))])
            vals1 = np.log10(sig[np.where((rbdata == max(rbdata)) & (sig > 0))])

            if compute_wsd and len(vals0) > 0 and len(vals1) > 0:
                wsd = wasserstein_distance(vals0, vals1)
                title_text = f"wsd={wsd:.3f}"
            else:
                title_text = ""

            _ = sns.kdeplot(
                vals0, ax=ax, c="b", label=f"{featname}=0", linewidth=3, bw_adjust=0.5
            )
            _ = sns.kdeplot(
                vals1, ax=ax, c="r", label=f"{featname}=1", linewidth=3, bw_adjust=0.5
            )
            ax.legend(loc="upper right")
            ax.set_xlabel("log(dF/F)", fontsize=20)
            ax.set_ylabel("density", fontsize=20)
            if title_text:
                ax.set_title(title_text)

        if data_type == "spikes":
            raise NotImplementedError(
                "Binary feature density plot for spike data not yet implemented"
            )

    else:
        x0, y0 = np.log10(sig + np.random.random(size=len(sig)) * 1e-8), np.log(
            bdata + np.random.random(size=len(bdata)) * 1e-8
        )

        jdata = np.vstack([x0, y0]).T
        # jplot = sns.jointplot(jdata, x=jdata[:,0], y=jdata[:,1], kind='hist', bins=100)
        nbins = 100
        k = gaussian_kde(jdata.T)
        xi, yi = np.mgrid[
            x0.min() : x0.max() : nbins * 1j, y0.min() : y0.max() : nbins * 1j
        ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # plot a density
        ax.set_title("Density")
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="coolwarm")
        ax.set_xlabel("log(signals)", fontsize=20)
        ax.set_ylabel(f"log({featname})", fontsize=20)

    return ax


def plot_shadowed_groups(ax, xvals, binary_series, color='gray', alpha=0.3, label='shadowed'):
    """
    Shade regions where binary series equals 1.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    xvals : array-like
        X-axis values corresponding to binary_series.
    binary_series : array-like
        Binary array (0s and 1s) indicating regions to shade.
    color : str, optional
        Color for shaded regions. Default: 'gray'.
    alpha : float, optional
        Transparency of shaded regions. Default: 0.3.
    label : str, optional
        Label for legend. Default: 'shadowed'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Modified axes object.
    """
    x = np.arange(len(binary_series))

    # Find and shadow groups of 1s
    i = 0
    n = len(binary_series)
    labelled = False
    while i < n:
        if binary_series[i] == 1:
            x_min = xvals[i]  # Start of group
            while i < n and binary_series[i] == 1:
                i += 1
            x_max = xvals[i - 1] if i > 0 else xvals[0]  # End of group
            # Shadow the region
            ax.axvspan(x_min, x_max + 1, alpha=alpha,
                        color=color, label=label if not labelled else "")
            labelled = True
        else:
            i += 1

    return ax


def plot_neuron_feature_pair(
    exp,
    cell_id,
    featname,
    ind1=0,
    ind2=100000,
    ds=1,
    add_density_plot=True,
    ax=None,
    title=None,
    bcolor='g',
    neuron_label=None,
    feature_label=None,
    non_feature_label=None
):
    """
    Plot neural activity alongside behavioral feature with density analysis.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing neurons and features.
    cell_id : int
        Index of the neuron.
    featname : str
        Name of the behavioral feature.
    ind1 : int, optional
        Start frame index. Default: 0.
    ind2 : int, optional
        End frame index. Default: 100000.
    ds : int, optional
        Downsampling factor. Default: 1.
    add_density_plot : bool, optional
        Whether to add density subplot. Default: True.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Forces add_density_plot=False when provided.
    title : str, optional
        Custom title for the plot.
    bcolor : str, optional
        Color for feature visualization. Default: 'g'.
    neuron_label : str, optional
        Custom label for neuron. Default: f'neuron {cell_id}'.
    feature_label : str, optional
        Custom label for feature. Default: featname.
    non_feature_label : str, optional
        Custom label for non-feature state. Default: f'non-{featname}'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot(s).

    Raises
    ------
    IndexError
        If cell_id >= number of neurons
    AttributeError
        If featname not found in experiment

    Notes
    -----
    - Discrete features shown as shaded regions where active
    - Uses make_beautiful() for axis styling with legends below
    - Y-axis formatted to 1 decimal place
    - Dark gray for non-feature distribution in density plot
    - Calls plt.tight_layout() with bottom adjustment for legends

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from driada.experiment import load_demo_experiment
    >>>
    >>> # Basic time series plot
    >>> exp = load_demo_experiment(verbose=False)
    >>> fig = plot_neuron_feature_pair(exp, 5, 'speed')
    >>> plt.close(fig)  # Suppress display
    >>>
    >>> # With custom labels
    >>> fig = plot_neuron_feature_pair(exp, 10, 'object1',
    ...                               feature_label='Object Interaction',
    ...                               non_feature_label='No Object')
    >>> plt.close(fig)
    """

    ind2 = min(exp.n_frames, ind2)
    ca = exp.neurons[cell_id].ca.scdata[ind1:ind2][::ds]
    rca = rescale(rankdata(ca))
    feature = getattr(exp, featname)
    bdata = feature.scdata[ind1:ind2][::ds]
    rbdata = rescale(rankdata(bdata))

    # Set default labels if not provided
    if neuron_label is None:
        neuron_label = f'neuron {cell_id}'
    if feature_label is None:
        feature_label = featname
    if non_feature_label is None:
        non_feature_label = f'non-{featname}'

    if ax is None:
        if add_density_plot:
            fig, axs = plt.subplots(1, 2, figsize=(20, 6), width_ratios=[0.7, 0.3], dpi=300)
            ax0, ax1 = axs
            ax1 = make_beautiful(ax1, legend_loc='below', legend_offset=0.25, legend_ncol=1)
        else:
            fig, ax0 = plt.subplots(figsize=(20, 6), dpi=300)
            ax1 = None
    else:
        ax0 = ax
        ax1 = None
        add_density_plot = False
        fig = ax0.figure

    xvals = np.arange(ind1, ind2)[::ds]/20.0

    ax0.plot(xvals, ca, c='b', linewidth=3, alpha=0.6, label=neuron_label)
    if feature.discrete:
        ax0 = plot_shadowed_groups(ax0, xvals,
                                feature.scdata[ind1:ind2][::ds], color=bcolor, alpha=0.5,
                                label=feature_label)
    else:
        ax0.plot(xvals, rbdata, c='r', linewidth=2, alpha=0.5, label=feature_label)

    # Changed ncol from 2 to 1 to stack labels vertically
    ax0.legend(fontsize=24, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    ax0.set_xlabel('time, s', fontsize=30)
    ax0.set_ylabel('signal', fontsize=30)

    if title is None:
        title = f'{exp.signature} Neuron {cell_id}, feature {featname}'

    # apply styling with ncol=1 for vertical stacking
    ax0 = make_beautiful(ax0, legend_loc='below', legend_offset=0.25, legend_fontsize=24,
                        legend_ncol=1)

    # Format y-axis tick labels to 1 decimal place with proper rounding
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{round(x, 6):.1f}'))

    if add_density_plot:
        if feature.discrete:
            vals0 = np.log10(ca[np.where((rbdata == min(rbdata)) & (ca > 0))] + 1e-10)
            vals1 = np.log10(ca[np.where((rbdata == max(rbdata)) & (ca > 0))] + 1e-10)
            if len(vals0) > 0 and len(vals1) > 0:
                wsd = wasserstein_distance(vals0, vals1)
                _ = sns.kdeplot(vals0, ax=ax1, c='dimgray', label=non_feature_label, linewidth=5,
                            bw_adjust=0.1)
                _ = sns.kdeplot(vals1, ax=ax1, c=bcolor, label=feature_label, linewidth=5,
                            bw_adjust=0.1)

                ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=20, ncol=1,
                        frameon=False)
                ax1.set_xlabel(r'$\log$(signal)', fontsize=30)
                ax1.set_ylabel('density', fontsize=30)

                # apply styling with ncol=1
                ax1 = make_beautiful(ax1, legend_loc='below', legend_offset=0.25, legend_fontsize=24,
                                    legend_ncol=1)

                ax1.set_xlim(-4.0, 0.5)
                # Format y-axis tick labels to 1 decimal place for density plot
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{round(x, 6):.1f}'))

        else:
            x0 = np.log10(ca + np.random.random(size=len(ca)) * 1e-8)
            y0 = np.log(bdata + np.random.random(size=len(bdata)) * 1e-8)

            jdata = np.vstack([x0, y0]).T
            nbins = 100
            k = gaussian_kde(jdata.T)
            xi, yi = np.mgrid[x0.min():x0.max():nbins * 1j, y0.min():y0.max():nbins * 1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            # plot a density
            ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='coolwarm')
            ax1.set_xlabel(r'$\log$(signal)', fontsize=30)
            ax1.set_ylabel(fr'$\log$({featname})', fontsize=30)

            # apply styling with ncol=1
            ax1 = make_beautiful(ax1, legend_loc='below', legend_offset=0.25, legend_fontsize=24,
                                legend_ncol=1)

            # Format y-axis tick labels to 1 decimal place for density plot
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

    plt.tight_layout()
    # Add extra space at bottom for legends
    plt.subplots_adjust(bottom=0.15)

    return fig


def plot_disentanglement_heatmap(
    disent_matrix,
    count_matrix,
    feat_names,
    title=None,
    figsize=(12, 10),
    dpi=100,
    cmap=None,
    vmin=0,
    vmax=100,
    cbar_label="Disentanglement score (%)",
    fontsize=14,
    title_fontsize=18,
    show_grid=True,
    grid_alpha=0.3,
):
    """Plot disentanglement analysis results as a heatmap.

    Creates a heatmap showing the relative disentanglement scores between
    feature pairs. Each cell (i,j) shows the percentage of neurons where
    feature i was primary when paired with feature j.

    Parameters
    ----------
    disent_matrix : ndarray
        Disentanglement matrix from disentangle_all_selectivities.
    count_matrix : ndarray
        Count matrix from disentangle_all_selectivities.
    feat_names : list of str
        Feature names corresponding to matrix indices.
    title : str, optional
        Plot title. Default: 'Disentanglement Analysis'.
    figsize : tuple, optional
        Figure size (width, height). Default: (12, 10).
    dpi : int, optional
        Figure DPI. Default: 100.
    cmap : str or Colormap, optional
        Colormap to use. Default: custom red-white-green gradient.
    vmin : float, optional
        Minimum value for colormap. Default: 0.
    vmax : float, optional
        Maximum value for colormap. Default: 100.
    cbar_label : str, optional
        Colorbar label. Default: 'Disentanglement score (%)'.
    fontsize : int, optional
        Font size for tick labels. Default: 14.
    title_fontsize : int, optional
        Font size for title. Default: 18.
    show_grid : bool, optional
        Whether to show grid lines. Default: True.
    grid_alpha : float, optional
        Grid transparency. Default: 0.3.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the heatmap.
    ax : matplotlib.axes.Axes
        Axes containing the heatmap.

    Raises
    ------
    ImportError
        If seaborn, pandas, or matplotlib.colors not available
    ValueError
        If matrix dimensions don't match or feat_names length doesn't match matrices
    
    Notes
    -----
    The heatmap uses a diverging colormap where:
    - Red indicates low disentanglement (feature is redundant)
    - Gray (0.7, 0.7, 0.7) indicates balanced contribution (~50%)
    - Green indicates high disentanglement (feature is primary)

    Cells are masked (shown in white) where no data is available.
    Uses pandas DataFrame internally for seaborn compatibility.
    Calls plt.tight_layout() which affects figure state.
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Create synthetic data for demonstration
    >>> n_features = 4
    >>> features = ['speed', 'position', 'direction', 'licking']
    >>> 
    >>> # Create synthetic matrices
    >>> # disent_matrix[i,j] = how many times feature i was primary vs j
    >>> disent_mat = np.array([
    ...     [0, 15, 8, 20],
    ...     [5, 0, 12, 18],
    ...     [12, 8, 0, 10],
    ...     [10, 7, 15, 0]
    ... ])
    >>> 
    >>> # count_matrix[i,j] = total comparisons between features i and j
    >>> count_mat = np.array([
    ...     [0, 20, 20, 30],
    ...     [20, 0, 20, 25],
    ...     [20, 20, 0, 25],
    ...     [30, 25, 25, 0]
    ... ])
    >>> 
    >>> # Basic heatmap
    >>> fig, ax = plot_disentanglement_heatmap(disent_mat, count_mat, features)
    >>> plt.close(fig)  # Suppress display
    >>> 
    >>> # Custom styling
    >>> fig, ax = plot_disentanglement_heatmap(
    ...     disent_mat, count_mat, features,
    ...     title="My Analysis", cmap='RdYlGn',
    ...     figsize=(8, 6), dpi=150
    ... )
    >>> plt.close(fig)
    """
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    import pandas as pd

    # Calculate relative disentanglement matrix (as percentage)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_disent_matrix = np.divide(disent_matrix, count_matrix) * 100
        rel_disent_matrix[count_matrix == 0] = np.nan

    # Create default colormap if not provided
    if cmap is None:
        # Red -> Gray -> Green gradient
        # Gray at 50% represents equal selectivity (no disentanglement)
        colors = [(1, 0, 0), (0.7, 0.7, 0.7), (0, 1, 0)]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list(
            "disentanglement_cmap", colors, N=n_bins
        )

    # Create DataFrame for seaborn
    df_heatmap = pd.DataFrame(rel_disent_matrix, columns=feat_names, index=feat_names)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap
    sns.heatmap(
        df_heatmap,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": cbar_label},
        mask=np.isnan(rel_disent_matrix),
        square=True,
        linewidths=0.5,
        linecolor="gray",
    )

    # Add grid if requested
    if show_grid:
        ax.grid(True, linestyle="-", alpha=grid_alpha, color="black")

    # Set title
    if title is None:
        title = "Disentanglement Analysis"
    ax.set_title(title, fontsize=title_fontsize, pad=20)

    # Configure tick labels
    ax.set_xticks(np.arange(len(feat_names)) + 0.5)
    ax.set_xticklabels(feat_names, fontsize=fontsize, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(feat_names)) + 0.5)
    ax.set_yticklabels(feat_names, fontsize=fontsize, rotation=0)

    # Set axis labels
    ax.set_xlabel("Feature (as secondary)", fontsize=fontsize + 2)
    ax.set_ylabel("Feature (as primary)", fontsize=fontsize + 2)

    plt.tight_layout()

    return fig, ax


def plot_disentanglement_summary(
    disent_matrix,
    count_matrix,
    feat_names,
    experiments=None,
    title_prefix="",
    figsize=(14, 10),
    dpi=100,
):
    """Plot comprehensive disentanglement analysis with multiple views.

    Creates a figure with multiple subplots showing:
    1. Disentanglement heatmap
    2. Feature dominance scores
    3. Pairwise interaction counts

    Parameters
    ----------
    disent_matrix : ndarray or list of ndarray
        Disentanglement matrix(es). If list, matrices are summed.
    count_matrix : ndarray or list of ndarray
        Count matrix(es). If list, matrices are summed.
    feat_names : list of str
        Feature names corresponding to matrix indices.
    experiments : list of str, optional
        Experiment names if multiple matrices provided. Currently not used.
    title_prefix : str, optional
        Prefix for the main title.
    figsize : tuple, optional
        Figure size. Default: (14, 10).
    dpi : int, optional
        Figure DPI. Default: 100.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing all subplots.
    
    Raises
    ------
    ImportError
        If matplotlib.colors, seaborn, or pandas not available
    ValueError
        If matrix dimensions don't match or feat_names length doesn't match matrices
    TypeError
        If disent_matrix/count_matrix not ndarray or list of ndarrays
    
    Notes
    -----
    - Creates 2x2 grid layout with custom ratios (3:1 for both dimensions)
    - Main heatmap uses red-white-green colormap (different from plot_disentanglement_heatmap)
    - Dominance scores show how often each feature is primary
    - Only displays feature pairs with non-zero counts
    - experiments parameter is accepted but not used in current implementation
    - Calls plt.tight_layout() which affects figure state
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Create synthetic data
    >>> features = ['speed', 'position', 'direction', 'licking']
    >>> 
    >>> # Synthetic matrices as before
    >>> disent_mat = np.array([
    ...     [0, 15, 8, 20],
    ...     [5, 0, 12, 18],
    ...     [12, 8, 0, 10],
    ...     [10, 7, 15, 0]
    ... ])
    >>> count_mat = np.array([
    ...     [0, 20, 20, 30],
    ...     [20, 0, 20, 25],
    ...     [20, 20, 0, 25],
    ...     [30, 25, 25, 0]
    ... ])
    >>> 
    >>> # Single experiment summary
    >>> fig = plot_disentanglement_summary(disent_mat, count_mat, features)
    >>> plt.close(fig)  # Suppress display
    >>> 
    >>> # Multiple experiments (matrices will be summed)
    >>> disent2 = disent_mat * 0.8  # Second synthetic experiment
    >>> count2 = count_mat  # Same comparison counts
    >>> fig = plot_disentanglement_summary(
    ...     [disent_mat, disent2], [count_mat, count2], features,
    ...     title_prefix="Combined: "
    ... )
    >>> plt.close(fig)
    """
    # Handle multiple experiments
    if isinstance(disent_matrix, list):
        total_disent = np.sum(disent_matrix, axis=0)
        total_count = np.sum(count_matrix, axis=0)
        n_exp = len(disent_matrix)
    else:
        total_disent = disent_matrix
        total_count = count_matrix
        n_exp = 1

    # Create figure with subplots
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

    # Main heatmap
    ax_main = fig.add_subplot(gs[0, 0])

    # Calculate relative disentanglement matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_disent_matrix = np.divide(total_disent, total_count) * 100
        rel_disent_matrix[total_count == 0] = np.nan

    # Create colormap
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    import pandas as pd

    colors = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]
    cmap = LinearSegmentedColormap.from_list("disentanglement_cmap", colors, N=100)

    # Create DataFrame and plot
    df_heatmap = pd.DataFrame(rel_disent_matrix, columns=feat_names, index=feat_names)
    sns.heatmap(
        df_heatmap,
        ax=ax_main,
        cmap=cmap,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Disentanglement score (%)"},
        mask=np.isnan(rel_disent_matrix),
        square=True,
        linewidths=0.5,
        linecolor="gray",
    )
    ax_main.set_title("Disentanglement Heatmap")

    # Feature dominance scores (how often each feature is primary)
    ax_dom = fig.add_subplot(gs[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        dominance_scores = np.nansum(total_disent / total_count, axis=1)
    y_pos = np.arange(len(feat_names))
    ax_dom.barh(y_pos, dominance_scores, color="green", alpha=0.7)
    ax_dom.set_yticks(y_pos)
    ax_dom.set_yticklabels(feat_names)
    ax_dom.set_xlabel("Dominance Score")
    ax_dom.set_title("Feature Dominance")
    ax_dom.grid(True, alpha=0.3)

    # Interaction counts
    ax_counts = fig.add_subplot(gs[1, :])
    pair_counts = []
    pair_labels = []
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):
            if total_count[i, j] > 0:
                pair_counts.append(total_count[i, j])
                pair_labels.append(f"{feat_names[i]}-{feat_names[j]}")

    x_pos = np.arange(len(pair_counts))
    ax_counts.bar(x_pos, pair_counts, color="blue", alpha=0.7)
    ax_counts.set_xticks(x_pos)
    ax_counts.set_xticklabels(pair_labels, rotation=45, ha="right")
    ax_counts.set_ylabel("Number of neurons")
    ax_counts.set_title("Pairwise interaction counts")
    ax_counts.grid(True, axis="y", alpha=0.3)

    # Main title
    if n_exp > 1:
        title = f"{title_prefix}Disentanglement Analysis ({n_exp} experiments)"
    else:
        title = f"{title_prefix}Disentanglement Analysis"
    fig.suptitle(title, fontsize=16, y=0.98)

    plt.tight_layout()
    return fig


def plot_selectivity_heatmap(
    exp,
    significant_neurons,
    metric="mi",
    cmap="viridis",
    use_log_scale=False,
    vmin=None,
    vmax=None,
    figsize=(10, 8),
    significance_threshold=None,
    ax=None,
):
    """Create a heatmap showing metric values for selective neuron-feature pairs.

    Parameters
    ----------
    exp : Experiment
        The experiment object containing all data and results
    significant_neurons : dict
        Dictionary mapping neuron IDs to lists of significant features
    metric : str, optional
        Which metric to display ('mi' for mutual information, 'corr' for correlation)
        Default: 'mi'
    cmap : str, optional
        Colormap to use. Default: 'viridis'
    use_log_scale : bool, optional
        Whether to use log scale for metric values. Default: False
    vmin : float, optional
        Minimum value for colormap. If None, auto-determined from data
    vmax : float, optional
        Maximum value for colormap. If None, auto-determined from data
    figsize : tuple, optional
        Figure size (ignored if ax provided). Default: (10, 8)
    significance_threshold : float, optional
        If provided, only show pairs with p-value below this threshold
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the heatmap
    ax : matplotlib.axes.Axes
        Axes containing the heatmap
    stats : dict
        Dictionary containing statistics about the data:
        - n_selective: number of selective neurons
        - n_pairs: total number of significant pairs
        - selectivity_rate: percentage of selective neurons
        - metric_values: list of all non-zero metric values
        - sparsity: percentage of zero entries in the matrix
    
    Raises
    ------
    AttributeError
        If experiment missing required attributes (dynamic_features, n_cells, get_neuron_feature_pair_stats)
    KeyError
        If neuron or feature not found in experiment data
    
    Notes
    -----
    - Only processes string-type features (tuple features are ignored)
    - Always uses mode='calcium' when retrieving stats
    - Calls plt.tight_layout() which affects figure state
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from driada.experiment import load_demo_experiment
    >>> 
    >>> # Load demo experiment and create synthetic selectivity data
    >>> exp = load_demo_experiment(verbose=False)
    >>> 
    >>> # Create synthetic significant_neurons dict
    >>> # In real usage, this comes from INTENSE analysis
    >>> significant_neurons = {
    ...     5: ['speed', 'x_pos'],   # Neuron 5 selective for speed and x_pos
    ...     10: ['speed'],           # Neuron 10 selective for speed only
    ...     15: ['x_pos', 'y_pos'],  # Neuron 15 selective for spatial features
    ...     20: ['speed', 'y_pos'],
    ...     25: ['x_pos']
    ... }
    >>> 
    >>> # Initialize stats_tables if not present
    >>> if not hasattr(exp, 'stats_tables'):
    ...     exp.stats_tables = {}
    >>> if 'calcium' not in exp.stats_tables:
    ...     exp.stats_tables['calcium'] = {}
    >>> 
    >>> # Add minimal stats to experiment for the example
    >>> # Using features that exist in demo data
    >>> # Each stat entry needs data_hash and other required fields
    >>> import numpy as np
    >>> hash_val = 'demo_hash'
    >>> exp.stats_tables['calcium']['speed'] = {
    ...     5: {'me': 0.3, 'pval': 0.001, 'rval': 0.5, 'data_hash': hash_val,
    ...         'opt_delay': 0, 'pre_pval': 0.1, 'pre_rval': 0.3, 
    ...         'rel_me_beh': 0.2, 'rel_me_ca': 0.15}, 
    ...     10: {'me': 0.4, 'pval': 0.0001, 'rval': 0.6, 'data_hash': hash_val,
    ...          'opt_delay': 0, 'pre_pval': 0.05, 'pre_rval': 0.4,
    ...          'rel_me_beh': 0.3, 'rel_me_ca': 0.2}, 
    ...     20: {'me': 0.25, 'pval': 0.005, 'rval': 0.4, 'data_hash': hash_val,
    ...          'opt_delay': 0, 'pre_pval': 0.2, 'pre_rval': 0.25,
    ...          'rel_me_beh': 0.15, 'rel_me_ca': 0.1}
    ... }
    >>> exp.stats_tables['calcium']['x_pos'] = {
    ...     5: {'me': 0.35, 'pval': 0.002, 'rval': 0.55, 'data_hash': hash_val,
    ...         'opt_delay': 0, 'pre_pval': 0.15, 'pre_rval': 0.35,
    ...         'rel_me_beh': 0.25, 'rel_me_ca': 0.2}, 
    ...     15: {'me': 0.45, 'pval': 0.0001, 'rval': 0.7, 'data_hash': hash_val,
    ...          'opt_delay': 0, 'pre_pval': 0.08, 'pre_rval': 0.5,
    ...          'rel_me_beh': 0.35, 'rel_me_ca': 0.3}, 
    ...     25: {'me': 0.3, 'pval': 0.003, 'rval': 0.5, 'data_hash': hash_val,
    ...          'opt_delay': 0, 'pre_pval': 0.18, 'pre_rval': 0.3,
    ...          'rel_me_beh': 0.2, 'rel_me_ca': 0.15}
    ... }
    >>> exp.stats_tables['calcium']['y_pos'] = {
    ...     15: {'me': 0.2, 'pval': 0.01, 'rval': 0.3, 'data_hash': hash_val,
    ...          'opt_delay': 0, 'pre_pval': 0.25, 'pre_rval': 0.15,
    ...          'rel_me_beh': 0.1, 'rel_me_ca': 0.08}, 
    ...     20: {'me': 0.15, 'pval': 0.02, 'rval': 0.25, 'data_hash': hash_val,
    ...          'opt_delay': 0, 'pre_pval': 0.3, 'pre_rval': 0.12,
    ...          'rel_me_beh': 0.08, 'rel_me_ca': 0.05}
    ... }
    >>> 
    >>> # Basic selectivity heatmap
    >>> fig, ax, stats = plot_selectivity_heatmap(exp, significant_neurons)
    >>> plt.close(fig)  # Suppress display
    >>> 
    >>> # With log scale and p-value filtering
    >>> fig, ax, stats = plot_selectivity_heatmap(
    ...     exp, significant_neurons,
    ...     use_log_scale=True,
    ...     significance_threshold=0.01
    ... )
    >>> plt.close(fig)
    >>> 
    >>> # Custom visualization
    >>> fig, ax, stats = plot_selectivity_heatmap(
    ...     exp, significant_neurons,
    ...     cmap='hot', vmin=0, vmax=0.5,
    ...     figsize=(12, 10)
    ... )
    >>> plt.close(fig)
    """
    # Get all features and create ordered lists
    all_features = sorted(
        [f for f in exp.dynamic_features.keys() if isinstance(f, str)]
    )
    all_neurons = list(range(exp.n_cells))

    # Create matrix with metric values (0 for non-selective pairs)
    selectivity_matrix = np.zeros((len(all_neurons), len(all_features)))

    # Collect all metric values for statistics
    all_metric_values = []

    for neuron_idx, cell_id in enumerate(all_neurons):
        for feat_idx, feat_name in enumerate(all_features):
            # Check if this neuron-feature pair is significant
            if (
                cell_id in significant_neurons
                and feat_name in significant_neurons[cell_id]
            ):
                # Get the statistics for this pair
                try:
                    pair_stats = exp.get_neuron_feature_pair_stats(
                        cell_id, feat_name, mode="calcium"
                    )
                    
                    # Skip if stats not available
                    if pair_stats is None:
                        continue

                    # Check significance threshold if provided
                    if significance_threshold is not None:
                        pval = pair_stats.get("pval", None)
                        # Skip if pval is None (failed stage 1) or above threshold
                        if pval is None or pval > significance_threshold:
                            continue

                    # Get the metric value - 'me' contains the metric value for whichever metric was used
                    value = pair_stats.get("me", 0)

                    selectivity_matrix[neuron_idx, feat_idx] = value
                    all_metric_values.append(value)
                except (KeyError, AttributeError):
                    # Skip if stats not available for this pair
                    continue

    # Apply log scale if requested
    if use_log_scale and len(all_metric_values) > 0:
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        selectivity_matrix = np.log10(selectivity_matrix + epsilon)
        # Set zeros back to a special value for visualization
        selectivity_matrix[selectivity_matrix < np.log10(epsilon * 2)] = np.nan

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine color limits
    if len(all_metric_values) > 0:
        if vmin is None:
            vmin = 0 if not use_log_scale else np.log10(min(all_metric_values))
        if vmax is None:
            vmax = (
                max(all_metric_values)
                if not use_log_scale
                else np.log10(max(all_metric_values))
            )
    else:
        vmin = 0
        vmax = 1

    # Create masked array to handle NaN values properly
    masked_matrix = np.ma.masked_invalid(selectivity_matrix)

    # Plot heatmap
    im = ax.imshow(
        masked_matrix,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    # Set ticks and labels
    ax.set_xticks(range(len(all_features)))
    ax.set_xticklabels(all_features, rotation=45, ha="right")
    ax.set_yticks(
        range(0, len(all_neurons), max(1, len(all_neurons) // 20))
    )  # Show ~20 neuron labels
    ax.set_yticklabels(range(0, len(all_neurons), max(1, len(all_neurons) // 20)))

    # Labels and title
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Neurons", fontsize=12)
    metric_name = "Mutual Information" if metric == "mi" else "Correlation"
    scale_text = " (log₁₀)" if use_log_scale else ""
    ax.set_title(
        f"Neuronal Selectivity: {metric_name}{scale_text}",
        fontsize=14,
        fontweight="bold",
    )

    # Add colorbar with appropriate label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{metric_name}{scale_text}", rotation=270, labelpad=20)

    # Calculate statistics
    n_selective = len(significant_neurons)
    n_pairs = sum(len(features) for features in significant_neurons.values())
    selectivity_rate = (n_selective / exp.n_cells) * 100
    sparsity = (1 - n_pairs / (len(all_neurons) * len(all_features))) * 100

    # Add summary text
    summary_lines = [
        f"Selective neurons: {n_selective}/{exp.n_cells} ({selectivity_rate:.1f}%)",
        f"Total selective pairs: {n_pairs}",
    ]

    if len(all_metric_values) > 0:
        summary_lines.extend(
            [
                f"{metric.upper()} range: [{min(all_metric_values):.3f}, {max(all_metric_values):.3f}]",
                f"Mean {metric.upper()}: {np.mean(all_metric_values):.3f}",
            ]
        )

    summary_text = "\n".join(summary_lines)
    # Position text in the lower right corner to avoid colorbar overlap
    fig.text(
        0.98,
        0.02,
        summary_text,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Add grid for better readability
    ax.set_xticks(np.arange(len(all_features)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(all_neurons)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Return statistics
    stats = {
        "n_selective": n_selective,
        "n_pairs": n_pairs,
        "selectivity_rate": selectivity_rate,
        "metric_values": all_metric_values,
        "sparsity": sparsity,
    }

    return fig, ax, stats
