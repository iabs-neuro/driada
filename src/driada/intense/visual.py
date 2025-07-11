import numpy as np

import matplotlib.pyplot as plt
from ..utils.plot import create_default_figure, make_beautiful
from ..utils.data import rescale
from scipy.stats import rankdata, gaussian_kde, wasserstein_distance
import seaborn as sns


def plot_pc_activity(exp, cell_ind, ds=None, ax=None):
    """
    Plot place cell activity overlaid on spatial trajectory.
    
    Parameters
    ----------
    exp : Experiment
        Experiment object with spatial data and neurons.
    cell_ind : int
        Index of the neuron to plot.
    ds : int, optional
        Downsampling factor. Default: 5.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    """
    pc_stats = exp.stats_table[('x', 'y')][cell_ind]
    pval = None if pc_stats['pval'] is None else np.round(pc_stats['pval'], 7)
    rel_mi_beh = None if pc_stats['rel_mi_beh'] is None else np.round(pc_stats['rel_mi_beh'], 4)

    if ds is None:
        ds = 5

    if ax is None:
        lenx = max(exp.x.data) - min(exp.x.data)
        leny = max(exp.y.data) - min(exp.y.data)
        xyratio = max(lenx / leny, leny / lenx)
        fig, ax = create_default_figure(6*xyratio, 6)

    #neur = np.roll(rescale(rankdata(exp.neurons[ind].ca.data)), 0)
    neur = rescale(np.log(exp.neurons[cell_ind].ca.data+1e-10))
    spinds = np.where(exp.neurons[cell_ind].sp.data != 0)[0]

    #ax.plot(exp.x.data[::ds], exp.y.data[::ds], c = 'k', alpha=0.3)
    ax.scatter(exp.x.data[::ds], exp.y.data[::ds], c=neur[::ds], cmap = 'plasma', alpha=0.8)
    ax.scatter(exp.x.data[spinds], exp.y.data[spinds], c='k', alpha=1, marker='*', linewidth=2, s=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Cell {cell_ind}, Rel MI={rel_mi_beh}, pval={pval}')

    return ax


def plot_neuron_feature_density(exp, data_type, cell_id, featname, ind1=0, ind2=100000, ds=1, shift=None, ax=None):
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
        Temporal shift (not implemented). Default: None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    """
    ind2 = min(exp.n_frames, ind2)

    if data_type == 'calcium':
        sig = exp.neurons[cell_id].ca.scdata[ind1:ind2][::ds]
    if data_type == 'spikes':
        sig = exp.neurons[cell_id].sp.scdata[ind1:ind2][::ds]

    feature = getattr(exp, featname)
    bdata = feature.scdata[ind1:ind2][::ds]
    rbdata = rescale(rankdata(bdata))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    if feature.is_binary:
        if data_type == 'calcium':
            vals0 = np.log10(sig[np.where((rbdata == min(rbdata)) & (sig > 0))])
            vals1 = np.log10(sig[np.where((rbdata == max(rbdata)) & (sig > 0))])

            wsd = wasserstein_distance(vals0, vals1)
            # _ = ax.hist(vals0, bins = 25, color = 'b', log = True, density = True, alpha = 0.7, label=f'{featname}=1')
            _ = sns.kdeplot(vals0, ax=ax, c='b', label=f'{featname}=0', linewidth=3, bw_adjust=0.5)
            _ = sns.kdeplot(vals1, ax=ax, c='r', label=f'{featname}=1', linewidth=3, bw_adjust=0.5)
            # _ = ax.hist(vals1, bins = 25, color = 'r', log = True, density = True, alpha = 0.7, label=f'{featname}=0')
            ax.legend(loc='upper right')
            ax.set_xlabel('log(dF/F)', fontsize=20)
            ax.set_ylabel('density', fontsize=20)
            ax.set_title(f'wsd={wsd}')

        if data_type == 'spikes':
            raise NotImplementedError('Binary feature density plot for spike data not yet implemented')

    else:
        x0, y0 = np.log10(sig + np.random.random(size=len(sig)) * 1e-8), np.log(
            bdata + np.random.random(size=len(bdata)) * 1e-8)

        jdata = np.vstack([x0, y0]).T
        # jplot = sns.jointplot(jdata, x=jdata[:,0], y=jdata[:,1], kind='hist', bins=100)
        nbins = 100
        k = gaussian_kde(jdata.T)
        xi, yi = np.mgrid[x0.min():x0.max():nbins * 1j, y0.min():y0.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # plot a density
        ax.set_title('Density')
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='coolwarm')
        ax.set_xlabel('log(signals)', fontsize=20)
        ax.set_ylabel(f'log({featname})', fontsize=20)

    return ax


def plot_neuron_feature_pair(exp, cell_id, featname, ind1=0, ind2=100000, ds=1,
                             add_density_plot=True, ax=None, title=None):
    """
    Plot neural activity time series alongside behavioral feature.
    
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
        Axes to plot on (ignored if add_density_plot=True).
    title : str, optional
        Custom title for the plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot(s).
    """

    ind2 = min(exp.n_frames, ind2)
    ca = exp.neurons[cell_id].ca.scdata[ind1:ind2][::ds]
    #rca = rescale(rankdata(ca))
    feature = getattr(exp, featname)
    bdata = feature.scdata[ind1:ind2][::ds]
    rbdata = rescale(rankdata(bdata))

    if ax is None:
        if add_density_plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), width_ratios=[0.6, 0.4])
            ax0, ax1 = axs
            ax1 = make_beautiful(ax1)
        else:
            fig, ax0 = plt.subplots(figsize=(10, 6))
            ax1 = None

    ax0 = make_beautiful(ax0)

    ax0.plot(np.arange(ind1, ind2)[::ds], ca, c='b', linewidth=2, alpha=0.5, label=f'neuron {cell_id}')
    if feature.discrete:
        ax0.scatter(np.arange(ind1, ind2)[::ds][np.where(rbdata == 1)], ca[np.where(rbdata == 1)], c='r', linewidth=2)
    else:
        ax0.plot(np.arange(ind1, ind2)[::ds], rbdata, c='r', linewidth=2, alpha=0.5)

    if add_density_plot:
        plot_neuron_feature_density(exp, 'calcium', cell_id, featname, ind1=ind1, ind2=ind2, ds=ds, ax=ax1)

    ax0.set_xlabel('timeframes', fontsize=20)
    ax0.set_ylabel('Signal/behavior', fontsize=20)

    if title is None:
        title = f'{exp.signature} Neuron {cell_id}, feature {featname}'

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()

    return fig


def plot_disentanglement_heatmap(disent_matrix, count_matrix, feat_names, 
                                 title=None, figsize=(12, 10), dpi=100,
                                 cmap=None, vmin=0, vmax=100,
                                 cbar_label='Disentanglement score (%)',
                                 fontsize=14, title_fontsize=18,
                                 show_grid=True, grid_alpha=0.3):
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
        
    Notes
    -----
    The heatmap uses a diverging colormap where:
    - Red indicates low disentanglement (feature is redundant)
    - White indicates balanced contribution (~50%)
    - Green indicates high disentanglement (feature is primary)
    
    Cells are masked (shown in white) where no data is available.
    """
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    import pandas as pd
    
    # Calculate relative disentanglement matrix (as percentage)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_disent_matrix = np.divide(disent_matrix, count_matrix) * 100
        rel_disent_matrix[count_matrix == 0] = np.nan
    
    # Create default colormap if not provided
    if cmap is None:
        # Red -> White -> Green gradient
        colors = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("disentanglement_cmap", colors, N=n_bins)
    
    # Create DataFrame for seaborn
    df_heatmap = pd.DataFrame(rel_disent_matrix, columns=feat_names, index=feat_names)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create heatmap
    sns.heatmap(df_heatmap,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': cbar_label},
                mask=np.isnan(rel_disent_matrix),
                square=True,
                linewidths=0.5,
                linecolor='gray')
    
    # Add grid if requested
    if show_grid:
        ax.grid(True, linestyle='-', alpha=grid_alpha, color='black')
    
    # Set title
    if title is None:
        title = 'Disentanglement Analysis'
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    
    # Configure tick labels
    ax.set_xticks(np.arange(len(feat_names)) + 0.5)
    ax.set_xticklabels(feat_names, fontsize=fontsize, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(feat_names)) + 0.5)
    ax.set_yticklabels(feat_names, fontsize=fontsize, rotation=0)
    
    # Set axis labels
    ax.set_xlabel('Feature (as secondary)', fontsize=fontsize + 2)
    ax.set_ylabel('Feature (as primary)', fontsize=fontsize + 2)
    
    plt.tight_layout()
    
    return fig, ax


def plot_disentanglement_summary(disent_matrix, count_matrix, feat_names,
                                experiments=None, title_prefix='',
                                figsize=(14, 10), dpi=100):
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
        Experiment names if multiple matrices provided.
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
    with np.errstate(divide='ignore', invalid='ignore'):
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
    sns.heatmap(df_heatmap, ax=ax_main, cmap=cmap, vmin=0, vmax=100,
                cbar_kws={'label': 'Disentanglement score (%)'},
                mask=np.isnan(rel_disent_matrix), square=True,
                linewidths=0.5, linecolor='gray')
    ax_main.set_title('Disentanglement Heatmap')
    
    # Feature dominance scores (how often each feature is primary)
    ax_dom = fig.add_subplot(gs[0, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        dominance_scores = np.nansum(total_disent / total_count, axis=1)
    y_pos = np.arange(len(feat_names))
    ax_dom.barh(y_pos, dominance_scores, color='green', alpha=0.7)
    ax_dom.set_yticks(y_pos)
    ax_dom.set_yticklabels(feat_names)
    ax_dom.set_xlabel('Dominance Score')
    ax_dom.set_title('Feature Dominance')
    ax_dom.grid(True, alpha=0.3)
    
    # Interaction counts
    ax_counts = fig.add_subplot(gs[1, :])
    pair_counts = []
    pair_labels = []
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):
            if total_count[i, j] > 0:
                pair_counts.append(total_count[i, j])
                pair_labels.append(f'{feat_names[i]}-{feat_names[j]}')
    
    x_pos = np.arange(len(pair_counts))
    ax_counts.bar(x_pos, pair_counts, color='blue', alpha=0.7)
    ax_counts.set_xticks(x_pos)
    ax_counts.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax_counts.set_ylabel('Number of neurons')
    ax_counts.set_title('Pairwise interaction counts')
    ax_counts.grid(True, axis='y', alpha=0.3)
    
    # Main title
    if n_exp > 1:
        title = f'{title_prefix}Disentanglement Analysis ({n_exp} experiments)'
    else:
        title = f'{title_prefix}Disentanglement Analysis'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    return fig
