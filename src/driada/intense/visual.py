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
            fig, ax0 = plt.subplots(figsize=(10, 6), width_ratios=[0.6, 0.4])
            ax1 = None

    ax0 = make_beautiful(ax0)

    ax0.plot(np.arange(ind1, ind2)[::ds], ca, c='b', linewidth=2, alpha=0.5, label=f'neuron {cell_id}')
    if feature.discrete:
        ax0.scatter(np.arange(ind1, ind2)[::ds][np.where(rbdata == 1)], ca[np.where(rbdata == 1)], c='r', linewidth=2)
    else:
        ax0.plot(np.arange(ind1, ind2)[::ds], rbdata, c='r', linewidth=2, alpha=0.5)

    if add_density_plot:
        plot_neuron_feature_density(exp, cell_id, featname, ind1=ind1, ind2=ind2, ds=ds, ax=ax1)

    ax0.set_xlabel('timeframes', fontsize=20)
    ax0.set_ylabel('Signal/behavior', fontsize=20)

    if title is None:
        title = f'{exp.signature} Neuron {cell_id}, feature {featname}'

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()

    return fig
