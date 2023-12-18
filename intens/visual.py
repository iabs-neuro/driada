import numpy as np

import matplotlib.pyplot as plt
from ..utils.plot import create_default_figure, make_beautiful
from ..utils.data import rescale
from scipy.stats import rankdata, gaussian_kde, wasserstein_distance
import seaborn as sns


def plot_pc_activity(exp, cell_ind, ds=None, ax=None):
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


def plot_neuron_feature_pair(exp, cell_id, featname, ind1=0, ind2=100000, ds=1,
                             add_density_plot=True, ax=None, title=None):

    ind2 = min(exp.n_frames, ind2)
    ca = exp.neurons[cell_id].ca.scdata[ind1:ind2][::ds]
    rca = rescale(rankdata(ca))
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
        if feature.discrete:
            vals0 = np.log10(ca[np.where((rbdata == 0) & (ca != 0))])
            vals1 = np.log10(ca[np.where((rbdata == 1) & (ca != 0))])
            wsd = wasserstein_distance(vals0, vals1)
            # _ = axs[1].hist(vals0, bins = 25, color = 'b', log = True, density = True, alpha = 0.7, label=f'{featname}=1')
            _ = sns.kdeplot(vals0, ax=axs[1], c='b', label=f'{featname}=0', linewidth=3, bw_adjust=0.5)
            _ = sns.kdeplot(vals1, ax=axs[1], c='r', label=f'{featname}=1', linewidth=3, bw_adjust=0.5)
            # _ = axs[1].hist(vals1, bins = 25, color = 'r', log = True, density = True, alpha = 0.7, label=f'{featname}=0')
            ax1.legend(loc='upper right')
            ax1.set_xlabel('log(signal)', fontsize=20)
            ax1.set_ylabel('density', fontsize=20)

        else:
            x0, y0 = np.log10(ca + np.random.random(size=len(ca)) * 1e-8), np.log(bdata + np.random.random(size=len(bdata)) * 1e-8)

            jdata = np.vstack([x0, y0]).T
            # jplot = sns.jointplot(jdata, x=jdata[:,0], y=jdata[:,1], kind='hist', bins=100)
            nbins = 100
            k = gaussian_kde(jdata.T)
            xi, yi = np.mgrid[x0.min():x0.max():nbins * 1j, y0.min():y0.max():nbins * 1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            # plot a density
            ax1.set_title('Density')
            ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='coolwarm')
            ax1.set_xlabel('log(signal)', fontsize=20)
            ax1.set_ylabel(f'log({featname})', fontsize=20)

    ax0.set_xlabel('timeframes', fontsize=20)
    ax0.set_ylabel('Signal/behavior', fontsize=20)

    if title is None:
        title = f'{exp.signature} Neuron {cell_id}, feature {featname}'

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
