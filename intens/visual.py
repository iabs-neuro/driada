import numpy as np

from ..utils.plot import create_default_figure
from ..utils.data import rescale


def show_pc_activity(exp, cell_ind, ds=None, ax=None):
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