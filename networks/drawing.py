from ..utils.plot import *
import numpy as np

def draw_degree_distr(net, mode=None, cumulative=0, survival=1, log_log=0):
    if not net.directed:
        mode = 'all'

    fig, ax = create_default_figure(10, 8)
    ax.set_title('Degree distribution', color='white')

    if mode is not None:
        distr = net.get_degree_distr(mode=mode)
        if cumulative:
            if survival:
                distr = 1 - np.cumsum(distr)
            else:
                distr = np.cumsum(distr)

        if log_log:
            degree, = ax.loglog(distr[:-1], linewidth=2, c='k', label='degree')
        else:
            degree, = ax.plot(distr, linewidth=2, c='k', label='degree')

        ax.legend(handles=[degree], fontsize=16)

    else:
        distr = net.get_degree_distr(mode='all')
        outdistr = net.get_degree_distr(mode='out')
        indistr = net.get_degree_distr(mode='in')
        distrlist = [distr, outdistr, indistr]
        if cumulative:
            if survival:
                distrlist = [1 - np.cumsum(d) for d in distrlist]
            else:
                distrlist = [np.cumsum(d) for d in distrlist]

        if log_log:
            degree, = ax.loglog(distrlist[0][:-1], linewidth=2, c='k', label='degree')
            outdegree, = ax.loglog(distrlist[1][:-1], linewidth=2, c='b', label='outdegree')
            indegree, = ax.loglog(distrlist[2][:-1], linewidth=2, c='r', label='indegree')
        else:
            degree, = ax.plot(distrlist[0], linewidth=2, c='k', label='degree')
            outdegree, = ax.plot(distrlist[1], linewidth=2, c='b', label='outdegree')
            indegree, = ax.plot(distrlist[2], linewidth=2, c='r', label='indegree')

        ax.legend(handles=[degree, outdegree, indegree], fontsize=16)