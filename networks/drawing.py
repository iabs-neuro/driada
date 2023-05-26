from ..utils.plot import *
from .matrix_utils import *
import numpy as np
import networkx as nx
from itertools import combinations

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
        
def draw_spectrum(net, mode='adj', ax=None, colors=None):
    spectrum = net.get_spectrum(mode)
    data = np.array(sorted(list(set(spectrum)), key=np.abs))

    if ax is None:
        fig, ax = create_default_figure(12,10)

    if net.directed:
        ax.scatter(data.real, data.imag, netap='Spectral', c=colors)
    else:
        ax.hist(data.real, bins=50)
        

def draw_eigenvectors(net, left_ind, right_ind, mode='adj'):
    spectrum = net.get_spectrum(mode)
    eigenvectors = net.get_eigenvectors(mode)

    vecs = np.real(eigenvectors[:, left_ind: right_ind + 1])
    # vecs = np.abs(net.eigenvectors[:, left_ind: right_ind+1])
    eigvals = np.real(spectrum[left_ind: right_ind + 1])

    npics = vecs.shape[1]
    pics_in_a_row = np.ceil(np.sqrt(npics))
    pics_in_a_col = np.ceil(1.0 * npics / pics_in_a_row)
    fig, ax = create_default_figure(16,12)
    # ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                        hspace=0.2, wspace=0.1)

    if net.pos is None:
        # pos = nx.layout.spring_layout(net.graph)
        pos = nx.drawing.layout.circular_layout(net.graph)
    else:
        pos = net.pos

    nodesize = np.sqrt(net.scaled_outdeg) * 100 + 10
    anchor_for_colorbar = None
    for i in range(npics):
        vec = vecs[:, i]
        ax = fig.add_subplot(pics_in_a_col, pics_in_a_row, i + 1)
        '''
        ax.set(xlim=(min(xpos)-0.1*abs(min(xpos)), max(xpos)+0.1*abs(max(xpos))),
               ylim=(min(ypos)-0.1*abs(min(ypos)), max(ypos)+0.1*abs(max(ypos))))
        '''
        text = 'eigenvector ' + str(i + 1) + ' lambda ' + str(np.round(eigvals[i], 3))
        ax.set_title(text)
        options = {
            'node_color': vec,
            'node_size': nodesize,
            'netap': net.get_netap('Spectral')
        }

        nodes = nx.draw_networkx_nodes(net.graph, pos, ax=ax, **options)
        if anchor_for_colorbar is None:
            anchor_for_colorbar = nodes
        # edges = nx.draw_networkx_edges(net.graph, pos, **options)
        # pc, = mpl.collections.PatchCollection(nodes, netap = options['netap'])
        # pc.set_array(edge_colors)
        nodes.set_clim(vmin=min(vec) * 1.1, vmax=max(vec) * 1.1)
        plt.colorbar(anchor_for_colorbar)

    plt.show()

def draw_net(net, colors=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))

    if net.pos is None:
        print('Node positions not found, auto layout was constructed')
        pos = nx.layout.spring_layout(net.graph)
        # pos = nx.drawing.layout.circular_layout(net.graph)
    else:
        pos = net.pos

    nodesize = np.sqrt(net.scaled_outdeg) * 100 + 10
    options = {
        'node_size': nodesize,
        'netap': net.get_netap('Spectral'),
        'ax': ax
    }

    nodes = nx.draw_networkx_nodes(net.graph, pos, node_color=colors, ax=ax, **options)
    edges = nx.draw_networkx_edges(net.graph, pos, ax=ax, **options)

    plt.show()
        

def show_mat(net, dtype=None, mode='adj', ax=None):
    mat = getattr(net, mode)
    if mat is None:
        if mode in ['lap', 'lap_out']:
            mat = get_laplacian(net.adj)
        elif mode == 'nlap':
            mat = get_norm_laplacian(net.adj)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if not dtype is None:
        # print(net.adj.astype(dtype).A)
        ax.matshow(mat.astype(dtype).A)
    else:
        ax.matshow(mat.A)
        
        
def plot_lem_embedding(net, ndim, colors=None):

    if net.lem_emb is None:
        net.construct_lem_embedding(ndim)

    if colors is None:
        colors = np.zeros(net.lem_emb.shape[1])
        colors = range(net.lem_emb.shape[1])

    psize = 10
    data = net.lem_emb.A
    pairs = list(combinations(np.arange(ndim), 2))
    npics = len(pairs)
    pics_in_a_row = np.ceil(np.sqrt(npics))
    pics_in_a_col = np.ceil(1.0 * npics / pics_in_a_row)

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Projections')
    for i in range(len(pairs)):
        ax = fig.add_subplot(pics_in_a_row, pics_in_a_col, i + 1)
        i1, i2 = pairs[i]
        scatter = ax.scatter(data[i1, :], data[i2, :], c=colors, s=psize)

        legend = ax.legend(*scatter.legend_elements(),
                           loc="upper left", title="Classes")

        ax.text(min(data[i1, :]), min(data[i2, :]), 'axes ' + str(i1 + 1) + ' vs.' + str(i2 + 1),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        # ax.add_artist(legend)