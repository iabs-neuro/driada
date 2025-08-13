from ..utils.plot import *
from .matrix_utils import *
import numpy as np
import networkx as nx
from itertools import combinations
from matplotlib import cm
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize as color_normalize


def draw_degree_distr(net, mode=None, cumulative=0, survival=1, log_log=0):
    if not net.directed:
        mode = "all"

    fig, ax = create_default_figure(10, 8)
    ax.set_title("Degree distribution", color="white")

    if mode is not None:
        distr = net.get_degree_distr(mode=mode)
        if cumulative:
            if survival:
                distr = 1 - np.cumsum(distr)
            else:
                distr = np.cumsum(distr)

        if log_log:
            (degree,) = ax.loglog(distr[:-1], linewidth=2, c="k", label="degree")
        else:
            (degree,) = ax.plot(distr, linewidth=2, c="k", label="degree")

        ax.legend(handles=[degree], fontsize=16)

    else:
        distr = net.get_degree_distr(mode="all")
        outdistr = net.get_degree_distr(mode="out")
        indistr = net.get_degree_distr(mode="in")
        distrlist = [distr, outdistr, indistr]
        if cumulative:
            if survival:
                distrlist = [1 - np.cumsum(d) for d in distrlist]
            else:
                distrlist = [np.cumsum(d) for d in distrlist]

        if log_log:
            (degree,) = ax.loglog(distrlist[0][:-1], linewidth=2, c="k", label="degree")
            (outdegree,) = ax.loglog(
                distrlist[1][:-1], linewidth=2, c="b", label="outdegree"
            )
            (indegree,) = ax.loglog(
                distrlist[2][:-1], linewidth=2, c="r", label="indegree"
            )
        else:
            (degree,) = ax.plot(distrlist[0], linewidth=2, c="k", label="degree")
            (outdegree,) = ax.plot(distrlist[1], linewidth=2, c="b", label="outdegree")
            (indegree,) = ax.plot(distrlist[2], linewidth=2, c="r", label="indegree")

        ax.legend(handles=[degree, outdegree, indegree], fontsize=16)


def draw_spectrum(net, mode="adj", ax=None, colors=None, cmap="plasma", nbins=None):
    spectrum = net.get_spectrum(mode)
    data = np.array(sorted(list(set(spectrum)), key=np.abs))

    if ax is None:
        fig, ax = create_default_figure(12, 10)

    if net.directed:
        ax.scatter(data.real, data.imag, cmap=cmap, c=colors)
    else:
        if nbins is None:
            nbins = len(spectrum) / 10
        ax.hist(data.real, bins=nbins)


def get_vector_coloring(vec, cmap="plasma"):
    cmap = plt.get_cmap(cmap)
    vec = np.array(vec).ravel()
    colors = cmap((vec - min(vec)) / (max(vec) - min(vec)))
    return colors


def draw_eigenvectors(
    net,
    left_ind,
    right_ind,
    mode="adj",
    nodesize=None,
    cmap="plasma",
    draw_edges=True,
    edge_options={},
):
    spectrum = net.get_spectrum(mode)
    eigenvectors = net.get_eigenvectors(mode)

    vecs = eigenvectors[:, left_ind : right_ind + 1]
    # vecs = np.abs(net.eigenvectors[:, left_ind: right_ind+1])
    eigvals = np.real(spectrum[left_ind : right_ind + 1])

    npics = vecs.shape[1]
    pics_in_a_row = int(np.ceil(np.sqrt(npics)))
    pics_in_a_col = int(np.ceil(1.0 * npics / pics_in_a_row))
    fig, axs = plt.subplots(
        nrows=pics_in_a_col,
        ncols=pics_in_a_row,
        figsize=(8 * pics_in_a_row, 8 * pics_in_a_col),
    )
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.1
    )

    if net.pos is None:
        pos = nx.layout.spring_layout(net.graph)
        # pos = nx.drawing.layout.circular_layout(net.graph)
    else:
        pos = net.pos

    if nodesize is None:
        nodesize = np.sqrt(net.scaled_outdeg) * 500 + 50

    for i in range(npics):
        vec = vecs[:, i]
        ax = fig.add_subplot(pics_in_a_col, pics_in_a_row, i + 1)

        text = "eigenvector " + str(i + 1) + " lambda " + str(np.round(eigvals[i], 3))
        ax.set_title(text)

        ncolors = get_vector_coloring(vec)
        options = {
            "node_color": ncolors,
            "node_size": nodesize,
            "cmap": mpl.colormaps[cmap],
        }

        nx.draw_networkx_nodes(net.graph, pos, ax=ax, **options)

        if draw_edges:
            nx.draw_networkx_edges(net.graph, pos, **edge_options)
        # pc, = mpl.collections.PatchCollection(nodes, cmap = options['cmap'])
        # pc.set_array(edge_colors)

        cmappable = ScalarMappable(color_normalize(0, 1), cmap=cmap)
        fig.colorbar(cmappable, ax=ax)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def draw_net(net, colors=None, nodesize=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))

    if net.pos is None:
        print("Node positions not found, auto layout was constructed")
        pos = nx.layout.spring_layout(net.graph)
        # pos = nx.drawing.layout.circular_layout(net.graph)
    else:
        pos = net.pos

    if nodesize is None:
        nodesize = np.sqrt(net.scaled_outdeg) * 100 + 10

    node_options = {
        "node_size": nodesize,
        "cmap": cm.get_cmap("Spectral"),
    }
    edge_options = {}

    nx.draw_networkx_nodes(
        net.graph, pos, node_color=colors, ax=ax, **node_options
    )
    nx.draw_networkx_edges(net.graph, pos, ax=ax, **edge_options)

    plt.show()


def show_mat(net, dtype=None, mode="adj", ax=None):
    mat = getattr(net, mode)
    if mat is None:
        if mode in ["lap", "lap_out"]:
            mat = get_laplacian(net.adj)
        elif mode == "nlap":
            mat = get_norm_laplacian(net.adj)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if dtype is not None:
        ax.matshow(mat.astype(dtype).toarray())
    else:
        ax.matshow(mat.toarray())


def plot_lem_embedding(net, ndim, colors=None):

    if net.lem_emb is None:
        net.construct_lem_embedding(ndim)

    if colors is None:
        colors = np.zeros(net.lem_emb.shape[1])
        colors = range(net.lem_emb.shape[1])

    psize = 10
    data = net.lem_emb.toarray()
    pairs = list(combinations(np.arange(ndim), 2))
    npics = len(pairs)
    pics_in_a_row = np.ceil(np.sqrt(npics))
    pics_in_a_col = np.ceil(1.0 * npics / pics_in_a_row)

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("Projections")
    for i in range(len(pairs)):
        ax = fig.add_subplot(pics_in_a_row, pics_in_a_col, i + 1)
        i1, i2 = pairs[i]
        scatter = ax.scatter(data[i1, :], data[i2, :], c=colors, s=psize)

        ax.legend(
            *scatter.legend_elements(), loc="upper left", title="Classes"
        )

        ax.text(
            min(data[i1, :]),
            min(data[i2, :]),
            "axes " + str(i1 + 1) + " vs." + str(i2 + 1),
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
        )
        # ax.add_artist(legend)
