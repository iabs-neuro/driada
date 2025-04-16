import networkx as nx
import numpy as np
from copy import deepcopy

from .randomization import *

def get_giant_cc_from_graph(G):
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    # print([len(c) for c in connected_components])
    gcc = connected_components[0]

    return nx.subgraph(G, gcc)


def get_giant_scc_from_graph(G):
    # for a directed graph, its largest strongly connected component is returned.
    if nx.is_directed(G):
        strongly_connected_components = sorted(nx.strongly_connected_components(G),
                                               key=len,
                                               reverse=True)

        gcc = strongly_connected_components[0]

    else:
        raise ValueError('Strongly connected components are meaningless for undirected graphs')

    return nx.subgraph(G, gcc)


def remove_selfloops_from_graph(graph):
    g = deepcopy(graph) # NetworkX graphs are highly nested, deepcopy is safer
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return g


def remove_isolates_and_selfloops_from_graph(graph):
    g = deepcopy(graph) # NetworkX graphs are highly nested, deepcopy is safer
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    g.remove_nodes_from(list(nx.isolates(g)))
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return g


def remove_isolates_from_graph(graph):
    g = deepcopy(graph)  # NetworkX graphs are highly nested, deepcopy is safer
    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def small_world_index(G, nrand=10, null_model='erdos-renyi'):
    asp = nx.average_shortest_path_length(G)
    acc = nx.average_clustering(G)

    r_asp_list = []
    r_acc_list = []
    i = 0
    while i < nrand:
        if null_model == 'maslov-sneppen':
            #ar = adj_random_rewiring_iom_preserving(sp.csr_array(data), is_weighted=True).todense()
            raise Exception('not implemented yet')

        elif null_model == 'erdos-renyi':
            n = G.number_of_nodes()
            p = 2.0 * G.number_of_edges() / n / (n - 1)
            Gr = nx.gnp_random_graph(n, p)

        if nx.is_connected(Gr):
            i += 1
            r_asp_list.append(nx.average_shortest_path_length(Gr))
            r_acc_list.append(nx.average_clustering(Gr))

    sw = (acc / np.mean(r_acc_list)) / (asp / np.mean(r_asp_list))
    return sw