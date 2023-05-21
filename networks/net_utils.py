import networkx as nx
from .randomization import *

def take_giant_component(G):
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    # IMPORTANT: for an undirected graph, its largest connected component is returned.
    # for a directed graph, its largest strongly connected component is returned.

    if nx.is_directed(G):
        strongly_connected_components = sorted(nx.strongly_connected_components(G),
                                               key=len, reverse=True)

        gcc = strongly_connected_components[0]

    else:
        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        # print([len(c) for c in connected_components])
        gcc = connected_components[0]

    return nx.subgraph(G, gcc)


def remove_isolates_and_selfloops_from_graph(g):
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    g.remove_nodes_from(list(nx.isolates(g)))
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return g

def remove_isolates(g):
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
            #ar = adj_random_rewiring_iom_preserving(sp.csr_matrix(data), is_weighted=True).A
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