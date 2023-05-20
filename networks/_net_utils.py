import networkx as nx


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