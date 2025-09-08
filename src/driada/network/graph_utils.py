import networkx as nx
import numpy as np
from copy import deepcopy


def get_giant_cc_from_graph(G):
    """Extract the giant (largest) connected component from a graph.

    For directed graphs, returns the largest weakly connected component.
    For undirected graphs, returns the largest connected component.
    The graph type is preserved (DiGraph remains DiGraph, Graph remains Graph).

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph from which to extract the giant component.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        Subgraph containing only the nodes in the giant connected component.
        The returned graph type matches the input graph type.

    Raises
    ------
    IndexError
        If the graph has no nodes or no connected components.

    See Also
    --------
    ~driada.network.graph_utils.get_giant_scc_from_graph : Extract giant strongly connected component.
    :func:`networkx.connected_components` : Find all connected components.
    :func:`networkx.weakly_connected_components` : Find weakly connected components.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> gcc = get_giant_cc_from_graph(G)
    >>> len(gcc) == len(G)  # Karate club is fully connected
    True    """
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    if nx.is_directed(G):
        connected_components = sorted(
            nx.weakly_connected_components(G), key=len, reverse=True
        )
    else:
        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    gcc = connected_components[0]

    return nx.subgraph(G, gcc)


def get_giant_scc_from_graph(G):
    """Extract the giant (largest) strongly connected component from a directed graph.

    A strongly connected component is a maximal subgraph where every node can
    reach every other node via directed paths.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph from which to extract the giant strongly connected component.

    Returns
    -------
    networkx.DiGraph
        Subgraph containing only the nodes in the giant strongly connected component.

    Raises
    ------
    ValueError
        If the input graph is undirected, as strongly connected components
        are only meaningful for directed graphs.
    IndexError
        If the graph has no nodes or no strongly connected components.

    See Also
    --------
    ~driada.network.graph_utils.get_giant_cc_from_graph : Extract giant connected component.
    :func:`networkx.strongly_connected_components` : Find all strongly connected components.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (4, 5)])
    >>> scc = get_giant_scc_from_graph(G)
    >>> sorted(scc.nodes())
    [1, 2, 3]    """
    # for a directed graph, its largest strongly connected component is returned.
    if nx.is_directed(G):
        strongly_connected_components = sorted(
            nx.strongly_connected_components(G), key=len, reverse=True
        )

        gcc = strongly_connected_components[0]

    else:
        raise ValueError(
            "Strongly connected components are meaningless for undirected graphs"
        )

    return nx.subgraph(G, gcc)


def remove_selfloops_from_graph(graph):
    """Remove all self-loop edges from a graph.

    Self-loops are edges that connect a node to itself.
    The graph type is preserved.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Input graph from which to remove self-loops.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A deep copy of the input graph with all self-loops removed.
        The graph type matches the input.

    See Also
    --------
    ~driada.network.graph_utils.remove_isolates_and_selfloops_from_graph : Remove both isolates and self-loops.
    :func:`networkx.selfloop_edges` : Find all self-loop edges in a graph.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph([(1, 2), (2, 2), (2, 3)])  # (2, 2) is a self-loop
    >>> G_clean = remove_selfloops_from_graph(G)
    >>> G_clean.number_of_edges()
    2    """
    g = deepcopy(graph)  # NetworkX graphs are highly nested, deepcopy is safer
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return g


def remove_isolates_and_selfloops_from_graph(graph):
    """Remove all isolated nodes and self-loop edges from a graph.

    Isolated nodes are nodes with no edges connecting them to other nodes.
    Self-loops are edges that connect a node to itself.
    The graph type is preserved.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Input graph to clean.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A deep copy of the input graph with isolated nodes and self-loops removed.
        The graph type matches the input.

    See Also
    --------
    ~driada.network.graph_utils.remove_selfloops_from_graph : Remove only self-loops.
    ~driada.network.graph_utils.remove_isolates_from_graph : Remove only isolated nodes.
    :func:`networkx.isolates` : Find isolated nodes in a graph.
    :func:`networkx.selfloop_edges` : Find self-loop edges in a graph.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph([(1, 1), (2, 3)])  # Node 1 has only a self-loop
    >>> G.add_node(4)  # Add isolated node
    >>> G_clean = remove_isolates_and_selfloops_from_graph(G)
    >>> sorted(G_clean.nodes())
    [2, 3]    """
    g = deepcopy(graph)  # NetworkX graphs are highly nested, deepcopy is safer
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    # First remove self-loops
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    # Then remove isolates (including nodes that became isolated after self-loop removal)
    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def remove_isolates_from_graph(graph):
    """Remove all isolated nodes from a graph.

    Isolated nodes are nodes with degree 0 (no edges connecting them).

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Input graph from which to remove isolated nodes.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A deep copy of the input graph with all isolated nodes removed.
        The graph type matches the input.

    See Also
    --------
    ~driada.network.graph_utils.remove_isolates_and_selfloops_from_graph : Remove both isolates and self-loops.
    :func:`networkx.isolates` : Find isolated nodes in a graph.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph([(1, 2), (2, 3)])
    >>> G.add_node(4)  # Add isolated node
    >>> G_clean = remove_isolates_from_graph(G)
    >>> sorted(G_clean.nodes())
    [1, 2, 3]    """
    g = deepcopy(graph)  # NetworkX graphs are highly nested, deepcopy is safer
    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def small_world_index(G, nrand=10, null_model="erdos-renyi"):
    """Calculate the small-world index of a graph.

    The small-world index quantifies how much a network exhibits small-world
    properties compared to random networks. It is defined as:
    SW = (C/C_rand) / (L/L_rand)
    where C is clustering coefficient and L is average shortest path length.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (must be connected).
    nrand : int, optional
        Number of random graphs to generate for comparison.
        Default is 10.
    null_model : {'erdos-renyi', 'maslov-sneppen'}, optional
        Type of random graph model to use for comparison.
        Currently only 'erdos-renyi' is implemented.
        Default is 'erdos-renyi'.

    Returns
    -------
    float
        Small-world index. Values > 1 indicate small-world properties.
        Typical small-world networks have SW >> 1.

    Raises
    ------
    NotImplementedError
        If 'maslov-sneppen' null model is requested (not implemented).
    networkx.NetworkXError
        If the input graph is not connected.

    See Also
    --------
    :func:`networkx.average_clustering` : Compute average clustering coefficient.
    :func:`networkx.average_shortest_path_length` : Compute average shortest path length.
    :func:`networkx.watts_strogatz_graph` : Generate small-world graphs.

    Notes
    -----
    The function only considers connected random graphs for comparison.
    Small-world networks have high clustering and short path lengths.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.watts_strogatz_graph(100, 6, 0.3)  # Small-world network
    >>> sw = small_world_index(G, nrand=5)
    >>> sw > 1  # Should be True for small-world networks
    True    """
    asp = nx.average_shortest_path_length(G)
    acc = nx.average_clustering(G)

    r_asp_list = []
    r_acc_list = []
    i = 0
    while i < nrand:
        if null_model == "maslov-sneppen":
            # ar = adj_random_rewiring_iom_preserving(sp.csr_array(data), is_weighted=True).todense()
            raise NotImplementedError("Maslov-Sneppen null model not implemented yet")

        elif null_model == "erdos-renyi":
            n = G.number_of_nodes()
            p = 2.0 * G.number_of_edges() / n / (n - 1)
            Gr = nx.gnp_random_graph(n, p)

        if nx.is_connected(Gr):
            i += 1
            r_asp_list.append(nx.average_shortest_path_length(Gr))
            r_acc_list.append(nx.average_clustering(Gr))

    sw = (acc / np.mean(r_acc_list)) / (asp / np.mean(r_asp_list))
    return sw
