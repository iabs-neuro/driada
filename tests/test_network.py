import networkx as nx
import numpy as np
import scipy.sparse as sp

from src.driada.network.net_base import Network
from src.driada.network.graph_utils import get_giant_scc_from_graph
from src.driada.network.matrix_utils import turn_to_partially_directed


def create_default_adj(directed=False):
    seed = 42
    G = nx.random_regular_graph(5, 100, seed=seed)
    adj = nx.adjacency_matrix(G)
    adj = sp.csr_matrix(turn_to_partially_directed(adj.A, int(directed)))
    return adj


def create_default_graph(directed=False):
    adj = create_default_adj(directed=directed)
    gtype = nx.DiGraph if directed else nx.Graph
    graph = nx.from_scipy_sparse_array(adj, create_using=gtype)
    return graph


def create_default_net():
    adj = create_default_adj()
    net = Network(adj=adj)
    return net


def test_init_from_adj():
    adj = create_default_adj()
    net = Network(adj=adj, create_nx_graph=False)
    assert np.allclose(net.adj.data, adj.data)


def test_init_from_graph():
    g = create_default_graph()
    net = Network(graph=g)
    assert np.allclose(net.adj.data, nx.adjacency_matrix(g).data)
    assert nx.is_isomorphic(net.graph, g)


def test_remove_isolates():
    g = create_default_graph()
    g2 = g.copy()
    g2.add_nodes_from(['a', 'b', 'c'])

    net1 = Network(graph=g2, preprocessing='remove_isolates', verbose=True)
    assert np.allclose(net1.adj.data, nx.adjacency_matrix(g).data)
    assert nx.is_isomorphic(net1.graph, g)

    net2 = Network(adj=nx.adjacency_matrix(g2), create_nx_graph=False, preprocessing='remove_isolates')
    assert np.allclose(net2.adj.data, nx.adjacency_matrix(g).data)


def test_take_giant_cс():
    g = create_default_graph()
    g2 = g.copy()
    g2.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
    g2.add_edges_from([('a', 'b'), ('c', 'd'), ('d', 'e'), ('e', 'c')])

    net1 = Network(graph=g2, preprocessing='giant_cc', verbose=True)
    assert np.allclose(net1.adj.data, nx.adjacency_matrix(g).data)
    assert nx.is_isomorphic(net1.graph, g)

    net2 = Network(adj=nx.adjacency_matrix(g2),
                   create_nx_graph=False,
                   preprocessing='giant_cc',
                   verbose=True)

    assert np.allclose(net2.adj.data, nx.adjacency_matrix(g).data)


def test_take_giant_scc():
    g = create_default_graph(directed=True)
    g2 = g.copy()
    g2.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
    g2.add_edges_from([('a', 'b'), ('c', 'd'), ('d', 'e'), ('e', 'c')])

    gcc = get_giant_scc_from_graph(g2)

    net1 = Network(graph=g2, preprocessing='giant_scc', verbose=True)
    assert np.allclose(net1.adj.data, nx.adjacency_matrix(gcc).data)
    assert nx.is_isomorphic(net1.graph, gcc)

    net2 = Network(adj=nx.adjacency_matrix(g2),
                   create_nx_graph=False,
                   preprocessing='giant_scc',
                   verbose=True)

    assert np.allclose(net2.adj.data, nx.adjacency_matrix(gcc).data)
