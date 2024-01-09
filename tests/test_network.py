import networkx as nx
import pytest
import numpy as np

from ..network.net_base import Network

def create_default_adj():
    seed = 42
    G = nx.gnm_random_graph(100, 500, seed=seed)
    adj = nx.adjacency_matrix(G)
    return adj

def create_default_net():
    adj = create_default_adj()
    network_args = {
        'directed': False,
        'weighted': False
    }
    net = Network(adj, network_args)
    return net

def test_init_from_adj():
    adj = create_default_adj()
    net = create_default_net()
    assert np.allclose(net.adj.data, adj.data)
