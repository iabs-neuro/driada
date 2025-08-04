import networkx as nx
import numpy as np
import scipy.sparse as sp
import pytest

from driada.network.net_base import Network
from driada.network.graph_utils import get_giant_scc_from_graph
from driada.network.matrix_utils import turn_to_partially_directed


def create_default_adj(directed=False):
    seed = 42
    G = nx.random_regular_graph(5, 100, seed=seed)
    adj = nx.adjacency_matrix(G)
    adj = sp.csr_matrix(turn_to_partially_directed(adj.toarray(), int(directed)))
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


def test_diagonalize():
    adj = create_default_adj()
    net = Network(adj=adj, create_nx_graph=False)
    for mode in ['adj', 'trans', 'lap', 'nlap', 'rwlap']:
        net.diagonalize(mode=mode)


def test_assign_random_weights():
    """Test random weight assignment to adjacency matrix."""
    from driada.network.matrix_utils import assign_random_weights
    
    # Create a simple adjacency matrix
    A = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])
    
    # Assign random weights
    W = assign_random_weights(A)
    
    # Check properties
    assert W.shape == A.shape
    assert np.allclose(W, W.T)  # Should be symmetric
    assert np.all(W[A == 0] == 0)  # Zero entries should remain zero
    assert np.all(W[A != 0] > 0)  # Non-zero entries should be positive


def test_matrix_utils_edge_cases():
    """Test edge cases in matrix utility functions."""
    from driada.network.matrix_utils import (
        symmetric_component, 
        non_symmetric_component,
        remove_duplicates,
        adj_input_to_csr_sparse_matrix,
        remove_selfloops_from_adj,
        get_norm_laplacian,
    )
    
    # Test symmetric component
    A = sp.csr_matrix([[0, 1, 0],
                       [1, 0, 2],
                       [3, 0, 0]])
    
    symm_unweighted = symmetric_component(A, is_weighted=False)
    expected_unweighted = np.array([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, 0]])
    assert np.allclose(symm_unweighted, expected_unweighted)
    
    symm_weighted = symmetric_component(A, is_weighted=True)
    assert np.allclose(symm_weighted, expected_unweighted)
    
    # Test non-symmetric component
    non_symm = non_symmetric_component(A, is_weighted=True)
    expected = np.array([[0, 0, 0],
                         [0, 0, 2],
                         [3, 0, 0]])
    assert np.allclose(non_symm, expected)
    
    # Test remove duplicates
    row = np.array([0, 0, 1, 1, 2])
    col = np.array([1, 1, 0, 2, 1])
    data = np.array([1, 2, 3, 4, 5])
    coo = sp.coo_matrix((data, (row, col)), shape=(3, 3))
    result = remove_duplicates(coo)
    assert result.nnz == 4  # Should have 4 unique entries
    
    # Test adj_input_to_csr with numpy array
    arr = np.array([[0, 1], [1, 0]])
    result = adj_input_to_csr_sparse_matrix(arr)
    assert isinstance(result, sp.csr_array)
    
    # Test adj_input_to_csr with COO matrix
    coo = sp.coo_matrix(arr)
    result = adj_input_to_csr_sparse_matrix(coo)
    assert isinstance(result, sp.csr_array)
    
    # Test adj_input_to_csr with invalid format
    lil = sp.lil_matrix(arr)
    with pytest.raises(Exception, match="Wrong input parsed"):
        adj_input_to_csr_sparse_matrix(lil)
    
    # Test remove self-loops
    A_loops = sp.csr_matrix([[1, 1, 0],
                             [1, 2, 1],
                             [0, 1, 3]])
    result = remove_selfloops_from_adj(A_loops)
    assert result.diagonal().sum() == 0
    
    # Test normalized Laplacian with non-symmetric matrix
    A_nonsym = sp.csr_matrix([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]])
    with pytest.raises(Exception, match="non-hermitian"):
        get_norm_laplacian(A_nonsym)


def test_turn_to_partially_directed_weighted():
    """Test turn_to_partially_directed with weighted option."""
    mat = np.array([[0, 2.5, 0],
                    [2.5, 0, 1.5],
                    [0, 1.5, 0]])
    
    # Test with weighted=1
    result = turn_to_partially_directed(mat, directed=0.0, weighted=1)
    # Result is a sparse matrix, convert to dense for comparison
    assert sp.issparse(result)
    assert np.allclose(result.toarray(), mat)
    
    # Test with non-ndarray input
    with pytest.raises(Exception, match="Wrong input parsed"):
        turn_to_partially_directed([1, 2, 3], directed=0.0)


class TestNetworkModuleImports:
    """Test imports for network module components."""
    
    def test_import_drawing(self):
        """Test importing network drawing utilities."""
        from driada.network import drawing
        assert hasattr(drawing, '__file__')
        
    def test_import_quantum(self):
        """Test importing quantum network utilities."""
        from driada.network import quantum
        assert hasattr(quantum, '__file__')
        
    def test_import_spectral(self):
        """Test importing spectral analysis utilities."""
        from driada.network import spectral
        assert hasattr(spectral, '__file__')
        
    def test_import_randomization(self):
        """Test importing network randomization utilities."""
        from driada.network import randomization
        assert hasattr(randomization, '__file__')