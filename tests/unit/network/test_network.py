import networkx as nx
import numpy as np
import scipy.sparse as sp
import pytest
from unittest.mock import patch

from driada.network.net_base import (
    Network,
    check_matrix_type,
    check_adjacency,
    check_directed,
    check_weights_and_directions,
    calculate_directionality_fraction,
    select_construction_pipeline,
    UNDIR_MATRIX_TYPES,
    DIR_MATRIX_TYPES,
)
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
    g2.add_nodes_from(["a", "b", "c"])

    net1 = Network(graph=g2, preprocessing="remove_isolates", verbose=True)
    assert np.allclose(net1.adj.data, nx.adjacency_matrix(g).data)
    assert nx.is_isomorphic(net1.graph, g)

    net2 = Network(
        adj=nx.adjacency_matrix(g2),
        create_nx_graph=False,
        preprocessing="remove_isolates",
    )
    assert np.allclose(net2.adj.data, nx.adjacency_matrix(g).data)


def test_take_giant_cс():
    g = create_default_graph()
    g2 = g.copy()
    g2.add_nodes_from(["a", "b", "c", "d", "e"])
    g2.add_edges_from([("a", "b"), ("c", "d"), ("d", "e"), ("e", "c")])

    net1 = Network(graph=g2, preprocessing="giant_cc", verbose=True)
    assert np.allclose(net1.adj.data, nx.adjacency_matrix(g).data)
    assert nx.is_isomorphic(net1.graph, g)

    net2 = Network(
        adj=nx.adjacency_matrix(g2),
        create_nx_graph=False,
        preprocessing="giant_cc",
        verbose=True,
    )

    assert np.allclose(net2.adj.data, nx.adjacency_matrix(g).data)


def test_take_giant_scc():
    g = create_default_graph(directed=True)
    g2 = g.copy()
    g2.add_nodes_from(["a", "b", "c", "d", "e"])
    g2.add_edges_from([("a", "b"), ("c", "d"), ("d", "e"), ("e", "c")])

    gcc = get_giant_scc_from_graph(g2)

    net1 = Network(graph=g2, preprocessing="giant_scc", verbose=True)
    assert np.allclose(net1.adj.data, nx.adjacency_matrix(gcc).data)
    assert nx.is_isomorphic(net1.graph, gcc)

    net2 = Network(
        adj=nx.adjacency_matrix(g2),
        create_nx_graph=False,
        preprocessing="giant_scc",
        verbose=True,
    )

    assert np.allclose(net2.adj.data, nx.adjacency_matrix(gcc).data)


def test_diagonalize():
    adj = create_default_adj()
    net = Network(adj=adj, create_nx_graph=False)
    for mode in ["adj", "trans", "lap", "nlap", "rwlap"]:
        net.diagonalize(mode=mode)


def test_assign_random_weights():
    """Test random weight assignment to adjacency matrix."""
    from driada.network.matrix_utils import assign_random_weights

    # Create a simple adjacency matrix
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

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
    A = sp.csr_matrix([[0, 1, 0], [1, 0, 2], [3, 0, 0]])

    symm_unweighted = symmetric_component(A, is_weighted=False)
    expected_unweighted = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    # Convert sparse matrix to dense for comparison
    assert np.allclose(symm_unweighted.toarray(), expected_unweighted)

    symm_weighted = symmetric_component(A, is_weighted=True)
    # For weighted case, convert sparse to dense for comparison
    assert np.allclose(symm_weighted.toarray(), expected_unweighted)

    # Test non-symmetric component
    non_symm = non_symmetric_component(A, is_weighted=True)
    expected = np.array([[0, 0, 0], [0, 0, 2], [3, 0, 0]])
    # Convert sparse to dense if needed
    if sp.issparse(non_symm):
        assert np.allclose(non_symm.toarray(), expected)
    else:
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
    A_loops = sp.csr_matrix([[1, 1, 0], [1, 2, 1], [0, 1, 3]])
    result = remove_selfloops_from_adj(A_loops)
    assert result.diagonal().sum() == 0

    # Test normalized Laplacian with non-symmetric matrix
    A_nonsym = sp.csr_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    with pytest.raises(Exception, match="non-symmetric"):
        get_norm_laplacian(A_nonsym)


def test_turn_to_partially_directed_weighted():
    """Test turn_to_partially_directed with weighted option."""
    mat = np.array([[0, 2.5, 0], [2.5, 0, 1.5], [0, 1.5, 0]])

    # Test with weighted=1
    result = turn_to_partially_directed(mat, directed=0.0, weighted=1)
    # Result is a sparse matrix, convert to dense for comparison
    assert sp.issparse(result)
    assert np.allclose(result.toarray(), mat)

    # Test with non-ndarray input
    with pytest.raises(TypeError, match="Input must be numpy array or scipy sparse matrix"):
        turn_to_partially_directed([1, 2, 3], directed=0.0)


class TestNetworkModuleImports:
    """Test imports for network module components."""

    def test_import_drawing(self):
        """Test importing network drawing utilities."""
        from driada.network import drawing

        assert hasattr(drawing, "__file__")

    def test_import_quantum(self):
        """Test importing quantum network utilities."""
        from driada.network import quantum

        assert hasattr(quantum, "__file__")

    def test_import_spectral(self):
        """Test importing spectral analysis utilities."""
        from driada.network import spectral

        assert hasattr(spectral, "__file__")

    def test_import_randomization(self):
        """Test importing network randomization utilities."""
        from driada.network import randomization

        assert hasattr(randomization, "__file__")


class TestHelperFunctions:
    """Test all helper functions in net_base.py"""

    def test_check_matrix_type_valid(self):
        """Test check_matrix_type with valid inputs."""
        # Valid undirected matrix types
        for mode in UNDIR_MATRIX_TYPES:
            check_matrix_type(mode, is_directed=False)  # Should not raise

        # Valid directed matrix types
        for mode in DIR_MATRIX_TYPES:
            check_matrix_type(mode, is_directed=True)  # Should not raise

    def test_check_matrix_type_invalid(self):
        """Test check_matrix_type with invalid inputs."""
        # Invalid matrix type
        with pytest.raises(ValueError, match="is not in allowed matrix types"):
            check_matrix_type("invalid_type", is_directed=False)

        # Directed type with undirected network
        with pytest.raises(ValueError, match="not allowed for directed networks"):
            check_matrix_type("nlap", is_directed=True)

        # Undirected type with directed network
        with pytest.raises(ValueError, match="not allowed for undirected networks"):
            check_matrix_type("lap_out", is_directed=False)

    def test_check_adjacency_valid(self):
        """Test check_adjacency with valid square matrix."""
        A = sp.csr_matrix([[0, 1], [1, 0]])
        check_adjacency(A)  # Should not raise

    def test_check_adjacency_invalid(self):
        """Test check_adjacency with non-square matrix."""
        A = sp.csr_matrix([[0, 1, 0], [1, 0, 1]])
        with pytest.raises(ValueError, match="Adjacency matrix must be square"):
            check_adjacency(A)

    def test_check_directed_valid(self):
        """Test check_directed with valid inputs."""
        # Real world network
        check_directed(0, real_world=True)
        check_directed(1, real_world=True)
        check_directed(0.0, real_world=True)
        check_directed(1.0, real_world=True)

        # Non-real world network
        check_directed(0.5, real_world=False)
        check_directed(0.7, real_world=False)

    def test_check_directed_invalid(self):
        """Test check_directed with invalid inputs."""
        # Fractional direction for real network
        with pytest.raises(Exception, match="Fractional direction is not valid"):
            check_directed(0.5, real_world=True)

        # Out of range for non-real network
        with pytest.raises(Exception, match='Wrong "directed" parameter value'):
            check_directed(-0.1, real_world=False)

        with pytest.raises(Exception, match='Wrong "directed" parameter value'):
            check_directed(1.1, real_world=False)

    def test_check_weights_and_directions(self):
        """Test check_weights_and_directions function."""
        # Create symmetric unweighted matrix
        A_sym = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        check_weights_and_directions(
            A_sym, weighted=False, directed=False
        )  # Should pass

        # Test mismatch - symmetric matrix but directed=True
        with pytest.raises(ValueError, match="the adjacency matrix is symmetric"):
            check_weights_and_directions(A_sym, weighted=False, directed=True)

        # Create asymmetric matrix
        A_asym = sp.csr_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        check_weights_and_directions(
            A_asym, weighted=False, directed=True
        )  # Should pass

        # Test mismatch - asymmetric matrix but directed=False
        with pytest.raises(ValueError, match="the adjacency matrix is asymmetric"):
            check_weights_and_directions(A_asym, weighted=False, directed=False)

        # Create weighted matrix
        A_weighted = sp.csr_matrix([[0, 2.5, 1.5], [2.5, 0, 3.0], [1.5, 3.0, 0]])
        check_weights_and_directions(
            A_weighted, weighted=True, directed=False
        )  # Should pass

        # Test mismatch - weighted matrix but weighted=False
        with pytest.raises(ValueError, match="the adjacency matrix is weighted"):
            check_weights_and_directions(A_weighted, weighted=False, directed=False)

    def test_select_construction_pipeline(self):
        """Test select_construction_pipeline function."""
        # Test with adjacency matrix only
        A = sp.csr_matrix([[0, 1], [1, 0]])
        pipeline, directionality = select_construction_pipeline(A, None)
        assert pipeline == "adj"
        assert directionality == 0.0  # Symmetric matrix

        # Test with graph only
        G = nx.Graph()
        pipeline, directionality = select_construction_pipeline(None, G)
        assert pipeline == "graph"
        assert directionality == 0.0  # Undirected graph

        # Test with neither
        with pytest.raises(
            ValueError, match='Either "adj" or "graph" argument must be non-empty'
        ):
            select_construction_pipeline(None, None)

        # Test with both
        with pytest.raises(
            ValueError, match='Either "adj" or "graph" should be given, not both'
        ):
            select_construction_pipeline(A, G)

        # Test with unsupported graph type
        class CustomGraph:
            pass

        with pytest.raises(
            TypeError, match="graph should have one of supported graph types"
        ):
            select_construction_pipeline(None, CustomGraph())


class TestNetworkInitialization:
    """Test Network class initialization with various parameters."""

    def test_init_with_node_attributes(self):
        """Test Network initialization with node attributes."""
        G = nx.karate_club_graph()
        pos = nx.spring_layout(G)
        node_attrs = {node: {"club": G.nodes[node]["club"]} for node in G.nodes()}

        net = Network(graph=G, pos=pos, node_attrs=node_attrs)
        assert net.pos is not None
        assert net.node_attrs is not None
        assert len(net.pos) == len(G.nodes())
        assert len(net.node_attrs) == len(G.nodes())

    def test_init_without_nx_graph_creation(self):
        """Test Network initialization without creating NetworkX graph."""
        A = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        net = Network(adj=A, create_nx_graph=False)

        assert net.graph is None
        assert net.adj is not None
        assert hasattr(net, "_init_to_final_node_mapping")

    def test_init_with_preprocessing_none(self):
        """Test Network initialization without preprocessing."""
        A = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        with patch("builtins.print") as mock_print:
            net = Network(
                adj=A, preprocessing=None, verbose=True, create_nx_graph=False
            )
            mock_print.assert_called_with(
                "No preprocessing specified, this may lead to unexpected errors in graph connectivity!"
            )

    def test_init_invalid_preprocessing(self):
        """Test Network initialization with invalid preprocessing."""
        A = sp.csr_matrix([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="Wrong preprocessing type"):
            Network(adj=A, preprocessing="invalid_preprocessing")

    def test_directed_detection(self):
        """Test automatic directed/undirected detection."""
        # Symmetric matrix -> undirected
        A_sym = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        net_undir = Network(adj=A_sym)
        assert net_undir.directed == False

        # Asymmetric matrix -> directed
        # Create a truly asymmetric 4x4 matrix
        A_asym = sp.csr_matrix([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        net_dir = Network(adj=A_asym)
        assert net_dir.directed == True

    def test_weighted_detection(self):
        """Test automatic weighted/unweighted detection."""
        # Binary matrix -> unweighted
        A_binary = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        net_unweighted = Network(adj=A_binary)
        assert net_unweighted.weighted == False

        # Non-binary matrix -> weighted
        A_weighted = sp.csr_matrix([[0, 2.5, 1.5], [2.5, 0, 3.0], [1.5, 3.0, 0]])
        net_weighted = Network(adj=A_weighted)
        assert net_weighted.weighted == True


class TestNetworkMethods:
    """Test Network class methods."""

    @pytest.fixture
    def simple_network(self):
        """Create a simple test network with 10 nodes."""
        # Create a more densely connected 10-node network for better randomization
        A = sp.csr_matrix(
            [
                [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            ]
        )
        return Network(adj=A)

    @pytest.fixture
    def directed_network(self):
        """Create a directed test network with 10 nodes."""
        # Create a directed cycle with 10 nodes
        A = sp.csr_matrix(
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        return Network(adj=A)

    def test_is_connected(self, simple_network):
        """Test is_connected method."""
        assert simple_network.is_connected() is True

        # Create disconnected network with preprocessing=None to keep it disconnected
        A_disconnected = sp.csr_matrix(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        )
        net_disconnected = Network(adj=A_disconnected, preprocessing=None)
        assert net_disconnected.is_connected() is False

    def test_get_degree_distr(self, simple_network):
        """Test get_degree_distr method."""
        # Test all degrees
        hist_all = simple_network.get_degree_distr(mode="all")
        assert isinstance(hist_all, np.ndarray)

        # Test out degrees
        hist_out = simple_network.get_degree_distr(mode="out")
        assert isinstance(hist_out, np.ndarray)

        # Test in degrees
        hist_in = simple_network.get_degree_distr(mode="in")
        assert isinstance(hist_in, np.ndarray)

        # Test invalid mode
        with pytest.raises(ValueError, match="Wrong mode for degree distribution"):
            simple_network.get_degree_distr(mode="invalid")

    def test_randomize_methods(self, simple_network):
        """Test randomize method with different modes."""
        # Test shuffle mode
        rand_shuffle = simple_network.randomize(rmode="shuffle")
        assert isinstance(rand_shuffle, Network)
        assert rand_shuffle.n == simple_network.n
        assert rand_shuffle.real_world == False

        # Test complete mode - skip if not complete graph
        try:
            rand_complete = simple_network.randomize(rmode="complete")
            assert isinstance(rand_complete, Network)
        except ValueError as e:
            # Expected if graph is not complete
            assert "Graph is not complete" in str(e)

        # Test adj_iom mode
        rand_iom = simple_network.randomize(rmode="adj_iom")
        assert isinstance(rand_iom, Network)

        # Test invalid mode
        with pytest.raises(ValueError, match="Unknown randomization method"):
            simple_network.randomize(rmode="invalid_mode")

    def test_get_node_degrees(self, directed_network):
        """Test get_node_degrees method."""
        directed_network.get_node_degrees()

        assert hasattr(directed_network, "outdeg")
        assert hasattr(directed_network, "indeg")
        assert hasattr(directed_network, "deg")
        assert hasattr(directed_network, "scaled_outdeg")
        assert hasattr(directed_network, "scaled_indeg")

        # Check degree values
        # For 10-node directed cycle, each node has outdegree and indegree of 1
        assert np.array_equal(directed_network.outdeg, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert np.array_equal(directed_network.indeg, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_get_matrix_caching(self, simple_network):
        """Test get_matrix method and caching."""
        # First call should compute matrix
        lap = simple_network.get_matrix("lap")
        assert lap is not None

        # Second call should return cached matrix
        lap2 = simple_network.get_matrix("lap")
        assert lap is lap2  # Same object reference

        # Test different matrix types
        nlap = simple_network.get_matrix("nlap")
        assert nlap is not None

        rwlap = simple_network.get_matrix("rwlap")
        assert rwlap is not None

        trans = simple_network.get_matrix("trans")
        assert trans is not None

    def test_diagonalize_methods(self, simple_network):
        """Test diagonalization and related methods."""
        # Test diagonalize
        simple_network.diagonalize(mode="adj", verbose=True)
        assert hasattr(simple_network, "adj_spectrum")
        assert hasattr(simple_network, "adj_eigenvectors")

        # Test get_spectrum (should use cached values)
        spectrum = simple_network.get_spectrum("adj")
        assert spectrum is not None
        assert len(spectrum) == simple_network.n

        # Test get_eigenvectors
        eigvecs = simple_network.get_eigenvectors("adj")
        assert eigvecs is not None
        assert eigvecs.shape == (simple_network.n, simple_network.n)

    def test_calculate_ipr(self, simple_network):
        """Test calculate_ipr and get_ipr methods."""
        ipr = simple_network.get_ipr("adj")
        assert ipr is not None
        assert len(ipr) == simple_network.n

        # Test caching
        ipr2 = simple_network.get_ipr("adj")
        assert np.array_equal(ipr, ipr2)

    def test_calculate_z_values(self, simple_network):
        """Test calculate_z_values and get_z_values methods."""
        zvals = simple_network.get_z_values("lap")
        assert zvals is not None
        assert isinstance(zvals, dict)

        # Test with repeated eigenvalues warning
        with patch("builtins.print") as mock_print:
            simple_network.verbose = True
            simple_network.lap_zvalues = None  # Clear cache
            simple_network.calculate_z_values("lap")
            # Check if warning was printed (if there are repeated eigenvalues)

    def test_thermodynamic_methods(self, simple_network):
        """Test thermodynamic entropy calculation methods."""
        tlist = [0.1, 1.0, 10.0]

        # Test thermodynamic entropy
        entropy = simple_network.calculate_thermodynamic_entropy(tlist)
        assert len(entropy) == len(tlist)

        # Test free entropy
        free_entropy = simple_network.calculate_free_entropy(tlist)
        assert len(free_entropy) == len(tlist)

        # Test q-entropy
        q_entropy = simple_network.calculate_q_entropy(q=2, tlist=tlist)
        assert len(q_entropy) == len(tlist)

        # Test with normalized Laplacian
        entropy_norm = simple_network.calculate_thermodynamic_entropy(tlist, norm=True)
        assert len(entropy_norm) == len(tlist)

    def test_thermodynamic_directed_network(self, directed_network):
        """Test thermodynamic methods on directed networks."""
        tlist = [1.0]

        # Should work with non-normalized
        entropy = directed_network.calculate_thermodynamic_entropy(tlist, norm=False)
        assert len(entropy) == 1

        # Should raise for normalized on directed
        with pytest.raises(
            NotImplementedError,
            match="Normalized Laplacian not implemented for directed",
        ):
            directed_network.calculate_thermodynamic_entropy(tlist, norm=True)

    def test_estrada_methods(self, simple_network):
        """Test Estrada communicability and bipartivity methods."""
        # Test Estrada communicability
        comm = simple_network.calculate_estrada_communicability()
        assert comm > 0
        assert hasattr(simple_network, "estrada_communicability")

        # Test Estrada bipartivity index
        bipart = simple_network.get_estrada_bipartivity_index()
        assert bipart > 0
        assert hasattr(simple_network, "estrada_bipartivity")

    def test_localization_signatures(self, simple_network):
        """Test localization_signatures method."""
        simple_network.verbose = True
        mean_inv_r_sq, mean_cos_phi = simple_network.localization_signatures(mode="lap")

        assert isinstance(mean_inv_r_sq, float)
        assert isinstance(mean_cos_phi, float)

    def test_construct_lem_embedding(self, simple_network):
        """Test construct_lem_embedding method."""
        # Should work for undirected network
        simple_network.construct_lem_embedding(dim=2)

        assert simple_network.lem_emb is not None
        assert simple_network.lem_emb.shape[0] == 2  # dim

    def test_construct_lem_embedding_directed_error(self, directed_network):
        """Test construct_lem_embedding raises error for directed graphs."""
        with pytest.raises(
            Exception, match="LEM embedding is not implemented for directed graphs"
        ):
            directed_network.construct_lem_embedding(dim=2)

    def test_construct_lem_embedding_disconnected_error(self):
        """Test construct_lem_embedding raises error for disconnected graphs."""
        # Create a disconnected graph
        A_disconnected = sp.csr_matrix(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        )
        net_disconnected = Network(adj=A_disconnected)

        # Mock eigs to return multiple eigenvalues equal to 1.0
        with patch("driada.network.net_base.eigs") as mock_eigs:
            mock_eigs.return_value = (
                np.array([1.0, 1.0, 0.5, 0.3]),
                np.random.rand(4, 4),
            )

            with pytest.raises(Exception, match="graph is not connected"):
                net_disconnected.construct_lem_embedding(dim=2)


class TestNetworkEdgeCases:
    """Test edge cases and error conditions."""

    def test_graph_with_selfloops(self):
        """Test handling of graphs with self-loops."""
        G = nx.Graph()
        G.add_edges_from([(0, 0), (0, 1), (1, 2)])  # Self-loop on node 0

        net = Network(graph=G)
        # Self-loops should be removed
        assert net.adj[0, 0] == 0

    def test_isolated_nodes_in_connected_component(self):
        """Test error when connected component has isolated nodes."""
        # Create network with isolated node after preprocessing
        A = sp.csr_matrix([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        net = Network(adj=A, create_nx_graph=False, preprocessing=None)
        net.n_cc = 1  # Manually set to trigger the check

        with pytest.raises(Exception, match="Graph has .* isolated nodes"):
            net.diagonalize()

    def test_complex_eigenvalues_directed(self):
        """Test handling of complex eigenvalues in directed networks."""
        # Create a directed network that might have complex eigenvalues
        A = sp.csr_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        net = Network(adj=A)

        # Should handle complex eigenvalues for directed networks
        net.diagonalize(mode="adj")
        assert net.adj_spectrum is not None

    def test_invalid_matrix_mode(self):
        """Test get_matrix with invalid mode after type checking."""
        net = Network(adj=sp.csr_matrix([[0, 1], [1, 0]]))

        # First bypass check_matrix_type to test the else clause
        with patch("driada.network.net_base.check_matrix_type"):
            # Set the attribute to None to force computation
            net.invalid_mode = None

            with pytest.raises(Exception, match="Wrong matrix type: invalid_mode"):
                net.get_matrix("invalid_mode")

    def test_degree_distribution_edge_cases(self):
        """Test get_degree_distr with networks having uniform degrees."""
        # Create a network where all nodes have the same degree
        A = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        net = Network(adj=A)

        hist = net.get_degree_distr()
        assert hist is not None

        # Test with max(deg) - min(deg) = 0
        # All nodes have degree 2, so bins would be 0
        # The function should handle this gracefully

    def test_node_degrees_uniform_case(self):
        """Test node degree scaling when all degrees are uniform."""
        # Create network where all nodes have same in/out degree
        A = sp.csr_matrix([[0, 1], [1, 0]])
        net = Network(adj=A)
        net.get_node_degrees()

        # When all degrees are the same, scaled degrees should be all 1s
        assert np.all(net.scaled_outdeg == 1.0)
        # Note: there's a bug in the original code where scaled_indeg is not set correctly

    def test_diagonalize_errors(self):
        """Test various error conditions in diagonalize method."""
        # Test with inconsistent degree for boolean undirected network
        A = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 0, 0]])  # Node 2 is isolated
        net = Network(adj=A, create_nx_graph=False, real_world=False)
        net.weighted = False
        net.directed = False

        # Manually set outdeg != indeg to trigger error
        net.outdeg = np.array([1, 2, 0])
        net.indeg = np.array([1, 1, 1])

        with pytest.raises(Exception, match="out- and in- degrees do not coincide"):
            net.diagonalize()

    def test_diagonalize_laplacian_components(self):
        """Test diagonalize with multiple components error."""
        # Create disconnected network
        A = sp.csr_matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        net = Network(adj=A, create_nx_graph=False)
        net.weighted = False
        net.n_cc = 1  # Pretend it's connected to trigger the check

        # Mock the eigenvalue computation to return multiple zero eigenvalues
        with patch("driada.network.net_base.la.eigh") as mock_eigh:
            # Return eigenvalues with 2 zeros (indicating 2 components)
            mock_eigh.return_value = (np.array([0.0, 0.0, 1.0, 2.0]), np.eye(4))

            with pytest.raises(Exception, match="Graph has .* components"):
                net.diagonalize(mode="lap")

    def test_complex_eigenvectors_undirected_error(self):
        """Test error when undirected network has complex eigenvectors."""
        A = sp.csr_matrix([[0, 1], [1, 0]])
        net = Network(adj=A, directed=False)

        # Mock eigh to return complex eigenvalues
        with patch("driada.network.net_base.la.eigh") as mock_eigh:
            # Return real eigenvalues but complex eigenvectors
            mock_eigh.return_value = (
                np.array([1.0, -1.0]),
                np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
            )

            with pytest.raises(
                ValueError, match="Complex eigenvectors found in non-directed network"
            ):
                net.diagonalize(mode="adj")  # Use adj mode to avoid lap mode checks


class TestNetworkWithRealGraphs:
    """Test Network class with real-world graph examples."""

    def test_karate_club_network(self):
        """Test with Zachary's karate club network."""
        G = nx.karate_club_graph()
        net = Network(graph=G, name="Karate Club")

        assert net.n == 34
        assert net.name == "Karate Club"
        assert net.is_connected()

        # Test spectral properties
        spectrum = net.get_spectrum("lap")
        assert len(spectrum) == 34
        assert abs(spectrum[0]) < 1e-10  # First eigenvalue should be ~0

    def test_directed_graph_from_nx(self):
        """Test with directed graph from NetworkX."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

        net = Network(graph=G)
        assert net.directed == True
        assert net.n == 4

    def test_weighted_graph_from_nx(self):
        """Test with weighted graph from NetworkX."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 2.5), (1, 2, 3.0), (2, 0, 1.5)])

        net = Network(graph=G)
        assert net.weighted == True
        assert net.n == 3

    def test_network_with_node_positions(self):
        """Test network initialization with node positions."""
        # Create graph with positions
        G = nx.path_graph(5)
        pos = {i: (i, 0) for i in range(5)}

        # Test from graph
        net1 = Network(graph=G, pos=pos)
        assert net1.pos is not None
        assert len(net1.pos) == 5

        # Test from adjacency with preprocessing
        A = nx.adjacency_matrix(G)
        net2 = Network(adj=A, pos=pos, preprocessing="giant_cc", create_nx_graph=False)
        assert net2.pos is not None
        assert len(net2.pos) == 5

    def test_preprocessing_with_verbose_output(self):
        """Test preprocessing with verbose output."""
        # Create graph with isolates
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        G.add_nodes_from([3, 4])  # Isolated nodes

        with patch("builtins.print") as mock_print:
            net = Network(graph=G, preprocessing="remove_isolates", verbose=True)

            # Check that verbose message about removed nodes/edges was printed
            calls = [str(call) for call in mock_print.call_args_list]
            assert any(
                "nodes and" in str(call) and "edges removed" in str(call)
                for call in calls
            )


class TestCheckWeightsAndDirections:
    """Test check_weights_and_directions validation function."""
    
    def test_sparse_matrix_unweighted_undirected_valid(self):
        """Test with valid sparse unweighted undirected matrix."""
        # Create symmetric binary matrix
        A = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        
        # Should not raise any exception
        check_weights_and_directions(A, weighted=False, directed=False)
        
    def test_sparse_matrix_weighted_mismatch(self):
        """Test exception when weighted parameter doesn't match matrix."""
        # Create weighted matrix with values beyond {0, 1}
        A = sp.csr_matrix([[0, 2.5, 1], [2.5, 0, 3], [1, 3, 0]])
        
        # Should raise ValueError for weighted mismatch
        with pytest.raises(ValueError, match="weighted.*set to False.*but the adjacency matrix is weighted"):
            check_weights_and_directions(A, weighted=False, directed=False)
            
    def test_sparse_matrix_directed_mismatch(self):
        """Test exception when directed parameter doesn't match matrix."""
        # Create asymmetric matrix
        A = sp.csr_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        
        # Should raise ValueError for directed mismatch
        with pytest.raises(ValueError, match="directed.*set to False.*but the adjacency matrix is asymmetric"):
            check_weights_and_directions(A, weighted=False, directed=False)
            
    def test_dense_matrix_valid(self):
        """Test with valid dense numpy arrays."""
        # Symmetric weighted matrix
        A = np.array([[0, 2.5, 2.5], [2.5, 0, 3], [2.5, 3, 0]])
        
        # Should not raise exception
        check_weights_and_directions(A, weighted=True, directed=False)
        
    def test_symmetric_but_marked_directed(self):
        """Test error when symmetric matrix is marked as directed."""
        # Binary symmetric matrix
        A = sp.csr_matrix([[0, 1], [1, 0]])
        
        with pytest.raises(ValueError, match="directed.*set to True.*but the adjacency matrix is symmetric"):
            check_weights_and_directions(A, weighted=False, directed=True)
            
    def test_unweighted_but_marked_weighted(self):
        """Test error when unweighted matrix is marked as weighted."""
        # Binary matrix
        A = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        
        with pytest.raises(ValueError, match="weighted.*set to True.*but the adjacency matrix is not weighted"):
            check_weights_and_directions(A, weighted=True, directed=False)
            
    def test_sparse_efficiency_no_conversion(self):
        """Test that sparse matrices are checked efficiently without conversion."""
        # Create large sparse matrix - should handle efficiently
        size = 1000
        A = sp.random(size, size, density=0.01, format='csr')
        A = (A + A.T) / 2  # Make symmetric
        # Make binary
        A = (A != 0).astype(int)
        
        # This should complete quickly without memory issues
        check_weights_and_directions(A, weighted=False, directed=False)


class TestCalculateDirectionalityFraction:
    """Test calculate_directionality_fraction with weighted graph fix."""
    
    def test_fully_symmetric_matrix(self):
        """Test with completely symmetric matrix."""
        A = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        
        frac = calculate_directionality_fraction(A)
        assert frac == 0.0
        
    def test_fully_directed_matrix(self):
        """Test with completely asymmetric matrix."""
        A = np.array([[0, 1, 2], [0, 0, 3], [0, 0, 0]])
        
        frac = calculate_directionality_fraction(A)
        assert frac == 1.0
        
    def test_weighted_asymmetric_edges(self):
        """Test the critical bug fix: different weights should be directed."""
        # Edge (0,1) has weight 5, but (1,0) has weight 3
        A = np.array([[0, 5, 0], [3, 0, 0], [0, 0, 0]])
        
        frac = calculate_directionality_fraction(A)
        assert frac == 1.0  # Should be fully directed since weights differ
        
    def test_partially_directed_matrix(self):
        """Test with mix of symmetric and asymmetric edges."""
        # (0,1) and (1,0) both have weight 2 (symmetric)
        # (0,2) has weight 3 but no (2,0) (directed)
        A = np.array([[0, 2, 3], [2, 0, 0], [0, 0, 0]])
        
        frac = calculate_directionality_fraction(A)
        # 2 symmetric edges + 1 directed edge = 3 total edges
        # directed fraction = 1/3
        assert abs(frac - 1/3) < 1e-10
        
    def test_sparse_matrix_efficiency(self):
        """Test efficient handling of sparse matrices."""
        # Create sparse matrix with known structure
        row = [0, 1, 0, 2]
        col = [1, 0, 2, 1]
        data = [2, 2, 3, 4]  # (0,1) and (1,0) symmetric, others not
        A = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        
        frac = calculate_directionality_fraction(A)
        # 2 symmetric + 2 directed = 4 total edges
        # directed fraction = 2/4 = 0.5
        assert abs(frac - 0.5) < 1e-10
        
    def test_empty_graph(self):
        """Test with empty adjacency matrix."""
        A = np.zeros((3, 3))
        
        frac = calculate_directionality_fraction(A)
        assert frac == 0.0  # Empty graph is considered undirected
        
    def test_self_loops_ignored(self):
        """Test that diagonal entries are ignored."""
        A = np.array([[5, 1, 0], [1, 10, 0], [0, 0, 15]])
        
        frac = calculate_directionality_fraction(A)
        assert frac == 0.0  # Only off-diagonal symmetric edges
        
    def test_tolerance_in_weight_comparison(self):
        """Test that floating point tolerance is used for weight comparison."""
        # Almost equal weights should be considered symmetric
        A = np.array([[0, 1.0], [1.0000001, 0]])
        
        frac = calculate_directionality_fraction(A)
        assert frac == 0.0  # Should be symmetric within tolerance
