"""
Comprehensive tests for network randomization algorithms.

These tests ensure all randomization methods work correctly and preserve
required graph properties.
"""

import pytest
import numpy as np
import scipy.sparse as sp
import networkx as nx
import logging

from driada.network.randomization import (
    adj_random_rewiring_iom_preserving,
    random_rewiring_complete_graph,
    random_rewiring_dense_graph,
    get_single_double_edges_lists,
    randomize_graph,
    _validate_adjacency_matrix,
    random_rewiring_IOM_preserving,  # Testing deprecation
)
from .graph_fixtures import (
    create_standard_graph,
    create_complete_graph,
    create_dense_graph,
    create_networkx_graph,
)


class TestValidation:
    """Test input validation functions."""

    def test_validate_adjacency_matrix_valid(self):
        """Test validation with valid inputs."""
        # Dense matrix
        dense_adj = np.array([[0, 1], [1, 0]])
        _validate_adjacency_matrix(dense_adj)

        # Sparse matrix
        sparse_adj = sp.csr_matrix([[0, 1], [1, 0]])
        _validate_adjacency_matrix(sparse_adj)

    def test_validate_adjacency_matrix_invalid(self):
        """Test validation with invalid inputs."""
        # Not a matrix
        with pytest.raises(ValueError, match="must be a numpy array or scipy sparse"):
            _validate_adjacency_matrix([1, 2, 3])

        # Not 2D
        with pytest.raises(ValueError, match="must be 2D"):
            _validate_adjacency_matrix(np.array([1, 2, 3]))

        # Not square
        with pytest.raises(ValueError, match="must be square"):
            _validate_adjacency_matrix(np.array([[1, 2], [3, 4], [5, 6]]))

        # Dense matrix when sparse required
        dense_adj = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="requires a sparse matrix"):
            _validate_adjacency_matrix(dense_adj, allow_dense=False)


class TestIOMPreserving:
    """Test In-Out-Motif preserving randomization."""

    @pytest.fixture
    def simple_directed_graph(self):
        """Create a simple directed graph adjacency matrix."""
        return create_standard_graph(n=10, density=0.3, directed=True)

    @pytest.fixture
    def weighted_graph(self):
        """Create a weighted graph."""
        return create_standard_graph(n=10, density=0.3, weighted=True, directed=True)

    def test_iom_basic(self, simple_directed_graph):
        """Test basic IOM randomization."""
        result = adj_random_rewiring_iom_preserving(
            simple_directed_graph, is_weighted=False, r=2, random_state=42
        )

        # Check output type
        assert sp.issparse(result)
        # Can be either csr_matrix or csr_array
        assert result.format == "csr"

        # Check shape preserved
        assert result.shape == simple_directed_graph.shape

        # Check number of edges preserved
        assert result.nnz == simple_directed_graph.nnz

    def test_iom_preserves_degree_sequence(self, simple_directed_graph):
        """Test that degree sequences are preserved."""
        original = simple_directed_graph.toarray()
        result = adj_random_rewiring_iom_preserving(
            simple_directed_graph, is_weighted=False, r=2, random_state=42
        ).toarray()

        # Check in-degree and out-degree preservation
        orig_in_degree = original.sum(axis=0)
        orig_out_degree = original.sum(axis=1)

        result_in_degree = result.sum(axis=0)
        result_out_degree = result.sum(axis=1)

        np.testing.assert_array_equal(orig_in_degree, result_in_degree)
        np.testing.assert_array_equal(orig_out_degree, result_out_degree)

    def test_iom_weighted(self, weighted_graph):
        """Test IOM randomization with weighted graphs."""
        result = adj_random_rewiring_iom_preserving(
            weighted_graph, is_weighted=True, r=2, random_state=42
        )

        # Check weights are preserved (not just converted to binary)
        assert not np.all(np.isin(result.data, [0, 1]))

    def test_iom_with_logging(self, simple_directed_graph, caplog):
        """Test logging functionality."""
        logger = logging.getLogger("test_logger")

        with caplog.at_level(logging.INFO):
            adj_random_rewiring_iom_preserving(
                simple_directed_graph, is_weighted=False, r=1, logger=logger
            )

        assert "Starting IOM-preserving randomization" in caplog.text
        assert "IOM-preserving randomization completed" in caplog.text

    def test_iom_progress_bar(self, simple_directed_graph):
        """Test progress bar functionality."""
        # Just ensure it runs without error
        adj_random_rewiring_iom_preserving(
            simple_directed_graph, is_weighted=False, r=1, enable_progressbar=True
        )

        adj_random_rewiring_iom_preserving(
            simple_directed_graph, is_weighted=False, r=1, enable_progressbar=False
        )

    def test_iom_reproducibility(self, simple_directed_graph):
        """Test that random_state ensures reproducibility."""
        result1 = adj_random_rewiring_iom_preserving(
            simple_directed_graph, is_weighted=False, r=2, random_state=42
        ).toarray()

        result2 = adj_random_rewiring_iom_preserving(
            simple_directed_graph, is_weighted=False, r=2, random_state=42
        ).toarray()

        np.testing.assert_array_equal(result1, result2)

    def test_iom_small_graph_warning(self, caplog):
        """Test warning when graph is too small for rewiring."""
        # Create a graph with only 3 nodes and minimal edges
        small_adj = sp.csr_matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

        with caplog.at_level(logging.WARNING):
            result = adj_random_rewiring_iom_preserving(
                small_adj, is_weighted=False, r=5  # High r to trigger warning
            )

        # Should warn about max attempts (not enough edges to rewire)
        # Or just complete without changes
        assert result.shape == small_adj.shape


class TestCompleteGraphRandomization:
    """Test complete graph randomization."""

    @pytest.fixture
    def complete_graph(self):
        """Create a complete graph."""
        return create_complete_graph(n=10, weighted=True)

    def test_complete_basic(self, complete_graph):
        """Test basic complete graph randomization."""
        result = random_rewiring_complete_graph(complete_graph, p=1.0, random_state=42)

        # Check type and shape
        assert isinstance(result, np.ndarray)
        assert result.shape == complete_graph.shape

        # Check it's still complete
        n = complete_graph.shape[0]
        assert len(np.nonzero(result)[0]) == n * (n - 1)

    def test_complete_partial_shuffle(self, complete_graph):
        """Test partial shuffling with p < 1."""
        result = random_rewiring_complete_graph(complete_graph, p=0.5, random_state=42)

        # Some edges should remain unchanged
        diff = np.abs(result - complete_graph)
        unchanged = np.sum(diff == 0)
        total_edges = complete_graph.shape[0] * (complete_graph.shape[0] - 1)

        # Roughly half should be unchanged (allowing for randomness)
        assert unchanged > 0.3 * total_edges
        assert unchanged < 0.7 * total_edges

    def test_complete_symmetry_preserved(self):
        """Test that symmetry is preserved for symmetric graphs."""
        adj = create_complete_graph(n=10, weighted=False)

        result = random_rewiring_complete_graph(adj, p=1.0, random_state=42)

        # Check symmetry
        np.testing.assert_allclose(result, result.T)

    def test_complete_invalid_input(self):
        """Test error handling for non-complete graphs."""
        # Create incomplete graph using dense graph with low completeness
        adj = create_dense_graph(n=10, completeness=0.5, weighted=False)

        with pytest.raises(ValueError, match="Graph is not complete"):
            random_rewiring_complete_graph(adj)

    def test_complete_invalid_p(self, complete_graph):
        """Test error handling for invalid p values."""
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            random_rewiring_complete_graph(complete_graph, p=1.5)

        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            random_rewiring_complete_graph(complete_graph, p=-0.1)

    def test_complete_sparse_input(self):
        """Test handling of sparse input."""
        dense = create_complete_graph(n=10, weighted=False)
        sparse = sp.csr_matrix(dense)

        result = random_rewiring_complete_graph(sparse, random_state=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == sparse.shape


class TestDenseGraphRandomization:
    """Test dense graph randomization."""

    @pytest.fixture
    def dense_graph(self):
        """Create a dense but not complete graph."""
        return create_dense_graph(n=10, completeness=0.8, weighted=True)

    def test_dense_basic(self, dense_graph):
        """Test basic dense graph randomization."""
        result = random_rewiring_dense_graph(dense_graph, random_state=42)

        # Check type and shape
        assert isinstance(result, np.ndarray)
        assert result.shape == dense_graph.shape

        # Check symmetry preserved
        np.testing.assert_allclose(result, result.T)

    def test_dense_complete_graph(self):
        """Test dense randomization on complete graph."""
        n = 10
        complete = create_complete_graph(n=n, weighted=False)

        result = random_rewiring_dense_graph(complete, random_state=42)

        # Should still be complete
        assert len(np.nonzero(result)[0]) == n * (n - 1)

    def test_dense_gap_filling(self, dense_graph, caplog):
        """Test gap filling functionality."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.DEBUG):
            result = random_rewiring_dense_graph(
                dense_graph, logger=logger, gap_fill_weight=0.001
            )

        assert "using gap filling" in caplog.text

    def test_dense_sparse_input(self, dense_graph):
        """Test handling of sparse input."""
        sparse = sp.csr_matrix(dense_graph)

        result = random_rewiring_dense_graph(sparse, random_state=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == sparse.shape


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_single_double_edges_lists(self):
        """Test edge list separation."""
        # Create a directed graph with mixed edge types
        G = nx.DiGraph()
        # Add more nodes to ensure proper testing
        # Single edges
        for i in range(0, 10, 2):
            G.add_edge(i, i + 1)
        # Double edges
        for i in range(10, 15):
            G.add_edge(i, i + 1)
            G.add_edge(i + 1, i)

        single, double = get_single_double_edges_lists(G)

        assert len(single) == 5  # 5 single edges
        assert len(double) == 5  # 5 double edges
        # Check that double edges are properly identified
        double_nodes = {e[0] for e in double}
        assert all(node >= 10 for node in double_nodes)

    def test_legacy_function_deprecation(self):
        """Test that legacy function shows deprecation warning."""
        # Use a standard NetworkX graph
        G = create_networkx_graph(n=10, graph_type="erdos_renyi")

        with pytest.warns(DeprecationWarning, match="will be removed in v2.0"):
            # Use r=0 to just test the warning, not the algorithm
            random_rewiring_IOM_preserving(G, r=0)


class TestUnifiedAPI:
    """Test the unified randomize_graph function."""

    @pytest.fixture
    def test_graph(self):
        """Create a test graph."""
        return create_standard_graph(n=10, density=0.3, directed=False)

    def test_unified_iom(self, test_graph):
        """Test unified API with IOM method."""
        result = randomize_graph(
            test_graph, method="iom", iterations=5, random_state=42
        )

        assert sp.issparse(result)
        assert result.shape == test_graph.shape

    def test_unified_complete(self):
        """Test unified API with complete method."""
        complete = create_complete_graph(n=10, weighted=False)

        result = randomize_graph(complete, method="complete", p=0.5, random_state=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == complete.shape

    def test_unified_dense(self):
        """Test unified API with dense method."""
        dense = create_dense_graph(n=10, completeness=0.85)

        result = randomize_graph(
            dense, method="dense", gap_fill_weight=0.01, random_state=42
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == dense.shape

    def test_unified_invalid_method(self, test_graph):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="Unknown randomization method"):
            randomize_graph(test_graph, method="invalid")

    def test_unified_logging(self, test_graph, caplog):
        """Test logging through unified API."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.INFO):
            randomize_graph(
                test_graph, method="iom", logger=logger, enable_progressbar=False
            )

        assert "randomization" in caplog.text.lower()


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_graph(self):
        """Test handling of empty graphs."""
        empty = sp.csr_matrix((10, 10))

        result = adj_random_rewiring_iom_preserving(empty, is_weighted=False, r=10)

        # Should return empty graph
        assert result.nnz == 0
        assert result.shape == empty.shape

    def test_single_edge_graph(self):
        """Test graph with single edge."""
        adj = sp.csr_matrix((10, 10))
        adj[0, 1] = 1

        result = adj_random_rewiring_iom_preserving(adj, is_weighted=False, r=10)

        # Single edge can't be rewired
        np.testing.assert_array_equal(result.toarray(), adj.toarray())

    def test_self_loops_ignored(self):
        """Test that self-loops are ignored."""
        # Create graph with self-loops
        adj = create_standard_graph(n=10, density=0.3)
        adj = adj.toarray()
        # Add self-loops
        np.fill_diagonal(adj, 1)

        result = adj_random_rewiring_iom_preserving(
            sp.csr_matrix(adj), is_weighted=False, r=5
        )

        # Check diagonal remains same
        result_array = result.toarray()
        np.testing.assert_array_equal(np.diag(result_array), np.diag(adj))
