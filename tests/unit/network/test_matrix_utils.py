"""Tests for network matrix utilities."""

import numpy as np
import pytest
import scipy.sparse as sp

from driada.network.matrix_utils import (
    get_laplacian,
    get_inv_sqrt_diag_matrix,
    get_norm_laplacian,
    get_inv_diag_matrix,
    get_rw_laplacian,
    get_trans_matrix,
    get_symmetry_index,
    _is_sparse,
)


class TestMatrixUtilsBranching:
    """Test that matrix utility functions work with both dense and sparse matrices."""
    
    @pytest.fixture
    def simple_graph_adj(self):
        """Create a simple 4-node path graph adjacency matrix: 0-1-2-3"""
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        return adj
    
    @pytest.fixture
    def cycle_graph_adj(self):
        """Create a 4-node cycle graph adjacency matrix: 0-1-2-3-0"""
        adj = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=float)
        return adj
    
    def test_is_sparse(self, simple_graph_adj):
        """Test sparse matrix detection."""
        assert not _is_sparse(simple_graph_adj)
        assert _is_sparse(sp.csr_matrix(simple_graph_adj))
        assert _is_sparse(sp.csc_matrix(simple_graph_adj))
        assert _is_sparse(sp.dok_matrix(simple_graph_adj))
    
    def test_get_laplacian(self, simple_graph_adj):
        """Test Laplacian computation."""
        # Dense version
        L_dense = get_laplacian(simple_graph_adj)
        assert isinstance(L_dense, np.ndarray)
        assert L_dense.shape == (4, 4)
        # Row sums should be zero
        np.testing.assert_allclose(L_dense.sum(axis=1), 0, atol=1e-10)
        
        # Sparse version
        L_sparse = get_laplacian(sp.csr_matrix(simple_graph_adj))
        assert sp.issparse(L_sparse)
        assert L_sparse.shape == (4, 4)
        # Results should match
        np.testing.assert_allclose(L_dense, L_sparse.toarray())
    
    def test_get_inv_sqrt_diag_matrix(self, simple_graph_adj):
        """Test inverse square root degree matrix."""
        # Dense version
        D_dense = get_inv_sqrt_diag_matrix(simple_graph_adj)
        assert isinstance(D_dense, np.ndarray)
        
        # Check diagonal values
        degrees = simple_graph_adj.sum(axis=0)
        expected_diag = 1.0 / np.sqrt(degrees)
        np.testing.assert_allclose(np.diag(D_dense), expected_diag)
        
        # Sparse version
        D_sparse = get_inv_sqrt_diag_matrix(sp.csr_matrix(simple_graph_adj))
        assert sp.issparse(D_sparse)
        np.testing.assert_allclose(D_dense, D_sparse.toarray())
    
    def test_get_norm_laplacian(self, simple_graph_adj):
        """Test normalized Laplacian computation."""
        # Dense version
        L_dense = get_norm_laplacian(simple_graph_adj)
        assert isinstance(L_dense, np.ndarray)
        
        # Check eigenvalues are in [0, 2] for connected graphs
        eigenvals = np.linalg.eigvalsh(L_dense)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
        assert np.all(eigenvals <= 2 + 1e-10)
        
        # Sparse version
        L_sparse = get_norm_laplacian(sp.csr_matrix(simple_graph_adj))
        assert sp.issparse(L_sparse)
        np.testing.assert_allclose(L_dense, L_sparse.toarray(), rtol=1e-10)
    
    def test_get_inv_diag_matrix(self, simple_graph_adj):
        """Test inverse degree matrix."""
        # Dense version
        D_inv_dense = get_inv_diag_matrix(simple_graph_adj)
        assert isinstance(D_inv_dense, np.ndarray)
        
        # Check it's actually the inverse
        degrees = simple_graph_adj.sum(axis=0)
        D = np.diag(degrees)
        np.testing.assert_allclose(D_inv_dense @ D, np.eye(4), atol=1e-10)
        
        # Sparse version
        D_inv_sparse = get_inv_diag_matrix(sp.csr_matrix(simple_graph_adj))
        assert sp.issparse(D_inv_sparse)
        np.testing.assert_allclose(D_inv_dense, D_inv_sparse.toarray())
    
    def test_get_trans_matrix(self, simple_graph_adj):
        """Test transition matrix computation."""
        # Dense version
        T_dense = get_trans_matrix(simple_graph_adj)
        assert isinstance(T_dense, np.ndarray)
        
        # Row sums should be 1 for non-isolated nodes
        row_sums = T_dense.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0)
        
        # Sparse version
        T_sparse = get_trans_matrix(sp.csr_matrix(simple_graph_adj))
        assert sp.issparse(T_sparse)
        np.testing.assert_allclose(T_dense, T_sparse.toarray())
    
    def test_get_rw_laplacian(self, simple_graph_adj):
        """Test random walk Laplacian."""
        # Dense version
        L_rw_dense = get_rw_laplacian(simple_graph_adj)
        assert isinstance(L_rw_dense, np.ndarray)
        
        # Should equal I - T
        T = get_trans_matrix(simple_graph_adj)
        expected = np.eye(4) - T
        np.testing.assert_allclose(L_rw_dense, expected)
        
        # Sparse version
        L_rw_sparse = get_rw_laplacian(sp.csr_matrix(simple_graph_adj))
        assert sp.issparse(L_rw_sparse)
        np.testing.assert_allclose(L_rw_dense, L_rw_sparse.toarray())
    
    def test_get_symmetry_index(self, simple_graph_adj):
        """Test symmetry index calculation."""
        # Symmetric matrix should have index 1
        assert get_symmetry_index(simple_graph_adj) == 1.0
        assert get_symmetry_index(sp.csr_matrix(simple_graph_adj)) == 1.0
        
        # Completely asymmetric matrix (directed cycle)
        asym_adj = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ])
        assert get_symmetry_index(asym_adj) == 0.0
        assert get_symmetry_index(sp.csr_matrix(asym_adj)) == 0.0
        
        # Partially symmetric matrix
        partial_adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        # 6 edges have symmetric counterpart out of 7 total edges
        expected = 6.0 / 7.0
        assert abs(get_symmetry_index(partial_adj) - expected) < 1e-10
        assert abs(get_symmetry_index(sp.csr_matrix(partial_adj)) - expected) < 1e-10
        
        # Matrix with diagonal elements
        diag_adj = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 0, 1]
        ])
        # 4 edges have counterpart (including self-loops) out of 5 total
        expected_diag = 4.0 / 5.0
        assert abs(get_symmetry_index(diag_adj) - expected_diag) < 1e-10
        
        # Empty matrix (no edges) - considered fully symmetric
        empty_adj = np.zeros((5, 5))
        assert get_symmetry_index(empty_adj) == 1.0
        assert get_symmetry_index(sp.csr_matrix(empty_adj)) == 1.0
        
        # Upper triangular matrix (no symmetric edges)
        upper_adj = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        assert get_symmetry_index(upper_adj) == 0.0
        assert get_symmetry_index(sp.csr_matrix(upper_adj)) == 0.0
    
    def test_isolated_nodes_handling(self):
        """Test functions handle isolated nodes correctly."""
        # Graph with isolated node
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],  # isolated
            [0, 0, 0, 0]   # isolated
        ], dtype=float)
        
        # Test all functions don't crash with isolated nodes
        for matrix_type in [adj, sp.csr_matrix(adj)]:
            D_inv_sqrt = get_inv_sqrt_diag_matrix(matrix_type)
            D_inv = get_inv_diag_matrix(matrix_type)
            T = get_trans_matrix(matrix_type)
            
            # Check isolated nodes have 0 in inverse matrices
            if sp.issparse(D_inv_sqrt):
                assert D_inv_sqrt[2, 2] == 0
                assert D_inv_sqrt[3, 3] == 0
            else:
                assert D_inv_sqrt[2, 2] == 0
                assert D_inv_sqrt[3, 3] == 0
    
    def test_type_preservation(self, simple_graph_adj):
        """Test that functions preserve input type (dense->dense, sparse->sparse)."""
        funcs = [
            get_laplacian,
            get_inv_sqrt_diag_matrix,
            get_norm_laplacian,
            get_inv_diag_matrix,
            get_rw_laplacian,
            get_trans_matrix,
        ]
        
        for func in funcs:
            # Dense input should give dense output
            result_dense = func(simple_graph_adj)
            assert isinstance(result_dense, np.ndarray), f"{func.__name__} failed to return ndarray"
            
            # Sparse input should give sparse output
            result_sparse = func(sp.csr_matrix(simple_graph_adj))
            assert sp.issparse(result_sparse), f"{func.__name__} failed to return sparse matrix"
    
    def test_different_sparse_formats(self, simple_graph_adj):
        """Test functions work with different sparse matrix formats."""
        formats = [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix]
        
        # Just test with Laplacian as representative
        L_dense = get_laplacian(simple_graph_adj)
        
        for sparse_format in formats:
            sparse_adj = sparse_format(simple_graph_adj)
            L_sparse = get_laplacian(sparse_adj)
            assert sp.issparse(L_sparse)
            np.testing.assert_allclose(L_dense, L_sparse.toarray())