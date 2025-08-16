"""Test turn_to_partially_directed function with both dense and sparse matrices."""

import numpy as np
import scipy.sparse as sp
import pytest

from driada.network.matrix_utils import turn_to_partially_directed


class TestTurnToPartiallyDirected:
    """Test suite for turn_to_partially_directed function."""

    def test_dense_fully_undirected(self):
        """Test with dense matrix, directed=0."""
        # Create symmetric dense matrix
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        
        result = turn_to_partially_directed(A, directed=0.0, weighted=0)
        
        # Should return sparse matrix
        assert sp.issparse(result)
        # Should be symmetric
        assert np.allclose(result.toarray(), result.toarray().T)
        # Should have same structure
        assert np.array_equal((result != 0).toarray(), (A != 0))

    def test_sparse_fully_undirected(self):
        """Test with sparse matrix, directed=0."""
        # Create symmetric sparse matrix
        A_dense = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        A = sp.csr_matrix(A_dense)
        
        result = turn_to_partially_directed(A, directed=0.0, weighted=0)
        
        # Should return sparse matrix
        assert sp.issparse(result)
        # Should be symmetric
        assert np.allclose(result.toarray(), result.toarray().T)
        # Should have same non-zero pattern (excluding diagonal)
        result_nonzero = (result != 0).toarray()
        A_nonzero = (A_dense != 0)
        np.fill_diagonal(A_nonzero, False)  # Ignore diagonal
        assert np.array_equal(result_nonzero, A_nonzero)

    def test_dense_fully_directed(self):
        """Test with dense matrix, directed=1.0."""
        # Create symmetric dense matrix
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        
        # Set random seed for reproducibility
        np.random.seed(42)
        result = turn_to_partially_directed(A, directed=1.0, weighted=0)
        
        # Result is now sparse
        assert sp.issparse(result)
        # Should be fully asymmetric (no symmetric pairs)
        result_dense = result.toarray()
        assert not np.allclose(result_dense, result_dense.T)
        # Should have removed some edges
        assert result.nnz < np.sum(A != 0)

    def test_sparse_fully_directed(self):
        """Test with sparse matrix, directed=1.0."""
        # Create symmetric sparse matrix
        A_dense = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        A = sp.csr_matrix(A_dense)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        result = turn_to_partially_directed(A, directed=1.0, weighted=0)
        
        # Should return sparse matrix
        assert sp.issparse(result)
        # Should be fully asymmetric
        result_dense = result.toarray()
        assert not np.allclose(result_dense, result_dense.T)
        # Should have removed some edges
        assert result.nnz < A.nnz

    def test_dense_partial_directed(self):
        """Test with dense matrix, directed=0.5."""
        # Create symmetric dense matrix
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        
        # Set random seed for reproducibility
        np.random.seed(42)
        result = turn_to_partially_directed(A, directed=0.5, weighted=0)
        
        # Result is now sparse
        assert sp.issparse(result)
        # Should have some asymmetry
        result_dense = result.toarray()
        assert not np.allclose(result_dense, result_dense.T)
        # Should have removed some but not all symmetric pairs
        assert result.nnz < np.sum(A != 0)
        assert result.nnz > np.sum(A != 0) / 2

    def test_sparse_partial_directed(self):
        """Test with sparse matrix, directed=0.5."""
        A_dense = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        A = sp.csr_matrix(A_dense)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        result = turn_to_partially_directed(A, directed=0.5, weighted=0)
        
        # Should return sparse matrix
        assert sp.issparse(result)
        # Should have some asymmetry
        result_dense = result.toarray()
        assert not np.allclose(result_dense, result_dense.T)
        # Should have removed some edges
        # Note: A has 12 edges, after removing diagonal it's 10 edges (5 symmetric pairs)
        # With directed=0.5, we expect to remove edges from about half the pairs
        assert result.nnz < A.nnz

    def test_weighted_matrix(self):
        """Test with weighted matrix."""
        # Create weighted symmetric matrix
        A = np.array([
            [0.0, 2.5, 0.0, 0.0],
            [2.5, 0.0, 1.5, 3.0],
            [0.0, 1.5, 0.0, 0.5],
            [0.0, 3.0, 0.5, 0.0]
        ])
        
        result = turn_to_partially_directed(A, directed=0.0, weighted=1)
        
        # Should preserve weights
        assert np.allclose(result.toarray(), A)

    def test_self_loops_removed(self):
        """Test that self-loops are removed."""
        # Matrix with self-loops
        A = np.array([
            [1, 1, 0],
            [1, 2, 1],
            [0, 1, 3]
        ])
        
        # Dense pathway - now returns sparse
        result_dense = turn_to_partially_directed(A, directed=0.0)
        assert sp.issparse(result_dense)
        assert np.all(np.diag(result_dense.toarray()) == 0)
        
        # Sparse pathway
        result_sparse = turn_to_partially_directed(sp.csr_matrix(A), directed=0.0)
        assert sp.issparse(result_sparse)
        assert np.all(np.diag(result_sparse.toarray()) == 0)

    def test_empty_matrix(self):
        """Test with empty matrix."""
        A = np.zeros((5, 5))
        
        result = turn_to_partially_directed(A, directed=0.5)
        assert sp.issparse(result)
        assert result.nnz == 0

    def test_invalid_input(self):
        """Test with invalid input."""
        with pytest.raises(TypeError):
            turn_to_partially_directed([1, 2, 3], directed=0.5)

    def test_directed_none_equivalent_to_zero(self):
        """Test that directed=None is equivalent to directed=0."""
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        result_none = turn_to_partially_directed(A, directed=None)
        result_zero = turn_to_partially_directed(A, directed=0.0)
        
        assert np.array_equal(result_none.toarray(), result_zero.toarray())

    def test_sparse_formats(self):
        """Test with different sparse matrix formats."""
        A_dense = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        for format in ['csr', 'csc', 'coo', 'lil', 'dok']:
            A_sparse = sp.csr_matrix(A_dense).asformat(format)
            result = turn_to_partially_directed(A_sparse, directed=0.0)
            assert sp.issparse(result)
            assert np.array_equal(result.toarray(), A_dense)