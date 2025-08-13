"""Tests for correlation_matrix function."""

import numpy as np
from driada.utils.data import correlation_matrix


class TestCorrelationMatrix:
    """Test correlation_matrix function."""

    def test_identity_correlation(self):
        """Test that identical rows have correlation 1."""
        A = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        corr = correlation_matrix(A)

        assert corr.shape == (2, 2)
        # Identical rows should have perfect correlation
        expected = np.array([[1, 1], [1, 1]])
        assert np.allclose(corr, expected)

    def test_perfect_correlation(self):
        """Test perfect positive and negative correlations."""
        A = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 4, 6, 8, 10],  # Perfect positive correlation
                [5, 4, 3, 2, 1],  # Perfect negative correlation with first
            ]
        )
        corr = correlation_matrix(A)

        expected = np.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])

        assert np.allclose(corr, expected)

    def test_matches_numpy_corrcoef(self):
        """Test that our implementation matches numpy.corrcoef."""
        # Random data
        np.random.seed(42)
        A = np.random.randn(5, 20)

        corr_ours = correlation_matrix(A)
        corr_numpy = np.corrcoef(A)

        assert np.allclose(corr_ours, corr_numpy)

    def test_single_observation(self):
        """Test edge case with single observation."""
        A = np.array([[1], [2], [3]])
        corr = correlation_matrix(A)

        # With single observation, correlation is undefined (NaN)
        assert corr.shape == (3, 3)
        assert np.all(np.isnan(corr))

    def test_zero_variance(self):
        """Test variables with zero variance."""
        A = np.array(
            [
                [1, 1, 1, 1, 1],  # Zero variance
                [1, 2, 3, 4, 5],  # Non-zero variance
                [2, 2, 2, 2, 2],  # Zero variance
            ]
        )
        corr = correlation_matrix(A)

        # Zero variance variables should have 1.0 on diagonal
        # and NaN correlations with other variables
        assert corr[0, 0] == 1.0
        assert corr[1, 1] == 1.0
        assert corr[2, 2] == 1.0
        assert np.isnan(corr[0, 1])
        assert np.isnan(corr[0, 2])
        assert np.isnan(corr[1, 0])
        assert np.isnan(corr[1, 2])
        assert np.isnan(corr[2, 0])
        assert np.isnan(corr[2, 1])

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors have zero correlation."""
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        corr = correlation_matrix(A)

        assert corr.shape == (2, 2)
        assert corr[0, 0] == 1.0
        assert corr[1, 1] == 1.0
        assert np.isclose(corr[0, 1], 0.0, atol=1e-10)
        assert np.isclose(corr[1, 0], 0.0, atol=1e-10)

    def test_partial_correlation(self):
        """Test intermediate correlation values."""
        A = np.array([[1, 2, 3, 4, 5], [1, 3, 2, 5, 4]])  # Partially correlated
        corr = correlation_matrix(A)

        assert corr.shape == (2, 2)
        assert corr[0, 0] == 1.0
        assert corr[1, 1] == 1.0
        assert 0 < corr[0, 1] < 1  # Positive but not perfect correlation
        assert corr[0, 1] == corr[1, 0]  # Symmetric
