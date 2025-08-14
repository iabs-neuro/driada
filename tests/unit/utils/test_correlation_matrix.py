"""Tests for correlation_matrix and norm_cross_corr functions."""

import numpy as np
import pytest
from driada.utils.data import correlation_matrix, norm_cross_corr


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


class TestNormCrossCorr:
    """Test norm_cross_corr function."""
    
    def test_perfect_correlation_same_signal(self):
        """Test perfect correlation with identical signals."""
        signal = np.sin(np.linspace(0, 2*np.pi, 100))
        result = norm_cross_corr(signal, signal, mode='valid')
        assert np.isclose(result[0], 1.0, atol=1e-10)
    
    def test_perfect_anticorrelation(self):
        """Test perfect anti-correlation with inverted signals."""
        signal1 = np.sin(np.linspace(0, 2*np.pi, 100))
        signal2 = -signal1
        result = norm_cross_corr(signal1, signal2, mode='valid')
        assert np.isclose(result[0], -1.0, atol=1e-10)
    
    def test_no_correlation_random(self):
        """Test low correlation between random signals."""
        np.random.seed(42)
        signal1 = np.random.randn(100)
        signal2 = np.random.randn(100)
        result = norm_cross_corr(signal1, signal2, mode='valid')
        # Random signals should have low correlation
        assert abs(result[0]) < 0.3
    
    def test_shifted_signal_detection(self):
        """Test detection of shifted signals."""
        signal = np.sin(np.linspace(0, 4*np.pi, 200))
        shift = 20
        signal_shifted = np.roll(signal, shift)
        result = norm_cross_corr(signal, signal_shifted, mode='full')
        
        # Find peak correlation
        max_idx = np.argmax(np.abs(result))
        peak_lag = max_idx - len(signal) + 1
        
        # Should find peak near the shift amount
        assert abs(peak_lag - (-shift)) < 5  # Within 5 samples
        assert np.abs(result[max_idx]) > 0.9  # Strong correlation at peak
    
    def test_amplitude_offset_invariance(self):
        """Test invariance to amplitude scaling and offset."""
        signal1 = np.sin(np.linspace(0, 2*np.pi, 100))
        # Scale and offset
        signal2 = 5 * signal1 + 10
        result = norm_cross_corr(signal1, signal2, mode='valid')
        assert np.isclose(result[0], 1.0, atol=1e-10)
    
    def test_constant_signals_edge_case(self):
        """Test handling of constant signals (zero variance)."""
        const1 = np.ones(100)
        const2 = np.ones(100) * 2
        result = norm_cross_corr(const1, const2, mode='valid')
        # Should handle zero variance gracefully
        assert np.isfinite(result[0])
        assert abs(result[0]) < 1e-6  # Should be close to 0
    
    def test_different_modes(self):
        """Test different correlation modes."""
        signal1 = np.array([1, 2, 3, 4, 5])
        signal2 = np.array([2, 3, 4])
        
        # Full mode
        result_full = norm_cross_corr(signal1, signal2, mode='full')
        assert len(result_full) == len(signal1) + len(signal2) - 1
        
        # Valid mode
        result_valid = norm_cross_corr(signal1, signal2, mode='valid')
        assert len(result_valid) == len(signal1) - len(signal2) + 1
        
        # Same mode
        result_same = norm_cross_corr(signal1, signal2, mode='same')
        assert len(result_same) == len(signal1)
    
    def test_empty_signals(self):
        """Test handling of empty signals."""
        empty = np.array([])
        signal = np.array([1, 2, 3])
        
        with pytest.raises(ValueError):
            norm_cross_corr(empty, signal)
    
    def test_single_value_signals(self):
        """Test signals with single value."""
        signal1 = np.array([5])
        signal2 = np.array([3])
        result = norm_cross_corr(signal1, signal2, mode='valid')
        assert len(result) == 1
        assert np.isfinite(result[0])
