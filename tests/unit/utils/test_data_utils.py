"""
Tests for data utility functions.
"""

import pytest
import numpy as np
from driada.utils.data import remove_outliers, create_correlated_gaussian_data, correlation_matrix, rescale


class TestRemoveOutliers:
    """Test cases for remove_outliers function."""
    
    def test_zscore_method(self):
        """Test z-score outlier removal."""
        # Create data with clear outliers
        data = np.array([1, 2, 3, 4, 5, 100, -100, 6, 7, 8])
        
        indices, cleaned = remove_outliers(data, method='zscore', threshold=2.0)
        
        # Should remove 100 and -100
        assert len(cleaned) == 8
        assert 100 not in cleaned
        assert -100 not in cleaned
        assert all(val in cleaned for val in [1, 2, 3, 4, 5, 6, 7, 8])
        
    def test_iqr_method(self):
        """Test IQR outlier removal."""
        # Create data with outliers
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 50])
        
        indices, cleaned = remove_outliers(data, method='iqr', threshold=1.5)
        
        # Should remove 50
        assert 50 not in cleaned
        assert len(cleaned) == 9
        
    def test_mad_method(self):
        """Test MAD outlier removal."""
        # MAD is more robust to outliers
        data = np.array([1, 2, 3, 4, 5, 100])
        
        indices, cleaned = remove_outliers(data, method='mad', threshold=2.5)
        
        # Should remove 100
        assert 100 not in cleaned
        assert len(cleaned) == 5
        
    def test_mad_with_identical_values(self):
        """Test MAD method with all identical values."""
        data = np.array([5, 5, 5, 5, 5])
        
        indices, cleaned = remove_outliers(data, method='mad', threshold=2.5)
        
        # Should keep all values
        assert len(cleaned) == 5
        assert all(val == 5 for val in cleaned)
        
    def test_quantile_method(self):
        """Test quantile-based outlier removal."""
        data = np.arange(100)  # 0 to 99
        
        indices, cleaned = remove_outliers(
            data, method='quantile', quantile_range=(0.1, 0.9)
        )
        
        # Should remove bottom 10% and top 10%
        assert len(cleaned) == 80
        assert min(cleaned) >= 10
        assert max(cleaned) <= 89
        
    def test_isolation_method_fallback(self):
        """Test isolation forest method (or fallback to MAD if sklearn not available)."""
        data = np.concatenate([np.random.normal(0, 1, 95), [10, -10, 15, -15, 20]])
        
        # This will either use IsolationForest or fall back to MAD
        indices, cleaned = remove_outliers(data, method='isolation', threshold=0.05)
        
        # Should remove some outliers
        assert len(cleaned) < 100
        assert len(cleaned) > 90  # But not too many
        
    def test_invalid_method(self):
        """Test invalid method raises error."""
        data = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            remove_outliers(data, method='invalid_method')
            
    def test_empty_data(self):
        """Test with empty data."""
        data = np.array([])
        
        indices, cleaned = remove_outliers(data, method='zscore')
        
        assert len(cleaned) == 0
        assert len(indices) == 0
        
    def test_single_value(self):
        """Test with single value."""
        data = np.array([42])
        
        indices, cleaned = remove_outliers(data, method='zscore')
        
        assert len(cleaned) == 1
        assert cleaned[0] == 42
        
    def test_return_indices(self):
        """Test that returned indices are correct."""
        # Use a longer series for stable statistics
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        
        # Insert extreme outliers at specific positions
        data = normal_data.copy()
        data[10] = 100  # Extreme positive outlier
        data[50] = -100  # Extreme negative outlier
        
        indices, cleaned = remove_outliers(data, method='zscore', threshold=2.0)
        
        # Check indices correspond to inliers
        assert all(data[idx] in cleaned for idx in indices)
        assert len(indices) == len(cleaned)
        
        # Outliers at positions 10 and 50 should be removed
        assert 10 not in indices
        assert 50 not in indices
        assert 100 not in cleaned
        assert -100 not in cleaned
        assert len(cleaned) == 98  # Should remove 2 outliers
        
    def test_2d_data_flattening(self):
        """Test that 2D data is properly flattened."""
        # Create longer series for stable statistics
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (10, 10))
        
        # Add extreme outlier
        data = normal_data.copy()
        data[5, 5] = 100  # Extreme outlier
        
        indices, cleaned = remove_outliers(data, method='zscore', threshold=2.0)
        
        # Should flatten and process
        assert cleaned.ndim == 1
        assert 100 not in cleaned
        assert len(cleaned) == 99  # Should remove 1 outlier from 100 total values
        

class TestCorrelatedGaussianData:
    """Test cases for create_correlated_gaussian_data function."""
    
    def test_default_correlations(self):
        """Test with default correlation structure."""
        n_features = 10
        n_samples = 1000
        
        data, cov_matrix = create_correlated_gaussian_data(
            n_features=n_features, n_samples=n_samples, seed=42
        )
        
        assert data.shape == (n_features, n_samples)
        assert cov_matrix.shape == (n_features, n_features)
        
        # Check specified correlations (approximately)
        corr_matrix = correlation_matrix(data)
        assert abs(corr_matrix[1, 9] - 0.9) < 0.1
        assert abs(corr_matrix[2, 8] - 0.8) < 0.1
        assert abs(corr_matrix[3, 7] - 0.7) < 0.1
        
    def test_custom_correlations(self):
        """Test with custom correlation pairs."""
        correlation_pairs = [(0, 1, 0.5), (2, 3, -0.7)]
        
        data, cov_matrix = create_correlated_gaussian_data(
            n_features=5, n_samples=1000, 
            correlation_pairs=correlation_pairs, seed=42
        )
        
        corr_matrix = correlation_matrix(data)
        assert abs(corr_matrix[0, 1] - 0.5) < 0.1
        assert abs(corr_matrix[2, 3] - (-0.7)) < 0.1
        

class TestRescale:
    """Test cases for rescale function."""
    
    def test_rescale_basic(self):
        """Test basic rescaling to [0, 1]."""
        data = np.array([0, 5, 10])
        rescaled = rescale(data)
        
        assert rescaled.min() == 0
        assert rescaled.max() == 1
        assert rescaled[1] == 0.5
        
    def test_rescale_negative_values(self):
        """Test rescaling with negative values."""
        data = np.array([-10, 0, 10])
        rescaled = rescale(data)
        
        assert rescaled.min() == 0
        assert rescaled.max() == 1
        assert rescaled[1] == 0.5
        
    def test_rescale_constant_values(self):
        """Test rescaling when all values are the same."""
        data = np.array([5, 5, 5, 5])
        rescaled = rescale(data)
        
        # Should handle gracefully - typically returns all zeros or original values
        assert len(rescaled) == 4
        assert np.all(rescaled == rescaled[0])  # All values should be the same