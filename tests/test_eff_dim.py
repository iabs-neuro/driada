"""Tests for effective dimension estimation."""
import numpy as np
import pytest
from driada.dimensionality import eff_dim


class TestEffDim:
    """Test effective dimension estimation function."""
    
    def test_eff_dim_basic(self):
        """Test basic functionality with random data."""
        np.random.seed(42)
        # n_features x n_samples format
        data = np.random.randn(50, 1000)
        
        # Test without correction
        ed_uncorrected = eff_dim(data, enable_correction=False)
        assert isinstance(ed_uncorrected, float)
        assert 0 < ed_uncorrected <= 50
        
    def test_eff_dim_with_low_rank_data(self):
        """Test with low-rank data."""
        np.random.seed(42)
        # Create rank-5 data
        n_features, n_samples = 50, 1000
        rank = 5
        U = np.random.randn(n_features, rank)
        V = np.random.randn(rank, n_samples)
        data = U @ V
        
        ed = eff_dim(data, enable_correction=False)
        assert 3 < ed < 6  # Should be close to true rank of 5
        
    def test_eff_dim_with_identity_covariance(self):
        """Test with identity covariance (all dimensions equally important)."""
        np.random.seed(42)
        n_features = 50
        # Create data with identity covariance
        data = np.random.randn(n_features, 1000)
        
        ed = eff_dim(data, enable_correction=False)
        # Should be close to n_features for identity covariance
        assert 40 < ed < 50
        
    def test_eff_dim_with_correction_small_ratio(self):
        """Test correction with small n/t ratio."""
        np.random.seed(42)
        # Small n/t ratio should work with correction
        data = np.random.randn(10, 1000)  # n/t = 0.01
        
        ed_corrected = eff_dim(data, enable_correction=True)
        ed_uncorrected = eff_dim(data, enable_correction=False)
        
        assert isinstance(ed_corrected, float)
        assert isinstance(ed_uncorrected, float)
        # Corrected should be different from uncorrected
        assert abs(ed_corrected - ed_uncorrected) > 0.01
        
    def test_eff_dim_different_q_values(self):
        """Test with different Renyi entropy orders."""
        np.random.seed(42)
        data = np.random.randn(30, 500)
        
        # Test q=1 (Shannon entropy)
        ed_q1 = eff_dim(data, enable_correction=False, q=1)
        # Test q=2 (default, quadratic entropy)
        ed_q2 = eff_dim(data, enable_correction=False, q=2)
        # Test q=inf (min-entropy)
        ed_qinf = eff_dim(data, enable_correction=False, q=np.inf)
        
        # All should be valid but different
        assert all(isinstance(ed, float) for ed in [ed_q1, ed_q2, ed_qinf])
        assert ed_q1 != ed_q2 != ed_qinf
        
    def test_eff_dim_warning_large_ratio(self):
        """Test that warning is issued for large n/t ratio."""
        np.random.seed(42)
        # Large n/t ratio
        data = np.random.randn(100, 200)  # n/t = 0.5
        
        with pytest.warns(UserWarning, match="Spectrum correction is recommended"):
            ed = eff_dim(data, enable_correction=False)
        
        assert isinstance(ed, float)
        
    @pytest.mark.xfail(reason="Known issue: correction can fail with certain data")
    def test_eff_dim_correction_edge_cases(self):
        """Test correction with edge cases that might fail."""
        np.random.seed(42)
        
        # Near-singular data that might cause correction to fail
        n_features = 100
        n_samples = 150
        # Create highly correlated features
        base = np.random.randn(1, n_samples)
        noise = 0.01 * np.random.randn(n_features, n_samples)
        data = base + noise
        
        # This might fail with current implementation
        ed_corrected = eff_dim(data, enable_correction=True)
        assert isinstance(ed_corrected, float)
        
    def test_eff_dim_single_dimension(self):
        """Test with effectively one-dimensional data."""
        np.random.seed(42)
        # Create 1D data embedded in higher dimensions
        t = np.linspace(0, 10, 1000)
        data = np.vstack([np.sin(t), np.cos(t), 2*np.sin(t), 3*np.cos(t)])
        data += 0.01 * np.random.randn(*data.shape)  # Small noise
        
        ed = eff_dim(data, enable_correction=False)
        # Should be close to 1 or 2 (circular manifold)
        assert 0.5 < ed < 3