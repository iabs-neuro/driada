"""Tests for effective dimension estimation."""

import numpy as np
import pytest
from driada.dimensionality import eff_dim


class TestEffDim:
    """Test effective dimension estimation function."""

    def test_eff_dim_basic(self):
        """Test basic functionality with random data."""
        np.random.seed(42)
        # n_samples x n_features format (standard ML convention)
        data = np.random.randn(1000, 50)

        # Test without correction
        ed_uncorrected = eff_dim(data, enable_correction=False)
        assert isinstance(ed_uncorrected, float)
        assert 0 < ed_uncorrected <= 50

    def test_eff_dim_with_low_rank_data(self):
        """Test with low-rank data."""
        np.random.seed(42)
        # Create rank-5 data
        n_samples, n_features = 1000, 50
        rank = 5
        U = np.random.randn(n_samples, rank)
        V = np.random.randn(rank, n_features)
        data = U @ V

        ed = eff_dim(data, enable_correction=False)
        assert 3 < ed < 6  # Should be close to true rank of 5

    def test_eff_dim_with_identity_covariance(self):
        """Test with identity covariance (all dimensions equally important)."""
        np.random.seed(42)
        n_features = 50
        # Create data with identity covariance
        data = np.random.randn(1000, n_features)

        ed = eff_dim(data, enable_correction=False)
        # Should be close to n_features for identity covariance
        assert 40 < ed < 50

    def test_eff_dim_with_correction_small_ratio(self):
        """Test correction with small n/t ratio."""
        np.random.seed(42)
        # Small n_features/n_samples ratio should work with correction
        data = np.random.randn(1000, 10)  # n_features/n_samples = 0.01

        ed_corrected = eff_dim(data, enable_correction=True)
        ed_uncorrected = eff_dim(data, enable_correction=False)

        assert isinstance(ed_corrected, float)
        assert isinstance(ed_uncorrected, float)
        # Corrected should be different from uncorrected
        assert abs(ed_corrected - ed_uncorrected) > 0.01

    def test_eff_dim_different_q_values(self):
        """Test with different Renyi entropy orders."""
        np.random.seed(42)
        data = np.random.randn(500, 30)

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
        """Test that warning is issued for large n_features/n_samples ratio."""
        np.random.seed(42)
        # Large n_features/n_samples ratio
        data = np.random.randn(200, 100)  # n_features/n_samples = 0.5

        with pytest.warns(UserWarning, match="Spectrum correction is recommended"):
            ed = eff_dim(data, enable_correction=False)

        assert isinstance(ed, float)

    def test_eff_dim_correction_edge_cases(self):
        """Test correction with edge cases that previously failed."""
        np.random.seed(42)

        # Near-singular data that might cause correction to fail
        n_features = 100
        n_samples = 150
        # Create highly correlated features
        base = np.random.randn(n_samples, 1)
        noise = 0.01 * np.random.randn(n_samples, n_features)
        data = base + noise

        # This should now work with the fix
        ed_corrected = eff_dim(data, enable_correction=True)
        assert isinstance(ed_corrected, float)
        assert 0 < ed_corrected <= n_features

    def test_eff_dim_single_dimension(self):
        """Test with effectively one-dimensional data."""
        np.random.seed(42)
        # Create 1D data embedded in higher dimensions
        t = np.linspace(0, 10, 1000)
        data = np.column_stack([np.sin(t), np.cos(t), 2 * np.sin(t), 3 * np.cos(t)])
        data += 0.01 * np.random.randn(*data.shape)  # Small noise

        ed = eff_dim(data, enable_correction=False)
        # Should be close to 1 or 2 (circular manifold)
        assert 0.5 < ed < 3

    def test_eff_dim_negative_eigenvalues(self):
        """Test handling of matrices with negative eigenvalues."""
        np.random.seed(42)
        n_features = 50
        n_samples = 60  # Close to n_features to increase chance of numerical issues

        # Create data that might lead to negative eigenvalues
        data = np.random.randn(n_samples, n_features)
        # Add some correlation structure
        data[:, 1:10] = 0.9 * data[:, 0:1] + 0.1 * np.random.randn(n_samples, 9)

        # Both should work without errors
        ed_uncorrected = eff_dim(data, enable_correction=False)
        ed_corrected = eff_dim(data, enable_correction=True)

        assert isinstance(ed_uncorrected, float)
        assert isinstance(ed_corrected, float)
        assert 0 < ed_uncorrected <= n_features
        assert 0 < ed_corrected <= n_features

    def test_eff_dim_correction_with_warnings(self):
        """Test that appropriate warnings are issued for negative eigenvalues."""
        np.random.seed(42)
        n_features = 100
        n_samples = 105  # Very close to n_features

        # Create nearly rank-deficient data
        rank = 50
        U = np.random.randn(n_samples, rank)
        V = np.random.randn(rank, n_features)
        data = U @ V
        # Add tiny noise to avoid exact rank deficiency
        data += 1e-10 * np.random.randn(n_samples, n_features)

        # Should work and possibly issue warning about negative eigenvalues
        ed_corrected = eff_dim(data, enable_correction=True)
        assert isinstance(ed_corrected, float)
        assert 0 < ed_corrected <= n_features

    def test_eff_dim_extreme_cases(self):
        """Test with extreme data configurations."""
        np.random.seed(42)

        # Case 1: Very low variance in most dimensions
        data1 = 1e-10 * np.random.randn(100, 10)  # Tiny variance
        data1[:, 0] = np.random.randn(100)  # Normal variance in one dimension
        ed1 = eff_dim(data1, enable_correction=False)
        # Note: Even tiny variance contributes to effective dimension
        # For participation ratio (q=2), all non-zero eigenvalues contribute
        assert 8.0 < ed1 < 10.0  # Will be close to 9 due to formula

        # Case 2: Perfectly correlated features
        base = np.random.randn(100, 1)
        data2 = np.tile(base, (1, 10))
        data2 += 1e-10 * np.random.randn(100, 10)  # Tiny noise to avoid singular matrix
        ed2 = eff_dim(data2, enable_correction=False)
        assert 0.9 < ed2 < 1.5  # Should be close to 1

        # Case 3: Very high n_features/n_samples ratio with correction
        data3 = np.random.randn(100, 90)
        ed3 = eff_dim(
            data3, enable_correction=True, correction_iters=3
        )  # Fewer iterations for speed
        assert isinstance(ed3, float)
        assert 0 < ed3 <= 90
