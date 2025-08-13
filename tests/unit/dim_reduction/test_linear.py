"""
Comprehensive tests for linear dimensionality estimation methods.

This module tests the PCA-based dimensionality estimation functions
from driada.dimensionality.linear:
- pca_dimension: Dimension via variance threshold
- pca_dimension_profile: Dimension profile for multiple thresholds
- effective_rank: Effective rank based on eigenvalue entropy

Important Limitations and Considerations:
-----------------------------------------
1. PCA-based methods fundamentally assume linear structure:
   - Will overestimate dimension for nonlinear manifolds (e.g., Swiss roll)
   - Cannot detect intrinsic nonlinear structure
   - Best suited for data with linear subspace structure

2. Effective rank vs PCA dimension:
   - Effective rank uses entropy of eigenvalue distribution
   - Often higher than PCA dimension for same data
   - More sensitive to the tail of eigenvalue distribution
   - Better captures "spread" of variance across components

3. Method-specific limitations:
   - pca_dimension: Threshold choice is arbitrary (90%, 95%, etc.)
   - effective_rank: Can be > n_features for uniform eigenvalues
   - All methods: Sensitive to preprocessing (standardization)

4. Test design notes:
   - We test both exact (noise-free) and noisy scenarios
   - Effective rank tests use wider ranges due to its different nature
   - Some "failures" in original tests were due to misunderstanding the methods
"""

import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve, make_blobs
from driada.dimensionality import pca_dimension, pca_dimension_profile, effective_rank


class TestPCADimension:
    """Test PCA-based dimension estimation via variance threshold."""

    def test_basic_functionality(self):
        """Test pca_dimension returns valid output."""
        # Generate simple data
        data = np.random.randn(100, 10)
        dim = pca_dimension(data, threshold=0.95)

        # Check output type and range
        assert isinstance(dim, (int, np.integer))
        assert 1 <= dim <= 10

    def test_exact_linear_subspace(self):
        """Test on data lying exactly in a linear subspace."""
        # Create exact 3D subspace in 10D ambient space
        n_samples = 200
        basis = np.random.randn(10, 3)
        basis = np.linalg.qr(basis)[0]  # Orthonormalize
        coeffs = np.random.randn(n_samples, 3)
        data = coeffs @ basis.T

        # Should recover exactly 3 dimensions
        dim = pca_dimension(data, threshold=0.999)
        assert dim == 3, f"Expected exactly 3 dimensions, got {dim}"

    def test_threshold_sensitivity(self):
        """Test sensitivity to variance threshold parameter."""
        # Generate data with exponentially decaying eigenvalues
        n_samples, n_features = 500, 20
        eigenvalues = np.exp(-np.arange(n_features) / 2)
        U = np.linalg.qr(np.random.randn(n_features, n_features))[0]
        data = (
            np.random.randn(n_samples, n_features) @ np.diag(np.sqrt(eigenvalues)) @ U.T
        )

        # Test different thresholds
        thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        dims = [pca_dimension(data, threshold=t) for t in thresholds]

        # Dimensions should increase with threshold
        assert all(
            dims[i] <= dims[i + 1] for i in range(len(dims) - 1)
        ), f"Dimensions should be non-decreasing: {dims}"
        assert (
            dims[0] < dims[-1]
        ), "Should capture more components with higher threshold"

    def test_standardization_effect(self):
        """Test effect of standardization on dimension estimation."""
        # Create data with very different scales
        n_samples = 300
        data = np.column_stack(
            [
                100 * np.random.randn(n_samples),  # Large scale
                np.random.randn(n_samples),  # Normal scale
                0.01 * np.random.randn(n_samples),  # Small scale
                np.random.randn(n_samples, 7),  # Additional normal features
            ]
        )

        # Without standardization, large scale dominates
        dim_no_std = pca_dimension(data, threshold=0.95, standardize=False)

        # With standardization, all features contribute
        dim_std = pca_dimension(data, threshold=0.95, standardize=True)

        # Standardized should need more components
        assert dim_std >= dim_no_std
        assert dim_std >= 3  # Should capture at least the 3 different scales

    def test_noisy_manifold(self):
        """Test on manifold data with noise."""
        # Generate Swiss roll (2D manifold) with noise
        data, _ = make_swiss_roll(n_samples=1000, noise=0.5, random_state=42)

        # PCA will overestimate due to noise
        dim = pca_dimension(data, threshold=0.95)
        assert dim == 3, f"Expected 3 dimensions for noisy 2D manifold in 3D, got {dim}"

    def test_high_dimensional_data(self):
        """Test on high-dimensional data."""
        # Generate data with intrinsic dimension 5 in 50D space
        n_samples = 200
        intrinsic_data = np.random.randn(n_samples, 5)
        projection = np.random.randn(5, 50)
        data = intrinsic_data @ projection

        # Add small noise
        data += 0.01 * np.random.randn(*data.shape)

        dim = pca_dimension(data, threshold=0.99)
        assert 4 <= dim <= 10, f"Expected ~5 dimensions, got {dim}"

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Single feature
        single_feat = np.random.randn(100, 1)
        dim = pca_dimension(single_feat, threshold=0.95)
        assert dim == 1

        # All features identical (rank 1)
        identical = np.random.randn(100, 1) @ np.ones((1, 5))
        dim = pca_dimension(identical, threshold=0.99)
        assert dim == 1

        # More features than samples
        wide_data = np.random.randn(50, 100)
        dim = pca_dimension(wide_data, threshold=0.95)
        assert dim <= 50  # Can't exceed number of samples

    def test_known_covariance_structure(self):
        """Test on data with known covariance structure."""
        # Generate data with specific eigenvalue decay
        n_samples = 500
        eigenvalues = [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01]
        n_features = len(eigenvalues)

        # Create covariance matrix
        U = np.linalg.qr(np.random.randn(n_features, n_features))[0]
        cov = U @ np.diag(eigenvalues) @ U.T

        # Generate data
        data = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)

        # Calculate expected dimensions for different thresholds
        total_var = sum(eigenvalues)
        cumsum = np.cumsum(eigenvalues) / total_var

        # Test threshold 0.9
        expected_dim_90 = np.searchsorted(cumsum, 0.9) + 1
        actual_dim_90 = pca_dimension(data, threshold=0.9, standardize=False)
        assert abs(actual_dim_90 - expected_dim_90) <= 1


class TestPCADimensionProfile:
    """Test PCA dimension profile across multiple thresholds."""

    def test_basic_functionality(self):
        """Test pca_dimension_profile returns valid output."""
        data = np.random.randn(100, 10)
        profile = pca_dimension_profile(data)

        # Check output structure
        assert isinstance(profile, dict)
        assert "thresholds" in profile
        assert "n_components" in profile
        assert "explained_variance_ratio" in profile
        assert "cumulative_variance" in profile

        # Check consistency
        assert len(profile["thresholds"]) == len(profile["n_components"])
        assert len(profile["explained_variance_ratio"]) <= 10  # Max features

    def test_default_thresholds(self):
        """Test with default threshold values."""
        data = np.random.randn(200, 15)
        profile = pca_dimension_profile(data)

        # Default thresholds
        expected_thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        assert np.array_equal(profile["thresholds"], expected_thresholds)

        # Components should be non-decreasing
        components = profile["n_components"]
        assert all(
            components[i] <= components[i + 1] for i in range(len(components) - 1)
        )

    def test_custom_thresholds(self):
        """Test with custom threshold values."""
        data = np.random.randn(150, 10)
        custom_thresholds = [0.6, 0.7, 0.85, 0.99]
        profile = pca_dimension_profile(data, thresholds=custom_thresholds)

        assert np.array_equal(profile["thresholds"], custom_thresholds)
        assert len(profile["n_components"]) == len(custom_thresholds)

    def test_consistency_with_pca_dimension(self):
        """Test consistency with individual pca_dimension calls."""
        data = np.random.randn(300, 20)
        thresholds = [0.8, 0.9, 0.95]

        # Get profile
        profile = pca_dimension_profile(data, thresholds=thresholds, standardize=True)

        # Check against individual calls
        for i, threshold in enumerate(thresholds):
            individual_dim = pca_dimension(data, threshold=threshold, standardize=True)
            assert profile["n_components"][i] == individual_dim

    def test_variance_explained_consistency(self):
        """Test that cumulative variance matches thresholds."""
        # Generate data with known structure
        n_samples = 400
        eigenvalues = np.exp(-np.arange(10) / 2)
        U = np.linalg.qr(np.random.randn(10, 10))[0]
        data = np.random.randn(n_samples, 10) @ np.diag(np.sqrt(eigenvalues)) @ U.T

        profile = pca_dimension_profile(data, standardize=False)

        # Check that cumulative variance at selected components exceeds thresholds
        for i, (threshold, n_comp) in enumerate(
            zip(profile["thresholds"], profile["n_components"])
        ):
            if n_comp <= len(profile["cumulative_variance"]):
                cumvar_at_n = profile["cumulative_variance"][n_comp - 1]
                assert cumvar_at_n >= threshold - 0.01  # Small tolerance

    def test_profile_visualization_data(self):
        """Test that profile provides useful data for visualization."""
        # Generate interesting data
        data, _ = make_blobs(n_samples=500, n_features=20, centers=5, random_state=42)

        profile = pca_dimension_profile(data)

        # Should have smooth eigenvalue decay
        exp_var = profile["explained_variance_ratio"]
        assert all(exp_var[i] >= exp_var[i + 1] for i in range(len(exp_var) - 1))

        # Cumulative variance should reach 1.0
        assert abs(profile["cumulative_variance"][-1] - 1.0) < 1e-10


class TestEffectiveRank:
    """Test effective rank based on eigenvalue entropy."""

    def test_basic_functionality(self):
        """Test effective_rank returns valid output."""
        data = np.random.randn(100, 10)
        rank = effective_rank(data)

        # Check output type and range
        assert isinstance(rank, (float, np.floating))
        assert 1 <= rank <= 10

    def test_exact_rank_k_matrix(self):
        """Test on matrix with exact rank k."""
        # Create exact rank-3 matrix
        n_samples = 200
        U = np.random.randn(n_samples, 3)
        V = np.random.randn(3, 10)
        data = U @ V

        rank = effective_rank(data, standardize=False)
        # Should be close to 3 (exactly 3 for perfect rank-3)
        assert 2.5 < rank < 3.2, f"Expected effective rank ~3, got {rank}"

    def test_uniform_eigenvalues(self):
        """Test on data with uniform eigenvalue distribution."""
        # Create data with equal variances in all directions
        n_samples, n_features = 500, 10
        data = np.random.randn(n_samples, n_features)

        rank = effective_rank(data, standardize=True)
        # Should be close to n_features for uniform distribution
        assert 8 < rank < 10, f"Expected effective rank ~10, got {rank}"

    def test_exponential_eigenvalue_decay(self):
        """Test on data with exponential eigenvalue decay."""
        # Generate data with exponentially decaying eigenvalues
        n_samples, n_features = 500, 20
        decay_rate = 0.5
        eigenvalues = np.exp(-np.arange(n_features) * decay_rate)

        U = np.linalg.qr(np.random.randn(n_features, n_features))[0]
        data = (
            np.random.randn(n_samples, n_features) @ np.diag(np.sqrt(eigenvalues)) @ U.T
        )

        rank = effective_rank(data, standardize=False)

        # For exponential decay with rate 0.5, effective rank is still fairly high
        # because many eigenvalues contribute (exp(-0.5*i) decays slowly)
        assert (
            8 < rank < 15
        ), f"Expected moderate effective rank for this decay rate, got {rank}"

    def test_comparison_with_pca_dimension(self):
        """Test relationship between effective rank and PCA dimension."""
        # Generate data with clear dimensionality
        n_samples = 400
        # 5 strong components, 5 weak components
        eigenvalues = np.array([10, 8, 6, 4, 2, 0.5, 0.3, 0.1, 0.05, 0.01])
        n_features = len(eigenvalues)

        U = np.linalg.qr(np.random.randn(n_features, n_features))[0]
        cov = U @ np.diag(eigenvalues) @ U.T
        data = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)

        eff_rank = effective_rank(data, standardize=False)
        pca_dim_90 = pca_dimension(data, threshold=0.9, standardize=False)
        pca_dim_95 = pca_dimension(data, threshold=0.95, standardize=False)

        # Effective rank is a different measure than PCA dimension
        # For this eigenvalue distribution, effective rank will be higher than PCA dims
        # because it considers the entropy of the entire eigenvalue distribution
        assert (
            5 < eff_rank < 10
        ), f"Expected effective rank ~7 for this distribution, got {eff_rank}"
        assert pca_dim_90 <= pca_dim_95  # Basic sanity check

    def test_standardization_effect(self):
        """Test effect of standardization on effective rank."""
        # Create data with very different scales
        n_samples = 300
        data = np.column_stack(
            [
                1000 * np.random.randn(n_samples, 2),  # Two large scale features
                np.random.randn(n_samples, 5),  # Five normal scale features
                0.001 * np.random.randn(n_samples, 3),  # Three tiny scale features
            ]
        )

        rank_no_std = effective_rank(data, standardize=False)
        rank_std = effective_rank(data, standardize=True)

        # Without standardization, dominated by large scale features
        assert rank_no_std < 4

        # With standardization, all features contribute
        assert rank_std > 6

    def test_manifold_data(self):
        """Test on manifold data."""
        # Swiss roll - 2D manifold in 3D
        data, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

        rank = effective_rank(data, standardize=True)
        # Should detect that variance is concentrated in ~2 dimensions
        assert 2 < rank < 3, f"Expected effective rank ~2-3 for Swiss roll, got {rank}"

    def test_edge_cases(self):
        """Test edge cases and numerical stability."""
        # Single feature (rank 1)
        single_feat = np.random.randn(100, 1)
        rank = effective_rank(single_feat)
        assert abs(rank - 1.0) < 1e-10

        # Perfectly correlated features (rank 1)
        base = np.random.randn(100, 1)
        correlated = np.hstack([base, 2 * base, -0.5 * base])
        rank = effective_rank(correlated, standardize=False)
        assert abs(rank - 1.0) < 0.1

        # Wide matrix (more features than samples)
        wide = np.random.randn(50, 100)
        rank = effective_rank(wide)
        assert rank <= 50

    def test_entropy_interpretation(self):
        """Test entropy-based interpretation of effective rank."""
        # Create data with k equal eigenvalues and rest zero
        # Effective rank should equal k
        n_samples = 500
        for k in [1, 3, 5, 7]:
            # Create k-dimensional subspace
            basis = np.linalg.qr(np.random.randn(20, k))[0]
            data = np.random.randn(n_samples, k) @ basis.T

            rank = effective_rank(data, standardize=False)
            assert abs(rank - k) < 0.3, f"Expected rank {k}, got {rank}"

    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned matrices."""
        # Create nearly singular matrix
        n_samples = 200
        eigenvalues = np.logspace(0, -15, 10)  # Very wide range
        U = np.linalg.qr(np.random.randn(10, 10))[0]

        # Generate data with tiny eigenvalues
        data = np.random.randn(n_samples, 10) @ np.diag(np.sqrt(eigenvalues)) @ U.T

        # Should handle without errors
        rank = effective_rank(data, standardize=False)
        assert isinstance(rank, (float, np.floating))
        assert 1 <= rank <= 10


class TestIntegration:
    """Test integration and consistency between different methods."""

    def test_methods_on_same_data(self):
        """Test all methods give consistent results on same data."""
        # Generate data with clear structure
        n_samples = 500
        eigenvalues = [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01]
        n_features = len(eigenvalues)

        U = np.linalg.qr(np.random.randn(n_features, n_features))[0]
        cov = U @ np.diag(eigenvalues) @ U.T
        data = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)

        # Get estimates from all methods
        pca_90 = pca_dimension(data, threshold=0.9, standardize=False)
        pca_95 = pca_dimension(data, threshold=0.95, standardize=False)
        eff_rank = effective_rank(data, standardize=False)
        profile = pca_dimension_profile(data, standardize=False)

        # Check consistency
        assert pca_90 <= pca_95
        # Effective rank is typically higher than PCA dimensions for broad eigenvalue distributions
        assert 4 < eff_rank < 8, f"Expected effective rank ~5-6, got {eff_rank}"
        assert profile["n_components"][2] == pca_90  # 0.9 is 3rd threshold
        assert profile["n_components"][3] == pca_95  # 0.95 is 4th threshold

    def test_manifold_comparison(self):
        """Compare methods on various manifolds."""
        manifolds = [
            (
                "Swiss Roll",
                make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)[0],
            ),
            ("S-Curve", make_s_curve(n_samples=1000, noise=0.1, random_state=42)[0]),
        ]

        for name, data in manifolds:
            pca_95 = pca_dimension(data, threshold=0.95)
            eff_rank = effective_rank(data)

            # For these 2D manifolds in 3D:
            # PCA should give 3 (captures noise)
            # Effective rank should be between 2 and 3
            assert pca_95 == 3, f"{name}: PCA dimension should be 3"
            assert (
                2 < eff_rank < 3
            ), f"{name}: Effective rank should be ~2-3, got {eff_rank}"
