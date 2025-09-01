"""Tests for KSG entropy and mutual information estimators."""

import numpy as np
import pytest
from driada.information.ksg import (
    nonparam_entropy_c,
    nonparam_cond_entropy_cc,
    nonparam_mi_cc,
    nonparam_mi_cd,
    nonparam_mi_dc,
    lnc_correction,
    build_tree,
)
from driada.information.info_utils import py_fast_digamma
from driada.information import mi_model_gd


class TestKSGEntropy:
    """Test KSG continuous entropy estimator."""

    def test_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        # Uniform distribution on [0, 1] has entropy log(1) = 0
        # But on [0, a] has entropy log(a)
        np.random.seed(42)
        x = np.random.uniform(0, 2, 1000)
        h = nonparam_entropy_c(x, k=5)
        # Theoretical entropy is log(2) ≈ 0.693
        assert 0.5 < h < 0.9

    def test_entropy_gaussian(self):
        """Test entropy of Gaussian distribution."""
        np.random.seed(42)
        # Standard normal has entropy 0.5 * log(2*pi*e) ≈ 1.419
        x = np.random.randn(1000)
        h = nonparam_entropy_c(x, k=5)
        assert 1.2 < h < 1.6

        # Scaled Gaussian: entropy increases with log(sigma)
        x_scaled = 2 * np.random.randn(1000)
        h_scaled = nonparam_entropy_c(x_scaled, k=5)
        assert h_scaled > h

    def test_entropy_multivariate(self):
        """Test entropy of multivariate data."""
        np.random.seed(42)
        # Independent Gaussians: H(X,Y) = H(X) + H(Y)
        x = np.random.randn(1000, 2)
        h_2d = nonparam_entropy_c(x, k=5)
        # Should be approximately 2 * 1.419
        assert 2.5 < h_2d < 3.0

        # 3D case
        x_3d = np.random.randn(1000, 3)
        h_3d = nonparam_entropy_c(x_3d, k=5)
        assert h_3d > h_2d

    def test_entropy_deterministic(self):
        """Test entropy of deterministic/low-entropy data."""
        np.random.seed(42)
        # Constant data (with small noise to avoid degeneracy)
        x = np.ones(100) + 1e-10 * np.random.randn(100)
        h = nonparam_entropy_c(x, k=3)
        # Should be very low
        assert h < 0  # Low entropy data should have negative differential entropy

    def test_entropy_base_e_vs_base_2(self):
        """Test entropy calculation with different bases."""
        np.random.seed(42)
        x = np.random.randn(500)

        h_e = nonparam_entropy_c(x, k=5, base=np.e)
        h_2 = nonparam_entropy_c(x, k=5, base=2)

        # h_2 = h_e / log(2)
        assert np.abs(h_2 - h_e / np.log(2)) < 0.1

    def test_entropy_1d_reshape(self):
        """Test that 1D arrays are properly reshaped."""
        np.random.seed(42)
        x_1d = np.random.randn(500)
        x_2d = x_1d.reshape(-1, 1)

        h_1d = nonparam_entropy_c(x_1d, k=5)
        h_2d = nonparam_entropy_c(x_2d, k=5)

        assert np.abs(h_1d - h_2d) < 0.1

    def test_entropy_different_k(self):
        """Test entropy estimation with different k values."""
        np.random.seed(42)
        x = np.random.randn(500)

        h_k3 = nonparam_entropy_c(x, k=3)
        h_k5 = nonparam_entropy_c(x, k=5)
        h_k10 = nonparam_entropy_c(x, k=10)

        # All should be similar for large enough sample
        assert np.abs(h_k3 - h_k5) < 0.2
        assert np.abs(h_k5 - h_k10) < 0.2


class TestKSGConditionalEntropy:
    """Test KSG conditional entropy estimator."""

    def test_cond_entropy_independent(self):
        """Test H(X|Y) = H(X) when X and Y are independent."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)

        h_x = nonparam_entropy_c(x, k=5)
        h_x_given_y = nonparam_cond_entropy_cc(x, y, k=5)

        # Should be approximately equal
        assert np.abs(h_x - h_x_given_y) < 0.2

    def test_cond_entropy_deterministic(self):
        """Test H(X|Y) = 0 when X is deterministic function of Y."""
        np.random.seed(42)
        y = np.random.randn(500)
        x = 2 * y + 3  # Deterministic function
        # Add tiny noise to avoid degeneracy
        x += 1e-10 * np.random.randn(500)

        h_x_given_y = nonparam_cond_entropy_cc(x, y, k=5)

        # Should be very low (near zero)
        assert h_x_given_y < 0

    def test_cond_entropy_partial_dependence(self):
        """Test H(X|Y) < H(X) when X partially depends on Y."""
        np.random.seed(42)
        y = np.random.randn(500)
        noise = np.random.randn(500)
        x = y + noise  # X = Y + noise

        h_x = nonparam_entropy_c(x, k=5)
        h_x_given_y = nonparam_cond_entropy_cc(x, y, k=5)

        # H(X|Y) should be less than H(X)
        assert h_x_given_y < h_x
        # But not zero (due to noise)
        assert h_x_given_y > 0

    def test_cond_entropy_multivariate(self):
        """Test conditional entropy with multivariate Y."""
        np.random.seed(42)
        y = np.random.randn(500, 2)
        x = y[:, 0] + 0.5 * y[:, 1] + 0.5 * np.random.randn(500)

        h_x_given_y = nonparam_cond_entropy_cc(x, y, k=5)

        # Should be positive but small
        assert 0 < h_x_given_y < 1


class TestKSGMutualInformation:
    """Test KSG mutual information estimator."""

    def test_mi_independent(self):
        """Test MI = 0 for independent variables."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)

        mi = nonparam_mi_cc(x, y, k=5)

        # Should be close to zero
        assert -0.1 < mi < 0.1

    def test_mi_perfect_dependence(self):
        """Test MI is high when Y = f(X) deterministically."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 500)
        y = x**2  # Deterministic function

        mi = nonparam_mi_cc(x, y, k=5)

        # For perfect dependence, MI should be very high
        assert mi > 2.0  # Should be high positive value

        # Compare with independent case
        y_indep = np.random.uniform(0, 1, 500)
        mi_indep = nonparam_mi_cc(x, y_indep, k=5)

        # MI for dependent should be much higher than independent
        assert mi > mi_indep + 2.0

    def test_mi_linear_correlation(self):
        """Test MI for linearly correlated variables."""
        np.random.seed(42)
        # Create correlated Gaussians
        mean = [0, 0]
        cov = [[1, 0.8], [0.8, 1]]
        data = np.random.multivariate_normal(mean, cov, 500)
        x, y = data[:, 0], data[:, 1]

        mi = nonparam_mi_cc(x, y, k=5)

        # For Gaussians: MI = -0.5 * log(1 - rho^2)
        theoretical_mi = -0.5 * np.log(1 - 0.8**2)
        assert np.abs(mi - theoretical_mi) < 0.2

    def test_mi_multivariate(self):
        """Test MI with multivariate inputs."""
        np.random.seed(42)
        # X is 2D, Y is 2D, with some dependence
        x = np.random.randn(500, 2)
        y = np.zeros((500, 2))
        y[:, 0] = x[:, 0] + 0.5 * x[:, 1] + 0.5 * np.random.randn(500)
        y[:, 1] = x[:, 1] + 0.5 * np.random.randn(500)

        mi = nonparam_mi_cc(x, y, k=5)

        # Should be positive and substantial
        assert mi > 0.5

    def test_conditional_mi(self):
        """Test conditional mutual information I(X;Y|Z)."""
        np.random.seed(42)
        # Create chain: X -> Z -> Y
        x = np.random.randn(500)
        z = x + 0.5 * np.random.randn(500)
        y = z + 0.5 * np.random.randn(500)

        # I(X;Y|Z) should be close to 0 (X and Y are conditionally independent given Z)
        cmi = nonparam_mi_cc(x, y, z=z, k=5)
        assert -0.15 < cmi < 0.15

        # I(X;Y) without conditioning should be positive
        mi = nonparam_mi_cc(x, y, k=5)
        assert mi > 0.1

    def test_mi_with_precomputed_trees(self):
        """Test MI calculation with pre-computed trees."""
        np.random.seed(42)
        x = np.random.randn(500, 2)
        y = np.random.randn(500, 2)

        # Pre-compute trees with same leaf size as default
        tree_x = build_tree(x, lf=5)
        tree_y = build_tree(y, lf=5)

        mi_with_trees = nonparam_mi_cc(
            x, y, k=5, lf=5, precomputed_tree_x=tree_x, precomputed_tree_y=tree_y
        )
        mi_without_trees = nonparam_mi_cc(x, y, k=5, lf=5)

        # Results should be very close (small numerical differences due to tree building)
        assert np.abs(mi_with_trees - mi_without_trees) < 0.02

    def test_mi_base_conversion(self):
        """Test MI calculation with different bases."""
        np.random.seed(42)
        x = np.random.randn(300)
        y = x + np.random.randn(300)

        mi_e = nonparam_mi_cc(x, y, k=5, base=np.e)
        mi_2 = nonparam_mi_cc(x, y, k=5, base=2)

        # Conversion factor
        assert np.abs(mi_2 - mi_e / np.log(2)) < 0.05

    def test_mi_assertion_errors(self):
        """Test assertion errors for invalid inputs."""
        x = np.random.randn(100)
        y = np.random.randn(50)  # Different length

        with pytest.raises(ValueError, match="Arrays should have same length"):
            nonparam_mi_cc(x, y)

        # k too large
        x = np.random.randn(10)
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="k must be less than n_samples"):
            nonparam_mi_cc(x, y, k=10)


class TestLNCCorrection:
    """Test Local Non-uniformity Correction."""

    def test_lnc_uniform_data(self):
        """Test LNC correction on uniform data."""
        np.random.seed(42)
        # Uniform data should have minimal correction
        points = np.random.uniform(0, 1, (100, 2))
        tree = build_tree(points)
        k = 5
        alpha = 0.9

        correction = lnc_correction(tree, points, k, alpha)

        # Should be small for uniform data
        assert 0 <= correction < 0.5

    def test_lnc_clustered_data(self):
        """Test LNC correction on clustered data."""
        np.random.seed(42)
        # Create two clusters
        cluster1 = np.random.randn(50, 2)
        cluster2 = np.random.randn(50, 2) + 5
        points = np.vstack([cluster1, cluster2])

        tree = build_tree(points)
        k = 5
        alpha = 0.9

        correction = lnc_correction(tree, points, k, alpha)

        # Should have some correction due to non-uniformity
        assert correction >= 0

    def test_lnc_different_alpha(self):
        """Test LNC correction with different alpha values."""
        np.random.seed(42)
        points = np.random.randn(100, 3)
        tree = build_tree(points)
        k = 5

        correction_low = lnc_correction(tree, points, k, alpha=0.5)
        correction_high = lnc_correction(tree, points, k, alpha=0.95)

        # Both should be non-negative
        assert correction_low >= 0
        assert correction_high >= 0

    def test_lnc_with_mi_estimation(self):
        """Test MI estimation with LNC correction."""
        np.random.seed(42)
        # Correlated data
        x = np.random.randn(200)
        y = x + 0.5 * np.random.randn(200)

        mi_no_lnc = nonparam_mi_cc(x, y, k=5, alpha=0)
        mi_with_lnc = nonparam_mi_cc(x, y, k=5, alpha=0.9)

        # Both should be positive
        assert mi_no_lnc > 0
        assert mi_with_lnc > 0
        # LNC might change the estimate slightly
        assert np.abs(mi_no_lnc - mi_with_lnc) < 0.5


class TestHelperFunctions:
    """Test helper functions in KSG module."""

    def test_build_tree(self):
        """Test tree building function."""
        np.random.seed(42)

        # Low dimensional data uses KDTree
        points_low = np.random.randn(100, 5)
        tree_low = build_tree(points_low)
        assert tree_low is not None

        # High dimensional data uses BallTree
        points_high = np.random.randn(100, 25)
        tree_high = build_tree(points_high)
        assert tree_high is not None

        # Tree should work for queries
        dists, indices = tree_low.query(points_low[:5], k=3)
        assert dists.shape == (5, 3)
        assert indices.shape == (5, 3)

    def test_py_fast_digamma(self):
        """Test digamma function."""
        # Test known values
        assert np.abs(py_fast_digamma(1) - (-0.5772156649)) < 0.001
        assert np.abs(py_fast_digamma(2) - 0.4227843351) < 0.001

        # Test multiple scalar inputs
        x_values = [1, 2, 3, 4, 5]
        results = [py_fast_digamma(x) for x in x_values]
        assert len(results) == len(x_values)
        # Results should be increasing
        for i in range(1, len(results)):
            assert results[i] > results[i - 1]


class TestKSGMixedType:
    """Test KSG mutual information estimators for mixed continuous/discrete variables."""
    
    def test_nonparam_mi_cd_basic(self):
        """Test MI between continuous and discrete variables."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create correlated continuous and discrete variables
        x_continuous = np.random.randn(n_samples)
        # Discretize based on x value
        y_discrete = (x_continuous > 0).astype(int)
        
        mi = nonparam_mi_cd(x_continuous, y_discrete)
        
        # MI should be positive for correlated variables
        assert mi > 0
        assert mi < 1  # Should not be perfect correlation
        
    def test_nonparam_mi_cd_independent(self):
        """Test MI for independent variables."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create independent variables
        x_continuous = np.random.randn(n_samples)
        y_discrete = np.random.randint(0, 3, n_samples)
        
        mi = nonparam_mi_cd(x_continuous, y_discrete)
        
        # MI should be close to 0 for independent variables
        assert mi < 0.1
        
    def test_nonparam_mi_dc_symmetry(self):
        """Test that MI(X,Y) = MI(Y,X) for discrete-continuous."""
        np.random.seed(42)
        n_samples = 500
        
        # Create correlated variables
        x_discrete = np.random.randint(0, 4, n_samples)
        y_continuous = x_discrete + 0.5 * np.random.randn(n_samples)
        
        mi_dc = nonparam_mi_dc(x_discrete, y_continuous)
        mi_cd = nonparam_mi_cd(y_continuous, x_discrete)
        
        # Should be equal (or very close due to numerical precision)
        assert abs(mi_dc - mi_cd) < 1e-9
        
    def test_comparison_with_gcmi(self):
        """Compare KSG estimator with GCMI for mixed types."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create correlated continuous and discrete variables
        y_discrete = np.random.randint(0, 3, n_samples)
        x_continuous = y_discrete + np.random.randn(n_samples) * 0.5
        
        # Reshape for GCMI (expects 2D array)
        x_continuous_2d = x_continuous.reshape(1, -1)
        
        # Calculate MI using both methods
        mi_ksg = nonparam_mi_cd(x_continuous, y_discrete)
        mi_gcmi = mi_model_gd(x_continuous_2d, y_discrete, 3)
        
        # They should give similar results (both positive)
        # Note: KSG and GCMI use very different methods, so results can differ substantially
        assert mi_ksg > 0
        assert mi_gcmi > 0
        # Both should detect the correlation
        assert mi_ksg > 0.3
        assert mi_gcmi > 0.3
        
    def test_multidimensional_continuous(self):
        """Test MI with multidimensional continuous variable."""
        np.random.seed(42)
        n_samples = 500
        
        # Create 2D continuous variable correlated with discrete
        y_discrete = np.random.randint(0, 2, n_samples)
        x_continuous = np.column_stack([
            y_discrete + 0.5 * np.random.randn(n_samples),
            y_discrete * 2 + 0.3 * np.random.randn(n_samples)
        ])
        
        mi = nonparam_mi_cd(x_continuous, y_discrete)
        
        # Should have high MI due to correlation
        assert mi > 0.3
        
    def test_edge_cases(self):
        """Test edge cases and input validation."""
        # Test with too few samples
        with pytest.raises(ValueError, match="k must be less than n_samples"):
            nonparam_mi_cd(np.array([1, 2, 3]), np.array([0, 1, 0]), k=5)
            
        # Test with mismatched lengths
        with pytest.raises(ValueError, match="Arrays should have same length"):
            nonparam_mi_cd(np.random.randn(10), np.random.randint(0, 2, 8))
            
        # Test with single class
        x_continuous = np.random.randn(100)
        y_discrete = np.zeros(100, dtype=int)  # All same class
        mi = nonparam_mi_cd(x_continuous, y_discrete)
        assert mi == 0  # No information if Y is constant
