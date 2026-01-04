"""Tests for entropy calculation functions."""

import numpy as np
import pytest
from driada.information.entropy import (
    entropy_d,
    probs_to_entropy,
    joint_entropy_dd,
    conditional_entropy_cdd,
    conditional_entropy_cd,
    joint_entropy_cdd,
    joint_entropy_cd,
)


def test_entropy_d():
    """Test discrete entropy calculation."""
    # Test 1: Uniform distribution (maximum entropy)
    x_uniform = np.array([0, 1, 2, 3] * 25)  # 100 samples, 4 states
    H_uniform = entropy_d(x_uniform)
    # For uniform distribution with 4 states: H = log2(4) = 2 bits
    assert abs(H_uniform - 2.0) < 0.01

    # Test 2: Deterministic case (zero entropy)
    x_det = np.ones(100)
    H_det = entropy_d(x_det)
    assert abs(H_det) < 1e-9

    # Test 3: Binary case with known probabilities
    # p = 0.8, q = 0.2: H = -0.8*log2(0.8) - 0.2*log2(0.2) ≈ 0.722
    x_binary = np.array([0] * 80 + [1] * 20)
    H_binary = entropy_d(x_binary)
    expected_H = -0.8 * np.log2(0.8) - 0.2 * np.log2(0.2)
    assert abs(H_binary - expected_H) < 0.01

    # Test 4: Edge case - single sample
    x_single = np.array([5])
    H_single = entropy_d(x_single)
    assert abs(H_single) < 1e-9

    # Test 5: Edge case - empty array
    x_empty = np.array([])
    H_empty = entropy_d(x_empty)
    assert H_empty == 0


def test_probs_to_entropy():
    """Test entropy calculation from probability distribution."""
    # Test 1: Uniform distribution
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    H = probs_to_entropy(p_uniform)
    assert abs(H - 2.0) < 1e-9

    # Test 2: Deterministic distribution
    p_det = np.array([1.0, 0.0, 0.0])
    H = probs_to_entropy(p_det)
    assert abs(H) < 1e-9

    # Test 3: Binary entropy
    p_binary = np.array([0.5, 0.5])
    H = probs_to_entropy(p_binary)
    assert abs(H - 1.0) < 1e-9

    # Test 4: Known case
    p = np.array([0.1, 0.2, 0.3, 0.4])
    H = probs_to_entropy(p)
    expected = -np.sum(p * np.log2(p))
    assert abs(H - expected) < 1e-9

    # Test 5: Handles zeros correctly
    p_with_zeros = np.array([0.5, 0.5, 0.0])
    H = probs_to_entropy(p_with_zeros)
    assert np.isfinite(H)


def test_joint_entropy_dd():
    """Test joint entropy for two discrete variables."""
    # Test 1: Independent variables
    # H(X,Y) = H(X) + H(Y) for independent variables
    x = np.array([0, 0, 1, 1] * 25)
    y = np.array([0, 1, 0, 1] * 25)
    H_joint = joint_entropy_dd(x, y)
    # Both are uniform binary, so H(X) = H(Y) = 1, H(X,Y) = 2
    assert abs(H_joint - 2.0) < 0.1

    # Test 2: Perfectly correlated variables
    # H(X,Y) = H(X) = H(Y) when X = Y
    x = np.array([0, 1, 2, 3] * 25)
    y = x.copy()
    H_joint = joint_entropy_dd(x, y)
    H_x = entropy_d(x)
    assert abs(H_joint - H_x) < 0.1

    # Test 3: Deterministic case
    x = np.ones(100)
    y = np.ones(100)
    H_joint = joint_entropy_dd(x, y)
    assert abs(H_joint) < 0.1

    # Test 4: Different cardinalities
    x = np.array([0, 1] * 50)  # Binary
    y = np.array([0, 1, 2, 3] * 25)  # 4 states
    H_joint = joint_entropy_dd(x, y)
    assert H_joint > 0 and H_joint <= 3  # Max is log2(2*4) = 3


def test_conditional_entropy_cdd():
    """Test conditional entropy H(Z|X,Y) for continuous Z, discrete X,Y."""
    np.random.seed(42)

    # Test 1: Z independent of X,Y
    # H(Z|X,Y) = H(Z) when independent
    n = 1000
    z = np.random.randn(n)
    x = np.random.randint(0, 2, n)
    y = np.random.randint(0, 2, n)

    H_cond = conditional_entropy_cdd(z, x, y, k=5)
    # For standard normal, differential entropy ≈ 0.5 * log2(2πe) ≈ 2.05 bits
    assert H_cond > 1.5 and H_cond < 2.5

    # Test 2: Z fully determined by X,Y with noise to avoid singularity
    z = x.astype(float) + 2 * y + 0.01 * np.random.randn(n)
    H_cond = conditional_entropy_cdd(z, x, y, k=5)
    # Differential entropy can be negative for small variance
    assert H_cond < -2

    # Test 3: Edge case - insufficient samples
    z_small = np.array([1.0, 2.0, 3.0])
    x_small = np.array([0, 1, 0])
    y_small = np.array([0, 0, 1])
    H_cond = conditional_entropy_cdd(z_small, x_small, y_small, k=5)
    assert np.isfinite(H_cond)
    assert H_cond == 0


def test_conditional_entropy_cd():
    """Test conditional entropy H(Z|X) for continuous Z, discrete X."""
    np.random.seed(42)

    # Test 1: Z independent of X
    n = 1000
    z = np.random.randn(n)
    x = np.random.randint(0, 2, n)

    H_cond = conditional_entropy_cd(z, x, k=5)
    # Should be close to H(Z) ≈ 2.05 for standard normal
    assert H_cond > 1.5 and H_cond < 2.5

    # Test 2: Z determined by X with small noise
    z = x.astype(float) * 5.0 + 1.0 + 0.01 * np.random.randn(n)
    H_cond = conditional_entropy_cd(z, x, k=5)
    # Differential entropy can be negative for small variance
    assert H_cond < -2

    # Test 3: Z partially dependent on X
    z = x.astype(float) + 0.1 * np.random.randn(n)
    H_cond = conditional_entropy_cd(z, x, k=5)
    # Differential entropy can be negative
    assert H_cond < 0

    # Test 4: Edge case - small sample size
    z_small = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x_small = np.array([0, 0, 0, 1, 1, 1])
    H_cond = conditional_entropy_cd(z_small, x_small, k=2)
    assert np.isfinite(H_cond)


def test_joint_entropy_cdd():
    """Test joint entropy H(X,Y,Z) for two discrete and one continuous variable."""
    np.random.seed(42)
    n = 1000

    # Test 1: All independent
    x = np.random.randint(0, 2, n)
    y = np.random.randint(0, 2, n)
    z = np.random.randn(n)

    H_xyz = joint_entropy_cdd(x, y, z, k=5)
    H_xy = joint_entropy_dd(x, y)
    # H(X,Y,Z) = H(X,Y) + H(Z|X,Y) ≈ H(X,Y) + H(Z) when independent
    # H(X,Y) ≈ 2, H(Z) ≈ 2.05
    assert H_xyz > 3.5 and H_xyz < 4.5

    # Test 2: Z depends on X,Y with small noise
    z = x.astype(float) + 2 * y + 0.01 * np.random.randn(n)
    H_xyz = joint_entropy_cdd(x, y, z, k=5)
    # Should be close to H(X,Y) + small negative value
    assert H_xyz < H_xy + 0.5

    # Test 3: Chain rule verification
    H_z_given_xy = conditional_entropy_cdd(z, x, y, k=5)
    assert abs(H_xyz - (H_xy + H_z_given_xy)) < 0.01


def test_joint_entropy_cd():
    """Test joint entropy H(X,Z) for one discrete and one continuous variable."""
    np.random.seed(42)
    n = 1000

    # Test 1: Independent variables
    x = np.random.randint(0, 3, n)
    z = np.random.randn(n)

    H_xz = joint_entropy_cd(x, z, k=5)
    H_x = entropy_d(x)
    # H(X,Z) = H(X) + H(Z|X) ≈ H(X) + H(Z) when independent
    # H(X) ≈ log2(3) ≈ 1.58, H(Z) ≈ 2.05
    assert H_xz > 3.0 and H_xz < 4.0

    # Test 2: Z depends on X with small noise
    z = x.astype(float) * 2.0 + 0.01 * np.random.randn(n)
    H_xz = joint_entropy_cd(x, z, k=5)
    # Should be close to H(X) + small negative value
    assert H_xz < H_x + 0.5

    # Test 3: Chain rule verification
    H_z_given_x = conditional_entropy_cd(z, x, k=5)
    assert abs(H_xz - (H_x + H_z_given_x)) < 0.01

    # Test 4: Edge case - binary X
    x = np.array([0, 1] * 500)
    z = np.random.randn(n)
    H_xz = joint_entropy_cd(x, z, k=5)
    assert H_xz > 2.5  # H(X) = 1, H(Z) ≈ 2.05


def test_edge_cases_and_errors():
    """Test edge cases and error handling."""
    # Test empty arrays
    empty = np.array([])

    # entropy_d with empty array returns 0
    H_empty = entropy_d(empty)
    assert H_empty == 0.0

    # probs_to_entropy with empty array should return 0
    H_prob_empty = probs_to_entropy(empty)
    assert H_prob_empty == 0.0

    # Test mismatched lengths
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    z = np.array([1.0, 2.0, 3.0])

    with pytest.raises((ValueError, IndexError)):
        joint_entropy_dd(x, y)

    # Test single element arrays
    x_single = np.array([1])
    z_single = np.array([1.5])

    H_x = entropy_d(x_single)
    assert abs(H_x) < 1e-9

    # Test all same values
    x_same = np.ones(100, dtype=int)
    y_same = np.zeros(100, dtype=int)
    z_same = np.ones(100) * 3.14

    H_x = entropy_d(x_same)
    assert abs(H_x) < 1e-9

    H_xy = joint_entropy_dd(x_same, y_same)
    assert abs(H_xy) < 1e-9

    # Test with negative values (should work fine)
    x_neg = np.array([-1, -2, -1, -2] * 25)
    H_neg = entropy_d(x_neg)
    assert H_neg > 0  # Should be same as [0,1,0,1]


def test_consistency_relationships():
    """Test mathematical relationships between entropy functions."""
    np.random.seed(42)
    n = 1000

    # Create correlated discrete variables
    x = np.random.randint(0, 3, n)
    y = (x + np.random.randint(0, 2, n)) % 3
    z = np.random.randn(n) + x.astype(float) * 0.5

    # Test 1: H(X,Y) >= H(X) and H(X,Y) >= H(Y)
    H_x = entropy_d(x)
    H_y = entropy_d(y)
    H_xy = joint_entropy_dd(x, y)
    assert H_xy >= H_x - 0.01  # Small tolerance for numerical errors
    assert H_xy >= H_y - 0.01

    # Test 2: H(X,Y) <= H(X) + H(Y) (subadditivity)
    assert H_xy <= H_x + H_y + 0.01

    # Test 3: H(X,Z) >= H(X) (adding a variable doesn't decrease entropy)
    H_xz = joint_entropy_cd(x, z, k=5)
    assert H_xz >= H_x - 0.01

    # Test 4: Chain rule H(X,Y,Z) = H(X,Y) + H(Z|X,Y)
    H_xyz = joint_entropy_cdd(x, y, z, k=5)
    H_z_given_xy = conditional_entropy_cdd(z, x, y, k=5)
    assert abs(H_xyz - (H_xy + H_z_given_xy)) < 0.01

    # Test 5: For independent variables
    x_ind = np.random.randint(0, 2, n)
    y_ind = np.random.randint(0, 2, n)
    z_ind = np.random.randn(n)

    H_x_ind = entropy_d(x_ind)
    H_y_ind = entropy_d(y_ind)
    H_xy_ind = joint_entropy_dd(x_ind, y_ind)
    # Should be approximately additive for independent variables
    assert abs(H_xy_ind - (H_x_ind + H_y_ind)) < 0.1


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    np.random.seed(42)

    # Test with very small probabilities
    p_extreme = np.array([0.999, 0.0005, 0.0005])
    H = probs_to_entropy(p_extreme)
    assert np.isfinite(H)
    assert H > 0

    # Test with very large discrete values
    x_large = np.array([1000000, 1000001, 1000000, 1000001] * 25)
    H_large = entropy_d(x_large)
    assert abs(H_large - 1.0) < 0.01  # Should be same as binary

    # Test with very small continuous values
    z_small = np.random.randn(1000) * 1e-10
    x = np.random.randint(0, 2, 1000)
    H_cond = conditional_entropy_cd(z_small, x, k=5)
    assert np.isfinite(H_cond)

    # Test with very large continuous values
    z_large = np.random.randn(1000) * 1e10
    H_cond_large = conditional_entropy_cd(z_large, x, k=5)
    assert np.isfinite(H_cond_large)
