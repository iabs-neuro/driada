"""Tests for information theory utility functions."""

import numpy as np
from scipy.special import digamma
from driada.information.info_utils import (
    py_fast_digamma,
    py_fast_digamma_arr,
    binary_mi_score,
)


class TestDigammaFunctions:
    """Test fast digamma implementations."""

    def test_py_fast_digamma_single_values(self):
        """Test fast digamma on single values."""
        # Test values where digamma is well-defined
        test_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0]

        for x in test_values:
            fast_result = py_fast_digamma(x)
            scipy_result = digamma(x)
            # Allow small numerical differences
            assert abs(fast_result - scipy_result) < 1e-6, f"Failed for x={x}"

    def test_py_fast_digamma_small_values(self):
        """Test fast digamma for small positive values."""
        # Small values exercise the while loop
        small_values = [0.1, 0.2, 0.3, 0.5, 0.8]

        for x in small_values:
            fast_result = py_fast_digamma(x)
            scipy_result = digamma(x)
            assert abs(fast_result - scipy_result) < 1e-6

    def test_py_fast_digamma_arr(self):
        """Test array version of fast digamma."""
        # Test array of various sizes
        x_arr = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0])

        fast_results = py_fast_digamma_arr(x_arr)
        scipy_results = digamma(x_arr)

        assert fast_results.shape == x_arr.shape
        assert np.allclose(fast_results, scipy_results, rtol=1e-6)

    def test_py_fast_digamma_arr_mixed(self):
        """Test array digamma with mixed small and large values."""
        x_arr = np.array([0.1, 0.5, 1.0, 3.0, 5.5, 10.0, 50.0, 100.0])

        fast_results = py_fast_digamma_arr(x_arr)
        scipy_results = digamma(x_arr)

        assert np.allclose(fast_results, scipy_results, rtol=1e-6)

    def test_py_fast_digamma_arr_edge_cases(self):
        """Test array digamma with edge cases."""
        # Empty array
        empty = np.array([])
        result = py_fast_digamma_arr(empty)
        assert result.shape == (0,)

        # Single element
        single = np.array([2.5])
        result = py_fast_digamma_arr(single)
        expected = digamma(2.5)
        assert abs(result[0] - expected) < 1e-6

    def test_py_fast_digamma_consistency(self):
        """Test consistency between single and array versions."""
        test_values = [0.5, 1.5, 2.5, 7.0, 15.0]

        for x in test_values:
            single_result = py_fast_digamma(x)
            array_result = py_fast_digamma_arr(np.array([x]))[0]
            assert abs(single_result - array_result) < 1e-10


class TestBinaryMIScore:
    """Test binary mutual information score calculation."""

    def test_binary_mi_score_independent(self):
        """Test MI score for independent variables."""
        # Create independent binary variables
        n = 100
        x = np.array([0, 1] * 50)
        y = np.array([0, 0, 1, 1] * 25)

        # Create contingency table
        contingency = np.zeros((2, 2), dtype=int)
        for i in range(n):
            contingency[x[i], y[i]] += 1

        mi = binary_mi_score(contingency)
        # MI should be close to 0 for independent variables
        assert mi >= 0  # MI is non-negative
        assert mi < 0.1  # Should be small for independent vars

    def test_binary_mi_score_perfect_dependence(self):
        """Test MI score for perfectly dependent variables."""
        # X and Y are identical
        n = 100
        x = np.array([0, 1] * 50)
        y = x.copy()

        contingency = np.zeros((2, 2), dtype=int)
        for i in range(n):
            contingency[x[i], y[i]] += 1

        mi = binary_mi_score(contingency)
        # MI should equal H(X) = H(Y) = log2(2) = 1 bit for uniform binary
        # But the function returns MI in nats, not bits
        expected_mi_nats = np.log(2)  # ~0.693
        assert abs(mi - expected_mi_nats) < 0.01

    def test_binary_mi_score_single_cluster(self):
        """Test MI score when one variable has single value."""
        # All X values are the same
        contingency = np.array([[50, 50], [0, 0]])
        mi = binary_mi_score(contingency)
        assert mi == 0.0  # Should return 0 for single cluster

        # All Y values are the same
        contingency = np.array([[50, 0], [50, 0]])
        mi = binary_mi_score(contingency)
        assert mi == 0.0

    def test_binary_mi_score_asymmetric(self):
        """Test MI score with asymmetric contingency table."""
        # Imbalanced but dependent
        contingency = np.array([[40, 10], [5, 45]])
        mi = binary_mi_score(contingency)

        # Should have significant MI
        assert mi > 0.2
        assert mi < 1.0  # Less than perfect dependence

    def test_binary_mi_score_sparse(self):
        """Test MI score with sparse contingency table."""
        # One cell is zero
        contingency = np.array([[30, 20], [25, 0]])
        mi = binary_mi_score(contingency)

        assert mi > 0  # Still has some MI
        assert mi < 0.5  # But not too high

    def test_binary_mi_score_numerical_stability(self):
        """Test numerical stability of MI calculation."""
        # Very small values
        contingency = np.array([[1, 1], [1, 1]])
        mi = binary_mi_score(contingency)
        assert mi >= 0
        assert not np.isnan(mi)
        assert not np.isinf(mi)

        # Large values
        contingency = np.array([[10000, 5000], [5000, 10000]])
        mi = binary_mi_score(contingency)
        assert mi >= 0
        assert not np.isnan(mi)
        assert not np.isinf(mi)

    def test_binary_mi_score_multiclass(self):
        """Test MI score with more than 2 classes."""
        # 3x3 contingency table
        contingency = np.array([[20, 5, 5], [5, 20, 5], [5, 5, 20]])
        mi = binary_mi_score(contingency)

        # Should have positive MI due to diagonal structure
        assert mi > 0.2  # Lower threshold since MI is in nats
        assert mi <= np.log(3)  # Maximum MI is log(min(|X|, |Y|)) in nats

    def test_binary_mi_score_empty_contingency(self):
        """Test MI score with empty contingency table."""
        # Empty table
        contingency = np.array([[0, 0], [0, 0]])
        mi = binary_mi_score(contingency)
        assert mi == 0.0

    def test_binary_mi_score_xor_pattern(self):
        """Test MI score with XOR-like pattern."""
        # XOR pattern has high MI
        contingency = np.array([[25, 25], [25, 25]])
        mi = binary_mi_score(contingency)
        assert abs(mi) < 0.01  # Uniform distribution has zero MI


class TestDigammaEdgeCases:
    """Test edge cases for digamma functions - non-positive values."""

    def test_py_fast_digamma_negative_returns_nan(self):
        """Test that negative values return NaN."""
        result = py_fast_digamma(-1.0)
        assert np.isnan(result)

        result = py_fast_digamma(-0.5)
        assert np.isnan(result)

        result = py_fast_digamma(-100.0)
        assert np.isnan(result)

    def test_py_fast_digamma_zero_returns_nan(self):
        """Test that zero returns NaN."""
        result = py_fast_digamma(0.0)
        assert np.isnan(result)

    def test_py_fast_digamma_int_input(self):
        """Test that integer input is handled correctly via float conversion."""
        # Integer inputs should work via float(x) conversion
        result_int = py_fast_digamma(5)
        result_float = py_fast_digamma(5.0)
        assert abs(result_int - result_float) < 1e-10

        result_int = py_fast_digamma(1)
        expected = digamma(1.0)
        assert abs(result_int - expected) < 1e-6

    def test_py_fast_digamma_numpy_scalar_types(self):
        """Test with various numpy scalar types."""
        x_float64 = np.float64(3.5)
        x_float32 = np.float32(3.5)
        x_int32 = np.int32(3)
        x_int64 = np.int64(3)

        # All should work and produce finite results
        assert not np.isnan(py_fast_digamma(x_float64))
        assert not np.isnan(py_fast_digamma(float(x_float32)))
        assert not np.isnan(py_fast_digamma(x_int32))
        assert not np.isnan(py_fast_digamma(x_int64))

    def test_py_fast_digamma_arr_negative_values(self):
        """Test array version with negative values returns NaN."""
        arr = np.array([-1.0, -0.5, -10.0])
        result = py_fast_digamma_arr(arr)
        assert all(np.isnan(result))

    def test_py_fast_digamma_arr_zero_values(self):
        """Test array version with zero returns NaN."""
        arr = np.array([0.0])
        result = py_fast_digamma_arr(arr)
        assert np.isnan(result[0])

    def test_py_fast_digamma_arr_mixed_positive_negative(self):
        """Test array version with mixed positive and negative values."""
        arr = np.array([1.0, -1.0, 2.0, -0.5, 3.0, 0.0])
        result = py_fast_digamma_arr(arr)

        # Positive values should have valid results
        assert not np.isnan(result[0])  # 1.0
        assert not np.isnan(result[2])  # 2.0
        assert not np.isnan(result[4])  # 3.0

        # Non-positive values should be NaN
        assert np.isnan(result[1])  # -1.0
        assert np.isnan(result[3])  # -0.5
        assert np.isnan(result[5])  # 0.0

        # Check positive values are accurate
        expected = digamma(np.array([1.0, 2.0, 3.0]))
        actual = np.array([result[0], result[2], result[4]])
        assert np.allclose(actual, expected, rtol=1e-6)

    def test_py_fast_digamma_arr_very_small_values(self):
        """Test with very small positive values exercising the while loop."""
        arr = np.array([0.001, 0.01, 0.1, 0.5])
        result = py_fast_digamma_arr(arr)
        expected = digamma(arr)
        assert np.allclose(result, expected, rtol=1e-5)


class TestBinaryMIScoreValidation:
    """Test input validation for binary_mi_score."""

    def test_binary_mi_score_1d_input_raises_error(self):
        """Test that 1D input raises ValueError."""
        import pytest
        contingency_1d = np.array([10, 20, 30])
        with pytest.raises(ValueError, match="must be 2D"):
            binary_mi_score(contingency_1d)

    def test_binary_mi_score_3d_input_raises_error(self):
        """Test that 3D input raises ValueError."""
        import pytest
        contingency_3d = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]])
        with pytest.raises(ValueError, match="must be 2D"):
            binary_mi_score(contingency_3d)

    def test_binary_mi_score_negative_values_raises_error(self):
        """Test that negative values in contingency table raise ValueError."""
        import pytest
        contingency = np.array([[10, -5], [20, 15]])
        with pytest.raises(ValueError, match="cannot contain negative"):
            binary_mi_score(contingency)

    def test_binary_mi_score_single_row_returns_zero(self):
        """Test that single-row contingency returns 0 (pi.size == 1)."""
        # Single row contingency table
        contingency = np.array([[50, 30, 20]])  # shape (1, 3)
        mi = binary_mi_score(contingency)
        assert mi == 0.0

    def test_binary_mi_score_single_column_returns_zero(self):
        """Test that single-column contingency returns 0 (pj.size == 1)."""
        # Single column contingency table
        contingency = np.array([[50], [30], [20]])  # shape (3, 1)
        mi = binary_mi_score(contingency)
        assert mi == 0.0
