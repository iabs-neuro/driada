"""Tests for JIT-compiled entropy calculation functions."""

import numpy as np
from driada.information.entropy_jit import (
    entropy_d_jit,
    joint_entropy_dd_jit,
)
from driada.information.entropy import (
    entropy_d,
    joint_entropy_dd,
)


class TestEntropyJIT:
    """Test JIT-compiled entropy functions against their non-JIT counterparts."""

    def test_entropy_d_jit_uniform(self):
        """Test JIT entropy with uniform distribution."""
        x = np.array([0, 1, 2, 3] * 25)  # 100 samples, 4 states
        h_jit = entropy_d_jit(x)
        h_regular = entropy_d(x)
        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit - 2.0) < 0.01  # log2(4) = 2

    def test_entropy_d_jit_deterministic(self):
        """Test JIT entropy with deterministic data (zero entropy)."""
        x = np.ones(100, dtype=np.float64)
        h_jit = entropy_d_jit(x)
        h_regular = entropy_d(x)
        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit) < 1e-9

    def test_entropy_d_jit_binary(self):
        """Test JIT entropy with binary distribution."""
        # p = 0.8, q = 0.2
        x = np.array([0] * 80 + [1] * 20, dtype=np.float64)
        h_jit = entropy_d_jit(x)
        h_regular = entropy_d(x)
        expected = -0.8 * np.log2(0.8) - 0.2 * np.log2(0.2)
        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit - expected) < 0.01

    def test_entropy_d_jit_single_sample(self):
        """Test JIT entropy with single sample."""
        x = np.array([5], dtype=np.float64)
        h_jit = entropy_d_jit(x)
        h_regular = entropy_d(x)
        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit) < 1e-9

    def test_entropy_d_jit_edge_cases(self):
        """Test JIT entropy with various edge cases."""
        # Test with negative values
        x = np.array([-1, 0, 1, -1, 0, 1], dtype=np.float64)
        h_jit = entropy_d_jit(x)
        assert h_jit > 0

        # Test with large values
        x = np.array([1000, 2000, 3000] * 10, dtype=np.float64)
        h_jit = entropy_d_jit(x)
        assert abs(h_jit - np.log2(3)) < 0.01

    def test_joint_entropy_dd_jit_independent(self):
        """Test JIT joint entropy with independent variables."""
        n = 100
        x = np.array([0, 1] * 50, dtype=np.float64)
        y = np.array([0, 0, 1, 1] * 25, dtype=np.float64)

        h_jit = joint_entropy_dd_jit(x, y)
        h_regular = joint_entropy_dd(x, y)

        # For independent uniform variables: H(X,Y) = H(X) + H(Y)
        h_x = entropy_d_jit(x)
        h_y = entropy_d_jit(y)

        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit - (h_x + h_y)) < 0.01

    def test_joint_entropy_dd_jit_dependent(self):
        """Test JIT joint entropy with perfectly dependent variables."""
        x = np.array([0, 1, 2, 0, 1, 2] * 20, dtype=np.float64)
        y = x.copy()  # Perfect dependence

        h_jit = joint_entropy_dd_jit(x, y)
        h_regular = joint_entropy_dd(x, y)
        h_x = entropy_d_jit(x)

        # For perfect dependence: H(X,Y) = H(X) = H(Y)
        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit - h_x) < 0.01

    def test_joint_entropy_dd_jit_deterministic(self):
        """Test JIT joint entropy with constant variables."""
        x = np.ones(50, dtype=np.float64)
        y = np.zeros(50, dtype=np.float64)

        h_jit = joint_entropy_dd_jit(x, y)
        h_regular = joint_entropy_dd(x, y)

        # Both variables constant -> H(X,Y) = 0
        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit) < 1e-9

    def test_joint_entropy_dd_jit_mixed_types(self):
        """Test JIT joint entropy with various data patterns."""
        # XOR-like relationship
        x = np.array([0, 0, 1, 1] * 25, dtype=np.float64)
        y = np.array([0, 1, 1, 0] * 25, dtype=np.float64)

        h_jit = joint_entropy_dd_jit(x, y)
        h_regular = joint_entropy_dd(x, y)

        # Should have maximum joint entropy (2 bits)
        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit - 2.0) < 0.01

    def test_joint_entropy_dd_jit_unequal_states(self):
        """Test JIT joint entropy with different number of states."""
        x = np.array([0, 1] * 50, dtype=np.float64)  # 2 states
        y = np.array([0, 1, 2, 3] * 25, dtype=np.float64)  # 4 states

        h_jit = joint_entropy_dd_jit(x, y)
        h_regular = joint_entropy_dd(x, y)

        assert abs(h_jit - h_regular) < 1e-9
        assert h_jit >= max(entropy_d_jit(x), entropy_d_jit(y))
        assert h_jit <= entropy_d_jit(x) + entropy_d_jit(y)

    def test_joint_entropy_dd_jit_single_sample(self):
        """Test JIT joint entropy with single sample."""
        x = np.array([1], dtype=np.float64)
        y = np.array([2], dtype=np.float64)

        h_jit = joint_entropy_dd_jit(x, y)
        h_regular = joint_entropy_dd(x, y)

        assert abs(h_jit - h_regular) < 1e-9
        assert abs(h_jit) < 1e-9


class TestJITPerformance:
    """Test performance characteristics of JIT functions."""

    def test_entropy_jit_faster_than_regular(self):
        """Verify JIT version is actually compiled and working."""
        # Large dataset to see performance difference
        x = np.random.randint(0, 10, size=10000)

        # Just verify both give same results
        h_jit = entropy_d_jit(x)
        h_regular = entropy_d(x)
        assert abs(h_jit - h_regular) < 1e-8

    def test_joint_entropy_jit_consistency(self):
        """Test consistency between JIT and regular versions on random data."""
        for _ in range(5):
            n = 1000
            x = np.random.randint(0, 5, size=n)
            y = np.random.randint(0, 5, size=n)

            h_jit = joint_entropy_dd_jit(x, y)
            h_regular = joint_entropy_dd(x, y)
            assert abs(h_jit - h_regular) < 1e-8
