"""Test automatic switching between JIT and regular implementations."""

import numpy as np
import pytest
from driada.information import entropy


class TestAutoSwitch:
    """Test automatic switching mechanism."""

    def test_entropy_d_auto_switch_small(self):
        """Test that small arrays use JIT implementation."""
        # Small array should use JIT
        x_small = np.random.randint(0, 10, size=500)
        h = entropy.entropy_d(x_small)

        # Result should be valid entropy
        assert 0 <= h <= np.log2(10)

    def test_entropy_d_auto_switch_large(self):
        """Test that large arrays use numpy implementation."""
        # Large array should use numpy
        x_large = np.random.randint(0, 10, size=5000)
        h = entropy.entropy_d(x_large)

        # Result should be valid entropy
        assert 0 <= h <= np.log2(10)

    def test_entropy_d_consistency_across_threshold(self):
        """Test that results are consistent across the switching threshold."""
        # Test right at the threshold
        np.random.seed(42)
        x_just_below = np.random.randint(0, 5, size=999)
        x_just_above = np.random.randint(0, 5, size=1001)

        # Make them identical except for size
        x_just_above[:999] = x_just_below
        x_just_above[999:] = x_just_below[0:2]

        h_below = entropy.entropy_d(x_just_below)
        h_above = entropy.entropy_d(x_just_above)

        # Entropies should be very close
        assert abs(h_below - h_above) < 0.01

    def test_joint_entropy_dd_always_jit(self):
        """Test that joint entropy always uses JIT."""
        # Test various sizes
        sizes = [100, 1000, 10000]

        for size in sizes:
            x = np.random.randint(0, 10, size=size)
            y = np.random.randint(0, 10, size=size)

            h = entropy.joint_entropy_dd(x, y)

            # Result should be valid joint entropy
            assert 0 <= h <= 2 * np.log2(10)

    def test_auto_switch_correctness(self):
        """Test that auto-switching produces correct results."""
        np.random.seed(42)

        # Test data with known entropy
        # Binary data with p=0.5 should have entropy = 1 bit
        x_binary = np.array([0, 1] * 250)  # 500 elements
        h_binary = entropy.entropy_d(x_binary)
        assert abs(h_binary - 1.0) < 1e-10

        # Uniform distribution over 4 values should have entropy = 2 bits
        x_uniform = np.array([0, 1, 2, 3] * 250)  # 1000 elements (at threshold)
        h_uniform = entropy.entropy_d(x_uniform)
        assert abs(h_uniform - 2.0) < 1e-8

        # Test with larger array
        x_uniform_large = np.array([0, 1, 2, 3] * 500)  # 2000 elements
        h_uniform_large = entropy.entropy_d(x_uniform_large)
        assert abs(h_uniform_large - 2.0) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
