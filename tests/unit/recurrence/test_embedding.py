"""Tests for driada.recurrence.embedding."""

import numpy as np
import pytest
from driada.recurrence.embedding import takens_embedding


class TestTakensEmbedding:
    """Tests for takens_embedding()."""

    def test_shape_basic(self):
        """Embedding of N=100 with tau=1, m=3 -> (3, 98)."""
        data = np.random.randn(100)
        result = takens_embedding(data, tau=1, m=3)
        assert result.shape == (3, 98)

    def test_shape_large_tau(self):
        """Embedding of N=100 with tau=5, m=4 -> (4, 85)."""
        data = np.random.randn(100)
        result = takens_embedding(data, tau=5, m=4)
        assert result.shape == (4, 85)

    def test_values_manual(self):
        """Verify embedding values match manual construction."""
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        result = takens_embedding(data, tau=2, m=3)
        # Column 0: [data[0], data[2], data[4]] = [0, 2, 4]
        # Column 1: [data[1], data[3], data[5]] = [1, 3, 5]
        np.testing.assert_array_equal(result[:, 0], [0, 2, 4])
        np.testing.assert_array_equal(result[:, 1], [1, 3, 5])

    def test_rejects_2d_input(self):
        """Must reject 2D arrays."""
        with pytest.raises(ValueError, match="1D"):
            takens_embedding(np.random.randn(10, 2), tau=1, m=2)

    def test_rejects_short_series(self):
        """Must reject series shorter than (m-1)*tau + 1."""
        with pytest.raises(ValueError, match="too short"):
            takens_embedding(np.array([1.0, 2.0]), tau=3, m=5)

    def test_sinusoid_period(self):
        """Sine wave embedding with quarter-period tau gives circular attractor."""
        t = np.linspace(0, 10 * np.pi, 1000)
        data = np.sin(t)
        # Period ~ 200 samples (5 full cycles in 1000 points).
        # Quarter-period tau ~ 50 gives sin vs cos -> near-zero correlation.
        result = takens_embedding(data, tau=50, m=2)
        corr = np.abs(np.corrcoef(result[0], result[1])[0, 1])
        assert corr < 0.3
