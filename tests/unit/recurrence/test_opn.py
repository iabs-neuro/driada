"""Tests for OrdinalPartitionNetwork class."""

import math
import numpy as np
import pytest
from driada.network import Network


class TestOrdinalPartitionNetwork:
    """Test OPN construction and properties."""

    def test_known_patterns_d3(self):
        """Verify exact pattern encoding for d=3, known example."""
        from driada.recurrence import OrdinalPartitionNetwork
        # Deterministic signal: always increasing -> pattern (0,1,2) = "123"
        data = np.arange(20, dtype=float)
        opn = OrdinalPartitionNetwork(data, d=3, tau=1)
        # With monotonic data, only one pattern should appear
        assert opn.missing_patterns == math.factorial(3) - 1  # 5 missing

    def test_noise_visits_all_patterns(self):
        """White noise with enough data should visit all d! patterns."""
        from driada.recurrence import OrdinalPartitionNetwork
        np.random.seed(42)
        data = np.random.randn(10000)
        opn = OrdinalPartitionNetwork(data, d=3, tau=1)
        assert opn.missing_patterns == 0

    def test_noise_high_permutation_entropy(self):
        """White noise should have permutation entropy near 1.0."""
        from driada.recurrence import OrdinalPartitionNetwork
        np.random.seed(42)
        data = np.random.randn(10000)
        opn = OrdinalPartitionNetwork(data, d=3, tau=1)
        assert opn.permutation_entropy > 0.95

    def test_periodic_low_permutation_entropy(self):
        """Periodic signal should have low permutation entropy."""
        from driada.recurrence import OrdinalPartitionNetwork
        t = np.arange(5000)
        data = np.sin(2 * np.pi * t / 50)
        opn = OrdinalPartitionNetwork(data, d=3, tau=1)
        assert opn.permutation_entropy < 0.5

    def test_inherits_network_directed(self):
        """OPN must be a directed weighted Network."""
        from driada.recurrence import OrdinalPartitionNetwork
        np.random.seed(42)
        data = np.random.randn(200)
        opn = OrdinalPartitionNetwork(data, d=3, tau=1)
        assert isinstance(opn, Network)
        assert opn.n <= math.factorial(3)  # at most d! nodes

    def test_transition_matrix_rows_sum_to_one(self):
        """Row sums of the adjacency (transition) matrix should be ~1."""
        from driada.recurrence import OrdinalPartitionNetwork
        np.random.seed(42)
        data = np.random.randn(1000)
        opn = OrdinalPartitionNetwork(data, d=3, tau=1)
        row_sums = np.array(opn.adj.sum(axis=1)).ravel()
        # Only rows with outgoing edges should sum to ~1
        nonzero_rows = row_sums[row_sums > 0]
        assert np.allclose(nonzero_rows, 1.0, atol=0.01)

    def test_rejects_d_too_large(self):
        """d > 7 should raise ValueError."""
        from driada.recurrence import OrdinalPartitionNetwork
        with pytest.raises(ValueError, match="<= 7"):
            OrdinalPartitionNetwork(np.random.randn(100), d=8, tau=1)

    def test_rejects_1d_check(self):
        """Must reject non-1D input."""
        from driada.recurrence import OrdinalPartitionNetwork
        with pytest.raises(ValueError, match="1D"):
            OrdinalPartitionNetwork(np.random.randn(10, 2), d=3, tau=1)
