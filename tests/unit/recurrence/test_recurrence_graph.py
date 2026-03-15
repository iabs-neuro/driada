"""Tests for RecurrenceGraph class."""

import numpy as np
import pytest
import scipy.sparse as sp
from driada.recurrence import RecurrenceGraph, takens_embedding
from driada.network import Network
from driada.dim_reduction.graph import ProximityGraph


class TestRecurrenceGraphConstruction:
    """Test RecurrenceGraph initialization."""

    @pytest.fixture
    def sine_embedded(self):
        """Sine wave embedded in 2D."""
        t = np.arange(500)
        data = np.sin(2 * np.pi * t / 40)
        return takens_embedding(data, tau=10, m=3)

    def test_is_proximity_graph(self, sine_embedded):
        """RecurrenceGraph must inherit from ProximityGraph with valid structure."""
        rg = RecurrenceGraph(sine_embedded, method='knn', k=5)
        assert isinstance(rg, ProximityGraph)
        assert isinstance(rg, Network)
        assert rg.n == sine_embedded.shape[1]
        assert rg.adj.nnz > 0

    def test_knn_method(self, sine_embedded):
        """k-NN method produces sparse binary adjacency."""
        rg = RecurrenceGraph(sine_embedded, method='knn', k=5)
        assert rg.adj.shape[0] == sine_embedded.shape[1]
        assert rg.adj.nnz > 0

    def test_eps_method(self, sine_embedded):
        """Epsilon-ball method produces sparse adjacency."""
        rg = RecurrenceGraph(sine_embedded, method='eps', epsilon=0.5)
        assert rg.adj.shape[0] == sine_embedded.shape[1]

    def test_rejects_1d(self):
        """Must reject 1D input with helpful message."""
        with pytest.raises(ValueError, match="takens_embedding"):
            RecurrenceGraph(np.random.randn(100), method='knn', k=5)

    def test_theiler_window_removes_band(self, sine_embedded):
        """Theiler window should remove near-diagonal entries.

        Uses eps method which produces dense enough graphs to have
        near-diagonal entries (k-NN on periodic signals connects
        points exactly one period apart, not temporally adjacent).
        """
        rg_no_theiler = RecurrenceGraph(
            sine_embedded, method='eps', epsilon=1.0, theiler_window=None,
        )
        rg_theiler = RecurrenceGraph(
            sine_embedded, method='eps', epsilon=1.0, theiler_window=5,
        )
        # Theiler version should have fewer entries
        assert rg_theiler.adj.nnz < rg_no_theiler.adj.nnz

    def test_no_preprocessing_default(self, sine_embedded):
        """All time points must be preserved (no giant_cc extraction)."""
        rg = RecurrenceGraph(sine_embedded, method='knn', k=5)
        assert rg.n == sine_embedded.shape[1]

    def test_spectral_analysis_works(self, sine_embedded):
        """Inherited Network spectral methods must work."""
        rg = RecurrenceGraph(sine_embedded, method='knn', k=5)
        spectrum = rg.get_spectrum('adj')
        assert len(spectrum) > 0

    def test_recurrence_rate_property(self, sine_embedded):
        """Recurrence rate should be between 0 and 1."""
        rg = RecurrenceGraph(sine_embedded, method='knn', k=5)
        assert 0 < rg.recurrence_rate < 1

    def test_from_adjacency(self):
        """from_adjacency should create valid RecurrenceGraph."""
        n = 50
        adj = sp.random(n, n, density=0.1, format='csr')
        adj = (adj + adj.T > 0).astype(float)  # Make symmetric binary
        rg = RecurrenceGraph.from_adjacency(adj)
        assert isinstance(rg, RecurrenceGraph)
        assert isinstance(rg, Network)
        assert rg.n == n
        assert rg.adj.nnz > 0

    def test_from_adjacency_theiler_recorded(self):
        """from_adjacency should record theiler_window."""
        adj = sp.eye(20, format='csr')
        rg = RecurrenceGraph.from_adjacency(adj, theiler_window=10)
        assert rg.theiler_window == 10

    def test_negative_theiler_window_raises(self, sine_embedded):
        """Negative theiler_window must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            RecurrenceGraph(sine_embedded, method='knn', k=5, theiler_window=-1)

    def test_theiler_degrees_consistent(self, sine_embedded):
        """Degree sequences must reflect the Theiler-filtered adjacency.

        Verifies that the construct_adjacency override applies the Theiler
        window BEFORE Network computes degrees — not after, which would
        leave stale degree sequences.
        """
        rg = RecurrenceGraph(
            sine_embedded, method='eps', epsilon=1.0, theiler_window=5,
        )
        # Recompute degrees directly from the adjacency matrix
        expected_deg = np.asarray(rg.adj.sum(axis=1)).ravel()
        np.testing.assert_array_equal(rg.deg, expected_deg)
