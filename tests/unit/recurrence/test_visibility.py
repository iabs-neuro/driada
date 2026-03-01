"""Tests for VisibilityGraph class."""

import numpy as np
import pytest
import scipy.sparse as sp
from driada.network import Network


class TestHorizontalVisibilityGraph:
    """Test HVG construction and properties."""

    def test_monotonic_is_path_graph(self):
        """Monotonic increasing signal: each point sees only adjacent points."""
        from driada.recurrence import VisibilityGraph
        data = np.arange(20, dtype=float)
        vg = VisibilityGraph(data, method='horizontal')
        # Path graph: n-1 edges (undirected, stored as 2*(n-1) nonzeros)
        assert vg.adj.nnz == 2 * (len(data) - 1)

    def test_constant_is_complete(self):
        """Constant signal: all points visible to all (no blocking)."""
        from driada.recurrence import VisibilityGraph
        data = np.ones(10)
        vg = VisibilityGraph(data, method='horizontal')
        n = len(data)
        # Complete graph: n*(n-1) nonzeros (symmetric)
        assert vg.adj.nnz == n * (n - 1)

    def test_spike_is_hub(self):
        """Single spike in flat signal: spike node has highest degree."""
        from driada.recurrence import VisibilityGraph
        data = np.zeros(20)
        data[10] = 10.0  # spike at position 10
        vg = VisibilityGraph(data, method='horizontal')
        degrees = np.array(vg.adj.sum(axis=1)).ravel()
        assert np.argmax(degrees) == 10

    def test_inherits_network(self):
        """Result must be a Network instance with working spectral methods."""
        from driada.recurrence import VisibilityGraph
        data = np.random.randn(50)
        vg = VisibilityGraph(data, method='horizontal')
        assert isinstance(vg, Network)
        assert hasattr(vg, 'adj')
        assert hasattr(vg, 'deg')
        assert vg.n == len(data)

    def test_adjacency_is_symmetric(self):
        """Undirected VG must produce symmetric adjacency."""
        from driada.recurrence import VisibilityGraph
        data = np.random.randn(30)
        vg = VisibilityGraph(data, method='horizontal')
        diff = vg.adj - vg.adj.T
        assert diff.nnz == 0
