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

    def test_constant_is_path(self):
        """Constant signal: HVG strict inequality means only adjacent visible.

        Standard definition (Luque et al. 2009): x_k < min(x_i, x_j).
        For constant data, x_k = min(x_i, x_j), so condition fails —
        only adjacent pairs (no intermediary) are connected → path graph.
        """
        from driada.recurrence import VisibilityGraph
        data = np.ones(10)
        vg = VisibilityGraph(data, method='horizontal')
        # Path graph: n-1 edges (undirected, stored as 2*(n-1) nonzeros)
        assert vg.adj.nnz == 2 * (len(data) - 1)

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
        np.random.seed(7)
        data = np.random.randn(50)
        vg = VisibilityGraph(data, method='horizontal')
        assert isinstance(vg, Network)
        assert hasattr(vg, 'adj')
        assert hasattr(vg, 'deg')
        assert vg.n == len(data)

    def test_adjacency_is_symmetric(self):
        """Undirected VG must produce symmetric adjacency."""
        from driada.recurrence import VisibilityGraph
        np.random.seed(8)
        data = np.random.randn(30)
        vg = VisibilityGraph(data, method='horizontal')
        diff = vg.adj - vg.adj.T
        assert diff.nnz == 0


class TestNaturalVisibilityGraph:
    """Test NVG construction."""

    def test_nvg_monotonic_is_path(self):
        """Monotonic signal: NVG is also a path graph."""
        from driada.recurrence import VisibilityGraph
        data = np.arange(15, dtype=float)
        vg = VisibilityGraph(data, method='natural')
        assert vg.adj.nnz == 2 * (len(data) - 1)

    def test_nvg_has_more_edges_than_hvg(self):
        """NVG is a supergraph of HVG (more or equal edges)."""
        from driada.recurrence import VisibilityGraph
        np.random.seed(42)
        data = np.random.randn(50)
        hvg = VisibilityGraph(data, method='horizontal')
        nvg = VisibilityGraph(data, method='natural')
        assert nvg.adj.nnz >= hvg.adj.nnz

    def test_hvg_is_subgraph_of_nvg(self):
        """Every HVG edge must also appear in NVG."""
        from driada.recurrence import VisibilityGraph
        np.random.seed(123)
        data = np.random.randn(30)
        hvg = VisibilityGraph(data, method='horizontal')
        nvg = VisibilityGraph(data, method='natural')
        # Every nonzero in HVG must be nonzero in NVG
        hvg_dense = hvg.adj.toarray()
        nvg_dense = nvg.adj.toarray()
        assert np.all(nvg_dense[hvg_dense > 0] > 0)

    def test_directed_is_upper_triangular(self):
        """Directed VG should only have forward-in-time edges."""
        from driada.recurrence import VisibilityGraph
        np.random.seed(9)
        data = np.random.randn(20)
        vg = VisibilityGraph(data, method='horizontal', directed=True)
        adj_dense = vg.adj.toarray()
        assert np.allclose(np.tril(adj_dense, k=-1), 0)

    def test_rejects_2d(self):
        """Must reject non-1D input."""
        from driada.recurrence import VisibilityGraph
        with pytest.raises(ValueError, match="1D"):
            VisibilityGraph(np.random.randn(10, 2))


class TestVisibilityGraphValidation:
    """Test input validation for VisibilityGraph."""

    def test_rejects_empty_array(self):
        """Empty array must raise ValueError."""
        from driada.recurrence import VisibilityGraph
        with pytest.raises(ValueError, match="at least 2"):
            VisibilityGraph(np.array([]))

    def test_rejects_single_point(self):
        """Single data point must raise ValueError."""
        from driada.recurrence import VisibilityGraph
        with pytest.raises(ValueError, match="at least 2"):
            VisibilityGraph(np.array([1.0]))

    def test_rejects_nan(self):
        """NaN values must raise ValueError."""
        from driada.recurrence import VisibilityGraph
        data = np.array([1.0, np.nan, 3.0, 4.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            VisibilityGraph(data)

    def test_rejects_inf(self):
        """Inf values must raise ValueError."""
        from driada.recurrence import VisibilityGraph
        data = np.array([1.0, np.inf, 3.0, 4.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            VisibilityGraph(data)

    def test_two_points_minimal(self):
        """Two data points should work and produce one edge."""
        from driada.recurrence import VisibilityGraph
        vg = VisibilityGraph(np.array([1.0, 2.0]))
        assert vg.adj.nnz == 2  # one undirected edge = 2 entries

    def test_nvg_large_n_warns(self):
        """NVG with N > 2000 should emit a warning."""
        from driada.recurrence import VisibilityGraph
        data = np.random.randn(2001)
        with pytest.warns(UserWarning, match="slow"):
            VisibilityGraph(data, method='natural')
