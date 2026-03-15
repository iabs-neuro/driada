"""Integration tests for VG/OPN via TimeSeries API."""

import numpy as np
import pytest
from driada.information import TimeSeries
from driada.network import Network


class TestTimeSeriesVisibilityGraph:
    """Test ts.visibility_graph() delegation."""

    def test_basic_call(self):
        data = np.random.randn(100)
        ts = TimeSeries(data, discrete=False)
        vg = ts.visibility_graph()
        assert isinstance(vg, Network)
        assert vg.n == len(data)

    def test_caching(self):
        data = np.random.randn(100)
        ts = TimeSeries(data, discrete=False)
        vg1 = ts.visibility_graph()
        vg2 = ts.visibility_graph()
        assert vg1 is vg2  # cached

    def test_different_method_invalidates_cache(self):
        data = np.random.randn(100)
        ts = TimeSeries(data, discrete=False)
        hvg = ts.visibility_graph(method='horizontal')
        nvg = ts.visibility_graph(method='natural')
        assert hvg is not nvg

    def test_discrete_raises(self):
        data = np.array([1, 2, 3, 1, 2])
        ts = TimeSeries(data, discrete=True)
        with pytest.raises(ValueError, match="continuous"):
            ts.visibility_graph()


class TestTimeSeriesOPN:
    """Test ts.ordinal_partition_network() and ts.permutation_entropy()."""

    def test_basic_call(self):
        np.random.seed(42)
        data = np.random.randn(500)
        ts = TimeSeries(data, discrete=False)
        opn = ts.ordinal_partition_network(d=3, tau=1)
        assert isinstance(opn, Network)
        assert opn.n > 0
        assert opn.adj.nnz > 0

    def test_auto_estimation(self):
        t = np.arange(500)
        data = np.sin(2 * np.pi * t / 40) + 0.1 * np.random.randn(500)
        ts = TimeSeries(data, discrete=False)
        opn = ts.ordinal_partition_network()  # auto d and tau
        assert isinstance(opn, Network)
        assert opn.n > 0
        pe = ts.permutation_entropy()
        assert 0.0 < pe <= 1.0

    def test_permutation_entropy_shortcut(self):
        np.random.seed(42)
        data = np.random.randn(500)
        ts = TimeSeries(data, discrete=False)
        pe = ts.permutation_entropy(d=3, tau=1)
        assert 0.0 <= pe <= 1.0

    def test_caching(self):
        np.random.seed(42)
        data = np.random.randn(200)
        ts = TimeSeries(data, discrete=False)
        opn1 = ts.ordinal_partition_network(d=3, tau=1)
        opn2 = ts.ordinal_partition_network(d=3, tau=1)
        assert opn1 is opn2

    def test_clear_caches(self):
        np.random.seed(42)
        data = np.random.randn(200)
        ts = TimeSeries(data, discrete=False)
        ts.visibility_graph()
        ts.ordinal_partition_network(d=3, tau=1)
        ts.clear_caches()
        assert ts._vg_cache is None
        assert ts._opn_cache is None
