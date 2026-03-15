"""Tests for TimeSeries/MultiTimeSeries recurrence integration."""

import numpy as np
import pytest
from driada.information.info_base import TimeSeries, MultiTimeSeries
from driada.recurrence import RecurrenceGraph


class TestTimeSeriesRecurrence:
    """Tests for TimeSeries recurrence methods."""

    @pytest.fixture
    def sine_ts(self):
        """Sine wave TimeSeries."""
        t = np.arange(1000)
        data = np.sin(2 * np.pi * t / 40)
        return TimeSeries(data, discrete=False)

    def test_estimate_tau(self, sine_ts):
        """TimeSeries.estimate_tau() returns reasonable value."""
        tau = sine_ts.estimate_tau(max_shift=50)
        assert isinstance(tau, (int, np.integer))
        assert 5 <= tau <= 15

    def test_estimate_tau_cached(self, sine_ts):
        """Second call returns cached value."""
        tau1 = sine_ts.estimate_tau(max_shift=50)
        tau2 = sine_ts.estimate_tau(max_shift=50)
        assert tau1 == tau2

    def test_estimate_embedding_dim(self, sine_ts):
        """TimeSeries.estimate_embedding_dim() returns reasonable value."""
        m = sine_ts.estimate_embedding_dim(tau=10, max_dim=8)
        assert isinstance(m, (int, np.integer))
        assert 2 <= m <= 4

    def test_takens_embedding(self, sine_ts):
        """TimeSeries.takens_embedding() returns correct shape."""
        emb = sine_ts.takens_embedding(tau=10, m=3)
        assert emb.shape[0] == 3
        assert emb.shape[1] == 1000 - 2 * 10

    def test_recurrence_graph(self, sine_ts):
        """TimeSeries.recurrence_graph() returns RecurrenceGraph."""
        rg = sine_ts.recurrence_graph(tau=10, m=3, method='knn', k=5)
        assert isinstance(rg, RecurrenceGraph)

    def test_recurrence_graph_auto_theiler(self, sine_ts):
        """Auto theiler_window = tau*(m-1)+1."""
        rg = sine_ts.recurrence_graph(tau=10, m=3, theiler_window='auto')
        assert rg.theiler_window == 10 * 2 + 1

    def test_rqa(self, sine_ts):
        """TimeSeries.rqa() returns dict with RQA measures."""
        result = sine_ts.rqa(tau=10, m=3)
        assert 'DET' in result
        assert 'LAM' in result

    def test_discrete_ts_raises(self):
        """Recurrence methods should raise for discrete TimeSeries."""
        ts = TimeSeries(np.array([0, 1, 0, 1, 1, 0]), discrete=True)
        with pytest.raises(ValueError, match="continuous"):
            ts.estimate_tau()

    def test_clear_caches_clears_recurrence(self, sine_ts):
        """clear_caches() should clear recurrence caches."""
        sine_ts.estimate_tau(max_shift=50)
        sine_ts.clear_caches()
        assert not hasattr(sine_ts, '_recurrence_tau') or sine_ts._recurrence_tau is None


class TestMultiTimeSeriesRecurrence:
    """Tests for MultiTimeSeries recurrence methods."""

    @pytest.fixture
    def sine_mts(self):
        """3 sine waves as MultiTimeSeries."""
        t = np.arange(500)
        data = np.vstack([
            np.sin(2 * np.pi * t / 30),
            np.sin(2 * np.pi * t / 35),
            np.sin(2 * np.pi * t / 25),
        ])
        return MultiTimeSeries(data, discrete=False, allow_zero_columns=True)

    def test_population_recurrence_graph(self, sine_mts):
        """MTS.population_recurrence_graph() returns a RecurrenceGraph."""
        pop = sine_mts.population_recurrence_graph(tau=8, m=3, method='joint', n_jobs=1)
        assert isinstance(pop, RecurrenceGraph)

    def test_population_caches_individual_graphs(self, sine_mts):
        """After population call, individual TS should have cached graphs."""
        # Verify no cache before
        ts0 = sine_mts.ts_list[0]
        assert not hasattr(ts0, '_recurrence_graph_cache') or ts0._recurrence_graph_cache is None

        sine_mts.population_recurrence_graph(tau=8, m=3, method='joint', n_jobs=1)

        # Cache must exist after population call
        assert ts0._recurrence_graph_cache is not None, (
            "population_recurrence_graph did not populate per-component cache"
        )
        cached_key, cached_rg = ts0._recurrence_graph_cache
        assert isinstance(cached_rg, RecurrenceGraph)


class TestCreateNxGraph:
    """Tests for create_nx_graph=True on recurrence graph types."""

    @pytest.fixture
    def sine_ts(self):
        t = np.arange(200)
        return TimeSeries(np.sin(2 * np.pi * t / 50))

    def test_recurrence_graph_creates_nx_graph(self, sine_ts):
        """RecurrenceGraph with create_nx_graph=True has .graph."""
        rg = sine_ts.recurrence_graph(tau=10, m=3, k=5, create_nx_graph=True)
        assert rg.graph is not None
        assert rg.graph.number_of_nodes() > 0
        assert rg.graph.number_of_edges() > 0

    def test_recurrence_graph_default_no_graph(self, sine_ts):
        """RecurrenceGraph default has .graph = None."""
        rg = sine_ts.recurrence_graph(tau=10, m=3, k=5)
        assert rg.graph is None

    def test_visibility_graph_creates_nx_graph(self, sine_ts):
        """VisibilityGraph with create_nx_graph=True has .graph."""
        vg = sine_ts.visibility_graph(method="horizontal", create_nx_graph=True)
        assert vg.graph is not None
        assert vg.graph.number_of_nodes() == len(sine_ts.data)

    def test_opn_creates_nx_graph(self, sine_ts):
        """OrdinalPartitionNetwork with create_nx_graph=True has .graph."""
        opn = sine_ts.ordinal_partition_network(tau=10, create_nx_graph=True)
        assert opn.graph is not None
        assert opn.graph.number_of_nodes() > 0

    def test_from_adjacency_creates_nx_graph(self, sine_ts):
        """RecurrenceGraph.from_adjacency with create_nx_graph=True."""
        rg = sine_ts.recurrence_graph(tau=10, m=3, k=5)
        rg2 = RecurrenceGraph.from_adjacency(rg.adj, create_nx_graph=True)
        assert rg2.graph is not None
        assert rg2.graph.number_of_nodes() == rg.n
