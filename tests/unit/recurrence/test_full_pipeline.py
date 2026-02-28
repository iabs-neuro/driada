"""End-to-end integration test for the recurrence analysis pipeline."""

import numpy as np
import pytest
from driada.information.info_base import TimeSeries, MultiTimeSeries
from driada.recurrence import RecurrenceGraph
from driada.network import Network


class TestFullPipeline:
    """Test the complete user workflow."""

    def test_single_neuron_workflow(self):
        """Full workflow: TimeSeries -> tau -> m -> RecurrenceGraph -> RQA."""
        rng = np.random.default_rng(0)
        t = np.arange(2000)
        data = np.sin(2 * np.pi * t / 50) + 0.1 * rng.standard_normal(2000)
        ts = TimeSeries(data, discrete=False)

        tau = ts.estimate_tau(max_shift=60)
        assert 5 <= tau <= 20

        m = ts.estimate_embedding_dim(tau=tau, max_dim=8)
        assert 2 <= m <= 5

        # Use larger k to produce enough off-diagonal recurrence bands
        # for long diagonal lines (DET).  With small k the RP of a
        # low-dimensional ring attractor is too sparse for high DET.
        rg = ts.recurrence_graph(tau=tau, m=m, method='knn', k=30,
                                  theiler_window=1)
        assert isinstance(rg, RecurrenceGraph)
        assert rg.n > 0

        rqa = rg.rqa()
        assert rqa['DET'] > 0.0  # Non-trivial determinism for periodic signal
        assert 0 < rqa['RR'] < 1

        spectrum = rg.get_spectrum('adj')
        assert len(spectrum) > 0

    def test_population_workflow(self):
        """Full workflow: MultiTimeSeries -> population recurrence graph."""
        t = np.arange(500)
        rng = np.random.default_rng(42)
        n_neurons = 5
        data = np.vstack([
            np.sin(2 * np.pi * t / (30 + i * 2)) + 0.05 * rng.standard_normal(500)
            for i in range(n_neurons)
        ])
        mts = MultiTimeSeries(data, discrete=False)

        pop = mts.population_recurrence_graph(
            tau=8, m=3, method='joint', threshold=0.4,
            rg_method='knn', k=5, n_jobs=1,
        )
        assert isinstance(pop, RecurrenceGraph)
        assert pop.n > 0
        assert pop.adj.nnz > 0

    def test_caching_workflow(self):
        """Verify caching: same params -> same object, different combo -> fast."""
        t = np.arange(500)
        data = np.vstack([np.sin(2 * np.pi * t / 30) for _ in range(3)])
        mts = MultiTimeSeries(data, discrete=False, allow_zero_columns=True)

        pop1 = mts.population_recurrence_graph(
            tau=7, m=3, method='joint', rg_method='knn', k=5, n_jobs=1,
        )
        pop2 = mts.population_recurrence_graph(
            tau=7, m=3, method='mean', rg_method='knn', k=5, n_jobs=1,
        )

        assert pop1.adj.nnz > 0
        assert pop2.adj.nnz > 0

    def test_recurrence_then_spectral(self):
        """RecurrenceGraph spectral analysis should match Network API."""
        t = np.arange(500)
        data = np.sin(2 * np.pi * t / 30)
        ts = TimeSeries(data, discrete=False)
        rg = ts.recurrence_graph(tau=7, m=3)

        assert hasattr(rg, 'get_spectrum')
        assert hasattr(rg, 'deg')
        assert rg.directed is False

    def test_top_level_import(self):
        """RecurrenceGraph importable from top-level driada."""
        from driada import RecurrenceGraph as RG
        assert RG is RecurrenceGraph
