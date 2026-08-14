"""Integration tests for routing and end-to-end equivalence of the high-d FFT path.

With ``MAX_FFT_MTS_DIMENSIONS = 5``, MultiTimeSeries of dimension 4 and 5 must be
routed through the FFT engine (continuous and discrete), while dimension 6 still
falls back to the loop engine. The FFT observed MI must match the loop engine.
"""
import numpy as np
import pytest

from driada.information.info_base import TimeSeries, MultiTimeSeries
from driada.intense.fft import get_fft_type, FFT_MULTIVARIATE, FFT_MTS_DISCRETE
from driada.intense import compute_me_stats


def _continuous_mts(d, n, seed):
    rng = np.random.RandomState(seed)
    return MultiTimeSeries([TimeSeries(rng.randn(n), discrete=False) for _ in range(d)])


class TestRouting:
    @pytest.mark.parametrize("d", [4, 5])
    def test_continuous_mts_routes_to_fft(self, d):
        n = 300
        mts = _continuous_mts(d, n, seed=d)
        ts = TimeSeries(np.random.RandomState(1).randn(n), discrete=False)
        assert get_fft_type(mts, ts, metric="mi", mi_estimator="gcmi",
                            count=50, engine="auto") == FFT_MULTIVARIATE

    @pytest.mark.parametrize("d", [4, 5])
    def test_discrete_mts_routes_to_fft(self, d):
        n = 300
        mts = _continuous_mts(d, n, seed=10 + d)
        disc = TimeSeries(np.random.RandomState(2).randint(0, 3, n).astype(float),
                          discrete=True)
        assert get_fft_type(mts, disc, metric="mi", mi_estimator="gcmi",
                            count=50, engine="auto") == FFT_MTS_DISCRETE

    def test_d6_still_falls_back(self):
        n = 300
        mts = _continuous_mts(6, n, seed=99)
        disc = TimeSeries(np.random.RandomState(3).randint(0, 3, n).astype(float),
                          discrete=True)
        assert get_fft_type(mts, disc, metric="mi", mi_estimator="gcmi",
                            count=50, engine="auto") is None
        with pytest.raises(ValueError, match="no FFT optimization is applicable"):
            get_fft_type(mts, disc, metric="mi", mi_estimator="gcmi",
                        count=50, engine="fft")


class TestEndToEndEquivalence:
    @pytest.mark.parametrize("d", [4, 5])
    def test_fft_matches_loop_observed_mi(self, d):
        """Observed MI (stats['me']) must match between FFT and loop engines."""
        rng = np.random.RandomState(500 + d)
        n = 600
        z = rng.randn(n)
        # Feature dimensions partially coupled to z so MI is non-trivial.
        rows = [(0.25 + 0.08 * k) * z
                + np.sqrt(1 - (0.25 + 0.08 * k) ** 2) * rng.randn(n)
                for k in range(d)]
        mts = MultiTimeSeries([TimeSeries(r, discrete=False) for r in rows])
        mts.name = "feature"
        ts = TimeSeries(z, discrete=False)
        ts.name = "neuron"

        common = dict(metric="mi", n_shuffles_stage2=30, mode="stage2",
                      mi_estimator="gcmi")
        stats_fft, _, _ = compute_me_stats([ts], [mts], engine="fft", **common)
        stats_loop, _, _ = compute_me_stats([ts], [mts], engine="loop", **common)

        me_fft = stats_fft[0][0]["me"]
        me_loop = stats_loop[0][0]["me"]
        np.testing.assert_allclose(me_fft, me_loop, rtol=1e-4, atol=1e-6)
