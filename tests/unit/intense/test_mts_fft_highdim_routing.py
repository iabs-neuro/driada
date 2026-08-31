"""Integration tests for routing and end-to-end equivalence of the high-d FFT path.

MultiTimeSeries up to ``MAX_FFT_MTS_DIMENSIONS`` must be routed through the FFT
engine (continuous and discrete), while anything above that bound falls back to
the loop engine. The FFT observed MI must match the loop engine.
"""
import numpy as np
import pytest

from driada.information.info_base import TimeSeries, MultiTimeSeries
from driada.intense.fft import (
    get_fft_type,
    FFT_MULTIVARIATE,
    FFT_MTS_DISCRETE,
    MAX_FFT_MTS_DIMENSIONS,
)
from driada.intense import compute_me_stats


def _continuous_mts(d, n, seed):
    rng = np.random.RandomState(seed)
    return MultiTimeSeries([TimeSeries(rng.randn(n), discrete=False) for _ in range(d)])


class TestRouting:
    @pytest.mark.parametrize("d", [4, 5, MAX_FFT_MTS_DIMENSIONS])
    def test_continuous_mts_routes_to_fft(self, d):
        n = 300
        mts = _continuous_mts(d, n, seed=d)
        ts = TimeSeries(np.random.RandomState(1).randn(n), discrete=False)
        assert get_fft_type(mts, ts, metric="mi", mi_estimator="gcmi",
                            count=50, engine="auto") == FFT_MULTIVARIATE

    @pytest.mark.parametrize("d", [4, 5, MAX_FFT_MTS_DIMENSIONS])
    def test_discrete_mts_routes_to_fft(self, d):
        n = 300
        mts = _continuous_mts(d, n, seed=10 + d)
        disc = TimeSeries(np.random.RandomState(2).randint(0, 3, n).astype(float),
                          discrete=True)
        assert get_fft_type(mts, disc, metric="mi", mi_estimator="gcmi",
                            count=50, engine="auto") == FFT_MTS_DISCRETE

    def test_above_limit_still_falls_back(self):
        n = 300
        mts = _continuous_mts(MAX_FFT_MTS_DIMENSIONS + 1, n, seed=99)
        disc = TimeSeries(np.random.RandomState(3).randint(0, 3, n).astype(float),
                          discrete=True)
        assert get_fft_type(mts, disc, metric="mi", mi_estimator="gcmi",
                            count=50, engine="auto") is None
        with pytest.raises(ValueError, match="no FFT optimization is applicable"):
            get_fft_type(mts, disc, metric="mi", mi_estimator="gcmi",
                        count=50, engine="fft")


class TestEndToEndEquivalence:
    @pytest.mark.parametrize("d", [4, 5, MAX_FFT_MTS_DIMENSIONS])
    def test_fft_matches_loop_observed_mi(self, d):
        """Observed MI (stats['me']) must match between FFT and loop engines."""
        rng = np.random.RandomState(500 + d)
        n = 600
        z = rng.randn(n)
        # Feature dimensions partially coupled to z so MI is non-trivial;
        # coupling coefficients stay below 1 for any d.
        coeffs = 0.2 + 0.6 * np.arange(d) / max(d - 1, 1)
        rows = [a * z + np.sqrt(1 - a ** 2) * rng.randn(n) for a in coeffs]
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
