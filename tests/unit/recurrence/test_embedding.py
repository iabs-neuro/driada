"""Tests for driada.recurrence.embedding."""

import numpy as np
import pytest
from driada.recurrence.embedding import (
    takens_embedding, estimate_tau, estimate_embedding_dim,
)


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


class TestEstimateTau:
    """Tests for estimate_tau()."""

    def test_sinusoid_first_minimum(self):
        """Sine wave: tau should be near quarter-period."""
        period = 40  # samples per cycle
        t = np.arange(2000)
        data = np.sin(2 * np.pi * t / period)
        tau = estimate_tau(data, max_shift=50, method='first_minimum')
        assert 7 <= tau <= 13

    def test_exponential_fit(self):
        """Exponential fit should also find reasonable tau for sine."""
        period = 40
        t = np.arange(2000)
        data = np.sin(2 * np.pi * t / period)
        tau = estimate_tau(data, max_shift=50, method='exponential_fit')
        assert 3 <= tau <= 20

    def test_white_noise_reasonable(self):
        """White noise: tau should be small (MI drops immediately)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000)
        tau = estimate_tau(data, max_shift=50, method='first_minimum')
        assert 1 <= tau <= 10

    def test_returns_int(self):
        """Result must be a positive integer."""
        data = np.random.randn(500)
        tau = estimate_tau(data, max_shift=30)
        assert isinstance(tau, (int, np.integer))
        assert tau >= 1

    def test_unknown_method_raises(self):
        """Unknown method must raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            estimate_tau(np.random.randn(100), method='unknown')


class TestEstimateEmbeddingDim:
    """Tests for estimate_embedding_dim() via FNN."""

    def test_sinusoid_finds_low_dim(self):
        """Sine wave is 1D dynamics -- FNN should find m <= 3."""
        t = np.arange(2000)
        data = np.sin(2 * np.pi * t / 40)
        m = estimate_embedding_dim(data, tau=10, max_dim=8)
        assert 2 <= m <= 3

    def test_returns_int(self):
        """Result must be a positive integer >= 2."""
        data = np.random.randn(1000)
        m = estimate_embedding_dim(data, tau=5, max_dim=6)
        assert isinstance(m, (int, np.integer))
        assert m >= 2

    def test_max_dim_respected(self):
        """If FNN never drops below threshold, return max_dim."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        m = estimate_embedding_dim(data, tau=1, max_dim=5)
        assert m <= 5

    def test_known_multicomponent_signal(self):
        """Sum of 3 incommensurate sines -- FNN should find m >= 3."""
        t = np.arange(5000)
        data = np.sin(t * 0.1) + np.sin(t * 0.0314) + np.sin(t * 0.00712)
        m = estimate_embedding_dim(data, tau=15, max_dim=8)
        assert 3 <= m <= 6

    def test_return_fractions_type(self):
        """return_fractions=True returns (int, list of tuples)."""
        t = np.arange(2000)
        data = np.sin(2 * np.pi * t / 40)
        result = estimate_embedding_dim(data, tau=10, max_dim=8,
                                        return_fractions=True)
        assert isinstance(result, tuple) and len(result) == 2
        dim, fractions = result
        assert isinstance(dim, (int, np.integer))
        assert isinstance(fractions, list)
        assert len(fractions) == 7  # max_dim - 1 = 8 - 1
        for m, frac in fractions:
            assert isinstance(m, (int, np.integer))
            assert 0.0 <= frac <= 1.0

    def test_return_fractions_matches_default(self):
        """Dimension from return_fractions=True matches the default call."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(800)
        dim_default = estimate_embedding_dim(data, tau=3, max_dim=6)
        dim_frac, fractions = estimate_embedding_dim(
            data, tau=3, max_dim=6, return_fractions=True)
        assert dim_default == dim_frac

    def test_return_fractions_false_is_int(self):
        """return_fractions=False (default) returns plain int."""
        data = np.random.randn(500)
        result = estimate_embedding_dim(data, tau=5, max_dim=6,
                                        return_fractions=False)
        assert isinstance(result, (int, np.integer))


class TestGetTdmiFFTEquivalence:
    """Verify FFT-accelerated GCMI TDMI matches loop-based computation."""

    def test_fft_matches_loop_gcmi(self):
        """FFT-accelerated GCMI TDMI must match loop-based mi_gg."""
        from driada.information.info_base import get_1d_mi, TimeSeries
        from driada.information.gcmi import copnorm
        from driada.information.info_fft import compute_mi_batch_fft

        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)

        # FFT path (what get_tdmi now does internally)
        cn = copnorm(data).ravel()
        shifts = np.arange(1, 61)
        mi_fft = compute_mi_batch_fft(cn, cn, shifts, biascorrect=True)

        # Loop path (previous behavior)
        ts = TimeSeries(data, discrete=False)
        mi_loop = np.array([
            get_1d_mi(ts, ts, shift=s, estimator='gcmi') for s in range(1, 61)
        ])

        np.testing.assert_allclose(mi_fft, mi_loop, atol=1e-12)

    def test_get_tdmi_returns_list(self):
        """get_tdmi must return list of float for API compatibility."""
        from driada.information.info_base import get_tdmi

        result = get_tdmi(np.random.default_rng(42).standard_normal(500),
                          min_shift=1, max_shift=20)
        assert isinstance(result, list)
        assert len(result) == 19

    def test_get_tdmi_empty_range(self):
        """min_shift >= max_shift returns empty list."""
        from driada.information.info_base import get_tdmi

        assert get_tdmi(np.random.randn(100), min_shift=50, max_shift=50) == []
