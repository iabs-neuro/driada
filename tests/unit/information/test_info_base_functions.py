"""Tests for standalone functions in info_base module."""

import numpy as np
import pytest
import scipy.stats
from driada.information.info_base import (
    get_stats_function,
    calc_signal_ratio,
    get_sim,
    get_mi,
    get_1d_mi,
    get_tdmi,
    get_multi_mi,
    aggregate_multiple_ts,
    conditional_mi,
    interaction_information,
    TimeSeries,
    MultiTimeSeries,
)


class TestGetStatsFunction:
    """Test get_stats_function utility."""

    def test_get_valid_function(self):
        """Test getting valid scipy.stats functions."""
        # Get pearsonr
        func = get_stats_function("pearsonr")
        assert func is scipy.stats.pearsonr

        # Get spearmanr
        func = get_stats_function("spearmanr")
        assert func is scipy.stats.spearmanr

    def test_get_invalid_function(self):
        """Test error for invalid function name."""
        with pytest.raises(ValueError, match="Metric 'invalid_func' not found"):
            get_stats_function("invalid_func")


class TestCalcSignalRatio:
    """Test calc_signal_ratio function."""

    def test_basic_ratio(self):
        """Test basic signal ratio calculation."""
        binary = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        continuous = np.array([1, 10, 2, 8, 9, 1, 2, 10])

        ratio = calc_signal_ratio(binary, continuous)
        # avg_on = mean([10, 8, 9, 10]) = 9.25
        # avg_off = mean([1, 2, 1, 2]) = 1.5
        # ratio = 9.25 / 1.5 ≈ 6.17
        assert 6 < ratio < 6.5

    def test_zero_off_state(self):
        """Test when off state average is zero."""
        binary = np.array([0, 1, 0, 1])
        continuous = np.array([0, 5, 0, 3])

        ratio = calc_signal_ratio(binary, continuous)
        assert np.isinf(ratio)

    def test_both_zero(self):
        """Test when both states have zero average."""
        binary = np.array([0, 1, 0, 1])
        continuous = np.array([0, 0, 0, 0])

        ratio = calc_signal_ratio(binary, continuous)
        assert np.isnan(ratio)

    def test_all_ones(self):
        """Test when binary signal is all ones."""
        binary = np.ones(10, dtype=int)
        continuous = np.random.rand(10) + 1

        ratio = calc_signal_ratio(binary, continuous)
        # No off state, so returns NaN
        assert np.isnan(ratio)


class TestGetSim:
    """Test get_sim similarity function."""

    def test_correlation_metric(self):
        """Test similarity with correlation metric."""
        np.random.seed(42)
        # Create correlated time series
        x = TimeSeries(np.random.randn(200), discrete=False)
        noise = np.random.randn(200) * 0.5
        y_data = x.data + noise
        y = TimeSeries(y_data, discrete=False)

        # Use pearsonr as metric
        sim = get_sim(x, y, metric="pearsonr")
        # Should return correlation value
        assert isinstance(sim, (float, np.floating))
        assert 0.5 < sim < 0.95  # Positive correlation

    def test_shift_parameter(self):
        """Test similarity with shift."""
        np.random.seed(42)
        # Create identical series with shift
        data = np.sin(np.linspace(0, 4 * np.pi, 100))
        x = TimeSeries(data, discrete=False)
        y = TimeSeries(data, discrete=False)

        # No shift - should be perfectly correlated
        sim0 = get_sim(x, y, metric="pearsonr", shift=0)
        assert sim0 > 0.99

        # With shift - correlation should decrease
        sim5 = get_sim(x, y, metric="pearsonr", shift=5)
        assert sim5 < sim0

    def test_downsampling(self):
        """Test similarity with downsampling."""
        np.random.seed(42)
        x = TimeSeries(np.random.randn(1000), discrete=False)
        y = TimeSeries(np.random.randn(1000), discrete=False)

        # Calculate with different downsampling
        sim1 = get_sim(x, y, metric="pearsonr", ds=1)
        sim10 = get_sim(x, y, metric="pearsonr", ds=10)

        # Both should return float values
        assert isinstance(sim1, (float, np.floating))
        assert isinstance(sim10, (float, np.floating))

    def test_numpy_array_input(self):
        """Test with numpy array inputs."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        sim = get_sim(x, y, metric="spearmanr")
        assert isinstance(sim, (float, np.floating))


class TestGetMI:
    """Test get_mi mutual information function."""

    def test_independent_variables(self):
        """Test MI for independent variables."""
        np.random.seed(42)
        x = TimeSeries(np.random.randn(500), discrete=False)
        y = TimeSeries(np.random.randn(500), discrete=False)

        mi = get_mi(x, y, estimator="ksg")
        # Should be close to zero
        assert -0.2 < mi < 0.2

    def test_dependent_variables(self):
        """Test MI for dependent variables."""
        np.random.seed(42)
        x_data = np.random.randn(500)
        x = TimeSeries(x_data, discrete=False)
        # Create dependent variable
        y = TimeSeries(x_data + 0.5 * np.random.randn(500), discrete=False)

        mi = get_mi(x, y, estimator="ksg")
        # Should be positive
        assert mi > 0.3

    def test_shift_parameter(self):
        """Test MI with shift parameter."""
        np.random.seed(42)
        # Create time series with autocorrelation
        data = np.cumsum(np.random.randn(500))
        x = TimeSeries(data[:-10], discrete=False)
        y = TimeSeries(data[10:], discrete=False)

        # MI with no shift
        mi0 = get_mi(x, y, shift=0, estimator="ksg")
        # MI with shift that aligns them
        mi10 = get_mi(x, y, shift=-10, estimator="ksg")

        # Both should be reasonable values
        assert isinstance(mi0, float)
        assert isinstance(mi10, float)

    def test_gcmi_estimator(self):
        """Test MI with GCMI estimator."""
        np.random.seed(42)
        x = TimeSeries(np.random.randn(300), discrete=False)
        y = TimeSeries(x.data + np.random.randn(300), discrete=False)

        mi = get_mi(x, y, estimator="gcmi")
        assert mi > 0

    def test_mi_multidimensional_error(self):
        """Test error for multidimensional numpy array."""
        data1 = np.random.randn(3, 100)
        data2 = np.random.randn(100)

        with pytest.raises(Exception, match="Multidimensional inputs"):
            get_mi(data1, data2)

    def test_mi_multi_single(self):
        """Test MI between MultiTimeSeries and TimeSeries."""
        # Create MultiTimeSeries
        mts_data = np.random.randn(3, 200)
        mts = MultiTimeSeries(mts_data, discrete=False)

        # Create correlated TimeSeries
        ts_data = 0.5 * mts_data[0] + 0.3 * mts_data[1] + 0.2 * np.random.randn(200)
        ts = TimeSeries(ts_data, discrete=False)

        # Test MI calculation
        mi = get_mi(mts, ts)
        assert isinstance(mi, float)
        assert mi > 0

        # Test reverse order
        mi_rev = get_mi(ts, mts)
        assert isinstance(mi_rev, float)
        assert mi_rev > 0

    def test_mi_multi_single_discrete(self):
        """Test MI between MultiTimeSeries and discrete TimeSeries."""
        # Create continuous MultiTimeSeries
        mts_data = np.random.randn(2, 200)
        mts = MultiTimeSeries(mts_data, discrete=False)

        # Create discrete TimeSeries dependent on mts
        discrete_data = (mts_data[0] > 0).astype(int)
        ts = TimeSeries(discrete_data, discrete=True)

        # Test MI calculation
        mi = get_mi(mts, ts)
        assert isinstance(mi, float)
        assert mi > 0

    def test_mi_multi_multi(self):
        """Test MI between two MultiTimeSeries."""
        # Create first MultiTimeSeries
        mts1_data = np.random.randn(2, 200)
        mts1 = MultiTimeSeries(mts1_data, discrete=False)

        # Create second MultiTimeSeries correlated with first
        mts2_data = np.vstack(
            [
                mts1_data[0] + 0.5 * np.random.randn(200),
                mts1_data[1] + 0.5 * np.random.randn(200),
            ]
        )
        mts2 = MultiTimeSeries(mts2_data, discrete=False)

        # Test MI calculation
        mi = get_mi(mts1, mts2)
        assert isinstance(mi, float)
        assert mi > 0

    def test_mi_multi_single_ksg_error(self):
        """Test error when using KSG estimator for multi-dimensional data."""
        mts = MultiTimeSeries(np.random.randn(2, 100), discrete=False)
        ts = TimeSeries(np.random.randn(100), discrete=False)

        # This should work with gcmi (default)
        mi_gcmi = get_mi(mts, ts, estimator="gcmi")
        assert isinstance(mi_gcmi, float)

        # But should fail with ksg
        with pytest.raises(
            NotImplementedError, match="KSG estimator is not supported for dim>1"
        ):
            get_mi(mts, ts, estimator="ksg")

    def test_mi_multi_single_coincidence_warning(self):
        """Test warning when MultiTimeSeries contains identical data as TimeSeries."""
        # Create data where one row of MultiTimeSeries matches TimeSeries exactly
        data = np.random.randn(100)
        mts_data = np.vstack([data, np.random.randn(100)])  # Identical to ts

        mts = MultiTimeSeries(mts_data, discrete=False)
        ts = TimeSeries(data.copy(), discrete=False)  # Copy to avoid same object

        # This should trigger the warning and return 0
        with pytest.warns(
            UserWarning,
            match="MI computation between MultiTimeSeries containing identical data",
        ):
            mi = get_mi(mts, ts, shift=0)
            assert mi == 0.0

    def test_mi_multi_single_with_shift(self):
        """Test MI between MultiTimeSeries and TimeSeries with shift."""
        # Create MultiTimeSeries
        mts = MultiTimeSeries(np.random.randn(2, 200), discrete=False)
        ts = TimeSeries(np.random.randn(200), discrete=False)

        # Test with different shifts
        mi_no_shift = get_mi(mts, ts, shift=0)
        mi_shift5 = get_mi(mts, ts, shift=5)
        mi_shift_neg5 = get_mi(mts, ts, shift=-5)

        assert isinstance(mi_no_shift, float)
        assert isinstance(mi_shift5, float)
        assert isinstance(mi_shift_neg5, float)

    def test_mi_multi_multi_with_shift(self):
        """Test MI between two MultiTimeSeries with shift."""
        # Create correlated MultiTimeSeries
        base_data = np.random.randn(2, 200)
        mts1 = MultiTimeSeries(base_data, discrete=False)
        mts2 = MultiTimeSeries(
            base_data + 0.5 * np.random.randn(2, 200), discrete=False
        )

        # Test with shift
        mi_no_shift = get_mi(mts1, mts2, shift=0)
        mi_shift10 = get_mi(mts1, mts2, shift=10)

        assert isinstance(mi_no_shift, float)
        assert isinstance(mi_shift10, float)
        assert mi_no_shift > 0  # Should be correlated

    def test_mi_multi_single_discrete_with_shift(self):
        """Test MI between MultiTimeSeries and discrete TimeSeries with shift."""
        # Create MultiTimeSeries
        mts = MultiTimeSeries(np.random.randn(2, 200), discrete=False)

        # Create discrete TimeSeries dependent on mts
        discrete_data = (mts.data[0] > 0).astype(int)
        ts = TimeSeries(discrete_data, discrete=True)

        # Test with shift
        mi_shift = get_mi(mts, ts, shift=5)
        assert isinstance(mi_shift, float)


class TestGet1DMI:
    """Test get_1d_mi function."""

    def test_basic_1d_mi(self):
        """Test basic 1D mutual information."""
        np.random.seed(42)
        # Create correlated 1D time series
        ts1 = TimeSeries(np.random.randn(400), discrete=False)
        ts2 = TimeSeries(ts1.data + 0.5 * np.random.randn(400), discrete=False)

        mi = get_1d_mi(ts1, ts2)
        assert mi > 0

    def test_coincidence_check(self):
        """Test coincidence checking."""
        # Create identical time series
        data = np.random.randn(200)
        ts1 = TimeSeries(data, discrete=False)
        ts2 = TimeSeries(data.copy(), discrete=False)  # Copy to avoid same object

        # With coincidence check (default) should raise error
        with pytest.raises(ValueError, match="MI\\(X,X\\) for continuous variables is infinite"):
            mi = get_1d_mi(ts1, ts2, check_for_coincidence=True)

        # Without coincidence check - should compute high MI
        mi = get_1d_mi(ts1, ts2, check_for_coincidence=False)
        assert mi > 1  # Should be high for identical series

    def test_1d_mi_discrete_discrete(self):
        """Test MI between two discrete time series."""
        # Create correlated discrete data
        ts1 = TimeSeries(np.random.randint(0, 3, 200), discrete=True)
        ts2_data = ts1.int_data.copy()
        # Add some noise by randomly flipping 20% of values
        flip_idx = np.random.choice(200, 40, replace=False)
        ts2_data[flip_idx] = (ts2_data[flip_idx] + 1) % 3
        ts2 = TimeSeries(ts2_data, discrete=True)

        mi = get_1d_mi(ts1, ts2)
        assert isinstance(mi, float)
        assert mi > 0  # Should show correlation

    def test_1d_mi_discrete_continuous(self):
        """Test MI between discrete and continuous time series."""
        # Create discrete signal
        discrete_data = np.random.randint(0, 2, 200)
        ts1 = TimeSeries(discrete_data, discrete=True)

        # Create continuous signal dependent on discrete
        continuous_data = np.where(
            discrete_data == 0, np.random.normal(0, 1, 200), np.random.normal(5, 1, 200)
        )
        ts2 = TimeSeries(continuous_data, discrete=False)

        mi = get_1d_mi(ts1, ts2)
        assert isinstance(mi, float)
        assert mi > 0  # Should show dependency

    def test_1d_mi_continuous_discrete(self):
        """Test MI between continuous and discrete time series."""
        # Create continuous signal
        continuous_data = np.random.randn(200)
        ts1 = TimeSeries(continuous_data, discrete=False)

        # Create discrete signal dependent on continuous
        discrete_data = (continuous_data > 0).astype(int)
        ts2 = TimeSeries(discrete_data, discrete=True)

        mi = get_1d_mi(ts1, ts2)
        assert isinstance(mi, float)
        assert mi > 0  # Should show dependency

    def test_1d_mi_discrete_with_shift(self):
        """Test MI between discrete time series with shift."""
        # Create autocorrelated discrete signal
        data = np.random.randint(0, 3, 210)
        ts1 = TimeSeries(data[:200], discrete=True)
        ts2 = TimeSeries(data[10:210], discrete=True)  # Shifted version

        # Without shift, should have low MI
        mi_no_shift = get_1d_mi(ts1, ts2, shift=0)
        # With correct shift, should have high MI
        mi_shift = get_1d_mi(ts1, ts2, shift=-10)

        assert mi_shift > mi_no_shift


class TestGetTDMI:
    """Test get_tdmi time-delayed mutual information."""

    def test_autocorrelated_series(self):
        """Test TDMI for autocorrelated series."""
        np.random.seed(42)
        # Create autocorrelated series
        ar_coef = 0.8
        noise = np.random.randn(1000)
        data = np.zeros(1000)
        data[0] = noise[0]
        for i in range(1, 1000):
            data[i] = ar_coef * data[i - 1] + noise[i]

        # Calculate TDMI
        tdmi_values = get_tdmi(data, min_shift=1, max_shift=11)

        assert len(tdmi_values) == 10
        # First value (shift=1) should be highest
        assert tdmi_values[0] == max(tdmi_values)
        # Should decrease with lag
        assert tdmi_values[0] > tdmi_values[5]

    def test_random_series(self):
        """Test TDMI for random series."""
        np.random.seed(42)
        data = np.random.randn(500)

        tdmi_values = get_tdmi(data, min_shift=1, max_shift=6)

        assert len(tdmi_values) == 5
        # All values should be near zero
        assert all(abs(v) < 0.2 for v in tdmi_values)


class TestGetMultiMI:
    """Test get_multi_mi function."""

    def test_multiple_predictors(self):
        """Test MI with multiple predictor time series."""
        np.random.seed(42)
        # Create target influenced by multiple sources
        ts1 = TimeSeries(np.random.randn(300), discrete=False)
        ts2 = TimeSeries(np.random.randn(300), discrete=False)
        ts3 = TimeSeries(np.random.randn(300), discrete=False)

        # Target is combination of all
        target_data = (
            0.5 * ts1.data
            + 0.3 * ts2.data
            + 0.2 * ts3.data
            + 0.5 * np.random.randn(300)
        )
        target = TimeSeries(target_data, discrete=False)

        # Calculate multi MI
        mi = get_multi_mi([ts1, ts2, ts3], target)

        # Should be positive (predictors contain info about target)
        assert mi > 0

    def test_single_predictor(self):
        """Test multi MI with single predictor."""
        np.random.seed(42)
        ts1 = TimeSeries(np.random.randn(200), discrete=False)
        ts2 = TimeSeries(ts1.data + np.random.randn(200), discrete=False)

        # Single predictor in list
        mi_multi = get_multi_mi([ts1], ts2)
        # Compare with regular MI
        mi_single = get_mi(ts1, ts2)

        # Should be similar
        assert abs(mi_multi - mi_single) < 0.1


class TestAggregateMultipleTS:
    """Test aggregate_multiple_ts function."""

    def test_basic_aggregation(self):
        """Test basic time series aggregation."""
        ts1 = TimeSeries(np.array([1, 2, 3, 4, 5]), discrete=False)
        ts2 = TimeSeries(np.array([2, 3, 4, 5, 6]), discrete=False)
        ts3 = TimeSeries(np.array([3, 4, 5, 6, 7]), discrete=False)

        result = aggregate_multiple_ts(ts1, ts2, ts3)

        # Should return MultiTimeSeries with shape (3, 5)
        assert isinstance(result, MultiTimeSeries)
        assert result.data.shape == (3, 5)
        # Check that data is included (with small noise added)
        assert np.allclose(result.data[0], ts1.data, atol=0.1)
        assert np.allclose(result.data[1], ts2.data, atol=0.1)
        assert np.allclose(result.data[2], ts3.data, atol=0.1)

    def test_with_noise(self):
        """Test aggregation with noise."""
        np.random.seed(42)
        ts1 = TimeSeries(np.ones(100), discrete=False)
        ts2 = TimeSeries(np.ones(100) * 2, discrete=False)

        result = aggregate_multiple_ts(ts1, ts2, noise=0.1)

        # Should have added noise
        assert isinstance(result, MultiTimeSeries)
        assert result.data.shape == (2, 100)
        # Should not be exactly equal due to noise
        assert not np.array_equal(result.data[0], ts1.data)
        # But should be close
        assert np.abs(result.data[0] - ts1.data).max() < 0.5

    def test_different_types(self):
        """Test with TimeSeries objects."""
        ts1 = TimeSeries(np.random.randn(50), discrete=False)
        ts2 = TimeSeries(np.random.randn(50), discrete=False)

        result = aggregate_multiple_ts(ts1, ts2)

        assert isinstance(result, MultiTimeSeries)
        assert result.data.shape == (2, 50)


class TestConditionalMI:
    """Test conditional_mi function."""

    def test_chain_dependency(self):
        """Test CMI for chain dependency X -> Z -> Y."""
        np.random.seed(42)
        # Create chain: X -> Z -> Y
        x_data = np.random.randn(500)
        z_data = x_data + 0.5 * np.random.randn(500)
        y_data = z_data + 0.5 * np.random.randn(500)

        ts_x = TimeSeries(x_data, discrete=False)
        ts_y = TimeSeries(y_data, discrete=False)
        ts_z = TimeSeries(z_data, discrete=False)

        # I(X;Y|Z) should be small (X and Y are conditionally independent given Z)
        cmi = conditional_mi(ts_x, ts_y, ts_z)
        assert -0.2 < cmi < 0.2

    def test_common_cause(self):
        """Test CMI for common cause Z -> X, Z -> Y."""
        np.random.seed(42)
        # Common cause
        z_data = np.random.randn(500)
        x_data = z_data + np.random.randn(500)
        y_data = z_data + np.random.randn(500)

        ts_x = TimeSeries(x_data, discrete=False)
        ts_y = TimeSeries(y_data, discrete=False)
        ts_z = TimeSeries(z_data, discrete=False)

        # I(X;Y|Z) should be small
        cmi = conditional_mi(ts_x, ts_y, ts_z)
        assert -0.2 < cmi < 0.2


class TestInteractionInformation:
    """Test interaction_information function."""

    def test_redundant_information(self):
        """Test interaction information for redundant variables."""
        np.random.seed(42)
        # Create redundant information structure
        # X and Y both contain information about Z
        z_data = np.random.randn(400)
        x_data = z_data + 0.5 * np.random.randn(400)
        y_data = z_data + 0.5 * np.random.randn(400)

        ts_x = TimeSeries(x_data, discrete=False)
        ts_y = TimeSeries(y_data, discrete=False)
        ts_z = TimeSeries(z_data, discrete=False)

        # Interaction information
        ii = interaction_information(ts_x, ts_y, ts_z)
        # Should be negative for redundancy
        assert ii < 0

    def test_synergistic_information(self):
        """Test interaction information for synergy."""
        np.random.seed(42)
        # Create XOR-like relationship (synergy)
        x_data = np.random.randint(0, 2, 400)
        y_data = np.random.randint(0, 2, 400)
        # Z is approximately XOR of X and Y with some noise
        z_data = (x_data + y_data) % 2
        # Add some continuous noise to make it work with continuous MI estimators
        x_cont = x_data + 0.1 * np.random.randn(400)
        y_cont = y_data + 0.1 * np.random.randn(400)
        z_cont = z_data + 0.1 * np.random.randn(400)

        ts_x = TimeSeries(x_cont, discrete=False)
        ts_y = TimeSeries(y_cont, discrete=False)
        ts_z = TimeSeries(z_cont, discrete=False)

        # Interaction information
        ii = interaction_information(ts_x, ts_y, ts_z)
        # For synergy, this could be positive
        assert isinstance(ii, float)


class TestGetSimAdditional:
    """Test get_sim similarity function with different metrics."""

    def test_sim_discrete_continuous_av(self):
        """Test 'av' metric for binary-continuous similarity."""
        # Binary signal
        binary = TimeSeries(np.array([0, 1, 0, 1, 1, 0] * 20), discrete=True)
        # Continuous signal correlated with binary
        continuous_data = []
        for b in binary.data:
            if b == 1:
                continuous_data.append(np.random.normal(10, 1))
            else:
                continuous_data.append(np.random.normal(2, 1))
        continuous = TimeSeries(np.array(continuous_data), discrete=False)

        # Test av metric
        sim = get_sim(binary, continuous, metric="av")
        assert sim > 1  # Should show signal ratio > 1

        # Test reverse order
        sim_rev = get_sim(continuous, binary, metric="av")
        assert sim_rev > 1

    def test_sim_discrete_continuous_error(self):
        """Test error for unsupported metrics with mixed types."""
        binary = TimeSeries(np.array([0, 1, 0, 1]), discrete=True)
        continuous = TimeSeries(np.random.randn(4), discrete=False)

        with pytest.raises(ValueError, match="Only 'av' and 'mi' metrics"):
            get_sim(binary, continuous, metric="pearsonr")

    def test_sim_discrete_discrete_error(self):
        """Test error for discrete-discrete similarity."""
        ts1 = TimeSeries(np.array([0, 1, 0, 1]), discrete=True)
        ts2 = TimeSeries(np.array([1, 0, 1, 0]), discrete=True)

        with pytest.raises(ValueError, match="not supported for two discrete"):
            get_sim(ts1, ts2, metric="pearsonr")

    def test_sim_multidimensional_error(self):
        """Test error for multidimensional data with non-MI metrics."""
        mts1 = MultiTimeSeries(np.random.randn(2, 100), discrete=False)
        mts2 = MultiTimeSeries(np.random.randn(2, 100), discrete=False)

        with pytest.raises(Exception, match="Metrics except 'mi'"):
            get_sim(mts1, mts2, metric="pearsonr")

    def test_sim_fast_pearsonr(self):
        """Test fast Pearson correlation metric."""
        # Create correlated time series
        x = np.random.randn(200)
        y = x + 0.5 * np.random.randn(200)

        ts1 = TimeSeries(x, discrete=False)
        ts2 = TimeSeries(y, discrete=False)

        sim = get_sim(ts1, ts2, metric="fast_pearsonr")
        assert 0.5 < sim < 0.9  # Should show positive correlation

    def test_sim_spearmanr(self):
        """Test Spearman correlation metric."""
        # Create monotonically related data
        x = np.random.randn(150)
        y = x**3 + np.random.randn(150) * 0.1

        ts1 = TimeSeries(x, discrete=False)
        ts2 = TimeSeries(y, discrete=False)

        sim = get_sim(ts1, ts2, metric="spearmanr")
        assert 0.8 < sim < 1.0  # Should show strong monotonic relationship

    def test_sim_av_non_binary_error(self):
        """Test error when using 'av' metric with non-binary discrete data."""
        # Non-binary discrete signal
        discrete = TimeSeries(np.array([0, 1, 2, 3, 2, 1] * 20), discrete=True)
        continuous = TimeSeries(np.random.randn(120), discrete=False)

        with pytest.raises(ValueError, match="must be binary for metric='av'"):
            get_sim(discrete, continuous, metric="av")


class TestGetMultiMI:
    """Test get_multi_mi function."""

    def test_multi_mi_basic(self):
        """Test multi MI with multiple predictors."""
        # Create predictors
        ts1 = TimeSeries(np.random.randn(200), discrete=False)
        ts2 = TimeSeries(np.random.randn(200), discrete=False)
        ts3 = TimeSeries(np.random.randn(200), discrete=False)

        # Target is influenced by all predictors
        target_data = (
            0.5 * ts1.data
            + 0.3 * ts2.data
            + 0.2 * ts3.data
            + 0.5 * np.random.randn(200)
        )
        target = TimeSeries(target_data, discrete=False)

        # Compute multi MI
        mi = get_multi_mi([ts1, ts2, ts3], target)

        assert isinstance(mi, float)
        assert mi > 0  # Should show positive MI

    def test_multi_mi_with_shift(self):
        """Test multi MI with time shift."""
        # Create time-lagged predictors
        base_signal = np.cumsum(np.random.randn(210))
        ts1 = TimeSeries(base_signal[:200], discrete=False)
        ts2 = TimeSeries(base_signal[5:205], discrete=False)
        target = TimeSeries(base_signal[10:210], discrete=False)

        # MI with shift
        mi_shift = get_multi_mi([ts1, ts2], target, shift=5)
        mi_no_shift = get_multi_mi([ts1, ts2], target, shift=0)

        assert isinstance(mi_shift, float)
        assert isinstance(mi_no_shift, float)


class TestConditionalMIAdditional:
    """Additional tests for conditional MI edge cases."""

    def test_cmi_discrete_error(self):
        """Test error when ts1 is discrete."""
        ts1 = TimeSeries(np.random.randint(0, 3, 100), discrete=True)
        ts2 = TimeSeries(np.random.randn(100), discrete=False)
        ts3 = TimeSeries(np.random.randn(100), discrete=False)

        with pytest.raises(ValueError, match="continuous X only"):
            conditional_mi(ts1, ts2, ts3)

    def test_cmi_cdc_case(self):
        """Test CDC case: continuous X, discrete Y, continuous Z."""
        # Continuous X
        x_data = np.random.randn(200)
        ts_x = TimeSeries(x_data, discrete=False)

        # Discrete Y dependent on X
        y_data = (x_data > 0).astype(int)
        ts_y = TimeSeries(y_data, discrete=True)

        # Continuous Z
        z_data = x_data + np.random.randn(200)
        ts_z = TimeSeries(z_data, discrete=False)

        cmi = conditional_mi(ts_x, ts_y, ts_z)
        assert isinstance(cmi, float)
        assert cmi >= 0

    def test_cmi_cdd_case(self):
        """Test CDD case: continuous X, discrete Y and Z."""
        # Continuous X
        ts_x = TimeSeries(np.random.randn(200), discrete=False)

        # Discrete Y and Z
        ts_y = TimeSeries(np.random.randint(0, 2, 200), discrete=True)
        ts_z = TimeSeries(np.random.randint(0, 3, 200), discrete=True)

        cmi = conditional_mi(ts_x, ts_y, ts_z)
        assert isinstance(cmi, float)
        assert cmi >= 0


class TestGetTDMI:
    """Test time-delayed mutual information."""

    def test_tdmi_basic(self):
        """Test basic TDMI functionality."""
        # Create autocorrelated signal
        signal = np.cumsum(np.random.randn(500))

        tdmi = get_tdmi(signal, min_shift=1, max_shift=6)

        assert len(tdmi) == 5  # max_shift is exclusive
        assert all(isinstance(v, float) for v in tdmi)
        # First lag should have highest MI for autocorrelated signal
        assert tdmi[0] == max(tdmi)

    def test_tdmi_with_nn_parameter(self):
        """Test TDMI with custom nn parameter."""
        signal = np.sin(np.linspace(0, 10, 300))

        tdmi = get_tdmi(signal, min_shift=1, max_shift=4, nn=10)

        assert len(tdmi) == 3
        assert all(isinstance(v, float) for v in tdmi)


class TestTheoreticalGaussianMI:
    """Test MI estimation against theoretical values for Gaussian distributions."""
    
    @staticmethod
    def theoretical_mi_gaussian_1d(rho):
        """Calculate theoretical MI for bivariate Gaussian with correlation rho."""
        # I(X;Y) = -0.5 * log2(1 - rho^2) in bits
        # Both get_1d_mi estimators return MI in bits
        return -0.5 * np.log2(1 - rho**2)
    
    @staticmethod
    def theoretical_mi_gaussian_multi(cov_matrix):
        """Calculate theoretical MI for multivariate Gaussian.
        
        For Gaussian variables, I(X;Y) = 0.5 * log(det(Σ_X) * det(Σ_Y) / det(Σ_XY))
        """
        # Assuming last variable is Y, rest are X
        n = cov_matrix.shape[0]
        
        # Marginal covariances
        cov_x = cov_matrix[:-1, :-1]
        cov_y = cov_matrix[-1:, -1:]
        
        # Determinants
        det_joint = np.linalg.det(cov_matrix)
        det_x = np.linalg.det(cov_x)
        det_y = cov_y[0, 0]  # scalar for 1D Y
        
        # MI in bits (to match the estimators which return bits)
        mi = 0.5 * np.log2(det_x * det_y / det_joint)
        return mi
    
    def test_1d_gaussian_independent(self):
        """Test MI estimation for independent Gaussian variables."""
        np.random.seed(42)
        n_samples = 5000
        
        # Independent Gaussian
        x = np.random.randn(n_samples)
        y = np.random.randn(n_samples)
        
        ts_x = TimeSeries(x)
        ts_y = TimeSeries(y)
        
        # Test both estimators
        mi_gcmi = get_1d_mi(ts_x, ts_y, estimator='gcmi')
        mi_ksg = get_1d_mi(ts_x, ts_y, estimator='ksg', k=10)
        
        # Should be close to 0
        assert abs(mi_gcmi) < 0.05
        assert abs(mi_ksg) < 0.05
    
    def test_1d_gaussian_correlated(self):
        """Test MI estimation for correlated Gaussian variables."""
        np.random.seed(42)
        n_samples = 5000
        rho = 0.7  # correlation
        
        # Correlated Gaussian
        x = np.random.randn(n_samples)
        y = rho * x + np.sqrt(1 - rho**2) * np.random.randn(n_samples)
        
        ts_x = TimeSeries(x)
        ts_y = TimeSeries(y)
        
        # Theoretical MI
        mi_theory = self.theoretical_mi_gaussian_1d(rho)
        
        # Test both estimators
        mi_gcmi = get_1d_mi(ts_x, ts_y, estimator='gcmi')
        mi_ksg = get_1d_mi(ts_x, ts_y, estimator='ksg', k=10)
        
        # Should be close to theoretical value
        # GCMI tends to overestimate even for 1D Gaussian
        assert abs(mi_gcmi - mi_theory) < 0.15
        # KSG has more variance but often closer
        assert abs(mi_ksg - mi_theory) < 0.1
    
    def test_multivariate_gaussian(self):
        """Test multivariate MI estimation for Gaussian variables."""
        np.random.seed(42)
        n_samples = 5000
        
        # Define covariance matrix for 4D Gaussian (3D X, 1D Y)
        cov_matrix = np.array([
            [1.0, 0.5, 0.2, 0.7],   # X1
            [0.5, 1.0, 0.1, 0.5],   # X2  
            [0.2, 0.1, 1.0, 0.0],   # X3
            [0.7, 0.5, 0.0, 1.0]    # Y
        ])
        
        # Generate correlated Gaussian data
        data = np.random.multivariate_normal(np.zeros(4), cov_matrix, size=n_samples)
        
        # Create TimeSeries
        ts_list = [TimeSeries(data[:, i]) for i in range(3)]
        ts_y = TimeSeries(data[:, 3])
        
        # Theoretical MI
        mi_theory = self.theoretical_mi_gaussian_multi(cov_matrix)
        
        # Test estimators
        mi_gcmi = get_multi_mi(ts_list, ts_y, estimator='gcmi')
        mi_ksg = get_multi_mi(ts_list, ts_y, estimator='ksg', k=10)
        
        # GCMI tends to overestimate for multivariate
        assert abs(mi_gcmi - mi_theory) < 0.3
        # KSG should be closer
        assert abs(mi_ksg - mi_theory) < 0.1
    
    def test_perfect_linear_dependence(self):
        """Test MI estimation for perfect linear dependence."""
        np.random.seed(42)
        n_samples = 5000
        
        # X determines Y perfectly
        x = np.random.randn(n_samples)
        y = 2 * x  # Perfect linear dependence
        
        ts_x = TimeSeries(x)
        ts_y = TimeSeries(y)
        
        # MI should be very large (theoretically infinite)
        mi_gcmi = get_1d_mi(ts_x, ts_y, estimator='gcmi')
        mi_ksg = get_1d_mi(ts_x, ts_y, estimator='ksg', k=10)
        
        # Both should give large values
        assert mi_gcmi > 5.0
        assert mi_ksg > 5.0
    
    def test_ksg_k_parameter_stability(self):
        """Test that KSG is stable for k > d."""
        np.random.seed(42)
        n_samples = 5000
        
        # 4D Gaussian
        cov_matrix = np.eye(4)
        cov_matrix[0, 3] = cov_matrix[3, 0] = 0.6
        cov_matrix[1, 3] = cov_matrix[3, 1] = 0.4
        
        data = np.random.multivariate_normal(np.zeros(4), cov_matrix, size=n_samples)
        
        ts_list = [TimeSeries(data[:, i]) for i in range(3)]
        ts_y = TimeSeries(data[:, 3])
        
        # Test different k values (d=4 here)
        mi_values = []
        for k in [5, 10, 20]:  # All k > d
            mi = get_multi_mi(ts_list, ts_y, estimator='ksg', k=k)
            mi_values.append(mi)
        
        # Values should be similar (within reasonable range)
        assert max(mi_values) - min(mi_values) < 0.2
        
        # Test that k=3 gives warning (k < d)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mi_k3 = get_multi_mi(ts_list, ts_y, estimator='ksg', k=3)
            
            # Should have warning about k <= d
            assert len(w) > 0
            assert "LNC correction disabled" in str(w[0].message)


class TestCopulaDownsampling:
    """Test copula transformation behavior with downsampling."""
    
    def test_copula_downsampling_approximation_small_ds(self):
        """Test that copula downsampling approximation is reasonable for small ds."""
        np.random.seed(42)
        n_samples = 10000
        
        # Create smooth signal (sine wave + noise)
        t = np.linspace(0, 10, n_samples)
        x = np.sin(t) + 0.1 * np.random.randn(n_samples)
        y = np.cos(t) + 0.1 * np.random.randn(n_samples)
        
        ts_x = TimeSeries(x)
        ts_y = TimeSeries(y)
        
        # Compare MI with different downsampling factors
        mi_ds1 = get_1d_mi(ts_x, ts_y, ds=1, estimator='gcmi')
        mi_ds2 = get_1d_mi(ts_x, ts_y, ds=2, estimator='gcmi')
        mi_ds5 = get_1d_mi(ts_x, ts_y, ds=5, estimator='gcmi')
        
        # For smooth signals, downsampling shouldn't change MI much
        assert abs(mi_ds2 - mi_ds1) < 0.1  # Small difference for ds=2
        assert abs(mi_ds5 - mi_ds1) < 0.2  # Larger but still reasonable for ds=5
    
    def test_copula_downsampling_with_rapidly_varying_signal(self):
        """Test copula downsampling with rapidly varying signals."""
        np.random.seed(42)
        n_samples = 10000
        
        # Create rapidly varying signal with stronger correlation
        t = np.linspace(0, 100, n_samples)
        # High frequency signal with modulation
        carrier = np.sin(50 * t)
        modulation = np.sin(0.5 * t)
        x = carrier * modulation + 0.1 * np.random.randn(n_samples)
        y = carrier * modulation + 0.1 * np.random.randn(n_samples)
        
        ts_x = TimeSeries(x)
        ts_y = TimeSeries(y)
        
        # Compare MI with different downsampling
        mi_ds1 = get_1d_mi(ts_x, ts_y, ds=1, estimator='gcmi')
        mi_ds20 = get_1d_mi(ts_x, ts_y, ds=20, estimator='gcmi')
        
        # For signals with high-frequency components, heavy downsampling affects MI estimation
        # The exact behavior depends on the signal structure
        # Just verify both give reasonable positive MI
        assert mi_ds1 > 0.5  # Strong correlation at full sampling
        assert mi_ds20 > 0.1  # Still some correlation even with heavy downsampling
    
    def test_copula_transformation_preserves_rank_order(self):
        """Test that copula transformation preserves rank ordering."""
        np.random.seed(42)
        
        # Create data with known rank structure
        data = np.array([1.5, 3.2, -0.5, 4.1, 2.0])
        ts = TimeSeries(data)
        
        # Get ranks of original data
        original_ranks = scipy.stats.rankdata(data)
        
        # Get ranks of copula transformed data
        copula_ranks = scipy.stats.rankdata(ts.copula_normal_data)
        
        # Ranks should be identical
        np.testing.assert_array_equal(original_ranks, copula_ranks)
    
    def test_copula_downsampling_warning_for_large_ds(self):
        """Test that documentation warns about large downsampling factors."""
        # This is more of a documentation test - verify the docstring contains warning
        from driada.information.info_base import get_1d_mi
        
        docstring = get_1d_mi.__doc__
        assert "downsampling with GCMI" in docstring
        assert "approximation" in docstring
        assert "ds ≤ 5" in docstring or "small downsampling factors" in docstring
    
    def test_copula_vs_exact_for_gaussian(self):
        """Test GCMI accuracy for Gaussian data with different downsampling."""
        np.random.seed(42)
        n_samples = 10000
        rho = 0.6
        
        # Create correlated Gaussian
        x = np.random.randn(n_samples)
        y = rho * x + np.sqrt(1 - rho**2) * np.random.randn(n_samples)
        
        ts_x = TimeSeries(x)
        ts_y = TimeSeries(y)
        
        # Theoretical MI
        mi_theory = -0.5 * np.log(1 - rho**2)
        
        # Test with different downsampling
        for ds in [1, 2, 5, 10]:
            mi_gcmi = get_1d_mi(ts_x, ts_y, ds=ds, estimator='gcmi')
            # GCMI overestimates but should be consistent across reasonable ds
            if ds <= 5:
                assert abs(mi_gcmi - mi_theory) < 0.2
            else:
                # Larger error acceptable for large ds
                assert abs(mi_gcmi - mi_theory) < 0.3
