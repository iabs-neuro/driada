"""Tests for TimeSeries class in info_base module."""

import numpy as np
import pytest
import warnings
from driada.information.info_base import TimeSeries


class TestTimeSeriesTypeDetection:
    """Test automatic type detection for TimeSeries."""

    def test_define_ts_type_discrete(self):
        """Test detection of discrete time series."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            # Binary data
            ts_binary = np.array([0, 1, 0, 1, 1, 0, 0, 1] * 20)
            assert TimeSeries.define_ts_type(ts_binary) == True

            # Small number of unique values with repetition
            ts_discrete = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3] * 20)
            assert TimeSeries.define_ts_type(ts_discrete) == True

            # Integer categorical data with heavy repetition
            np.random.seed(42)
            ts_categorical = np.random.choice([0, 1, 2], size=200, p=[0.5, 0.3, 0.2])
            assert TimeSeries.define_ts_type(ts_categorical) == True

    def test_define_ts_type_continuous(self):
        """Test detection of continuous time series."""
        # Suppress deprecation warning for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            # Gaussian data
            ts_gaussian = np.random.randn(200)
            assert TimeSeries.define_ts_type(ts_gaussian) == False

            # Uniform continuous
            ts_uniform = np.random.uniform(0, 1, 200)
            assert TimeSeries.define_ts_type(ts_uniform) == False

            # Use truly continuous data (not linspace which creates timeline)
            ts_continuous = np.random.exponential(2, 200)
            assert TimeSeries.define_ts_type(ts_continuous) == False

    def test_define_ts_type_short_series_warning(self):
        """Test warning for short time series."""
        ts_short = np.array([1, 2, 3, 4, 5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TimeSeries.define_ts_type(ts_short)
            # May get multiple warnings (deprecation + short series)
            assert len(w) >= 1
            # Check that at least one warning is about short series
            warning_messages = [str(warn.message) for warn in w]
            assert any(
                "too short" in msg or "samples available" in msg
                for msg in warning_messages
            )

    def test_define_ts_type_ambiguous(self):
        """Test behavior for ambiguous time series."""
        # The new system is more sophisticated, so data that was previously ambiguous
        # might now be properly classified (e.g., timeline data)
        np.random.seed(42)

        # This concatenates timeline data with repeated values - will be detected as discrete/timeline
        ts_timeline_like = np.concatenate(
            [
                np.repeat(np.arange(50), 2),  # 50 values repeated twice
                np.arange(100),  # 100 unique values in sequence
            ]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            # This should return True (discrete) because it's timeline-like data
            result = TimeSeries.define_ts_type(ts_timeline_like)
            assert result == True  # Timeline data is discrete

        # Test with truly ambiguous data (mix of continuous and discrete characteristics)
        ts_truly_ambiguous = np.concatenate(
            [
                np.random.normal(0, 1, 100),  # Continuous part
                np.repeat([1, 2, 3, 4, 5], 20),  # Discrete part
            ]
        )
        np.random.shuffle(ts_truly_ambiguous)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            # Should still return a boolean result (no error)
            result = TimeSeries.define_ts_type(ts_truly_ambiguous)
            assert isinstance(result, bool)


class TestTimeSeriesInitialization:
    """Test TimeSeries class initialization."""

    def test_init_continuous_auto(self):
        """Test initialization with continuous data (auto-detected)."""
        data = np.random.randn(200)
        ts = TimeSeries(data)

        assert ts.discrete == False
        assert ts.data.shape == (200,)
        assert ts.scdata.shape == (200,)
        assert 0 <= ts.scdata.min() <= ts.scdata.max() <= 1
        assert ts.copula_normal_data is not None
        assert ts.shuffle_mask.all()

    def test_init_discrete_auto(self):
        """Test initialization with discrete data (auto-detected)."""
        # Create discrete data with heavy repetition to ensure detection
        np.random.seed(42)
        data = np.random.choice([0, 1, 2, 3], size=200, p=[0.4, 0.3, 0.2, 0.1])
        ts = TimeSeries(data)

        assert ts.discrete == True
        assert hasattr(ts, "int_data")
        assert ts.int_data.dtype == np.int64
        assert ts.is_binary == False

    def test_init_binary_data(self):
        """Test initialization with binary data."""
        data = np.array([0, 1, 1, 0, 1, 0] * 30)
        ts = TimeSeries(data, discrete=True)

        assert ts.discrete == True
        assert ts.is_binary == True
        assert hasattr(ts, "bool_data")
        assert ts.bool_data.dtype == bool

    def test_init_explicit_type(self):
        """Test initialization with explicit type specification."""
        # Force continuous interpretation of integer-like data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 40)
        ts = TimeSeries(data, discrete=False)

        assert ts.discrete == False
        assert ts.copula_normal_data is not None

    def test_init_with_shuffle_mask(self):
        """Test initialization with custom shuffle mask."""
        data = np.random.randn(100)
        mask = np.array([True, False] * 50)
        ts = TimeSeries(data, shuffle_mask=mask)

        assert ts.shuffle_mask.shape == (100,)
        assert ts.shuffle_mask.sum() == 50
        assert ts.shuffle_mask.dtype == bool

    def test_init_scaling(self):
        """Test data scaling initialization."""
        data = np.array([10, 20, 30, 40, 50])
        ts = TimeSeries(data, discrete=False)

        # Check scaled data is in [0, 1]
        assert np.abs(ts.scdata.min() - 0) < 1e-10
        assert np.abs(ts.scdata.max() - 1) < 1e-10
        # Check scale factor is stored
        assert ts.data_scale > 0

    def test_init_list_input(self):
        """Test initialization with list input."""
        # Create continuous data from list
        np.random.seed(42)
        data = list(np.random.randn(100))
        ts = TimeSeries(data)

        assert isinstance(ts.data, np.ndarray)
        assert ts.data.shape == (100,)

    def test_init_empty_dicts(self):
        """Test that empty dictionaries are initialized."""
        data = np.random.randn(100)
        ts = TimeSeries(data)

        assert isinstance(ts.entropy, dict)
        assert len(ts.entropy) == 0
        assert ts.kdtree is None
        assert isinstance(ts.kdtree_query, dict)
        assert len(ts.kdtree_query) == 0


class TestTimeSeriesKDTree:
    """Test KDTree functionality in TimeSeries."""

    def test_get_kdtree_first_call(self):
        """Test KDTree creation on first call."""
        data = np.random.randn(200)
        ts = TimeSeries(data)

        # Initially None
        assert ts.kdtree is None

        # Get tree
        tree = ts.get_kdtree()
        assert tree is not None
        assert ts.kdtree is not None

        # Should be cached
        tree2 = ts.get_kdtree()
        assert tree2 is tree

    def test_compute_kdtree(self):
        """Test _compute_kdtree method."""
        data = np.random.randn(100)
        ts = TimeSeries(data)

        tree = ts._compute_kdtree()
        assert tree is not None

        # Test tree functionality
        distances, indices = tree.query(data.reshape(-1, 1)[:5], k=3)
        assert distances.shape == (5, 3)
        assert indices.shape == (5, 3)

    def test_get_kdtree_query(self):
        """Test KDTree query caching."""
        data = np.random.randn(150)  # TimeSeries expects 1D data
        ts = TimeSeries(data)

        # Initially empty dict
        assert ts.kdtree_query == {}

        # Get query
        query = ts.get_kdtree_query(k=5)
        assert query is not None
        assert 5 in ts.kdtree_query

        # Should be cached
        query2 = ts.get_kdtree_query(k=5)
        assert query2 is query

    def test_compute_kdtree_query(self):
        """Test _compute_kdtree_query method."""
        data = np.random.randn(100)  # TimeSeries expects 1D data
        ts = TimeSeries(data)

        # This should create the tree if needed
        query = ts._compute_kdtree_query(k=3)
        assert query is not None
        # Query returns tuple of (distances, indices)
        assert len(query) == 2  # Should be tuple


class TestTimeSeriesEntropy:
    """Test entropy computation in TimeSeries."""

    def test_get_entropy_continuous(self):
        """Test entropy computation for continuous data."""
        data = np.random.randn(200)
        ts = TimeSeries(data, discrete=False)

        # Get entropy with default parameters
        h = ts.get_entropy()
        assert isinstance(h, float)
        assert h > 0  # Gaussian has positive differential entropy

        # Should be cached
        assert 1 in ts.entropy  # Default ds=1

    def test_get_entropy_discrete(self):
        """Test entropy computation for discrete data."""
        data = np.random.randint(0, 4, 200)
        ts = TimeSeries(data, discrete=True)

        h = ts.get_entropy()
        assert isinstance(h, float)
        assert 0 <= h <= np.log2(4)  # Max entropy for 4 states in bits

        # Should be cached
        assert 1 in ts.entropy  # Default ds=1

    def test_get_entropy_with_downsampling(self):
        """Test entropy with downsampling."""
        data = np.random.randn(1000)
        ts = TimeSeries(data, discrete=False)

        # Entropy with different downsampling
        h1 = ts.get_entropy(ds=1)
        h2 = ts.get_entropy(ds=2)
        h5 = ts.get_entropy(ds=5)

        # All should be reasonable
        assert h1 > 0
        assert h2 > 0
        assert h5 > 0

        # Should be cached separately
        assert 1 in ts.entropy
        assert 2 in ts.entropy
        assert 5 in ts.entropy

    def test_entropy_caching(self):
        """Test that entropy values are properly cached."""
        data = np.random.randn(200)
        ts = TimeSeries(data, discrete=False)

        # First call computes entropy
        h1 = ts.get_entropy(ds=1)
        # Second call should return cached value
        h2 = ts.get_entropy(ds=1)

        assert h1 == h2
        assert 1 in ts.entropy


class TestTimeSeriesProperties:
    """Test various properties and edge cases."""

    def test_copula_normal_transform(self):
        """Test copula normal transformation."""
        data = np.random.exponential(2, 200)  # Non-normal data
        ts = TimeSeries(data, discrete=False)

        assert ts.copula_normal_data is not None
        # Should be approximately standard normal
        assert -4 < ts.copula_normal_data.min() < -1
        assert 1 < ts.copula_normal_data.max() < 4
        assert abs(ts.copula_normal_data.mean()) < 0.3

    def test_discrete_no_copula(self):
        """Test that discrete data doesn't compute copula transform."""
        data = np.random.randint(0, 10, 200)
        ts = TimeSeries(data, discrete=True)

        # Copula transform should not be computed for discrete data
        assert ts.copula_normal_data is None

    def test_non_binary_discrete(self):
        """Test non-binary discrete data."""
        data = np.random.randint(0, 10, 200)
        ts = TimeSeries(data, discrete=True)

        assert ts.discrete == True
        assert ts.is_binary == False
        assert hasattr(ts, "int_data")
        assert not hasattr(ts, "bool_data")

    def test_edge_case_single_value(self):
        """Test time series with single repeated value."""
        data = np.ones(100)

        # Should be detected as discrete
        ts = TimeSeries(data)
        assert ts.discrete == True

        # Scaled data should handle constant
        assert np.all(ts.scdata == ts.scdata[0])


class TestTimeSeriesFilter:
    """Test TimeSeries filter method."""

    def test_filter_gaussian(self):
        """Test Gaussian filtering."""
        # Create noisy sine wave
        t = np.linspace(0, 10, 500)
        clean_signal = np.sin(t)
        noise = np.random.randn(500) * 0.5
        noisy_signal = clean_signal + noise

        ts = TimeSeries(noisy_signal, discrete=False)

        # Apply Gaussian filter
        filtered = ts.filter(method="gaussian", sigma=2.0)

        assert isinstance(filtered, TimeSeries)
        assert len(filtered.data) == len(ts.data)
        # Filtered should have less variance
        assert np.var(filtered.data) < np.var(ts.data)

    def test_filter_savgol(self):
        """Test Savitzky-Golay filtering."""
        # Create noisy data
        data = np.random.randn(100) + np.sin(np.linspace(0, 10, 100))
        ts = TimeSeries(data, discrete=False)

        # Apply Savgol filter
        filtered = ts.filter(method="savgol", window_length=7, polyorder=3)

        assert isinstance(filtered, TimeSeries)
        assert len(filtered.data) == len(ts.data)

    def test_filter_wavelet(self):
        """Test wavelet denoising."""
        # Create noisy data
        t = np.linspace(0, 1, 256)
        clean = np.sin(2 * np.pi * 5 * t)
        noisy = clean + 0.5 * np.random.randn(256)

        ts = TimeSeries(noisy, discrete=False)

        # Apply wavelet filter
        filtered = ts.filter(method="wavelet", wavelet="db4")

        assert isinstance(filtered, TimeSeries)
        assert len(filtered.data) == len(ts.data)

    def test_filter_none(self):
        """Test no filtering (copy)."""
        data = np.random.randn(100)
        ts = TimeSeries(data, discrete=False)

        # No filtering
        filtered = ts.filter(method="none")

        assert isinstance(filtered, TimeSeries)
        assert np.array_equal(filtered.data, ts.data)
        # Should be a copy
        assert filtered is not ts
        assert filtered.data is not ts.data

    def test_filter_discrete_warning(self):
        """Test warning when filtering discrete data."""
        data = np.random.randint(0, 5, 100)
        ts = TimeSeries(data, discrete=True)

        with pytest.warns(UserWarning, match="Filtering discrete"):
            filtered = ts.filter(method="gaussian")

        assert isinstance(filtered, TimeSeries)


class TestTimeSeriesApproximateEntropy:
    """Test approximate entropy method."""

    def test_approximate_entropy_random(self):
        """Test ApEn for random data."""
        # Random data should have high ApEn
        data = np.random.randn(500)
        ts = TimeSeries(data, discrete=False)

        apen = ts.approximate_entropy(m=2, r=None)

        assert isinstance(apen, float)
        assert apen > 1.0  # Random data has high ApEn

    def test_approximate_entropy_periodic(self):
        """Test ApEn for periodic data."""
        # Periodic data should have low ApEn
        t = np.linspace(0, 10, 500)
        data = np.sin(t)
        ts = TimeSeries(data, discrete=False)

        apen = ts.approximate_entropy(m=2, r=0.2)

        assert isinstance(apen, float)
        assert apen < 1.0  # Periodic data has low ApEn

    def test_approximate_entropy_custom_r(self):
        """Test ApEn with custom tolerance."""
        # Use more structured data where the relationship holds
        t = np.linspace(0, 20, 500)
        data = np.sin(t) + 0.1 * np.random.randn(500)
        ts = TimeSeries(data, discrete=False)

        # Different r values
        apen1 = ts.approximate_entropy(m=2, r=0.1)
        apen2 = ts.approximate_entropy(m=2, r=0.5)

        assert isinstance(apen1, float)
        assert isinstance(apen2, float)
        # For structured data, larger r should give smaller ApEn
        assert apen2 < apen1

    def test_approximate_entropy_discrete_error(self):
        """Test error for discrete time series."""
        data = np.random.randint(0, 5, 200)
        ts = TimeSeries(data, discrete=True)

        with pytest.raises(ValueError, match="only valid for continuous"):
            ts.approximate_entropy()


class TestTimeSeriesInputValidation:
    """Test input validation for TimeSeries."""
    
    def test_valid_input(self):
        """Test that valid input passes validation."""
        # Normal continuous data
        data = np.random.randn(100)
        ts = TimeSeries(data)
        assert ts.data.shape == (100,)
        
        # Normal discrete data
        data = np.array([0, 1, 0, 1, 1, 0] * 10)
        ts = TimeSeries(data, discrete=True)
        assert ts.discrete == True
        
    def test_invalid_data_type(self):
        """Test that non-numpy data is converted properly."""
        # List should be converted
        data = [1, 2, 3, 4, 5]
        ts = TimeSeries(data)
        assert isinstance(ts.data, np.ndarray)
        
    def test_multidimensional_data(self):
        """Test that multidimensional data raises error."""
        data = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="Time series must be 1D"):
            TimeSeries(data)
            
    def test_short_series(self):
        """Test that series with less than 2 points raises error."""
        with pytest.raises(ValueError, match="must have at least 2 points"):
            TimeSeries([1])
            
        with pytest.raises(ValueError, match="must have at least 2 points"):
            TimeSeries(np.array([]))
            
    def test_nan_values(self):
        """Test that NaN values raise error."""
        data = np.array([1, 2, np.nan, 4, 5])
        with pytest.raises(ValueError, match="contains NaN values"):
            TimeSeries(data)
            
    def test_inf_values(self):
        """Test that infinite values raise error."""
        data = np.array([1, 2, np.inf, 4, 5])
        with pytest.raises(ValueError, match="contains infinite values"):
            TimeSeries(data)
            
        data = np.array([1, 2, -np.inf, 4, 5])
        with pytest.raises(ValueError, match="contains infinite values"):
            TimeSeries(data)
            
    def test_shuffle_mask_validation(self):
        """Test shuffle mask validation."""
        data = np.random.randn(100)
        
        # Valid shuffle mask
        mask = np.random.randint(0, 2, 100)
        ts = TimeSeries(data, shuffle_mask=mask)
        assert np.array_equal(ts.shuffle_mask, mask)
        
        # Wrong length
        mask = np.random.randint(0, 2, 50)
        with pytest.raises(ValueError, match="Shuffle mask must have same length"):
            TimeSeries(data, shuffle_mask=mask)
            
        # Invalid values
        mask = np.array([0, 1, 2, 1, 0] * 20)
        with pytest.raises(ValueError, match="Shuffle mask must contain only 0s and 1s"):
            TimeSeries(data, shuffle_mask=mask)
            
    def test_edge_cases(self):
        """Test edge cases for input validation."""
        # Minimum valid series
        data = np.array([1, 2])
        ts = TimeSeries(data)
        assert len(ts.data) == 2
        
        # All zeros (valid)
        data = np.zeros(100)
        ts = TimeSeries(data)
        assert np.all(ts.data == 0)
        
        # All same value (valid)
        data = np.ones(100) * 42
        ts = TimeSeries(data)
        assert np.all(ts.data == 42)
