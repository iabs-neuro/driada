"""Tests for MultiTimeSeries class in info_base module."""

import numpy as np
import pytest
from driada.information.info_base import TimeSeries, MultiTimeSeries


class TestMultiTimeSeriesInitialization:
    """Test MultiTimeSeries initialization."""

    def test_init_from_numpy_array(self):
        """Test initialization from numpy array."""
        # Create 2D array: 5 time series, 100 time points each
        data = np.random.randn(5, 100)

        mts = MultiTimeSeries(data, discrete=False)

        assert mts.data.shape == (5, 100)
        assert mts.discrete == False
        assert mts.shape == (5, 100)
        assert mts.n_points == 100  # From MVData
        assert mts.n_dim == 5  # From MVData

    def test_init_from_timeseries_list(self):
        """Test initialization from list of TimeSeries."""
        ts_list = [
            TimeSeries(np.random.randn(100), discrete=False),
            TimeSeries(np.random.randn(100), discrete=False),
            TimeSeries(np.random.randn(100), discrete=False),
        ]

        mts = MultiTimeSeries(ts_list)

        assert mts.data.shape == (3, 100)
        assert mts.n_dim == 3
        assert mts.discrete == False

    def test_init_discrete_data(self):
        """Test initialization with discrete data."""
        data = np.random.randint(0, 5, size=(4, 80))

        mts = MultiTimeSeries(data, discrete=True)

        assert mts.discrete == True
        assert hasattr(mts, "int_data")
        assert mts.int_data.shape == (4, 80)

    def test_init_with_labels(self):
        """Test initialization with point labels."""
        data = np.random.randn(3, 50)
        # Labels should be for points (time steps), not dimensions
        labels = np.arange(50)

        mts = MultiTimeSeries(data, labels=labels, discrete=False)

        assert np.array_equal(mts.labels, labels)
        assert len(mts.labels) == mts.n_points

    def test_init_with_downsampling(self):
        """Test initialization with downsampling."""
        data = np.random.randn(3, 1000)

        mts = MultiTimeSeries(data, downsampling=10, discrete=False)

        # Should be downsampled
        assert mts.data.shape == (3, 100)
        assert mts.n_points == 100

    def test_init_mixed_lengths_error(self):
        """Test error when TimeSeries have different lengths."""
        ts_list = [
            TimeSeries(np.random.randn(100), discrete=False),
            TimeSeries(np.random.randn(80), discrete=False),  # Different length
        ]

        with pytest.raises(
            ValueError, match="All TimeSeries must have the same length"
        ):
            MultiTimeSeries(ts_list)

    def test_init_mixed_types_error(self):
        """Test error when TimeSeries have different types."""
        ts_list = [
            TimeSeries(np.random.randn(100), discrete=False),
            TimeSeries(np.random.randint(0, 5, 100), discrete=True),  # Different type
        ]

        with pytest.raises(
            ValueError,
            match="All components of MultiTimeSeries must be either continuous or discrete",
        ):
            MultiTimeSeries(ts_list)

    def test_init_with_shuffle_mask(self):
        """Test initialization with shuffle mask."""
        data = np.random.randn(3, 100)
        mask = np.array([True, False] * 50)

        mts = MultiTimeSeries(data, shuffle_mask=mask, discrete=False)

        assert hasattr(mts, "shuffle_mask")
        assert mts.shuffle_mask.shape == (100,)

    def test_init_missing_discrete_param(self):
        """Test error when discrete parameter missing for numpy input."""
        data = np.random.randn(3, 100)

        with pytest.raises(ValueError, match="'discrete' parameter must be specified"):
            MultiTimeSeries(data)  # Missing discrete parameter

    def test_init_wrong_dimensions(self):
        """Test error with wrong array dimensions."""
        data_1d = np.random.randn(100)
        data_3d = np.random.randn(3, 10, 10)

        with pytest.raises(ValueError, match="must be 2D"):
            MultiTimeSeries(data_1d, discrete=False)

        with pytest.raises(ValueError, match="must be 2D"):
            MultiTimeSeries(data_3d, discrete=False)


class TestMultiTimeSeriesProperties:
    """Test MultiTimeSeries properties and attributes."""

    def test_shape_property(self):
        """Test shape property from parent class."""
        data = np.random.randn(5, 200)
        mts = MultiTimeSeries(data, discrete=False)

        assert mts.shape == (5, 200)
        assert mts.n_dim == 5
        assert mts.n_points == 200

    def test_data_attributes(self):
        """Test data-related attributes."""
        data = np.random.randn(3, 100)
        mts = MultiTimeSeries(data, discrete=False)

        # Should have scaled data
        assert hasattr(mts, "scdata")
        assert mts.scdata.shape == (3, 100)

        # Should have copula normal data for continuous
        assert hasattr(mts, "copula_normal_data")
        assert mts.copula_normal_data.shape == (3, 100)

    def test_discrete_attributes(self):
        """Test attributes for discrete MultiTimeSeries."""
        # Create discrete data with at least some variation to avoid zero columns
        np.random.seed(42)  # For reproducibility
        data = np.random.randint(1, 6, size=(4, 80))  # Use 1-5 instead of 0-4
        mts = MultiTimeSeries(data, discrete=True)

        assert hasattr(mts, "int_data")
        assert mts.int_data.shape == (4, 80)
        assert mts.copula_normal_data is None

    def test_binary_properties(self):
        """Test binary data properties."""
        # Create binary data
        data = np.random.randint(0, 2, size=(4, 100))
        mts = MultiTimeSeries(data, discrete=True, allow_zero_columns=True)

        # Check if components are binary
        ts_list = [TimeSeries(data[i, :], discrete=True) for i in range(4)]
        for ts in ts_list:
            assert hasattr(ts, "is_binary")
            if len(set(ts.int_data)) == 2:
                assert ts.is_binary == True

    def test_continuous_scaling(self):
        """Test continuous data scaling."""
        data = np.random.rand(3, 100) * 100 + 50  # Data in [50, 150]
        mts = MultiTimeSeries(data, discrete=False)

        # Should have scaled data
        assert hasattr(mts, "scdata")
        # Each row should be scaled
        for i in range(3):
            assert mts.scdata[i].min() >= 0
            assert mts.scdata[i].max() <= 1

    def test_rescale_rows(self):
        """Test row rescaling option."""
        # Create data with different scales per row
        data = np.array(
            [
                np.random.randn(100) * 10,  # Large scale
                np.random.randn(100) * 0.1,  # Small scale
                np.random.randn(100) * 1,  # Medium scale
            ]
        )

        mts = MultiTimeSeries(data, rescale_rows=True, discrete=False)

        # Check that data has been rescaled per row
        # Note: rescale_rows in MVData uses MinMaxScaler to rescale each row to [0, 1]
        for i in range(3):
            row_min = mts.data[i].min()
            row_max = mts.data[i].max()
            # Check that each row is scaled to approximately [0, 1]
            assert row_min >= -1e-10  # Allow small numerical errors
            assert row_max <= 1.0 + 1e-10
            # Check that the range is close to 1 (unless all values were the same)
            if np.std(data[i]) > 1e-10:
                assert row_max - row_min > 0.99


class TestMultiTimeSeriesEntropy:
    """Test entropy methods for MultiTimeSeries."""

    def test_get_entropy_continuous(self):
        """Test entropy calculation for continuous MultiTimeSeries."""
        data = np.random.randn(3, 200)
        mts = MultiTimeSeries(data, discrete=False)

        # Get entropy of the whole MultiTimeSeries
        h = mts.get_entropy()
        assert isinstance(h, float)
        assert h > 0  # Gaussian has positive entropy

        # Test with downsampling
        h_ds = mts.get_entropy(ds=2)
        assert isinstance(h_ds, float)
        assert h_ds > 0

    def test_get_entropy_discrete(self):
        """Test entropy calculation for discrete MultiTimeSeries."""
        # Test with 2 variables (supported)
        data2 = np.random.randint(0, 4, size=(2, 200))
        mts2 = MultiTimeSeries(data2, discrete=True, allow_zero_columns=True)

        # Get joint entropy of 2 series
        h2 = mts2.get_entropy()
        assert isinstance(h2, float)
        assert h2 >= 0  # Entropy is non-negative

        # Test with 3 variables (not yet supported)
        data3 = np.random.randint(0, 4, size=(3, 200))
        mts3 = MultiTimeSeries(data3, discrete=True, allow_zero_columns=True)

        with pytest.raises(
            NotImplementedError, match="Joint entropy for 3 discrete variables"
        ):
            h3 = mts3.get_entropy()


class TestMultiTimeSeriesFilter:
    """Test filtering methods for MultiTimeSeries."""

    def test_filter_gaussian(self):
        """Test Gaussian filtering."""
        # Create noisy data
        t = np.linspace(0, 10, 500)
        clean_data = np.array([np.sin(t), np.cos(t), np.sin(2 * t)])
        noise = np.random.randn(3, 500) * 0.5
        noisy_data = clean_data + noise

        mts = MultiTimeSeries(noisy_data, discrete=False)

        # Apply Gaussian filter
        filtered = mts.filter(method="gaussian", sigma=2.0)

        assert isinstance(filtered, MultiTimeSeries)
        assert filtered.data.shape == mts.data.shape
        # Should be smoother (less variance)
        assert np.var(filtered.data) < np.var(mts.data)

    def test_filter_none(self):
        """Test no filtering (copy)."""
        data = np.random.randn(2, 100)
        mts = MultiTimeSeries(data, discrete=False)

        # No filtering
        filtered = mts.filter(method="none")

        assert isinstance(filtered, MultiTimeSeries)
        assert np.array_equal(filtered.data, mts.data)
        # Should be a copy, not same object
        assert filtered is not mts
        assert filtered.data is not mts.data

    def test_filter_discrete_warning(self):
        """Test warning when filtering discrete data."""
        data = np.random.randint(0, 5, size=(2, 100))
        mts = MultiTimeSeries(data, discrete=True, allow_zero_columns=True)

        with pytest.warns(UserWarning, match="Filtering discrete"):
            filtered = mts.filter(method="gaussian")

        assert isinstance(filtered, MultiTimeSeries)

    def test_filter_preserves_attributes(self):
        """Test that filtering preserves MultiTimeSeries attributes."""
        data = np.random.randn(3, 100)
        # Labels should be for points, not dimensions
        labels = np.arange(100)
        mts = MultiTimeSeries(
            data, labels=labels, data_name="test_data", discrete=False
        )

        filtered = mts.filter(method="gaussian")

        assert np.array_equal(filtered.labels, labels)
        assert filtered.data_name == "test_data"
        assert filtered.discrete == False
        assert filtered.shape == mts.shape
