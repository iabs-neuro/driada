"""
Tests for spatial analysis utilities.
"""

import pytest
import numpy as np
from unittest.mock import patch

from driada.utils.spatial import (
    compute_occupancy_map,
    compute_rate_map,
    extract_place_fields,
    compute_spatial_information_rate,
    compute_spatial_decoding_accuracy,
    compute_spatial_information,
    compute_spatial_metrics,
)
from driada.information import TimeSeries, MultiTimeSeries


class TestOccupancyMap:
    """Test occupancy map computation."""

    def test_compute_occupancy_map_basic(self):
        """Test basic occupancy map computation."""
        # Create simple trajectory
        positions = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.1],
                [0.4, 0.2],
            ]
        )

        occupancy, x_edges, y_edges = compute_occupancy_map(
            positions, arena_bounds=((0, 0.5), (0, 0.5)), bin_size=0.1, min_occupancy=0
        )

        # Check shape
        assert occupancy.shape == (5, 5)
        assert len(x_edges) == 6
        assert len(y_edges) == 6

        # Check that visited bins have non-zero occupancy
        assert occupancy[0, 0] > 0  # Bottom-left visited
        assert np.sum(occupancy > 0) > 0

    def test_compute_occupancy_map_smoothing(self):
        """Test occupancy map with smoothing."""
        positions = np.random.rand(100, 2)

        # Without smoothing
        occ_no_smooth, _, _ = compute_occupancy_map(positions, bin_size=0.1, smooth_sigma=None)

        # With smoothing
        occ_smooth, _, _ = compute_occupancy_map(positions, bin_size=0.1, smooth_sigma=1.0)

        # Smoothed should have fewer NaN values
        assert np.sum(np.isnan(occ_smooth)) <= np.sum(np.isnan(occ_no_smooth))

    def test_compute_occupancy_map_min_occupancy(self):
        """Test minimum occupancy threshold."""
        positions = np.array([[0.05, 0.05]])  # Single point

        occupancy, _, _ = compute_occupancy_map(
            positions,
            arena_bounds=((0, 0.1), (0, 0.1)),
            bin_size=0.05,
            min_occupancy=2.0,  # Higher than single visit
        )

        # All bins should be NaN due to min occupancy
        assert np.all(np.isnan(occupancy))

    def test_compute_occupancy_map_auto_bounds(self):
        """Test automatic arena bounds detection."""
        positions = np.random.rand(100, 2) * 2.0  # Range 0-2

        occupancy, x_edges, y_edges = compute_occupancy_map(
            positions, arena_bounds=None, bin_size=0.1  # Auto-detect
        )

        # Check bounds include all data with margin
        assert x_edges[0] < positions[:, 0].min()
        assert x_edges[-1] > positions[:, 0].max()
        assert y_edges[0] < positions[:, 1].min()
        assert y_edges[-1] > positions[:, 1].max()


class TestRateMap:
    """Test firing rate map computation."""

    def test_compute_rate_map_basic(self):
        """Test basic rate map computation."""
        # Create occupancy map
        positions = np.random.rand(1000, 2)
        occupancy, x_edges, y_edges = compute_occupancy_map(
            positions, bin_size=0.1, min_occupancy=0
        )

        # Create continuous neural signal (e.g., calcium)
        neural_signal = np.random.exponential(1.0, 1000)  # Positive continuous signal

        rate_map = compute_rate_map(
            neural_signal, positions, occupancy, x_edges, y_edges, smooth_sigma=None
        )

        # Check shape matches occupancy
        assert rate_map.shape == occupancy.shape

        # Check no negative rates
        assert np.all(rate_map >= 0)

        # Check NaN handling
        assert not np.any(np.isnan(rate_map[~np.isnan(occupancy)]))

    def test_compute_rate_map_smoothing(self):
        """Test rate map smoothing."""
        positions = np.random.rand(1000, 2)
        occupancy, x_edges, y_edges = compute_occupancy_map(positions, bin_size=0.1)

        # Create continuous signal with some structure
        neural_signal = np.abs(np.random.normal(2.0, 0.5, 1000))

        # Without smoothing
        rate_no_smooth = compute_rate_map(
            neural_signal, positions, occupancy, x_edges, y_edges, smooth_sigma=None
        )

        # With smoothing
        rate_smooth = compute_rate_map(
            neural_signal, positions, occupancy, x_edges, y_edges, smooth_sigma=2.0
        )

        # Smoothed map should be smoother (have valid values)
        valid_mask = ~np.isnan(occupancy)
        if np.sum(valid_mask) > 10:
            # Check that smoothing was applied (values should differ)
            assert not np.allclose(rate_smooth[valid_mask], rate_no_smooth[valid_mask])

    def test_compute_rate_map_no_spikes(self):
        """Test rate map with no activity."""
        positions = np.random.rand(100, 2)
        occupancy, x_edges, y_edges = compute_occupancy_map(positions)

        neural_signal = np.zeros(100)  # No activity

        rate_map = compute_rate_map(neural_signal, positions, occupancy, x_edges, y_edges)

        # Should be all zeros
        assert np.all(rate_map == 0)


class TestPlaceFields:
    """Test place field extraction."""

    def test_extract_place_fields_single_field(self):
        """Test extraction of single place field."""
        # Create rate map with single peak
        rate_map = np.zeros((20, 20))
        rate_map[8:12, 8:12] = 5.0  # Central place field
        rate_map[10, 10] = 10.0  # Peak

        fields = extract_place_fields(
            rate_map, min_peak_rate=5.0, min_field_size=4, peak_to_mean_ratio=1.2
        )

        assert len(fields) == 1
        assert fields[0]["peak_rate"] == 10.0
        assert fields[0]["size"] >= 4
        # Center should be near (10, 10) - allow for center of mass calculation differences
        center = fields[0]["center"]
        assert abs(center[0] - 10) <= 1
        assert abs(center[1] - 10) <= 1

    def test_extract_place_fields_multiple(self):
        """Test extraction of multiple place fields."""
        # Create rate map with two peaks
        rate_map = np.zeros((30, 30))
        rate_map[5:8, 5:8] = 3.0
        rate_map[6, 6] = 6.0  # First peak
        rate_map[20:23, 20:23] = 3.0
        rate_map[21, 21] = 6.0  # Second peak

        fields = extract_place_fields(rate_map, min_peak_rate=5.0, min_field_size=4)

        assert len(fields) == 2
        assert all(f["peak_rate"] >= 5.0 for f in fields)

    def test_extract_place_fields_size_threshold(self):
        """Test field size threshold."""
        # Small field that should be rejected
        rate_map = np.zeros((20, 20))
        rate_map[10, 10] = 10.0  # Single bin peak

        fields = extract_place_fields(
            rate_map, min_peak_rate=5.0, min_field_size=4  # Requires at least 4 bins
        )

        assert len(fields) == 0


class TestSpatialInformation:
    """Test spatial information metrics."""

    def test_compute_spatial_information_rate_uniform(self):
        """Test spatial information for uniform firing."""
        # Uniform rate map and occupancy
        rate_map = np.ones((10, 10))
        occupancy_map = np.ones((10, 10))

        info = compute_spatial_information_rate(rate_map, occupancy_map)

        # Uniform firing has zero spatial information
        assert info == pytest.approx(0.0, abs=1e-10)

    def test_compute_spatial_information_rate_localized(self):
        """Test spatial information for localized firing."""
        # Highly localized firing
        rate_map = np.zeros((10, 10))
        rate_map[5, 5] = 10.0  # Single location

        occupancy_map = np.ones((10, 10))

        info = compute_spatial_information_rate(rate_map, occupancy_map)

        # Localized firing has high spatial information
        assert info > 0.0

    def test_compute_spatial_information_rate_no_firing(self):
        """Test spatial information with no firing."""
        rate_map = np.zeros((10, 10))
        occupancy_map = np.ones((10, 10))

        info = compute_spatial_information_rate(rate_map, occupancy_map)

        # No firing means no information
        assert info == 0.0


class TestSpatialDecoding:
    """Test position decoding from neural activity."""

    def test_compute_spatial_decoding_accuracy_perfect(self):
        """Test decoding with perfect position encoding."""
        # Create neural activity that perfectly encodes position
        n_samples = 1000
        positions = np.column_stack(
            [
                np.sin(np.linspace(0, 4 * np.pi, n_samples)),
                np.cos(np.linspace(0, 4 * np.pi, n_samples)),
            ]
        )

        # Neural activity directly encodes position
        neural_activity = positions.T  # 2 neurons encoding x and y

        metrics = compute_spatial_decoding_accuracy(
            neural_activity, positions, test_size=0.3, n_estimators=10, random_state=42
        )

        # Should have high accuracy
        assert metrics["r2_avg"] > 0.8
        assert metrics["r2_x"] > 0.8
        assert metrics["r2_y"] > 0.8
        assert metrics["mse"] < 0.1

    def test_compute_spatial_decoding_accuracy_random(self):
        """Test decoding with random neural activity."""
        n_samples = 500
        positions = np.random.rand(n_samples, 2)
        neural_activity = np.random.rand(10, n_samples)  # Random activity

        metrics = compute_spatial_decoding_accuracy(
            neural_activity, positions, test_size=0.3, random_state=42
        )

        # Random activity should have poor decoding
        assert metrics["r2_avg"] < 0.3  # Low R²
        assert metrics["mse"] > 0.05  # High error

    def test_compute_spatial_decoding_accuracy_with_logger(self):
        """Test decoding with logger."""
        import logging

        logger = logging.getLogger("test")

        positions = np.random.rand(100, 2)
        neural_activity = np.random.rand(5, 100)

        with patch.object(logger, "info") as mock_info:
            metrics = compute_spatial_decoding_accuracy(neural_activity, positions, logger=logger)

            # Should log progress
            assert mock_info.call_count >= 2


class TestSpatialMI:
    """Test spatial mutual information computation."""

    def test_compute_spatial_information_arrays(self):
        """Test MI computation with numpy arrays."""
        # Create correlated data
        n_samples = 1000
        positions = np.random.rand(n_samples, 2)

        # Neural activity correlates with position
        neural_activity = positions[:, 0] + 0.1 * np.random.randn(n_samples)

        metrics = compute_spatial_information(neural_activity, positions)

        assert "mi_x" in metrics
        assert "mi_y" in metrics
        assert "mi_total" in metrics

        # Should have higher MI with X than Y
        assert metrics["mi_x"] > metrics["mi_y"]
        assert metrics["mi_total"] > 0

    def test_compute_spatial_information_timeseries(self):
        """Test MI computation with TimeSeries objects."""
        n_samples = 500
        positions = np.random.rand(n_samples, 2)
        neural_ts = TimeSeries(np.random.rand(n_samples), discrete=False)

        metrics = compute_spatial_information(neural_ts, positions)

        assert all(k in metrics for k in ["mi_x", "mi_y", "mi_total"])
        assert all(v >= 0 for v in metrics.values())

    def test_compute_spatial_information_multitimeseries(self):
        """Test MI computation with MultiTimeSeries."""
        n_samples = 500
        positions = np.random.rand(n_samples, 2)

        # Multiple neurons
        neural_data = [TimeSeries(np.random.rand(n_samples), discrete=False) for _ in range(3)]
        neural_mts = MultiTimeSeries(neural_data)

        metrics = compute_spatial_information(neural_mts, positions)

        assert all(k in metrics for k in ["mi_x", "mi_y", "mi_total"])

    def test_compute_spatial_information_invalid_positions(self):
        """Test MI computation with invalid position dimensions."""
        neural_activity = np.random.rand(100)
        positions = np.random.rand(100, 3)  # 3D positions

        with pytest.raises(ValueError, match="Positions must be 2D"):
            compute_spatial_information(neural_activity, positions)


class TestComputeSpatialMetrics:
    """Test selective metric computation."""

    def test_compute_spatial_metrics_all(self):
        """Test computing all metrics."""
        positions = np.random.rand(200, 2)
        neural_activity = np.random.rand(5, 200)

        results = compute_spatial_metrics(neural_activity, positions, metrics=None)  # Compute all

        # Should have all metric types (decoding and information only)
        assert "decoding" in results
        assert "information" in results
        assert "place_fields" not in results

    def test_compute_spatial_metrics_subset(self):
        """Test computing subset of metrics."""
        positions = np.random.rand(200, 2)
        neural_activity = np.random.rand(5, 200)

        results = compute_spatial_metrics(
            neural_activity, positions, metrics=["decoding", "information"]
        )

        # Should only have requested metrics
        assert "decoding" in results
        assert "information" in results
        assert "place_fields" not in results

    def test_compute_spatial_metrics_with_kwargs(self):
        """Test passing kwargs to analysis functions."""
        positions = np.random.rand(200, 2)
        neural_activity = np.random.rand(5, 200)

        results = compute_spatial_metrics(
            neural_activity,
            positions,
            metrics=["decoding"],
            test_size=0.2,  # Kwarg for decoding
            n_estimators=5,  # Another kwarg
        )

        # Should complete without error
        assert "decoding" in results
        assert "r2_avg" in results["decoding"]

    def test_compute_spatial_metrics_place_fields_error(self):
        """Verify requesting 'place_fields' metric raises ValueError."""
        positions = np.random.rand(200, 2)
        neural_activity = np.random.rand(5, 200)

        with pytest.raises(ValueError, match="Place field detection removed"):
            compute_spatial_metrics(neural_activity, positions, metrics=["place_fields"])
