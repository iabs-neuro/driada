"""Tests for synthetic time series generation."""

import numpy as np
import pytest
from driada.experiment.synthetic.time_series import generate_binary_time_series


class TestBinaryTimeSeries:
    """Test binary time series generation."""

    def test_generate_binary_time_series_basic(self):
        """Test basic binary time series generation."""
        length = 1000
        avg_islands = 10
        avg_duration = 20

        series = generate_binary_time_series(length, avg_islands, avg_duration)

        # Check basic properties
        assert len(series) == length
        assert set(np.unique(series)).issubset({0, 1})

        # Count actual islands
        islands = []
        in_island = False
        island_start = 0

        for i, val in enumerate(series):
            if val == 1 and not in_island:
                in_island = True
                island_start = i
            elif val == 0 and in_island:
                in_island = False
                islands.append((island_start, i))

        if in_island:  # Handle case where series ends with 1
            islands.append((island_start, len(series)))

        n_islands = len(islands)

        # Check number of islands is reasonable (within 50% of target)
        assert (
            0.5 * avg_islands <= n_islands <= 1.5 * avg_islands
        ), f"Expected ~{avg_islands} islands, got {n_islands}"

        # Check average duration if we have islands
        if n_islands > 0:
            durations = [end - start for start, end in islands]
            mean_duration = np.mean(durations)
            # Average duration should be within 50% of target
            assert (
                0.5 * avg_duration <= mean_duration <= 1.5 * avg_duration
            ), f"Expected avg duration ~{avg_duration}, got {mean_duration:.1f}"

        # Check active fraction
        active_fraction = np.mean(series)
        expected_fraction = (avg_islands * avg_duration) / length
        # Should be within reasonable range
        assert (
            0.5 * expected_fraction <= active_fraction <= 1.5 * expected_fraction
        ), f"Expected active fraction ~{expected_fraction:.3f}, got {active_fraction:.3f}"

    def test_generate_binary_time_series_edge_cases(self):
        """Test edge cases for binary time series generation."""
        # Very short series
        series = generate_binary_time_series(10, avg_islands=2, avg_duration=3)
        assert len(series) == 10

        # More islands than can fit
        series = generate_binary_time_series(100, avg_islands=50, avg_duration=10)
        assert len(series) == 100
        assert np.sum(series) <= 100  # Can't have more active points than total

        # Very long duration
        series = generate_binary_time_series(100, avg_islands=2, avg_duration=30)
        assert len(series) == 100

    def test_generate_binary_time_series_reproducibility(self):
        """Test that setting numpy seed gives reproducible results."""
        length = 500
        avg_islands = 5
        avg_duration = 20

        np.random.seed(42)
        series1 = generate_binary_time_series(length, avg_islands, avg_duration)

        np.random.seed(42)
        series2 = generate_binary_time_series(length, avg_islands, avg_duration)

        assert np.array_equal(series1, series2)

    def test_realistic_parameters(self):
        """Test with realistic parameters from mixed selectivity generation."""
        # Parameters from mixed_selectivity.py
        fps = 20
        duration = 600  # seconds
        length = duration * fps
        avg_islands = 10
        avg_duration = int(0.5 * fps)  # 0.5 seconds

        series = generate_binary_time_series(length, avg_islands, avg_duration)

        # Check active fraction - should be reasonable for neural data
        active_fraction = np.mean(series)
        print(f"Active fraction with realistic params: {active_fraction:.3f}")

        # For neural data, we typically want 5-20% active time
        assert (
            0.001 <= active_fraction <= 0.3
        ), f"Active fraction {active_fraction:.3f} is unrealistic for neural data"

        # Count actual islands
        islands = 0
        for i in range(1, len(series)):
            if series[i] == 1 and series[i - 1] == 0:
                islands += 1
        if series[0] == 1:
            islands += 1

        print(f"Number of islands: {islands}")
        assert islands > 0, "Should have at least some active periods"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
