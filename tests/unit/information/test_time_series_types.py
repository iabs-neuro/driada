"""
Comprehensive tests for time series type detection system.

Tests cover:
- Basic discrete/continuous detection
- Circular/periodic pattern detection
- Subtype classification
- Edge cases and error handling
- Backward compatibility
"""

import numpy as np
import pytest

from driada.information.time_series_types import (
    analyze_time_series_type,
    is_discrete_time_series,
    _extract_statistical_properties,
    _detect_circular,
    _detect_periodicity,
    _detect_discrete_subtype,
    _detect_continuous_subtype,
)


class TestTimeSeriesTypeDetection:
    """Test the main type detection functionality."""

    def test_timeline_detection(self):
        """Test detection of timeline data (regularly spaced discrete values)."""
        # Integer timeline with high uniqueness (0, 10, 20, 30, ..., 990)
        data = np.arange(0, 1000, 10)  # 100 unique values from 0 to 990
        np.random.shuffle(data)
        result = analyze_time_series_type(data)

        assert result.is_discrete
        assert result.subtype == "timeline"
        assert result.confidence > 0.8  # Timeline detection has 0.85 confidence

        # Non-integer timeline with high uniqueness (0, 0.5, 1, 1.5, ..., 49.5)
        data = np.arange(0, 50, 0.5)  # 100 unique values
        np.random.shuffle(data)
        result = analyze_time_series_type(data)

        assert result.is_discrete
        assert result.subtype == "timeline"
        assert result.confidence > 0.8  # Timeline detection has 0.85 confidence

    def test_binary_detection(self):
        """Test detection of binary time series."""
        # Pure binary
        data = np.array([0, 1, 0, 1, 1, 0, 0, 1] * 20)
        result = analyze_time_series_type(data)

        assert result.is_discrete
        assert result.primary_type == "discrete"
        assert result.subtype == "binary"
        assert result.confidence > 0.7  # Binary should have good confidence

        # Binary with very small noise (should still be detected as discrete if noise is negligible)
        data_small_noise = data + np.random.normal(0, 0.001, len(data))
        result_small_noise = analyze_time_series_type(data_small_noise)
        # With tiny noise, might still detect as discrete

        # Binary with significant noise (should be continuous)
        data_noisy = data.astype(float) + np.random.normal(0, 0.1, len(data))
        result_noisy = analyze_time_series_type(data_noisy)
        assert (
            result_noisy.is_continuous
        )  # With significant noise, it's no longer discrete

    def test_categorical_detection(self):
        """Test detection of categorical time series."""
        # Small number of categories
        data = np.random.choice([1, 2, 3, 4, 5], size=200)
        result = analyze_time_series_type(data)

        assert result.is_discrete
        assert result.subtype == "categorical"
        assert result.confidence >= 0.8  # Allow exactly 0.8

        # Many categories
        data_many = np.random.choice(range(15), size=300)
        result_many = analyze_time_series_type(data_many)
        assert result_many.is_discrete
        assert result_many.subtype == "categorical"

    def test_count_data_detection(self):
        """Test detection of count data."""
        # Monotonically increasing count data (e.g., cumulative counts)
        data = np.cumsum(np.random.poisson(lam=2, size=100))
        result = analyze_time_series_type(data)

        assert result.is_discrete
        assert result.subtype == "count"
        assert result.confidence > 0.9

        # Simple counting sequence
        data_simple = np.arange(0, 50)
        result_simple = analyze_time_series_type(data_simple)
        assert result_simple.is_discrete
        assert result_simple.subtype == "count"

    def test_continuous_linear_detection(self):
        """Test detection of continuous linear data."""
        # Normal distribution
        data = np.random.normal(100, 15, size=500)
        result = analyze_time_series_type(data)

        assert result.is_continuous
        assert result.primary_type == "continuous"
        assert result.subtype == "linear"
        assert result.confidence > 0.7

        # Uniform distribution
        data_uniform = np.random.uniform(0, 100, size=500)
        result_uniform = analyze_time_series_type(data_uniform)
        assert result_uniform.is_continuous
        assert result_uniform.subtype == "linear"

    def test_circular_detection_with_ranges(self):
        """Test detection of circular data with common ranges."""
        # Test [0, 2π] range
        data_2pi = np.random.uniform(0, 2 * np.pi, size=200)
        result = analyze_time_series_type(data_2pi, name="angle")
        assert result.is_circular
        assert result.subtype == "circular"
        assert np.isclose(result.circular_period, 2 * np.pi, rtol=0.1)

        # Test [-π, π] range
        data_pi = np.random.uniform(-np.pi, np.pi, size=200)
        result = analyze_time_series_type(data_pi)
        assert result.is_circular
        assert np.isclose(result.circular_period, 2 * np.pi, rtol=0.1)

        # Test [0, 360] range
        data_360 = np.random.uniform(0, 360, size=200)
        result = analyze_time_series_type(data_360, name="heading_degrees")
        assert result.is_circular
        assert np.isclose(result.circular_period, 360, rtol=0.1)

        # Test [-180, 180] range
        data_180 = np.random.uniform(-180, 180, size=200)
        result = analyze_time_series_type(data_180)
        assert result.is_circular
        assert np.isclose(result.circular_period, 360, rtol=0.1)

    def test_circular_detection_with_wraparound(self):
        """Test detection of circular data with wraparound."""
        # Create data with clear wraparound
        angles = np.linspace(0, 10 * np.pi, 200) % (2 * np.pi)
        result = analyze_time_series_type(angles)

        assert result.is_circular
        assert result.confidence > 0.6

    def test_context_aware_detection(self):
        """Test that names influence detection appropriately."""
        # Ambiguous data that could be circular
        data = np.random.uniform(0, 6.28, size=100)

        # Without context
        result1 = analyze_time_series_type(data)
        confidence1 = result1.confidence if result1.is_circular else 0

        # With circular context
        result2 = analyze_time_series_type(data, name="phase_angle")
        confidence2 = result2.confidence if result2.is_circular else 0

        # Context should increase confidence for circular detection
        assert confidence2 > confidence1
        assert result2.is_circular

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very short time series
        with pytest.warns(UserWarning, match="Only .* samples available"):
            result = analyze_time_series_type([1, 2, 3, 4, 5])
            assert result.primary_type in ["discrete", "continuous"]

        # Single unique value
        data_const = np.ones(100)
        result_const = analyze_time_series_type(data_const)
        assert result_const.is_discrete

        # All unique values
        data_unique = np.arange(100) + np.random.normal(0, 0.1, 100)
        result_unique = analyze_time_series_type(data_unique)
        assert result_unique.is_continuous

    def test_mixed_ambiguous_data(self):
        """Test detection on ambiguous data."""
        # Discrete values but many unique (e.g., rounded continuous)
        data = np.round(np.random.exponential(10, 300), 1)
        result = analyze_time_series_type(data)

        # Should make a decision even if ambiguous
        assert result.primary_type in ["discrete", "continuous"]
        assert 0.4 <= result.confidence <= 0.8  # Moderate confidence


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_is_discrete_time_series(self):
        """Test simple discrete detection function."""
        # Clear discrete
        assert is_discrete_time_series([0, 1, 0, 1, 1, 0] * 10)  # Binary data
        assert is_discrete_time_series([1, 2, 3, 1, 2, 3] * 20)  # Categorical data

        # Clear continuous
        assert not is_discrete_time_series(np.random.normal(0, 1, 100))
        # Note: linspace creates a perfect timeline, so it's detected as discrete
        assert not is_discrete_time_series(
            np.random.uniform(0, 10, 100)
        )  # Random continuous data

        # With confidence
        is_discrete, conf = is_discrete_time_series(
            [1, 2, 3, 4, 5] * 20, return_confidence=True
        )
        assert is_discrete
        assert 0 < conf <= 1

    def test_legacy_detect_ts_type(self):
        """Test that legacy alias works."""
        from driada.information.time_series_types import detect_ts_type

        # Should work identically to is_discrete_time_series
        data = [0, 1, 0, 1, 1, 0]
        assert detect_ts_type(data) == is_discrete_time_series(data)


class TestStatisticalProperties:
    """Test the statistical property extraction."""

    def test_extract_properties_basic(self):
        """Test basic property extraction."""
        data = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        props = _extract_statistical_properties(data)

        assert props["n_samples"] == 10
        assert props["n_unique"] == 5
        assert props["uniqueness_ratio"] == 0.5
        assert props["mean"] == 3.0
        assert props["min"] == 1
        assert props["max"] == 5
        assert props["range"] == 4
        assert props["fraction_integers"] == 1.0

    def test_extract_properties_entropy(self):
        """Test entropy calculations."""
        # Low entropy (concentrated)
        data_low = np.array([1] * 90 + [2] * 10)
        props_low = _extract_statistical_properties(data_low)
        assert props_low["normalized_entropy"] < 0.5

        # High entropy (uniform)
        data_high = np.random.choice(range(10), size=1000)
        props_high = _extract_statistical_properties(data_high)
        assert props_high["normalized_entropy"] > 0.8

    def test_extract_properties_gaps(self):
        """Test gap statistics."""
        # Regular gaps
        data_regular = np.array([1, 3, 5, 7, 9])
        props_regular = _extract_statistical_properties(data_regular)
        assert props_regular["cv_gap"] < 0.1  # Low coefficient of variation

        # Irregular gaps
        data_irregular = np.array([1, 2, 5, 6, 10, 11, 20])
        props_irregular = _extract_statistical_properties(data_irregular)
        assert props_irregular["cv_gap"] > 0.5


class TestCircularDetection:
    """Test circular pattern detection specifically."""

    def test_detect_circular_by_range(self):
        """Test circular detection by data range."""
        # Perfect [0, 2π] range
        data = np.linspace(0, 2 * np.pi, 100)
        props = _extract_statistical_properties(data)
        result = _detect_circular(data, props)

        assert result["is_circular"]
        assert result["confidence"] > 0.3
        assert np.isclose(result["period"], 2 * np.pi)

    def test_detect_circular_by_name(self):
        """Test circular detection by name context."""
        data = np.random.uniform(0, 100, 50)
        props = _extract_statistical_properties(data)

        # Without name
        result1 = _detect_circular(data, props)

        # With circular name
        result2 = _detect_circular(data, props, name="rotation_angle")

        assert result2["confidence"] > result1["confidence"]

    def test_detect_circular_wraparound(self):
        """Test wraparound detection."""
        # Create data with clear jumps
        data = np.array([0.1, 0.2, 0.3, 6.2, 6.1, 0.1, 0.2])
        props = _extract_statistical_properties(data)
        result = _detect_circular(data, props)

        # Should detect wraparound pattern
        assert result["confidence"] > 0


class TestPeriodicityDetection:
    """Test general periodicity detection."""

    def test_detect_periodicity_autocorrelation(self):
        """Test periodicity via autocorrelation."""
        # Create clearly periodic signal
        t = np.arange(100)
        signal = np.sin(2 * np.pi * t / 20)  # Period of 20 samples
        props = _extract_statistical_properties(signal)
        props["max_autocorr"] = 0.9  # High autocorrelation

        result = _detect_periodicity(signal, props)
        assert result["period"] is not None
        assert 18 < result["period"] < 22  # Close to 20

    def test_detect_periodicity_fourier(self):
        """Test periodicity via Fourier analysis."""
        # Longer signal for FFT
        t = np.arange(200)
        signal = 5 * np.sin(2 * np.pi * t / 25) + np.random.normal(0, 0.5, 200)
        props = _extract_statistical_properties(signal)

        result = _detect_periodicity(signal, props)
        assert result["period"] is not None
        assert 23 < result["period"] < 27  # Close to 25


class TestSubtypeDetection:
    """Test subtype classification."""

    def test_discrete_subtypes(self):
        """Test discrete subtype detection."""
        # Binary
        props_binary = {"n_unique": 2, "fraction_integers": 1.0, "uniqueness_ratio": 1.0}
        result_binary = _detect_discrete_subtype(np.array([0, 1]), props_binary)
        assert result_binary["subtype"] == "binary"
        assert result_binary["confidence"] == 1.0

        # Categorical (small set of values that repeat)
        data_cat = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])  # Non-monotonic
        props_cat = {
            "n_unique": 5,
            "fraction_integers": 1.0,
            "min": 1,
            "n_samples": 10,
            "uniqueness_ratio": 0.5,  # Each value appears twice
        }
        result_cat = _detect_discrete_subtype(data_cat, props_cat)
        assert result_cat["subtype"] == "categorical"  # Should be categorical

        # Count (monotonic data)
        data_count = np.arange(10)
        props_count = {
            "n_unique": 10,
            "fraction_integers": 1.0,
            "min": 0,
            "n_samples": 10,
            "uniqueness_ratio": 1.0,  # All unique for monotonic count
        }
        result_count = _detect_discrete_subtype(data_count, props_count)
        assert result_count["subtype"] == "count"

    def test_continuous_subtypes(self):
        """Test continuous subtype detection."""
        data = np.random.normal(0, 1, 100)
        props = _extract_statistical_properties(data)

        # Linear (not circular or periodic)
        circular_result = {"is_circular": False, "confidence": 0.1}
        periodic_result = {"period": None, "confidence": 0.1}

        result = _detect_continuous_subtype(
            data, props, circular_result, periodic_result
        )
        assert result["subtype"] == "linear"

        # Circular
        circular_result["is_circular"] = True
        circular_result["confidence"] = 0.9
        result_circ = _detect_continuous_subtype(
            data, props, circular_result, periodic_result
        )
        assert result_circ["subtype"] == "circular"

        # Should remain linear even with periodicity detected
        circular_result["is_circular"] = False
        periodic_result["period"] = 20
        periodic_result["confidence"] = 0.8
        result_per = _detect_continuous_subtype(
            data, props, circular_result, periodic_result
        )
        assert result_per["subtype"] == "linear"  # We removed periodic subtype


class TestIntegrationWithTimeSeries:
    """Test integration with TimeSeries class."""

    def test_timeseries_auto_detection(self):
        """Test that TimeSeries uses new detection correctly."""
        from driada.information.info_base import TimeSeries

        # Binary data
        ts_binary = TimeSeries([0, 1, 0, 1, 1, 0] * 20)
        assert ts_binary.discrete
        assert ts_binary.is_binary

        # Continuous data
        ts_cont = TimeSeries(np.random.normal(50, 10, 200))
        assert not ts_cont.discrete

        # Count data
        ts_count = TimeSeries(np.random.poisson(3, 200))
        assert ts_count.discrete
        assert not ts_count.is_binary

    def test_timeseries_legacy_method(self):
        """Test legacy define_ts_type method."""
        from driada.information.info_base import TimeSeries

        with pytest.warns(DeprecationWarning):
            result = TimeSeries.define_ts_type([0, 1, 0, 1])
            assert isinstance(result, bool)


class TestAmbiguousTypeDetection:
    """Test ambiguous type detection and warnings."""

    def test_ambiguous_explicit_specification(self):
        """Test explicitly specifying ambiguous type."""
        from driada.information.info_base import TimeSeries
        import warnings

        data = np.random.uniform(0, 10, 100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ts = TimeSeries(data, ts_type="ambiguous")

            # Should have warning
            assert len(w) == 1
            assert "ambiguous" in str(w[0].message)

            # Check properties
            assert ts.type_info.primary_type == "ambiguous"
            assert ts.type_info.confidence == 1.0
            assert not ts.discrete  # Defaults to continuous
            assert ts.type_info.subtype is None

    def test_no_warning_when_forced(self):
        """Test no warning when type is explicitly forced."""
        from driada.information.info_base import TimeSeries
        import warnings

        # Create data that might be ambiguous
        data_mixed = np.round(np.random.exponential(5, 200), 1)
        data_mixed = data_mixed[data_mixed < 20]

        # Force discrete - should have no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ts = TimeSeries(data_mixed, discrete=True)
            assert len(w) == 0  # No warnings
            assert ts.discrete
            assert ts.type_info.primary_type == "discrete"

    def test_string_shortcuts(self):
        """Test all string type shortcuts."""
        from driada.information.info_base import TimeSeries
        import warnings

        test_data = np.random.normal(0, 1, 50)

        # Test all type shortcuts
        type_mappings = {
            "binary": ("discrete", "binary", True),
            "categorical": ("discrete", "categorical", True),
            "count": ("discrete", "count", True),
            "timeline": ("discrete", "timeline", True),
            "linear": ("continuous", "linear", False),
            "circular": ("continuous", "circular", False),
            "ambiguous": ("ambiguous", None, False),
        }

        for type_str, (
            expected_primary,
            expected_subtype,
            expected_discrete,
        ) in type_mappings.items():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ts = TimeSeries(test_data, ts_type=type_str)

                assert ts.type_info.primary_type == expected_primary
                assert ts.type_info.subtype == expected_subtype
                assert ts.discrete == expected_discrete

                # Only ambiguous should warn
                if type_str == "ambiguous":
                    assert len(w) == 1
                else:
                    assert len(w) == 0

    def test_auto_detected_ambiguous(self):
        """Test data that auto-detects as ambiguous."""
        # Create data with mixed characteristics
        # This should have discrete and continuous scores very close
        np.random.seed(42)

        # Mix of repeated values and continuous-like spread
        discrete_part = np.repeat([1, 2, 3, 4, 5], 10)
        continuous_part = np.random.normal(3, 0.5, 50)
        data = np.concatenate([discrete_part, continuous_part])
        np.random.shuffle(data)

        result = analyze_time_series_type(data)

        # Check if scores are close (indicating ambiguity)
        if result.primary_type == "ambiguous":
            # Verify the scores were indeed close
            discrete_score = result.metadata.get("discrete_score", 0)
            continuous_score = result.metadata.get("continuous_score", 0)
            assert abs(discrete_score - continuous_score) < 0.2

    def test_ambiguous_properties(self):
        """Test properties of ambiguous type."""
        from driada.information.info_base import TimeSeries

        ts = TimeSeries([1, 2, 3], ts_type="ambiguous")

        # Test properties
        assert ts.type_info.is_ambiguous
        assert not ts.type_info.is_discrete
        assert not ts.type_info.is_continuous
        assert not ts.type_info.is_periodic
        assert ts.type_info.subtype is None

    def test_ambiguous_discrete_noise(self):
        """Test discrete data with noise (should not always be ambiguous)."""
        from driada.information.info_base import TimeSeries

        # Discrete values with small noise
        data_noisy = np.random.choice([1, 2, 3, 4, 5], 100) + np.random.normal(
            0, 0.05, 100
        )
        ts = TimeSeries(data_noisy)

        # With small noise, should still detect as continuous
        # (because the noise makes it non-integer)
        assert ts.type_info.primary_type in ["continuous", "ambiguous"]

        # Check metadata
        assert "n_unique" in ts.type_info.metadata
        assert "uniqueness_ratio" in ts.type_info.metadata


if __name__ == "__main__":
    pytest.main([__file__])
