"""Tests for circular feature transformation."""

import numpy as np
import pytest

from driada.information.circular_transform import (
    circular_to_cos_sin,
    cos_sin_to_circular,
    detect_circular_period,
    normalize_to_radians,
    get_circular_2d_name,
    is_circular_2d_feature,
)
from driada.information import TimeSeries, MultiTimeSeries


class TestCircularToCosIn:
    """Tests for circular_to_cos_sin transformation."""

    def test_radians_basic(self):
        """Test transformation of radian data."""
        angles = np.linspace(0, 2 * np.pi, 100)
        mts = circular_to_cos_sin(angles, period=2 * np.pi, name="test_2d")

        assert isinstance(mts, MultiTimeSeries)
        assert mts.n_dim == 2
        np.testing.assert_allclose(mts.ts_list[0].data, np.cos(angles), rtol=1e-10)
        np.testing.assert_allclose(mts.ts_list[1].data, np.sin(angles), rtol=1e-10)

    def test_degrees(self):
        """Test transformation of degree data."""
        angles_deg = np.linspace(0, 360, 100)
        mts = circular_to_cos_sin(angles_deg, period=360, name="test_2d")

        angles_rad = np.deg2rad(angles_deg)
        np.testing.assert_allclose(mts.ts_list[0].data, np.cos(angles_rad), rtol=1e-10)
        np.testing.assert_allclose(mts.ts_list[1].data, np.sin(angles_rad), rtol=1e-10)

    def test_name_assignment(self):
        """Test that name is correctly assigned."""
        angles = np.linspace(0, 2 * np.pi, 100)
        mts = circular_to_cos_sin(angles, name="headdirection_2d")
        assert mts.name == "headdirection_2d"

    def test_auto_period_detection(self):
        """Test auto period detection when period is None."""
        # Radians
        angles_rad = np.random.uniform(0, 2 * np.pi, 100)
        mts = circular_to_cos_sin(angles_rad, period=None, name="test")
        # Should detect radians and produce correct transformation
        np.testing.assert_allclose(mts.ts_list[0].data, np.cos(angles_rad), rtol=1e-10)

    def test_timeseries_input(self):
        """Test transformation with TimeSeries input."""
        angles = np.linspace(0, 2 * np.pi, 100)
        ts = TimeSeries(angles, discrete=False, ts_type="circular", name="hd")

        mts = circular_to_cos_sin(ts)
        assert isinstance(mts, MultiTimeSeries)
        assert mts.name == "hd_2d"  # Auto-named with _2d suffix
        assert mts.ts_list[0].name == "cos"
        assert mts.ts_list[1].name == "sin"

    def test_timeseries_input_with_period(self):
        """Test that period from TimeSeries type_info is used."""
        angles = np.linspace(0, 2 * np.pi, 100)
        ts = TimeSeries(angles, discrete=False, ts_type="circular", name="hd")

        mts = circular_to_cos_sin(ts)
        np.testing.assert_allclose(mts.ts_list[0].data, np.cos(angles), rtol=1e-10)


class TestRoundtripTransformation:
    """Test that transformation is reversible."""

    def test_roundtrip_radians(self):
        """Test roundtrip for radian data."""
        original = np.random.uniform(0, 2 * np.pi, 1000)
        mts = circular_to_cos_sin(original, period=2 * np.pi)

        recovered = cos_sin_to_circular(mts.ts_list[0].data, mts.ts_list[1].data)
        np.testing.assert_allclose(recovered, original, rtol=1e-10)

    def test_roundtrip_degrees(self):
        """Test roundtrip for degree data."""
        original = np.random.uniform(0, 360, 1000)
        mts = circular_to_cos_sin(original, period=360)

        recovered = cos_sin_to_circular(
            mts.ts_list[0].data, mts.ts_list[1].data, period=360
        )
        np.testing.assert_allclose(recovered, original, rtol=1e-9)

    def test_roundtrip_negative_angles(self):
        """Test roundtrip with negative angles (gets normalized to [0, 2pi))."""
        original = np.random.uniform(-np.pi, np.pi, 1000)
        mts = circular_to_cos_sin(original, period=2 * np.pi)

        recovered = cos_sin_to_circular(mts.ts_list[0].data, mts.ts_list[1].data)
        # Original is in [-pi, pi], recovered is in [0, 2pi)
        # Normalize original to [0, 2pi) for comparison
        original_normalized = original % (2 * np.pi)
        np.testing.assert_allclose(recovered, original_normalized, rtol=1e-10)


class TestDetectCircularPeriod:
    """Tests for period auto-detection."""

    def test_radians_detection(self):
        """Test detection of radian data."""
        rad_data = np.random.uniform(0, 2 * np.pi, 100)
        period = detect_circular_period(rad_data)
        assert abs(period - 2 * np.pi) < 0.1

    def test_degrees_detection(self):
        """Test detection of degree data."""
        deg_data = np.random.uniform(0, 360, 100)
        period = detect_circular_period(deg_data)
        assert abs(period - 360) < 1

    def test_small_range_defaults_to_radians(self):
        """Test that small range data defaults to radians."""
        small_data = np.random.uniform(0, 1, 100)
        period = detect_circular_period(small_data)
        assert abs(period - 2 * np.pi) < 0.1


class TestNormalizeToRadians:
    """Tests for normalization function."""

    def test_degrees_to_radians(self):
        """Test conversion from degrees to radians."""
        deg_data = np.array([0, 90, 180, 270, 360])
        rad_data = normalize_to_radians(deg_data, 360)
        expected = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        np.testing.assert_allclose(rad_data, expected)

    def test_radians_unchanged(self):
        """Test that radian data is unchanged."""
        rad_data = np.linspace(0, 2 * np.pi, 100)
        result = normalize_to_radians(rad_data, 2 * np.pi)
        np.testing.assert_allclose(result, rad_data)

    def test_none_period_unchanged(self):
        """Test that None period returns data unchanged."""
        data = np.linspace(0, 2 * np.pi, 100)
        result = normalize_to_radians(data, None)
        np.testing.assert_allclose(result, data)


class TestCosSinToCircular:
    """Tests for inverse transformation."""

    def test_basic_inverse(self):
        """Test basic inverse transformation."""
        original = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        cos_data = np.cos(original)
        sin_data = np.sin(original)

        recovered = cos_sin_to_circular(cos_data, sin_data)
        np.testing.assert_allclose(recovered, original, atol=1e-10)

    def test_inverse_with_different_period(self):
        """Test inverse transformation with degree period."""
        original_deg = np.array([0, 90, 180, 270])
        original_rad = np.deg2rad(original_deg)
        cos_data = np.cos(original_rad)
        sin_data = np.sin(original_rad)

        recovered = cos_sin_to_circular(cos_data, sin_data, period=360)
        np.testing.assert_allclose(recovered, original_deg, atol=1e-9)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_circular_2d_name(self):
        """Test _2d naming helper."""
        assert get_circular_2d_name("headdirection") == "headdirection_2d"
        assert get_circular_2d_name("angle") == "angle_2d"
        assert get_circular_2d_name("bodydirection") == "bodydirection_2d"

    def test_is_circular_2d_feature(self):
        """Test _2d feature detection."""
        assert is_circular_2d_feature("headdirection_2d") is True
        assert is_circular_2d_feature("headdirection") is False
        assert is_circular_2d_feature("speed") is False
        assert is_circular_2d_feature("angle_2d") is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_boundary_values(self):
        """Test handling of values at and near boundaries."""
        # Values very close to 0 and 2pi
        angles = np.array([0.001, np.pi, 2 * np.pi - 0.001])
        mts = circular_to_cos_sin(angles, period=2 * np.pi)

        # Should handle without issues
        assert mts.n_dim == 2
        assert len(mts.ts_list[0].data) == 3

    def test_single_value_fails(self):
        """Test that single value raises error (TimeSeries requires at least 2 points)."""
        angles = np.array([np.pi])
        with pytest.raises(ValueError, match="at least 2 points"):
            circular_to_cos_sin(angles, period=2 * np.pi)

    def test_large_array(self):
        """Test with large array."""
        n = 100000
        angles = np.random.uniform(0, 2 * np.pi, n)
        mts = circular_to_cos_sin(angles, period=2 * np.pi)

        assert mts.ts_list[0].data.shape == (n,)
        assert mts.ts_list[1].data.shape == (n,)
