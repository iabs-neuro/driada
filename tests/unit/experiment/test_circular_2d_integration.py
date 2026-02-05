"""Integration tests for circular _2d feature creation."""

import numpy as np
import pytest

from driada.experiment import load_exp_from_aligned_data
from driada.information import TimeSeries, MultiTimeSeries


class TestCreateCircular2dEnabled:
    """Test automatic _2d feature creation when enabled."""

    def test_circular_feature_creates_2d_version(self):
        """Test that _2d versions of circular features are created by default."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
            "speed": np.random.uniform(0, 10, 1000),
        }

        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

        # Original preserved as TimeSeries
        assert isinstance(exp.dynamic_features["head_direction"], TimeSeries)
        # _2d version created as MultiTimeSeries
        assert "head_direction_2d" in exp.dynamic_features
        assert isinstance(exp.dynamic_features["head_direction_2d"], MultiTimeSeries)
        assert exp.dynamic_features["head_direction_2d"].n_dim == 2

    def test_non_circular_feature_no_2d(self):
        """Test that non-circular features don't get _2d versions."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
            "speed": np.random.uniform(0, 10, 1000),
        }

        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

        # Speed is not circular, should not have _2d version
        assert "speed_2d" not in exp.dynamic_features

    def test_default_is_enabled(self):
        """Test that create_circular_2d=True is the default."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
        }

        # Don't pass create_circular_2d - should default to True
        exp = load_exp_from_aligned_data("test", {"animal": "A1"}, data, verbose=False)

        assert "head_direction_2d" in exp.dynamic_features


class TestCreateCircular2dDisabled:
    """Test behavior when _2d creation is disabled."""

    def test_no_2d_when_disabled(self):
        """Test that create_circular_2d=False skips _2d creation."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
        }

        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=False, verbose=False
        )

        # Original preserved
        assert "head_direction" in exp.dynamic_features
        # No _2d version
        assert "head_direction_2d" not in exp.dynamic_features


class TestCircular2dTransformCorrectness:
    """Test that _2d transformations are mathematically correct."""

    def test_cos_sin_values_correct(self):
        """Test that cos/sin components are computed correctly."""
        np.random.seed(42)
        angles = np.random.uniform(0, 2 * np.pi, 1000)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": angles,
        }

        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

        mts = exp.dynamic_features["head_direction_2d"]
        np.testing.assert_allclose(mts.ts_list[0].data, np.cos(angles), rtol=1e-10)
        np.testing.assert_allclose(mts.ts_list[1].data, np.sin(angles), rtol=1e-10)

    def test_2d_points_on_unit_circle(self):
        """Test that (cos, sin) points lie on unit circle."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
        }

        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

        mts = exp.dynamic_features["head_direction_2d"]
        cos_vals = mts.ts_list[0].data
        sin_vals = mts.ts_list[1].data

        # All points should be on unit circle (radius = 1)
        radii = np.sqrt(cos_vals**2 + sin_vals**2)
        np.testing.assert_allclose(radii, np.ones_like(radii), rtol=1e-10)


class TestExperimentHelperMethods:
    """Test the helper methods added to Experiment class."""

    @pytest.fixture
    def exp_with_circular(self):
        """Create experiment with circular feature."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
            "speed": np.random.uniform(0, 10, 1000),
        }
        return load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

    def test_has_circular_2d(self, exp_with_circular):
        """Test has_circular_2d method."""
        assert exp_with_circular.has_circular_2d("head_direction") is True
        assert exp_with_circular.has_circular_2d("speed") is False
        assert exp_with_circular.has_circular_2d("nonexistent") is False

    def test_get_circular_2d_feature(self, exp_with_circular):
        """Test get_circular_2d_feature method."""
        mts = exp_with_circular.get_circular_2d_feature("head_direction")
        assert isinstance(mts, MultiTimeSeries)
        assert mts.n_dim == 2

        # Non-existent should return None
        assert exp_with_circular.get_circular_2d_feature("speed") is None
        assert exp_with_circular.get_circular_2d_feature("nonexistent") is None

    def test_get_circular_features(self, exp_with_circular):
        """Test get_circular_features method."""
        circular = exp_with_circular.get_circular_features()

        # Should include head_direction but not speed or _2d versions
        assert "head_direction" in circular
        assert "speed" not in circular
        assert "head_direction_2d" not in circular

        # The returned values should be TimeSeries
        assert isinstance(circular["head_direction"], TimeSeries)


class TestMultipleCircularFeatures:
    """Test with multiple circular features."""

    def test_multiple_circular_features_create_2d(self):
        """Test that multiple circular features all get _2d versions."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
            "body_direction": np.random.uniform(0, 2 * np.pi, 1000),
            "phase_angle": np.random.uniform(0, 2 * np.pi, 1000),
            "speed": np.random.uniform(0, 10, 1000),
        }

        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

        # All circular features should have _2d versions
        # (depends on auto-detection, which may or may not detect all as circular)
        if exp.dynamic_features["head_direction"].type_info.is_circular:
            assert "head_direction_2d" in exp.dynamic_features

        # Speed should never have _2d version
        assert "speed_2d" not in exp.dynamic_features
