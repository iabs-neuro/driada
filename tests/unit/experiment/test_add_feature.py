"""Tests for Experiment.add_feature() method."""

import pytest
import numpy as np
import warnings

from driada.experiment.exp_base import Experiment
from driada.information.info_base import TimeSeries, MultiTimeSeries


@pytest.fixture
def basic_exp():
    """Create a minimal experiment for testing add_feature."""
    calcium = np.random.RandomState(42).rand(5, 500)
    feat = TimeSeries(np.random.RandomState(42).rand(500), name="speed")
    return Experiment(
        signature="test_add_feat",
        calcium=calcium,
        spikes=None,
        exp_identificators={},
        static_features={"fps": 20.0},
        dynamic_features={"speed": feat},
    )


class TestAddFeature:

    def test_add_discrete_feature(self, basic_exp):
        """Added discrete feature is accessible as attribute and dict entry."""
        data = np.random.RandomState(0).randint(0, 3, 500).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            basic_exp.add_feature("zone", data, ts_type="discrete")

        assert hasattr(basic_exp, "zone")
        assert "zone" in basic_exp.dynamic_features
        ts = basic_exp.dynamic_features["zone"]
        assert isinstance(ts, TimeSeries)
        assert ts.discrete

    def test_add_continuous_feature_from_ndarray(self, basic_exp):
        """Raw ndarray is wrapped in TimeSeries automatically."""
        data = np.random.RandomState(1).rand(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            basic_exp.add_feature("acceleration", data)

        ts = basic_exp.dynamic_features["acceleration"]
        assert isinstance(ts, TimeSeries)
        np.testing.assert_array_equal(ts.data, data)

    def test_add_timeseries_directly(self, basic_exp):
        """Passing a TimeSeries object is accepted without re-wrapping."""
        ts_in = TimeSeries(np.random.RandomState(2).rand(500), name="acc")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            basic_exp.add_feature("acceleration", ts_in)

        assert basic_exp.dynamic_features["acceleration"] is ts_in

    def test_reject_protected_attribute_name(self, basic_exp):
        """Protected attribute names raise ValueError."""
        data = np.random.RandomState(3).rand(500)
        with pytest.raises(ValueError, match="protected attribute"):
            basic_exp.add_feature("calcium", data)

    def test_reject_wrong_length(self, basic_exp):
        """Feature with wrong number of timepoints raises ValueError."""
        data = np.random.RandomState(4).rand(100)  # 100 != 500
        with pytest.raises(ValueError, match="500"):
            basic_exp.add_feature("bad_feat", data)

    def test_shuffle_mask_applied(self, basic_exp):
        """Shuffle mask edges are set to False."""
        data = np.random.RandomState(5).rand(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            basic_exp.add_feature("test_mask", data)

        ts = basic_exp.dynamic_features["test_mask"]
        # MIN_FEAT_SHIFT_SEC=2.0, fps=20 → shift=40
        assert not ts.shuffle_mask[0]
        assert not ts.shuffle_mask[39]
        assert not ts.shuffle_mask[-1]
        assert not ts.shuffle_mask[-40]
        # Middle should be True
        assert ts.shuffle_mask[250]

    def test_data_hashes_updated(self, basic_exp):
        """_data_hashes includes the new feature after add_feature."""
        data = np.random.RandomState(6).rand(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            basic_exp.add_feature("new_feat", data)

        assert "new_feat" in basic_exp._data_hashes["calcium"]

    def test_warns_about_cached_stats(self, basic_exp):
        """add_feature issues a warning about cached stats."""
        data = np.random.RandomState(7).rand(500)
        with pytest.warns(UserWarning, match="Cached INTENSE stats"):
            basic_exp.add_feature("warn_feat", data)

    def test_2d_array_creates_multitimeseries(self, basic_exp):
        """2D ndarray is wrapped in MultiTimeSeries."""
        data = np.random.RandomState(8).rand(3, 500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            basic_exp.add_feature("multi_feat", data)

        ts = basic_exp.dynamic_features["multi_feat"]
        assert isinstance(ts, MultiTimeSeries)

    def test_non_string_name_raises_type_error(self, basic_exp):
        """Non-string feature name raises TypeError."""
        data = np.random.RandomState(9).rand(500)
        with pytest.raises(TypeError, match="string"):
            basic_exp.add_feature(123, data)
