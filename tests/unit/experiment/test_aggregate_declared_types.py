"""Components of an aggregated feature keep the type declared for them."""

import numpy as np
import pytest

from driada.experiment import load_exp_from_aligned_data
from driada.information.info_base import TimeSeries, aggregate_multiple_ts
from driada.intense.representations import substitute_by_type


def _data():
    rng = np.random.default_rng(0)
    # x covers [0, 2*pi), which auto-detection takes for an angle
    return {
        "calcium": rng.normal(size=(3, 5000)),
        "x": rng.uniform(0, 2 * np.pi, 5000),
        "y": np.cumsum(rng.normal(size=5000)),
    }


def _build(feature_types=None):
    return load_exp_from_aligned_data(
        data_source="T",
        exp_params={"name": "t"},
        data=_data(),
        static_features={"fps": 20.0},
        aggregate_features={("x", "y"): "place"},
        feature_types=feature_types,
        verbose=False,
        seed=0,
    )


def _component_circular(exp):
    return [ts.type_info.is_circular for ts in exp.dynamic_features["place"].ts_list]


def test_undeclared_components_are_auto_detected():
    assert _component_circular(_build()) == [True, False]


def test_declared_component_type_reaches_the_aggregate():
    exp = _build({"x": "linear", "y": "linear"})
    assert _component_circular(exp) == [False, False]
    assert not exp.dynamic_features["x"].type_info.is_circular


def test_declared_linear_place_gets_quadratic_representation():
    assert substitute_by_type(["place"], _build())[0] == ["place"]
    declared = _build({"x": "linear", "y": "linear"})
    assert substitute_by_type(["place"], declared)[0] == ["place_quad"]


def test_aggregate_multiple_ts_keeps_declared_types():
    rng = np.random.default_rng(1)
    angle_like = TimeSeries(rng.uniform(0, 2 * np.pi, 5000), discrete=False)
    other = TimeSeries(np.cumsum(rng.normal(size=5000)), discrete=False)

    auto = aggregate_multiple_ts(angle_like, other, name="p", seed=0)
    assert auto.ts_list[0].type_info.is_circular

    declared = aggregate_multiple_ts(
        angle_like, other, name="p", seed=0, ts_types=["linear", None]
    )
    assert not declared.ts_list[0].type_info.is_circular
    assert np.array_equal(auto.data, declared.data)


def test_aggregate_multiple_ts_rejects_wrong_number_of_types():
    ts = TimeSeries(np.random.default_rng(2).normal(size=100), discrete=False)
    with pytest.raises(ValueError, match="ts_types"):
        aggregate_multiple_ts(ts, ts, ts_types=["linear"])


def test_runner_configs_declare_position_linear():
    from tools.selectivity_dynamics.filters import EXPERIMENT_CONFIGS

    for name, cfg in EXPERIMENT_CONFIGS.items():
        for components in cfg["aggregate_features"]:
            if set(components) <= {"x", "y"}:
                assert cfg["feature_types"]["x"] == "linear", name
                assert cfg["feature_types"]["y"] == "linear", name
