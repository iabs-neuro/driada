"""Tie-breaking noise added at experiment construction is reproducible with a seed."""

import numpy as np

from driada.experiment import load_exp_from_aligned_data
from driada.experiment.signal_preprocessing import calcium_preprocessing
from driada.information.info_base import TimeSeries, aggregate_multiple_ts


def _data():
    rng = np.random.default_rng(0)
    # Many exact zeros after clipping, as in real calcium traces
    return {
        "calcium": rng.normal(size=(4, 600)),
        "x": rng.random(600),
        "y": rng.random(600),
    }


def _build(seed):
    return load_exp_from_aligned_data(
        data_source="T",
        exp_params={"name": "t"},
        data=_data(),
        static_features={"fps": 20.0},
        aggregate_features={("x", "y"): "position"},
        verbose=False,
        seed=seed,
    )


def test_calcium_preprocessing_seed_is_reproducible():
    ca = np.zeros(500)
    assert np.array_equal(calcium_preprocessing(ca, seed=1), calcium_preprocessing(ca, seed=1))
    assert not np.array_equal(calcium_preprocessing(ca, seed=1), calcium_preprocessing(ca, seed=2))


def test_calcium_preprocessing_clips_and_keeps_input():
    ca = np.array([1.0, -0.5, 2.0, 0.5])
    out = calcium_preprocessing(ca, seed=0)
    assert ca[1] == -0.5
    assert 0 <= out[1] < 1e-7
    assert (out > 0).all()


def test_experiment_seed_gives_identical_traces():
    a, b, c = _build(7), _build(7), _build(8)
    assert np.array_equal(a.calcium.data, b.calcium.data)
    assert not np.array_equal(a.calcium.data, c.calcium.data)
    assert np.array_equal(
        a.dynamic_features["position"].data, b.dynamic_features["position"].data
    )


def test_neurons_get_different_noise():
    exp = _build(7)
    noise = exp.calcium.data - np.clip(_data()["calcium"], 0, None)
    assert not np.array_equal(noise[0], noise[1])


def test_aggregate_multiple_ts_seed():
    ts1 = TimeSeries(np.zeros(300), discrete=False)
    ts2 = TimeSeries(np.ones(300), discrete=False)
    m1 = aggregate_multiple_ts(ts1, ts2, seed=3)
    m2 = aggregate_multiple_ts(ts1, ts2, seed=3)
    assert np.array_equal(m1.data, m2.data)
