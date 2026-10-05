"""Tests for the place-vs-zone rules of the analysis-tool filters."""

import copy

import numpy as np
import pytest

from driada.information.gcmi import mi_model_gd
from tools.selectivity_dynamics.filters import (
    GENERAL_PRIORITY_RULES,
    _class_conditional_mi,
    build_priority_filter,
    compose_filters,
    get_filter_for_experiment,
    nof_filter,
    spatial_filter,
    zone_share_filter,
)

FPS = 20
ARENA = 50.0
EVENT_RATE = 0.08  # events per second
ZONE_CELL, FAR_CELL, EDGE_CELL = 0, 1, 2


def _trajectory(rng, n_frames):
    """Smooth random walk reflected inside a square arena."""
    pos = np.empty((2, n_frames))
    p, v = np.full(2, ARENA / 2), np.zeros(2)
    for t in range(n_frames):
        v = 0.95 * v + rng.standard_normal(2) * 0.25
        p = p + v
        for k in range(2):
            if p[k] < 0:
                p[k], v[k] = -p[k], -v[k]
            if p[k] > ARENA:
                p[k], v[k] = 2 * ARENA - p[k], -v[k]
        pos[:, t] = p
    return pos


def _calcium(rng, tuning, tuned_fraction=0.9):
    """Poisson events following the tuning, convolved with a calcium transient."""
    n_frames = tuning.size
    rate = EVENT_RATE * (1 - tuned_fraction) + EVENT_RATE * tuned_fraction * tuning / tuning.mean()
    events = rng.poisson(np.minimum(rate, 3.0) / FPS)
    t = np.arange(6 * FPS) / FPS
    kernel = np.exp(-t / 1.2) - np.exp(-t / 0.1)
    amplitudes = events * rng.lognormal(0, 0.5, n_frames)
    return np.convolve(amplitudes, kernel)[:n_frames] + rng.standard_normal(n_frames) * 0.1


@pytest.fixture(scope="module")
def session():
    """A walls cell, a place field far from the walls and a field centred beyond the edge."""
    rng = np.random.default_rng(1)
    pos = _trajectory(rng, 15 * 60 * FPS)
    x, y = pos
    walls = (np.minimum(x, ARENA - x) < 5) | (np.minimum(y, ARENA - y) < 5)
    tuning = {
        ZONE_CELL: walls.astype(float),
        FAR_CELL: np.exp(-((x - 25) ** 2 + (y - 28) ** 2) / (2 * 4 ** 2)),
        EDGE_CELL: np.exp(-((x + 15) ** 2 + (y - 25) ** 2) / (2 * 10 ** 2)),
    }
    return {
        "calcium_data": {nid: _calcium(rng, tun) for nid, tun in tuning.items()},
        "feature_data": {"walls": walls.astype(float)},
        "position_data": pos,
        "discrete_place_features": ["walls"],
        "place_feat_name": "place",
        "fps": FPS,
    }


def _run(filter_func, session, **overrides):
    sels = {nid: ["place", "walls", "speed"] for nid in session["calcium_data"]}
    decisions = {nid: {} for nid in sels}
    renames = {nid: {} for nid in sels}
    filter_func(sels, decisions, renames, **{**session, **overrides})
    return sels, decisions, renames


@pytest.fixture(scope="module")
def default_result(session):
    return _run(get_filter_for_experiment("NOF", zone_rule="information_share"), session)


def test_estimator_matches_library():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 16, 3000)
    labels[labels == 7] = 6  # an empty class
    x = rng.standard_normal(3000) + 0.3 * (labels % 3)
    with pytest.warns(RuntimeWarning, match="no samples"):
        expected = mi_model_gd(x[None], labels, Ym=16, biascorrect=True)
    got = _class_conditional_mi(x[None], labels[None], 16)[0]
    assert got == pytest.approx(expected, abs=1e-9)


def test_zone_cell_goes_to_zone(default_result):
    sels, decisions, renames = default_result
    assert decisions[ZONE_CELL] == {("place", "walls"): 1}
    assert sels[ZONE_CELL] == ["place", "walls", "speed"]
    assert renames[ZONE_CELL] == {}


@pytest.mark.parametrize("nid", [FAR_CELL, EDGE_CELL])
def test_place_field_stays_place(default_result, nid):
    sels, decisions, renames = default_result
    assert decisions[nid] == {("place", "walls"): 0}
    assert sels[nid] == ["place", "walls", "speed"]
    assert renames[nid] == {}


def test_information_share_rule_is_zone_share_filter(session, default_result):
    assert default_result == _run(zone_share_filter, session)


def test_top_activity_rule_is_spatial_filter(session, default_result):
    result = _run(get_filter_for_experiment("NOF", zone_rule="top_activity"), session)
    previous_chain = compose_filters(
        build_priority_filter(GENERAL_PRIORITY_RULES), nof_filter, spatial_filter
    )
    assert result == _run(previous_chain, session)

    sels, decisions, renames = result
    assert sels[ZONE_CELL] == ["speed", "place-walls"]
    assert renames[ZONE_CELL] == {"place-walls": ("place", "walls")}
    assert decisions[FAR_CELL] == {("place", "walls"): 0}
    assert result != default_result


def test_unknown_zone_rule_raises():
    with pytest.raises(ValueError, match="zone rule"):
        get_filter_for_experiment("NOF", zone_rule="unknown")


def test_decisions_are_reproducible(session, default_result):
    assert _run(get_filter_for_experiment("NOF", zone_rule="information_share"), session) == default_result


def test_delay_is_taken_from_intense_stats(session, default_result):
    delay = 37
    shifted = copy.copy(session)
    # Calcium lagging behaviour by `delay` frames.
    shifted["calcium_data"] = {
        nid: np.roll(ca, delay) for nid, ca in session["calcium_data"].items()
    }
    stats = {nid: {"walls": {"opt_delay": delay}} for nid in session["calcium_data"]}
    assert _run(zone_share_filter, shifted, cell_feat_stats=stats) == default_result


def test_threshold_is_configurable(session):
    _, decisions, _ = _run(zone_share_filter, session, zone_share_threshold=np.inf)
    assert decisions[ZONE_CELL] == {("place", "walls"): 0}


def test_losing_zone_is_skipped(session):
    sels = {ZONE_CELL: ["place", "walls", "corners"]}
    decisions = {ZONE_CELL: {("corners", "walls"): 0}}
    zone_share_filter(sels, decisions, {ZONE_CELL: {}}, **session)
    assert decisions[ZONE_CELL] == {("corners", "walls"): 0}


def test_without_data_pairs_stay_undecided():
    sels = {0: ["place", "walls"], 1: ["place", "speed"]}
    decisions = {0: {}, 1: {}}
    zone_share_filter(sels, decisions, {0: {}, 1: {}}, discrete_place_features=["walls"])
    assert decisions == {0: {("place", "walls"): 0.5}, 1: {}}
