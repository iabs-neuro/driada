"""Tests for the place-field rule of the analysis-tool place-vs-zone filters."""

import numpy as np
import pytest

from tools.selectivity_dynamics.filters import (
    DEFAULT_ZONE_RULE,
    _position_bins,
    get_filter_for_experiment,
    place_field_filter,
    place_field_in_zone,
    place_field_map,
)

FPS = 20
ARENA = 50.0
OBJECT_CELL, FAR_CELL, MANY_FIELDS_CELL, NEAR_CELL = 0, 1, 2, 3
OBJECT_XY, OBJECT_RADIUS = (15.0, 35.0), 4.0


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


def _field(x, y, centre, sigma=5.0):
    return np.exp(-((x - centre[0]) ** 2 + (y - centre[1]) ** 2) / (2 * sigma ** 2))


@pytest.fixture(scope="module")
def session():
    """Noisy activity following a field on the object, far from it, three comparable fields and one beside it."""
    rng = np.random.default_rng(3)
    pos = _trajectory(rng, 20 * 60 * FPS)
    x, y = pos
    zone = (x - OBJECT_XY[0]) ** 2 + (y - OBJECT_XY[1]) ** 2 < OBJECT_RADIUS ** 2
    centre = np.full(2, ARENA / 2)
    walls = (np.minimum(x, ARENA - x) < 6) | (np.minimum(y, ARENA - y) < 6)
    tuning = {
        OBJECT_CELL: _field(x, y, OBJECT_XY),
        FAR_CELL: _field(x, y, (38.0, 12.0)),
        MANY_FIELDS_CELL: (_field(x, y, OBJECT_XY) + 0.9 * _field(x, y, (38.0, 12.0))
                           + 0.9 * _field(x, y, (38.0, 38.0))),
        NEAR_CELL: _field(x, y, (OBJECT_XY[0] + 14.0, OBJECT_XY[1])),
    }
    return {
        "calcium_data": {nid: t + rng.standard_normal(t.size) * 0.3 for nid, t in tuning.items()},
        "feature_data": {"object1": zone.astype(float), "walls": walls.astype(float),
                         "center": (np.abs(pos - centre[:, None]).max(axis=0) < 12).astype(float)},
        "position_data": pos,
        "discrete_place_features": ["object1", "walls", "center"],
        "place_feat_name": "place",
    }


def _run(filter_func, session, features=("place", "object1", "speed"), **overrides):
    sels = {nid: list(features) for nid in session["calcium_data"]}
    decisions = {nid: {} for nid in sels}
    renames = {nid: {} for nid in sels}
    filter_func(sels, decisions, renames, **{**session, **overrides})
    return sels, decisions, renames


def test_place_field_is_default(session):
    assert DEFAULT_ZONE_RULE == "place_field"
    default = _run(get_filter_for_experiment("NOF"), session)
    explicit = _run(get_filter_for_experiment("NOF", zone_rule="place_field"), session)
    assert default == explicit == _run(place_field_filter, session)


def test_field_on_object_is_merged(session):
    sels, decisions, renames = _run(place_field_filter, session)
    assert sels[OBJECT_CELL] == ["speed", "place-object1"]
    assert renames[OBJECT_CELL] == {"place-object1": ("place", "object1")}
    assert decisions[OBJECT_CELL] == {}


@pytest.mark.parametrize("nid", [FAR_CELL, NEAR_CELL])
def test_field_elsewhere_stays_place(session, nid):
    sels, decisions, renames = _run(place_field_filter, session)
    assert sels[nid] == ["place", "object1", "speed"]
    assert decisions[nid] == {("place", "object1"): 0}
    assert renames[nid] == {}


def test_several_comparable_fields_stay_place(session):
    sels, decisions, _ = _run(place_field_filter, session)
    assert decisions[MANY_FIELDS_CELL] == {("place", "object1"): 0}
    # The other fields stop mattering once the main field may be a minority.
    sels, _, _ = _run(place_field_filter, session, main_field_threshold=0.3)
    assert "place-object1" in sels[MANY_FIELDS_CELL]


def test_decision_does_not_depend_on_time_in_zone(session):
    """A field in the centre is a centre cell and not a walls cell, however long the animal stays at the walls."""
    bins = _position_bins(session["position_data"])
    walls = session["feature_data"]["walls"] > 0.5
    centre = session["feature_data"]["center"] > 0.5
    x, y = session["position_data"]
    calcium = _field(x, y, (ARENA / 2, ARENA / 2))
    assert place_field_in_zone(calcium, centre, bins)[0]
    assert not place_field_in_zone(calcium, walls, bins)[0]
    # Keep every frame at the walls and one in ten elsewhere: the map is unchanged.
    keep = walls | (np.arange(walls.size) % 10 == 0)
    assert walls[keep].mean() > 0.5
    assert place_field_in_zone(calcium[keep], centre[keep], bins[keep])[0]
    assert not place_field_in_zone(calcium[keep], walls[keep], bins[keep])[0]


def test_one_zone_is_merged_and_the_rest_lose(session):
    """A field on the object inside the walls strip: the zone holding more of the field is merged."""
    x, y = session["position_data"]
    near_wall = (np.minimum(x, ARENA - x) < 22) | (np.minimum(y, ARENA - y) < 22)
    data = {**session, "feature_data": {**session["feature_data"], "walls": near_wall.astype(float)}}
    sels, decisions, renames = _run(place_field_filter, data, features=("place", "object1", "walls"))
    assert sels[OBJECT_CELL] == ["object1", "place-walls"]
    assert decisions[OBJECT_CELL] == {("place", "object1"): 0}


def test_delay_is_taken_from_intense_stats(session):
    shift = 12 * FPS
    data = {**session, "calcium_data": {nid: np.roll(c, shift) for nid, c in session["calcium_data"].items()}}
    sels, _, _ = _run(place_field_filter, data)
    assert "place-object1" not in sels[OBJECT_CELL]
    stats = {nid: {"object1": {"opt_delay": shift}} for nid in data["calcium_data"]}
    sels, _, _ = _run(place_field_filter, data, cell_feat_stats=stats)
    assert "place-object1" in sels[OBJECT_CELL]


def test_losing_zone_is_skipped(session):
    sels = {OBJECT_CELL: ["place", "object1", "objects"]}
    decisions = {OBJECT_CELL: {("objects", "object1"): 0}}
    renames = {OBJECT_CELL: {}}
    place_field_filter(sels, decisions, renames, **session)
    assert sels[OBJECT_CELL] == ["place", "object1", "objects"]
    assert ("place", "object1") not in decisions[OBJECT_CELL]


def test_without_data_pairs_stay_undecided():
    sels = {0: ["place", "walls"]}
    decisions, renames = {0: {}}, {0: {}}
    place_field_filter(sels, decisions, renames, discrete_place_features=["walls"])
    assert decisions[0] == {("place", "walls"): 0.5}
    assert sels[0] == ["place", "walls"]


def test_unvisited_bins_are_not_in_the_map(session):
    x, y = session["position_data"]
    visited = x < ARENA / 2
    bins = _position_bins(session["position_data"])[visited]
    amap = place_field_map(session["calcium_data"][OBJECT_CELL][visited], bins)
    assert np.isnan(amap).any() and np.isfinite(amap).any()
    assert np.isnan(amap[-1]).all()
