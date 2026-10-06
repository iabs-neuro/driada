"""Tests for zone masks in the place-field rule of the analysis-tool place-vs-zone filters."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.selectivity_dynamics.filters import (
    _MAP_BINS,
    _axis_bins,
    _position_bins,
    extract_filter_data,
    find_zone_mask,
    load_zone_masks,
    place_field_filter,
    place_field_in_zone,
    place_field_map,
    zone_mask_share,
)

FPS = 20
ARENA = 50.0
STEP = 0.1
OBJECT_XY = (15.0, 35.0)


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


def _raster(inside):
    """Boolean raster [iy, ix] of a zone given as a function of the cell centre (x, y)."""
    n = int(round(ARENA / STEP)) + 1
    x, y = np.meshgrid(np.arange(n) * STEP, np.arange(n) * STEP)
    return inside(x, y)


def _disc(centre, radius):
    return _raster(lambda x, y: (x - centre[0]) ** 2 + (y - centre[1]) ** 2 < radius ** 2)


@pytest.fixture(scope="module")
def session():
    """One cell with a noisy field on the object; the zone indicators say the animal was never in a zone."""
    rng = np.random.default_rng(3)
    pos = _trajectory(rng, 20 * 60 * FPS)
    x, y = pos
    tuning = np.exp(-((x - OBJECT_XY[0]) ** 2 + (y - OBJECT_XY[1]) ** 2) / (2 * 5.0 ** 2))
    never = np.zeros(x.size)
    return {
        "calcium_data": {0: tuning + rng.standard_normal(x.size) * 0.3},
        "feature_data": {"object1": never, "corners": never, "walls": never},
        "position_data": pos,
        "discrete_place_features": ["object1", "corners", "walls"],
        "place_feat_name": "place",
    }


def _run(session, features, **overrides):
    sels = {0: list(features)}
    decisions, renames = {0: {}}, {0: {}}
    place_field_filter(sels, decisions, renames, **{**session, **overrides})
    return sels[0], decisions[0], renames[0]


def test_axis_bins_are_the_bins_of_the_trajectory(session):
    pos = session["position_data"]
    assert np.array_equal(_axis_bins(pos[0], pos[0]) * _MAP_BINS + _axis_bins(pos[1], pos[1]),
                          _position_bins(pos))
    assert list(_axis_bins(np.array([pos[0].min() - 1.0, pos[0].max() + 1.0]), pos[0])) == [-1, -1]


def test_mask_share_is_the_covered_area_of_the_bin():
    # Map bins are 2 units wide and start at 0.05, so each holds raster cells 20b+1 .. 20b+20
    position = np.array([[0.05, 40.05], [0.05, 40.05]])
    mask = np.zeros((401, 401), bool)
    mask[41:51, 111:201] = True  # y: first half of bin 2; x: second half of bin 5 and bins 6-9
    share = zone_mask_share(mask, STEP, position)
    expected = np.zeros((_MAP_BINS, _MAP_BINS))
    expected[5, 2] = 0.25
    expected[6:10, 2] = 0.5
    assert np.allclose(share, expected)


def test_mask_of_the_whole_arena_covers_every_bin(session):
    share = zone_mask_share(_raster(lambda x, y: x >= 0), STEP, session["position_data"])
    assert share.shape == (_MAP_BINS, _MAP_BINS)
    assert np.all(share == 1)


def test_zone_share_replaces_the_zone_indicator(session):
    bins = _position_bins(session["position_data"])
    calcium = session["calcium_data"][0]
    amap = np.nan_to_num(place_field_map(calcium, bins), nan=-np.inf)
    px, py = np.unravel_index(np.argmax(amap), amap.shape)

    def decide(dx, **kwargs):
        share = np.zeros((_MAP_BINS, _MAP_BINS))
        share[px + dx, py] = 1.0
        return place_field_in_zone(calcium, None, bins, zone_share=share, **kwargs)[0]

    assert decide(0)
    assert not decide(3)
    assert decide(1)
    assert not decide(1, peak_tolerance_bins=0)


def test_field_on_the_object_in_a_corner_gets_both_zones(session):
    """The masks decide: the indicators of both zones are off in every frame."""
    masks = {"step": STEP, "masks": {"object1": _disc(OBJECT_XY, 4.0),
                                     "corners": _raster(lambda x, y: (x < 22) & (y > 28))}}
    sels, _, renames = _run(session, ("place", "object1", "corners", "speed"), zone_masks=masks)
    assert sorted(sels) == ["place-corners", "place-object1", "speed"]
    assert renames == {"place-corners": ("place", "corners"), "place-object1": ("place", "object1")}
    # Without the masks the same data gives no zone
    sels, decisions, _ = _run(session, ("place", "object1", "corners", "speed"))
    assert sels == ["place", "object1", "corners", "speed"]
    assert decisions == {("place", "object1"): 0, ("place", "corners"): 0}


def test_mask_elsewhere_overrides_an_indicator_on_the_field(session):
    x, y = session["position_data"]
    on_field = ((x - OBJECT_XY[0]) ** 2 + (y - OBJECT_XY[1]) ** 2 < 4.0 ** 2).astype(float)
    data = {**session, "feature_data": {**session["feature_data"], "object1": on_field}}
    assert _run(data, ("place", "object1"))[0] == ["place-object1"]
    masks = {"step": STEP, "masks": {"object1": _disc((38.0, 12.0), 4.0)}}
    sels, decisions, _ = _run(data, ("place", "object1"), zone_masks=masks)
    assert sels == ["place", "object1"]
    assert decisions == {("place", "object1"): 0}


def test_zone_without_a_mask_is_decided_by_its_indicator(session):
    x, y = session["position_data"]
    near_wall = ((np.minimum(x, ARENA - x) < 22) | (np.minimum(y, ARENA - y) < 22)).astype(float)
    data = {**session, "feature_data": {**session["feature_data"], "walls": near_wall}}
    masks = {"step": STEP, "masks": {"object1": _disc((38.0, 12.0), 4.0)}}
    sels, _, _ = _run(data, ("place", "object1", "walls"), zone_masks=masks)
    assert sels == ["place-walls"]


def _write_masks(path, zones):
    names = list(zones)
    np.savez(path, masks=np.stack([zones[name] for name in names]).astype(np.uint8),
             zone_names=np.array(names), grid_step_cm=np.float64(STEP))


def test_load_zone_masks(tmp_path, session):
    path = tmp_path / "NOF_X01_1D_zones_cm.npz"
    arena = _raster(lambda x, y: x >= 0)
    _write_masks(path, {"ArenaReal": arena, "Object1RealOut": _disc(OBJECT_XY, 4.0),
                        "ArenaCornersAllRealOut": _raster(lambda x, y: (x < 9) & (y < 9)),
                        "Object1Real": _disc(OBJECT_XY, 2.0)})
    zones = load_zone_masks(path, session["position_data"])
    assert zones["step"] == STEP
    assert sorted(zones["masks"]) == ["corners", "object1"]
    assert zones["masks"]["object1"].dtype == bool
    assert np.array_equal(zones["masks"]["object1"], _disc(OBJECT_XY, 4.0))
    assert np.array_equal(zones["arena"], arena)


def test_masks_of_another_arena_are_refused(tmp_path, session):
    path = tmp_path / "NOF_X01_1D_zones_cm.npz"
    _write_masks(path, {"ArenaReal": _raster(lambda x, y: x >= 0), "Object1RealOut": _disc(OBJECT_XY, 4.0)})
    with pytest.raises(ValueError, match="arena floor"):
        load_zone_masks(path, session["position_data"] + 30.0)


def test_find_zone_mask(tmp_path):
    (tmp_path / "NOF").mkdir()
    path = tmp_path / "NOF" / "NOF_X01_1D_zones_cm.npz"
    _write_masks(path, {"ArenaReal": _raster(lambda x, y: x >= 0)})
    assert find_zone_mask(tmp_path, "NOF_X01_1D") == path
    # The mask of another day of the same animal is not a substitute
    with pytest.warns(UserWarning, match="NOF_X01_2D"):
        assert find_zone_mask(tmp_path, "NOF_X01_2D") is None


def test_extract_filter_data_adds_the_masks_of_the_session(tmp_path, session):
    path = tmp_path / "NOF_X01_1D_zones_cm.npz"
    _write_masks(path, {"ArenaReal": _raster(lambda x, y: x >= 0), "Object1RealOut": _disc(OBJECT_XY, 4.0)})
    exp = SimpleNamespace(neurons=[SimpleNamespace(ca=SimpleNamespace(data=session["calcium_data"][0]))],
                          fps=FPS, place=SimpleNamespace(data=session["position_data"]),
                          object1=SimpleNamespace(data=session["feature_data"]["object1"]))
    assert "zone_masks" not in extract_filter_data(exp, discrete_place_features=["object1"])
    data = extract_filter_data(exp, discrete_place_features=["object1"], zone_mask_path=path)
    assert sorted(data["zone_masks"]["masks"]) == ["object1"]
    sels = {0: ["place", "object1"]}
    place_field_filter(sels, {0: {}}, {0: {}}, **data)
    assert sels[0] == ["place-object1"]
