"""The INTENSE runner tools pass the seed to experiment construction and to INTENSE."""

import numpy as np

from tools.selectivity_dynamics.loader import load_experiment_from_npz


def _write_npz(path):
    rng = np.random.default_rng(0)
    np.savez(path, calcium=rng.normal(size=(3, 3000)), x=rng.random(3000), y=rng.random(3000))
    return path


def test_loader_seed_is_reproducible(tmp_path):
    npz = _write_npz(tmp_path / "FOF_F01_1D_aligned.npz")
    agg = {("x", "y"): "place"}
    a = load_experiment_from_npz(npz, agg_features=agg, verbose=False, seed=5)
    b = load_experiment_from_npz(npz, agg_features=agg, verbose=False, seed=5)
    c = load_experiment_from_npz(npz, agg_features=agg, verbose=False, seed=6)
    assert np.array_equal(a.calcium.data, b.calcium.data)
    assert np.array_equal(a.dynamic_features["place"].data, b.dynamic_features["place"].data)
    assert not np.array_equal(a.calcium.data, c.calcium.data)
