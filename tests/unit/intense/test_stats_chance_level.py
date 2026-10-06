"""get_table_of_stats reports the chance level of every pair next to its metric value."""

import numpy as np
import pytest
from scipy.stats import norm

from driada.intense.stats import get_mi_distr_pvalue, get_table_of_stats

NSH = 100
TRUE_VALUES = [[0.5, 0.001], [0.03, 0.2], [0.015, 0.08]]
DELAYS = np.array([[0, 3], [-2, 5], [1, 0]])
# Number of shuffles strictly below the true value of each pair, plus one.
RANK_COUNTS = [[101, 1], [85, 101], [56, 100]]


@pytest.fixture
def metable():
    rng = np.random.default_rng(0)
    table = rng.gamma(2, 0.01, size=(3, 2, NSH + 1))
    table[:, :, 0] = TRUE_VALUES
    return table


@pytest.mark.parametrize("stage", [1, 2])
def test_chance_level_is_the_shuffle_mean(metable, stage):
    stats = get_table_of_stats(metable, DELAYS, nsh=NSH, stage=stage)
    for i in range(3):
        for j in range(2):
            pair = stats[i][j]
            assert pair["me_null"] == metable[i, j, 1:].mean()
            assert pair["me_excess"] == pair["me"] - pair["me_null"]
            assert isinstance(pair["me_null"], float)
            assert isinstance(pair["me_excess"], float)


@pytest.mark.parametrize("stage", [1, 2])
def test_excess_is_not_floored_at_zero(metable, stage):
    stats = get_table_of_stats(metable, DELAYS, nsh=NSH, stage=stage)
    assert stats[0][1]["me"] == 0.001
    assert stats[0][1]["me_excess"] < -0.01


def test_single_precision_table(metable):
    table = metable.astype(np.float32)
    stats = get_table_of_stats(table, DELAYS, nsh=NSH, stage=2, metric_distr_type="norm")
    for i in range(3):
        for j in range(2):
            null = float(table[i, j, 1:].astype(np.float64).mean())
            assert stats[i][j]["me_null"] == pytest.approx(null, abs=1e-7)
            assert stats[i][j]["me_excess"] == pytest.approx(float(table[i, j, 0]) - null, abs=1e-7)


def test_masked_pairs_get_no_chance_level(metable):
    mask = np.array([[1, 0], [0, 1], [1, 1]])
    stats = get_table_of_stats(metable, DELAYS, precomputed_mask=mask, nsh=NSH, stage=1)
    assert "me_null" in stats[0][0]
    assert stats[0][1] == {}


def test_stage1_statistics_are_unchanged(metable):
    stats = get_table_of_stats(metable, DELAYS, nsh=NSH, stage=1)
    for i in range(3):
        for j in range(2):
            pair = stats[i][j]
            assert pair["pre_rval"] == RANK_COUNTS[i][j] / (NSH + 1)
            assert pair["pre_pval"] is None
            assert pair["me"] == TRUE_VALUES[i][j]
            assert pair["opt_delay"] == DELAYS[i, j]
            assert set(pair) == {"pre_rval", "pre_pval", "me", "opt_delay", "me_null", "me_excess"}


@pytest.mark.parametrize("distr_type", ["gamma_zi", "gamma", "norm"])
def test_stage2_statistics_are_unchanged(metable, distr_type):
    stats = get_table_of_stats(metable, DELAYS, nsh=NSH, stage=2, metric_distr_type=distr_type)
    for i in range(3):
        for j in range(2):
            pair = stats[i][j]
            assert pair["rval"] == RANK_COUNTS[i][j] / (NSH + 1)
            assert pair["me"] == TRUE_VALUES[i][j]
            assert pair["opt_delay"] == DELAYS[i, j]
            assert set(pair) == {"rval", "pval", "me", "opt_delay", "me_null", "me_excess"}
            if distr_type == "norm":
                shuffles = metable[i, j, 1:]
                z_score = (metable[i, j, 0] - shuffles.mean()) / (shuffles.std() + 1e-30)
                assert pair["pval"] == pytest.approx(float(norm.sf(z_score)), rel=1e-12, abs=1e-300)
            else:
                expected = get_mi_distr_pvalue(metable[i, j, 1:], metable[i, j, 0], distr_type=distr_type)
                assert pair["pval"] == expected


def _ar1(rng, n_frames, phi, size=None):
    shape = (n_frames,) if size is None else (size, n_frames)
    noise = rng.standard_normal(shape)
    out = np.empty(shape)
    out[..., 0] = noise[..., 0] / np.sqrt(1 - phi**2)
    for t in range(1, n_frames):
        out[..., t] = phi * out[..., t - 1] + noise[..., t]
    return out


def test_excess_of_unrelated_autocorrelated_signals_is_zero():
    """Ten minutes at 4 Hz, correlation time 4 s, a five-component feature."""
    from driada.information.info_base import MultiTimeSeries, TimeSeries
    from driada.intense.intense_base import compute_me_stats

    rng = np.random.default_rng(0)
    n_frames, phi = 10 * 60 * 4, np.exp(-1 / 16)
    signals = [TimeSeries(_ar1(rng, n_frames, phi), discrete=False) for _ in range(20)]
    features = [
        MultiTimeSeries([TimeSeries(row, discrete=False) for row in _ar1(rng, n_frames, phi, size=5)])
        for _ in range(10)
    ]
    stats, _, _ = compute_me_stats(
        signals, features, mode="stage1", n_shuffles_stage1=100,
        verbose=False, seed=1, enable_parallelization=False,
    )
    pairs = [stats[i][j] for i in stats for j in stats[i]]
    assert len(pairs) == 200
    assert np.mean([pair["me"] for pair in pairs]) > 0.01
    assert abs(np.mean([pair["me_excess"] for pair in pairs])) < 0.002
