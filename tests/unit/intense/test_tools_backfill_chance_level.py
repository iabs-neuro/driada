"""Chance levels recovered from saved shuffles match a fresh run of the library."""

import copy

import numpy as np
import pandas as pd
import pytest

import driada
from driada.experiment.synthetic import generate_synthetic_exp
from tools.neuron_database.loaders import parse_stats_csv
from tools.selectivity_dynamics.analysis import _combine_feature_stats
from tools.selectivity_dynamics.backfill_chance_level import (
    backfill_output_dir,
    chance_levels_from_results,
)
from tools.selectivity_dynamics.export import save_all_results, save_stats_csv

# Shuffled values are saved in half precision.
TOLERANCE = 2e-4
NAME = 'NOF_T01_1D'


def _without_chance_level(stats):
    stripped = copy.deepcopy(stats)
    for feats in stripped.values():
        for pair in feats.values():
            pair.pop('me_null', None)
            pair.pop('me_excess', None)
    return stripped


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    """A saved result as written before chance levels existed, and the fresh statistics."""
    exp = generate_synthetic_exp(n_dfeats=2, n_cfeats=2, nneurons=6, duration=120, seed=3, verbose=False)
    stats, significance, info, results, _ = driada.compute_cell_feat_significance(
        exp, mode='two_stage', n_shuffles_stage1=50, n_shuffles_stage2=300, ds=2,
        find_optimal_delays=False, with_disentanglement=False,
        enable_parallelization=False, seed=1, verbose=False,
        # Loose thresholds keep enough significant pairs for the tables.
        pval_thr=0.5, multicomp_correction=None,
    )
    fresh = copy.deepcopy(stats)
    old = _without_chance_level(stats)
    results.update('stats', old)
    src = tmp_path_factory.mktemp('saved')
    save_all_results(NAME, exp, old, significance, info, results, None, src)
    return src, fresh


def test_fixture_has_stage2_pairs(saved):
    _, fresh = saved
    assert sum('rval' in pair for feats in fresh.values() for pair in feats.values()) >= 3


def test_levels_match_a_fresh_run(saved):
    src, fresh = saved
    levels = chance_levels_from_results(src / 'results' / f'{NAME}_results.npz')
    expected = {(str(cell), feat): pair for cell, feats in fresh.items()
                for feat, pair in feats.items() if pair.get('me') is not None}
    assert set(levels) == set(expected) and expected
    for key, pair in expected.items():
        assert levels[key]['me_null'] == pytest.approx(pair['me_null'], abs=TOLERANCE)
        assert levels[key]['me_excess'] == pytest.approx(pair['me_excess'], abs=TOLERANCE)


def test_tables_are_written_to_a_new_folder(saved, tmp_path):
    src, fresh = saved
    out = tmp_path / 'backfilled'
    before = (src / 'tables' / f'{NAME} INTENSE stats.csv').read_bytes()
    assert backfill_output_dir(src, out) == 1
    assert (src / 'tables' / f'{NAME} INTENSE stats.csv').read_bytes() == before

    old = parse_stats_csv(src / 'tables' / f'{NAME} INTENSE stats.csv')
    new = parse_stats_csv(out / 'tables' / f'{NAME} INTENSE stats.csv')
    assert set(new) == set(old) and new
    for cell, feats in new.items():
        assert set(feats) == set(old[cell])
        for feat, pair in feats.items():
            reference = fresh[cell][feat] if cell in fresh else fresh[str(cell)][feat]
            assert pair['me_null'] == pytest.approx(reference['me_null'], abs=TOLERANCE)
            assert pair['me_excess'] == pytest.approx(reference['me_excess'], abs=TOLERANCE)
            assert {k: v for k, v in pair.items() if k not in ('me_null', 'me_excess')} == old[cell][feat]
    assert ((out / 'tables' / f'{NAME} INTENSE significance.csv').read_bytes()
            == (src / 'tables' / f'{NAME} INTENSE significance.csv').read_bytes())


def test_existing_output_is_not_overwritten(saved, tmp_path):
    src, _ = saved
    out = tmp_path / 'backfilled'
    backfill_output_dir(src, out)
    with pytest.raises(FileExistsError):
        backfill_output_dir(src, out)
    with pytest.raises(ValueError, match='new folder'):
        backfill_output_dir(src, src)


def test_merged_zone_entry_takes_the_levels_of_its_components(saved, tmp_path):
    src, fresh = saved
    raw = parse_stats_csv(src / 'tables' / f'{NAME} INTENSE stats.csv')
    cell, feats = next((cell, feats) for cell, feats in raw.items() if len(feats) >= 2)
    first, second = sorted(feats)[:2]
    merged_name = f'{first}-{second}'
    merged = {str(cell): {
        merged_name: _combine_feature_stats(feats[first], feats[second], names=(first, second)),
        first: feats[first],
    }}
    disent_dir = src / 'tables_disentangled'
    disent_dir.mkdir(exist_ok=True)
    save_stats_csv(merged, [first, merged_name], disent_dir / f'{NAME} INTENSE stats.csv')
    pd.DataFrame({first: ["{'stage2': True}"], merged_name: ["{'stage2': True}"]},
                 index=[cell]).to_csv(disent_dir / f'{NAME} INTENSE significance.csv')

    out = tmp_path / 'backfilled'
    backfill_output_dir(src, out)
    entry = parse_stats_csv(out / 'tables_disentangled' / f'{NAME} INTENSE stats.csv')[cell][merged_name]
    reference = fresh[cell] if cell in fresh else fresh[str(cell)]
    dominant = first if feats[first]['me'] >= feats[second]['me'] else second
    assert entry['me_null'] == pytest.approx(reference[dominant]['me_null'], abs=TOLERANCE)
    assert entry['me_excess'] == pytest.approx(reference[dominant]['me_excess'], abs=TOLERANCE)
    for name in (first, second):
        component = entry['component_stats'][name]
        assert component['me_null'] == pytest.approx(reference[name]['me_null'], abs=TOLERANCE)
        assert component['me_excess'] == pytest.approx(reference[name]['me_excess'], abs=TOLERANCE)


def test_misaligned_shuffles_are_refused(saved, tmp_path):
    src, _ = saved
    path = src / 'results' / f'{NAME}_results.npz'
    arrays = dict(np.load(path, allow_pickle=True))
    for stage in ('1', '2'):
        key = f'info_me_total{stage}_data'
        if key in arrays:
            arrays[key] = arrays[key][::-1].copy()
    broken = tmp_path / f'{NAME}_results.npz'
    np.savez(broken, **arrays)
    with pytest.raises(ValueError, match='do not match'):
        chance_levels_from_results(broken)


def test_stage1_shuffles_are_used_when_there_is_no_stage2(tmp_path):
    exp = generate_synthetic_exp(n_dfeats=1, n_cfeats=1, nneurons=3, duration=120, seed=3, verbose=False)
    stats, significance, info, results, _ = driada.compute_cell_feat_significance(
        exp, mode='stage1', n_shuffles_stage1=50, ds=2, find_optimal_delays=False,
        with_disentanglement=False, enable_parallelization=False, seed=1, verbose=False,
    )
    fresh = copy.deepcopy(stats)
    results.update('stats', _without_chance_level(stats))
    save_all_results(NAME, exp, results.stats, significance, info, results, None, tmp_path)
    levels = chance_levels_from_results(tmp_path / 'results' / f'{NAME}_results.npz')
    assert len(levels) == sum(len(feats) for feats in fresh.values()) > 0
    for (cell, feat), level in levels.items():
        assert level['me_null'] == pytest.approx(fresh[int(cell)][feat]['me_null'], abs=TOLERANCE)
