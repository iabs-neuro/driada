"""A merged place-<zone> label reaches the cross-session tables with the zone's own statistics."""

import numpy as np
import pandas as pd
import pytest

from tools.neuron_database.database import pretransform_merge_composite_place
from tools.neuron_database.loaders import load_session_from_csvs
from tools.selectivity_dynamics.analysis import _combine_feature_stats
from tools.selectivity_dynamics.export import save_significance_csv, save_stats_csv

PLACE = {'me': 0.20, 'pval': 1e-9, 'opt_delay': 5, 'signal_ratio': None}
ZONE = {'me': 0.03, 'pval': 1e-4, 'opt_delay': 40, 'signal_ratio': 1.8}


def test_merged_entry_keeps_component_stats():
    merged = _combine_feature_stats(PLACE, ZONE, names=('place', 'object1'))
    assert merged['me'] == 0.20 and merged['pval'] == 1e-9
    assert merged['merged_from'] == ['place', 'object1']
    assert merged['component_stats'] == {'place': PLACE, 'object1': ZONE}


def test_without_names_nothing_is_added():
    assert 'component_stats' not in _combine_feature_stats(PLACE, ZONE)


@pytest.fixture
def session_records(tmp_path):
    stats = {'0': {'place-object1': _combine_feature_stats(PLACE, ZONE, names=('place', 'object1')),
                   'speed': {'me': 0.02, 'pval': 1e-3, 'opt_delay': 0}},
             '1': {'place-object1': _combine_feature_stats(PLACE, ZONE)}}
    sig = {nid: {feat: {'stage1': True, 'stage2': True} for feat in feats} for nid, feats in stats.items()}
    features = ['place-object1', 'speed']
    save_stats_csv(stats, features, tmp_path / 'S INTENSE stats.csv')
    save_significance_csv(sig, features, tmp_path / 'S INTENSE significance.csv')
    records, _ = load_session_from_csvs(tmp_path / 'S INTENSE stats.csv', tmp_path / 'S INTENSE significance.csv')
    return pd.DataFrame(records)


def test_zone_takes_its_own_stats_in_cross_session_tables(session_records):
    data = pretransform_merge_composite_place(session_records, ['object1'])
    row = data[(data.neuron_idx == 0) & (data.feature == 'object1')].iloc[0]
    assert row['me'] == pytest.approx(0.03)
    assert row['pval'] == pytest.approx(1e-4)
    assert row['opt_delay'] == 40
    assert row['signal_ratio'] == pytest.approx(1.8)
    speed = data[data.feature == 'speed'].iloc[0]
    assert speed['me'] == pytest.approx(0.02)


def test_tables_without_component_stats_keep_the_merged_values(session_records):
    data = pretransform_merge_composite_place(session_records, ['object1'])
    row = data[(data.neuron_idx == 1) & (data.feature == 'object1')].iloc[0]
    assert row['me'] == pytest.approx(0.20)


def test_composite_outside_the_zone_list_is_left_alone(session_records):
    data = pretransform_merge_composite_place(session_records, ['walls'])
    row = data[data.neuron_idx == 0].iloc[0]
    assert row['feature'] == 'place-object1'
    assert row['me'] == pytest.approx(0.20)
    assert not np.isnan(row['opt_delay'])
