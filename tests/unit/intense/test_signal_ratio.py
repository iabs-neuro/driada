"""Tests for signal_ratio computation and anti-selectivity filtering.

Tests that signal_ratio is correctly computed for binary discrete features,
set to NaN for continuous features, and properly filtered in neuron_database.
"""

import numpy as np
import pandas as pd
import pytest

from driada.information.info_base import TimeSeries, calc_signal_ratio
from driada.intense.pipelines import compute_cell_feat_significance
from driada.experiment.synthetic.generators import generate_tuned_selectivity_exp


class TestCalcSignalRatioDirect:
    """Direct unit tests for calc_signal_ratio with known data."""

    def test_known_ratio(self):
        """Binary ON region has 4x the mean of OFF region."""
        binary = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        continuous = np.array([1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0])
        ratio = calc_signal_ratio(binary, continuous)
        assert ratio == pytest.approx(4.0)

    def test_ratio_below_one(self):
        """Suppression case: neurons fire less during ON state."""
        binary = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        continuous = np.array([4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0])
        ratio = calc_signal_ratio(binary, continuous)
        assert ratio == pytest.approx(0.25)

    def test_equal_activity(self):
        """No modulation: ratio should be 1.0."""
        binary = np.array([0, 0, 1, 1])
        continuous = np.array([2.0, 2.0, 2.0, 2.0])
        ratio = calc_signal_ratio(binary, continuous)
        assert ratio == pytest.approx(1.0)

    def test_zero_baseline_nonzero_on(self):
        """OFF mean is zero, ON is nonzero -> inf."""
        binary = np.array([0, 0, 1, 1])
        continuous = np.array([0.0, 0.0, 3.0, 3.0])
        ratio = calc_signal_ratio(binary, continuous)
        assert np.isinf(ratio)

    def test_both_zero(self):
        """Both means zero -> NaN."""
        binary = np.array([0, 0, 1, 1])
        continuous = np.array([0.0, 0.0, 0.0, 0.0])
        ratio = calc_signal_ratio(binary, continuous)
        assert np.isnan(ratio)


class TestSignalRatioInPipeline:
    """Integration tests: signal_ratio appears in pipeline output."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        """Run pipeline once on a small synthetic experiment with both
        binary (event) and continuous (speed) features."""
        population = [
            {"name": "event_cells", "count": 2, "features": ["event_0"]},
            {"name": "speed_cells", "count": 2, "features": ["speed"]},
            {"name": "nonselective", "count": 1, "features": []},
        ]
        exp = generate_tuned_selectivity_exp(
            population, duration=30, fps=20, seed=42, verbose=False
        )
        result = compute_cell_feat_significance(
            exp,
            data_type="calcium",
            mode="two_stage",
            n_shuffles_stage1=100,
            n_shuffles_stage2=500,
            ds=5,
            find_optimal_delays=False,
            save_computed_stats=False,
            use_precomputed_stats=False,
            verbose=False,
        )
        stats = result[0]
        return stats, exp

    def test_signal_ratio_exists_for_binary_feature(self, pipeline_result):
        """signal_ratio should be a positive float for binary features."""
        stats, exp = pipeline_result
        # event_0 is a binary discrete feature
        for cell_id in stats:
            if "event_0" in stats[cell_id]:
                sr = stats[cell_id]["event_0"]["signal_ratio"]
                assert isinstance(sr, (float, np.floating)), (
                    f"signal_ratio should be float, got {type(sr)}"
                )
                assert sr > 0 or np.isnan(sr) or np.isinf(sr), (
                    f"signal_ratio should be positive, got {sr}"
                )

    def test_signal_ratio_none_for_continuous_feature(self, pipeline_result):
        """signal_ratio should be None for continuous (non-binary) features."""
        stats, exp = pipeline_result
        for cell_id in stats:
            if "speed" in stats[cell_id]:
                sr = stats[cell_id]["speed"]["signal_ratio"]
                assert sr is None, (
                    f"signal_ratio for continuous feature should be None, got {sr}"
                )

    def test_signal_ratio_key_present_in_all_pairs(self, pipeline_result):
        """Every cell-feature pair should have a signal_ratio key."""
        stats, exp = pipeline_result
        for cell_id in stats:
            for feat_id in stats[cell_id]:
                assert "signal_ratio" in stats[cell_id][feat_id], (
                    f"signal_ratio missing for cell={cell_id}, feat={feat_id}"
                )


class TestSignalRatioCsvRoundtrip:
    """signal_ratio survives CSV save/load cycle."""

    def test_roundtrip(self, tmp_path):
        from tools.selectivity_dynamics.export import save_stats_csv

        stats = {
            '0': {
                'freezing': {'me': 0.05, 'pval': 0.001, 'opt_delay': 0,
                             'signal_ratio': 1.35},
                'speed': {'me': 0.03, 'pval': 0.01, 'opt_delay': 5,
                          'signal_ratio': None},
            }
        }
        csv_path = tmp_path / 'test_stats.csv'
        save_stats_csv(stats, ['freezing', 'speed'], csv_path)

        from tools.neuron_database.loaders import parse_stats_csv
        loaded = parse_stats_csv(csv_path)
        assert loaded[0]['freezing']['signal_ratio'] == pytest.approx(1.35)
        assert loaded[0]['speed']['signal_ratio'] is None


class TestLoaderExtractsSignalRatio:
    """neuron_database loader picks up signal_ratio from stats CSV."""

    def test_load_session(self, tmp_path):
        from tools.selectivity_dynamics.export import save_stats_csv, save_significance_csv
        from tools.neuron_database.loaders import load_session_from_csvs

        stats = {
            '0': {
                'freezing': {'me': 0.05, 'pval': 0.001, 'opt_delay': 0,
                             'signal_ratio': 1.35},
                'speed': {'me': 0.03, 'pval': 0.01, 'opt_delay': 5,
                          'signal_ratio': None},
            }
        }
        sig = {'0': {'freezing': True, 'speed': False}}

        stats_path = tmp_path / 'stats.csv'
        sig_path = tmp_path / 'sig.csv'
        save_stats_csv(stats, ['freezing', 'speed'], stats_path)
        save_significance_csv(sig, ['freezing', 'speed'], sig_path)

        records = load_session_from_csvs(stats_path, sig_path)

        freezing_rec = [r for r in records if r['feature'] == 'freezing'][0]
        assert freezing_rec['signal_ratio'] == pytest.approx(1.35)

        speed_rec = [r for r in records if r['feature'] == 'speed'][0]
        assert np.isnan(speed_rec['signal_ratio'])


class TestApplySignificanceFiltersAntiSelectivity:
    """Anti-selectivity filtering in apply_significance_filters."""

    def test_filters_suppressed_neurons(self):
        from tools.neuron_database.tables import apply_significance_filters

        df = pd.DataFrame({
            'significant': [True, True, True, True],
            'me': [0.05, 0.06, 0.05, 0.07],
            'pval': [0.0001, 0.0001, 0.0001, 0.0001],
            'delay_sign': [1, 1, 1, 1],
            'signal_ratio': [1.35, 0.8, float('nan'), 1.1],
        })

        # Default (filter on): remove suppressed, keep NaN
        filtered = apply_significance_filters(df, filter_anti_selectivity=True)
        assert len(filtered) == 3
        assert 0.8 not in filtered['signal_ratio'].values

        # Filter off: keep all
        unfiltered = apply_significance_filters(df, filter_anti_selectivity=False)
        assert len(unfiltered) == 4

    def test_no_signal_ratio_column(self):
        """Graceful when column is missing (old data)."""
        from tools.neuron_database.tables import apply_significance_filters

        df = pd.DataFrame({
            'significant': [True, True],
            'me': [0.05, 0.06],
            'pval': [0.0001, 0.0001],
            'delay_sign': [1, 1],
        })
        filtered = apply_significance_filters(df, filter_anti_selectivity=True)
        assert len(filtered) == 2


class TestQueryAntiSelectivity:
    """NeuronDatabase.query() anti_selectivity parameter."""

    def test_query_filters_suppressed(self):
        from tools.neuron_database.database import NeuronDatabase

        data = pd.DataFrame({
            'mouse': ['H01'] * 3,
            'session': ['1D'] * 3,
            'matched_id': [0, 1, 2],
            'neuron_idx': [0, 1, 2],
            'feature': ['freezing', 'freezing', 'speed'],
            'significant': [True, True, True],
            'me': [0.05, 0.06, 0.04],
            'pval': [0.0001, 0.0001, 0.0001],
            'opt_delay': [0, 0, 0],
            'delay_sign': [0, 0, 0],
            'signal_ratio': [1.3, 0.7, float('nan')],
        })
        matching = {'H01': pd.DataFrame({'1D': [1, 2, 3]})}
        db = NeuronDatabase(['1D'], matching, data)

        # No filtering by default
        result = db.query(feature='freezing')
        assert len(result) == 2

        # Filter anti-selectivities
        result = db.query(feature='freezing', anti_selectivity=True)
        assert len(result) == 1
        assert result.iloc[0]['signal_ratio'] == pytest.approx(1.3)

        # NaN passes through
        result = db.query(feature='speed', anti_selectivity=True)
        assert len(result) == 1
