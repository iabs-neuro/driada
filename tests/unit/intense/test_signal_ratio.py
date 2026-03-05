"""Tests for signal_ratio computation in INTENSE pipeline.

Tests that signal_ratio is correctly computed for binary discrete features
and set to NaN for continuous features.
"""

import numpy as np
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

    def test_signal_ratio_nan_for_continuous_feature(self, pipeline_result):
        """signal_ratio should be NaN for continuous (non-binary) features."""
        stats, exp = pipeline_result
        for cell_id in stats:
            if "speed" in stats[cell_id]:
                sr = stats[cell_id]["speed"]["signal_ratio"]
                assert np.isnan(sr), (
                    f"signal_ratio for continuous feature should be NaN, got {sr}"
                )

    def test_signal_ratio_key_present_in_all_pairs(self, pipeline_result):
        """Every cell-feature pair should have a signal_ratio key."""
        stats, exp = pipeline_result
        for cell_id in stats:
            for feat_id in stats[cell_id]:
                assert "signal_ratio" in stats[cell_id][feat_id], (
                    f"signal_ratio missing for cell={cell_id}, feat={feat_id}"
                )
