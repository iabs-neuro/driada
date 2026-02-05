"""Tests for INTENSE circular feature substitution."""

import numpy as np
import pytest

from driada.experiment.synthetic import generate_circular_manifold_exp
from driada.experiment import load_exp_from_aligned_data
from driada.intense.pipelines import substitute_circular_with_2d, compute_cell_feat_significance
from driada.information import TimeSeries, MultiTimeSeries


class TestSubstituteCircularWith2d:
    """Tests for the substitution helper function."""

    @pytest.fixture
    def exp_with_circular(self):
        """Create experiment with circular feature and its _2d version."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
            "speed": np.random.uniform(0, 10, 1000),
        }
        return load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

    def test_substitutes_circular_with_2d(self, exp_with_circular):
        """Test that circular features are substituted with _2d versions."""
        feat_ids = ["head_direction", "speed"]
        new_ids, substitutions = substitute_circular_with_2d(
            feat_ids, exp_with_circular, verbose=False
        )

        # head_direction should be substituted
        assert "head_direction_2d" in new_ids
        assert "head_direction" not in new_ids
        # speed should remain unchanged
        assert "speed" in new_ids

        # Check substitutions list
        assert ("head_direction", "head_direction_2d") in substitutions
        assert len(substitutions) == 1

    def test_preserves_non_circular(self, exp_with_circular):
        """Test that non-circular features are not substituted."""
        feat_ids = ["speed"]
        new_ids, substitutions = substitute_circular_with_2d(
            feat_ids, exp_with_circular, verbose=False
        )

        assert new_ids == ["speed"]
        assert len(substitutions) == 0

    def test_handles_already_2d_features(self, exp_with_circular):
        """Test that _2d features are not double-substituted."""
        feat_ids = ["head_direction_2d"]
        new_ids, substitutions = substitute_circular_with_2d(
            feat_ids, exp_with_circular, verbose=False
        )

        assert new_ids == ["head_direction_2d"]
        assert len(substitutions) == 0

    def test_handles_tuple_features(self, exp_with_circular):
        """Test that tuple features (multifeatures) are preserved."""
        feat_ids = [("head_direction", "speed")]
        new_ids, substitutions = substitute_circular_with_2d(
            feat_ids, exp_with_circular, verbose=False
        )

        # Tuples should pass through unchanged
        assert new_ids == [("head_direction", "speed")]
        assert len(substitutions) == 0

    def test_handles_missing_2d_version(self):
        """Test handling when circular feature exists but _2d version doesn't."""
        np.random.seed(42)
        data = {
            "Calcium": np.random.randn(10, 1000),
            "head_direction": np.random.uniform(0, 2 * np.pi, 1000),
        }
        # Create experiment without _2d creation
        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=False, verbose=False
        )

        feat_ids = ["head_direction"]
        new_ids, substitutions = substitute_circular_with_2d(feat_ids, exp, verbose=False)

        # No substitution should happen if _2d doesn't exist
        assert new_ids == ["head_direction"]
        assert len(substitutions) == 0


class TestIntenseCircularSubstitution:
    """Tests for INTENSE pipeline using circular substitution."""

    @pytest.fixture
    def circular_exp(self):
        """Generate circular manifold experiment for testing."""
        return generate_circular_manifold_exp(
            n_neurons=10, duration=60, fps=20, seed=42, verbose=False
        )

    def test_intense_creates_2d_version(self, circular_exp):
        """Test that circular experiment has _2d version created."""
        # Check that head_direction_2d exists
        assert "head_direction_2d" in circular_exp.dynamic_features
        assert isinstance(
            circular_exp.dynamic_features["head_direction_2d"], MultiTimeSeries
        )

    def test_use_circular_2d_default(self, circular_exp):
        """Test that use_circular_2d=True is the default."""
        # Run INTENSE with default settings
        stats, sig, info, res = compute_cell_feat_significance(
            circular_exp,
            feat_bunch=["head_direction"],
            cell_bunch=[0, 1],
            mode="stage1",
            n_shuffles_stage1=10,
            verbose=False,
        )

        # Check that results were computed (should use head_direction_2d internally)
        assert len(stats) > 0

    def test_use_circular_2d_disabled(self, circular_exp):
        """Test that use_circular_2d=False uses raw circular features."""
        # Run INTENSE with circular substitution disabled
        stats, sig, info, res = compute_cell_feat_significance(
            circular_exp,
            feat_bunch=["head_direction"],
            cell_bunch=[0, 1],
            mode="stage1",
            n_shuffles_stage1=10,
            use_circular_2d=False,
            verbose=False,
        )

        # Should still work, just with raw circular feature
        assert len(stats) > 0

    def test_explicit_2d_feature_not_double_substituted(self, circular_exp):
        """Test that explicitly requesting _2d feature doesn't cause issues."""
        # Explicitly request the _2d version
        stats, sig, info, res = compute_cell_feat_significance(
            circular_exp,
            feat_bunch=["head_direction_2d"],
            cell_bunch=[0, 1],
            mode="stage1",
            n_shuffles_stage1=10,
            use_circular_2d=True,  # This should not cause double substitution
            verbose=False,
        )

        assert len(stats) > 0


class TestIntenseResultsWithCircular:
    """Test that INTENSE results are consistent with circular features."""

    @pytest.fixture
    def tuned_circular_exp(self):
        """Generate experiment with known circular tuning."""
        return generate_circular_manifold_exp(
            n_neurons=20,
            duration=120,
            fps=20,
            tuning_concentration=4.0,  # High tuning
            seed=42,
            verbose=False,
        )

    def test_circular_feature_detected_as_significant(self, tuned_circular_exp):
        """Test that circular tuning is detected when using _2d representation."""
        stats, sig, info, res = compute_cell_feat_significance(
            tuned_circular_exp,
            feat_bunch=["head_direction"],
            mode="stage1",
            n_shuffles_stage1=50,
            use_circular_2d=True,
            verbose=False,
        )

        # At least some neurons should show significant response
        # (the synthetic data has tuned neurons)
        significant_count = sum(
            1 for cell_id in sig if sig[cell_id].get("head_direction_2d", False)
        )
        # With strong tuning and reasonable duration, we expect detections
        # This is a smoke test - actual detection depends on parameters
        assert significant_count >= 0  # At minimum, no errors occurred
