"""
Tests for principled selectivity generation module.

This module tests the tuning-based synthetic data generation with biologically
meaningful selectivity patterns (von Mises, Gaussian, sigmoid tuning curves).
"""

import numpy as np
import pytest
from driada.experiment.synthetic import (
    sigmoid_tuning_curve,
    compute_speed_from_positions,
    compute_head_direction_from_positions,
    combine_responses,
    generate_tuned_selectivity_exp,
    ground_truth_to_selectivity_matrix,
)


class TestSigmoidTuningCurve:
    """Tests for sigmoid tuning curve function."""

    def test_basic_output_shape(self):
        """Test that output shape matches input."""
        x = np.linspace(0, 1, 100)
        response = sigmoid_tuning_curve(x, threshold=0.5, slope=10.0)
        assert response.shape == x.shape

    def test_output_range(self):
        """Test that output is in valid range [0, max_response]."""
        x = np.linspace(0, 1, 100)
        response = sigmoid_tuning_curve(x, threshold=0.5, slope=10.0, max_response=1.0)
        assert np.all(response >= 0)
        assert np.all(response <= 1.0)

    def test_threshold_behavior(self):
        """Test that response at threshold is 50% of max."""
        threshold = 0.5
        response = sigmoid_tuning_curve(
            np.array([threshold]), threshold=threshold, slope=10.0, max_response=1.0
        )
        assert np.isclose(response[0], 0.5, atol=0.01)

    def test_monotonic_increase(self):
        """Test that response increases monotonically with input."""
        x = np.linspace(0, 1, 100)
        response = sigmoid_tuning_curve(x, threshold=0.5, slope=10.0)
        diffs = np.diff(response)
        assert np.all(diffs >= 0)

    def test_slope_effect(self):
        """Test that higher slope produces sharper transition."""
        x = np.linspace(0, 1, 100)
        response_low_slope = sigmoid_tuning_curve(x, threshold=0.5, slope=2.0)
        response_high_slope = sigmoid_tuning_curve(x, threshold=0.5, slope=20.0)

        # High slope should have steeper gradient near threshold
        grad_low = np.max(np.abs(np.diff(response_low_slope)))
        grad_high = np.max(np.abs(np.diff(response_high_slope)))
        assert grad_high > grad_low

    def test_max_response_scaling(self):
        """Test that max_response scales output correctly."""
        x = np.linspace(0, 1, 100)
        response_1 = sigmoid_tuning_curve(x, threshold=0.5, slope=10.0, max_response=1.0)
        response_2 = sigmoid_tuning_curve(x, threshold=0.5, slope=10.0, max_response=2.0)
        assert np.allclose(response_2, response_1 * 2.0, rtol=0.01)


class TestComputeSpeedFromPositions:
    """Tests for speed computation from positions."""

    def test_basic_output_shape(self):
        """Test that output shape matches input time dimension."""
        positions = np.random.randn(2, 100) * 0.1
        positions = np.cumsum(positions, axis=1)
        speed = compute_speed_from_positions(positions, fps=20)
        assert speed.shape == (100,)

    def test_positive_speed(self):
        """Test that speed is always non-negative."""
        positions = np.random.randn(2, 100) * 0.1
        positions = np.cumsum(positions, axis=1)
        speed = compute_speed_from_positions(positions, fps=20)
        assert np.all(speed >= 0)

    def test_stationary_speed(self):
        """Test that stationary positions give zero speed."""
        # Create stationary positions
        positions = np.zeros((2, 100))
        positions[0, :] = 0.5  # constant x
        positions[1, :] = 0.5  # constant y
        speed = compute_speed_from_positions(positions, fps=20)
        assert np.allclose(speed, 0, atol=1e-10)

    def test_constant_velocity(self):
        """Test that constant velocity gives constant speed."""
        # Create linear trajectory
        positions = np.zeros((2, 100))
        positions[0, :] = np.linspace(0, 1, 100)  # constant x velocity
        positions[1, :] = np.linspace(0, 0.5, 100)  # constant y velocity
        speed = compute_speed_from_positions(positions, fps=20, smooth_sigma=1)

        # Exclude edges due to smoothing effects
        core_speed = speed[10:-10]
        assert np.std(core_speed) < np.mean(core_speed) * 0.1  # Low variance

    def test_fps_scaling(self):
        """Test that fps scales speed correctly."""
        positions = np.random.randn(2, 100) * 0.1
        positions = np.cumsum(positions, axis=1)
        speed_10 = compute_speed_from_positions(positions, fps=10, smooth_sigma=1)
        speed_20 = compute_speed_from_positions(positions, fps=20, smooth_sigma=1)
        # Higher fps should give higher speed values (same displacement in less time)
        assert np.mean(speed_20) > np.mean(speed_10)


class TestComputeHeadDirectionFromPositions:
    """Tests for head direction computation from positions."""

    def test_basic_output_shape(self):
        """Test that output shape matches input time dimension."""
        np.random.seed(42)
        positions = np.random.randn(2, 100) * 0.1
        positions = np.cumsum(positions, axis=1)
        hd = compute_head_direction_from_positions(positions)
        assert hd.shape == (100,)

    def test_output_range(self):
        """Test that head direction is in [0, 2*pi)."""
        np.random.seed(42)
        positions = np.random.randn(2, 100) * 0.1
        positions = np.cumsum(positions, axis=1)
        hd = compute_head_direction_from_positions(positions)
        assert np.all(hd >= 0)
        assert np.all(hd < 2 * np.pi)

    def test_rightward_movement(self):
        """Test that rightward movement gives ~0 or ~2pi direction."""
        positions = np.zeros((2, 100))
        positions[0, :] = np.linspace(0, 1, 100)  # Moving right
        positions[1, :] = 0.5  # constant y
        hd = compute_head_direction_from_positions(positions, smooth_sigma=1)

        # Middle portion should point roughly right (0 radians)
        core_hd = hd[20:-20]
        # Account for circular wrapping
        assert np.mean(np.cos(core_hd)) > 0.9  # cos should be near 1

    def test_upward_movement(self):
        """Test that upward movement gives ~pi/2 direction."""
        positions = np.zeros((2, 100))
        positions[0, :] = 0.5  # constant x
        positions[1, :] = np.linspace(0, 1, 100)  # Moving up
        hd = compute_head_direction_from_positions(positions, smooth_sigma=1)

        # Middle portion should point roughly up (pi/2 radians)
        core_hd = hd[20:-20]
        assert np.mean(np.sin(core_hd)) > 0.9  # sin should be near 1


class TestCombineResponses:
    """Tests for response combination function."""

    def test_or_mode(self):
        """Test OR (max) combination mode."""
        r1 = np.array([0.2, 0.8, 0.1])
        r2 = np.array([0.5, 0.3, 0.9])
        combined = combine_responses([r1, r2], mode="or")
        expected = np.array([0.5, 0.8, 0.9])
        assert np.allclose(combined, expected)

    def test_and_mode(self):
        """Test AND (min) combination mode."""
        r1 = np.array([0.2, 0.8, 0.1])
        r2 = np.array([0.5, 0.3, 0.9])
        combined = combine_responses([r1, r2], mode="and")
        expected = np.array([0.2, 0.3, 0.1])
        assert np.allclose(combined, expected)

    def test_single_response(self):
        """Test that single response is returned unchanged."""
        r1 = np.array([0.2, 0.8, 0.1])
        combined = combine_responses([r1], mode="or")
        assert np.allclose(combined, r1)

    def test_three_responses(self):
        """Test combining three responses."""
        r1 = np.array([0.2, 0.8, 0.1])
        r2 = np.array([0.5, 0.3, 0.9])
        r3 = np.array([0.3, 0.6, 0.4])
        combined_or = combine_responses([r1, r2, r3], mode="or")
        combined_and = combine_responses([r1, r2, r3], mode="and")
        assert np.allclose(combined_or, np.array([0.5, 0.8, 0.9]))
        assert np.allclose(combined_and, np.array([0.2, 0.3, 0.1]))

    def test_empty_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            combine_responses([], mode="or")

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        r1 = np.array([0.2, 0.8])
        r2 = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="Unknown combination mode"):
            combine_responses([r1, r2], mode="invalid")

    def test_weighted_sum_mode(self):
        """Test weighted_sum combination mode."""
        r1 = np.array([0.2, 0.8, 0.1])
        r2 = np.array([0.5, 0.3, 0.9])
        combined = combine_responses([r1, r2], weights=[0.7, 0.3], mode="weighted_sum")
        # Expected: 0.2*0.7 + 0.5*0.3 = 0.29, 0.8*0.7 + 0.3*0.3 = 0.65, 0.1*0.7 + 0.9*0.3 = 0.34
        expected = np.array([0.29, 0.65, 0.34])
        assert np.allclose(combined, expected)

    def test_weighted_or_mode(self):
        """Test weighted_or combination mode."""
        r1 = np.array([0.2, 0.8, 0.1])
        r2 = np.array([0.5, 0.3, 0.9])
        combined = combine_responses([r1, r2], weights=[0.7, 0.3], mode="weighted_or")
        # Expected: max(0.2*0.7, 0.5*0.3)=0.15, max(0.8*0.7, 0.3*0.3)=0.56, max(0.1*0.7, 0.9*0.3)=0.27
        expected = np.array([0.15, 0.56, 0.27])
        assert np.allclose(combined, expected)

    def test_weighted_sum_default_weights(self):
        """Test weighted_sum with default equal weights."""
        r1 = np.array([0.2, 0.8])
        r2 = np.array([0.6, 0.4])
        combined = combine_responses([r1, r2], mode="weighted_sum")
        # Expected with equal weights (0.5, 0.5): (0.2+0.6)/2=0.4, (0.8+0.4)/2=0.6
        expected = np.array([0.4, 0.6])
        assert np.allclose(combined, expected)

    def test_weighted_sum_clipping(self):
        """Test that weighted_sum clips to [0, 1]."""
        r1 = np.array([0.8, 0.9])
        r2 = np.array([0.9, 0.8])
        combined = combine_responses([r1, r2], weights=[0.8, 0.8], mode="weighted_sum")
        # 0.8*0.8 + 0.9*0.8 = 1.36, should clip to 1.0
        assert np.all(combined <= 1.0)
        assert np.all(combined >= 0.0)

    def test_weighted_mismatched_lengths_raises_error(self):
        """Test that mismatched weights/responses raises ValueError."""
        r1 = np.array([0.2, 0.8])
        r2 = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="Number of weights"):
            combine_responses([r1, r2], weights=[0.5, 0.3, 0.2], mode="weighted_sum")


class TestGenerateTunedSelectivityExp:
    """Tests for the main experiment generation function."""

    def test_basic_generation(self):
        """Test basic experiment generation."""
        population = [
            {"name": "hd_cells", "count": 2, "features": ["head_direction"]},
            {"name": "nonselective", "count": 2, "features": []},
        ]
        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        ground_truth = exp.ground_truth

        assert exp.n_cells == 4
        assert exp.n_frames == 30 * 20  # 30s at 20 fps

    def test_ground_truth_structure(self):
        """Test that ground truth has correct structure."""
        population = [
            {"name": "hd_cells", "count": 2, "features": ["head_direction"]},
            {"name": "place_cells", "count": 2, "features": ["x", "y"]},
        ]
        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        ground_truth = exp.ground_truth

        # Check required keys
        assert "expected_pairs" in ground_truth
        assert "neuron_types" in ground_truth
        assert "tuning_parameters" in ground_truth
        assert "population_config" in ground_truth

    def test_expected_pairs_correctness(self):
        """Test that expected pairs match population config."""
        population = [
            {"name": "hd_cells", "count": 2, "features": ["head_direction"]},
            {"name": "speed_cells", "count": 2, "features": ["speed"]},
            {"name": "nonselective", "count": 1, "features": []},
        ]
        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        ground_truth = exp.ground_truth

        expected_pairs = ground_truth["expected_pairs"]

        # HD cells: 2 neurons x 1 feature = 2 pairs
        # Speed cells: 2 neurons x 1 feature = 2 pairs
        # Nonselective: 0 pairs
        # Total: 4 pairs
        assert len(expected_pairs) == 4

        # Check specific pairs
        hd_pairs = [(n, f) for n, f in expected_pairs if f == "head_direction"]
        speed_pairs = [(n, f) for n, f in expected_pairs if f == "speed"]
        assert len(hd_pairs) == 2
        assert len(speed_pairs) == 2

    def test_place_cells_expect_position_2d(self):
        """Test that place cells expect position_2d detection (not x/y marginals)."""
        population = [
            {"name": "place_cells", "count": 2, "features": ["x", "y"], "combination": "and"},
        ]
        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        ground_truth = exp.ground_truth

        expected_pairs = ground_truth["expected_pairs"]

        # Place cells: 2 neurons x 1 feature (position_2d) = 2 pairs
        assert len(expected_pairs) == 2

        pos2d_pairs = [(n, f) for n, f in expected_pairs if f == "position_2d"]
        assert len(pos2d_pairs) == 2

        # Should NOT have x/y marginal expectations
        x_pairs = [(n, f) for n, f in expected_pairs if f == "x"]
        y_pairs = [(n, f) for n, f in expected_pairs if f == "y"]
        assert len(x_pairs) == 0
        assert len(y_pairs) == 0

    def test_neuron_types_mapping(self):
        """Test that neuron_types correctly maps indices to names."""
        population = [
            {"name": "hd_cells", "count": 2, "features": ["head_direction"]},
            {"name": "speed_cells", "count": 3, "features": ["speed"]},
        ]
        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        ground_truth = exp.ground_truth

        neuron_types = ground_truth["neuron_types"]

        # First 2 neurons should be hd_cells
        assert neuron_types[0] == "hd_cells"
        assert neuron_types[1] == "hd_cells"

        # Next 3 should be speed_cells
        assert neuron_types[2] == "speed_cells"
        assert neuron_types[3] == "speed_cells"
        assert neuron_types[4] == "speed_cells"

    def test_tuning_parameters_present(self):
        """Test that tuning parameters are auto-generated."""
        population = [
            {"name": "hd_cells", "count": 1, "features": ["head_direction"]},
            {"name": "speed_cells", "count": 1, "features": ["speed"]},
        ]
        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        ground_truth = exp.ground_truth

        tuning_params = ground_truth["tuning_parameters"]

        # HD cell should have pref_dir and kappa
        hd_params = tuning_params[0]["head_direction"]
        assert "pref_dir" in hd_params
        assert "kappa" in hd_params
        assert 0 <= hd_params["pref_dir"] < 2 * np.pi

        # Speed cell should have threshold and slope
        speed_params = tuning_params[1]["speed"]
        assert "threshold" in speed_params
        assert "slope" in speed_params

    def test_custom_tuning_defaults(self):
        """Test that custom tuning defaults are applied."""
        population = [
            {"name": "hd_cells", "count": 1, "features": ["head_direction"]},
        ]
        tuning_defaults = {"head_direction": {"kappa": 5.0}}  # Custom kappa

        exp = generate_tuned_selectivity_exp(
            population,
            tuning_defaults=tuning_defaults,
            duration=30,
            seed=42,
            verbose=False,
        )
        ground_truth = exp.ground_truth

        hd_params = ground_truth["tuning_parameters"][0]["head_direction"]
        assert hd_params["kappa"] == 5.0

    def test_user_tuning_params_override(self):
        """Test that user tuning_params in population config override defaults."""
        population = [
            {
                "name": "hd_cells",
                "count": 1,
                "features": ["head_direction"],
                "tuning_params": {"kappa": 8.0},
            },
        ]

        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        ground_truth = exp.ground_truth

        hd_params = ground_truth["tuning_parameters"][0]["head_direction"]
        assert hd_params["kappa"] == 8.0

    def test_event_features(self):
        """Test that event features work correctly."""
        population = [
            {"name": "event_cells", "count": 2, "features": ["event_0"]},
        ]

        exp = generate_tuned_selectivity_exp(
            population,
            n_discrete_features=2,
            duration=30,
            seed=42,
            verbose=False,
        )
        ground_truth = exp.ground_truth

        # Check that event_0 feature exists
        assert "event_0" in exp.dynamic_features

        # Check expected pairs
        expected_pairs = ground_truth["expected_pairs"]
        event_pairs = [(n, f) for n, f in expected_pairs if f == "event_0"]
        assert len(event_pairs) == 2

    def test_mixed_selectivity_or(self):
        """Test mixed selectivity with OR combination."""
        population = [
            {
                "name": "mixed",
                "count": 2,
                "features": ["head_direction", "event_0"],
                "combination": "or",
            },
        ]

        exp = generate_tuned_selectivity_exp(
            population,
            n_discrete_features=1,
            duration=30,
            seed=42,
            verbose=False,
        )
        ground_truth = exp.ground_truth

        # Should have 2 neurons x 2 features = 4 expected pairs
        assert len(ground_truth["expected_pairs"]) == 4

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        population = [
            {"name": "hd_cells", "count": 2, "features": ["head_direction"]},
        ]

        exp1 = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        exp2 = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )

        # Calcium signals should be identical (access raw calcium data)
        calcium1 = np.array([exp1.neurons[i].ca.data for i in range(exp1.n_cells)])
        calcium2 = np.array([exp2.neurons[i].ca.data for i in range(exp2.n_cells)])
        assert np.allclose(calcium1, calcium2)

        # Tuning parameters should be identical
        assert (
            exp1.ground_truth["tuning_parameters"][0]["head_direction"]["pref_dir"]
            == exp2.ground_truth["tuning_parameters"][0]["head_direction"]["pref_dir"]
        )

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        population = [
            {"name": "hd_cells", "count": 2, "features": ["head_direction"]},
        ]

        exp1 = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        exp2 = generate_tuned_selectivity_exp(
            population, duration=30, seed=123, verbose=False
        )

        # Calcium signals should differ (access raw calcium data)
        calcium1 = np.array([exp1.neurons[i].ca.data for i in range(exp1.n_cells)])
        calcium2 = np.array([exp2.neurons[i].ca.data for i in range(exp2.n_cells)])
        assert not np.allclose(calcium1, calcium2)

    def test_all_features_present(self):
        """Test that all expected features are in experiment.

        Only features referenced in the population config should be included.
        Unreferenced features (like position_2d) should NOT be added.
        """
        population = [
            {"name": "hd_cells", "count": 1, "features": ["head_direction"]},
            {"name": "place_cells", "count": 1, "features": ["x", "y"]},
            {"name": "speed_cells", "count": 1, "features": ["speed"]},
            {"name": "event_cells", "count": 1, "features": ["event_0"]},
        ]

        exp = generate_tuned_selectivity_exp(
            population,
            n_discrete_features=2,
            duration=30,
            seed=42,
            verbose=False,
        )

        # Features referenced in population config should be present
        assert "head_direction" in exp.dynamic_features
        assert "x" in exp.dynamic_features
        assert "y" in exp.dynamic_features
        assert "speed" in exp.dynamic_features
        assert "event_0" in exp.dynamic_features
        assert "event_1" in exp.dynamic_features
        # position_2d should NOT be present since it's not referenced
        assert "position_2d" not in exp.dynamic_features

    def test_weighted_combination_in_generator(self):
        """Test weighted combination mode in experiment generation."""
        population = [
            {
                "name": "weighted_mixed",
                "count": 2,
                "features": ["head_direction", "event_0"],
                "weights": [0.7, 0.3],
                "combination": "weighted_sum",
            },
        ]

        exp = generate_tuned_selectivity_exp(
            population,
            n_discrete_features=1,
            duration=30,
            seed=42,
            verbose=False,
        )
        ground_truth = exp.ground_truth

        # Check that _combination info is stored in ground truth
        for idx in range(2):
            params = ground_truth["tuning_parameters"][idx]
            assert "_combination" in params
            assert params["_combination"]["mode"] == "weighted_sum"
            assert params["_combination"]["weights"]["head_direction"] == 0.7
            assert params["_combination"]["weights"]["event_0"] == 0.3

    def test_unknown_feature_raises_error(self):
        """Test that unknown feature raises ValueError."""
        population = [
            {"name": "test", "count": 1, "features": ["unknown_feature"]},
        ]

        with pytest.raises(ValueError, match="Unknown feature"):
            generate_tuned_selectivity_exp(
                population, duration=30, seed=42, verbose=False
            )


class TestIntegrationWithIntenseResults:
    """Test integration with IntenseResults validation."""

    def test_validate_against_ground_truth_method(self):
        """Test that IntenseResults.validate_against_ground_truth works."""
        from driada.intense.intense_base import IntenseResults

        # Create mock results
        results = IntenseResults()
        results.significance = {
            0: {"head_direction": {"criterion": True}},
            1: {"head_direction": {"criterion": False}},
            2: {"speed": {"criterion": True}},
        }

        ground_truth = {
            "expected_pairs": [(0, "head_direction"), (1, "head_direction"), (2, "speed")],
            "neuron_types": {0: "hd_cell", 1: "hd_cell", 2: "speed_cell"},
        }

        metrics = results.validate_against_ground_truth(ground_truth, verbose=False)

        # Check metrics structure
        assert "true_positives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics
        assert "sensitivity" in metrics
        assert "precision" in metrics
        assert "f1" in metrics

        # Check values
        assert metrics["true_positives"] == 2  # 0->hd, 2->speed
        assert metrics["false_negatives"] == 1  # 1->hd was missed
        assert metrics["sensitivity"] == 2 / 3

    def test_validate_no_significance_raises_error(self):
        """Test that validation without significance raises error."""
        from driada.intense.intense_base import IntenseResults

        results = IntenseResults()
        ground_truth = {"expected_pairs": [(0, "hd")]}

        with pytest.raises(ValueError, match="no 'significance' attribute"):
            results.validate_against_ground_truth(ground_truth)


class TestGroundTruthToSelectivityMatrix:
    """Tests for ground truth to selectivity matrix conversion."""

    def test_basic_conversion(self):
        """Test basic conversion to matrix format."""
        ground_truth = {
            "expected_pairs": [(0, "hd"), (1, "hd"), (2, "speed")],
            "neuron_types": {0: "hd_cell", 1: "hd_cell", 2: "speed_cell"},
        }

        result = ground_truth_to_selectivity_matrix(ground_truth)

        assert "matrix" in result
        assert "feature_names" in result
        assert result["matrix"].shape == (2, 3)  # 2 features, 3 neurons

    def test_feature_order(self):
        """Test that feature names are sorted alphabetically by default."""
        ground_truth = {
            "expected_pairs": [(0, "z_feature"), (1, "a_feature")],
            "neuron_types": {0: "type1", 1: "type2"},
        }

        result = ground_truth_to_selectivity_matrix(ground_truth)

        assert result["feature_names"] == ["a_feature", "z_feature"]

    def test_custom_feature_order(self):
        """Test that custom feature order is respected."""
        ground_truth = {
            "expected_pairs": [(0, "speed"), (1, "hd")],
            "neuron_types": {0: "speed_cell", 1: "hd_cell"},
        }

        result = ground_truth_to_selectivity_matrix(
            ground_truth, feature_names=["hd", "speed"]
        )

        assert result["feature_names"] == ["hd", "speed"]
        # Check matrix values are in correct positions
        assert result["matrix"][0, 1] == 1.0  # hd at neuron 1
        assert result["matrix"][1, 0] == 1.0  # speed at neuron 0

    def test_matrix_values(self):
        """Test that matrix values are binary (0 or 1)."""
        ground_truth = {
            "expected_pairs": [(0, "hd"), (2, "hd"), (1, "speed")],
            "neuron_types": {0: "hd", 1: "speed", 2: "hd"},
        }

        result = ground_truth_to_selectivity_matrix(ground_truth)

        # All values should be 0 or 1
        assert np.all((result["matrix"] == 0) | (result["matrix"] == 1))

    def test_empty_expected_pairs(self):
        """Test handling of empty expected pairs."""
        ground_truth = {
            "expected_pairs": [],
            "neuron_types": {0: "nonselective", 1: "nonselective"},
        }

        result = ground_truth_to_selectivity_matrix(ground_truth)

        assert result["matrix"].shape == (0, 2)
        assert result["feature_names"] == []

    def test_multiple_features_per_neuron(self):
        """Test neurons with multiple features (place cells)."""
        ground_truth = {
            "expected_pairs": [(0, "x"), (0, "y"), (1, "hd")],
            "neuron_types": {0: "place_cell", 1: "hd_cell"},
        }

        result = ground_truth_to_selectivity_matrix(ground_truth)

        # Neuron 0 should have both x and y
        feat_idx_hd = result["feature_names"].index("hd")
        feat_idx_x = result["feature_names"].index("x")
        feat_idx_y = result["feature_names"].index("y")

        assert result["matrix"][feat_idx_x, 0] == 1.0
        assert result["matrix"][feat_idx_y, 0] == 1.0
        assert result["matrix"][feat_idx_hd, 1] == 1.0

    def test_integration_with_generate_tuned_selectivity_exp(self):
        """Test conversion of ground truth from generate_tuned_selectivity_exp."""
        population = [
            {"name": "hd_cells", "count": 2, "features": ["head_direction"]},
            {"name": "speed_cells", "count": 2, "features": ["speed"]},
            {"name": "nonselective", "count": 1, "features": []},
        ]

        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )
        ground_truth = exp.ground_truth

        result = ground_truth_to_selectivity_matrix(ground_truth)

        # Check shape: 2 features (head_direction, speed), 5 neurons
        assert result["matrix"].shape[1] == 5
        assert len(result["feature_names"]) == 2

        # Check that selectivity is correct
        feat_idx_hd = result["feature_names"].index("head_direction")
        feat_idx_speed = result["feature_names"].index("speed")

        # Neurons 0,1 are HD cells
        assert result["matrix"][feat_idx_hd, 0] == 1.0
        assert result["matrix"][feat_idx_hd, 1] == 1.0

        # Neurons 2,3 are speed cells
        assert result["matrix"][feat_idx_speed, 2] == 1.0
        assert result["matrix"][feat_idx_speed, 3] == 1.0

        # Neuron 4 is nonselective
        assert result["matrix"][:, 4].sum() == 0


class TestThresholdResponse:
    """Tests for threshold_response function."""

    def test_binary_discretization(self):
        """Test binary threshold discretization."""
        from driada.experiment.synthetic import threshold_response

        feature = np.array([0.1, 0.3, 0.7, 0.9, 0.2])
        response = threshold_response(feature, discretization="binary", threshold=0.5)
        expected = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(response, expected)

    def test_roi_discretization(self):
        """Test ROI-based discretization."""
        from driada.experiment.synthetic import threshold_response

        # Generate a feature with known structure
        np.random.seed(42)
        feature = np.random.randn(100)
        feature = (feature - feature.min()) / (feature.max() - feature.min())

        response = threshold_response(feature, discretization="roi", seed=42)

        # Response should be binary
        assert set(np.unique(response)).issubset({0.0, 1.0})
        # ROI should capture approximately 15% of values
        active_fraction = response.mean()
        assert 0.05 < active_fraction < 0.30  # Allow some variance

    def test_invalid_discretization_raises_error(self):
        """Test that invalid discretization mode raises error."""
        from driada.experiment.synthetic import threshold_response

        feature = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="Unknown discretization"):
            threshold_response(feature, discretization="invalid")


class TestThresholdTuningType:
    """Tests for tuning_type='threshold' in generate_tuned_selectivity_exp."""

    def test_threshold_tuning_type_basic(self):
        """Test basic threshold tuning type."""
        population = [
            {
                "name": "threshold_cells",
                "count": 3,
                "features": ["fbm_0"],
                "tuning_type": "threshold",
            },
        ]

        exp = generate_tuned_selectivity_exp(
            population,
            duration=30,
            seed=42,
            verbose=False,
        )

        assert exp.n_cells == 3
        # Check tuning parameters recorded
        for neuron_idx in range(3):
            params = exp.ground_truth["tuning_parameters"][neuron_idx]
            assert "fbm_0" in params
            assert params["fbm_0"]["tuning_type"] == "threshold"

    def test_threshold_tuning_with_events(self):
        """Test threshold tuning with event features (already binary)."""
        population = [
            {
                "name": "event_threshold",
                "count": 2,
                "features": ["event_0"],
                "tuning_type": "threshold",
            },
        ]

        exp = generate_tuned_selectivity_exp(
            population,
            duration=30,
            seed=42,
            n_discrete_features=1,
            verbose=False,
        )

        assert exp.n_cells == 2
        # Events should still work with threshold mode
        for neuron_idx in range(2):
            params = exp.ground_truth["tuning_parameters"][neuron_idx]
            assert "event_0" in params
            assert params["event_0"].get("binary", False)

    def test_threshold_vs_default_different_responses(self):
        """Test that threshold mode produces different responses than default."""
        # Default (sigmoid) tuning
        pop_sigmoid = [
            {"name": "sigmoid", "count": 2, "features": ["fbm_0"]},
        ]
        exp_sigmoid = generate_tuned_selectivity_exp(
            pop_sigmoid, duration=30, seed=42, verbose=False
        )

        # Threshold tuning
        pop_threshold = [
            {"name": "threshold", "count": 2, "features": ["fbm_0"],
             "tuning_type": "threshold"},
        ]
        exp_threshold = generate_tuned_selectivity_exp(
            pop_threshold, duration=30, seed=42, verbose=False
        )

        # The tuning parameters should be different
        sig_params = exp_sigmoid.ground_truth["tuning_parameters"][0]["fbm_0"]
        thresh_params = exp_threshold.ground_truth["tuning_parameters"][0]["fbm_0"]

        # Sigmoid has "slope" and "threshold", threshold mode has "tuning_type"
        assert "slope" in sig_params
        assert "tuning_type" in thresh_params
        assert thresh_params["tuning_type"] == "threshold"

    def test_threshold_binary_discretization_mode(self):
        """Test threshold mode with binary discretization."""
        population = [
            {
                "name": "threshold_binary",
                "count": 2,
                "features": ["speed"],
                "tuning_type": "threshold",
                "discretization": "binary",
            },
        ]

        exp = generate_tuned_selectivity_exp(
            population, duration=30, seed=42, verbose=False
        )

        # Check discretization mode recorded
        params = exp.ground_truth["tuning_parameters"][0]["speed"]
        assert params["tuning_type"] == "threshold"
        assert params["discretization"] == "binary"
