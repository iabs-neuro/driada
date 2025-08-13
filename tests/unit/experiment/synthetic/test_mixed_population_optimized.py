"""
Optimized tests for mixed population generator.

This module provides fast tests for the generate_mixed_population_exp function
with reduced parameters for quick execution while maintaining proper test coverage.
"""

import numpy as np
import pytest
from driada.experiment import generate_mixed_population_exp, Experiment


# Fast test parameters
FAST_PARAMS = {
    "duration": 30,  # 30s instead of 60-300s
    "fps": 10,  # 10 instead of 20
    "verbose": False,
    "seed": 42,
}


class TestGenerateMixedPopulationExpFast:
    """Fast tests for mixed population experiment generation."""

    def test_basic_generation_fast(self):
        """Fast test for basic mixed population generation."""
        exp, info = generate_mixed_population_exp(
            n_neurons=10,  # Reduced from 20
            manifold_fraction=0.6,
            manifold_type="2d_spatial",
            return_info=True,
            **FAST_PARAMS,
        )

        # Check experiment structure
        assert isinstance(exp, Experiment)
        assert exp.n_cells == 10
        assert exp.n_frames == 300  # 30s * 10fps

        # Check population composition
        composition = info["population_composition"]
        assert composition["n_manifold"] == 6  # 60% of 10
        assert composition["n_feature_selective"] == 4  # 40% of 10
        assert composition["manifold_type"] == "2d_spatial"
        assert len(composition["manifold_indices"]) == 6
        assert len(composition["feature_indices"]) == 4

        # Check calcium signals shape
        assert exp.calcium.shape == (10, 300)

        # Check features exist
        assert "x_position" in exp.dynamic_features
        assert "y_position" in exp.dynamic_features
        assert "position_2d" in exp.dynamic_features

    @pytest.mark.parametrize("manifold_type", ["2d_spatial", "circular", "3d_spatial"])
    def test_different_manifold_types_fast(self, manifold_type):
        """Fast test for different manifold types."""
        exp, info = generate_mixed_population_exp(
            n_neurons=8,  # Minimal neurons
            manifold_fraction=0.5,
            manifold_type=manifold_type,
            return_info=True,
            **FAST_PARAMS,
        )

        assert exp.n_cells == 8
        composition = info["population_composition"]
        assert composition["manifold_type"] == manifold_type

        # Check appropriate features exist
        if manifold_type == "2d_spatial":
            assert "x_position" in exp.dynamic_features
            assert "y_position" in exp.dynamic_features
            assert "position_2d" in exp.dynamic_features
        elif manifold_type == "circular":
            assert "circular_angle" in exp.dynamic_features
        elif manifold_type == "3d_spatial":
            assert "x_position" in exp.dynamic_features
            assert "y_position" in exp.dynamic_features
            assert "z_position" in exp.dynamic_features

    def test_feature_selective_params_fast(self):
        """Fast test for feature-selective neuron parameters."""
        exp, info = generate_mixed_population_exp(
            n_neurons=8,
            manifold_fraction=0.0,  # All feature-selective
            n_discrete_features=1,
            n_continuous_features=2,
            return_info=True,
            **FAST_PARAMS,
        )

        assert exp.n_cells == 8
        assert info["population_composition"]["n_feature_selective"] == 8

        # Check behavioral features
        discrete_feats = [f for f in exp.dynamic_features if f.startswith("d_feat_")]
        continuous_feats = [f for f in exp.dynamic_features if f.startswith("c_feat_")]
        assert len(discrete_feats) == 1
        assert len(continuous_feats) == 2

    def test_correlation_effects_minimal(self):
        """Minimal test for correlation effects."""
        # The API doesn't support correlation_effects parameter
        # Just test basic generation works
        exp = generate_mixed_population_exp(
            n_neurons=8, manifold_fraction=0.5, **FAST_PARAMS
        )

        # Verify it produces valid experiment
        assert exp.n_cells == 8

        # Check that neurons have some variance (not all identical)
        ca_signals = [exp.neurons[i].ca.data for i in range(8)]
        correlations = []
        for i in range(7):
            corr = np.corrcoef(ca_signals[i], ca_signals[i + 1])[0, 1]
            correlations.append(corr)

        # Should have some diversity in correlations
        assert np.std(correlations) > 0.01

    def test_calcium_signal_properties(self):
        """Fast test for calcium signal properties."""
        exp = generate_mixed_population_exp(
            n_neurons=6, manifold_fraction=0.5, **FAST_PARAMS
        )

        # Check calcium signals are valid
        for i in range(exp.n_cells):
            ca_data = exp.neurons[i].ca.data
            assert len(ca_data) == 300
            assert np.all(np.isfinite(ca_data))
            assert np.std(ca_data) > 0  # Should have variance

    def test_edge_cases(self):
        """Test edge cases with minimal parameters."""
        # Pure manifold population
        exp, info = generate_mixed_population_exp(
            n_neurons=5, manifold_fraction=1.0, return_info=True, **FAST_PARAMS
        )
        assert info["population_composition"]["n_manifold"] == 5
        assert info["population_composition"]["n_feature_selective"] == 0

        # Pure feature-selective population
        exp, info = generate_mixed_population_exp(
            n_neurons=5, manifold_fraction=0.0, return_info=True, **FAST_PARAMS
        )
        assert info["population_composition"]["n_manifold"] == 0
        assert info["population_composition"]["n_feature_selective"] == 5

        # Single neuron
        exp, info = generate_mixed_population_exp(
            n_neurons=1, manifold_fraction=0.5, return_info=True, **FAST_PARAMS
        )
        assert exp.n_cells == 1
        total = (
            info["population_composition"]["n_manifold"]
            + info["population_composition"]["n_feature_selective"]
        )
        assert total == 1

    def test_feature_configuration(self):
        """Test different feature configurations quickly."""
        # No discrete features
        exp = generate_mixed_population_exp(
            n_neurons=6,
            manifold_fraction=0.5,
            n_discrete_features=0,
            n_continuous_features=3,
            **FAST_PARAMS,
        )

        assert "c_feat_0" in exp.dynamic_features
        assert "c_feat_2" in exp.dynamic_features
        assert "d_feat_0" not in exp.dynamic_features

        # No continuous features
        exp = generate_mixed_population_exp(
            n_neurons=6,
            manifold_fraction=0.5,
            n_discrete_features=2,
            n_continuous_features=0,
            **FAST_PARAMS,
        )

        assert "d_feat_0" in exp.dynamic_features
        assert "d_feat_1" in exp.dynamic_features
        assert "c_feat_0" not in exp.dynamic_features


class TestParameterValidation:
    """Fast parameter validation tests."""

    def test_invalid_manifold_fraction(self):
        """Test invalid manifold fraction."""
        with pytest.raises(ValueError):
            generate_mixed_population_exp(
                n_neurons=10, manifold_fraction=1.5, **FAST_PARAMS  # Invalid
            )

        with pytest.raises(ValueError):
            generate_mixed_population_exp(
                n_neurons=10, manifold_fraction=-0.1, **FAST_PARAMS  # Invalid
            )

    def test_invalid_manifold_type(self):
        """Test invalid manifold type."""
        with pytest.raises(ValueError):
            generate_mixed_population_exp(
                n_neurons=10, manifold_type="invalid_type", **FAST_PARAMS
            )

    def test_invalid_neuron_count(self):
        """Test invalid neuron count."""
        with pytest.raises(ValueError):
            generate_mixed_population_exp(n_neurons=0, **FAST_PARAMS)  # Invalid


class TestIntegrationWithINTENSE:
    """Minimal integration test with INTENSE."""

    def test_mixed_population_with_intense_minimal(self):
        """Minimal test that mixed population works with INTENSE."""
        from driada.intense.pipelines import compute_cell_feat_significance

        # Generate minimal mixed population
        exp = generate_mixed_population_exp(
            n_neurons=5,
            manifold_fraction=0.4,
            duration=60,  # Need longer for INTENSE
            fps=10,
            seed=42,
            verbose=False,
        )

        # Run minimal INTENSE analysis
        result = compute_cell_feat_significance(
            exp,
            cell_bunch=[0, 1, 2],  # Just a few neurons
            mode="stage1",
            n_shuffles_stage1=5,  # Minimal shuffles
            verbose=False,
            ds=5,  # Aggressive downsampling
            allow_mixed_dimensions=True,  # Allow MultiTimeSeries
            find_optimal_delays=False,  # Disable to avoid MultiTimeSeries issue
        )

        # Just verify it runs
        assert len(result) == 4
        stats, _, _, _ = result
        assert isinstance(stats, dict)


class TestPerformanceBenchmarks:
    """Verify optimizations work."""

    def test_generation_under_1s(self):
        """Ensure generation completes quickly."""
        import time

        start = time.time()
        exp = generate_mixed_population_exp(
            n_neurons=10, manifold_fraction=0.5, **FAST_PARAMS
        )
        duration = time.time() - start

        assert duration < 1.0, f"Generation took {duration:.2f}s, should be <1s"
        assert exp.n_cells == 10

    def test_multiple_generations_under_5s(self):
        """Test multiple generations are still fast."""
        import time

        start = time.time()
        for _ in range(5):
            exp = generate_mixed_population_exp(
                n_neurons=8, manifold_fraction=0.5, **FAST_PARAMS
            )
        duration = time.time() - start

        assert duration < 5.0, f"5 generations took {duration:.2f}s, should be <5s"
