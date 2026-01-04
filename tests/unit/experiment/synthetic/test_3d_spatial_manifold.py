import numpy as np
import pytest
from driada.experiment import (
    generate_3d_random_walk,
    gaussian_place_field_3d,
    generate_3d_manifold_neurons,
    generate_3d_manifold_data,
    generate_3d_manifold_exp,
)
from driada.intense.pipelines import compute_cell_feat_significance


class TestGenerate3DRandomWalk:
    """Test 3D random walk trajectory generation."""

    def test_basic_generation(self):
        """Test basic 3D random walk generation."""
        length = 1000
        positions = generate_3d_random_walk(length, seed=42)

        assert positions.shape == (3, length)
        assert np.all(positions >= 0)
        assert np.all(positions <= 1)

    def test_custom_bounds(self):
        """Test with custom boundaries."""
        length = 500
        bounds = (-2, 2)
        positions = generate_3d_random_walk(length, bounds=bounds, seed=42)

        assert positions.shape == (3, length)
        assert np.all(positions >= bounds[0])
        assert np.all(positions <= bounds[1])

    def test_step_size_effect(self):
        """Test that step size affects movement."""
        length = 1000

        # Small steps
        pos_small = generate_3d_random_walk(length, step_size=0.001, seed=42)
        # Large steps
        pos_large = generate_3d_random_walk(length, step_size=0.1, seed=42)

        # Large steps should cover more distance
        dist_small = np.sum(np.sqrt(np.sum(np.diff(pos_small, axis=1) ** 2, axis=0)))
        dist_large = np.sum(np.sqrt(np.sum(np.diff(pos_large, axis=1) ** 2, axis=0)))

        assert dist_large > dist_small * 5

    def test_momentum_effect(self):
        """Test momentum creates smoother trajectories."""
        length = 1000

        # No momentum - more random
        pos_random = generate_3d_random_walk(length, momentum=0.0, seed=42)
        # High momentum - smoother
        pos_smooth = generate_3d_random_walk(length, momentum=0.95, seed=42)

        # Calculate direction changes in 3D
        def direction_changes(positions):
            velocities = np.diff(positions, axis=1)
            # Normalize velocities
            norms = np.linalg.norm(velocities, axis=0)
            norms[norms == 0] = 1  # Avoid division by zero
            velocities_norm = velocities / norms[np.newaxis, :]

            # Calculate angle changes using dot product
            dot_products = np.sum(
                velocities_norm[:, :-1] * velocities_norm[:, 1:], axis=0
            )
            dot_products = np.clip(dot_products, -1, 1)  # Numerical stability
            angles = np.arccos(dot_products)
            return np.sum(angles > np.pi / 2)  # Count sharp turns

        changes_random = direction_changes(pos_random)
        changes_smooth = direction_changes(pos_smooth)

        assert changes_random > changes_smooth * 2

    def test_reproducibility(self):
        """Test random seed reproducibility."""
        pos1 = generate_3d_random_walk(100, seed=123)
        pos2 = generate_3d_random_walk(100, seed=123)
        pos3 = generate_3d_random_walk(100, seed=456)

        np.testing.assert_array_equal(pos1, pos2)
        assert not np.array_equal(pos1, pos3)


class TestGaussianPlaceField3D:
    """Test 3D Gaussian place field response function."""

    def test_basic_response(self):
        """Test basic place field response in 3D."""
        # Create positions
        positions = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.6, 0.5, 0.5],
                [0.5, 0.6, 0.5],
                [0.5, 0.5, 0.6],
                [0.0, 0.0, 0.0],
            ]
        )
        center = np.array([0.5, 0.5, 0.5])

        response = gaussian_place_field_3d(positions, center, sigma=0.1)

        assert len(response) == len(positions)
        assert response[0] == pytest.approx(1.0)  # At center
        assert response[4] < 0.01  # Far from center
        assert np.all(response >= 0)
        assert np.all(response <= 1)

        # Check that responses at same distance are equal
        assert response[1] == pytest.approx(response[2])
        assert response[2] == pytest.approx(response[3])

    def test_sigma_effect(self):
        """Test that sigma controls field width in 3D."""
        positions = np.array([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5]])
        center = np.array([0.5, 0.5, 0.5])

        # Narrow field
        response_narrow = gaussian_place_field_3d(positions, center, sigma=0.05)
        # Wide field
        response_wide = gaussian_place_field_3d(positions, center, sigma=0.2)

        # At center, both should be 1
        assert response_narrow[0] == pytest.approx(1.0)
        assert response_wide[0] == pytest.approx(1.0)

        # Away from center, wide field should have higher response
        assert response_wide[1] > response_narrow[1]


class TestGenerate3DManifoldNeurons:
    """Test 3D manifold neuron generation."""

    def test_basic_generation(self):
        """Test basic 3D place cell generation."""
        n_neurons = 27  # 3x3x3 grid
        positions = generate_3d_random_walk(1000, seed=42)

        firing_rates, centers = generate_3d_manifold_neurons(
            n_neurons, positions, seed=42
        )

        assert firing_rates.shape == (n_neurons, positions.shape[1])
        assert centers.shape == (n_neurons, 3)
        assert np.all(firing_rates >= 0)
        assert np.all(centers >= 0)
        assert np.all(centers <= 1)

    def test_grid_arrangement(self):
        """Test 3D grid arrangement of place fields."""
        n_neurons = 27  # 3x3x3 grid
        positions = generate_3d_random_walk(1000, seed=42)

        _, centers_grid = generate_3d_manifold_neurons(
            n_neurons, positions, grid_arrangement=True, seed=42
        )

        # Check that centers form a 3D grid structure
        # Sort centers by x, then y, then z
        centers_sorted = centers_grid[
            np.lexsort((centers_grid[:, 2], centers_grid[:, 1], centers_grid[:, 0]))
        ]

        # First 9 should have similar x coordinate (first layer)
        first_layer_x = centers_sorted[:9, 0]
        assert np.std(first_layer_x) < 0.1  # Small variation due to jitter

    def test_random_arrangement(self):
        """Test random arrangement of place fields in 3D."""
        n_neurons = 27
        positions = generate_3d_random_walk(1000, seed=42)

        _, centers_random = generate_3d_manifold_neurons(
            n_neurons, positions, grid_arrangement=False, seed=42
        )

        # Should be more randomly distributed
        assert np.std(centers_random[:, 0]) > 0.2
        assert np.std(centers_random[:, 1]) > 0.2
        assert np.std(centers_random[:, 2]) > 0.2

    def test_firing_rate_modulation(self):
        """Test firing rate modulation by 3D place fields."""
        n_neurons = 10
        # Create positions at specific locations
        positions = np.array([[0.5, 0.5, 0.5]] * 100 + [[0.0, 0.0, 0.0]] * 100)

        # Generate neurons with one forced to be at center
        firing_rates, centers = generate_3d_manifold_neurons(
            n_neurons,
            positions.T,  # Transpose to (3, n_timepoints)
            baseline_rate=0.1,
            peak_rate=2.0,
            field_sigma=0.1,
            seed=42,
        )

        # Manually set first neuron center to test location
        centers[0] = np.array([0.5, 0.5, 0.5])

        # Recalculate firing rates for first neuron
        place_response = gaussian_place_field_3d(positions, centers[0], 0.1)
        firing_rates[0] = 0.1 + (2.0 - 0.1) * place_response

        # First neuron should fire more at (0.5, 0.5, 0.5) than (0, 0, 0)
        mean_rate_center = np.mean(firing_rates[0, :100])
        mean_rate_corner = np.mean(firing_rates[0, 100:])

        # At center should be near peak rate, at corner near baseline
        assert mean_rate_center > 1.5  # Should be close to 2.0
        assert mean_rate_corner < 0.2  # Should be close to 0.1
        assert mean_rate_center > mean_rate_corner * 5


class TestGenerate3DManifoldData:
    """Test complete 3D manifold data generation."""

    def test_basic_generation(self):
        """Test basic 3D data generation."""
        n_neurons = 20
        duration = 30  # seconds
        fps = 20

        calcium, positions, centers, rates = generate_3d_manifold_data(
            n_neurons, duration, fps, seed=42, verbose=False
        )

        n_timepoints = int(duration * fps)

        assert calcium.shape == (n_neurons, n_timepoints)
        assert positions.shape == (3, n_timepoints)
        assert centers.shape == (n_neurons, 3)
        assert rates.shape == (n_neurons, n_timepoints)

        # Check calcium properties
        assert np.mean(calcium) > 0  # Should have activity
        assert np.std(calcium) > 0  # Should have variability


class TestGenerate3DManifoldExp:
    """Test experiment generation with 3D manifold."""

    def test_basic_experiment(self):
        """Test basic 3D experiment generation."""
        n_neurons = 27  # 3x3x3
        duration = 30

        exp, info = generate_3d_manifold_exp(
            n_neurons, duration, verbose=False, seed=42, return_info=True
        )

        # Check experiment structure
        assert exp.n_cells == n_neurons
        assert exp.n_frames == int(duration * 20)  # Default fps=20

        # Check features
        assert "x" in exp.dynamic_features
        assert "y" in exp.dynamic_features
        assert "z" in exp.dynamic_features
        assert "position_3d" in exp.dynamic_features

        # Check info
        assert info["manifold_type"] == "3d_spatial"
        assert info["n_neurons"] == n_neurons
        assert "positions" in info
        assert "place_field_centers" in info
        assert info["positions"].shape == (3, exp.n_frames)

    def test_intense_analysis_compatibility(self):
        """Test that generated 3D data works with INTENSE analysis."""
        # Generate experiment with 3D place cells
        # Use parameters that ensure strong selectivity
        exp = generate_3d_manifold_exp(
            n_neurons=8,  # 2x2x2 grid for faster tests
            duration=120,  # Longer duration for better statistics
            fps=20,  # Higher fps for more samples
            field_sigma=0.15,  # Tighter fields for stronger selectivity
            step_size=0.04,  # Smaller steps for smoother trajectory
            momentum=0.7,  # More momentum for smoother paths
            peak_rate=5.0,  # Higher peak rate for stronger signal
            baseline_rate=0.01,  # Very low baseline for high SNR
            noise_std=0.001,  # Minimal noise
            calcium_noise_std=0.01,  # Very low calcium noise
            decay_time=1.0,  # Faster decay for sharper responses
            verbose=False,
            seed=42,
        )

        # Test 1: Individual x, y, z position analysis
        stats1, significance1, _, _ = compute_cell_feat_significance(
            exp,
            feat_bunch=["x", "y", "z"],
            mode="two_stage",
            n_shuffles_stage1=10,
            n_shuffles_stage2=200,  # Increased for more reliable p-values
            pval_thr=0.05,  # Less conservative threshold for small sample test
            ds=5,  # Downsample by 5x
            enable_parallelization=False,  # Disable parallelization
            verbose=False,
        )

        # Count significant neurons for individual features
        significant_neurons = exp.get_significant_neurons()
        individual_selective = sum(
            1
            for features in significant_neurons.values()
            if any(f in features for f in ["x", "y", "z"])
        )

        # Store the count before clearing
        individual_count = individual_selective

        # Clear previous results for second test
        exp.selectivity_tables_initialized = False
        exp.stats_tables = {}

        # Test 2: 3D position multifeature analysis
        stats2, significance2, _, _ = compute_cell_feat_significance(
            exp,
            feat_bunch=["position_3d"],
            find_optimal_delays=False,  # Must disable for MultiTimeSeries
            mode="two_stage",
            n_shuffles_stage1=10,
            n_shuffles_stage2=200,  # Increased for more reliable p-values
            pval_thr=0.05,  # Less conservative threshold for small sample test
            ds=5,  # Downsample by 5x
            enable_parallelization=False,  # Disable parallelization
            allow_mixed_dimensions=True,
            verbose=False,
        )

        # Count significant neurons for 3D position
        significant_neurons = exp.get_significant_neurons()
        position_3d_selective = sum(
            1 for features in significant_neurons.values() if "position_3d" in features
        )

        # Verify both approaches detect selective neurons
        # Note: MultiTimeSeries approach might be more conservative than individual features
        assert (
            position_3d_selective >= 1
        ), f"3D position should detect at least 1 neuron, got {position_3d_selective}"

        # Individual features should detect some selective neurons
        assert (
            individual_count >= 3
        ), f"Expected at least 3/8 neurons with individual features, got {individual_count}"

        # Both methods should find selectivity
        print(f"Individual features detected: {individual_count} neurons")
        print(f"3D position detected: {position_3d_selective} neurons")

    def test_parameter_effects(self):
        """Test various parameter settings for 3D."""
        # Wide place fields
        exp_wide, _ = generate_3d_manifold_exp(
            n_neurons=10,
            duration=30,
            field_sigma=0.3,  # Wide fields
            verbose=False,
            seed=42,
            return_info=True,
        )

        # Narrow place fields
        exp_narrow, _ = generate_3d_manifold_exp(
            n_neurons=10,
            duration=30,
            field_sigma=0.05,  # Narrow fields
            verbose=False,
            seed=42,
            return_info=True,
        )

        # Wide fields should have more correlated activity
        calcium_wide = exp_wide.calcium.data
        calcium_narrow = exp_narrow.calcium.data

        # Calculate mean pairwise correlation
        def mean_correlation(data):
            corr_matrix = np.corrcoef(data)
            # Get upper triangle without diagonal
            upper_idx = np.triu_indices(len(data), k=1)
            return np.mean(corr_matrix[upper_idx])

        corr_wide = mean_correlation(calcium_wide)
        corr_narrow = mean_correlation(calcium_narrow)

        # Wide fields overlap more, so higher correlation
        assert corr_wide > corr_narrow

    def test_default_neuron_count(self):
        """Test that default neuron count is 125 (5x5x5 grid)."""
        exp, info = generate_3d_manifold_exp(
            duration=30, verbose=False, seed=42, return_info=True
        )

        assert exp.n_cells == 125
        assert info["n_neurons"] == 125


if __name__ == "__main__":
    pytest.main([__file__])
