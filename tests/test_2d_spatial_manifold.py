import numpy as np
import pytest
from driada.experiment import (
    generate_2d_random_walk,
    gaussian_place_field,
    generate_2d_manifold_neurons,
    generate_2d_manifold_data,
    generate_2d_manifold_exp,
)
from driada.intense.pipelines import compute_cell_feat_significance


class TestGenerate2DRandomWalk:
    """Test 2D random walk trajectory generation."""
    
    def test_basic_generation(self):
        """Test basic 2D random walk generation."""
        length = 1000
        positions = generate_2d_random_walk(length, seed=42)
        
        assert positions.shape == (length, 2)
        assert np.all(positions >= 0)
        assert np.all(positions <= 1)
    
    def test_custom_bounds(self):
        """Test with custom boundaries."""
        length = 500
        bounds = (-2, 2)
        positions = generate_2d_random_walk(length, bounds=bounds, seed=42)
        
        assert positions.shape == (length, 2)
        assert np.all(positions >= bounds[0])
        assert np.all(positions <= bounds[1])
    
    def test_step_size_effect(self):
        """Test that step size affects movement."""
        length = 1000
        
        # Small steps
        pos_small = generate_2d_random_walk(length, step_size=0.001, seed=42)
        # Large steps
        pos_large = generate_2d_random_walk(length, step_size=0.1, seed=42)
        
        # Large steps should cover more distance
        dist_small = np.sum(np.sqrt(np.sum(np.diff(pos_small, axis=0)**2, axis=1)))
        dist_large = np.sum(np.sqrt(np.sum(np.diff(pos_large, axis=0)**2, axis=1)))
        
        assert dist_large > dist_small * 5
    
    def test_momentum_effect(self):
        """Test momentum creates smoother trajectories."""
        length = 1000
        
        # No momentum - more random
        pos_random = generate_2d_random_walk(length, momentum=0.0, seed=42)
        # High momentum - smoother
        pos_smooth = generate_2d_random_walk(length, momentum=0.95, seed=42)
        
        # Calculate direction changes
        def direction_changes(positions):
            velocities = np.diff(positions, axis=0)
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            angle_changes = np.abs(np.diff(angles))
            return np.sum(angle_changes > np.pi/2)  # Count sharp turns
        
        changes_random = direction_changes(pos_random)
        changes_smooth = direction_changes(pos_smooth)
        
        assert changes_random > changes_smooth * 2
    
    def test_reproducibility(self):
        """Test random seed reproducibility."""
        pos1 = generate_2d_random_walk(100, seed=123)
        pos2 = generate_2d_random_walk(100, seed=123)
        pos3 = generate_2d_random_walk(100, seed=456)
        
        np.testing.assert_array_equal(pos1, pos2)
        assert not np.array_equal(pos1, pos3)


class TestGaussianPlaceField:
    """Test Gaussian place field response function."""
    
    def test_basic_response(self):
        """Test basic place field response."""
        # Create positions
        positions = np.array([[0.5, 0.5], [0.6, 0.5], [0.5, 0.6], [0.0, 0.0]])
        center = np.array([0.5, 0.5])
        
        response = gaussian_place_field(positions, center, sigma=0.1)
        
        assert len(response) == len(positions)
        assert response[0] == pytest.approx(1.0)  # At center
        assert response[3] < 0.01  # Far from center
        assert np.all(response >= 0)
        assert np.all(response <= 1)
    
    def test_sigma_effect(self):
        """Test that sigma controls field width."""
        positions = np.array([[0.5, 0.5], [0.6, 0.5]])
        center = np.array([0.5, 0.5])
        
        # Narrow field
        response_narrow = gaussian_place_field(positions, center, sigma=0.05)
        # Wide field
        response_wide = gaussian_place_field(positions, center, sigma=0.2)
        
        # At center, both should be 1
        assert response_narrow[0] == pytest.approx(1.0)
        assert response_wide[0] == pytest.approx(1.0)
        
        # Away from center, wide field should have higher response
        assert response_wide[1] > response_narrow[1]


class TestGenerate2DManifoldNeurons:
    """Test 2D manifold neuron generation."""
    
    def test_basic_generation(self):
        """Test basic place cell generation."""
        n_neurons = 25
        positions = generate_2d_random_walk(1000, seed=42)
        
        firing_rates, centers = generate_2d_manifold_neurons(
            n_neurons, positions, seed=42
        )
        
        assert firing_rates.shape == (n_neurons, len(positions))
        assert centers.shape == (n_neurons, 2)
        assert np.all(firing_rates >= 0)
        assert np.all(centers >= 0)
        assert np.all(centers <= 1)
    
    def test_grid_arrangement(self):
        """Test grid arrangement of place fields."""
        n_neurons = 25  # 5x5 grid
        positions = generate_2d_random_walk(1000, seed=42)
        
        _, centers_grid = generate_2d_manifold_neurons(
            n_neurons, positions, grid_arrangement=True, seed=42
        )
        
        # Check grid-like arrangement (should be roughly evenly spaced)
        # Remove jitter for testing
        centers_sorted = centers_grid[np.lexsort((centers_grid[:, 1], centers_grid[:, 0]))]
        
        # First 5 should have similar x, increasing y
        first_col_x = centers_sorted[:5, 0]
        assert np.std(first_col_x) < 0.1  # Small variation due to jitter
    
    def test_random_arrangement(self):
        """Test random arrangement of place fields."""
        n_neurons = 25
        positions = generate_2d_random_walk(1000, seed=42)
        
        _, centers_random = generate_2d_manifold_neurons(
            n_neurons, positions, grid_arrangement=False, seed=42
        )
        
        # Should be more randomly distributed
        assert np.std(centers_random[:, 0]) > 0.2
        assert np.std(centers_random[:, 1]) > 0.2
    
    def test_firing_rate_modulation(self):
        """Test firing rate modulation by place fields."""
        n_neurons = 10
        positions = np.array([[0.5, 0.5]] * 100 + [[0.0, 0.0]] * 100)
        
        firing_rates, centers = generate_2d_manifold_neurons(
            n_neurons, positions, 
            baseline_rate=0.1, peak_rate=2.0,
            field_sigma=0.1, seed=42
        )
        
        # Find neuron with center closest to (0.5, 0.5)
        distances = np.sqrt(np.sum((centers - [0.5, 0.5])**2, axis=1))
        closest_neuron = np.argmin(distances)
        
        # This neuron should fire more at (0.5, 0.5) than (0, 0)
        mean_rate_center = np.mean(firing_rates[closest_neuron, :100])
        mean_rate_corner = np.mean(firing_rates[closest_neuron, 100:])
        
        assert mean_rate_center > mean_rate_corner * 2


class TestGenerate2DManifoldData:
    """Test complete 2D manifold data generation."""
    
    def test_basic_generation(self):
        """Test basic data generation."""
        n_neurons = 20
        duration = 30  # seconds
        fps = 20
        
        calcium, positions, centers, rates = generate_2d_manifold_data(
            n_neurons, duration, fps, seed=42, verbose=False
        )
        
        n_timepoints = int(duration * fps)
        
        assert calcium.shape == (n_neurons, n_timepoints)
        assert positions.shape == (n_timepoints, 2)
        assert centers.shape == (n_neurons, 2)
        assert rates.shape == (n_neurons, n_timepoints)
        
        # Check calcium properties
        # Note: calcium can be negative due to noise
        assert np.mean(calcium) > 0  # Should have activity
        assert np.std(calcium) > 0   # Should have variability
    
    def test_multiple_environments(self):
        """Test generation with multiple environments (remapping)."""
        n_neurons = 20
        duration = 60
        fps = 20
        n_environments = 3
        
        calcium, positions_list, centers_list, rates = generate_2d_manifold_data(
            n_neurons, duration, fps, 
            n_environments=n_environments,
            seed=42, verbose=False
        )
        
        assert len(positions_list) == n_environments
        assert len(centers_list) == n_environments
        
        # Check remapping occurred
        # At least some centers should be different between environments
        centers_diff = np.sum(np.abs(centers_list[0] - centers_list[1]))
        assert centers_diff > 1.0  # Significant remapping


class TestGenerate2DManifoldExp:
    """Test experiment generation with 2D manifold."""
    
    def test_basic_experiment(self):
        """Test basic experiment generation."""
        n_neurons = 20
        duration = 30
        
        exp, info = generate_2d_manifold_exp(
            n_neurons, duration, verbose=False, seed=42
        )
        
        # Check experiment structure
        assert exp.n_cells == n_neurons
        assert exp.n_frames == int(duration * 20)  # Default fps=20
        
        # Check features
        assert 'x_position' in exp.dynamic_features
        assert 'y_position' in exp.dynamic_features
        assert 'position_2d' in exp.dynamic_features
        
        # Check info
        assert info['manifold_type'] == '2d_spatial'
        assert info['n_neurons'] == n_neurons
        assert 'positions' in info
        assert 'place_field_centers' in info
    
    def test_with_head_direction(self):
        """Test adding head direction feature."""
        exp, info = generate_2d_manifold_exp(
            n_neurons=20, duration=30,
            add_head_direction=True,
            verbose=False, seed=42
        )
        
        assert 'head_direction' in exp.dynamic_features
        assert 'head_direction_circular' in exp.dynamic_features
        
        # Check head direction is properly calculated
        hd = exp.dynamic_features['head_direction'].data
        assert np.all(hd >= 0)
        assert np.all(hd <= 2 * np.pi)
    
    def test_intense_analysis_compatibility(self):
        """Test that generated data works with INTENSE analysis."""
        # Generate experiment with place cells
        # Use 16 neurons (4x4 grid) with larger fields for better coverage
        exp, info = generate_2d_manifold_exp(
            n_neurons=16,       # 4x4 grid
            duration=300,
            field_sigma=0.15,   # Larger fields for better coverage (overlapping)
            step_size=0.04,     # Good exploration
            momentum=0.7,       # Smoother movement
            peak_rate=4.0,      # High peak rate
            baseline_rate=0.05, # Low baseline  
            noise_std=0.02,     # Low noise
            calcium_noise_std=0.05,  # Moderate calcium noise
            decay_time=1.5,     # Reasonable decay
            verbose=False, 
            seed=42
        )
        
        # Test 1: Individual x,y position analysis
        stats1, significance1, _, _ = compute_cell_feat_significance(
            exp,
            feat_bunch=['x_position', 'y_position'],
            mode='two_stage',
            n_shuffles_stage1=30,
            n_shuffles_stage2=200,
            verbose=False
        )
        
        # Count significant neurons for individual features
        significant_neurons = exp.get_significant_neurons()
        individual_selective = sum(1 for features in significant_neurons.values() 
                                 if 'x_position' in features or 'y_position' in features)
        
        # Clear previous results for second test
        exp.selectivity_tables_initialized = False
        exp.stats_tables = {}
        
        # Test 2: 2D position multifeature analysis
        stats2, significance2, _, _ = compute_cell_feat_significance(
            exp,
            feat_bunch=['position_2d'],
            find_optimal_delays=False,  # Must disable for MultiTimeSeries
            mode='two_stage',
            n_shuffles_stage1=30,
            n_shuffles_stage2=200,
            allow_mixed_dimensions=True,
            verbose=False
        )
        
        # Count significant neurons for 2D position
        significant_neurons = exp.get_significant_neurons()
        position_2d_selective = sum(1 for features in significant_neurons.values() 
                                   if 'position_2d' in features)
        
        # Verify 2D position approach detects neurons
        assert position_2d_selective >= individual_selective, \
            f"2D position ({position_2d_selective}) should detect at least as many neurons as individual ({individual_selective})"
        
        # Should detect most place cells
        assert position_2d_selective >= 12, \
            f"Expected at least 12/16 neurons with 2D position, got {position_2d_selective}"
        
        # Individual features should also work
        assert individual_selective >= 8, \
            f"Expected at least 8/16 neurons with individual features, got {individual_selective}"
    
    def test_multiple_environments_experiment(self):
        """Test experiment with multiple environments."""
        exp, info = generate_2d_manifold_exp(
            n_neurons=20, duration=60,
            n_environments=2,
            verbose=False, seed=42
        )
        
        # Should have environment indicator
        assert 'environment' in exp.dynamic_features
        
        env_data = exp.dynamic_features['environment'].data
        assert len(np.unique(env_data)) == 2
        
        # Check info contains list of centers
        assert len(info['place_field_centers']) == 2
    
    def test_parameter_effects(self):
        """Test various parameter settings."""
        # Wide place fields
        exp_wide, _ = generate_2d_manifold_exp(
            n_neurons=10, duration=30,
            field_sigma=0.3,  # Wide fields
            verbose=False, seed=42
        )
        
        # Narrow place fields  
        exp_narrow, _ = generate_2d_manifold_exp(
            n_neurons=10, duration=30,
            field_sigma=0.05,  # Narrow fields
            verbose=False, seed=42
        )
        
        # Wide fields should have more correlated activity
        calcium_wide = exp_wide.calcium
        calcium_narrow = exp_narrow.calcium
        
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


if __name__ == '__main__':
    pytest.main([__file__])