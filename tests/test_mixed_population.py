"""
Comprehensive tests for mixed population generator.

This module tests the generate_mixed_population_exp function which creates
synthetic neural populations combining manifold cells (place cells, head direction)
with feature-selective cells responding to behavioral variables.
"""

import numpy as np
import pytest
from driada.experiment import (
    generate_mixed_population_exp,
    Experiment
)
from driada.information.info_base import TimeSeries, MultiTimeSeries


class TestGenerateMixedPopulationExp:
    """Test mixed population experiment generation."""
    
    def test_basic_generation(self):
        """Test basic mixed population generation."""
        exp, info = generate_mixed_population_exp(
            n_neurons=20,
            manifold_fraction=0.6,
            manifold_type='2d_spatial',
            duration=60,
            seed=42,
            verbose=False, return_info=True)
        
        # Check experiment structure
        assert isinstance(exp, Experiment)
        assert exp.n_cells == 20
        assert exp.n_frames == int(60 * 20)  # 60s * 20fps
        
        # Check population composition
        composition = info['population_composition']
        assert composition['n_manifold'] == 12  # 60% of 20
        assert composition['n_feature_selective'] == 8  # 40% of 20
        assert composition['manifold_type'] == '2d_spatial'
        assert len(composition['manifold_indices']) == 12
        assert len(composition['feature_indices']) == 8
        
        # Check calcium signals shape
        assert exp.calcium.shape == (20, 1200)  # 20 neurons, 1200 timepoints
        
        # Check features exist
        assert 'x_position' in exp.dynamic_features
        assert 'y_position' in exp.dynamic_features
        assert 'position_2d' in exp.dynamic_features
        assert 'd_feat_0' in exp.dynamic_features
        assert 'c_feat_0' in exp.dynamic_features
    
    def test_manifold_types(self):
        """Test all manifold types work correctly."""
        manifold_types = ['circular', '2d_spatial', '3d_spatial']
        
        for manifold_type in manifold_types:
            exp, info = generate_mixed_population_exp(
                n_neurons=15,
                manifold_fraction=0.8,
                manifold_type=manifold_type,
                duration=30,
                seed=42,
                verbose=False,
                return_info=True)
            
            assert info['population_composition']['manifold_type'] == manifold_type
            assert info['population_composition']['n_manifold'] == 12  # 80% of 15
            assert info['population_composition']['n_feature_selective'] == 3
            
            # Check manifold-specific features
            if manifold_type == 'circular':
                assert 'head_direction' in exp.dynamic_features
                assert 'circular_angle' in exp.dynamic_features
                assert isinstance(exp.dynamic_features['circular_angle'], MultiTimeSeries)
            elif manifold_type == '2d_spatial':
                assert 'x_position' in exp.dynamic_features
                assert 'y_position' in exp.dynamic_features
                assert 'position_2d' in exp.dynamic_features
            elif manifold_type == '3d_spatial':
                assert 'x_position' in exp.dynamic_features
                assert 'y_position' in exp.dynamic_features
                assert 'z_position' in exp.dynamic_features
                assert 'position_3d' in exp.dynamic_features
    
    def test_edge_cases(self):
        """Test edge cases for population fractions."""
        # Pure manifold population
        exp, info = generate_mixed_population_exp(
            n_neurons=10,
            manifold_fraction=1.0,
            manifold_type='2d_spatial',
            duration=30,
            seed=42,
            verbose=False, return_info=True)
        
        assert info['population_composition']['n_manifold'] == 10
        assert info['population_composition']['n_feature_selective'] == 0
        assert exp.n_cells == 10
        
        # Pure feature-selective population
        exp, info = generate_mixed_population_exp(
            n_neurons=10,
            manifold_fraction=0.0,
            manifold_type='2d_spatial',
            duration=30,
            seed=42,
            verbose=False, return_info=True)
        
        assert info['population_composition']['n_manifold'] == 0
        assert info['population_composition']['n_feature_selective'] == 10
        assert exp.n_cells == 10
        
        # Single neuron
        exp, info = generate_mixed_population_exp(
            n_neurons=1,
            manifold_fraction=0.5,
            manifold_type='2d_spatial',
            duration=30,
            seed=42,
            verbose=False, return_info=True)
        
        assert exp.n_cells == 1
        total_neurons = info['population_composition']['n_manifold'] + info['population_composition']['n_feature_selective']
        assert total_neurons == 1
    
    def test_correlation_modes(self):
        """Test different correlation modes."""
        correlation_modes = ['independent', 'spatial_correlated', 'feature_correlated']
        
        for mode in correlation_modes:
            exp, info = generate_mixed_population_exp(
                n_neurons=20,
                manifold_fraction=0.5,
                manifold_type='2d_spatial',
                correlation_mode=mode,
                correlation_strength=0.5,
                duration=60,
                seed=42,
                verbose=False,
                return_info=True)
            
            assert info['correlation_applied'] == mode
            if mode == 'independent':
                assert info['correlation_strength'] == 0.0
            else:
                assert info['correlation_strength'] == 0.5
            
            # Check that correlation doesn't break basic structure
            assert exp.n_cells == 20
            assert 'x_position' in exp.dynamic_features
            assert 'c_feat_0' in exp.dynamic_features
    
    def test_feature_configuration(self):
        """Test different feature configurations."""
        # No discrete features
        exp, info = generate_mixed_population_exp(
            n_neurons=15,
            manifold_fraction=0.6,
            n_discrete_features=0,
            n_continuous_features=5,
            duration=30,
            seed=42,
            verbose=False, return_info=True)
        
        assert 'c_feat_0' in exp.dynamic_features
        assert 'c_feat_4' in exp.dynamic_features
        assert 'd_feat_0' not in exp.dynamic_features
        
        # No continuous features
        exp, info = generate_mixed_population_exp(
            n_neurons=15,
            manifold_fraction=0.6,
            n_discrete_features=4,
            n_continuous_features=0,
            duration=30,
            seed=42,
            verbose=False, return_info=True)
        
        assert 'd_feat_0' in exp.dynamic_features
        assert 'd_feat_3' in exp.dynamic_features
        assert 'c_feat_0' not in exp.dynamic_features
        
        # No behavioral features
        exp, info = generate_mixed_population_exp(
            n_neurons=15,
            manifold_fraction=0.6,
            n_discrete_features=0,
            n_continuous_features=0,
            duration=30,
            seed=42,
            verbose=False, return_info=True)
        
        # Should still work with just manifold features
        assert exp.n_cells == 15
        assert 'x_position' in exp.dynamic_features
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid manifold fraction
        with pytest.raises(ValueError, match="manifold_fraction must be between 0.0 and 1.0"):
            generate_mixed_population_exp(manifold_fraction=1.5)
        
        with pytest.raises(ValueError, match="manifold_fraction must be between 0.0 and 1.0"):
            generate_mixed_population_exp(manifold_fraction=-0.1)
        
        # Invalid manifold type
        with pytest.raises(ValueError, match="manifold_type must be"):
            generate_mixed_population_exp(manifold_type='invalid')
        
        # Invalid correlation mode
        with pytest.raises(ValueError, match="Invalid correlation_mode"):
            generate_mixed_population_exp(correlation_mode='invalid')
        
        # Invalid correlation strength
        with pytest.raises(ValueError, match="correlation_strength must be between 0.0 and 1.0"):
            generate_mixed_population_exp(correlation_strength=1.5)
    
    def test_custom_parameters(self):
        """Test custom parameter dictionaries."""
        custom_manifold_params = {
            'field_sigma': 0.2,
            'baseline_rate': 0.05,
            'peak_rate': 3.0,
            'noise_std': 0.02,
            'decay_time': 1.5,
            'calcium_noise_std': 0.05
        }
        
        custom_feature_params = {
            'rate_0': 0.05,
            'rate_1': 2.0,
            'skip_prob': 0.05,
            'hurst': 0.4,
            'ampl_range': (0.8, 1.5),
            'decay_time': 1.5,
            'noise_std': 0.05
        }
        
        exp, info = generate_mixed_population_exp(
            n_neurons=20,
            manifold_fraction=0.5,
            manifold_params=custom_manifold_params,
            feature_params=custom_feature_params,
            duration=60,
            seed=42,
            verbose=False, return_info=True)
        
        # Check parameters were stored
        assert info['parameters']['manifold_params']['field_sigma'] == 0.2
        assert info['parameters']['feature_params']['rate_1'] == 2.0
        
        # Check basic functionality
        assert exp.n_cells == 20
        assert isinstance(exp, Experiment)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        exp1, info1 = generate_mixed_population_exp(
            n_neurons=20,
            seed=123,
            verbose=False,
            return_info=True
        )
        
        exp2, info2 = generate_mixed_population_exp(
            n_neurons=20,
            seed=123,
            verbose=False,
            return_info=True
        )
        
        # Should produce identical results
        # Now calcium is a MultiTimeSeries, so compare the underlying data
        np.testing.assert_array_equal(exp1.calcium.data, exp2.calcium.data)
        
        # Check different seeds produce different results
        exp3, info3 = generate_mixed_population_exp(
            n_neurons=20,
            seed=456,
            verbose=False,
            return_info=True
        )
        
        assert not np.array_equal(exp1.calcium.data, exp3.calcium.data)
    
    def test_info_dictionary_completeness(self):
        """Test that info dictionary contains all expected information."""
        exp, info = generate_mixed_population_exp(
            n_neurons=25,
            manifold_fraction=0.6,
            manifold_type='2d_spatial',
            correlation_mode='spatial_correlated',
            correlation_strength=0.3,
            duration=60,
            seed=42,
            verbose=False, return_info=True)
        
        # Check all required keys exist
        required_keys = [
            'population_composition',
            'manifold_info',
            'feature_selectivity',
            'spatial_data',
            'behavioral_features',
            'correlation_applied',
            'correlation_strength',
            'parameters'
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        # Check population_composition details
        comp = info['population_composition']
        assert 'n_manifold' in comp
        assert 'n_feature_selective' in comp
        assert 'manifold_type' in comp
        assert 'manifold_indices' in comp
        assert 'feature_indices' in comp
        assert 'manifold_fraction' in comp
        
        # Check manifold_info details
        manifold_info = info['manifold_info']
        assert 'manifold_type' in manifold_info
        assert 'positions' in manifold_info
        assert 'place_field_centers' in manifold_info
        assert 'firing_rates' in manifold_info
        
        # Check parameters
        params = info['parameters']
        assert 'manifold_params' in params
        assert 'feature_params' in params
        assert 'n_discrete_features' in params
        assert 'n_continuous_features' in params
    
    def test_timeseries_types(self):
        """Test that features have correct TimeSeries types."""
        exp, info = generate_mixed_population_exp(
            n_neurons=20,
            manifold_fraction=0.5,
            n_discrete_features=2,
            n_continuous_features=2,
            duration=30,
            seed=42,
            verbose=False, return_info=True)
        
        # Check spatial features
        assert isinstance(exp.dynamic_features['x_position'], TimeSeries)
        assert not exp.dynamic_features['x_position'].discrete
        assert isinstance(exp.dynamic_features['position_2d'], MultiTimeSeries)
        
        # Check behavioral features
        assert isinstance(exp.dynamic_features['d_feat_0'], TimeSeries)
        assert exp.dynamic_features['d_feat_0'].discrete
        assert isinstance(exp.dynamic_features['c_feat_0'], TimeSeries)
        assert not exp.dynamic_features['c_feat_0'].discrete
    
    def test_calcium_signal_properties(self):
        """Test properties of generated calcium signals."""
        exp, info = generate_mixed_population_exp(
            n_neurons=30,
            manifold_fraction=0.5,
            duration=120,
            seed=42,
            verbose=False, return_info=True)
        
        # Basic shape checks
        # Now calcium is a MultiTimeSeries
        assert exp.calcium.data.shape[0] == 30
        assert exp.calcium.data.shape[1] == int(120 * 20)  # 120s * 20fps
        
        # Check calcium signals have reasonable properties
        assert np.all(exp.calcium.data >= 0)  # Non-negative
        assert np.mean(exp.calcium.data) > 0  # Some activity
        assert np.std(exp.calcium.data) > 0   # Some variability
        
        # Check manifold vs feature-selective neurons have different indices
        manifold_indices = info['population_composition']['manifold_indices']
        feature_indices = info['population_composition']['feature_indices']
        
        assert len(set(manifold_indices).intersection(set(feature_indices))) == 0
        assert len(manifold_indices) + len(feature_indices) == 30
    
    def test_spatial_correlation_effects(self):
        """Test that spatial correlation actually affects behavioral features."""
        # Generate with no correlation
        exp_indep, info_indep = generate_mixed_population_exp(
            n_neurons=20,
            manifold_fraction=0.5,
            correlation_mode='independent',
            seed=42,
            verbose=False,
            return_info=True
        )
        
        # Generate with spatial correlation
        exp_corr, info_corr = generate_mixed_population_exp(
            n_neurons=20,
            manifold_fraction=0.5,
            correlation_mode='spatial_correlated',
            correlation_strength=0.8,
            seed=42,
            verbose=False,
            return_info=True
        )
        
        # Features should be different due to correlation
        feat_indep = exp_indep.dynamic_features['c_feat_0'].data
        feat_corr = exp_corr.dynamic_features['c_feat_0'].data
        
        # They should be different (correlation changes the features)
        correlation = np.corrcoef(feat_indep, feat_corr)[0, 1]
        assert correlation < 0.99  # Not identical
    
    def test_integration_with_existing_functions(self):
        """Test that mixed population integrates properly with existing functions."""
        from driada.intense.pipelines import compute_cell_feat_significance
        
        # Generate small mixed population for quick testing
        exp, info = generate_mixed_population_exp(
            n_neurons=10,
            manifold_fraction=0.5,
            duration=60,
            seed=42,
            verbose=False, return_info=True)
        
        # Should be able to run INTENSE analysis without errors
        stats, significance, analysis_info, results = compute_cell_feat_significance(
            exp,
            feat_bunch=['x_position', 'd_feat_0'],
            mode='two_stage',
            n_shuffles_stage1=10,
            n_shuffles_stage2=50,
            verbose=False
        )
        
        # Basic checks - should complete without errors
        assert stats is not None
        assert significance is not None
        assert len(stats) > 0  # Should have some results


class TestMixedPopulationIntegration:
    """Test integration of mixed population with DRIADA ecosystem."""
    
    def test_experiment_methods_work(self):
        """Test that Experiment methods work with mixed population."""
        exp, info = generate_mixed_population_exp(
            n_neurons=15,
            duration=30,
            seed=42,
            verbose=False, return_info=True)
        
        # Test basic Experiment methods
        assert exp.n_cells == 15
        assert exp.n_frames > 0
        assert len(exp.dynamic_features) > 0
        
        # Test feature access
        feature_names = list(exp.dynamic_features.keys())
        assert len(feature_names) > 0
        
        # Test that we can access calcium data
        # Now calcium is a MultiTimeSeries, access the underlying data
        calcium_subset = exp.calcium.data[:5, :100]  # First 5 neurons, first 100 timepoints
        assert calcium_subset.shape == (5, 100)
    
    def test_mixed_population_with_intense(self):
        """Test mixed population works with INTENSE analysis pipeline."""
        exp, info = generate_mixed_population_exp(
            n_neurons=20,
            manifold_fraction=0.6,
            n_discrete_features=2,
            n_continuous_features=2,
            manifold_params={
                'field_sigma': 0.15,  # Tighter place fields
                'peak_rate': 5.0,     # Higher peak rate
                'baseline_rate': 0.05,
                'noise_std': 0.02,
                'decay_time': 1.5,
                'calcium_noise_std': 0.05
            },
            duration=300,  # Increased duration for better statistics
            seed=42,
            verbose=False, return_info=True)
        
        # Test with position features (should detect manifold cells)
        from driada.intense.pipelines import compute_cell_feat_significance
        
        stats, significance, analysis_info, results = compute_cell_feat_significance(
            exp,
            feat_bunch=['x_position', 'y_position'],
            mode='two_stage',
            n_shuffles_stage1=20,
            n_shuffles_stage2=100,
            verbose=False
        )
        
        # Should detect some spatial selectivity
        significant_neurons = exp.get_significant_neurons()
        assert len(significant_neurons) > 0  # Should find some selective neurons
        
        # Check that manifold cells are more likely to be spatially selective
        manifold_indices = set(info['population_composition']['manifold_indices'])
        spatial_selective_neurons = set()
        
        for cell_id, features in significant_neurons.items():
            if any(feat in ['x_position', 'y_position'] for feat in features):
                spatial_selective_neurons.add(cell_id)
        
        # At least some spatially selective neurons should be from manifold population
        manifold_selective = spatial_selective_neurons.intersection(manifold_indices)
        assert len(manifold_selective) > 0


class TestMixedPopulationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_neurons(self):
        """Test behavior with zero neurons."""
        with pytest.raises((ValueError, AssertionError)):
            generate_mixed_population_exp(n_neurons=0, verbose=False)
    
    def test_very_short_duration(self):
        """Test with very short duration."""
        exp, info = generate_mixed_population_exp(
            n_neurons=5,
            duration=1,  # 1 second
            seed=42,
            verbose=False, return_info=True)
        
        assert exp.n_frames == 20  # 1s * 20fps
        assert exp.n_cells == 5
    
    
    def test_extreme_correlation_strength(self):
        """Test with extreme correlation strengths."""
        # Very weak correlation
        exp, info = generate_mixed_population_exp(
            n_neurons=10,
            correlation_mode='spatial_correlated',
            correlation_strength=0.01,
            seed=42,
            verbose=False, return_info=True)
        assert info['correlation_strength'] == 0.01
        
        # Very strong correlation
        exp, info = generate_mixed_population_exp(
            n_neurons=10,
            correlation_mode='spatial_correlated',
            correlation_strength=0.99,
            seed=42,
            verbose=False, return_info=True)
        assert info['correlation_strength'] == 0.99


if __name__ == '__main__':
    pytest.main([__file__])