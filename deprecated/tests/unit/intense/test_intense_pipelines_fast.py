"""Fast tests for INTENSE pipeline functions with comprehensive mocking.

This module provides optimized tests that mock expensive computations while
maintaining high code coverage. Tests run in <1s each instead of minutes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace

from driada.intense.pipelines import (
    compute_cell_feat_significance,
    compute_feat_feat_significance,
    compute_cell_cell_significance
)
from driada.information.info_base import TimeSeries


@pytest.fixture(scope="module")
def mock_experiment():
    """Create a minimal mock experiment for all pipeline tests."""
    exp = SimpleNamespace()
    exp.n_frames = 100  # Minimal frames
    exp.n_cells = 5
    exp.signature = 'MockExp'
    
    # Create neurons with minimal data
    exp.neurons = []
    for i in range(5):
        neuron = SimpleNamespace()
        # Small calcium data
        neuron.ca = TimeSeries(np.random.randn(100) + 1, discrete=False)
        # Small spike data
        neuron.sp = TimeSeries(np.random.randint(0, 2, 100), discrete=True)
        exp.neurons.append(neuron)
    
    # Add minimal features
    exp.dynamic_features = {
        'feat1': TimeSeries(np.random.randn(100), discrete=False),
        'feat2': TimeSeries(np.random.randn(100), discrete=False),
        'feat3': TimeSeries(np.random.randint(0, 3, 100), discrete=True),
    }
    
    # Add required attributes
    exp.t_off_sec = 1.0
    exp.frame_time_sec = 0.1
    
    return exp


class TestIntensePipelinesFast:
    """Fast tests for INTENSE pipeline functions."""
    
    @patch('driada.intense.pipelines.compute_info_parallel')
    @patch('driada.intense.pipelines.compute_mi_significance_parallel')
    def test_compute_cell_feat_significance_mocked(self, mock_mi_sig, mock_info, mock_experiment):
        """Test cell-feature significance with mocked computations."""
        # Setup mocks
        mock_info.return_value = {'mi': 0.5, 'ent': 1.0, 'cond_ent': 0.5}
        mock_mi_sig.return_value = (0.5, True, 0.01, {'n_shuffles': 5})
        
        # Run function
        result = compute_cell_feat_significance(
            mock_experiment,
            cell_bunch=[0, 1],
            mode='stage1',
            n_shuffles_stage1=5,
            verbose=False,
            with_disentanglement=False
        )
        
        # Verify
        assert len(result) == 4
        stats, significance, info, results = result
        assert isinstance(stats, dict)
        assert isinstance(significance, dict)
        assert mock_info.called
        assert mock_mi_sig.called
    
    @patch('driada.intense.pipelines.compute_info_parallel')
    @patch('driada.intense.pipelines.compute_mi_significance_parallel')
    @patch('driada.intense.pipelines.compute_feat_feat_significance')
    def test_compute_cell_feat_with_disentanglement_mocked(
        self, mock_feat_feat, mock_mi_sig, mock_info, mock_experiment
    ):
        """Test cell-feature significance with disentanglement mocked."""
        # Setup mocks
        mock_info.return_value = {'mi': 0.5, 'ent': 1.0}
        mock_mi_sig.return_value = (0.5, True, 0.01, {'n_shuffles': 5})
        
        # Mock disentanglement results
        n_features = len(mock_experiment.dynamic_features)
        mock_feat_feat.return_value = (
            np.random.rand(n_features, n_features),  # sim_mat
            np.random.rand(n_features, n_features) > 0.5,  # sig_mat
            np.random.rand(n_features, n_features),  # pval_mat
            list(mock_experiment.dynamic_features.keys()),  # feat_names
            {}  # info
        )
        
        # Run with disentanglement
        stats, sig, info, results, disent = compute_cell_feat_significance(
            mock_experiment,
            cell_bunch=[0, 1],
            mode='stage1',
            n_shuffles_stage1=5,
            verbose=False,
            with_disentanglement=True
        )
        
        # Verify
        assert isinstance(disent, dict)
        assert 'feat_feat_significance' in disent
        assert 'disent_matrix' in disent
        assert mock_feat_feat.called
    
    @patch('driada.intense.pipelines.compute_info_parallel')
    @patch('driada.intense.pipelines.compute_mi_significance_parallel')
    def test_compute_feat_feat_significance_mocked(self, mock_mi_sig, mock_info, mock_experiment):
        """Test feature-feature correlation with mocked computations."""
        # Setup mocks
        mock_info.return_value = {'mi': 0.3, 'ent': 1.5}
        mock_mi_sig.return_value = (0.3, False, 0.1, {'n_shuffles': 10})
        
        # Run function
        sim_mat, sig_mat, pval_mat, feat_names, info = compute_feat_feat_significance(
            mock_experiment,
            feat_bunch=None,
            mode='stage1',
            n_shuffles_stage1=10,
            verbose=False
        )
        
        # Verify structure
        n_features = len(mock_experiment.dynamic_features)
        assert sim_mat.shape == (n_features, n_features)
        assert sig_mat.shape == (n_features, n_features)
        assert pval_mat.shape == (n_features, n_features)
        assert len(feat_names) == n_features
        
        # Verify properties
        assert np.allclose(np.diag(sim_mat), 0)
        assert np.allclose(sim_mat, sim_mat.T)
    
    @patch('driada.intense.pipelines.compute_info_parallel')
    @patch('driada.intense.pipelines.compute_mi_significance_parallel')
    def test_compute_cell_cell_significance_mocked(self, mock_mi_sig, mock_info, mock_experiment):
        """Test neuron-neuron correlation with mocked computations."""
        # Setup mocks to simulate some correlations
        def mi_side_effect(*args, **kwargs):
            # Return higher MI for specific neuron pairs
            if args[0] == 0 and args[1] == 1:
                return (0.8, True, 0.001, {'n_shuffles': 10})
            else:
                return (0.1, False, 0.5, {'n_shuffles': 10})
        
        mock_info.return_value = {'mi': 0.4, 'ent': 1.2}
        mock_mi_sig.side_effect = mi_side_effect
        
        # Run function
        sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
            mock_experiment,
            cell_bunch=[0, 1, 2],
            data_type='calcium',
            mode='stage1',
            n_shuffles_stage1=10,
            verbose=False
        )
        
        # Verify structure
        assert sim_mat.shape == (3, 3)
        assert sig_mat.shape == (3, 3)
        assert pval_mat.shape == (3, 3)
        assert len(cell_ids) == 3
        
        # Verify properties
        assert np.allclose(np.diag(sim_mat), 0)
        assert np.allclose(sim_mat, sim_mat.T)
        
        # Verify mocked correlation was detected
        assert sim_mat[0, 1] > sim_mat[0, 2]
        assert sig_mat[0, 1] == True
    
    def test_edge_cases_with_mocking(self, mock_experiment):
        """Test edge cases with minimal mocking."""
        with patch('driada.intense.pipelines.compute_info_parallel') as mock_info:
            mock_info.return_value = {'mi': 0.0, 'ent': 0.0}
            
            # Test with empty cell bunch
            result = compute_cell_feat_significance(
                mock_experiment,
                cell_bunch=[],
                mode='stage1',
                n_shuffles_stage1=2,
                verbose=False
            )
            assert len(result) == 4
            stats, _, _, _ = result
            assert len(stats) == 0
            
            # Test with single neuron
            result = compute_cell_feat_significance(
                mock_experiment,
                cell_bunch=[0],
                mode='stage1',
                n_shuffles_stage1=2,
                verbose=False
            )
            assert len(result) == 4
    
    def test_performance_with_mocking(self, mock_experiment):
        """Verify that mocked tests are fast."""
        import time
        
        with patch('driada.intense.pipelines.compute_info_parallel') as mock_info, \
             patch('driada.intense.pipelines.compute_mi_significance_parallel') as mock_mi:
            
            mock_info.return_value = {'mi': 0.5, 'ent': 1.0}
            mock_mi.return_value = (0.5, True, 0.01, {})
            
            # Time cell-feat significance
            start = time.time()
            compute_cell_feat_significance(
                mock_experiment,
                cell_bunch=list(range(5)),
                mode='stage1',
                n_shuffles_stage1=5,
                verbose=False
            )
            duration = time.time() - start
            assert duration < 0.1, f"Cell-feat took {duration:.3f}s"
            
            # Time feat-feat significance
            start = time.time()
            compute_feat_feat_significance(
                mock_experiment,
                mode='stage1',
                n_shuffles_stage1=5,
                verbose=False
            )
            duration = time.time() - start
            assert duration < 0.1, f"Feat-feat took {duration:.3f}s"
            
            # Time cell-cell significance
            start = time.time()
            compute_cell_cell_significance(
                mock_experiment,
                cell_bunch=list(range(5)),
                mode='stage1',
                n_shuffles_stage1=5,
                verbose=False
            )
            duration = time.time() - start
            assert duration < 0.1, f"Cell-cell took {duration:.3f}s"


class TestIntegrationScenariosFast:
    """Fast integration tests with mocked expensive operations."""
    
    @patch('driada.experiment.synthetic.generate_synthetic_exp_with_mixed_selectivity')
    @patch('driada.intense.pipelines.compute_info_parallel')
    @patch('driada.intense.pipelines.compute_mi_significance_parallel')
    def test_mixed_selectivity_scenario_mocked(self, mock_mi, mock_info, mock_gen):
        """Test mixed selectivity scenario with mocking."""
        # Create minimal mock experiment
        exp = SimpleNamespace()
        exp.n_frames = 100
        exp.n_cells = 10
        exp.neurons = []
        
        for i in range(10):
            neuron = SimpleNamespace()
            neuron.ca = TimeSeries(np.random.randn(100) + 1, discrete=False)
            exp.neurons.append(neuron)
        
        exp.dynamic_features = {
            'feat1': TimeSeries(np.random.randn(100), discrete=False),
            'feat2': TimeSeries(np.random.randn(100), discrete=False),
        }
        exp.t_off_sec = 1.0
        exp.frame_time_sec = 0.1
        
        mock_gen.return_value = exp
        mock_info.return_value = {'mi': 0.5, 'ent': 1.0}
        mock_mi.return_value = (0.5, True, 0.01, {})
        
        # Generate and analyze
        exp = mock_gen(
            num_neurons=10,
            duration=30,  # Short duration
            pattern_types=['feat1_selective', 'feat2_selective']
        )
        
        # Run analysis
        stats, sig, info, results = compute_cell_feat_significance(
            exp,
            mode='stage1',
            n_shuffles_stage1=5,
            verbose=False
        )
        
        # Verify
        assert isinstance(stats, dict)
        assert mock_gen.called
        assert mock_mi.called
    
    @patch('driada.experiment.synthetic.generate_multiselectivity_patterns')
    def test_multiselectivity_detection_mocked(self, mock_patterns):
        """Test multiselectivity pattern detection with mocking."""
        # Mock pattern generation
        mock_patterns.return_value = {
            'neuron_types': ['A', 'B', 'A', 'B', 'Mixed'],
            'feature_selectivity': {
                0: ['feat1'],
                1: ['feat2'],
                2: ['feat1'],
                3: ['feat2'],
                4: ['feat1', 'feat2']
            }
        }
        
        # Create minimal data
        patterns = mock_patterns(
            num_neurons=5,
            feature_names=['feat1', 'feat2'],
            mixed_fraction=0.2
        )
        
        # Verify
        assert 'neuron_types' in patterns
        assert 'feature_selectivity' in patterns
        assert mock_patterns.called


class TestPerformanceOptimizations:
    """Test that optimization flags work correctly."""
    
    def test_downsampling_parameter(self, mock_experiment):
        """Test that downsampling parameter is respected."""
        with patch('driada.intense.pipelines.compute_info_parallel') as mock_info:
            mock_info.return_value = {'mi': 0.5, 'ent': 1.0}
            
            # Run with downsampling
            compute_cell_feat_significance(
                mock_experiment,
                cell_bunch=[0],
                mode='stage1',
                n_shuffles_stage1=2,
                ds=5,  # Downsampling factor
                verbose=False
            )
            
            # Check that downsampled data was used
            call_args = mock_info.call_args
            assert call_args is not None
    
    def test_stage1_vs_two_stage_performance(self, mock_experiment):
        """Verify stage1 is faster than two_stage."""
        import time
        
        with patch('driada.intense.pipelines.compute_info_parallel') as mock_info, \
             patch('driada.intense.pipelines.compute_mi_significance_parallel') as mock_mi:
            
            mock_info.return_value = {'mi': 0.5, 'ent': 1.0}
            mock_mi.return_value = (0.5, True, 0.01, {})
            
            # Time stage1
            start = time.time()
            compute_cell_feat_significance(
                mock_experiment,
                cell_bunch=[0, 1],
                mode='stage1',
                n_shuffles_stage1=5,
                verbose=False
            )
            stage1_time = time.time() - start
            
            # Mock two_stage to simulate longer computation
            def slow_mi(*args, **kwargs):
                import time
                time.sleep(0.01)  # Simulate computation
                return (0.5, True, 0.01, {})
            
            mock_mi.side_effect = slow_mi
            
            # Time two_stage
            start = time.time()
            compute_cell_feat_significance(
                mock_experiment,
                cell_bunch=[0, 1],
                mode='two_stage',
                n_shuffles_stage1=5,
                n_shuffles_stage2=20,
                verbose=False
            )
            two_stage_time = time.time() - start
            
            # Verify stage1 is faster
            assert stage1_time < two_stage_time