"""Fast tests for the INTENSE visual module - no actual plotting."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from types import SimpleNamespace

# Import functions to test
from driada.intense.visual import (
    plot_pc_activity,
    plot_neuron_feature_density,
    plot_neuron_feature_pair,
    plot_disentanglement_heatmap,
    plot_disentanglement_summary,
    plot_selectivity_heatmap,
)


class TestVisualFunctions:
    """Test visual functions with mocked matplotlib."""
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_pc_activity(self, mock_subplots, mock_figure, visual_experiment):
        """Test place cell activity plotting - logic only."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Test that function runs and returns axes
        ax = plot_pc_activity(visual_experiment, 0)
        assert ax is not None
        
        # Verify key methods were called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called
        mock_ax.plot.assert_called()
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_neuron_feature_density(self, mock_subplots, visual_experiment):
        """Test neuron feature density plotting - logic only."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        ax = plot_neuron_feature_density(visual_experiment, 'calcium', 0, 'speed', ind1=0, ind2=100)
        assert ax is not None
        
        # Should create density plot
        mock_ax.contourf.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    def test_plot_neuron_feature_pair(self, mock_figure, visual_experiment):
        """Test neuron feature pair plotting - logic only."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        fig = plot_neuron_feature_pair(visual_experiment, 0, 'speed', ind1=0, ind2=100)
        assert fig is not None
        assert mock_fig.add_subplot.called
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_disentanglement_heatmap(self, mock_subplots):
        """Test disentanglement heatmap - logic only."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Create test data
        n_features = 4
        disent_matrix = np.random.rand(n_features, n_features) * 20
        count_matrix = np.random.randint(10, 50, (n_features, n_features))
        feat_names = [f'feat{i}' for i in range(n_features)]
        
        fig, ax = plot_disentanglement_heatmap(disent_matrix, count_matrix, feat_names)
        assert fig is not None
        assert ax is not None
        
        # Should create heatmap
        mock_ax.imshow.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    def test_plot_disentanglement_summary(self, mock_figure):
        """Test disentanglement summary - logic only."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        
        # Create test data
        n_features = 4
        disent_matrix = np.random.rand(n_features, n_features) * 20
        count_matrix = np.random.randint(10, 50, (n_features, n_features))
        feat_names = [f'feat{i}' for i in range(n_features)]
        
        fig = plot_disentanglement_summary(disent_matrix, count_matrix, feat_names)
        assert fig is not None
        
        # Should create multiple subplots
        assert mock_fig.add_subplot.call_count >= 3
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_selectivity_heatmap(self, mock_subplots):
        """Test selectivity heatmap - logic only."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Create minimal mock experiment
        exp = SimpleNamespace()
        exp.n_cells = 10
        exp.dynamic_features = {f'feat_{i}': None for i in range(4)}
        
        # Mock stats
        exp.stats_table = {}
        for feat_name in exp.dynamic_features:
            exp.stats_table[feat_name] = {}
            for neuron_id in range(10):
                exp.stats_table[feat_name][neuron_id] = {
                    'pre_rval': 0.3,
                    'pval': 0.01,
                    'me': 0.3
                }
        
        def get_stats(cell_id, feat_name, mode='calcium'):
            return exp.stats_table[feat_name][cell_id]
        exp.get_neuron_feature_pair_stats = get_stats
        
        significant_neurons = {0: ['feat_0'], 5: ['feat_1', 'feat_2']}
        
        fig, ax, stats = plot_selectivity_heatmap(exp, significant_neurons)
        assert fig is not None
        assert stats['n_selective'] == 2
        assert stats['n_pairs'] == 3
        
        # Should create heatmap
        mock_ax.imshow.assert_called()
    
    def test_edge_cases_with_mocking(self, visual_experiment):
        """Test edge cases without creating real plots."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Test with None values in stats
            visual_experiment.stats_table[('x', 'y')][0]['pval'] = None
            ax = plot_pc_activity(visual_experiment, 0)
            assert ax is not None
            
            # Test with very small data range
            ax = plot_neuron_feature_density(visual_experiment, 'calcium', 0, 'speed', ind1=0, ind2=5)
            assert ax is not None
            
            # Test with zero count matrix
            zero_count = np.zeros((4, 4))
            disent_matrix = np.ones((4, 4))
            feat_names = ['a', 'b', 'c', 'd']
            fig, ax = plot_disentanglement_heatmap(disent_matrix, zero_count, feat_names)
            assert fig is not None
    
    @patch('matplotlib.pyplot.subplots')
    def test_binary_feature_handling(self, mock_subplots, visual_experiment):
        """Test binary feature handling without plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Add binary feature
        visual_experiment.binary_feat = SimpleNamespace(
            data=np.random.randint(0, 2, visual_experiment.n_frames),
            scdata=np.random.rand(visual_experiment.n_frames),
            is_binary=True,
            discrete=True
        )
        
        # Should raise NotImplementedError for spikes with binary features
        with pytest.raises(NotImplementedError):
            plot_neuron_feature_density(visual_experiment, 'spikes', 0, 'binary_feat')
        
        # Should work for calcium
        ax = plot_neuron_feature_density(visual_experiment, 'calcium', 0, 'binary_feat')
        assert ax is not None


class TestVisualPerformance:
    """Test that visual functions are fast when mocked."""
    
    @patch('matplotlib.pyplot.subplots')
    def test_all_functions_fast(self, mock_subplots, visual_experiment):
        """All visual functions should complete in <0.1s when mocked."""
        import time
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Each function should be fast
        functions_to_test = [
            lambda: plot_pc_activity(visual_experiment, 0),
            lambda: plot_neuron_feature_density(visual_experiment, 'calcium', 0, 'speed'),
            lambda: plot_neuron_feature_pair(visual_experiment, 0, 'speed'),
        ]
        
        for func in functions_to_test:
            start = time.time()
            with patch('matplotlib.pyplot.figure', return_value=mock_fig):
                result = func()
            duration = time.time() - start
            assert duration < 0.1, f"Function took {duration}s, should be <0.1s"
            assert result is not None