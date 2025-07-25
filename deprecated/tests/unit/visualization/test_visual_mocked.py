"""Fast visual tests with comprehensive mocking - no actual plotting or computation."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
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


@pytest.fixture
def mock_experiment():
    """Create a minimal mock experiment with all required attributes."""
    exp = SimpleNamespace()
    exp.n_frames = 100
    exp.n_cells = 5
    exp.signature = 'MockExp'
    
    # Create position data
    exp.x = SimpleNamespace(data=np.random.randn(100), scdata=np.random.randn(100))
    exp.y = SimpleNamespace(data=np.random.randn(100), scdata=np.random.randn(100))
    exp.speed = SimpleNamespace(
        data=np.abs(np.random.randn(100)), 
        scdata=np.abs(np.random.randn(100)),
        is_binary=False,
        discrete=False
    )
    
    # Create neurons with proper structure
    exp.neurons = []
    for i in range(5):
        neuron = SimpleNamespace()
        # Ensure positive values for log operations
        neuron.ca = SimpleNamespace(
            data=np.abs(np.random.randn(100)) + 0.1,
            scdata=np.abs(np.random.randn(100)) + 0.1
        )
        neuron.sp = SimpleNamespace(
            data=np.random.randint(0, 2, 100),
            scdata=np.random.randint(0, 2, 100).astype(float)
        )
        exp.neurons.append(neuron)
    
    # Add stats table
    exp.stats_table = {('x', 'y'): []}
    for i in range(5):
        exp.stats_table[('x', 'y')].append({
            'pval': 0.01,
            'rel_mi_beh': 0.5
        })
    
    # Add dynamic features for selectivity heatmap
    exp.dynamic_features = {f'feat_{i}': None for i in range(4)}
    
    return exp


class TestVisualFunctionsWithMocking:
    """Test visual functions with comprehensive mocking."""
    
    @patch('matplotlib.pyplot')
    def test_plot_pc_activity(self, mock_plt, mock_experiment):
        """Test place cell activity plotting."""
        # Setup mocks
        fig = MagicMock()
        ax = MagicMock()
        mock_plt.subplots.return_value = (fig, ax)
        mock_plt.gca.return_value = ax
        
        # Mock axis methods
        ax.plot.return_value = [MagicMock()]
        ax.scatter.return_value = MagicMock()
        ax.set_xlim.return_value = None
        ax.set_ylim.return_value = None
        ax.get_xlim.return_value = (0, 10)
        ax.get_ylim.return_value = (0, 10)
        
        # Run function
        result = plot_pc_activity(mock_experiment, 0)
        
        # Verify
        assert result is ax
        assert ax.plot.called
        assert ax.set_xlabel.called
        assert ax.set_ylabel.called
    
    @patch('scipy.stats.gaussian_kde')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_neuron_feature_density(self, mock_subplots, mock_kde, mock_experiment):
        """Test neuron feature density plotting."""
        # Setup mocks
        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)
        
        # Mock KDE
        kde_instance = MagicMock()
        kde_instance.return_value = np.ones((10, 10))
        mock_kde.return_value = kde_instance
        
        # Run function
        result = plot_neuron_feature_density(
            mock_experiment, 'calcium', 0, 'speed', ind1=0, ind2=50
        )
        
        # Verify
        assert result is ax
        assert mock_kde.called
        ax.contourf.assert_called()
    
    @patch('matplotlib.pyplot')
    def test_plot_neuron_feature_pair(self, mock_plt, mock_experiment):
        """Test neuron feature pair plotting."""
        # Setup figure with subplots
        fig = MagicMock()
        ax1 = MagicMock()
        ax2 = MagicMock()
        
        # Mock figure creation and subplot addition
        mock_plt.figure.return_value = fig
        fig.add_subplot.side_effect = [ax1, ax2]
        
        # Mock axis properties
        for ax in [ax1, ax2]:
            ax.plot.return_value = [MagicMock()]
            ax.set_xlim.return_value = None
            ax.set_ylim.return_value = None
            ax.set_xlabel.return_value = None
            ax.set_ylabel.return_value = None
            ax.legend.return_value = MagicMock()
        
        # Run function
        result = plot_neuron_feature_pair(mock_experiment, 0, 'speed')
        
        # Verify
        assert result is fig
        assert fig.add_subplot.call_count >= 2
    
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_disentanglement_heatmap(self, mock_subplots, mock_heatmap):
        """Test disentanglement heatmap plotting."""
        # Setup mocks
        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)
        
        # Create test data
        n_features = 4
        disent_matrix = np.random.rand(n_features, n_features) * 20
        count_matrix = np.random.randint(10, 50, (n_features, n_features))
        feat_names = [f'feat{i}' for i in range(n_features)]
        
        # Run function
        result_fig, result_ax = plot_disentanglement_heatmap(
            disent_matrix, count_matrix, feat_names
        )
        
        # Verify
        assert result_fig is fig
        assert result_ax is ax
        assert mock_heatmap.called
    
    @patch('matplotlib.pyplot')
    @patch('seaborn.heatmap')
    def test_plot_disentanglement_summary(self, mock_heatmap, mock_plt):
        """Test disentanglement summary plotting."""
        # Setup mocks
        fig = MagicMock()
        mock_plt.figure.return_value = fig
        
        # Mock subplots
        ax = MagicMock()
        fig.add_subplot.return_value = ax
        
        # Mock axis methods
        ax.set_xlim.return_value = None
        ax.set_ylim.return_value = None
        ax.set_xlabel.return_value = None
        ax.set_ylabel.return_value = None
        ax.set_title.return_value = None
        ax.bar.return_value = MagicMock()
        ax.errorbar.return_value = MagicMock()
        ax.imshow.return_value = MagicMock()
        ax.set_xticks.return_value = None
        ax.set_xticklabels.return_value = None
        
        # Create test data
        n_features = 4
        disent_matrix = np.random.rand(n_features, n_features) * 20
        count_matrix = np.random.randint(10, 50, (n_features, n_features))
        feat_names = [f'feat{i}' for i in range(n_features)]
        
        # Run function
        result = plot_disentanglement_summary(disent_matrix, count_matrix, feat_names)
        
        # Verify
        assert result is fig
        assert fig.add_subplot.call_count >= 3  # Should create multiple subplots
        assert mock_heatmap.called
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_selectivity_heatmap(self, mock_subplots, mock_experiment):
        """Test selectivity heatmap plotting."""
        # Setup mocks
        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)
        
        # Mock imshow
        im = MagicMock()
        ax.imshow.return_value = im
        
        # Create stats for mock experiment
        def get_stats(cell_id, feat_name, mode='calcium'):
            return {
                'me': 0.3,
                'pval': 0.01,
                'pre_rval': 0.3
            }
        mock_experiment.get_neuron_feature_pair_stats = get_stats
        
        # Define significant neurons
        significant_neurons = {0: ['feat_0'], 2: ['feat_1', 'feat_2']}
        
        # Run function
        result_fig, result_ax, stats = plot_selectivity_heatmap(
            mock_experiment, significant_neurons
        )
        
        # Verify
        assert result_fig is fig
        assert result_ax is ax
        assert ax.imshow.called
        assert stats['n_selective'] == 2
        assert stats['n_pairs'] == 3
        assert len(stats['metric_values']) == 3
    
    @patch('matplotlib.pyplot.subplots')
    def test_edge_cases(self, mock_subplots, mock_experiment):
        """Test edge cases without errors."""
        # Setup mocks
        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)
        
        # Test with None pval
        mock_experiment.stats_table[('x', 'y')][0]['pval'] = None
        result = plot_pc_activity(mock_experiment, 0)
        assert result is ax
        
        # Test with empty significant neurons
        result_fig, result_ax, stats = plot_selectivity_heatmap(mock_experiment, {})
        assert stats['n_selective'] == 0
        assert stats['n_pairs'] == 0
    
    @patch('scipy.stats.gaussian_kde')
    @patch('matplotlib.pyplot')
    @patch('seaborn.violinplot')
    def test_binary_features(self, mock_violinplot, mock_plt, mock_kde, mock_experiment):
        """Test binary feature handling."""
        # Setup mocks
        fig = MagicMock()
        ax = MagicMock()
        mock_plt.subplots.return_value = (fig, ax)
        
        # Mock violinplot for binary features
        mock_violinplot.return_value = ax
        
        # Add binary feature
        mock_experiment.binary_feat = SimpleNamespace(
            data=np.random.randint(0, 2, 100),
            scdata=np.random.randint(0, 2, 100).astype(float),
            is_binary=True,
            discrete=True
        )
        
        # Test that spikes with binary features raises NotImplementedError
        with pytest.raises(NotImplementedError):
            plot_neuron_feature_density(mock_experiment, 'spikes', 0, 'binary_feat')
        
        # Test that calcium with binary features works
        result = plot_neuron_feature_density(mock_experiment, 'calcium', 0, 'binary_feat')
        assert result is ax
        # Should use violinplot for binary features
        assert mock_violinplot.called


class TestPerformanceWithMocking:
    """Verify that mocked tests are fast."""
    
    def test_mocking_performance(self, mock_experiment):
        """All mocked operations should be very fast."""
        import time
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('scipy.stats.gaussian_kde') as mock_kde, \
             patch('seaborn.heatmap') as mock_heatmap:
            
            # Setup basic mocks
            fig = MagicMock()
            ax = MagicMock()
            mock_subplots.return_value = (fig, ax)
            mock_figure.return_value = fig
            fig.add_subplot.return_value = ax
            
            kde_instance = MagicMock()
            kde_instance.return_value = np.ones((10, 10))
            mock_kde.return_value = kde_instance
            
            # Time each function
            functions = [
                lambda: plot_pc_activity(mock_experiment, 0),
                lambda: plot_neuron_feature_density(mock_experiment, 'calcium', 0, 'speed'),
                lambda: plot_neuron_feature_pair(mock_experiment, 0, 'speed'),
            ]
            
            for func in functions:
                start = time.time()
                result = func()
                duration = time.time() - start
                assert duration < 0.1, f"Function took {duration:.3f}s, should be <0.1s"
                assert result is not None