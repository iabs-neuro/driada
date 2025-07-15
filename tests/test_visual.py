"""Tests for the INTENSE visual module."""

import numpy as np
import pytest
from src.driada.intense.visual import (
    plot_pc_activity,
    plot_neuron_feature_density,
    plot_neuron_feature_pair,
    plot_disentanglement_heatmap,
    plot_disentanglement_summary,
    plot_selectivity_heatmap,
)
import matplotlib.pyplot as plt
from src.driada.experiment.synthetic import generate_synthetic_exp
from types import SimpleNamespace


def create_minimal_experiment_for_visual(T=500, num_neurons=3):
    """Create minimal experiment for visual testing only."""
    # Use synthetic module to create a proper experiment
    exp = generate_synthetic_exp(n_dfeats=3, n_cfeats=3, nneurons=num_neurons, seed=42, fps=20)
    
    # The visual functions expect exp.x, exp.y to be features with .data attribute
    # Create simple spatial trajectory
    T_actual = exp.n_frames
    x_data = np.cumsum(np.random.randn(T_actual) * 0.1)
    y_data = np.cumsum(np.random.randn(T_actual) * 0.1)
    speed_data = np.abs(np.random.randn(T_actual)) * 0.5
    
    # Create mock objects with data and scdata attributes for visual functions
    exp.x = SimpleNamespace(data=x_data, scdata=x_data)
    exp.y = SimpleNamespace(data=y_data, scdata=y_data)
    exp.speed = SimpleNamespace(data=speed_data, scdata=speed_data, is_binary=False, discrete=False)
    
    # Add signature attribute if missing
    if not hasattr(exp, 'signature'):
        exp.signature = 'TestExp'
    
    # Ensure neurons have sp and ca attributes with data and scdata
    for neuron in exp.neurons:
        if not hasattr(neuron, 'sp'):
            sp_data = np.zeros(T_actual)
            spike_times = np.random.choice(T_actual, size=int(T_actual*0.01), replace=False)
            sp_data[spike_times] = 1
            neuron.sp = SimpleNamespace(data=sp_data)
        # Ensure ca has scdata attribute
        if hasattr(neuron, 'ca'):
            if not hasattr(neuron.ca, 'scdata'):
                neuron.ca.scdata = neuron.ca.data
    
    return exp


def test_plot_pc_activity():
    """Test place cell activity plotting."""
    # Create minimal experiment for visual testing
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=5)
    
    # Add stats table for the plot
    exp.stats_table = {('x', 'y'): []}
    for i in range(5):
        exp.stats_table[('x', 'y')].append({
            'pval': 0.01 * (i+1),
            'rel_mi_beh': 0.5 / (i+1)
        })
    
    # Should run without error
    ax = plot_pc_activity(exp, 0)
    assert ax is not None
    plt.close('all')


def test_plot_neuron_feature_density():
    """Test neuron feature density plotting."""
    # Create minimal experiment for visual testing
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Should run without error
    ax = plot_neuron_feature_density(exp, 'calcium', 0, 'speed', ind1=0, ind2=100)
    assert ax is not None
    plt.close('all')


def test_plot_neuron_feature_pair():
    """Test neuron feature pair plotting."""
    # Create minimal experiment for visual testing
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Should run without error
    fig = plot_neuron_feature_pair(exp, 0, 'speed', ind1=0, ind2=100)
    assert fig is not None
    plt.close('all')


def test_plot_neuron_feature_density_discrete():
    """Test neuron feature density plotting with discrete features."""
    # Create minimal experiment for visual testing  
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Add a binary feature
    exp.binary_feat = SimpleNamespace(
        data=np.random.randint(0, 2, exp.n_frames),
        scdata=np.random.rand(exp.n_frames),
        is_binary=True,
        discrete=True
    )
    
    # Should run without error for spikes (currently raises NotImplementedError)
    with pytest.raises(NotImplementedError, match="Binary feature density plot for spike data"):
        plot_neuron_feature_density(exp, 'spikes', 0, 'binary_feat', ind1=0, ind2=100)
    
    plt.close('all')


def test_plot_neuron_feature_pair_no_density():
    """Test neuron feature pair plotting without density subplot."""
    # Create minimal experiment for visual testing  
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Should run without error
    fig = plot_neuron_feature_pair(exp, 0, 'speed', ind1=0, ind2=100, add_density_plot=False)
    assert fig is not None
    plt.close('all')


def test_plot_disentanglement_heatmap():
    """Test disentanglement heatmap plotting."""
    # Create test data
    n_features = 4
    disent_matrix = np.array([
        [0, 10, 20, 5],
        [20, 0, 15, 25],
        [5, 15, 0, 10],
        [25, 5, 20, 0]
    ])
    count_matrix = np.array([
        [0, 20, 30, 10],
        [20, 0, 30, 50],
        [30, 30, 0, 20],
        [10, 50, 20, 0]
    ])
    feat_names = ['feat1', 'feat2', 'feat3', 'feat4']
    
    # Test with default parameters
    fig, ax = plot_disentanglement_heatmap(disent_matrix, count_matrix, feat_names)
    assert fig is not None
    assert ax is not None
    plt.close('all')
    
    # Test with custom parameters
    fig, ax = plot_disentanglement_heatmap(
        disent_matrix, count_matrix, feat_names,
        title='Custom Title',
        figsize=(10, 8),
        dpi=150,
        vmin=0, vmax=100,
        fontsize=12,
        show_grid=False
    )
    assert fig is not None
    assert ax is not None
    plt.close('all')
    
    # Test with zero count matrix (should handle division by zero)
    zero_count = np.zeros_like(count_matrix)
    fig, ax = plot_disentanglement_heatmap(disent_matrix, zero_count, feat_names)
    assert fig is not None
    plt.close('all')
    
    # Test with custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 1), (1, 1, 1), (1, 0, 0)])
    fig, ax = plot_disentanglement_heatmap(
        disent_matrix, count_matrix, feat_names,
        cmap=custom_cmap
    )
    assert fig is not None
    plt.close('all')


def test_plot_disentanglement_summary():
    """Test disentanglement summary plotting."""
    # Create test data for single experiment
    n_features = 4
    disent_matrix = np.array([
        [0, 10, 20, 5],
        [20, 0, 15, 25],
        [5, 15, 0, 10],
        [25, 5, 20, 0]
    ])
    count_matrix = np.array([
        [0, 20, 30, 10],
        [20, 0, 30, 50],
        [30, 30, 0, 20],
        [10, 50, 20, 0]
    ])
    feat_names = ['feat1', 'feat2', 'feat3', 'feat4']
    
    # Test with single experiment
    fig = plot_disentanglement_summary(disent_matrix, count_matrix, feat_names)
    assert fig is not None
    plt.close('all')
    
    # Test with multiple experiments (list input)
    disent_matrices = [disent_matrix, disent_matrix * 0.8]
    count_matrices = [count_matrix, count_matrix * 0.9]
    fig = plot_disentanglement_summary(
        disent_matrices, count_matrices, feat_names,
        title_prefix='Multi-Exp: '
    )
    assert fig is not None
    plt.close('all')
    
    # Test with custom parameters
    fig = plot_disentanglement_summary(
        disent_matrix, count_matrix, feat_names,
        title_prefix='Test: ',
        figsize=(16, 12),
        dpi=200
    )
    assert fig is not None
    plt.close('all')
    
    # Test with experiments parameter (though not used in current implementation)
    fig = plot_disentanglement_summary(
        disent_matrices, count_matrices, feat_names,
        experiments=['Exp1', 'Exp2']
    )
    assert fig is not None
    plt.close('all')
    
    # Test with zero counts (edge case)
    zero_count = np.zeros_like(count_matrix)
    fig = plot_disentanglement_summary(disent_matrix, zero_count, feat_names)
    assert fig is not None
    plt.close('all')


def test_plot_neuron_feature_density_edge_cases():
    """Test edge cases for neuron feature density plotting."""
    # Create minimal experiment
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Test with shift parameter (currently unused but in signature)
    ax = plot_neuron_feature_density(exp, 'calcium', 0, 'speed', shift=10)
    assert ax is not None
    plt.close('all')
    
    # Test with very small data range
    ax = plot_neuron_feature_density(exp, 'calcium', 0, 'speed', ind1=0, ind2=10)
    assert ax is not None
    plt.close('all')
    
    # Test with downsampling
    ax = plot_neuron_feature_density(exp, 'calcium', 0, 'speed', ds=5)
    assert ax is not None
    plt.close('all')
    
    # Test with provided axes
    fig, ax_provided = plt.subplots()
    ax = plot_neuron_feature_density(exp, 'calcium', 0, 'speed', ax=ax_provided)
    assert ax is ax_provided
    plt.close('all')


def test_plot_pc_activity_edge_cases():
    """Test edge cases for place cell activity plotting."""
    # Create experiment
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=5)
    
    # Add stats table
    exp.stats_table = {('x', 'y'): []}
    for i in range(5):
        exp.stats_table[('x', 'y')].append({
            'pval': None if i == 0 else 0.01 * (i+1),  # Test None pval
            'rel_mi_beh': None if i == 1 else 0.5 / (i+1)  # Test None rel_mi_beh
        })
    
    # Test with None values in stats
    ax = plot_pc_activity(exp, 0)  # Has None pval
    assert ax is not None
    plt.close('all')
    
    ax = plot_pc_activity(exp, 1)  # Has None rel_mi_beh
    assert ax is not None
    plt.close('all')
    
    # Test with custom downsampling
    ax = plot_pc_activity(exp, 2, ds=10)
    assert ax is not None
    plt.close('all')
    
    # Test with provided axes and aspect ratio calculation
    fig, ax_provided = plt.subplots()
    ax = plot_pc_activity(exp, 3, ax=ax_provided)
    assert ax is ax_provided
    plt.close('all')


def test_plot_neuron_feature_pair_edge_cases():
    """Test edge cases for neuron feature pair plotting."""
    # Create experiment
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Test with custom title
    fig = plot_neuron_feature_pair(exp, 0, 'speed', title='Custom Title')
    assert fig is not None
    plt.close('all')
    
    # Test with binary feature
    exp.binary_feat = type('obj', (object,), {
        'data': np.random.randint(0, 2, exp.n_frames),
        'scdata': np.random.rand(exp.n_frames),
        'discrete': True,
        'is_binary': True
    })()
    
    fig = plot_neuron_feature_pair(exp, 0, 'binary_feat')
    assert fig is not None
    plt.close('all')
    
    # Test with ind2 > n_frames
    fig = plot_neuron_feature_pair(exp, 0, 'speed', ind1=0, ind2=10000)
    assert fig is not None
    plt.close('all')


def test_plot_neuron_feature_density_continuous():
    """Test neuron feature density plotting with continuous features."""
    # Create minimal experiment
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Test continuous feature density plot
    ax = plot_neuron_feature_density(exp, 'calcium', 0, 'speed')
    assert ax is not None
    plt.close('all')


def test_plot_neuron_feature_pair_binary():
    """Test neuron feature pair plotting with binary features."""
    # Create experiment
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Add a binary feature with is_binary attribute
    exp.binary_feat = SimpleNamespace(
        data=np.random.randint(0, 2, exp.n_frames),
        scdata=np.random.rand(exp.n_frames),
        discrete=True,
        is_binary=True
    )
    
    # Test plotting with binary feature
    fig = plot_neuron_feature_pair(exp, 0, 'binary_feat')
    assert fig is not None
    plt.close('all')


def test_plot_neuron_feature_density_binary_calcium():
    """Test neuron feature density plotting with binary features for calcium data."""
    # Create experiment
    exp = create_minimal_experiment_for_visual(T=500, num_neurons=3)
    
    # Add a binary feature
    exp.binary_feat = SimpleNamespace(
        data=np.random.randint(0, 2, exp.n_frames),
        scdata=np.random.rand(exp.n_frames),
        is_binary=True,
        discrete=True
    )
    
    # Test binary feature density plot for calcium (should work)
    ax = plot_neuron_feature_density(exp, 'calcium', 0, 'binary_feat')
    assert ax is not None
    plt.close('all')


def test_plot_disentanglement_summary_with_nan():
    """Test disentanglement summary with NaN values in dominance calculation."""
    # Create test data with some zero counts that will produce NaN
    disent_matrix = np.array([
        [0, 10, 0, 5],
        [20, 0, 15, 0],
        [0, 15, 0, 10],
        [25, 0, 20, 0]
    ])
    count_matrix = np.array([
        [0, 20, 0, 10],
        [20, 0, 30, 0],
        [0, 30, 0, 20],
        [10, 0, 20, 0]
    ])
    feat_names = ['feat1', 'feat2', 'feat3', 'feat4']
    
    # Should handle NaN values gracefully
    fig = plot_disentanglement_summary(disent_matrix, count_matrix, feat_names)
    assert fig is not None
    plt.close('all')


def test_plot_selectivity_heatmap():
    """Test plot_selectivity_heatmap function."""
    # Create mock experiment
    exp = SimpleNamespace()
    exp.n_cells = 10
    exp.dynamic_features = {
        'd_feat_0': None,
        'd_feat_1': None, 
        'c_feat_0': None,
        'c_feat_1': None,
        (0, 1): None  # Tuple key to test filtering
    }
    
    # Create mock stats_table with MI values
    exp.stats_table = {}
    for feat_name in ['d_feat_0', 'd_feat_1', 'c_feat_0', 'c_feat_1']:
        exp.stats_table[feat_name] = {}
        for neuron_id in range(10):
            # Make some neurons selective
            if (neuron_id < 3 and feat_name == 'd_feat_0') or \
               (neuron_id >= 7 and feat_name == 'c_feat_0') or \
               (neuron_id == 5 and feat_name in ['d_feat_1', 'c_feat_1']):
                mi_value = np.random.uniform(0.2, 0.5)
                pval = 0.001
            else:
                mi_value = np.random.uniform(0.01, 0.05)
                pval = 0.5
            
            exp.stats_table[feat_name][neuron_id] = {
                'pre_rval': mi_value,
                'pval': pval,
                'shift_used': 0.0,
                'me': mi_value  # 'me' stores the metric value
            }
    
    # Mock get_neuron_feature_pair_stats method
    def get_stats(cell_id, feat_name, mode='calcium'):
        return exp.stats_table[feat_name][cell_id]
    exp.get_neuron_feature_pair_stats = get_stats
    
    # Define significant neurons
    significant_neurons = {
        0: ['d_feat_0'],
        1: ['d_feat_0'],
        2: ['d_feat_0'],
        5: ['d_feat_1', 'c_feat_1'],
        7: ['c_feat_0'],
        8: ['c_feat_0'],
        9: ['c_feat_0']
    }
    
    # Test basic functionality
    fig, ax, stats = plot_selectivity_heatmap(exp, significant_neurons)
    assert fig is not None
    assert ax is not None
    assert isinstance(stats, dict)
    assert stats['n_selective'] == len(significant_neurons)
    assert stats['n_pairs'] == sum(len(v) for v in significant_neurons.values())
    assert len(stats['metric_values']) == stats['n_pairs']
    assert all(v > 0 for v in stats['metric_values'])  # All MI values should be positive
    plt.close('all')
    
    # Test with custom parameters
    fig, ax, stats = plot_selectivity_heatmap(
        exp, significant_neurons,
        metric='corr',
        cmap='plasma',
        use_log_scale=True,
        figsize=(12, 8)
    )
    assert fig is not None
    plt.close('all')
    
    # Test with significance threshold
    fig, ax, stats = plot_selectivity_heatmap(
        exp, significant_neurons,
        significance_threshold=0.01
    )
    assert fig is not None
    plt.close('all')
    
    # Test with provided axes
    fig_custom, ax_custom = plt.subplots(figsize=(10, 6))
    fig2, ax2, stats2 = plot_selectivity_heatmap(
        exp, significant_neurons,
        ax=ax_custom
    )
    assert ax2 is ax_custom
    assert fig2 is fig_custom
    plt.close('all')
    
    # Test with empty significant neurons
    fig, ax, stats = plot_selectivity_heatmap(exp, {})
    assert stats['n_selective'] == 0
    assert stats['n_pairs'] == 0
    assert len(stats['metric_values']) == 0
    plt.close('all')
    
    # Test with custom vmin/vmax
    fig, ax, stats = plot_selectivity_heatmap(
        exp, significant_neurons,
        vmin=0.1,
        vmax=0.6
    )
    assert fig is not None
    plt.close('all')
    
    # Test that tuple keys in dynamic_features are filtered out
    # The heatmap should only show 4 features, not 5
    fig, ax, stats = plot_selectivity_heatmap(exp, significant_neurons)
    assert len(ax.get_xticklabels()) == 4  # Only string keys
    plt.close('all')


def test_plot_selectivity_heatmap_edge_cases():
    """Test edge cases for plot_selectivity_heatmap."""
    # Create minimal mock experiment
    exp = SimpleNamespace()
    exp.n_cells = 3
    exp.dynamic_features = {'d_feat_0': None, 'c_feat_0': None}
    
    # Create stats with some None values
    exp.stats_table = {}
    for feat_name in ['d_feat_0', 'c_feat_0']:
        exp.stats_table[feat_name] = {}
        for neuron_id in range(3):
            exp.stats_table[feat_name][neuron_id] = {
                'pre_rval': 0.3 if neuron_id == 0 else 0.1,
                'pval': None if neuron_id == 1 else 0.001,  # Test None pval
                'shift_used': 0.0,
                'me': 0.3 if neuron_id == 0 else 0.1  # Add 'me' key
            }
    
    # Mock get_neuron_feature_pair_stats method
    def get_stats(cell_id, feat_name, mode='calcium'):
        return exp.stats_table[feat_name][cell_id]
    exp.get_neuron_feature_pair_stats = get_stats
    
    # Significant neurons including one with None pval
    significant_neurons = {0: ['d_feat_0'], 1: ['c_feat_0']}
    
    # Should handle None pval gracefully when no threshold
    fig, ax, stats = plot_selectivity_heatmap(exp, significant_neurons)
    assert fig is not None
    assert len(stats['metric_values']) == 2  # Both neurons included
    plt.close('all')
    
    # With threshold, should skip neuron 1 with None pval
    fig, ax, stats = plot_selectivity_heatmap(
        exp, significant_neurons,
        significance_threshold=0.05
    )
    assert fig is not None
    assert len(stats['metric_values']) == 1  # Only neuron 0
    plt.close('all')
    
    # Test with missing corr in stats when metric='corr'
    exp.stats_table['d_feat_0'][0].pop('corr', None)  # Remove corr if exists
    fig, ax, stats = plot_selectivity_heatmap(
        exp, {0: ['d_feat_0']},
        metric='corr'
    )
    assert fig is not None
    # Should fall back to pre_rval when corr not available
    assert len(stats['metric_values']) == 1
    plt.close('all')