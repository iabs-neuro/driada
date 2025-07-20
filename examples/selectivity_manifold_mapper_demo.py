"""
Comprehensive SelectivityManifoldMapper Demonstration
=====================================================

This example demonstrates the complete workflow of analyzing how individual neuron
selectivity relates to population-level manifold structure using DRIADA's integrated
INTENSE and dimensionality reduction capabilities.

Key concepts demonstrated:
1. Synthetic data generation with mixed selectivity neurons
2. INTENSE analysis to identify feature-selective neurons
3. Creating population embeddings with multiple DR methods
4. Analyzing neuron selectivity to embedding components
5. Visualizing functional organization in manifolds
6. Demonstrating how behavioral features map to embedding dimensions

Usage:
    python selectivity_manifold_mapper_demo.py [--quick] [--save-plots] [--methods METHOD1,METHOD2,...]
    
    Options:
    --quick         Run with smaller dataset for quick testing (200 neurons, 300s)
    --save-plots    Save generated plots to files
    --methods       Comma-separated list of DR methods to use (default: pca,umap,isomap)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import argparse
from typing import Dict, List, Tuple, Optional
import time

# Import DRIADA modules
from driada import (
    generate_mixed_population_exp,
    compute_cell_feat_significance
)
from driada.dim_reduction import SelectivityManifoldMapper, METHODS_DICT
from driada.intense import compute_embedding_selectivity


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate SelectivityManifoldMapper functionality"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run with smaller dataset for quick testing"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save generated plots to files"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pca,umap,isomap",
        help="Comma-separated list of DR methods to use"
    )
    return parser.parse_args()


def generate_rich_synthetic_data(quick_mode: bool = False):
    """
    Generate synthetic neural data with rich structure for demonstration.
    
    Parameters
    ----------
    quick_mode : bool
        If True, use smaller dataset for faster execution
        
    Returns
    -------
    exp : Experiment
        Generated experiment with mixed selectivity neurons
    info : dict
        Information about the generated data
    """
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC DATA")
    print("="*70)
    
    # Parameters for data generation
    if quick_mode:
        n_neurons = 200
        duration = 300  # 5 minutes
        print("Quick mode: 200 neurons, 5 minutes")
    else:
        n_neurons = 500
        duration = 600  # 10 minutes
        print("Full mode: 500 neurons, 10 minutes")
    
    # Generate mixed population with 2D spatial manifold
    exp, info = generate_mixed_population_exp(
        n_neurons=n_neurons,
        manifold_type='2d_spatial',
        manifold_fraction=0.6,  # 60% place cells
        n_discrete_features=1,   # Reward feature
        n_continuous_features=2, # Speed and head direction
        duration=duration,
        fps=20.0,
        correlation_mode='spatial_correlated',
        seed=42,
        manifold_params={
            'grid_arrangement': True,
            'field_sigma': 0.15,
            'noise_std': 0.05,
            'baseline_rate': 0.1,
            'peak_rate': 1.0,  # Realistic firing rate
            'decay_time': 2.0,
            'calcium_noise_std': 0.05
        },
        feature_params={
            'selectivity_prob': 0.9,
            'multi_select_prob': 0.6,  # High mixed selectivity
            'rate_0': 0.5,
            'rate_1': 2.0,
            'noise_std': 0.05,
            'hurst': 0.3,
            'skip_prob': 0.0,
            'ampl_range': (1.5, 3.5),
            'decay_time': 2.0
        },
        return_info=True
    )
    
    # Rename features for clarity
    feature_mapping = {
        'd_feat_0': 'reward',
        'c_feat_0': 'speed',
        'c_feat_1': 'head_direction'
    }
    
    print(f"\nGenerated {exp.n_cells} neurons:")
    print(f"  - Pure manifold cells: ~{int(exp.n_cells * 0.6)}")
    print(f"  - Feature-selective cells: ~{int(exp.n_cells * 0.4)}")
    print(f"  - Expected mixed selectivity: ~{int(exp.n_cells * 0.4 * 0.6)}")
    print(f"  - Recording duration: {duration}s at 20 Hz")
    
    return exp, info


def run_intense_analysis(exp, quick_mode: bool = False):
    """
    Run INTENSE analysis to identify feature-selective neurons.
    
    Parameters
    ----------
    exp : Experiment
        Experiment object with neural data
    quick_mode : bool
        If True, use fewer shuffles for faster execution
        
    Returns
    -------
    results : dict
        INTENSE analysis results
    """
    print("\n" + "="*70)
    print("RUNNING INTENSE ANALYSIS")
    print("="*70)
    
    # Parameters for INTENSE
    if quick_mode:
        n_shuffles_stage1 = 50
        n_shuffles_stage2 = 500
        print("Quick mode: 50/500 shuffles")
    else:
        n_shuffles_stage1 = 100
        n_shuffles_stage2 = 2000
        print("Full mode: 100/2000 shuffles")
    
    # Analyze all behavioral features
    features_to_analyze = ['position_2d', 'c_feat_0', 'c_feat_1', 'd_feat_0']
    available_features = [f for f in features_to_analyze if f in exp.dynamic_features]
    
    print(f"Analyzing {exp.n_cells} neurons × {len(available_features)} features")
    
    # Skip delays for MultiTimeSeries features
    skip_delays = {'position_2d': True}
    
    # Run INTENSE analysis
    start_time = time.time()
    stats, significance, info, intense_results = compute_cell_feat_significance(
        exp,
        feat_bunch=available_features,
        mode='two_stage',
        n_shuffles_stage1=n_shuffles_stage1,
        n_shuffles_stage2=n_shuffles_stage2,
        metric_distr_type='norm',
        pval_thr=0.01,
        multicomp_correction=None,
        verbose=True,
        find_optimal_delays=True,
        skip_delays=skip_delays,
        shift_window=2,
        ds=5,  # Downsample for efficiency
        allow_mixed_dimensions=True  # For position_2d MultiTimeSeries
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nINTENSE analysis completed in {elapsed_time:.1f} seconds")
    
    # Get significant neurons
    significant_neurons = exp.get_significant_neurons()
    mixed_selectivity_neurons = exp.get_significant_neurons(min_nspec=2)
    
    print(f"\nSelectivity Summary:")
    print(f"  - Total selective neurons: {len(significant_neurons)}/{exp.n_cells}")
    print(f"  - Mixed selectivity neurons: {len(mixed_selectivity_neurons)}")
    
    # Count selectivity by feature
    feature_counts = {}
    for neuron_id, features in significant_neurons.items():
        for feat in features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    
    print(f"\nSelectivity by feature:")
    for feat, count in sorted(feature_counts.items()):
        print(f"  - {feat}: {count} neurons")
    
    return {
        'stats': stats,
        'significance': significance,
        'significant_neurons': significant_neurons,
        'mixed_selectivity_neurons': mixed_selectivity_neurons,
        'feature_counts': feature_counts
    }


def create_embeddings_and_analyze(exp, methods: List[str], quick_mode: bool = False):
    """
    Create embeddings using SelectivityManifoldMapper and analyze component selectivity.
    
    Parameters
    ----------
    exp : Experiment
        Experiment with INTENSE results
    methods : list of str
        DR methods to use
    quick_mode : bool
        If True, use fewer components and shuffles
        
    Returns
    -------
    mapper : SelectivityManifoldMapper
        Initialized mapper object
    embedding_results : dict
        Results from compute_embedding_selectivity
    """
    print("\n" + "="*70)
    print("CREATING EMBEDDINGS AND ANALYZING COMPONENT SELECTIVITY")
    print("="*70)
    
    # Initialize SelectivityManifoldMapper
    mapper = SelectivityManifoldMapper(exp)
    
    # Parameters for embeddings
    if quick_mode:
        n_components = 5
        n_shuffles = 500
        print("Quick mode: 5 components, 500 shuffles")
    else:
        n_components = 10
        n_shuffles = 1000
        print("Full mode: 10 components, 1000 shuffles")
    
    # Create embeddings for each method
    for method in methods:
        print(f"\nCreating {method.upper()} embedding...")
        
        # Method-specific parameters
        if method == 'pca':
            dr_kwargs = {}
        elif method == 'umap':
            dr_kwargs = {'n_neighbors': 30}  # min_dist is handled internally
        elif method == 'isomap':
            dr_kwargs = {'n_neighbors': 30}
        else:
            dr_kwargs = {}
        
        # Create embedding using all neurons
        try:
            embedding = mapper.create_embedding(
                method=method,
                n_components=n_components,
                data_type='calcium',
                neuron_selection='all',
                **dr_kwargs
            )
            print(f"  Created {method} embedding: shape {embedding.shape}")
        except Exception as e:
            print(f"  Failed to create {method} embedding: {e}")
            continue
    
    # Check which methods were successfully created
    successful_methods = []
    for method in methods:
        try:
            exp.get_embedding(method, 'calcium')
            successful_methods.append(method)
        except:
            pass
    
    if not successful_methods:
        print("No embeddings were successfully created!")
        return mapper, {}
    
    # Analyze component selectivity for all embeddings
    print("\n" + "-"*50)
    print("Analyzing neuron selectivity to embedding components...")
    print("-"*50)
    
    embedding_results = compute_embedding_selectivity(
        exp,
        embedding_methods=successful_methods,
        mode='two_stage',
        n_shuffles_stage1=50,
        n_shuffles_stage2=n_shuffles,
        metric_distr_type='norm',
        pval_thr=0.05,  # More lenient for components
        multicomp_correction=None,  # No correction for exploratory analysis
        find_optimal_delays=False,  # Components are instantaneous
        verbose=True,
        ds=5
    )
    
    # Summarize results
    print("\n" + "="*50)
    print("COMPONENT SELECTIVITY SUMMARY")
    print("="*50)
    
    for method, results in embedding_results.items():
        n_sig_neurons = len(results['significant_neurons'])
        n_components = results['n_components']
        
        print(f"\n{method.upper()}:")
        print(f"  - Neurons selective to components: {n_sig_neurons}")
        print(f"  - Components with selective neurons:")
        
        for comp_idx, neuron_list in results['component_selectivity'].items():
            if neuron_list:
                print(f"    Component {comp_idx}: {len(neuron_list)} neurons")
    
    return mapper, embedding_results


def visualize_results(exp, mapper, embedding_results, methods: List[str], save_plots: bool = False):
    """
    Create comprehensive visualizations of the results.
    
    Parameters
    ----------
    exp : Experiment
        Experiment object
    mapper : SelectivityManifoldMapper
        Mapper with stored embeddings
    embedding_results : dict
        Component selectivity results
    methods : list of str
        DR methods to visualize
    save_plots : bool
        Whether to save plots to files
    """
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Embedding comparison figure
    create_embedding_comparison_figure(exp, methods, save_plots)
    
    # 2. Component selectivity heatmap
    create_component_selectivity_heatmap(exp, embedding_results, save_plots)
    
    # 3. Functional organization analysis
    create_functional_organization_figure(exp, mapper, embedding_results, save_plots)
    
    # 4. Component interpretation figure
    create_component_interpretation_figure(exp, embedding_results, save_plots)
    
    plt.show()


def create_embedding_comparison_figure(exp, methods: List[str], save_plots: bool = False):
    """Create figure comparing embeddings colored by behavioral features."""
    n_methods = len(methods)
    fig = plt.figure(figsize=(5*n_methods, 10))
    gs = gridspec.GridSpec(2, n_methods, hspace=0.3, wspace=0.3)
    
    # Get behavioral data - no downsampling needed, embeddings are full length
    if 'position_2d' in exp.dynamic_features:
        positions = exp.dynamic_features['position_2d'].data.T
        pos_angle = np.arctan2(positions[:, 1] - 0.5, positions[:, 0] - 0.5)
    else:
        pos_angle = None
    
    if 'c_feat_0' in exp.dynamic_features:
        speed = exp.dynamic_features['c_feat_0'].data
    else:
        speed = None
    
    for i, method in enumerate(methods):
        # Get embedding
        embedding_dict = exp.get_embedding(method, 'calcium')
        if embedding_dict is None:
            continue
        
        embedding = embedding_dict['data']
        
        # Plot colored by position angle
        ax1 = fig.add_subplot(gs[0, i])
        if pos_angle is not None:
            # Normalize angle to [0, 1] for color mapping
            angle_norm = (pos_angle + np.pi) / (2 * np.pi)
            scatter = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                                c=angle_norm, cmap='hsv', alpha=0.6, s=1, vmin=0, vmax=1)
            cbar = plt.colorbar(scatter, ax=ax1, label='Position angle')
            # Set colorbar ticks to show actual angles
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        else:
            ax1.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=1)
        
        ax1.set_xlabel('Component 0')
        ax1.set_ylabel('Component 1')
        ax1.set_title(f'{method.upper()} - Spatial structure')
        
        # Plot colored by speed
        ax2 = fig.add_subplot(gs[1, i])
        if speed is not None:
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                                c=speed, cmap='viridis', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax2, label='Speed')
        else:
            ax2.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=1)
        
        ax2.set_xlabel('Component 0')
        ax2.set_ylabel('Component 1')
        ax2.set_title(f'{method.upper()} - Speed modulation')
    
    plt.suptitle('Population Embeddings Colored by Behavioral Features', fontsize=16)
    
    if save_plots:
        plt.savefig('selectivity_mapper_embeddings.png', dpi=150, bbox_inches='tight')
        print("Saved: selectivity_mapper_embeddings.png")


def create_component_selectivity_heatmap(exp, embedding_results: Dict, save_plots: bool = False):
    """Create heatmap showing neuron selectivity to embedding components."""
    # Prepare data for heatmap
    methods = list(embedding_results.keys())
    max_components = max(res['n_components'] for res in embedding_results.values())
    
    # Create figure
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 8))
    if len(methods) == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        results = embedding_results[method]
        n_components = results['n_components']
        
        # Create selectivity matrix
        n_neurons = exp.n_cells
        selectivity_matrix = np.zeros((n_neurons, n_components))
        
        # Fill in MI values for significant pairs
        for feat_name, feat_stats in results['stats'].items():
            # Check if this is a component feature
            if isinstance(feat_name, str) and feat_name.startswith(f"{method}_comp"):
                comp_idx = int(feat_name.split('_comp')[-1])
                
                for neuron_id, stats in feat_stats.items():
                    if stats.get('me') is not None:
                        # Check if significant
                        if feat_name in results['significance'] and \
                           neuron_id in results['significance'][feat_name] and \
                           results['significance'][feat_name][neuron_id].get('stage2', False):
                            selectivity_matrix[neuron_id, comp_idx] = stats['me']
        
        # Plot heatmap
        im = ax.imshow(selectivity_matrix.T, aspect='auto', cmap='hot', 
                      interpolation='nearest')
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Component')
        ax.set_title(f'{method.upper()} Component Selectivity')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mutual Information (bits)')
        
        # Add grid
        ax.set_yticks(range(n_components))
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Neuron Selectivity to Embedding Components', fontsize=16)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('selectivity_mapper_component_heatmap.png', dpi=150, bbox_inches='tight')
        print("Saved: selectivity_mapper_component_heatmap.png")


def create_functional_organization_figure(exp, mapper, embedding_results: Dict, save_plots: bool = False):
    """Analyze and visualize functional organization in the manifold."""
    fig = plt.figure(figsize=(15, 5))
    
    # Get original feature selectivity
    significant_neurons = exp.get_significant_neurons()
    
    for i, method in enumerate(embedding_results.keys()):
        ax = fig.add_subplot(1, len(embedding_results), i+1)
        
        results = embedding_results[method]
        
        # Analyze overlap between feature-selective and component-selective neurons
        feature_selective = set(significant_neurons.keys())
        component_selective = set(results['significant_neurons'].keys())
        
        # Create Venn diagram data
        only_features = len(feature_selective - component_selective)
        both = len(feature_selective & component_selective)
        only_components = len(component_selective - feature_selective)
        
        # Simple bar plot instead of Venn diagram
        categories = ['Features\nonly', 'Both', 'Components\nonly']
        values = [only_features, both, only_components]
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('Number of neurons')
        ax.set_title(f'{method.upper()} - Functional Organization')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}', ha='center', va='bottom')
        
        # Add summary statistics
        total_selective = only_features + both + only_components
        if total_selective > 0:
            overlap_pct = (both / total_selective) * 100
            ax.text(0.5, 0.95, f'Overlap: {overlap_pct:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Functional Organization: Feature vs Component Selectivity', fontsize=16)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('selectivity_mapper_functional_org.png', dpi=150, bbox_inches='tight')
        print("Saved: selectivity_mapper_functional_org.png")


def create_component_interpretation_figure(exp, embedding_results: Dict, save_plots: bool = False):
    """Visualize how components relate to behavioral features."""
    # Focus on PCA for interpretability
    if 'pca' not in embedding_results:
        print("PCA not found in results, skipping component interpretation")
        return
    
    pca_results = embedding_results['pca']
    
    # Create correlation matrix between components and features
    embedding_dict = exp.get_embedding('pca', 'calcium')
    pca_embedding = embedding_dict['data']
    
    # Get behavioral features
    features_data = {}
    feature_names = []
    
    if 'position_2d' in exp.dynamic_features:
        pos_data = exp.dynamic_features['position_2d'].data
        features_data['X position'] = pos_data[0]
        features_data['Y position'] = pos_data[1]
        feature_names.extend(['X position', 'Y position'])
    
    if 'c_feat_0' in exp.dynamic_features:
        features_data['Speed'] = exp.dynamic_features['c_feat_0'].data
        feature_names.append('Speed')
    
    if 'c_feat_1' in exp.dynamic_features:
        features_data['Head direction'] = exp.dynamic_features['c_feat_1'].data
        feature_names.append('Head direction')
    
    if 'd_feat_0' in exp.dynamic_features:
        features_data['Reward'] = exp.dynamic_features['d_feat_0'].data.astype(float)
        feature_names.append('Reward')
    
    # Compute correlations
    n_components = min(5, pca_embedding.shape[1])  # Show top 5 components
    correlation_matrix = np.zeros((len(feature_names), n_components))
    
    for i, feat_name in enumerate(feature_names):
        for j in range(n_components):
            correlation_matrix[i, j] = np.corrcoef(features_data[feat_name], 
                                                  pca_embedding[:, j])[0, 1]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correlation heatmap
    im = ax1.imshow(correlation_matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-1, vmax=1)
    ax1.set_xticks(range(n_components))
    ax1.set_xticklabels([f'PC{i}' for i in range(n_components)])
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('Principal Components')
    ax1.set_title('Component-Feature Correlations')
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(n_components):
            text = ax1.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
    
    plt.colorbar(im, ax=ax1, label='Correlation')
    
    # Explained variance for PCA
    if 'metadata' in embedding_dict and 'explained_variance_ratio' in embedding_dict['metadata']:
        var_exp = embedding_dict['metadata']['explained_variance_ratio'][:n_components]
        ax2.bar(range(n_components), var_exp * 100)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance (%)')
        ax2.set_title('PCA Explained Variance')
        ax2.set_xticks(range(n_components))
        ax2.set_xticklabels([f'PC{i}' for i in range(n_components)])
        
        # Add cumulative variance
        cum_var = np.cumsum(var_exp) * 100
        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(n_components), cum_var, 'ro-', label='Cumulative')
        ax2_twin.set_ylabel('Cumulative Variance (%)')
        ax2_twin.set_ylim(0, 100)
        ax2_twin.legend()
    
    plt.suptitle('Component Interpretation: Relating Embeddings to Behavior', fontsize=16)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('selectivity_mapper_component_interpretation.png', dpi=150, bbox_inches='tight')
        print("Saved: selectivity_mapper_component_interpretation.png")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    methods = [m.strip() for m in args.methods.split(',')]
    
    # Validate methods
    valid_methods = []
    for method in methods:
        if method in METHODS_DICT:
            valid_methods.append(method)
        else:
            print(f"Warning: Unknown method '{method}', skipping")
    
    if not valid_methods:
        print("Error: No valid DR methods specified")
        return
    
    print("\n" + "="*70)
    print("SELECTIVITY MANIFOLD MAPPER DEMONSTRATION")
    print("="*70)
    print(f"DR methods: {', '.join(valid_methods)}")
    print(f"Quick mode: {args.quick}")
    print(f"Save plots: {args.save_plots}")
    
    # Step 1: Generate synthetic data
    exp, info = generate_rich_synthetic_data(args.quick)
    
    # Step 2: Run INTENSE analysis
    intense_results = run_intense_analysis(exp, args.quick)
    
    # Step 3: Create embeddings and analyze component selectivity
    mapper, embedding_results = create_embeddings_and_analyze(exp, valid_methods, args.quick)
    
    # Step 4: Create visualizations
    visualize_results(exp, mapper, embedding_results, valid_methods, args.save_plots)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    # Summary insights
    print("\nKey Insights:")
    print("1. SelectivityManifoldMapper bridges single-neuron and population analyses")
    print("2. Neurons selective to behavioral features often contribute to specific manifold dimensions")
    print("3. Different DR methods capture different aspects of population structure")
    print("4. Component selectivity reveals functional organization in neural manifolds")
    print("\nThis demonstrates DRIADA's unique capability to connect scales of neural analysis!")


if __name__ == "__main__":
    main()