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
        "--no-viz",
        action="store_true",
        help="Skip visualization generation for faster execution"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pca,umap,le",  # Changed from isomap to le
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
    
    # Generate mixed population with circular manifold (head direction cells)
    exp, info = generate_mixed_population_exp(
        n_neurons=n_neurons,
        manifold_type='circular',
        manifold_fraction=0.6,  # 60% head direction cells
        n_discrete_features=1,   # Task/trial type feature
        n_continuous_features=2, # Speed and reward magnitude
        duration=duration,
        fps=20.0,
        correlation_mode='independent',  # Features NOT correlated with head direction
        seed=42,
        manifold_params={
            'kappa': 2.0,  # Von Mises concentration parameter
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
        'd_feat_0': 'task_type',
        'c_feat_0': 'speed',
        'c_feat_1': 'reward_magnitude'
    }
    
    print(f"\nGenerated {exp.n_cells} neurons:")
    print(f"  - Pure head direction cells: ~{int(exp.n_cells * 0.6)}")
    print(f"  - Feature-selective cells: ~{int(exp.n_cells * 0.4)}")
    print(f"  - Expected mixed selectivity: ~{int(exp.n_cells * 0.4 * 0.6)}")
    print(f"  - Recording duration: {duration}s at 20 Hz")
    print(f"  - Manifold type: Circular (head direction)")
    print(f"  - Features: Independent of head direction")
    
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
    features_to_analyze = ['circular_position', 'c_feat_0', 'c_feat_1', 'd_feat_0']
    available_features = [f for f in features_to_analyze if f in exp.dynamic_features]
    
    print(f"Analyzing {exp.n_cells} neurons × {len(available_features)} features")
    
    # Skip delays for MultiTimeSeries features
    skip_delays = {'circular_position': True} if 'circular_position' in available_features else {}
    
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
        n_components = 5
        n_shuffles = 1000
        print("Full mode: 5 components, 1000 shuffles")
    
    # Create embeddings for each method
    for method in methods:
        print(f"\nCreating {method.upper()} embedding...")
        
        # Method-specific parameters
        if method == 'pca':
            dr_kwargs = {}
        elif method == 'umap':
            # For circular manifolds, UMAP needs specific tuning
            dr_kwargs = {
                'n_neighbors': 30,  # Reduced for tighter local structure
                'min_dist': 0.8,    # Balanced for circular preservation
                'metric': 'euclidean',
                'n_epochs': 500,    # More epochs for better convergence
                'init': 'spectral', # Better initialization for manifolds
                'spread': 1.0       # Standard spread
            }
        elif method == 'le':
            # Laplacian Eigenmaps for circular manifolds
            dr_kwargs = {
                'n_neighbors': 40,  # Smaller neighborhood for circular structure
            }
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
        pval_thr=0.01,  # More lenient for components
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
    
    # 2. Separate trajectory visualization
    create_trajectory_figure(exp, methods, save_plots)
    
    # 3. Component selectivity heatmap
    create_component_selectivity_heatmap(exp, embedding_results, save_plots)
    
    # 4. Functional organization analysis
    create_functional_organization_figure(exp, mapper, embedding_results, save_plots)
    
    # 5. Component interpretation figure
    create_component_interpretation_figure(exp, embedding_results, save_plots)
    
    plt.show()


def create_embedding_comparison_figure(exp, methods: List[str], save_plots: bool = False):
    """Create figure comparing embeddings colored by behavioral features."""
    from driada.utils.visual import plot_embedding_comparison
    
    # Prepare embeddings dict
    embeddings = {}
    for method in methods:
        embedding_dict = exp.get_embedding(method, 'calcium')
        if embedding_dict is not None:
            embeddings[method] = embedding_dict['data']
    
    # Get behavioral data
    features = {}
    if 'circular_position' in exp.dynamic_features:
        features['angle'] = exp.dynamic_features['circular_position'].data
    elif 'position_2d' in exp.dynamic_features:
        positions = exp.dynamic_features['position_2d'].data.T
        features['angle'] = np.arctan2(positions[:, 1] - 0.5, positions[:, 0] - 0.5)
    
    if 'c_feat_0' in exp.dynamic_features:
        features['speed'] = exp.dynamic_features['c_feat_0'].data
    
    # Create figure using visual utility
    fig = plot_embedding_comparison(
        embeddings=embeddings,
        features=features,
        methods=methods,
        with_trajectory=True,
        compute_metrics=True,
        save_path='selectivity_mapper_embeddings.png' if save_plots else None
    )
    
    if save_plots:
        print("Saved: selectivity_mapper_embeddings.png")
    
    return fig


def create_trajectory_figure(exp, methods: List[str], save_plots: bool = False):
    """Create separate figure showing trajectories in embedding space."""
    from driada.utils.visual import plot_trajectories
    
    # Prepare embeddings dict
    embeddings = {}
    for method in methods:
        embedding_dict = exp.get_embedding(method, 'calcium')
        if embedding_dict is not None:
            embeddings[method] = embedding_dict['data']
    
    # Create figure using visual utility
    fig = plot_trajectories(
        embeddings=embeddings,
        methods=methods,
        save_path='selectivity_mapper_trajectories.png' if save_plots else None
    )
    
    if save_plots:
        print("Saved: selectivity_mapper_trajectories.png")
    
    return fig


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
        # Note: stats structure is stats[neuron_id][feat_name]
        for neuron_id, neuron_stats in results['stats'].items():
            for feat_name, stats in neuron_stats.items():
                # Check if this is a component feature
                if isinstance(feat_name, str) and feat_name.startswith(f"{method}_comp"):
                    comp_idx = int(feat_name.split('_comp')[-1])
                    
                    if stats.get('me') is not None:
                        # Check if significant (correct significance structure)
                        if neuron_id in results['significance'] and \
                           feat_name in results['significance'][neuron_id] and \
                           results['significance'][neuron_id][feat_name].get('stage2', False):
                            selectivity_matrix[neuron_id, comp_idx] = stats['me']
        
        # Plot heatmap with proper scaling
        max_val = np.max(selectivity_matrix)
        if max_val > 0:
            im = ax.imshow(selectivity_matrix.T, aspect='auto', cmap='hot', 
                          interpolation='nearest', vmin=0, vmax=max_val)
        else:
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
    """Visualize how components relate to behavioral features using MI values."""
    from driada.utils.visual import plot_component_interpretation
    from driada.information.info_base import get_sim, TimeSeries
    
    # Get list of available methods
    available_methods = [method for method in embedding_results.keys() 
                        if method in ['pca', 'umap', 'le'] and embedding_results[method] is not None]
    
    if not available_methods:
        print("No DR methods found in results, skipping component interpretation")
        return
    
    # Get behavioral feature names and keys
    feature_names = []
    feature_keys = []
    
    if 'c_feat_0' in exp.dynamic_features:
        feature_names.append('Speed')
        feature_keys.append('c_feat_0')
    
    if 'c_feat_1' in exp.dynamic_features:
        feature_names.append('Reward magnitude')
        feature_keys.append('c_feat_1')
    
    if 'd_feat_0' in exp.dynamic_features:
        feature_names.append('Task type')
        feature_keys.append('d_feat_0')
    
    # Prepare MI matrices and metadata
    mi_matrices = {}
    metadata = {}
    
    ds = 5  # Same as used in INTENSE analysis
    
    for method in available_methods:
        try:
            embedding_dict = exp.get_embedding(method, 'calcium')
            embedding = embedding_dict['data']
            
            # Compute MI between components and behavioral features
            n_components = min(5, embedding_results[method]['n_components'])
            mi_matrix = np.zeros((len(feature_keys), n_components))
            
            for comp_idx in range(n_components):
                comp_data = embedding[:, comp_idx]
                
                for feat_idx, feat_key in enumerate(feature_keys):
                    try:
                        feat_data = exp.dynamic_features[feat_key].data
                        is_discrete = exp.dynamic_features[feat_key].discrete
                        
                        # Create TimeSeries objects
                        comp_ts = TimeSeries(comp_data, discrete=False)
                        feat_ts = TimeSeries(feat_data, discrete=is_discrete)
                        
                        # Compute MI
                        mi = get_sim(comp_ts, feat_ts, metric='mi', 
                                   shift=0, ds=ds, k=5, estimator='gcmi')
                        
                        mi_matrix[feat_idx, comp_idx] = mi
                        
                    except Exception as e:
                        print(f"Error computing MI for {method} comp{comp_idx} vs {feat_key}: {e}")
                        mi_matrix[feat_idx, comp_idx] = 0
            
            mi_matrices[method] = mi_matrix
            
            # Add metadata if available
            if method == 'pca' and 'metadata' in embedding_dict:
                metadata[method] = embedding_dict['metadata']
                
        except Exception as e:
            print(f"Failed to process {method}: {e}")
            continue
    
    # Create figure using visual utility
    fig = plot_component_interpretation(
        mi_matrices=mi_matrices,
        feature_names=feature_names,
        metadata=metadata,
        n_components=5,
        save_path='selectivity_mapper_component_interpretation.png' if save_plots else None
    )
    
    if save_plots:
        print("Saved: selectivity_mapper_component_interpretation.png")
    
    return fig


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
    
    # Step 4: Create visualizations (unless disabled)
    if not args.no_viz:
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