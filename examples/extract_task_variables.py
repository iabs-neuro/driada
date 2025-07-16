"""
Extract task-relevant dimensions from mixed selectivity neural populations.

This example demonstrates how population-level dimensionality reduction reveals
task structure that is distributed across neurons with mixed selectivity:

1. Generate a mixed population combining place cells (2D manifold) with 
   task-modulated neurons responding to speed, direction, and rewards
2. Show how individual neurons have mixed selectivity to multiple variables
3. Apply dimensionality reduction to extract latent task-relevant dimensions
4. Demonstrate that population analysis reveals task structure better than
   single-cell analysis alone
5. Validate extracted dimensions against ground truth task variables

Key insights:
- Single neurons often respond to multiple correlated task variables
- Population activity contains low-dimensional structure reflecting task demands
- Combining INTENSE (single-cell) with DR (population) provides complete picture
- Task-relevant dimensions emerge naturally from neural population geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

# Import DRIADA modules
from driada.experiment import generate_mixed_population_exp
from driada.intense import compute_cell_feat_significance
from driada.dimensionality import pca_dimension, effective_rank, nn_dimension
from driada.dim_reduction import MVData, knn_preservation_rate, procrustes_analysis
from driada.dim_reduction.manifold_metrics import trustworthiness, continuity


def generate_task_data(duration=300, fps=20, seed=42):
    """
    Generate a mixed neural population engaged in a navigation task.
    
    The task involves:
    - Spatial navigation (place cells on 2D manifold)
    - Speed modulation (some neurons encode running speed)
    - Reward signals (some neurons respond to reward locations)
    - Mixed selectivity (many neurons respond to combinations)
    
    Parameters
    ----------
    duration : float
        Duration of experiment in seconds
    fps : float
        Sampling rate in Hz
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    exp : Experiment
        DRIADA Experiment object with mixed population
    info : dict
        Information about population composition
    """
    print("\n=== GENERATING TASK DATA ===")
    print("Task: 2D navigation with speed modulation and reward signals")
    
    # Generate mixed population with spatial manifold and task features
    exp, info = generate_mixed_population_exp(
        n_neurons=100,              # Total population size
        manifold_type='2d_spatial', # Place cells for spatial navigation
        manifold_fraction=0.4,      # 40% pure place cells
        n_discrete_features=1,      # Reward states (0/1)
        n_continuous_features=2,    # Speed and head direction
        duration=duration,
        fps=fps,
        correlation_mode='spatial_correlated',  # Task features correlate with position
        seed=seed,
        manifold_params={
            'grid_arrangement': True,  # Use grid arrangement
            'field_sigma': 0.15,      # Place field size
            'noise_std': 0.1,         # Noise level
            'baseline_rate': 0.1,     # Baseline firing rate
            'peak_rate': 2.0,         # Peak firing rate
            'decay_time': 2.0,        # Calcium decay time
            'calcium_noise_std': 0.1  # Calcium signal noise
        },
        feature_params={
            'selectivity_prob': 0.8,  # High selectivity to task features
            'multi_select_prob': 0.6, # Many neurons have mixed selectivity
            'rate_0': 0.5,
            'rate_1': 3.0,
            'noise_std': 0.1,
            'hurst': 0.8,             # Hurst parameter for fractional Brownian motion
            'skip_prob': 0.0,         # Skip probability for sparse firing
            'ampl_range': (1.5, 3.5), # Amplitude range for calcium events
            'decay_time': 2.0         # Decay time for calcium signals
        }
    )
    
    # Feature names are already set - just create a mapping for reference
    feature_mapping = {
        'd_feat_0': 'reward',
        'c_feat_0': 'speed', 
        'c_feat_1': 'head_direction'
    }
    
    print(f"\nFeature mapping:")
    for old_name, new_meaning in feature_mapping.items():
        if old_name in exp.dynamic_features:
            print(f"  - {old_name} represents {new_meaning}")
    
    # Generate additional task structure
    # Speed modulates with actual movement
    if 'position_2d' in exp.dynamic_features:
        pos = exp.dynamic_features['position_2d'].data
        velocity = np.diff(pos, axis=1)
        speed = np.sqrt(np.sum(velocity**2, axis=0))
        speed = np.concatenate([[0], speed])  # Pad first timepoint
        # Smooth and normalize
        from scipy.ndimage import gaussian_filter1d
        speed = gaussian_filter1d(speed, sigma=5)
        speed = (speed - speed.min()) / (speed.max() - speed.min() + 1e-8)
        exp.dynamic_features['c_feat_0'].data = speed
    
    # Rewards at specific locations
    if 'd_feat_0' in exp.dynamic_features and 'position_2d' in exp.dynamic_features:
        reward_locations = np.array([[0.2, 0.2], [0.8, 0.8]])  # Two reward zones
        reward_radius = 0.1
        
        pos = exp.dynamic_features['position_2d'].data.T
        rewards = np.zeros(len(pos))
        
        for loc in reward_locations:
            distances = np.sqrt(np.sum((pos - loc)**2, axis=1))
            rewards[distances < reward_radius] = 1
            
        exp.dynamic_features['d_feat_0'].data = rewards.astype(int)
    
    print(f"Generated {exp.n_cells} neurons:")
    print(f"  - Pure manifold cells: ~{int(100 * 0.4)}")
    print(f"  - Feature-selective cells: ~{int(100 * 0.6)}")
    print(f"  - Expected mixed selectivity in feature cells: ~{int(100 * 0.6 * 0.6)}")
    print(f"  - Task variables: position (2D), speed, head_direction, reward")
    print(f"  - Recording duration: {duration}s at {fps} Hz")
    
    return exp  # Return only the experiment object


def analyze_single_cell_selectivity(exp):
    """
    Perform INTENSE analysis to identify single-neuron selectivity patterns.
    
    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object
        
    Returns
    -------
    results : dict
        Analysis results including selectivity profiles
    """
    print("\n=== SINGLE-CELL SELECTIVITY ANALYSIS ===")
    
    # Focus on key task variables (use actual feature names)
    task_features = ['position_2d', 'c_feat_0', 'c_feat_1', 'd_feat_0']  # speed, head_direction, reward
    available_features = [f for f in task_features if f in exp.dynamic_features]
    
    # Run INTENSE analysis
    results = compute_cell_feat_significance(
        exp,
        feat_bunch=available_features,
        mode='two_stage',
        n_shuffles_stage1=20,
        n_shuffles_stage2=100,
        verbose=False,
        find_optimal_delays=False,  # Skip for speed
        allow_mixed_dimensions=True,  # Allow MultiTimeSeries
        with_disentanglement=False  # Skip for speed
    )
    
    # Unpack results (4 values when with_disentanglement=False)
    stats, significance, info, intense_results = results
    disentanglement_results = None
    
    # Analyze selectivity patterns
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
    
    # Analyze mixed selectivity patterns
    if mixed_selectivity_neurons:
        print(f"\nMixed selectivity patterns (top 5):")
        for i, (neuron_id, features) in enumerate(list(mixed_selectivity_neurons.items())[:5]):
            print(f"  Neuron {neuron_id}: {', '.join(features)}")
    
    return {
        'stats': stats,
        'significance': significance,
        'significant_neurons': significant_neurons,
        'mixed_selectivity_neurons': mixed_selectivity_neurons,
        'disentanglement': disentanglement_results
    }


def extract_population_structure(exp, neural_data=None):
    """
    Apply dimensionality reduction to extract task-relevant population structure.
    
    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object
    neural_data : ndarray, optional
        Neural activity matrix. If None, uses exp.calcium
        
    Returns
    -------
    embeddings : dict
        Dictionary of embeddings from different DR methods
    """
    print("\n=== POPULATION-LEVEL ANALYSIS ===")
    
    if neural_data is None:
        neural_data = exp.calcium
    
    # Transpose for sklearn format (n_samples, n_features)
    data_t = neural_data.T
    
    # Estimate dimensionality
    print("\nEstimating intrinsic dimensionality:")
    print(f"  - PCA 90% variance: {pca_dimension(data_t, threshold=0.90)} dimensions")
    print(f"  - PCA 95% variance: {pca_dimension(data_t, threshold=0.95)} dimensions")
    print(f"  - Effective rank: {effective_rank(data_t):.2f}")
    
    # Sample subset for nonlinear estimators
    n_samples = min(1000, data_t.shape[0])
    sample_idx = np.random.choice(data_t.shape[0], n_samples, replace=False)
    data_sample = data_t[sample_idx]
    
    try:
        nn_dim = nn_dimension(data_sample, k=5)
        print(f"  - k-NN dimension: {nn_dim:.2f}")
    except:
        print(f"  - k-NN dimension: Failed")
    
    # Apply dimensionality reduction methods
    embeddings = {}
    
    # PCA - captures global linear structure
    print("\nApplying dimensionality reduction:")
    pca = PCA(n_components=10)
    embeddings['pca'] = pca.fit_transform(data_t)
    print(f"  - PCA: captured {np.sum(pca.explained_variance_ratio_[:3]):.1%} variance in 3D")
    
    # Isomap - preserves geodesic distances, good for manifolds
    isomap = Isomap(n_components=3, n_neighbors=15)
    embeddings['isomap'] = isomap.fit_transform(data_t)
    print(f"  - Isomap: embedded into 3D manifold")
    
    # UMAP - preserves local and global structure
    umap_reducer = umap.UMAP(n_components=3, n_neighbors=15, random_state=42)
    embeddings['umap'] = umap_reducer.fit_transform(data_t)
    print(f"  - UMAP: embedded into 3D space")
    
    return embeddings


def compare_with_task_variables(exp, embeddings):
    """
    Compare extracted dimensions with ground truth task variables.
    
    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object
    embeddings : dict
        Dictionary of embeddings from different methods
        
    Returns
    -------
    correlations : dict
        Correlations between embedding dimensions and task variables
    """
    print("\n=== COMPARING WITH TASK VARIABLES ===")
    
    correlations = {}
    
    # Extract task variables
    if 'position_2d' in exp.dynamic_features:
        pos_data = exp.dynamic_features['position_2d'].data
        pos_x = pos_data[0]
        pos_y = pos_data[1]
    else:
        pos_x = pos_y = None
        
    speed = exp.dynamic_features.get('c_feat_0', None)  # speed
    if speed is not None:
        speed = speed.data
        
    reward = exp.dynamic_features.get('d_feat_0', None)  # reward
    if reward is not None:
        reward = reward.data
    
    # Compare each embedding with task variables
    for method_name, embedding in embeddings.items():
        correlations[method_name] = {}
        
        print(f"\n{method_name.upper()} correlations with task variables:")
        
        # Check spatial encoding (first 2 dimensions)
        if pos_x is not None:
            # Use Procrustes analysis to find best alignment
            true_pos = np.column_stack([pos_x, pos_y])
            aligned_embedding, _ = procrustes_analysis(true_pos, embedding[:, :2])
            
            r_x = pearsonr(aligned_embedding[:, 0], pos_x)[0]
            r_y = pearsonr(aligned_embedding[:, 1], pos_y)[0]
            
            # Overall spatial correlation
            spatial_corr = np.sqrt(r_x**2 + r_y**2) / np.sqrt(2)
            correlations[method_name]['spatial'] = spatial_corr
            print(f"  - Spatial encoding (2D): {spatial_corr:.3f}")
        
        # Check speed encoding (3rd dimension or best correlating dimension)
        if speed is not None:
            speed_corrs = [abs(pearsonr(embedding[:, i], speed)[0]) 
                           for i in range(min(3, embedding.shape[1]))]
            best_speed_corr = max(speed_corrs)
            correlations[method_name]['speed'] = best_speed_corr
            print(f"  - Speed encoding: {best_speed_corr:.3f}")
        
        # Check reward encoding
        if reward is not None:
            # Use Spearman for discrete variable
            reward_corrs = [abs(spearmanr(embedding[:, i], reward)[0]) 
                           for i in range(min(3, embedding.shape[1]))]
            best_reward_corr = max(reward_corrs)
            correlations[method_name]['reward'] = best_reward_corr
            print(f"  - Reward encoding: {best_reward_corr:.3f}")
        
        # Compute manifold quality metrics
        if pos_x is not None:
            # k-NN preservation between true positions and embedding
            k = 10
            knn_score = knn_preservation_rate(true_pos, embedding[:, :2], k=k)
            correlations[method_name]['knn_preservation'] = knn_score
            print(f"  - k-NN preservation (k={k}): {knn_score:.3f}")
    
    return correlations


def visualize_results(exp, embeddings, selectivity_results, correlations):
    """
    Create comprehensive visualization of results.
    
    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object
    embeddings : dict
        Dictionary of embeddings
    selectivity_results : dict
        Results from INTENSE analysis
    correlations : dict
        Correlations with task variables
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Single-cell selectivity summary
    ax1 = plt.subplot(3, 4, 1)
    features = list(selectivity_results['significant_neurons'].values())
    feature_counts = {}
    for neuron_features in features:
        for feat in neuron_features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    
    if feature_counts:
        ax1.bar(range(len(feature_counts)), list(feature_counts.values()))
        ax1.set_xticks(range(len(feature_counts)))
        ax1.set_xticklabels(list(feature_counts.keys()), rotation=45, ha='right')
        ax1.set_ylabel('Number of selective neurons')
        ax1.set_title('Single-cell selectivity')
    
    # 2. Mixed selectivity patterns
    ax2 = plt.subplot(3, 4, 2)
    n_features_per_neuron = [len(features) for features in selectivity_results['significant_neurons'].values()]
    if n_features_per_neuron:
        hist_data = np.bincount(n_features_per_neuron)
        ax2.bar(range(len(hist_data)), hist_data)
        ax2.set_xlabel('Number of features')
        ax2.set_ylabel('Number of neurons')
        ax2.set_title('Mixed selectivity distribution')
    
    # 3. Population embeddings (2D projections)
    for i, (method_name, embedding) in enumerate(embeddings.items()):
        ax = plt.subplot(3, 4, 4 + i + 1)
        
        # Color by position if available
        if 'position_2d' in exp.dynamic_features:
            pos_data = exp.dynamic_features['position_2d'].data
            pos_x = pos_data[0]
            pos_y = pos_data[1]
            colors = np.arctan2(pos_y - 0.5, pos_x - 0.5)  # Angular position
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=colors, cmap='hsv', alpha=0.5, s=1)
            plt.colorbar(scatter, ax=ax, label='Position angle')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=1)
        
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_title(f'{method_name.upper()} embedding')
    
    # 4. 3D visualization of best method
    best_method = max(correlations.keys(), 
                     key=lambda m: correlations[m].get('spatial', 0))
    ax3d = plt.subplot(3, 4, 8, projection='3d')
    embedding = embeddings[best_method]
    
    if 'c_feat_0' in exp.dynamic_features:  # speed
        colors = exp.dynamic_features['c_feat_0'].data
        scatter = ax3d.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                             c=colors, cmap='viridis', alpha=0.5, s=1)
        plt.colorbar(scatter, ax=ax3d, label='Speed', pad=0.1)
    else:
        ax3d.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                    alpha=0.5, s=1)
    
    ax3d.set_xlabel('Dim 1')
    ax3d.set_ylabel('Dim 2')
    ax3d.set_zlabel('Dim 3')
    ax3d.set_title(f'Best embedding ({best_method.upper()}) - 3D')
    
    # 5. Correlation summary
    ax4 = plt.subplot(3, 4, 10)
    methods = list(correlations.keys())
    metrics = ['spatial', 'speed', 'reward']
    
    corr_matrix = np.zeros((len(methods), len(metrics)))
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            corr_matrix[i, j] = correlations[method].get(metric, 0)
    
    im = ax4.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels(metrics)
    ax4.set_yticks(range(len(methods)))
    ax4.set_yticklabels(methods)
    ax4.set_title('Task variable encoding')
    plt.colorbar(im, ax=ax4)
    
    # Add values to heatmap
    for i in range(len(methods)):
        for j in range(len(metrics)):
            ax4.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                    ha='center', va='center')
    
    # 6. Comparison: single-cell vs population
    ax5 = plt.subplot(3, 4, 12)
    
    # Single-cell: fraction of neurons encoding each variable
    single_cell_encoding = {}
    total_neurons = exp.n_cells
    for feat, count in feature_counts.items():
        single_cell_encoding[feat] = count / total_neurons
    
    # Population: best correlation across methods
    population_encoding = {}
    for metric in metrics:
        best_corr = max(correlations[m].get(metric, 0) for m in methods)
        population_encoding[metric] = best_corr
    
    # Map feature names for comparison
    feature_map = {
        'position_2d': 'spatial',
        'c_feat_0': 'speed',
        'd_feat_0': 'reward',
        'c_feat_1': 'head_direction'
    }
    
    # Create comparison bars
    compared_features = []
    single_values = []
    population_values = []
    
    for feat, mapped in feature_map.items():
        if feat in single_cell_encoding and mapped in population_encoding:
            compared_features.append(mapped)
            single_values.append(single_cell_encoding[feat])
            population_values.append(population_encoding[mapped])
    
    if compared_features:
        x = np.arange(len(compared_features))
        width = 0.35
        
        ax5.bar(x - width/2, single_values, width, label='Single-cell (fraction)', alpha=0.7)
        ax5.bar(x + width/2, population_values, width, label='Population (correlation)', alpha=0.7)
        
        ax5.set_xlabel('Task variable')
        ax5.set_ylabel('Encoding strength')
        ax5.set_xticks(x)
        ax5.set_xticklabels(compared_features)
        ax5.legend()
        ax5.set_title('Single-cell vs Population encoding')
        ax5.set_ylim(0, 1)
    
    plt.suptitle('Task Variable Extraction from Mixed Selectivity Population', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('task_variable_extraction.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run the complete analysis pipeline."""
    print("=" * 70)
    print("TASK VARIABLE EXTRACTION FROM MIXED SELECTIVITY POPULATIONS")
    print("=" * 70)
    
    # Generate task data
    exp = generate_task_data(duration=300, fps=20, seed=42)
    
    # Analyze single-cell selectivity
    selectivity_results = analyze_single_cell_selectivity(exp)
    
    # Extract population structure
    embeddings = extract_population_structure(exp)
    
    # Compare with task variables
    correlations = compare_with_task_variables(exp, embeddings)
    
    # Visualize results
    visualize_results(exp, embeddings, selectivity_results, correlations)
    
    # Summary insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    print("\n1. SINGLE-CELL ANALYSIS:")
    print(f"   - {len(selectivity_results['significant_neurons'])} neurons show selectivity")
    print(f"   - {len(selectivity_results['mixed_selectivity_neurons'])} have mixed selectivity")
    print("   - Individual neurons encode combinations of task variables")
    
    print("\n2. POPULATION ANALYSIS:")
    best_method = max(correlations.keys(), 
                     key=lambda m: correlations[m].get('spatial', 0))
    print(f"   - Best method for spatial encoding: {best_method.upper()}")
    print(f"   - Spatial correlation: {correlations[best_method].get('spatial', 0):.3f}")
    print("   - Population activity reveals continuous task structure")
    
    print("\n3. ADVANTAGE OF POPULATION APPROACH:")
    print("   - Single cells: discrete, mixed selectivity patterns")
    print("   - Population: continuous, interpretable task dimensions")
    print("   - Dimensionality reduction extracts latent task variables")
    print("   - Noise averaging across population improves signal")
    
    print("\n4. TASK SPACE GEOMETRY:")
    print("   - Neural manifold reflects task structure")
    print("   - Different methods capture different aspects:")
    print("     * PCA: global linear relationships")
    print("     * Isomap: manifold geometry")  
    print("     * UMAP: local and global structure")
    
    print("\n" + "=" * 70)
    print("Visualization saved as 'task_variable_extraction.png'")
    
    return exp, selectivity_results, embeddings, correlations


if __name__ == "__main__":
    # Run the analysis
    exp, selectivity_results, embeddings, correlations = main()