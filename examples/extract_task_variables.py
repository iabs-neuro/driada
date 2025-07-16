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
        n_neurons=200,              # Total population size
        manifold_type='2d_spatial', # Place cells for spatial navigation
        manifold_fraction=0.6,      # 60% pure place cells
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
            'hurst': 0.3,             # Hurst parameter for fractional Brownian motion
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
    print(f"  - Pure manifold cells: ~{int(100 * 0.6)}")
    print(f"  - Feature-selective cells: ~{int(100 * 0.4)}")
    print(f"  - Expected mixed selectivity in feature cells: ~{int(100 * 0.4 * 0.4)}")
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
    import time
    print("\n=== SINGLE-CELL SELECTIVITY ANALYSIS ===")
    
    # Focus on key task variables (use actual feature names)
    task_features = ['position_2d', 'c_feat_0', 'c_feat_1', 'd_feat_0']  # speed, head_direction, reward
    available_features = [f for f in task_features if f in exp.dynamic_features]
    
    print(f"Analyzing {exp.n_cells} neurons × {len(available_features)} features = {exp.n_cells * len(available_features)} pairs")
    print(f"Stage 1: 50 shuffles, Stage 2: 500 shuffles")
    
    # Run INTENSE analysis
    start = time.time()
    results = compute_cell_feat_significance(
        exp,
        feat_bunch=available_features,
        mode='two_stage',
        n_shuffles_stage1=50,      # Reduced for faster screening
        n_shuffles_stage2=500,
        metric_distr_type='norm',  # NOTE: norm's conservative p-values reduce false positives (see docs)
        pval_thr=0.05,             # More lenient threshold
        verbose=True,              # Enable to see progress
        find_optimal_delays=False,  # Disabled due to MultiTimeSeries incompatibility
        allow_mixed_dimensions=True,  # Allow MultiTimeSeries (position_2d)
        with_disentanglement=False  # Disable for faster execution
    )
    print(f"INTENSE computation time: {time.time() - start:.2f}s")
    
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


def extract_population_structure(exp, neural_data=None, ds=5):
    """
    Apply dimensionality reduction to extract task-relevant population structure.
    
    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object
    neural_data : ndarray, optional
        Neural activity matrix. If None, uses exp.calcium
    ds : int
        Downsampling factor for dimensionality reduction
        
    Returns
    -------
    embeddings : dict
        Dictionary of embeddings from different DR methods
    """
    print("\n=== POPULATION-LEVEL ANALYSIS ===")
    print(f"Downsampling factor: {ds}")

    if neural_data is None:
        neural_data = exp.calcium[:, ::ds]
    
    # Create MVData object for DRIADA's dimensionality reduction
    from driada.dim_reduction import MVData, METHODS_DICT
    mvdata = MVData(neural_data)  # MVData expects (n_features, n_samples)
    
    # Estimate dimensionality
    print("\nEstimating dimensionality:")
    data_t = neural_data.T  # For dimensionality estimation functions
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
    
    # Apply dimensionality reduction methods using DRIADA
    embeddings = {}
    
    print("\nApplying dimensionality reduction:")
    
    # PCA - captures global linear structure
    pca_params = {
        'e_method_name': 'pca',
        'dim': 10,
        'e_method': METHODS_DICT['pca']
    }
    pca_emb = mvdata.get_embedding(pca_params)
    embeddings['pca'] = pca_emb.coords.T  # Transpose to (n_samples, n_dims)
    # Calculate variance explained
    pca_var = np.var(pca_emb.coords, axis=1)
    pca_var_ratio = pca_var / np.sum(pca_var)
    print(f"  - PCA: captured {np.sum(pca_var_ratio[:3]):.1%} variance in 3D")
    
    # Define metric and graph parameters for nonlinear methods
    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }
    
    # Isomap - preserves geodesic distances, good for manifolds
    isomap_graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 30,  # Increased from 15
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    isomap_params = {
        'e_method_name': 'isomap',
        'dim': 3,
        'e_method': METHODS_DICT['isomap']
    }
    isomap_emb = mvdata.get_embedding(isomap_params, g_params=isomap_graph_params, m_params=metric_params)
    embeddings['isomap'] = isomap_emb.coords.T
    print(f"  - Isomap: embedded into 3D manifold")
    
    # UMAP - preserves local and global structure
    umap_graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 30,  # Increased from 15
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    umap_params = {
        'e_method_name': 'umap',
        'dim': 3,
        'min_dist': 0.1,
        'e_method': METHODS_DICT['umap']
    }
    umap_emb = mvdata.get_embedding(umap_params, g_params=umap_graph_params, m_params=metric_params)
    embeddings['umap'] = umap_emb.coords.T
    print(f"  - UMAP: embedded into 3D space")
    
    return embeddings


def compare_with_task_variables(exp, embeddings, ds=5):
    """
    Compare extracted dimensions with ground truth task variables.
    
    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object
    embeddings : dict
        Dictionary of embeddings from different methods
    ds : int
        Downsampling factor used in embeddings
        
    Returns
    -------
    correlations : dict
        Correlations between embedding dimensions and task variables
    """
    print("\n=== COMPARING WITH TASK VARIABLES ===")
    
    correlations = {}
    
    # Extract task variables with same downsampling
    if 'position_2d' in exp.dynamic_features:
        pos_data = exp.dynamic_features['position_2d'].data[:, ::ds]
        pos_x = pos_data[0]
        pos_y = pos_data[1]
    else:
        pos_x = pos_y = None
        
    speed = exp.dynamic_features.get('c_feat_0', None)  # speed
    if speed is not None:
        speed = speed.data[::ds]
        
    reward = exp.dynamic_features.get('d_feat_0', None)  # reward
    if reward is not None:
        reward = reward.data[::ds]
    
    # Compare each embedding with task variables
    for method_name, embedding in embeddings.items():
        correlations[method_name] = {}
        
        print(f"\n{method_name.upper()} correlations with task variables:")
        
        # Check spatial encoding (first 2 dimensions)
        if pos_x is not None:
            # Use Procrustes analysis to find best alignment
            from driada.dim_reduction import procrustes_analysis
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
            # Import additional metrics
            from driada.dim_reduction.manifold_metrics import trustworthiness, continuity
            from sklearn.metrics import pairwise_distances
            
            # k-NN preservation between true positions and embedding
            k = 10
            knn_score = knn_preservation_rate(true_pos, embedding[:, :2], k=k)
            correlations[method_name]['knn_preservation'] = knn_score
            print(f"  - k-NN preservation (k={k}): {knn_score:.3f}")
            
            # Trustworthiness: measures if neighbors in embedding are true neighbors
            trust = trustworthiness(true_pos, embedding[:, :2], k=k)
            correlations[method_name]['trustworthiness'] = trust
            print(f"  - Trustworthiness (k={k}): {trust:.3f}")
            
            # Continuity: measures if true neighbors remain neighbors in embedding
            cont = continuity(true_pos, embedding[:, :2], k=k)
            correlations[method_name]['continuity'] = cont
            print(f"  - Continuity (k={k}): {cont:.3f}")
            
            # Procrustes distance after alignment
            _, procrustes_dist = procrustes_analysis(true_pos, embedding[:, :2])
            correlations[method_name]['procrustes_distance'] = procrustes_dist
            print(f"  - Procrustes distance: {procrustes_dist:.3f}")
            
            # Spearman correlation of pairwise distances
            true_dists = pairwise_distances(true_pos).flatten()
            emb_dists = pairwise_distances(embedding[:, :2]).flatten()
            dist_corr = spearmanr(true_dists, emb_dists)[0]
            correlations[method_name]['distance_correlation'] = dist_corr
            print(f"  - Distance correlation: {dist_corr:.3f}")
    
    return correlations


def visualize_trajectories(exp, embeddings, ds=5):
    """
    Visualize true and reconstructed trajectories.
    
    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object
    embeddings : dict
        Dictionary of embeddings from different methods
    ds : int
        Downsampling factor used in embeddings
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Trajectory visualization figure
    """
    if 'position_2d' not in exp.dynamic_features:
        return None
        
    # Extract true trajectory with same downsampling
    pos_data = exp.dynamic_features['position_2d'].data[:, ::ds]
    pos_x = pos_data[0]
    pos_y = pos_data[1]
    time_points = np.arange(len(pos_x))
    
    # Create figure
    n_methods = len(embeddings)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(5*(n_methods + 1), 4))
    
    # Plot true trajectory
    ax = axes[0]
    scatter = ax.scatter(pos_x, pos_y, c=time_points, cmap='viridis', s=1, alpha=0.5)
    ax.plot(pos_x[:100], pos_y[:100], 'r-', alpha=0.5, linewidth=1, label='Start')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('True trajectory')
    ax.set_aspect('equal')
    ax.legend()
    
    # Plot reconstructed trajectories
    for i, (method_name, embedding) in enumerate(embeddings.items()):
        ax = axes[i + 1]
        
        # Use Procrustes to align with true trajectory
        from driada.dim_reduction import procrustes_analysis
        aligned_embedding, _ = procrustes_analysis(
            np.column_stack([pos_x, pos_y]), 
            embedding[:, :2]
        )
        
        # Plot
        scatter = ax.scatter(aligned_embedding[:, 0], aligned_embedding[:, 1], 
                           c=time_points, cmap='viridis', s=1, alpha=0.5)
        ax.plot(aligned_embedding[:100, 0], aligned_embedding[:100, 1], 
                'r-', alpha=0.5, linewidth=1, label='Start')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_title(f'{method_name.upper()} reconstruction')
        ax.set_aspect('equal')
        
    plt.tight_layout()
    return fig


def visualize_results(exp, embeddings, selectivity_results, correlations, ds=5):
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
    ds : int
        Downsampling factor used
    """
    fig = plt.figure(figsize=(24, 18))
    
    # Adjust subplot layout to prevent overlap
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.4)
    
    # 1. Single-cell selectivity summary
    ax1 = fig.add_subplot(gs[0, 0])
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
    ax2 = fig.add_subplot(gs[0, 1])
    n_features_per_neuron = [len(features) for features in selectivity_results['significant_neurons'].values()]
    if n_features_per_neuron:
        hist_data = np.bincount(n_features_per_neuron)
        ax2.bar(range(len(hist_data)), hist_data)
        ax2.set_xlabel('Number of features')
        ax2.set_ylabel('Number of neurons')
        ax2.set_title('Mixed selectivity distribution')
    
    # 3. Population embeddings (2D projections)
    for i, (method_name, embedding) in enumerate(embeddings.items()):
        ax = fig.add_subplot(gs[1, i])
        
        # Color by position if available - use downsampled data
        if 'position_2d' in exp.dynamic_features:
            pos_data = exp.dynamic_features['position_2d'].data[:, ::ds]
            pos_x = pos_data[0]
            pos_y = pos_data[1]
            colors = np.arctan2(pos_y - 0.5, pos_x - 0.5)  # Angular position
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=colors, cmap='hsv', alpha=0.5, s=10)
            cbar = plt.colorbar(scatter, ax=ax, label='Position angle', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=10)
        
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_title(f'{method_name.upper()} embedding')
    
    # 4. 3D visualization of best method
    best_method = max(correlations.keys(), 
                     key=lambda m: correlations[m].get('spatial', 0))
    ax3d = fig.add_subplot(gs[1, 3], projection='3d')
    embedding = embeddings[best_method]
    
    if 'c_feat_0' in exp.dynamic_features:  # speed - use downsampled
        colors = exp.dynamic_features['c_feat_0'].data[::ds]
        scatter = ax3d.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                             c=colors, cmap='viridis', alpha=0.6, s=20)
        cbar = plt.colorbar(scatter, ax=ax3d, label='Speed', pad=0.15, fraction=0.046, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
    else:
        ax3d.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                    alpha=0.6, s=20)
    
    ax3d.set_xlabel('Dim 1', labelpad=10)
    ax3d.set_ylabel('Dim 2', labelpad=10)
    ax3d.set_zlabel('Dim 3', labelpad=10)
    ax3d.set_title(f'Best embedding ({best_method.upper()}) - 3D', pad=20)
    ax3d.view_init(elev=20, azim=45)
    
    # 5. Correlation summary - expanded with spatial metrics
    ax4 = fig.add_subplot(gs[2:3, 0:4])  # Use more space for expanded metrics
    methods = list(correlations.keys())
    
    # All metrics to display
    all_metrics = ['spatial', 'speed', 'reward', 'knn_preservation', 
                   'trustworthiness', 'continuity', 'distance_correlation', 'procrustes_distance']
    
    # Create matrix with available metrics
    available_metrics = []
    for metric in all_metrics:
        if any(metric in correlations[m] for m in methods):
            available_metrics.append(metric)
    
    corr_matrix = np.zeros((len(methods), len(available_metrics)))
    for i, method in enumerate(methods):
        for j, metric in enumerate(available_metrics):
            value = correlations[method].get(metric, 0)
            # Normalize Procrustes distance (lower is better -> higher score)
            if metric == 'procrustes_distance':
                # Normalize to [0, 1] where lower distance = higher score
                max_proc_dist = max(correlations[m].get('procrustes_distance', 0) for m in methods)
                if max_proc_dist > 0:
                    value = 1 - (value / max_proc_dist)
                else:
                    value = 1.0
            corr_matrix[i, j] = value
    
    im = ax4.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
    ax4.set_xticks(range(len(available_metrics)))
    ax4.set_xticklabels(available_metrics, rotation=45, ha='right')
    ax4.set_yticks(range(len(methods)))
    ax4.set_yticklabels(methods)
    ax4.set_title('Task variable encoding and manifold quality metrics')
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation / Quality Score')
    
    # Add values to heatmap
    for i in range(len(methods)):
        for j in range(len(available_metrics)):
            text_color = 'white' if abs(corr_matrix[i, j] - 0.5) > 0.3 else 'black'
            ax4.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                    ha='center', va='center', color=text_color, fontsize=8)
    
    # 6. Comparison: single-cell vs population
    ax5 = fig.add_subplot(gs[3, 1:3])
    
    # Single-cell: fraction of neurons encoding each variable
    single_cell_encoding = {}
    total_neurons = exp.n_cells
    for feat, count in feature_counts.items():
        single_cell_encoding[feat] = count / total_neurons
    
    # Population: best correlation across methods
    population_encoding = {}
    encoding_metrics = ['spatial', 'speed', 'reward']  # Only task variables for comparison
    for metric in encoding_metrics:
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
    
    plt.suptitle('Task Variable Extraction from Mixed Selectivity Population', fontsize=16, y=0.99)
    plt.savefig('task_variable_extraction.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run the complete analysis pipeline."""
    import time
    
    print("=" * 70)
    print("TASK VARIABLE EXTRACTION FROM MIXED SELECTIVITY POPULATIONS")
    print("=" * 70)
    
    # Generate task data
    start_time = time.time()
    exp = generate_task_data(duration=300, fps=20, seed=42)
    print(f"\nTime for data generation: {time.time() - start_time:.2f}s")
    
    # Analyze single-cell selectivity
    start_time = time.time()
    selectivity_results = analyze_single_cell_selectivity(exp)
    print(f"Time for INTENSE analysis: {time.time() - start_time:.2f}s")
    
    # Extract population structure
    start_time = time.time()
    ds = 5  # Downsampling factor for DR methods
    embeddings = extract_population_structure(exp, ds=ds)
    print(f"Time for dimensionality reduction: {time.time() - start_time:.2f}s")
    
    # Compare with task variables
    start_time = time.time()
    correlations = compare_with_task_variables(exp, embeddings, ds=ds)
    print(f"Time for comparison: {time.time() - start_time:.2f}s")
    
    # Visualize results
    start_time = time.time()
    visualize_results(exp, embeddings, selectivity_results, correlations, ds=ds)
    print(f"Time for main visualization: {time.time() - start_time:.2f}s")
    
    # Visualize trajectories
    start_time = time.time()
    traj_fig = visualize_trajectories(exp, embeddings, ds=ds)
    if traj_fig:
        plt.savefig('task_variable_trajectories.png', dpi=150, bbox_inches='tight')
        plt.show()
    print(f"Time for trajectory visualization: {time.time() - start_time:.2f}s")
    
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
    
    print("\n3. MANIFOLD QUALITY METRICS:")
    best_method = max(correlations.keys(), 
                     key=lambda m: correlations[m].get('spatial', 0))
    print(f"   Best method: {best_method.upper()}")
    if best_method in correlations:
        metrics = correlations[best_method]
        print(f"   - Spatial correlation: {metrics.get('spatial', 0):.3f}")
        print(f"   - k-NN preservation: {metrics.get('knn_preservation', 0):.3f}")
        print(f"   - Trustworthiness: {metrics.get('trustworthiness', 0):.3f}")
        print(f"   - Continuity: {metrics.get('continuity', 0):.3f}")
        print(f"   - Distance correlation: {metrics.get('distance_correlation', 0):.3f}")
        print(f"   - Procrustes distance: {metrics.get('procrustes_distance', 0):.3f}")
    
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