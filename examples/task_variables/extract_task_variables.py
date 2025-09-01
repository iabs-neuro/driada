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

from driada.experiment import generate_mixed_population_exp
from driada.intense import compute_cell_feat_significance
from driada.dimensionality import pca_dimension, eff_dim, nn_dimension
from driada.dim_reduction import knn_preservation_rate


def generate_task_data(duration=600, fps=20, seed=42):
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

    exp, info = generate_mixed_population_exp(
        n_neurons=200,
        manifold_type="2d_spatial",
        manifold_fraction=0.6,
        n_discrete_features=1,
        n_continuous_features=2,
        duration=duration,
        fps=fps,
        correlation_mode="spatial_correlated",
        seed=seed,
        manifold_params={
            "grid_arrangement": True,
            "field_sigma": 0.15,
            "noise_std": 0.05,
            "baseline_rate": 0.1,
            "peak_rate": 1.0,  # Realistic for calcium imaging
            "decay_time": 2.0,
            "calcium_noise_std": 0.05,
        },
        feature_params={
            "selectivity_prob": 0.9,
            "multi_select_prob": 0.6,
            "rate_0": 0.5,
            "rate_1": 4.0,
            "noise_std": 0.05,
            "hurst": 0.3,
            "skip_prob": 0.0,
            "ampl_range": (1.5, 3.5),
            "decay_time": 2.0,
        },
        return_info=True,
    )

    feature_mapping = {
        "d_feat_0": "reward",
        "c_feat_0": "speed",
        "c_feat_1": "head_direction",
    }

    print("\nFeature mapping:")
    for old_name, new_meaning in feature_mapping.items():
        if old_name in exp.dynamic_features:
            print(f"  - {old_name} represents {new_meaning}")

    if "position_2d" in exp.dynamic_features:
        pos = exp.dynamic_features["position_2d"].data
        velocity = np.diff(pos, axis=1)
        speed = np.sqrt(np.sum(velocity**2, axis=0))
        speed = np.concatenate([[0], speed])
        from scipy.ndimage import gaussian_filter1d

        speed = gaussian_filter1d(speed, sigma=5)
        speed = (speed - speed.min()) / (speed.max() - speed.min() + 1e-8)
        exp.dynamic_features["c_feat_0"].data = speed

    if "d_feat_0" in exp.dynamic_features and "position_2d" in exp.dynamic_features:
        reward_locations = np.array([[0.2, 0.2], [0.8, 0.8]])
        reward_radius = 0.1

        pos = exp.dynamic_features["position_2d"].data.T
        rewards = np.zeros(len(pos))

        for loc in reward_locations:
            distances = np.sqrt(np.sum((pos - loc) ** 2, axis=1))
            rewards[distances < reward_radius] = 1

        exp.dynamic_features["d_feat_0"].data = rewards.astype(int)

    print(f"Generated {exp.n_cells} neurons:")
    print(f"  - Pure manifold cells: ~{int(exp.n_cells * 0.6)}")
    print(f"  - Feature-selective cells: ~{int(exp.n_cells * 0.4)}")
    print(f"  - Expected mixed selectivity: ~{int(exp.n_cells * 0.4 * 0.6)}")
    print("  - Task variables: position (2D), speed, head_direction, reward")
    print(f"  - Recording duration: {duration}s at {fps} Hz")

    return exp, info


def analyze_single_cell_selectivity(exp, ds=5):
    """
    Perform INTENSE analysis to identify single-neuron selectivity patterns.

    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object
    ds : int
        Downsampling factor for INTENSE analysis

    Returns
    -------
    results : dict
        Analysis results including selectivity profiles
    """
    import time

    print("\n=== SINGLE-CELL SELECTIVITY ANALYSIS ===")
    print(f"Downsampling factor: {ds}")

    task_features = ["position_2d", "c_feat_0", "c_feat_1", "d_feat_0"]
    available_features = [f for f in task_features if f in exp.dynamic_features]

    print(
        f"Analyzing {exp.n_cells} neurons × {len(available_features)} features = {exp.n_cells * len(available_features)} pairs"
    )

    skip_delays = {}
    for feat_name in available_features:
        if (
            hasattr(exp.dynamic_features[feat_name], "data")
            and hasattr(exp.dynamic_features[feat_name].data, "ndim")
            and exp.dynamic_features[feat_name].data.ndim > 1
        ):
            skip_delays[feat_name] = True

    start = time.time()
    results = compute_cell_feat_significance(
        exp,
        feat_bunch=available_features,
        mode="two_stage",
        n_shuffles_stage1=50,
        n_shuffles_stage2=500,
        metric_distr_type="norm",
        pval_thr=0.01,
        multicomp_correction=None,
        verbose=True,
        find_optimal_delays=False,
        allow_mixed_dimensions=True,
        skip_delays=skip_delays,
        with_disentanglement=False,
        ds=ds,
    )
    print(f"INTENSE computation time: {time.time() - start:.2f}s")

    stats, significance, info, intense_results = results

    significant_neurons = exp.get_significant_neurons()
    mixed_selectivity_neurons = exp.get_significant_neurons(min_nspec=2)

    print("\nSelectivity Summary:")
    print(f"  - Total selective neurons: {len(significant_neurons)}/{exp.n_cells}")
    print(f"  - Mixed selectivity neurons: {len(mixed_selectivity_neurons)}")

    feature_counts = {}
    for neuron_id, features in significant_neurons.items():
        for feat in features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    print("\nSelectivity by feature:")
    for feat, count in sorted(feature_counts.items()):
        print(f"  - {feat}: {count} neurons")

    if mixed_selectivity_neurons:
        print("\nMixed selectivity patterns (top 5):")
        for i, (neuron_id, features) in enumerate(
            list(mixed_selectivity_neurons.items())[:5]
        ):
            print(f"  Neuron {neuron_id}: {', '.join(features)}")

    return {
        "stats": stats,
        "significance": significance,
        "significant_neurons": significant_neurons,
        "mixed_selectivity_neurons": mixed_selectivity_neurons,
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
        # exp.calcium is already a MultiTimeSeries which inherits from MVData
        # We can downsample it if needed
        if ds > 1:
            neural_data = exp.calcium.scdata[
                :, ::ds
            ]  # Use scaled data for equal neuron contributions
            from driada import MultiTimeSeries

            mvdata = MultiTimeSeries(neural_data, discrete=False)
        else:
            mvdata = exp.calcium  # Use directly - it's already an MVData!
    else:
        from driada.dim_reduction import MVData

        mvdata = MVData(neural_data)

    print("\nEstimating dimensionality:")
    # Get the actual data array for dimensionality estimation
    if hasattr(mvdata, "data"):
        data_array = mvdata.data
    else:
        data_array = neural_data
    data_t = data_array.T
    print(f"  - PCA 90% variance: {pca_dimension(data_t, threshold=0.90)} dimensions")
    print(f"  - PCA 95% variance: {pca_dimension(data_t, threshold=0.95)} dimensions")

    # Try eff_dim with correction, fallback to without if it fails
    try:
        eff_dim_value = eff_dim(data_t, enable_correction=True)
        print(f"  - Effective dimension (corrected): {eff_dim_value:.2f}")
    except Exception:
        # Correction can fail with near-singular matrices
        try:
            eff_dim_value = eff_dim(data_t, enable_correction=False)
            print(f"  - Effective dimension: {eff_dim_value:.2f}")
        except Exception as e2:
            print(f"  - Effective dimension: Failed ({str(e2)})")

    n_samples = min(1000, data_t.shape[0])
    sample_idx = np.random.choice(data_t.shape[0], n_samples, replace=False)
    data_sample = data_t[sample_idx]

    try:
        nn_dim = nn_dimension(data_sample, k=5)
        print(f"  - k-NN dimension: {nn_dim:.2f}")
    except:
        print("  - k-NN dimension: Failed")

    embeddings = {}

    print("\nApplying dimensionality reduction:")

    # Use the new simplified API
    pca_emb = mvdata.get_embedding(method="pca", dim=10)
    embeddings["pca"] = pca_emb.coords.T
    pca_var = np.var(pca_emb.coords, axis=1)
    pca_var_ratio = pca_var / np.sum(pca_var)
    print(f"  - PCA: captured {np.sum(pca_var_ratio[:2]):.1%} variance in 2D")

    # Use the new simplified API for Isomap
    isomap_emb = mvdata.get_embedding(
        method="isomap", dim=2, n_neighbors=30, max_deleted_nodes=0.2
    )
    embeddings["isomap"] = isomap_emb.coords.T
    print("  - Isomap: embedded into 2D manifold")

    # Use the new simplified API for UMAP
    umap_emb = mvdata.get_embedding(method="umap", dim=2, n_neighbors=50, min_dist=0.1)
    embeddings["umap"] = umap_emb.coords.T
    print("  - UMAP: embedded into 2D space")

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

    if "position_2d" in exp.dynamic_features:
        pos_data = exp.dynamic_features["position_2d"].data[:, ::ds]
        pos_x = pos_data[0]
        pos_y = pos_data[1]
    else:
        pos_x = pos_y = None

    speed = exp.dynamic_features.get("c_feat_0", None)
    if speed is not None:
        speed = speed.data[::ds]

    reward = exp.dynamic_features.get("d_feat_0", None)
    if reward is not None:
        reward = reward.data[::ds]

    for method_name, embedding in embeddings.items():
        correlations[method_name] = {}

        print(f"\n{method_name.upper()} correlations with task variables:")

        if pos_x is not None:
            from driada.dim_reduction import procrustes_analysis

            true_pos = np.column_stack([pos_x, pos_y])
            aligned_embedding, _ = procrustes_analysis(true_pos, embedding[:, :2])

            r_x = pearsonr(aligned_embedding[:, 0], pos_x)[0]
            r_y = pearsonr(aligned_embedding[:, 1], pos_y)[0]

            spatial_corr = np.sqrt(r_x**2 + r_y**2) / np.sqrt(2)
            correlations[method_name]["spatial"] = spatial_corr
            print(f"  - Spatial encoding (2D): {spatial_corr:.3f}")

        if speed is not None:
            speed_corrs = [
                abs(pearsonr(embedding[:, i], speed)[0])
                for i in range(min(3, embedding.shape[1]))
            ]
            best_speed_corr = max(speed_corrs)
            correlations[method_name]["speed"] = best_speed_corr
            print(f"  - Speed encoding: {best_speed_corr:.3f}")

        if reward is not None:
            reward_corrs = [
                abs(spearmanr(embedding[:, i], reward)[0])
                for i in range(min(3, embedding.shape[1]))
            ]
            best_reward_corr = max(reward_corrs)
            correlations[method_name]["reward"] = best_reward_corr
            print(f"  - Reward encoding: {best_reward_corr:.3f}")

        if pos_x is not None:
            from driada.dim_reduction.manifold_metrics import (
                trustworthiness,
                continuity,
            )
            from sklearn.metrics import pairwise_distances

            k = 10
            knn_score = knn_preservation_rate(true_pos, embedding[:, :2], k=k)
            correlations[method_name]["knn_preservation"] = knn_score
            print(f"  - k-NN preservation (k={k}): {knn_score:.3f}")

            trust = trustworthiness(true_pos, embedding[:, :2], k=k)
            correlations[method_name]["trustworthiness"] = trust
            print(f"  - Trustworthiness (k={k}): {trust:.3f}")

            cont = continuity(true_pos, embedding[:, :2], k=k)
            correlations[method_name]["continuity"] = cont
            print(f"  - Continuity (k={k}): {cont:.3f}")

            _, procrustes_dist = procrustes_analysis(true_pos, embedding[:, :2])
            correlations[method_name]["procrustes_distance"] = procrustes_dist
            print(f"  - Procrustes distance: {procrustes_dist:.3f}")

            true_dists = pairwise_distances(true_pos).flatten()
            emb_dists = pairwise_distances(embedding[:, :2]).flatten()
            dist_corr = spearmanr(true_dists, emb_dists)[0]
            correlations[method_name]["distance_correlation"] = dist_corr
            print(f"  - Distance correlation: {dist_corr:.3f}")

    return correlations


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
    from driada.utils.visual import plot_embedding_comparison, DEFAULT_DPI

    # Create figure 1: Embeddings using visual utility
    # Prepare features for coloring
    features = {}
    if "position_2d" in exp.dynamic_features:
        pos_data = exp.dynamic_features["position_2d"].data[:, ::ds]
        pos_x = pos_data[0]
        pos_y = pos_data[1]
        features["angle"] = np.arctan2(pos_y - 0.5, pos_x - 0.5)

    fig1 = plot_embedding_comparison(
        embeddings=embeddings,
        features=features,
        feature_names={"angle": "Position angle"},
        with_trajectory=False,  # Simple visualization without trajectory
        compute_metrics=False,  # No density contours needed
        figsize=(15, 5),
        save_path="task_variable_embeddings.png",
        dpi=DEFAULT_DPI,
    )
    plt.show()

    # Create figure 2: Metrics comparison
    fig2 = plt.figure(figsize=(12, 6))

    ax4 = fig2.add_subplot(1, 1, 1)
    methods = list(correlations.keys())

    all_metrics = [
        "spatial",
        "speed",
        "reward",
        "knn_preservation",
        "trustworthiness",
        "continuity",
        "distance_correlation",
        "procrustes_distance",
    ]

    available_metrics = []
    for metric in all_metrics:
        if any(metric in correlations[m] for m in methods):
            available_metrics.append(metric)

    corr_matrix = np.zeros((len(methods), len(available_metrics)))
    for i, method in enumerate(methods):
        for j, metric in enumerate(available_metrics):
            value = correlations[method].get(metric, 0)
            if metric == "procrustes_distance":
                max_proc_dist = max(
                    correlations[m].get("procrustes_distance", 0) for m in methods
                )
                if max_proc_dist > 0:
                    value = 1 - (value / max_proc_dist)
                else:
                    value = 1.0
            corr_matrix[i, j] = value

    im = ax4.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax4.set_xticks(range(len(available_metrics)))
    ax4.set_xticklabels(available_metrics, rotation=45, ha="right")
    ax4.set_yticks(range(len(methods)))
    ax4.set_yticklabels(methods)
    ax4.set_title("Task variable encoding and manifold quality metrics")
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation / Quality Score")

    for i in range(len(methods)):
        for j in range(len(available_metrics)):
            text_color = "white" if abs(corr_matrix[i, j] - 0.5) > 0.3 else "black"
            ax4.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig("task_variable_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Create figure 3: Single-cell vs Population encoding comparison
    fig3 = plt.figure(figsize=(8, 6))
    ax5 = fig3.add_subplot(1, 1, 1)

    # Get feature counts for single-cell encoding
    features = list(selectivity_results["significant_neurons"].values())
    feature_counts = {}
    for neuron_features in features:
        for feat in neuron_features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    single_cell_encoding = {}
    total_neurons = exp.n_cells
    for feat, count in feature_counts.items():
        single_cell_encoding[feat] = count / total_neurons

    population_encoding = {}
    encoding_metrics = ["spatial", "speed", "reward"]
    for metric in encoding_metrics:
        best_corr = max(correlations[m].get(metric, 0) for m in methods)
        population_encoding[metric] = best_corr

    feature_map = {
        "position_2d": "spatial",
        "c_feat_0": "speed",
        "d_feat_0": "reward",
        "c_feat_1": "head_direction",
    }

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

        ax5.bar(
            x - width / 2,
            single_values,
            width,
            label="Single-cell (fraction)",
            alpha=0.7,
        )
        ax5.bar(
            x + width / 2,
            population_values,
            width,
            label="Population (correlation)",
            alpha=0.7,
        )

        ax5.set_xlabel("Task variable")
        ax5.set_ylabel("Encoding strength")
        ax5.set_xticks(x)
        ax5.set_xticklabels(compared_features)
        ax5.legend()
        ax5.set_title("Single-cell vs Population encoding")
        ax5.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("task_variable_encoding_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Run the complete analysis pipeline."""
    import time

    print("=" * 70)
    print("TASK VARIABLE EXTRACTION FROM MIXED SELECTIVITY POPULATIONS")
    print("=" * 70)

    start_time = time.time()
    exp, info = generate_task_data(duration=600, fps=20, seed=42)
    print(f"\nTime for data generation: {time.time() - start_time:.2f}s")

    # Set downsampling factor for both INTENSE and DR analysis
    ds = 5

    start_time = time.time()
    selectivity_results = analyze_single_cell_selectivity(exp, ds=ds)
    print(f"Time for INTENSE analysis: {time.time() - start_time:.2f}s")

    start_time = time.time()
    embeddings = extract_population_structure(exp, ds=ds)
    print(f"Time for dimensionality reduction: {time.time() - start_time:.2f}s")

    start_time = time.time()
    correlations = compare_with_task_variables(exp, embeddings, ds=ds)
    print(f"Time for comparison: {time.time() - start_time:.2f}s")

    start_time = time.time()
    visualize_results(exp, embeddings, selectivity_results, correlations, ds=ds)
    print(f"Time for visualization: {time.time() - start_time:.2f}s")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("\n1. SINGLE-CELL ANALYSIS:")
    print(
        f"   - {len(selectivity_results['significant_neurons'])} neurons show selectivity"
    )
    print(
        f"   - {len(selectivity_results['mixed_selectivity_neurons'])} have mixed selectivity"
    )
    print("   - Individual neurons encode combinations of task variables")

    print("\n2. POPULATION ANALYSIS:")
    best_method = max(
        correlations.keys(), key=lambda m: correlations[m].get("spatial", 0)
    )
    print(f"   - Best method for spatial encoding: {best_method.upper()}")
    print(
        f"   - Spatial correlation: {correlations[best_method].get('spatial', 0):.3f}"
    )
    print("   - Population activity reveals continuous task structure")

    print("\n3. MANIFOLD QUALITY METRICS:")
    best_method = max(
        correlations.keys(), key=lambda m: correlations[m].get("spatial", 0)
    )
    print(f"   Best method: {best_method.upper()}")
    if best_method in correlations:
        metrics = correlations[best_method]
        print(f"   - Spatial correlation: {metrics.get('spatial', 0):.3f}")
        print(f"   - k-NN preservation: {metrics.get('knn_preservation', 0):.3f}")
        print(f"   - Trustworthiness: {metrics.get('trustworthiness', 0):.3f}")
        print(f"   - Continuity: {metrics.get('continuity', 0):.3f}")
        print(
            f"   - Distance correlation: {metrics.get('distance_correlation', 0):.3f}"
        )
        print(f"   - Procrustes distance: {metrics.get('procrustes_distance', 0):.3f}")

    print("\n4. TASK SPACE GEOMETRY:")
    print("   - Neural manifold reflects task structure")
    print("   - Different methods capture different aspects:")
    print("     * PCA: global linear relationships")
    print("     * Isomap: manifold geometry")
    print("     * UMAP: local and global structure")

    print("\n" + "=" * 70)
    print("Visualizations saved as:")
    print("  - task_variable_embeddings.png")
    print("  - task_variable_metrics.png")
    print("  - task_variable_encoding_comparison.png")

    return exp, selectivity_results, embeddings, correlations


if __name__ == "__main__":
    exp, selectivity_results, embeddings, correlations = main()
