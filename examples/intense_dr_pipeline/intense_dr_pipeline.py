"""
INTENSE → Dimensionality Reduction Pipeline Example
===================================================

This example demonstrates how INTENSE selectivity analysis can guide
dimensionality reduction to improve spatial manifold reconstruction.

Key concepts:
1. Use INTENSE to identify spatially selective neurons
2. Compare DR quality using all neurons vs. selective neurons only
3. Evaluate reconstruction using spatial correspondence metrics
"""

import numpy as np
import matplotlib.pyplot as plt

from driada import (
    compute_cell_feat_significance,
    generate_mixed_population_exp,
)
from driada.dim_reduction import MVData
from driada.utils import filter_signals, adaptive_filter_signals
from driada.utils import (
    compute_spatial_decoding_accuracy,
    compute_spatial_information,
)


def compute_spatial_correspondence_metrics(embedding, true_positions):
    """
    Compute spatial-specific metrics for evaluating embedding quality.

    Parameters
    ----------
    embedding : ndarray, shape (n_samples, n_dims)
        Low-dimensional embedding
    true_positions : ndarray, shape (n_samples, n_spatial_dims)
        True spatial positions

    Returns
    -------
    metrics : dict
        Dictionary containing spatial correspondence metrics
    """
    from scipy.spatial.distance import pdist
    from scipy.spatial import procrustes

    metrics = {}

    # 1. SPATIAL DECODING ACCURACY - Use library function
    # Need to transpose embedding for library function (expects n_neurons x n_samples)
    decoding_metrics = compute_spatial_decoding_accuracy(
        embedding.T,  # Transpose to (n_dims, n_samples)
        true_positions,
        test_size=0.5,
        n_estimators=20,
        max_depth=3,
        min_samples_leaf=50,
        random_state=42,
    )

    # Map to expected metric names
    metrics["spatial_decoding_r2_x"] = decoding_metrics["r2_x"]
    metrics["spatial_decoding_r2_y"] = decoding_metrics["r2_y"]
    metrics["spatial_decoding_r2_avg"] = decoding_metrics["r2_avg"]
    metrics["spatial_decoding_mse"] = decoding_metrics["mse"]

    # 2. SPATIAL INFORMATION CONTENT - Use library function
    mi_metrics = compute_spatial_information(
        embedding.T, true_positions  # Transpose to (n_dims, n_samples)
    )

    # Map to expected metric names
    metrics["spatial_mi_x"] = mi_metrics["mi_x"]
    metrics["spatial_mi_y"] = mi_metrics["mi_y"]
    metrics["spatial_mi_total"] = mi_metrics["mi_total"]

    # 3. DISTANCE CORRELATION
    # Pearson correlation between pairwise distances
    try:
        dist_embed = pdist(embedding)
        dist_true = pdist(true_positions)

        # Pearson correlation
        metrics["distance_correlation"] = np.corrcoef(dist_embed, dist_true)[0, 1]
    except:
        metrics["distance_correlation"] = 0.0

    # 4. IMPROVED PROCRUSTES ANALYSIS
    # Proper implementation without arbitrary padding
    try:
        # Only use first 2 dimensions of embedding for fair comparison
        embedding_2d = embedding[:, :2] if embedding.shape[1] >= 2 else embedding

        # Center and normalize
        true_centered = true_positions - true_positions.mean(axis=0)
        embed_centered = embedding_2d - embedding_2d.mean(axis=0)

        # Procrustes analysis
        _, _, disparity = procrustes(true_centered, embed_centered)
        metrics["procrustes_disparity"] = disparity
    except:
        metrics["procrustes_disparity"] = 1.0

    return metrics


def main(
    quick_test=False,
    seed=42,
    enable_visualizations=True,
    filter_method="none",
    filter_params=None,
    include_noisy=False,
):
    """Run the complete INTENSE → DR pipeline example

    Parameters
    ----------
    quick_test : bool, optional
        If True, use smaller parameters for faster execution
    seed : int, optional
        Random seed for reproducible results (default: 42)
    enable_visualizations : bool, optional
        If True, create and save visualization plots (default: True)
    filter_method : str, optional
        Signal filtering method: 'none', 'gaussian', 'savgol', 'adaptive'
        Default is 'none' (no filtering)
    filter_params : dict, optional
        Parameters for the chosen filter method:
        - For 'gaussian': {'sigma': float} (default: 1.0)
        - For 'savgol': {'window_length': int, 'polyorder': int}
        - For 'adaptive': {'snr_threshold': float} (default: 2.0)
    include_noisy : bool, optional
        If True, include noisy scenarios in the analysis (default: False)
    """

    print("=" * 70)
    print("INTENSE-Guided Dimensionality Reduction for Spatial Data")
    print("=" * 70)

    # 1. Generate mixed population data
    print("\n1. Generating mixed population with spatial and non-spatial neurons...")

    # Use smaller parameters for quick test
    if quick_test:
        n_neurons = 50  # Minimal for fast testing
        duration = 300  # 5 minutes
        n_shuffles_1 = 10
        n_shuffles_2 = 50
    else:
        n_neurons = 300  # More neurons for clearer effects
        duration = 1000  # ~17 minutes for better statistics
        n_shuffles_1 = 50
        n_shuffles_2 = 500

    # Always use downsampling for efficiency
    ds = 5

    exp = generate_mixed_population_exp(
        n_neurons=n_neurons,
        manifold_type="2d_spatial",
        manifold_fraction=0.5,  # 1/2 place cells, 1/2 feature cells
        manifold_params={
            "field_sigma": 0.15,  # Slightly wider place fields for better coverage
            "peak_rate": 1.0,  # Lower peak rate to preserve spatial selectivity
            "baseline_rate": 0.05,  # Lower baseline for better contrast
            "noise_std": 0.02,  # Lower noise for cleaner signal
            "decay_time": 2.0,
            "calcium_noise_std": 0.05,
        },
        n_discrete_features=3,
        n_continuous_features=3,
        correlation_mode="independent",
        duration=duration,
        seed=seed,
        verbose=True,
    )
    print(f"  Created experiment with {exp.n_cells} neurons, {exp.n_frames} timepoints")
    print(f"  Available features: {list(exp.dynamic_features.keys())}")

    # 2. Run INTENSE with position_2d MultiTimeSeries
    print("\n2. Running INTENSE analysis on 2D position (MultiTimeSeries)...")
    stats, significance, info, results = compute_cell_feat_significance(
        exp,
        feat_bunch=["position_2d", "x_position", "y_position"],  # Using MultiTimeSeries
        find_optimal_delays=False,  # Required for MultiTimeSeries
        mode="two_stage",
        n_shuffles_stage1=n_shuffles_1,
        n_shuffles_stage2=n_shuffles_2,
        ds=ds,  # Downsample for efficiency
        pval_thr=0.01,
        multicomp_correction=None,  # No multiple comparison correction
        allow_mixed_dimensions=True,  # Required for MultiTimeSeries
        verbose=True,
    )

    # 3. Categorize neurons by selectivity
    print("\n3. Categorizing neurons by selectivity...")

    # Get neurons selective to each spatial feature
    sig_neurons_2d = list(exp.get_significant_neurons(fbunch="position_2d").keys())
    sig_neurons_x = list(exp.get_significant_neurons(fbunch="x_position").keys())
    sig_neurons_y = list(exp.get_significant_neurons(fbunch="y_position").keys())

    # Combine all spatial neurons (remove duplicates)
    spatial_neurons = list(set(sig_neurons_2d + sig_neurons_x + sig_neurons_y))

    print(f"  Spatial neurons (position_2d): {len(sig_neurons_2d)}")
    print(f"  Spatial neurons (x_position): {len(sig_neurons_x)}")
    print(f"  Spatial neurons (y_position): {len(sig_neurons_y)}")
    print(f"  Spatial neurons (total unique): {len(spatial_neurons)}")
    print(f"  Non-spatial neurons: {exp.n_cells - len(spatial_neurons)}")

    # Check if we have enough spatial neurons
    if len(spatial_neurons) < 5:
        print("\nERROR: Not enough spatial neurons detected!")
        print("This example requires at least 5 spatially selective neurons.")
        print("Try running with more neurons or adjusting detection parameters.")
        return None, None

    # Extract true positions first (needed for verification)
    x_pos = exp.dynamic_features["x_position"].data
    y_pos = exp.dynamic_features["y_position"].data
    true_positions = np.column_stack([x_pos, y_pos])

    # Downsample positions to match calcium data if using downsampling
    if ds > 1:
        true_positions = true_positions[::ds]

    # 🔍 VERIFICATION: Check ground truth vs detected spatial neurons
    print("\n🔍 VERIFICATION: Analyzing ground truth vs detected spatial neurons...")

    # Get ground truth spatial neurons (first 50% are spatial by construction)
    n_true_spatial = int(exp.n_cells * 0.5)  # 50% are spatial
    true_spatial_neurons = list(range(n_true_spatial))
    true_nonspatial_neurons = list(range(n_true_spatial, exp.n_cells))

    print(
        f"  Ground truth spatial neurons: {n_true_spatial} (indices 0-{n_true_spatial-1})"
    )
    print(
        f"  Ground truth non-spatial neurons: {exp.n_cells - n_true_spatial} (indices {n_true_spatial}-{exp.n_cells-1})"
    )

    # Check detection accuracy
    detected_spatial_set = set(spatial_neurons)
    true_spatial_set = set(true_spatial_neurons)
    true_nonspatial_set = set(true_nonspatial_neurons)

    # True positives: correctly identified spatial neurons
    true_positives = detected_spatial_set & true_spatial_set
    # False positives: non-spatial neurons incorrectly identified as spatial
    false_positives = detected_spatial_set & true_nonspatial_set
    # False negatives: spatial neurons missed by detection
    false_negatives = true_spatial_set - detected_spatial_set

    print(f"  True positives (correctly detected spatial): {len(true_positives)}")
    print(
        f"  False positives (non-spatial detected as spatial): {len(false_positives)}"
    )
    print(f"  False negatives (spatial missed): {len(false_negatives)}")

    # Calculate detection metrics
    precision = (
        len(true_positives) / len(detected_spatial_set) if detected_spatial_set else 0
    )
    recall = len(true_positives) / len(true_spatial_set) if true_spatial_set else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"  Detection Precision: {precision:.3f}")
    print(f"  Detection Recall: {recall:.3f}")
    print(f"  Detection F1-score: {f1:.3f}")

    # 🔍 VERIFICATION: Test spatial decoding with ground truth neurons
    print("\n🔍 VERIFICATION: Testing spatial decoding with ground truth neurons...")

    # Test with true spatial neurons
    print("  Computing spatial decoding with TRUE spatial neurons...")
    calcium_true_spatial = (
        exp.calcium.scdata[true_spatial_neurons, ::ds]
        if ds > 1
        else exp.calcium.scdata[true_spatial_neurons]
    )

    # Simple spatial decoding test with regularized decoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    # Use 50/50 split to reduce overfitting
    X_train, X_test, y_train, y_test = train_test_split(
        calcium_true_spatial.T, true_positions, test_size=0.5, random_state=42
    )
    # Use regularized Random Forest to prevent overfitting
    decoder_true = RandomForestRegressor(
        n_estimators=20,  # Fewer trees
        max_depth=3,  # Limited depth
        min_samples_leaf=50,  # Require more samples per leaf
        random_state=42,
        n_jobs=-1,
    )
    decoder_true.fit(X_train, y_train)
    y_pred_true = decoder_true.predict(X_test)
    r2_true_spatial = (
        r2_score(y_test[:, 0], y_pred_true[:, 0])
        + r2_score(y_test[:, 1], y_pred_true[:, 1])
    ) / 2

    # Test with true non-spatial neurons
    print("  Computing spatial decoding with TRUE non-spatial neurons...")
    calcium_true_nonspatial = (
        exp.calcium.scdata[true_nonspatial_neurons, ::ds]
        if ds > 1
        else exp.calcium.scdata[true_nonspatial_neurons]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        calcium_true_nonspatial.T, true_positions, test_size=0.5, random_state=42
    )
    # Same regularized decoder for non-spatial neurons
    decoder_nonspatial = RandomForestRegressor(
        n_estimators=20, max_depth=3, min_samples_leaf=50, random_state=42, n_jobs=-1
    )
    decoder_nonspatial.fit(X_train, y_train)
    y_pred_nonspatial = decoder_nonspatial.predict(X_test)
    r2_true_nonspatial = (
        r2_score(y_test[:, 0], y_pred_nonspatial[:, 0])
        + r2_score(y_test[:, 1], y_pred_nonspatial[:, 1])
    ) / 2

    print(f"  Spatial decoding R² - TRUE spatial neurons: {r2_true_spatial:.3f}")
    print(f"  Spatial decoding R² - TRUE non-spatial neurons: {r2_true_nonspatial:.3f}")
    print(
        f"  Spatial vs Non-spatial ratio: {r2_true_spatial/max(r2_true_nonspatial, 0.001):.2f}x"
    )

    # Verify that non-spatial neurons are truly independent
    if r2_true_nonspatial > 0.4:
        print("  ⚠️  WARNING: Non-spatial neurons show significant spatial decoding!")
        print("  This suggests the synthetic data generation may have issues.")
    else:
        print("  ✅ Non-spatial neurons show minimal spatial decoding (as expected)")

    print(f"  Calcium data shape: {exp.calcium.scdata.shape}")
    print(f"  Downsampled positions shape: {true_positions.shape}")

    # 5. Create scenarios to demonstrate benefit
    print("\n4. Creating test scenarios...")

    # Get all neurons
    calcium_all = exp.calcium.scdata[
        :, ::ds
    ]  # Use scaled data for equal neuron contributions

    # Get spatial neurons (detected by INTENSE)
    calcium_spatial = exp.calcium.scdata[spatial_neurons, ::ds]

    # Get non-selective neurons (neither spatial nor detected for other features)
    all_neurons = set(range(exp.n_cells))
    selective_neurons = set(spatial_neurons)
    # Also get neurons selective to other features
    for feat in [
        "d_feat_0",
        "d_feat_1",
        "d_feat_2",
        "c_feat_0",
        "c_feat_1",
        "c_feat_2",
    ]:
        try:
            feat_neurons = exp.get_significant_neurons(fbunch=feat)
            if feat_neurons:
                selective_neurons.update(feat_neurons.keys())
        except:
            # Feature might not have been tested
            pass
    non_selective_neurons = list(all_neurons - selective_neurons)
    calcium_non_selective = (
        exp.calcium.scdata[non_selective_neurons, ::ds]
        if non_selective_neurons
        else None
    )

    # Get random half of all neurons
    np.random.seed(seed)  # For reproducibility
    random_half_idx = np.random.choice(
        exp.n_cells, size=exp.n_cells // 2, replace=False
    )
    calcium_random_half = exp.calcium.scdata[random_half_idx, ::ds]

    print(f"  All neurons: {calcium_all.shape[0]} neurons")
    print(f"  Spatial neurons (INTENSE): {calcium_spatial.shape[0]} neurons")
    print(f"  Random half: {calcium_random_half.shape[0]} neurons")
    print(f"  Non-selective neurons: {len(non_selective_neurons)} neurons")

    # Add noise variants for robustness testing
    noise_level = 0.5  # Moderate noise
    calcium_all_noisy = calcium_all + np.random.normal(
        0, noise_level, calcium_all.shape
    )
    calcium_spatial_noisy = calcium_spatial + np.random.normal(
        0, noise_level, calcium_spatial.shape
    )

    # 4.5 Apply optional signal filtering
    if filter_method != "none":
        print(f"\n4.5 Applying {filter_method} filtering to neural signals...")

        # Set default filter parameters if not provided
        if filter_params is None:
            filter_params = {
                "gaussian": {"sigma": 1.0},
                "savgol": {"window_length": 5, "polyorder": 2},
                "adaptive": {"snr_threshold": 2.0},
            }.get(filter_method, {})

        print(f"  Filter parameters: {filter_params}")

        # Apply filtering to all scenarios
        if filter_method == "adaptive":
            calcium_all = adaptive_filter_signals(calcium_all, **filter_params)
            calcium_spatial = adaptive_filter_signals(calcium_spatial, **filter_params)
            calcium_random_half = adaptive_filter_signals(
                calcium_random_half, **filter_params
            )
            if calcium_non_selective is not None:
                calcium_non_selective = adaptive_filter_signals(
                    calcium_non_selective, **filter_params
                )
            calcium_all_noisy = adaptive_filter_signals(
                calcium_all_noisy, **filter_params
            )
            calcium_spatial_noisy = adaptive_filter_signals(
                calcium_spatial_noisy, **filter_params
            )
        else:
            calcium_all = filter_signals(
                calcium_all, method=filter_method, **filter_params
            )
            calcium_spatial = filter_signals(
                calcium_spatial, method=filter_method, **filter_params
            )
            calcium_random_half = filter_signals(
                calcium_random_half, method=filter_method, **filter_params
            )
            if calcium_non_selective is not None:
                calcium_non_selective = filter_signals(
                    calcium_non_selective, method=filter_method, **filter_params
                )
            calcium_all_noisy = filter_signals(
                calcium_all_noisy, method=filter_method, **filter_params
            )
            calcium_spatial_noisy = filter_signals(
                calcium_spatial_noisy, method=filter_method, **filter_params
            )

        print("  Filtering completed successfully")

    # 6. Apply various DR methods using DRIADA syntax
    print("\n5. Applying dimensionality reduction methods...")

    # Define DR methods with DRIADA parameters
    dr_methods = {
        "PCA": {"method": "pca", "params": {"dim": 2}},
        "Isomap": {"method": "isomap", "params": {"dim": 2, "n_neighbors": 30}},
        "UMAP": {
            "method": "umap",
            "params": {
                "dim": 2,
                "n_neighbors": 80,
                "min_dist": 0.8,
                "random_state": 42,
            },
        },
    }

    results = {}

    # Test key scenarios
    scenarios = [
        ("All neurons", calcium_all),
        ("Spatial neurons", calcium_spatial),
        ("Random half", calcium_random_half),
        ("Non-selective", calcium_non_selective),
    ]

    # Optionally add noisy scenarios
    if include_noisy:
        scenarios.extend(
            [
                ("All neurons (noisy)", calcium_all_noisy),
                ("Spatial neurons (noisy)", calcium_spatial_noisy),
            ]
        )

    for method_name, method_config in dr_methods.items():
        print(f"\n  {method_name}:")
        results[method_name] = {}

        for scenario_name, calcium_data in scenarios:
            # Skip if no data or too few neurons
            if calcium_data is None:
                print(f"    - {scenario_name}: No neurons in this category, skipping")
                continue
            if calcium_data.shape[0] < 10:
                print(
                    f"    - {scenario_name}: Too few neurons ({calcium_data.shape[0]}), skipping"
                )
                continue

            print(f"    - {scenario_name}...")

            try:
                mvdata = MVData(calcium_data)  # MVData expects (n_features, n_samples)

                # Adjust n_neighbors for smaller datasets if needed
                params = method_config["params"].copy()
                if "n_neighbors" in params:
                    params["n_neighbors"] = min(
                        params["n_neighbors"], calcium_data.shape[1] // 10
                    )

                # Get embedding using new API
                embedding_obj = mvdata.get_embedding(
                    method=method_config["method"], **params
                )
                embedding = (
                    embedding_obj.coords.T
                )  # Extract coordinates, shape: (n_samples, n_dims)

                # Compute metrics
                metrics = compute_spatial_correspondence_metrics(
                    embedding, true_positions
                )

                results[method_name][scenario_name] = {
                    "embedding": embedding,
                    "metrics": metrics,
                }

                # Print key metrics
                print(
                    f"      Spatial decoding R²: {metrics['spatial_decoding_r2_avg']:.3f}, "
                    f"Distance corr: {metrics['distance_correlation']:.3f}, "
                    f"MI: {metrics['spatial_mi_total']:.3f}"
                )

            except Exception as e:
                print(f"      Failed: {e}")
                results[method_name][scenario_name] = None

        # Calculate improvements
        if (
            "All neurons" in results[method_name]
            and "Spatial neurons" in results[method_name]
        ):
            if (
                results[method_name]["All neurons"]
                and results[method_name]["Spatial neurons"]
            ):
                r2_all = results[method_name]["All neurons"]["metrics"][
                    "spatial_decoding_r2_avg"
                ]
                r2_spatial = results[method_name]["Spatial neurons"]["metrics"][
                    "spatial_decoding_r2_avg"
                ]
                improvement = (r2_spatial / max(r2_all, 0.001) - 1) * 100
                print(f"    Spatial vs All improvement: {improvement:+.1f}%")

        if (
            "All neurons" in results[method_name]
            and "Random half" in results[method_name]
        ):
            if (
                results[method_name]["All neurons"]
                and results[method_name]["Random half"]
            ):
                r2_all = results[method_name]["All neurons"]["metrics"][
                    "spatial_decoding_r2_avg"
                ]
                r2_random = results[method_name]["Random half"]["metrics"][
                    "spatial_decoding_r2_avg"
                ]
                ratio = r2_random / max(r2_all, 0.001)
                print(f"    Random half performance: {ratio:.2f}x of all neurons")

    # 6. Visualize results
    if enable_visualizations:
        print("\n5. Creating visualizations...")
        from driada.utils.visual import plot_embeddings_grid, DEFAULT_DPI

        # Prepare embeddings for grid plot
        grid_embeddings = {}
        grid_metrics = {}
        for method_name in dr_methods.keys():
            grid_embeddings[method_name] = {}
            grid_metrics[method_name] = {}
            for scenario in [
                "All neurons",
                "Spatial neurons",
                "Random half",
                "Non-selective",
            ]:
                if scenario in results[method_name] and results[method_name][scenario]:
                    grid_embeddings[method_name][scenario] = results[method_name][
                        scenario
                    ]["embedding"]
                    grid_metrics[method_name][scenario] = {
                        "R²": results[method_name][scenario]["metrics"][
                            "spatial_decoding_r2_avg"
                        ]
                    }

        # Create labels for coloring (time/trajectory position)
        labels = np.arange(len(true_positions))

        # Plot embeddings grid
        title = "INTENSE-Guided DR: Benefit of Spatial Neuron Selection"
        if filter_method != "none":
            title += f"\n(Filtered with {filter_method} method)"

        fig1 = plot_embeddings_grid(
            embeddings=grid_embeddings,
            labels=labels,
            metrics=grid_metrics,
            colormap="viridis",
            figsize=(18, 12),
            n_cols=4,
            save_path="intense_dr_pipeline_results.png",
            dpi=DEFAULT_DPI,
        )

        # Add custom title and true trajectory subplot
        fig1.suptitle(title, fontsize=14)

        # Find an empty subplot position to add true trajectory
        ax_true = fig1.add_subplot(3, 4, 1)
        ax_true.scatter(
            true_positions[:, 0],
            true_positions[:, 1],
            c=labels,
            cmap="viridis",
            s=1,
            alpha=0.5,
        )
        ax_true.set_title("True 2D Trajectory")
        ax_true.set_xlabel("X position")
        ax_true.set_ylabel("Y position")
        ax_true.set_aspect("equal")

        plt.show()

        # Create neuron selectivity summary using visual utility
        from driada.utils.visual import plot_neuron_selectivity_summary

        selectivity_counts = {
            "Spatial\n(position_2d)": len(sig_neurons_2d),
            "Spatial\n(x_position)": len(sig_neurons_x),
            "Spatial\n(y_position)": len(sig_neurons_y),
            "Non-spatial": exp.n_cells - len(spatial_neurons),
        }

        fig_summary = plot_neuron_selectivity_summary(
            selectivity_counts=selectivity_counts,
            total_neurons=exp.n_cells,
            figsize=(8, 6),
            save_path="intense_dr_neuron_selectivity.png",
            dpi=DEFAULT_DPI,
        )
        plt.show()

        # 6. Create quality metrics comparison figure
        print("\n6. Creating quality metrics comparison...")

        # Create comprehensive metrics visualization
        fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig2.suptitle(
            "Dimensionality Reduction Quality Metrics Comparison", fontsize=16
        )

        # Define metrics to plot
        metrics_to_show = [
            ("spatial_decoding_r2_avg", "Spatial Decoding R²"),
            ("distance_correlation", "Distance Correlation"),
            ("spatial_mi_total", "Spatial Information (MI)"),
            ("procrustes_disparity", "Procrustes Disparity"),
        ]

        # Key scenarios to show
        scenarios_to_show = [
            ("All neurons", "All"),
            ("Spatial neurons", "Spatial"),
            ("Random half", "Random"),
            ("Non-selective", "Non-sel"),
        ]

        # Add noisy scenarios if included
        if include_noisy:
            scenarios_to_show.extend(
                [
                    ("All neurons (noisy)", "All+Noise"),
                    ("Spatial neurons (noisy)", "Spatial+Noise"),
                ]
            )

        # Plot each metric
        for idx, (metric_key, metric_title) in enumerate(metrics_to_show):
            ax = axes[idx // 2, idx % 2]

            # Prepare data for this metric
            methods = list(dr_methods.keys())

            x = np.arange(len(scenarios_to_show))
            width = 0.25

            for i, method in enumerate(methods):
                values = []
                for scenario, _ in scenarios_to_show:
                    if scenario in results[method] and results[method][scenario]:
                        value = results[method][scenario]["metrics"][metric_key]
                        values.append(value)
                    else:
                        values.append(0)

                offset = (i - len(methods) / 2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=method, alpha=0.8)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    if value != 0:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{value:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            ax.set_ylabel(metric_title)
            ax.set_title(metric_title)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [label for _, label in scenarios_to_show], rotation=45, ha="right"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Set y-axis limits
            if "r2" in metric_key:
                ax.set_ylim(0, 1.0)
            elif metric_key == "procrustes_disparity":
                ax.set_ylim(0, max(1.0, max(values) * 1.1))

        plt.tight_layout()
        plt.savefig("intense_dr_pipeline_metrics.png", dpi=150, bbox_inches="tight")
        plt.show()

    # 7. Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if filter_method != "none":
        print(f"\nSignal filtering applied: {filter_method}")
        print(f"Filter parameters: {filter_params}")

    print("\nBest performing method for spatial reconstruction:")
    best_method = None
    best_score = -1
    for method_name in results.keys():
        if (
            "Spatial neurons" in results[method_name]
            and results[method_name]["Spatial neurons"]
        ):
            score = results[method_name]["Spatial neurons"]["metrics"][
                "spatial_decoding_r2_avg"
            ]
            if score > best_score:
                best_score = score
                best_method = method_name

    if best_method:
        print(f"  {best_method} with spatial neurons")
        print(f"  Spatial decoding R²: {best_score:.3f}")

    print("\nSummary of performance across scenarios:")

    print("\nSpatial decoding R² comparison:")
    for method_name in results.keys():
        print(f"\n  {method_name}:")
        scenarios_order = [
            "All neurons",
            "Spatial neurons",
            "Random half",
            "Non-selective",
        ]
        for scenario in scenarios_order:
            if scenario in results[method_name] and results[method_name][scenario]:
                r2 = results[method_name][scenario]["metrics"][
                    "spatial_decoding_r2_avg"
                ]
                print(f"    {scenario:20s}: {r2:.3f}")

        # Calculate key comparisons
        if (
            "All neurons" in results[method_name]
            and results[method_name]["All neurons"]
        ):
            r2_all = results[method_name]["All neurons"]["metrics"][
                "spatial_decoding_r2_avg"
            ]

            if (
                "Spatial neurons" in results[method_name]
                and results[method_name]["Spatial neurons"]
            ):
                r2_spatial = results[method_name]["Spatial neurons"]["metrics"][
                    "spatial_decoding_r2_avg"
                ]
                imp = (r2_spatial / max(r2_all, 0.001) - 1) * 100
                print(f"    → Spatial vs All improvement: {imp:+.1f}%")

            if (
                "Random half" in results[method_name]
                and results[method_name]["Random half"]
            ):
                r2_random = results[method_name]["Random half"]["metrics"][
                    "spatial_decoding_r2_avg"
                ]
                ratio = r2_random / max(r2_all, 0.001)
                print(f"    → Random half / All ratio: {ratio:.2f}")

    if include_noisy:
        print("\nNoisy data performance:")
        for method_name in results.keys():
            if (
                "All neurons (noisy)" in results[method_name]
                and "Spatial neurons (noisy)" in results[method_name]
            ):
                if (
                    results[method_name]["All neurons (noisy)"]
                    and results[method_name]["Spatial neurons (noisy)"]
                ):
                    r2_all = results[method_name]["All neurons (noisy)"]["metrics"][
                        "spatial_decoding_r2_avg"
                    ]
                    r2_spatial = results[method_name]["Spatial neurons (noisy)"][
                        "metrics"
                    ]["spatial_decoding_r2_avg"]
                    imp = (r2_spatial / max(r2_all, 0.001) - 1) * 100
                    print(
                        f"  {method_name}: All {r2_all:.3f} → Spatial {r2_spatial:.3f} ({imp:+.1f}%)"
                    )

    return exp, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="INTENSE-guided dimensionality reduction pipeline"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with reduced parameters"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for reproducible results (default: 42)",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization generation"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="none",
        choices=["none", "gaussian", "savgol", "adaptive"],
        help="Signal filtering method (default: none)",
    )
    parser.add_argument(
        "--filter-sigma",
        type=float,
        default=1.0,
        help="Sigma for Gaussian filter (default: 1.0)",
    )
    parser.add_argument(
        "--filter-window",
        type=int,
        default=5,
        help="Window length for Savitzky-Golay filter (default: 5)",
    )
    parser.add_argument(
        "--filter-polyorder",
        type=int,
        default=2,
        help="Polynomial order for Savitzky-Golay filter (default: 2)",
    )
    parser.add_argument(
        "--filter-snr",
        type=float,
        default=2.0,
        help="SNR threshold for adaptive filter (default: 2.0)",
    )
    parser.add_argument(
        "--include-noisy",
        action="store_true",
        help="Include noisy scenarios in the analysis",
    )

    args = parser.parse_args()

    # Prepare filter parameters based on method
    filter_params = None
    if args.filter != "none":
        if args.filter == "gaussian":
            filter_params = {"sigma": args.filter_sigma}
        elif args.filter == "savgol":
            filter_params = {
                "window_length": args.filter_window,
                "polyorder": args.filter_polyorder,
            }
        elif args.filter == "adaptive":
            filter_params = {"snr_threshold": args.filter_snr}

    exp, results = main(
        quick_test=args.quick,
        seed=args.seed,
        enable_visualizations=not args.no_viz,
        filter_method=args.filter,
        filter_params=filter_params,
        include_noisy=args.include_noisy,
    )
