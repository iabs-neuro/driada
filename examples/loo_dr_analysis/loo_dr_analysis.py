#!/usr/bin/env python
"""
Leave-One-Out Dimension Reduction Analysis for DRIADA

This script performs dimension reduction using any available method in DRIADA
with a leave-one-out approach for neurons, computing comprehensive metrics
for space reconstruction quality.

The LOO importance scores are compared with INTENSE selectivity metrics to
validate whether neurons important for manifold reconstruction are the same
neurons that encode the behavioral feature.

Works with any DR method available in DRIADA and directly with Experiment objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Any, List
from tqdm import tqdm
import warnings
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# DRIADA imports
from driada.experiment import Experiment
from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.dim_reduction.manifold_metrics import (
    knn_preservation_rate,
    trustworthiness,
    continuity,
    geodesic_distance_correlation,
    stress,
    compute_reconstruction_error,
    compute_decoding_accuracy,
    compute_embedding_alignment_metrics,
    circular_structure_preservation
)
from driada.intense import compute_cell_feat_significance
from driada.utils.visual import DEFAULT_DPI, visualize_circular_manifold


def loo_dr_analysis(
    exp: Experiment,
    method: str = 'umap',
    data_type: str = 'calcium',
    use_scaled: bool = True,
    method_params: Optional[Dict[str, Any]] = None,
    metrics_to_compute: Optional[list] = None,
    k_neighbors: int = 15,
    cbunch: Optional[list] = None,
    downsampling: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform leave-one-out dimension reduction analysis on neural data.

    This function systematically removes one neuron at a time and measures
    the change in manifold reconstruction quality. For experiments with ground
    truth (e.g., synthetic manifolds), it uses reconstruction metrics that
    compare embeddings against known behavioral variables.

    Parameters
    ----------
    exp : Experiment
        DRIADA Experiment object containing neural data
    method : str, default='umap'
        Dimension reduction method. Options:
        Linear: 'pca'
        Manifold: 'le', 'auto_le', 'dmaps', 'auto_dmaps', 'isomap', 'lle',
                 'hlle', 'umap', 'mvu'
        Distance: 'mds'
        Neural: 'ae', 'vae', 'flexible_ae'
        Probabilistic: 'tsne'
    data_type : str, default='calcium'
        Type of data to use: 'calcium' or 'spikes'
    use_scaled : bool, default=True
        If True, use scaled data (recommended for equal neuron contributions)
    method_params : dict, optional
        Parameters specific to the DR method. If None, uses defaults.
        Common parameters:
        - dim: target dimensions (default 2)
        - n_neighbors: for graph-based methods (default 15)
        - min_dist: for UMAP (default 0.1)
        - perplexity: for t-SNE (default 30)
    metrics_to_compute : list, optional
        List of metrics to compute. If None, computes all available.
        Options: 'knn_preservation', 'trustworthiness', 'continuity',
                'geodesic_corr', 'stress', 'reconstruction_error',
                'manifold_score'
    k_neighbors : int, default=15
        Number of neighbors for metric computations
    cbunch : list, optional
        List of neuron indices to check. If None, checks all neurons.
        Example: [0, 5, 10] will only test neurons 0, 5, and 10.
    downsampling : int, optional
        Downsampling factor for temporal data. If provided, uses every
        downsampling-th timepoint. Useful for speeding up computation.
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with metrics as columns and neurons as rows.
        First row contains metrics for full neuron set (labeled 'all').
        Subsequent rows contain metrics with each neuron removed.
        Index is neuron identifier (0, 1, 2, ... for removed neurons).

    Examples
    --------
    >>> import driada
    >>> # Generate synthetic experiment
    >>> exp = driada.experiment.generate_circular_manifold_exp(
    ...     n_neurons=50, duration=300, seed=42
    ... )
    >>> # Run LOO analysis with UMAP
    >>> results = loo_dr_analysis(exp, method='umap', dim=2)
    >>> # Get neuron importance scores
    >>> baseline = results.loc['all']
    >>> degradation = (baseline - results.iloc[1:]) / baseline
    >>> importance = degradation.mean(axis=1)
    >>> print(f"Top 5 important neurons: {importance.nlargest(5).index.tolist()}")

    >>> # Try different DR methods
    >>> results_pca = loo_dr_analysis(exp, method='pca', dim=3)
    >>> results_tsne = loo_dr_analysis(exp, method='tsne',
    ...                                method_params={'perplexity': 20})
    """

    # Get the appropriate MultiTimeSeries object (which inherits from MVData)
    if data_type == 'calcium':
        data_obj = exp.calcium
        if use_scaled:
            neural_data = data_obj.scdata  # (n_neurons, n_timepoints)
        else:
            neural_data = data_obj.data
    elif data_type == 'spikes':
        if exp.spikes is None:
            raise ValueError("No spike data available in experiment")
        data_obj = exp.spikes
        neural_data = data_obj.data
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Use 'calcium' or 'spikes'")

    n_neurons, n_timepoints = neural_data.shape

    # Detect ground truth and manifold type
    ground_truth = None
    manifold_type = None

    if 'head_direction' in exp.dynamic_features:
        ground_truth = exp.dynamic_features['head_direction'].data
        manifold_type = 'circular'
        if verbose:
            print(f"Detected circular manifold with head_direction ground truth")
    elif 'position_2d' in exp.dynamic_features:
        ground_truth = exp.dynamic_features['position_2d'].data.T  # (n_timepoints, 2)
        manifold_type = 'spatial'
        if verbose:
            print(f"Detected 2D spatial manifold with position ground truth")
    elif 'x' in exp.dynamic_features and 'y' in exp.dynamic_features:
        ground_truth = np.column_stack([
            exp.dynamic_features['x'].data,
            exp.dynamic_features['y'].data
        ])
        manifold_type = 'spatial'
        if verbose:
            print(f"Detected 2D spatial manifold with x,y ground truth")

    # Apply downsampling to ground truth if needed
    if ground_truth is not None and downsampling is not None:
        if len(ground_truth.shape) == 1:
            ground_truth = ground_truth[::downsampling]
        else:
            ground_truth = ground_truth[::downsampling, :]
        if verbose:
            print(f"Downsampled ground truth by factor {downsampling}: new shape {ground_truth.shape}")

    if verbose:
        print(f"Starting LOO analysis for {n_neurons} neurons using {method}")
        print(f"Data shape: {neural_data.shape}")
        if ground_truth is not None:
            print(f"Ground truth shape: {ground_truth.shape}, type: {manifold_type}")

    # Default parameters for each method if not provided
    if method_params is None:
        method_params = _get_default_params(method)

    # Default metrics if not specified
    if metrics_to_compute is None:
        if ground_truth is not None:
            # Use ground truth reconstruction metrics
            metrics_to_compute = [
                'reconstruction_error',
                'decoding_accuracy',
                'alignment_corr'
            ]
        else:
            # Use unsupervised manifold quality metrics
            metrics_to_compute = [
                'knn_preservation', 'trustworthiness', 'continuity',
                'geodesic_corr', 'stress'
            ]

    # Initialize results storage
    results = []

    # Compute baseline with all neurons
    if verbose:
        print("Computing baseline embedding with all neurons...")

    baseline_metrics = _compute_embedding_metrics(
        data_obj,
        None,  # No mask for baseline
        method,
        method_params,
        metrics_to_compute,
        k_neighbors,
        ground_truth,
        manifold_type,
        label='all',
        downsampling=downsampling,
        verbose=verbose
    )
    results.append(baseline_metrics)

    # Determine which neurons to check
    if cbunch is None:
        neurons_to_check = list(range(n_neurons))
    else:
        # Validate cbunch indices
        neurons_to_check = []
        for idx in cbunch:
            if 0 <= idx < n_neurons:
                neurons_to_check.append(idx)
            else:
                if verbose:
                    print(f"Warning: Neuron index {idx} out of range (0-{n_neurons-1}), skipping")

        if not neurons_to_check:
            raise ValueError("No valid neuron indices in cbunch")

    # Leave-one-out analysis
    if verbose:
        iterator = tqdm(neurons_to_check, desc=f"LOO {method}")
        print(f"Checking {len(neurons_to_check)} neurons: {neurons_to_check[:10]}{'...' if len(neurons_to_check) > 10 else ''}")
    else:
        iterator = neurons_to_check

    for neuron_idx in iterator:
        # Create mask for removing neuron
        neuron_mask = np.ones(n_neurons, dtype=bool)
        neuron_mask[neuron_idx] = False

        # Compute metrics for reduced data
        try:
            metrics = _compute_embedding_metrics(
                data_obj,
                neuron_mask,
                method,
                method_params,
                metrics_to_compute,
                k_neighbors,
                ground_truth,
                manifold_type,
                label=neuron_idx,
                downsampling=downsampling,
                verbose=False  # Suppress verbose for individual neurons
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Failed for neuron {neuron_idx}: {e}")
            # Fill with NaN for failed embeddings
            metrics = {'neuron': neuron_idx}
            for metric in metrics_to_compute:
                metrics[metric] = np.nan

        results.append(metrics)

    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('neuron', inplace=True)

    if verbose:
        print(f"Completed LOO analysis. Shape: {results_df.shape}")

        # Report summary statistics
        baseline = results_df.loc['all']
        degradations = []
        # Only use numeric columns for degradation calculation
        numeric_cols = results_df.select_dtypes(include=['float64', 'int64']).columns
        metric_cols = [col for col in numeric_cols if col not in ['node_loss_rate', 'n_lost_nodes']]

        for idx in results_df.index[1:]:
            if not results_df.loc[idx, metric_cols].isna().all():
                deg = (baseline[metric_cols] - results_df.loc[idx, metric_cols]) / (baseline[metric_cols] + 1e-10)
                degradations.append(deg.mean())

        if degradations:
            top_5_idx = np.argsort(degradations)[-5:]
            print(f"Top 5 neurons causing most degradation: {top_5_idx.tolist()}")

        # Report node loss statistics if present
        if 'node_loss_rate' in results_df.columns:
            node_loss_neurons = results_df[results_df['node_loss_rate'] > 0]
            if len(node_loss_neurons) > 1:  # Exclude baseline 'all'
                print(f"\nNode loss statistics:")
                print(f"  Neurons with node loss: {len(node_loss_neurons)-1}/{len(results_df)-1}")
                avg_loss = node_loss_neurons['node_loss_rate'].mean()
                max_loss = node_loss_neurons['node_loss_rate'].max()
                print(f"  Average node loss rate: {avg_loss:.1%}")
                print(f"  Maximum node loss rate: {max_loss:.1%}")

                # Report graph disconnections
                if 'status' in results_df.columns:
                    disconnected = results_df[results_df['status'] == 'graph_disconnected']
                    if len(disconnected) > 0:
                        print(f"  Graph disconnections: {len(disconnected)} neurons")

    return results_df


def _get_default_params(method: str) -> Dict[str, Any]:
    """Get default parameters for each DR method with proper nn parameter."""
    defaults = {
        'pca': {'dim': 2},
        'le': {'dim': 2, 'nn': 15, 'max_deleted_nodes': 0.3},
        'auto_le': {'dim': 2, 'nn': 15, 'max_deleted_nodes': 0.3},
        'dmaps': {'dim': 2, 'dm_alpha': 0.5, 'dm_t': 1, 'nn': 15, 'max_deleted_nodes': 0.3},
        'auto_dmaps': {'dim': 2, 'nn': 15, 'max_deleted_nodes': 0.3},
        'isomap': {'dim': 2, 'nn': 30, 'max_deleted_nodes': 0.3},
        'lle': {'dim': 2, 'nn': 10, 'max_deleted_nodes': 0.3},
        'hlle': {'dim': 2, 'nn': 10, 'max_deleted_nodes': 0.3},
        'umap': {'dim': 2, 'nn': 30, 'min_dist': 0.1, 'max_deleted_nodes': 0.3},
        'mvu': {'dim': 2, 'nn': 10, 'max_deleted_nodes': 0.3},
        'mds': {'dim': 2},
        'ae': {'dim': 2, 'epochs': 100, 'lr': 0.001},
        'vae': {'dim': 2, 'epochs': 100, 'beta': 1.0},
        'flexible_ae': {'dim': 2, 'epochs': 100},
        'tsne': {'dim': 2, 'perplexity': 30, 'n_iter': 1000}
    }
    return defaults.get(method, {'dim': 2})


def _compute_embedding_metrics(
    data_obj,
    neuron_mask: Optional[np.ndarray],
    method: str,
    params: Dict[str, Any],
    metrics_list: list,
    k_neighbors: int,
    ground_truth: Optional[np.ndarray],
    manifold_type: Optional[str],
    label: Union[int, str],
    downsampling: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute embedding and calculate metrics.

    Parameters
    ----------
    data_obj : MultiTimeSeries
        MultiTimeSeries object (calcium or spikes) that inherits from MVData
    neuron_mask : np.ndarray or None
        Boolean mask for neurons to keep. None means use all neurons.
    method : str
        DR method name
    params : dict
        Method parameters
    metrics_list : list
        List of metrics to compute
    k_neighbors : int
        Number of neighbors for metrics
    ground_truth : np.ndarray or None
        Ground truth behavioral variables if available
    manifold_type : str or None
        Type of manifold ('circular', '2d_spatial', etc.)
    label : int or str
        Label for this computation (neuron index or 'all')
    verbose : bool
        Print progress

    Returns
    -------
    metrics : dict
        Computed metrics with 'neuron' key
    """
    # Use the MultiTimeSeries object directly (it inherits from MVData)
    # Apply neuron mask if provided
    if neuron_mask is not None:
        # Create a subset using the mask
        data = data_obj.data[neuron_mask, :]
        from driada.dim_reduction import MVData
        mvdata = MVData(data, downsampling=downsampling)
    else:
        # Use the full MultiTimeSeries object directly
        if downsampling is not None:
            # Need to create a new MVData with downsampling
            from driada.dim_reduction import MVData
            mvdata = MVData(data_obj.data, downsampling=downsampling)
        else:
            mvdata = data_obj

    # Get embedding with proper error handling for graph disconnection
    try:
        embedding = mvdata.get_embedding(method=method, **params)
        coords = embedding.coords.T  # (n_timepoints, n_dims)
    except Exception as e:
        # Check if it's a node loss error
        if "nodes discarded" in str(e) or "disconnected" in str(e).lower():
            if verbose:
                print(f"  Warning: Graph disconnected for neuron {label}: {e}")
            # Return NaN metrics with failure note
            metrics = {'neuron': label, 'status': 'graph_disconnected', 'node_loss_rate': 1.0}
            for metric in metrics_list:
                metrics[metric] = np.nan
            return metrics
        else:
            raise RuntimeError(f"Embedding failed for {method}: {e}")

    # Check for node loss in graph-based methods
    node_loss_rate = 0.0
    lost_nodes = []

    if hasattr(embedding, 'graph') and hasattr(embedding.graph, 'lost_nodes'):
        lost_nodes = list(embedding.graph.lost_nodes)
        total_samples = mvdata.data.shape[1]
        node_loss_rate = len(lost_nodes) / total_samples

        if node_loss_rate > 0:
            # Get surviving indices for ground truth alignment
            all_indices = set(range(total_samples))
            surviving_indices = sorted(all_indices - set(lost_nodes))

            if verbose and node_loss_rate > 0.1:  # Warn if > 10% loss
                print(f"  Warning: {node_loss_rate:.1%} nodes lost for neuron {label}")

            # Align ground truth to surviving nodes if needed
            if ground_truth is not None and node_loss_rate > 0:
                if len(ground_truth.shape) == 1:
                    ground_truth = ground_truth[surviving_indices]
                else:
                    ground_truth = ground_truth[surviving_indices, :]

    # Get data for metrics (they expect n_samples x n_features)
    if neuron_mask is not None:
        data_T = data.T
    else:
        data_T = data_obj.data.T

    # Compute requested metrics
    metrics = {'neuron': label, 'node_loss_rate': node_loss_rate, 'status': 'success'}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Ground truth reconstruction metrics
        if ground_truth is not None and manifold_type is not None:
            if 'reconstruction_error' in metrics_list:
                try:
                    result = compute_reconstruction_error(
                        coords, ground_truth, manifold_type=manifold_type
                    )
                    # Extract scalar value if result is a dict
                    if isinstance(result, dict):
                        # Use mean error or main metric
                        if 'mean_error' in result:
                            metrics['reconstruction_error'] = result['mean_error']
                        elif 'error' in result:
                            metrics['reconstruction_error'] = result['error']
                        else:
                            # Use first numeric value found
                            for key, val in result.items():
                                if isinstance(val, (int, float)):
                                    metrics['reconstruction_error'] = val
                                    break
                            else:
                                metrics['reconstruction_error'] = np.nan
                    else:
                        metrics['reconstruction_error'] = result
                except Exception as e:
                    metrics['reconstruction_error'] = np.nan

            if 'decoding_accuracy' in metrics_list:
                try:
                    result = compute_decoding_accuracy(
                        coords, ground_truth, manifold_type=manifold_type
                    )
                    metrics['decoding_accuracy'] = result['test_r2']
                except Exception as e:
                    metrics['decoding_accuracy'] = np.nan

            if 'alignment_corr' in metrics_list:
                try:
                    alignment_results = compute_embedding_alignment_metrics(
                        coords, ground_truth, manifold_type=manifold_type
                    )
                    # Extract correlation from results dict
                    if isinstance(alignment_results, dict) and 'correlation' in alignment_results:
                        metrics['alignment_corr'] = alignment_results['correlation']
                    else:
                        metrics['alignment_corr'] = np.nan
                except Exception as e:
                    metrics['alignment_corr'] = np.nan

        # Standard unsupervised metrics
        if 'knn_preservation' in metrics_list:
            try:
                metrics['knn_preservation'] = knn_preservation_rate(
                    data_T, coords, k=k_neighbors
                )
            except:
                metrics['knn_preservation'] = np.nan

        if 'trustworthiness' in metrics_list:
            try:
                metrics['trustworthiness'] = trustworthiness(
                    data_T, coords, k=k_neighbors
                )
            except:
                metrics['trustworthiness'] = np.nan

        if 'continuity' in metrics_list:
            try:
                metrics['continuity'] = continuity(
                    data_T, coords, k=k_neighbors
                )
            except:
                metrics['continuity'] = np.nan

        if 'geodesic_corr' in metrics_list:
            try:
                metrics['geodesic_corr'] = geodesic_distance_correlation(
                    data_T, coords, k_neighbors=k_neighbors
                )
            except:
                metrics['geodesic_corr'] = np.nan

        if 'stress' in metrics_list:
            try:
                metrics['stress'] = stress(data_T, coords)
            except:
                metrics['stress'] = np.nan


    return metrics


def compare_with_intense(
    loo_metrics_df: pd.DataFrame,
    intense_stats: Dict,
    feature_name: str,
    metrics_to_add: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add INTENSE metrics to LOO analysis results for a specific ground truth feature.

    This function extracts INTENSE selectivity metrics for a given feature
    and adds them as columns to the LOO metrics DataFrame.

    Parameters
    ----------
    loo_metrics_df : pd.DataFrame
        Output from loo_dr_analysis. Rows are neurons (with 'all' as baseline),
        columns are DR metrics.
    intense_stats : dict
        Raw stats dictionary from INTENSE: stats[neuron_id][feature_name][metric]
    feature_name : str
        Name of the ground truth feature to extract INTENSE metrics for
        (e.g., 'head_direction', 'x', 'y', 'position_2d')
    metrics_to_add : list of str, optional
        Which INTENSE metrics to add. Default: ['me', 'pval']
        Available metrics: 'me' (MI in bits), 'pval' (stage 2 p-value)
        Note: 'me' is the mutual information estimate and is renamed to 'intense_mi'
    verbose : bool, default=True
        Print extraction information

    Returns
    -------
    combined_df : pd.DataFrame
        Copy of loo_metrics_df with additional columns for INTENSE metrics.
        Column names: 'intense_{metric}' (e.g., 'intense_me', 'intense_pval')

    Examples
    --------
    >>> # After running LOO analysis and INTENSE
    >>> loo_results = loo_dr_analysis(exp, method='umap')
    >>> stats, sig, info, intense_res = compute_cell_feat_significance(exp)
    >>>
    >>> # Add INTENSE metrics for head_direction feature
    >>> combined = compare_with_intense(loo_results, stats, 'head_direction')
    >>>
    >>> # Now you can analyze correlations
    >>> from scipy.stats import spearmanr
    >>> corr, pval = spearmanr(combined['reconstruction_error'], combined['intense_mi'])
    >>> print(f"Correlation: {corr:.3f} (p={pval:.3e})")
    """
    import numpy as np
    import pandas as pd

    # Default metrics to extract
    if metrics_to_add is None:
        metrics_to_add = ['me', 'pval']

    # Make a copy of the LOO DataFrame
    combined_df = loo_metrics_df.copy()

    # Extract INTENSE metrics for each neuron
    n_neurons = len(loo_metrics_df) - 1  # Exclude 'all' row

    if verbose:
        print(f"Extracting INTENSE metrics for feature '{feature_name}'")
        print(f"  Metrics to add: {metrics_to_add}")
        print(f"  Number of neurons: {n_neurons}")

    # Initialize columns for INTENSE metrics
    for metric in metrics_to_add:
        combined_df[f'intense_{metric}'] = np.nan

    # Extract metrics for each neuron
    extracted_count = 0
    missing_neurons = []
    missing_features = []

    for neuron_id in range(n_neurons):
        # Check if neuron exists in INTENSE results
        if neuron_id not in intense_stats:
            missing_neurons.append(neuron_id)
            continue

        # Check if feature exists for this neuron
        if feature_name not in intense_stats[neuron_id]:
            missing_features.append(neuron_id)
            continue

        # Extract metrics
        neuron_stats = intense_stats[neuron_id][feature_name]
        for metric in metrics_to_add:
            if metric in neuron_stats:
                combined_df.loc[neuron_id, f'intense_{metric}'] = neuron_stats[metric]
                extracted_count += 1

    # No INTENSE metrics for the 'all' baseline row
    # Leave as NaN

    if verbose:
        print(f"  Successfully extracted metrics for {len(set(range(n_neurons)) - set(missing_neurons) - set(missing_features))}/{n_neurons} neurons")
        if missing_neurons:
            print(f"  Neurons not in INTENSE results: {len(missing_neurons)}")
        if missing_features:
            print(f"  Neurons without '{feature_name}' feature: {len(missing_features)}")

        # Show summary statistics
        print("\nINTENSE metrics summary:")
        for metric in metrics_to_add:
            col = f'intense_{metric}'
            valid_values = combined_df[col].dropna()
            if len(valid_values) > 0:
                print(f"  {col}:")
                print(f"    Range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
                print(f"    Mean: {valid_values.mean():.4f}")
                print(f"    Non-NaN: {len(valid_values)}")

    return combined_df


def analyze_neuron_importance(results_df: pd.DataFrame) -> pd.Series:
    """
    Calculate neuron importance scores from LOO results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from loo_dr_analysis

    Returns
    -------
    importance : pd.Series
        Importance score for each neuron
    """
    # Get numeric columns only (exclude 'status' and other non-numeric)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude metadata columns
    metric_cols = [c for c in numeric_cols if c not in ['node_loss_rate', 'n_lost_nodes']]

    # Get baseline (all neurons)
    baseline = results_df.loc['all', metric_cols]

    # Calculate degradation for each neuron removal
    importance_scores = []

    for idx in results_df.index[1:]:
        row_metrics = results_df.loc[idx, metric_cols]
        if row_metrics.isna().all():
            importance_scores.append(np.nan)
        else:
            # Calculate relative degradation
            degradation = (baseline - row_metrics) / (baseline + 1e-10)

            # Equal weights for ground-truth metrics
            weights = {
                'alignment_corr': 0.33,
                'decoding_accuracy': 0.33,
                'reconstruction_error': 0.33
            }

            # Compute weighted importance
            score = 0.0
            for metric, weight in weights.items():
                if metric in degradation.index and not np.isnan(degradation[metric]):
                    # For error metrics, higher error = worse, so positive degradation is good
                    if metric == 'reconstruction_error':
                        score += weight * (-degradation[metric])  # Flip sign: higher error when removed = important
                    else:
                        # For quality metrics, lower value when removed = important
                        score += weight * degradation[metric]

            importance_scores.append(score)

    return pd.Series(importance_scores, index=results_df.index[1:], name='importance')


def main():
    """
    Example demonstrating LOO-DR analysis with INTENSE validation.

    Key insight: Neurons important for manifold reconstruction (identified by LOO-DR)
    should be the same neurons that encode the behavioral feature (identified by INTENSE).
    """
    print("=" * 70)
    print("LOO Dimensionality Reduction Analysis with INTENSE Validation")
    print("=" * 70)

    # Generate mixed population: HD-coding + non-coding neurons
    # This tests whether LOO importance correlates with INTENSE selectivity
    print("\n[1/4] Generating mixed population experiment...")
    population = [
        {"name": "hd_broad", "count": 15, "features": ["head_direction"],
         "tuning_params": {"kappa": 2.0}},  # Broad tuning
        {"name": "hd_medium", "count": 15, "features": ["head_direction"],
         "tuning_params": {"kappa": 5.0}},  # Medium tuning
        {"name": "hd_sharp", "count": 15, "features": ["head_direction"],
         "tuning_params": {"kappa": 10.0}},  # Sharp tuning
        {"name": "nonselective", "count": 15, "features": []},  # No selectivity
    ]
    exp = generate_tuned_selectivity_exp(
        population=population,
        duration=600,  # 10 minutes for better statistics
        seed=42
    )
    print(f"  Created: {exp.n_neurons} neurons, {exp.calcium.data.shape[1]} timepoints")
    print(f"  Population: 45 HD (15 broad/15 medium/15 sharp) + 15 non-selective")

    # Run LOO analysis
    dr_method = 'isomap'
    dr_params = {'dim': 2, 'nn': 20, 'max_deleted_nodes': 0.3}
    print(f"\n[2/4] Running LOO-DR analysis with {dr_method}...")
    results = loo_dr_analysis(
        exp,
        method=dr_method,
        method_params=dr_params,
        downsampling=5,
        verbose=True
    )

    # Calculate importance scores from LOO results
    importance = analyze_neuron_importance(results)

    print(f"\nLOO-DR Results:")
    baseline = results.loc['all']
    print(f"  Baseline metrics:")
    print(f"    alignment_corr:      {baseline['alignment_corr']:.4f}")
    print(f"    decoding_accuracy:   {baseline['decoding_accuracy']:.4f}")
    print(f"    reconstruction_error: {baseline['reconstruction_error']:.4f}")

    if not importance.isna().all():
        print(f"\n  Top 5 most important neurons (per-metric degradation):")
        for neuron, score in importance.nlargest(5).items():
            row = results.loc[neuron]
            # Calculate per-metric change (baseline - loo)
            align_delta = baseline['alignment_corr'] - row['alignment_corr']
            decode_delta = baseline['decoding_accuracy'] - row['decoding_accuracy']
            error_delta = row['reconstruction_error'] - baseline['reconstruction_error']  # Higher error = worse
            print(f"    Neuron {neuron}: importance={score:.4f}")
            print(f"      align: {align_delta:+.4f}, decode: {decode_delta:+.4f}, error: {error_delta:+.4f}")

    # Verify the manifold is actually circular
    # Use the same downsampled MVData as the LOO analysis to get identical embedding
    print("\n  Verifying circular structure...")
    from driada.dim_reduction import MVData
    ds = 5
    mvdata_vis = MVData(exp.calcium.data, downsampling=ds)
    baseline_emb = mvdata_vis.get_embedding(method=dr_method, **dr_params)
    baseline_coords = baseline_emb.coords.T  # (n_samples, 2)
    ground_truth_full = exp.dynamic_features['head_direction'].data[::ds]

    # Handle lost nodes if any
    if hasattr(baseline_emb, 'graph') and hasattr(baseline_emb.graph, 'lost_nodes'):
        lost = set(baseline_emb.graph.lost_nodes)
        surviving = [i for i in range(len(ground_truth_full)) if i not in lost]
        ground_truth_aligned = ground_truth_full[surviving]
        print(f"    Note: {len(lost)} nodes lost in graph construction")
    else:
        ground_truth_aligned = ground_truth_full

    circular_metrics = circular_structure_preservation(
        baseline_coords,
        true_angles=ground_truth_aligned,
        k_neighbors=3
    )
    print(f"  Circular structure verification:")
    print(f"    Distance CV: {circular_metrics['distance_cv']:.3f} (lower = more circular)")
    print(f"    Consecutive preservation: {circular_metrics['consecutive_preservation']:.1%}")
    print(f"    Circular correlation: {circular_metrics['circular_correlation']:.3f}")

    # Visualize embedding
    print("\n  Visualizing baseline embedding...")
    fig = visualize_circular_manifold(
        [baseline_coords],
        ground_truth_aligned,
        [dr_method.upper()],
        save_path="examples/loo_dr_analysis/loo_dr_embedding.png"
    )
    plt.close(fig)
    print("    Saved: examples/loo_dr_analysis/loo_dr_embedding.png")

    # Run INTENSE to identify selective neurons
    # Use head_direction_2d (cos/sin representation) instead of raw head_direction
    # to preserve circular topology - raw angles break continuity
    print("\n[3/4] Running INTENSE analysis...")
    stats, significant, info, intense_res = compute_cell_feat_significance(
        exp,
        feat_bunch=['head_direction_2d'],  # Preserves circular topology via cos/sin
        n_shuffles_stage1=100,
        n_shuffles_stage2=10000,
        find_optimal_delays=True,
        ds=5,
        verbose=True
    )

    # Combine LOO metrics with INTENSE selectivity
    print("\n[4/4] Comparing LOO importance with INTENSE selectivity...")
    combined = compare_with_intense(
        results, stats, 'head_direction_2d',
        metrics_to_add=['me', 'pval'],
        verbose=True
    )

    # Rename 'intense_me' to 'intense_mi' for clarity
    if 'intense_me' in combined.columns:
        combined.rename(columns={'intense_me': 'intense_mi'}, inplace=True)

    # Add LOO importance to combined dataframe
    combined['loo_importance'] = np.nan
    for idx in importance.index:
        combined.loc[idx, 'loo_importance'] = importance[idx]

    # Add per-metric LOO values to combined dataframe
    baseline = results.loc['all']
    for metric in ['alignment_corr', 'decoding_accuracy', 'reconstruction_error']:
        combined[f'loo_{metric}'] = np.nan
        for idx in results.index:
            if idx != 'all':
                if metric == 'reconstruction_error':
                    # Higher error when neuron removed = more important
                    delta = results.loc[idx, metric] - baseline[metric]
                else:
                    # Lower score when neuron removed = more important
                    delta = baseline[metric] - results.loc[idx, metric]
                combined.loc[idx, f'loo_{metric}'] = delta

    # Compute correlation between LOO importance and INTENSE selectivity (MI)
    # Exclude 'all' row and rows with NaN
    neuron_data = combined.iloc[1:].dropna(subset=['loo_importance', 'intense_mi'])

    if len(neuron_data) >= 5:
        corr, pval = spearmanr(neuron_data['loo_importance'], neuron_data['intense_mi'])
        print(f"\n" + "=" * 70)
        print("KEY RESULT: Correlation between LOO importance and INTENSE selectivity")
        print("=" * 70)
        print(f"  Spearman correlation: r = {corr:.3f}")
        print(f"  p-value: {pval:.3e}")
        if pval < 0.05:
            if corr > 0:
                print("  -> Significant POSITIVE correlation: neurons important for")
                print("     manifold reconstruction ARE the ones selective for head_direction")
            else:
                print("  -> Significant NEGATIVE correlation (unexpected)")
        else:
            print("  -> No significant correlation found")

        # Report per-metric correlations with INTENSE MI
        print(f"\nPer-metric correlations with INTENSE MI:")
        for metric in ['loo_importance', 'loo_alignment_corr', 'loo_decoding_accuracy', 'loo_reconstruction_error']:
            valid = neuron_data.dropna(subset=[metric, 'intense_mi'])
            if len(valid) >= 5:
                metric_corr, metric_pval = spearmanr(valid[metric], valid['intense_mi'])
                sig = "*" if metric_pval < 0.05 else ""
                print(f"  {metric}: r={metric_corr:.3f}, p={metric_pval:.3f} {sig}")

        # Show comparison of top neurons
        print(f"\nTop 5 neurons by LOO importance vs their INTENSE MI:")
        top_loo = neuron_data.nlargest(5, 'loo_importance')
        for idx, row in top_loo.iterrows():
            sig_marker = "*" if row['intense_pval'] < 0.05 else " "
            print(f"  Neuron {idx}: LOO={row['loo_importance']:.4f}, MI={row['intense_mi']:.3f} bits {sig_marker}")

        print(f"\nTop 5 neurons by INTENSE MI vs their LOO importance:")
        top_intense = neuron_data.nlargest(5, 'intense_mi')
        for idx, row in top_intense.iterrows():
            print(f"  Neuron {idx}: MI={row['intense_mi']:.3f} bits, LOO={row['loo_importance']:.4f}")
    else:
        print(f"\nInsufficient data for correlation ({len(neuron_data)} neurons with complete data)")

    return {
        'results': results,
        'combined': combined,
        'importance': importance,
        'intense_stats': stats
    }


if __name__ == "__main__":
    results = main()