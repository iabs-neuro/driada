#!/usr/bin/env python
"""
Simplified Leave-One-Out analysis focused on spatial manifolds and optimal alignment.
Removes redundant code and focuses on essential metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from tqdm import tqdm

from driada.experiment import Experiment
from driada.dim_reduction import MVData
from driada.dim_reduction.manifold_metrics import (
    compute_reconstruction_error,
    compute_decoding_accuracy,
    compute_embedding_alignment_metrics
)


def extract_metric_value(result: Any, keys: list) -> float:
    """
    Extract scalar value from metric result (dict or scalar).

    Parameters
    ----------
    result : Any
        Result from metric function (dict or scalar)
    keys : list
        Priority list of keys to try for dict results

    Returns
    -------
    float
        Extracted metric value or np.nan
    """
    if result is None:
        return np.nan

    if isinstance(result, (int, float)):
        return float(result)

    if isinstance(result, dict):
        # Try keys in priority order
        for key in keys:
            if key in result:
                value = result[key]
                if isinstance(value, (int, float)):
                    return float(value)

        # Fallback: return first numeric value found
        for value in result.values():
            if isinstance(value, (int, float)):
                return float(value)

    return np.nan


def compute_alignment_metrics(
    embedding: np.ndarray,
    ground_truth: np.ndarray,
    manifold_type: str = 'spatial'
) -> Dict[str, float]:
    """
    Compute only alignment-based metrics for spatial manifolds.

    Parameters
    ----------
    embedding : np.ndarray
        Shape (n_samples, n_dims) - embedding coordinates
    ground_truth : np.ndarray
        Shape (n_samples, n_dims) - true positions
    manifold_type : str
        'spatial' or 'circular'

    Returns
    -------
    dict
        Alignment metrics (reconstruction_error, alignment_corr, decoding_r2)
    """
    metrics = {}

    # Reconstruction error with optimal alignment
    try:
        result = compute_reconstruction_error(
            embedding, ground_truth,
            manifold_type=manifold_type,
            allow_rotation=True,
            allow_reflection=True,
            allow_scaling=True
        )
        metrics['reconstruction_error'] = extract_metric_value(
            result, ['error', 'mean_error']
        )
        metrics['alignment_corr'] = extract_metric_value(
            result, ['correlation', 'corr']
        )
    except Exception:
        metrics['reconstruction_error'] = np.nan
        metrics['alignment_corr'] = np.nan

    # Decoding accuracy
    try:
        result = compute_decoding_accuracy(
            embedding, ground_truth,
            manifold_type=manifold_type,
            train_fraction=0.8
        )
        # Use the proper R² score returned by the function
        metrics['decoding_r2'] = extract_metric_value(result, ['test_r2', 'r2'])
        # Also store the test error for reference
        metrics['test_error'] = extract_metric_value(result, ['test_error', 'error'])
    except Exception:
        metrics['decoding_r2'] = np.nan
        metrics['test_error'] = np.nan

    return metrics


def get_ground_truth(exp: Experiment) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Extract ground truth behavioral variables from experiment.

    Parameters
    ----------
    exp : Experiment
        DRIADA experiment object

    Returns
    -------
    ground_truth : np.ndarray or None
        Ground truth data (n_samples, n_dims)
    manifold_type : str or None
        Type of manifold ('spatial' or 'circular')
    """
    if 'head_direction' in exp.dynamic_features:
        # Circular manifold
        angles = exp.dynamic_features['head_direction'].data
        return angles.reshape(-1, 1), 'circular'

    elif 'position_2d' in exp.dynamic_features:
        # 2D spatial manifold
        positions = exp.dynamic_features['position_2d'].data.T  # (n_samples, 2)
        return positions, 'spatial'

    elif 'x' in exp.dynamic_features and 'y' in exp.dynamic_features:
        # 2D spatial manifold (separate x,y)
        x = exp.dynamic_features['x'].data
        y = exp.dynamic_features['y'].data
        positions = np.column_stack([x, y])
        return positions, 'spatial'

    return None, None


def loo_analysis_simplified(
    exp: Experiment,
    method: str = 'umap',
    data_type: str = 'calcium',
    use_scaled: bool = True,
    method_params: Optional[Dict[str, Any]] = None,
    neurons_to_test: Optional[list] = None,
    downsampling: Optional[int] = None,
    verbose: bool = True,
    shuffled: bool = False,
    shuffle_seed: int = 42
) -> pd.DataFrame:
    """
    Simplified LOO analysis for spatial manifolds.

    Parameters
    ----------
    exp : Experiment
        DRIADA experiment with neural data
    method : str
        DR method ('umap', 'pca', 'isomap', etc.)
    data_type : str, default='calcium'
        Type of data: 'calcium' or 'spikes'
    use_scaled : bool, default=True
        If True, use scaled data (recommended for equal neuron contributions)
    method_params : dict, optional
        Parameters for DR method
    neurons_to_test : list, optional
        Specific neuron indices to test (None = all)
    downsampling : int, optional
        Temporal downsampling factor
    verbose : bool
        Print progress
    shuffled : bool, default=False
        If True, use temporally shuffled neural data to test null hypothesis
    shuffle_seed : int, default=42
        Random seed for shuffling reproducibility

    Returns
    -------
    pd.DataFrame
        Metrics for each neuron removal (first row = baseline)
    """
    # Get neural data based on data_type and scaling preference
    if data_type == 'calcium':
        if shuffled:
            # Generate shuffled calcium data
            if verbose:
                print("Generating shuffled calcium data...")
            neural_data = exp.get_multicell_shuffled_calcium(
                n_shuffles=1,
                method='roll_based',
                return_array=True,
                seed=shuffle_seed
            )
            # Shuffled data is not scaled, apply scaling if needed
            if use_scaled:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                neural_data = scaler.fit_transform(neural_data.T).T
        else:
            if use_scaled:
                neural_data = exp.calcium.scdata  # (n_neurons, n_samples)
            else:
                neural_data = exp.calcium.data
    elif data_type == 'spikes':
        if exp.spikes is None:
            raise ValueError("No spike data available in experiment")
        if shuffled:
            if verbose:
                print("Generating shuffled spike data...")
            neural_data = exp.get_multicell_shuffled_spikes(
                n_shuffles=1,
                method='roll_based',
                return_array=True,
                seed=shuffle_seed
            )
        else:
            neural_data = exp.spikes.data  # Spikes are typically not scaled
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Use 'calcium' or 'spikes'")

    n_neurons, n_samples = neural_data.shape

    # Get ground truth
    ground_truth, manifold_type = get_ground_truth(exp)

    if ground_truth is None:
        raise ValueError("No ground truth found in experiment")

    # Apply downsampling
    if downsampling:
        neural_data = neural_data[:, ::downsampling]
        if len(ground_truth.shape) == 1:
            ground_truth = ground_truth[::downsampling]
        else:
            ground_truth = ground_truth[::downsampling, :]
        n_samples = neural_data.shape[1]  # Update n_samples after downsampling

    if verbose:
        print(f"LOO Analysis: {n_neurons} neurons, {method} method")
        print(f"Data type: {data_type} ({'scaled' if use_scaled else 'unscaled'})")
        print(f"Manifold type: {manifold_type}")
        print(f"Data shape: {neural_data.shape}")

    # Default parameters
    if method_params is None:
        method_params = {'dim': 2}

    # Determine neurons to test
    if neurons_to_test is None:
        neurons_to_test = list(range(n_neurons))

    results = []

    # Baseline with all neurons
    if verbose:
        print("Computing baseline...")

    mvdata = MVData(neural_data, downsampling=1)
    embedding = mvdata.get_embedding(method=method, **method_params)
    coords = embedding.coords.T  # (n_samples, n_dims) - may have fewer samples if nodes lost

    # Check for node loss in baseline
    node_loss_rate = 0.0
    ground_truth_aligned = ground_truth

    if hasattr(embedding, 'graph') and hasattr(embedding.graph, 'lost_nodes'):
        lost_nodes = list(embedding.graph.lost_nodes)
        node_loss_rate = len(lost_nodes) / n_samples

        if lost_nodes:
            # Align ground truth to surviving nodes
            all_indices = set(range(n_samples))
            surviving = sorted(all_indices - set(lost_nodes))

            if len(ground_truth.shape) == 1:
                ground_truth_aligned = ground_truth[surviving]
            else:
                ground_truth_aligned = ground_truth[surviving, :]

            if verbose:
                print(f"  Warning: {node_loss_rate:.1%} nodes lost in baseline")
                print(f"  Embedding shape: {coords.shape}, aligned ground truth: {ground_truth_aligned.shape}")

    baseline_metrics = compute_alignment_metrics(coords, ground_truth_aligned, manifold_type)
    baseline_metrics['neuron'] = 'all'
    baseline_metrics['n_neurons'] = n_neurons
    baseline_metrics['node_loss_rate'] = node_loss_rate
    results.append(baseline_metrics)

    # Leave-one-out
    iterator = tqdm(neurons_to_test, desc=f"LOO {method}") if verbose else neurons_to_test

    for neuron_idx in iterator:
        # Remove neuron
        mask = np.ones(n_neurons, dtype=bool)
        mask[neuron_idx] = False
        reduced_data = neural_data[mask, :]

        try:
            # Compute embedding without this neuron
            mvdata_reduced = MVData(reduced_data, downsampling=1)
            embedding_reduced = mvdata_reduced.get_embedding(method=method, **method_params)
            coords_reduced = embedding_reduced.coords.T

            # Handle node loss in graph methods
            node_loss_rate = 0.0
            ground_truth_aligned = ground_truth

            if hasattr(embedding_reduced, 'graph') and hasattr(embedding_reduced.graph, 'lost_nodes'):
                lost_nodes = list(embedding_reduced.graph.lost_nodes)
                node_loss_rate = len(lost_nodes) / n_samples

                if lost_nodes:
                    # Align ground truth to surviving nodes
                    all_indices = set(range(n_samples))
                    surviving = sorted(all_indices - set(lost_nodes))

                    if len(ground_truth.shape) == 1:
                        ground_truth_aligned = ground_truth[surviving]
                    else:
                        ground_truth_aligned = ground_truth[surviving, :]

                    if verbose and node_loss_rate > 0.1:  # Warn if > 10% loss
                        print(f"  Warning: {node_loss_rate:.1%} nodes lost for neuron {neuron_idx}")

            # Compute metrics
            metrics = compute_alignment_metrics(coords_reduced, ground_truth_aligned, manifold_type)
            metrics['neuron'] = neuron_idx
            metrics['n_neurons'] = n_neurons - 1
            metrics['node_loss_rate'] = node_loss_rate

        except Exception as e:
            # Failed embedding
            metrics = {
                'neuron': neuron_idx,
                'n_neurons': n_neurons - 1,
                'reconstruction_error': np.nan,
                'alignment_corr': np.nan,
                'decoding_r2': np.nan,
                'test_error': np.nan,
                'node_loss_rate': np.nan
            }
            if verbose:
                print(f"Failed for neuron {neuron_idx}: {e}")

        results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)
    df.set_index('neuron', inplace=True)

    if verbose:
        print(f"\nCompleted. Shape: {df.shape}")
        print("\nBaseline metrics:")
        print(df.loc['all'])

        # Find most important neurons
        baseline_vals = df.loc['all'][['reconstruction_error', 'alignment_corr', 'decoding_r2']]
        degradations = []

        for idx in df.index[1:]:
            neuron_vals = df.loc[idx][['reconstruction_error', 'alignment_corr', 'decoding_r2']]
            # Higher error = worse, lower correlation = worse
            deg_error = (neuron_vals['reconstruction_error'] - baseline_vals['reconstruction_error']) / (baseline_vals['reconstruction_error'] + 1e-10)
            deg_corr = (baseline_vals['alignment_corr'] - neuron_vals['alignment_corr']) / (baseline_vals['alignment_corr'] + 1e-10)
            deg_r2 = (baseline_vals['decoding_r2'] - neuron_vals['decoding_r2']) / (baseline_vals['decoding_r2'] + 1e-10)

            # Average degradation (handle NaNs)
            degs = [deg_error, deg_corr, deg_r2]
            valid_degs = [d for d in degs if not np.isnan(d)]
            avg_deg = np.mean(valid_degs) if valid_degs else np.nan
            degradations.append((idx, avg_deg))

        # Sort by degradation
        degradations = [(idx, deg) for idx, deg in degradations if not np.isnan(deg)]
        degradations.sort(key=lambda x: x[1], reverse=True)

        if degradations:
            print(f"\nTop 5 most important neurons:")
            for idx, deg in degradations[:5]:
                print(f"  Neuron {idx}: {deg:.3f} degradation")

    return df


def batch_loo_analysis(
    exp: Experiment,
    methods: list = ['pca', 'umap', 'isomap'],
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Run LOO analysis with multiple DR methods.

    Parameters
    ----------
    exp : Experiment
        DRIADA experiment
    methods : list
        List of DR methods to test
    **kwargs
        Additional arguments for loo_analysis_simplified

    Returns
    -------
    dict
        Results for each method
    """
    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()}")
        print('='*60)

        try:
            df = loo_analysis_simplified(exp, method=method, **kwargs)
            results[method] = df
        except Exception as e:
            print(f"Failed: {e}")
            results[method] = None

    return results


if __name__ == "__main__":
    # Example usage
    from driada.experiment.synthetic import generate_2d_manifold_exp

    # Generate test data
    exp = generate_2d_manifold_exp(
        n_neurons=30,
        duration=200,
        fps=20,
        seed=42
    )

    # Run simplified LOO
    results = loo_analysis_simplified(
        exp,
        method='umap',
        neurons_to_test=[0, 5, 10, 15, 20],  # Test subset for speed
        verbose=True
    )

    print("\nResults:")
    print(results)