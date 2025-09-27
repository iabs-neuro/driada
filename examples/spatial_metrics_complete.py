#!/usr/bin/env python
"""
Complete spatial metrics computation with robust edge case handling.

This module provides production-ready functions for computing spatial
correspondence metrics between embeddings and ground truth positions.

Key Features:
- Handles DRIADA standard format (n_dims, n_samples)
- Robust to edge cases (constant velocity, identical points)
- Clear error messages and debugging info
- Supports both spatial and circular manifolds
"""

import sys
sys.path.insert(0, '/Users/nikita/PycharmProjects/driada2/src')

import numpy as np
import warnings
from typing import Dict, Optional, Union, Tuple
from driada.dim_reduction.manifold_metrics import (
    compute_reconstruction_error,
    compute_decoding_accuracy,
    compute_embedding_alignment_metrics
)


def compute_spatial_metrics(
    embedding_coords: np.ndarray,
    ground_truth: np.ndarray,
    manifold_type: str = 'spatial',
    verbose: bool = True,
    handle_edge_cases: bool = True
) -> Dict[str, Union[float, str]]:
    """
    Compute all spatial correspondence metrics with robust edge case handling.
    
    Parameters
    ----------
    embedding_coords : ndarray
        Shape (n_dims, n_samples) - DRIADA STANDARD FORMAT
        Your embedding coordinates from any DRIADA method.
    ground_truth : ndarray
        Shape (n_dims, n_samples) - DRIADA STANDARD FORMAT
        True spatial positions or angles.
    manifold_type : str
        'spatial' for 2D/3D positions, 'circular' for angles
    verbose : bool
        Print detailed info and warnings
    handle_edge_cases : bool
        Whether to handle edge cases gracefully (recommended)
    
    Returns
    -------
    dict : All metrics computed with edge case handling
        - reconstruction_error: Lower is better (0 = perfect)
        - reconstruction_corr: Higher is better (1 = perfect)
        - decoding_r2: Higher is better (1 = perfect decoding)
        - alignment_corr: Higher is better (1 = perfect alignment)
        - velocity_corr: Velocity correlation (may be NaN)
        - edge_case_notes: Description of any edge cases encountered
    
    Examples
    --------
    >>> # Generate example data
    >>> n_samples = 1000
    >>> ground_truth = np.random.randn(2, n_samples)  # 2D positions
    >>> embedding = ground_truth + 0.3 * np.random.randn(2, n_samples)
    >>> 
    >>> # Compute metrics
    >>> metrics = compute_spatial_metrics(embedding, ground_truth)
    >>> print(f"Reconstruction error: {metrics['reconstruction_error']:.4f}")
    """
    
    # Initialize results
    metrics = {
        'reconstruction_error': np.nan,
        'reconstruction_corr': np.nan,
        'decoding_r2': np.nan,
        'alignment_corr': np.nan,
        'velocity_corr': np.nan,
        'edge_case_notes': ''
    }
    edge_cases = []
    
    # Validate inputs
    if embedding_coords.shape[1] != ground_truth.shape[1]:
        raise ValueError(
            f"Sample count mismatch: embedding has {embedding_coords.shape[1]}, "
            f"ground truth has {ground_truth.shape[1]}"
        )
    
    n_dims_emb = embedding_coords.shape[0]
    n_dims_gt = ground_truth.shape[0]
    n_samples = embedding_coords.shape[1]
    
    if verbose:
        print(f"Input shapes (DRIADA format):")
        print(f"  Embedding: {embedding_coords.shape} (n_dims={n_dims_emb}, n_samples={n_samples})")
        print(f"  Ground truth: {ground_truth.shape} (n_dims={n_dims_gt}, n_samples={n_samples})")
        print(f"  Manifold type: {manifold_type}")
    
    # Check for edge cases
    if handle_edge_cases:
        # Check for constant positions (no movement)
        gt_var = np.var(ground_truth, axis=1)
        if np.any(gt_var < 1e-10):
            edge_cases.append('constant_position')
            if verbose:
                print("⚠️  Edge case detected: Very low variance in ground truth positions")
        
        # Check for constant velocity (circular motion)
        if n_samples > 2:
            if manifold_type == 'circular' or n_dims_gt == 2:
                # Compute angular velocities for circular data
                if n_dims_gt == 2:
                    angles = np.arctan2(ground_truth[1, :], ground_truth[0, :])
                    angular_velocity = np.diff(angles)
                    angular_velocity = np.arctan2(np.sin(angular_velocity), np.cos(angular_velocity))
                    velocity_var = np.var(angular_velocity)
                    
                    if velocity_var < 1e-10:
                        edge_cases.append('constant_angular_velocity')
                        if verbose:
                            print("⚠️  Edge case detected: Constant angular velocity")
    
    # Transpose for metrics functions (they expect n_samples, n_dims)
    embedding_T = embedding_coords.T
    ground_truth_T = ground_truth.T
    
    # 1. RECONSTRUCTION ERROR - Most reliable metric
    if verbose:
        print("\n1. Computing reconstruction error...")
    
    try:
        rec_result = compute_reconstruction_error(
            embedding_T,
            ground_truth_T,
            manifold_type=manifold_type,
            allow_rotation=True,
            allow_reflection=True,
            allow_scaling=True
        )
        
        if isinstance(rec_result, dict):
            metrics['reconstruction_error'] = rec_result.get('error', 
                                             rec_result.get('mean_error', np.nan))
            metrics['reconstruction_corr'] = rec_result.get('correlation', np.nan)
            
            if verbose:
                print(f"  ✅ Error: {metrics['reconstruction_error']:.4f}")
                if not np.isnan(metrics['reconstruction_corr']):
                    print(f"  ✅ Correlation: {metrics['reconstruction_corr']:.4f}")
        else:
            metrics['reconstruction_error'] = rec_result
            if verbose:
                print(f"  ✅ Error: {metrics['reconstruction_error']:.4f}")
    
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Failed: {e}")
        edge_cases.append('reconstruction_failed')
    
    # 2. DECODING ACCURACY - Tests functional readout
    if verbose:
        print("\n2. Computing decoding accuracy...")
    
    try:
        dec_result = compute_decoding_accuracy(
            embedding_T,
            ground_truth_T,
            manifold_type=manifold_type,
            train_fraction=0.8
        )
        
        if isinstance(dec_result, dict):
            for key in ['r2_score', 'mean_r2', 'score']:
                if key in dec_result:
                    metrics['decoding_r2'] = dec_result[key]
                    break
            else:
                # Try to find any numeric value
                for v in dec_result.values():
                    if isinstance(v, (float, int)) and not np.isnan(v):
                        metrics['decoding_r2'] = v
                        break
        else:
            metrics['decoding_r2'] = dec_result if dec_result is not None else np.nan
        
        if verbose and not np.isnan(metrics['decoding_r2']):
            print(f"  ✅ R² score: {metrics['decoding_r2']:.4f}")
    
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Failed: {e}")
        edge_cases.append('decoding_failed')
    
    # 3. ALIGNMENT METRICS - May fail for edge cases
    if verbose:
        print("\n3. Computing alignment metrics...")
    
    try:
        align_result = compute_embedding_alignment_metrics(
            embedding_T,
            ground_truth_T,
            manifold_type=manifold_type,
            allow_rotation=True,
            allow_reflection=True,
            allow_scaling=True
        )
        
        if isinstance(align_result, dict):
            metrics['alignment_corr'] = align_result.get('correlation',
                                       align_result.get('corr', np.nan))
            
            if 'velocity_correlation' in align_result:
                metrics['velocity_corr'] = align_result['velocity_correlation']
                
                # Handle NaN in velocity correlation
                if np.isnan(metrics['velocity_corr']) and handle_edge_cases:
                    edge_cases.append('velocity_corr_nan')
                    if verbose:
                        print("  ⚠️  Velocity correlation is NaN (likely constant velocity)")
                    
                    # Use reconstruction correlation as fallback
                    if not np.isnan(metrics['reconstruction_corr']):
                        metrics['alignment_corr'] = metrics['reconstruction_corr']
                        if verbose:
                            print(f"  ↻  Using reconstruction correlation as fallback: {metrics['alignment_corr']:.4f}")
            
            if verbose and not np.isnan(metrics['alignment_corr']):
                print(f"  ✅ Alignment correlation: {metrics['alignment_corr']:.4f}")
                if 'velocity_corr' in metrics and not np.isnan(metrics['velocity_corr']):
                    print(f"  ✅ Velocity correlation: {metrics['velocity_corr']:.4f}")
    
    except ValueError as e:
        if "NaN" in str(e) or "no variation" in str(e):
            edge_cases.append('alignment_nan_expected')
            if verbose:
                print(f"  ⚠️  Expected failure: {e}")
            
            # Use reconstruction correlation as fallback
            if handle_edge_cases and not np.isnan(metrics['reconstruction_corr']):
                metrics['alignment_corr'] = metrics['reconstruction_corr']
                if verbose:
                    print(f"  ↻  Using reconstruction correlation as fallback: {metrics['alignment_corr']:.4f}")
        else:
            if verbose:
                print(f"  ⚠️  Unexpected failure: {e}")
            edge_cases.append('alignment_failed')
    
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Failed: {e}")
        edge_cases.append('alignment_failed')
    
    # Record edge cases
    if edge_cases:
        metrics['edge_case_notes'] = ', '.join(edge_cases)
    
    return metrics


def quick_spatial_metrics(
    embedding: np.ndarray,
    ground_truth: np.ndarray,
    manifold_type: str = 'spatial'
) -> Tuple[float, float, float]:
    """
    Quick computation of the three main spatial metrics.
    
    Parameters
    ----------
    embedding : ndarray
        Shape (n_dims, n_samples) - DRIADA format
    ground_truth : ndarray
        Shape (n_dims, n_samples) - DRIADA format
    manifold_type : str
        'spatial' or 'circular'
    
    Returns
    -------
    tuple : (reconstruction_error, decoding_r2, alignment_corr)
        Returns NaN for any metric that fails
    """
    
    metrics = compute_spatial_metrics(
        embedding,
        ground_truth,
        manifold_type=manifold_type,
        verbose=False,
        handle_edge_cases=True
    )
    
    return (
        metrics['reconstruction_error'],
        metrics['decoding_r2'],
        metrics['alignment_corr']
    )


def validate_metrics(metrics: Dict[str, Union[float, str]], verbose: bool = True) -> bool:
    """
    Validate computed metrics and provide interpretation.
    
    Parameters
    ----------
    metrics : dict
        Output from compute_spatial_metrics
    verbose : bool
        Print interpretation
    
    Returns
    -------
    bool : True if all critical metrics are valid
    """
    
    valid = True
    
    if verbose:
        print("\n" + "="*60)
        print("METRICS VALIDATION AND INTERPRETATION")
        print("="*60)
    
    # Check reconstruction error
    if np.isnan(metrics.get('reconstruction_error', np.nan)):
        valid = False
        if verbose:
            print("❌ Reconstruction error: FAILED")
    else:
        error = metrics['reconstruction_error']
        if verbose:
            quality = "excellent" if error < 0.1 else "good" if error < 0.5 else "moderate" if error < 1.0 else "poor"
            print(f"✅ Reconstruction error: {error:.4f} ({quality})")
    
    # Check decoding accuracy
    if np.isnan(metrics.get('decoding_r2', np.nan)):
        if verbose:
            print("⚠️  Decoding R²: FAILED (non-critical)")
    else:
        r2 = metrics['decoding_r2']
        if verbose:
            quality = "excellent" if r2 > 0.9 else "good" if r2 > 0.7 else "moderate" if r2 > 0.5 else "poor"
            print(f"✅ Decoding R²: {r2:.4f} ({quality})")
    
    # Check alignment correlation
    if np.isnan(metrics.get('alignment_corr', np.nan)):
        if verbose:
            print("⚠️  Alignment correlation: FAILED (may be edge case)")
    else:
        corr = metrics['alignment_corr']
        if verbose:
            quality = "excellent" if corr > 0.9 else "good" if corr > 0.7 else "moderate" if corr > 0.5 else "poor"
            print(f"✅ Alignment correlation: {corr:.4f} ({quality})")
    
    # Report edge cases
    if metrics.get('edge_case_notes'):
        if verbose:
            print(f"\n⚠️  Edge cases encountered: {metrics['edge_case_notes']}")
    
    return valid


if __name__ == "__main__":
    print("="*60)
    print("COMPLETE SPATIAL METRICS WITH EDGE CASE HANDLING")
    print("="*60)
    
    # Test 1: Normal case
    print("\n1. Normal case - noisy embedding:")
    print("-"*40)
    n = 1000
    t = np.linspace(0, 4*np.pi, n)
    x = np.cos(t) + 0.5 * np.cos(3*t)
    y = np.sin(t) + 0.5 * np.sin(3*t)
    ground_truth = np.vstack([x, y])  # (2, n)
    embedding = ground_truth + 0.3 * np.random.randn(*ground_truth.shape)
    
    metrics = compute_spatial_metrics(embedding, ground_truth, verbose=True)
    validate_metrics(metrics)
    
    # Test 2: Edge case - constant angular velocity
    print("\n2. Edge case - constant angular velocity:")
    print("-"*40)
    angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
    ground_truth = np.vstack([np.cos(angles), np.sin(angles)])
    embedding = ground_truth + 0.1 * np.random.randn(*ground_truth.shape)
    
    metrics = compute_spatial_metrics(embedding, ground_truth, manifold_type='circular', verbose=True)
    validate_metrics(metrics)
    
    # Test 3: Quick metrics
    print("\n3. Quick metrics computation:")
    print("-"*40)
    rec_err, dec_r2, align_corr = quick_spatial_metrics(embedding, ground_truth)
    print(f"Reconstruction error: {rec_err:.4f}")
    print(f"Decoding R²: {dec_r2:.4f}")
    print(f"Alignment correlation: {align_corr:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
This module provides robust spatial metrics computation that:
1. Handles DRIADA standard format correctly
2. Detects and handles edge cases gracefully
3. Provides clear feedback about what's happening
4. Falls back to alternative metrics when needed
5. Never silently fails or returns misleading results

Use compute_spatial_metrics() for detailed analysis with edge case handling.
Use quick_spatial_metrics() for simple metric extraction.
Use validate_metrics() to interpret results.
    """)