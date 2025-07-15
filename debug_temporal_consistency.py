#!/usr/bin/env python
"""Debug why temporal consistency is near zero."""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from driada.experiment import generate_circular_manifold_exp
from driada.dim_reduction.data import MVData
from driada.dim_reduction.dr_base import METHODS_DICT
from driada.signals import manifold_preprocessing
from driada.dim_reduction.manifold_metrics import extract_angles_from_embedding
from scipy.stats import pearsonr

def debug_temporal_consistency():
    """Debug temporal consistency calculation step by step."""
    
    print("=== TEMPORAL CONSISTENCY DEBUGGING ===")
    
    # Generate data
    exp, info = generate_circular_manifold_exp(
        n_neurons=40, duration=200, fps=20.0, kappa=4.0, noise_std=0.1, seed=42
    )
    
    filtered_neural = manifold_preprocessing(exp.calcium.T, method='gaussian', sigma=1.5)
    neural_data = filtered_neural.T
    true_angles = info['head_direction']
    
    D = MVData(neural_data)
    
    # Get Isomap embedding
    isomap_params = {'e_method_name': 'isomap', 'dim': 2, 'e_method': METHODS_DICT['isomap'], 'nn': 7}
    metric_params = {'metric_name': 'l2', 'sigma': 1, 'p': 2}
    graph_params = {'g_method_name': 'knn', 'weighted': 0, 'nn': 7, 'max_deleted_nodes': 0.2, 'dist_to_aff': 'hk'}
    
    emb = D.get_embedding(isomap_params, g_params=graph_params, m_params=metric_params)
    coords = emb.coords.T
    
    # Handle lost nodes
    if hasattr(emb, 'graph') and emb.graph is not None and hasattr(emb.graph, 'lost_nodes'):
        lost_nodes = emb.graph.lost_nodes
        valid_indices = np.setdiff1d(np.arange(len(true_angles)), np.array(list(lost_nodes)))
        filtered_angles = true_angles[valid_indices]
    else:
        filtered_angles = true_angles
    
    print(f"Embedding shape: {coords.shape}")
    print(f"Filtered angles shape: {filtered_angles.shape}")
    
    # Step 1: Extract angles from embedding
    extracted_angles = extract_angles_from_embedding(coords)
    print(f"\nExtracted angles range: [{np.min(extracted_angles):.3f}, {np.max(extracted_angles):.3f}]")
    print(f"True angles range: [{np.min(filtered_angles):.3f}, {np.max(filtered_angles):.3f}]")
    
    # Step 2: Compute velocities
    true_velocity = np.diff(filtered_angles)
    reconstructed_velocity = np.diff(extracted_angles)
    
    # Handle circular wrapping
    true_velocity = np.arctan2(np.sin(true_velocity), np.cos(true_velocity))
    reconstructed_velocity = np.arctan2(np.sin(reconstructed_velocity), np.cos(reconstructed_velocity))
    
    print(f"\nTrue velocity stats:")
    print(f"  Mean: {np.mean(true_velocity):.6f}")
    print(f"  Std: {np.std(true_velocity):.6f}")
    print(f"  Range: [{np.min(true_velocity):.3f}, {np.max(true_velocity):.3f}]")
    
    print(f"\nReconstructed velocity stats:")
    print(f"  Mean: {np.mean(reconstructed_velocity):.6f}")
    print(f"  Std: {np.std(reconstructed_velocity):.6f}")
    print(f"  Range: [{np.min(reconstructed_velocity):.3f}, {np.max(reconstructed_velocity):.3f}]")
    
    # Step 3: Compute correlation
    correlation, p_value = pearsonr(true_velocity, reconstructed_velocity)
    print(f"\nTemporal consistency (correlation): {correlation:.6f}")
    print(f"P-value: {p_value:.6e}")
    
    # Visualize to understand the problem
    plt.figure(figsize=(15, 5))
    
    # Plot 1: True vs reconstructed angles over time
    plt.subplot(131)
    time_indices = np.arange(len(filtered_angles))
    plt.plot(time_indices[:500], filtered_angles[:500], 'b-', label='True angles', alpha=0.7)
    plt.plot(time_indices[:500], extracted_angles[:500], 'r-', label='Reconstructed angles', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Angle (rad)')
    plt.title('Angles over time (first 500 frames)')
    plt.legend()
    
    # Plot 2: Velocities
    plt.subplot(132)
    plt.plot(time_indices[:499], true_velocity[:499], 'b-', label='True velocity', alpha=0.7)
    plt.plot(time_indices[:499], reconstructed_velocity[:499], 'r-', label='Reconstructed velocity', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Angular velocity (rad/frame)')
    plt.title('Angular velocities (first 500 frames)')
    plt.legend()
    
    # Plot 3: Embedding structure
    plt.subplot(133)
    plt.scatter(coords[:, 0], coords[:, 1], c=extracted_angles, cmap='hsv', alpha=0.5, s=1)
    plt.colorbar(label='Extracted angle')
    plt.xlabel('Embedding dim 1')
    plt.ylabel('Embedding dim 2')
    plt.title('2D Embedding colored by angle')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('temporal_consistency_debug.png', dpi=150)
    print(f"\nDebug plots saved to temporal_consistency_debug.png")
    
    # Additional analysis: Check if the embedding preserves order
    # For a circular manifold, nearby points in time should be nearby in embedding
    time_diffs = []
    embedding_diffs = []
    for i in range(0, len(coords)-10, 10):
        time_diff = 10  # 10 frames apart
        embedding_diff = np.linalg.norm(coords[i+10] - coords[i])
        time_diffs.append(time_diff)
        embedding_diffs.append(embedding_diff)
    
    time_vs_embedding_corr = np.corrcoef(time_diffs, embedding_diffs)[0, 1]
    print(f"\nCorrelation between time distance and embedding distance: {time_vs_embedding_corr:.6f}")
    
    # Check if angle extraction is the issue
    # Try a different approach: use PCA on embedding
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(coords)
    pca_angles = np.arctan2(pca_coords[:, 1], pca_coords[:, 0])
    
    pca_velocity = np.diff(pca_angles)
    pca_velocity = np.arctan2(np.sin(pca_velocity), np.cos(pca_velocity))
    pca_correlation, _ = pearsonr(true_velocity, pca_velocity)
    print(f"\nAlternative (PCA-based) angle extraction correlation: {pca_correlation:.6f}")
    
    return {
        'correlation': correlation,
        'embedding_structure': time_vs_embedding_corr,
        'alternative_correlation': pca_correlation
    }

if __name__ == "__main__":
    results = debug_temporal_consistency()