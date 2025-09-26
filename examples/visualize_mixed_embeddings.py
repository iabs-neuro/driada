#!/usr/bin/env python
"""
Visualize embeddings and explain space reconstruction metrics.

This script:
1. Creates embeddings with different methods
2. Shows how they reconstruct the 2D spatial manifold
3. Explains the reconstruction metrics used
"""

import sys
sys.path.insert(0, '/Users/nikita/PycharmProjects/driada2/src')

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.synthetic import generate_mixed_population_exp
from driada.dim_reduction import MVData

def main():
    print("="*70)
    print("Embedding Visualization and Reconstruction Metrics")
    print("="*70)

    # Generate the same mixed population as in the test
    print("\n1. Generating mixed population...")
    exp, info = generate_mixed_population_exp(
        n_neurons=30,
        manifold_fraction=0.6,
        manifold_type='2d_spatial',
        duration=100,  # Short for quick demo
        fps=20.0,
        seed=42,
        verbose=False,
        return_info=True
    )

    print(f"   Created {exp.n_cells} neurons: {info['population_composition']['n_manifold']} manifold, "
          f"{info['population_composition']['n_feature_selective']} feature-selective")

    # Get neural data and ground truth
    calcium_data = exp.calcium.scdata  # Scaled calcium data

    # Get ground truth spatial positions
    x_pos = exp.dynamic_features['x_position'].data
    y_pos = exp.dynamic_features['y_position'].data

    # Downsample for speed
    downsample = 5
    calcium_downsampled = calcium_data[:, ::downsample]
    x_pos_ds = x_pos[::downsample]
    y_pos_ds = y_pos[::downsample]

    print(f"   Data shape: {calcium_downsampled.shape}")
    print(f"   Downsampled by {downsample}: {calcium_data.shape[1]} → {calcium_downsampled.shape[1]} timepoints")

    # Create embeddings with different methods
    print("\n2. Computing embeddings...")

    methods = ['pca', 'auto_le', 'umap']
    embeddings = {}

    for method in methods:
        print(f"   Computing {method.upper()}...")
        mvdata = MVData(calcium_downsampled)

        if method == 'pca':
            emb = mvdata.get_embedding(method=method, dim=2)
        elif method == 'auto_le':
            emb = mvdata.get_embedding(method=method, dim=2, n_neighbors=50)
        elif method == 'umap':
            emb = mvdata.get_embedding(method=method, dim=2, n_neighbors=30)

        embeddings[method] = emb.coords.T  # (n_timepoints, 2)
        print(f"      Shape: {embeddings[method].shape}")

    # Visualize embeddings
    print("\n3. Creating visualizations...")

    fig = plt.figure(figsize=(18, 10))

    # First row: Ground truth and embeddings colored by spatial position
    for i, (method, coords) in enumerate([('Ground Truth', np.column_stack([x_pos_ds, y_pos_ds]))] +
                                          [(m, embeddings[m]) for m in methods]):
        ax = plt.subplot(2, 4, i+1)

        if method == 'Ground Truth':
            # Show actual spatial positions
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                               c=np.arange(len(coords)), cmap='viridis',
                               s=5, alpha=0.6)
            ax.set_title("Ground Truth\n2D Spatial Position")
        else:
            # Color by true spatial position (using distance from origin)
            spatial_dist = np.sqrt(x_pos_ds**2 + y_pos_ds**2)
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                               c=spatial_dist, cmap='viridis',
                               s=5, alpha=0.6)
            ax.set_title(f"{method.upper()} Embedding\nColored by true position")

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        if i == 1:
            plt.colorbar(scatter, ax=ax, label="Spatial distance")

    # Second row: Show trajectory segments to see structure preservation
    n_segments = min(300, len(x_pos_ds))  # Show first N points as trajectory

    for i, (method, coords) in enumerate([(m, embeddings[m]) for m in methods]):
        ax = plt.subplot(2, 4, 5+i)

        # Plot trajectory
        ax.plot(coords[:n_segments, 0], coords[:n_segments, 1],
                'b-', alpha=0.3, linewidth=0.5)
        ax.scatter(coords[:n_segments, 0], coords[:n_segments, 1],
                  c=np.arange(n_segments), cmap='coolwarm', s=10)

        ax.set_title(f"{method.upper()}\nTrajectory preservation")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    plt.tight_layout()
    plt.savefig('mixed_embeddings_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved to: mixed_embeddings_comparison.png")
    plt.show()

    # Explain reconstruction metrics
    print("\n" + "="*70)
    print("SPACE RECONSTRUCTION METRICS EXPLAINED")
    print("="*70)

    print("""
The LOO analysis uses three main reconstruction metrics:

1. RECONSTRUCTION ERROR
   - Measures how well the 2D embedding matches the true 2D spatial positions
   - Computed using Procrustes analysis (optimal alignment + scaling + rotation)
   - Lower is better (perfect = 0)
   - For spatial data: Mean squared error after optimal alignment
   - For circular data: Circular correlation distance

2. DECODING ACCURACY
   - Tests if spatial position can be decoded from the embedding
   - Uses k-nearest neighbors regression (k=15 by default)
   - Cross-validated: train on part of data, test on rest
   - Higher is better (perfect = 1.0)
   - Measures: R² score for continuous variables

3. ALIGNMENT CORRELATION
   - Correlation between pairwise distances in:
     * Original high-dimensional neural space
     * Low-dimensional embedding space
     * Ground truth behavioral space
   - Higher correlation = better structure preservation
   - Range: -1 to 1 (perfect = 1)

WHY THESE METRICS MATTER:
- Reconstruction error: Direct measure of spatial map quality
- Decoding accuracy: Functional test - can we read out position?
- Alignment correlation: Global structure preservation

In our test:
- Removing manifold neurons should INCREASE reconstruction error
- Removing manifold neurons should DECREASE decoding accuracy
- Removing unrelated neurons should have minimal effect
""")

    # Compute and show actual metrics
    print("\n4. Computing actual metrics for comparison...")

    from driada.dim_reduction.manifold_metrics import (
        compute_reconstruction_error,
        compute_decoding_accuracy
    )

    ground_truth = np.column_stack([x_pos_ds, y_pos_ds])

    for method, coords in embeddings.items():
        print(f"\n   {method.upper()} metrics:")

        # Reconstruction error
        rec_error = compute_reconstruction_error(
            coords, ground_truth, manifold_type='spatial'
        )
        if isinstance(rec_error, dict):
            rec_error = rec_error.get('mean_error', rec_error.get('error', 0))
        print(f"     Reconstruction error: {rec_error:.4f}")

        # Decoding accuracy
        dec_acc = compute_decoding_accuracy(
            coords, ground_truth, k_neighbors=15
        )
        if isinstance(dec_acc, dict):
            dec_acc = dec_acc.get('mean_r2', dec_acc.get('r2_score', 0))
        print(f"     Decoding accuracy: {dec_acc:.4f}")

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)

if __name__ == "__main__":
    main()