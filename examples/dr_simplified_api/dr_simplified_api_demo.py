#!/usr/bin/env python
"""
Dimensionality Reduction API Demo
=================================

This example demonstrates DRIADA's dimensionality reduction API, showing how
to use various DR methods with minimal code.

Basic usage:
    embedding = mvdata.get_embedding(method='pca')

With custom parameters:
    embedding = mvdata.get_embedding(method='umap', n_neighbors=30, min_dist=0.1)

The API automatically handles:
- Default parameters for each method
- Graph construction for methods that need it
- Metric parameters
- Method-specific configurations
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

from driada.dim_reduction import MVData


def generate_demo_data():
    """Generate Swiss roll data for demonstration."""
    n_samples = 1000
    X, color = make_swiss_roll(n_samples, noise=0.1, random_state=42)

    # Transpose to match MVData format (features x samples)
    X = X.T

    return X, color


def demo_dr_methods():
    """Demonstrate various DR methods with the DRIADA API."""
    print("Generating Swiss Roll dataset...")
    X, color = generate_demo_data()

    # Create MVData object
    mvdata = MVData(X)

    print("\n" + "=" * 60)
    print("DIMENSIONALITY REDUCTION METHODS")
    print("=" * 60)

    # Linear methods
    print("\n--- Linear Methods ---")

    print("\n1. PCA (Principal Component Analysis):")
    print("   Default usage:")
    print("     emb = mvdata.get_embedding(method='pca')")
    print("   With parameters:")
    print("     emb = mvdata.get_embedding(method='pca', dim=3)")
    emb_pca = mvdata.get_embedding(method="pca")
    print(f"   → Result shape: {emb_pca.coords.shape}")

    # Manifold learning methods
    print("\n--- Manifold Learning Methods ---")

    print("\n2. Isomap (Isometric Mapping):")
    print("   Default usage:")
    print("     emb = mvdata.get_embedding(method='isomap')")
    print("   With parameters:")
    print("     emb = mvdata.get_embedding(method='isomap', n_neighbors=30, dim=3)")
    emb_iso = mvdata.get_embedding(method="isomap")
    print(f"   → Result shape: {emb_iso.coords.shape}")

    print("\n3. LLE (Locally Linear Embedding):")
    print("   Default usage:")
    print("     emb = mvdata.get_embedding(method='lle')")
    print("   With parameters:")
    print("     emb = mvdata.get_embedding(method='lle', n_neighbors=20)")
    emb_lle = mvdata.get_embedding(method="lle")
    print(f"   → Result shape: {emb_lle.coords.shape}")

    print("\n4. Laplacian Eigenmaps:")
    print("   Default usage:")
    print("     emb = mvdata.get_embedding(method='le')")
    print("   With parameters:")
    print("     emb = mvdata.get_embedding(method='le', n_neighbors=20)")
    emb_le = mvdata.get_embedding(method="le")
    print(f"   → Result shape: {emb_le.coords.shape}")

    # Visualization methods
    print("\n--- Visualization Methods ---")

    print("\n5. t-SNE (t-distributed Stochastic Neighbor Embedding):")
    print("   Default usage:")
    print("     emb = mvdata.get_embedding(method='tsne')")
    print("   With parameters:")
    print("     emb = mvdata.get_embedding(method='tsne', perplexity=50)")
    emb_tsne = mvdata.get_embedding(method="tsne", perplexity=50)
    print(f"   → Result shape: {emb_tsne.coords.shape}")

    print("\n6. UMAP (Uniform Manifold Approximation and Projection):")
    print("   Default usage:")
    print("     emb = mvdata.get_embedding(method='umap')")
    print("   With parameters:")
    print(
        "     emb = mvdata.get_embedding(method='umap', n_neighbors=50, min_dist=0.3)"
    )
    emb_umap = mvdata.get_embedding(method="umap", n_neighbors=50, min_dist=0.3)
    print(f"   → Result shape: {emb_umap.coords.shape}")

    # Distance-based methods
    print("\n--- Distance-based Methods ---")

    print("\n7. MDS (Multidimensional Scaling):")
    print("   Usage (requires distance matrix):")
    print("     mvdata.get_distmat()  # Compute distance matrix")
    print("     emb = mvdata.get_embedding(method='mds')")
    mvdata.get_distmat()
    emb_mds = mvdata.get_embedding(method="mds")
    print(f"   → Result shape: {emb_mds.coords.shape}")

    # Show parameter options
    print("\n" + "=" * 60)
    print("COMMON PARAMETERS")
    print("=" * 60)
    print("\nAll methods accept:")
    print("  dim: int - Number of output dimensions (default: 2)")
    print("\nGraph-based methods accept:")
    print("  n_neighbors: int - Number of nearest neighbors")
    print("\nt-SNE specific:")
    print("  perplexity: float - Balance between local and global structure")
    print("\nUMAP specific:")
    print("  min_dist: float - Minimum distance between points in embedding")
    print("\nDiffusion maps specific:")
    print("  dm_alpha: float - Diffusion map alpha parameter")

    # Visualize results
    visualize_embeddings(
        [emb_pca, emb_iso, emb_umap, emb_tsne],
        ["PCA", "Isomap", "UMAP", "t-SNE"],
        color,
    )


def visualize_embeddings(embeddings, names, color):
    """Visualize multiple embeddings side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (emb, name) in enumerate(zip(embeddings, names)):
        ax = axes[i]
        coords = emb.coords

        scatter = ax.scatter(
            coords[0, :], coords[1, :], c=color, cmap="viridis", s=20, alpha=0.7
        )
        ax.set_title(f"{name} Embedding")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        # Add colorbar to first subplot
        if i == 0:
            plt.colorbar(scatter, ax=ax, label="Position on roll")

    plt.tight_layout()
    plt.savefig("dr_simplified_api_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved visualization to: dr_simplified_api_comparison.png")
    plt.show()


def show_advanced_usage():
    """Show advanced usage patterns."""
    print("\n" + "=" * 60)
    print("ADVANCED USAGE PATTERNS")
    print("=" * 60)

    # Generate data
    X, _ = generate_demo_data()
    mvdata = MVData(X)

    print("\n1. Working with high-dimensional data:")
    print("   # Project to higher dimensions first, then visualize")
    print("   emb_5d = mvdata.get_embedding(method='pca', dim=5)")
    print("   # Create new MVData from embedding for further reduction")
    print("   mvdata_5d = MVData(emb_5d.coords)")
    print("   emb_2d = mvdata_5d.get_embedding(method='tsne')")

    print("\n2. Using custom metrics:")
    print("   # For methods that support custom metrics")
    print("   emb = mvdata.get_embedding(method='isomap', metric='cosine')")

    print("\n3. Handling sparse data:")
    print("   # MVData automatically handles sparse matrices")
    print("   from scipy.sparse import csr_matrix")
    print("   sparse_data = csr_matrix(X)")
    print("   mvdata_sparse = MVData(sparse_data)")
    print("   emb = mvdata_sparse.get_embedding(method='pca')")

    print("\n4. Chaining embeddings:")
    print("   # Use one embedding as input to another")
    print("   emb1 = mvdata.get_embedding(method='pca', dim=50)")
    print("   mvdata2 = MVData(emb1.coords)")
    print("   emb2 = mvdata2.get_embedding(method='umap')")


def main():
    """Run the demonstration."""
    print("=" * 60)
    print("DRIADA Dimensionality Reduction API Demo")
    print("=" * 60)

    demo_dr_methods()
    show_advanced_usage()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
