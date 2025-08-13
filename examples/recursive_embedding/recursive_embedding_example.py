"""
Example: Recursive Embedding Pipeline with to_mvdata()

This example demonstrates how to use the new to_mvdata() method to create
recursive embedding pipelines, where embeddings can be further reduced.
"""

import numpy as np
import matplotlib.pyplot as plt
from driada.dim_reduction import MVData
from driada.utils.visual import plot_embedding_comparison

# Generate sample high-dimensional data
np.random.seed(42)
n_features = 100
n_samples = 200

# Create three clusters in high-dimensional space with some overlap
# Use smaller separation to ensure connectivity
cluster1 = np.random.randn(n_features, n_samples // 3) * 0.5
cluster2 = np.random.randn(n_features, n_samples // 3) * 0.5 + 0.8
cluster3 = np.random.randn(n_features, n_samples - 2 * (n_samples // 3)) * 0.5 + 0.4

# Combine clusters
data = np.hstack([cluster1, cluster2, cluster3])
labels = np.array(
    [0] * (n_samples // 3)
    + [1] * (n_samples // 3)
    + [2] * (n_samples - 2 * (n_samples // 3))
)

# Create MVData object
mvdata = MVData(data, labels=labels)
print(f"Original data shape: {mvdata.data.shape} (features × samples)")

# First reduction: 100D → 20D using PCA
print("\nStep 1: Reducing 100D → 20D with PCA...")
pca_embedding = mvdata.get_embedding(method="pca", dim=20)
print(f"PCA embedding shape: {pca_embedding.coords.shape}")

# Convert embedding to MVData for further processing
pca_mvdata = pca_embedding.to_mvdata()
print(f"Converted to MVData with shape: {pca_mvdata.data.shape}")

# Second reduction: 20D → 2D using UMAP
print("\nStep 2: Reducing 20D → 2D with UMAP...")
umap_embedding = pca_mvdata.get_embedding(method="umap", dim=2, n_neighbors=30)
print(f"Final UMAP embedding shape: {umap_embedding.coords.shape}")

# For comparison, direct 100D → 2D reduction
print("\nFor comparison: Direct 100D → 2D with UMAP...")
direct_umap = mvdata.get_embedding(method="umap", dim=2, n_neighbors=30)

# Visualize results using plot_embedding_comparison
embeddings = {
    "Recursive: PCA→UMAP": umap_embedding.coords.T,  # Transpose to (n_samples, n_components)
    "Direct: UMAP": direct_umap.coords.T,
}

# Create dummy features for visualization (using cluster labels)
# Convert labels to continuous "feature" for color mapping
label_feature = labels.astype(float)
features = {"cluster": label_feature}
feature_names = {"cluster": "Cluster ID"}

# Create comparison plot
fig = plot_embedding_comparison(
    embeddings=embeddings,
    features=features,
    feature_names=feature_names,
    with_trajectory=False,  # No trajectory for this example
    compute_metrics=True,
    figsize=(12, 8),
    save_path="recursive_embedding_comparison.png",
)
plt.show()

# Advanced example: Multi-stage pipeline with different methods
print("\n" + "=" * 50)
print("Advanced: Multi-stage dimensionality reduction pipeline")
print("=" * 50)

# Stage 1: Autoencoder for non-linear reduction (if torch available)
try:
    import torch

    print("\nStage 1: Autoencoder 100D → 50D...")
    ae_embedding = mvdata.get_embedding(method="ae", dim=50, epochs=20, verbose=False)
    ae_mvdata = ae_embedding.to_mvdata()

    # Stage 2: PCA for linear reduction
    print("Stage 2: PCA 50D → 10D...")
    pca_embedding2 = ae_mvdata.get_embedding(method="pca", dim=10)
    pca_mvdata2 = pca_embedding2.to_mvdata()

    # Stage 3: t-SNE for final visualization
    print("Stage 3: t-SNE 10D → 2D...")
    tsne_embedding = pca_mvdata2.get_embedding(method="tsne", dim=2)

    print("\nPipeline complete: 100D → 50D → 10D → 2D")
    print(f"Final embedding shape: {tsne_embedding.coords.shape}")

except ImportError:
    print("\nPyTorch not available. Skipping autoencoder example.")

    # Alternative pipeline without neural methods
    print("\nAlternative pipeline:")
    print("Stage 1: PCA 100D → 30D...")
    pca1 = mvdata.get_embedding(method="pca", dim=30)

    print("Stage 2: Isomap 30D → 5D...")
    isomap = pca1.to_mvdata().get_embedding(method="isomap", dim=5, n_neighbors=20)

    print("Stage 3: UMAP 5D → 2D...")
    final = isomap.to_mvdata().get_embedding(method="umap", dim=2, n_neighbors=15)

    print("\nPipeline complete: 100D → 30D → 5D → 2D")
    print(f"Final embedding shape: {final.coords.shape}")

print("\nKey benefits of to_mvdata():")
print("1. Enables multi-stage dimensionality reduction pipelines")
print("2. Allows mixing linear and non-linear methods")
print("3. Can help with computational efficiency (reduce dimensions gradually)")
print("4. Preserves labels and metadata through the pipeline")
print("5. Each stage can use different algorithms optimized for that scale")
