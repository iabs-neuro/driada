# Dimensionality Reduction Module

## Overview

The dimensionality reduction module provides a comprehensive suite of algorithms for extracting low-dimensional representations from high-dimensional neural population data. It supports both linear and nonlinear methods with a unified interface.

## Key components

### MVData class
The core data container that `MultiTimeSeries` inherits from. Provides:
- Automatic downsampling
- Preprocessing (normalization, standardization)
- Direct access to dimensionality reduction methods via `get_embedding()`

### Supported methods

**Linear methods:**
- PCA (Principal Component Analysis)
- FA (Factor Analysis)
- LLE (Locally Linear Embedding)
- LEM (Laplacian Eigenmaps)
- DM (Diffusion Maps)

**Nonlinear methods:**
- Isomap (Isometric Mapping)
- UMAP (Uniform Manifold Approximation and Projection)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Autoencoders (standard and variational)
- MVU (Maximum Variance Unfolding)

### Manifold quality metrics
- `knn_preservation_rate()` - Proportion of neighbors preserved
- `trustworthiness()` - Are close points in low-D truly close in high-D?
- `continuity()` - Are close points in high-D still close in low-D?
- `procrustes_analysis()` - Optimal alignment between embeddings
- `circular_structure_preservation()` - For ring-like manifolds

## Example usage

```python
from driada.dim_reduction import MVData

# Create MVData object (or use experiment.calcium directly)
mvdata = MVData(neural_data, downsampling=5)

# Apply dimensionality reduction
embedding = mvdata.get_embedding(method='umap', n_components=2, n_neighbors=30)

# Access coordinates
coords = embedding.coords.T  # Shape: (n_samples, n_dims)

# Evaluate quality
from driada.dim_reduction import knn_preservation_rate
quality = knn_preservation_rate(neural_data, coords, k=10)
```

## Advanced features

### Sequential reduction
Chain multiple methods for optimal results:
```python
from driada.dim_reduction import dr_sequence

result = dr_sequence(
    mvdata,
    steps=[
        ('pca', {'dim': 50}),    # Initial denoising
        ('umap', {'dim': 2})     # Final embedding
    ]
)
```

### Graph-based methods
Graph-based DR methods (Isomap, LLE, Laplacian Eigenmaps, diffusion maps, UMAP) construct a `ProximityGraph` internally. `ProximityGraph` inherits from `Network` (`driada.network`), so the resulting graph has full spectral analysis, entropy, community detection, and visualization capabilities. Access it via `embedding.graph` after running a graph-based embedding.