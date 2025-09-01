# Dimensionality Estimation Module

## Overview

This module provides methods for estimating the intrinsic dimensionality of neural data. Understanding the true dimensionality helps determine how many components are needed for dimensionality reduction and reveals the complexity of neural representations.

## Methods

### Linear Dimensionality
- `pca_dimension()` - Number of PCs needed to capture a threshold of variance
- `pca_dimension_profile()` - Full variance explained curve
- `effective_rank()` - Based on eigenvalue entropy

### Effective Dimensionality
- `eff_dim()` - Participation ratio using Rényi entropy
  - Supports finite-sample correction for small datasets
  - Configurable entropy order (q parameter)

### Intrinsic Dimensionality
- `nn_dimension()` - k-nearest neighbor based estimation
- `correlation_dimension()` - Grassberger-Procaccia algorithm
- `geodesic_dimension()` - Based on geodesic distances

## Example Usage

```python
from driada.dimensionality import eff_dim, pca_dimension, nn_dimension

# Standard format: (n_samples, n_features)
neural_data = exp.calcium.scdata.T

# Linear methods - fast and interpretable
pca_90 = pca_dimension(neural_data, threshold=0.90)
print(f"Need {pca_90} PCs for 90% variance")

# Effective dimension - robust measure
eff_d = eff_dim(neural_data, enable_correction=True, q=2)
print(f"Effective dimension: {eff_d:.2f}")

# Intrinsic dimension - captures nonlinear structure
intrinsic_d = nn_dimension(neural_data, k=5)
print(f"Intrinsic dimension: {intrinsic_d:.2f}")
```

## When to Use Each Method

- **PCA dimension**: Quick assessment, linear relationships only
- **Effective dimension**: Robust to noise, good general-purpose metric
- **k-NN dimension**: When data lies on nonlinear manifold
- **Correlation dimension**: For fractal or self-similar data

## Technical Notes

All methods expect data in scikit-learn format: `(n_samples, n_features)` where:
- Rows are samples (e.g., time points)
- Columns are features (e.g., neurons)