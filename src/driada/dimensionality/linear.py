"""
Linear dimensionality estimation methods.

This module implements dimensionality estimation based on linear methods
such as Principal Component Analysis (PCA).
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import entropy


def pca_dimension(data, threshold=0.95, standardize=True):
    """
    Estimate dimensionality using Principal Component Analysis (PCA).

    This method determines the number of principal components needed to
    explain a specified fraction of the total variance in the data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data matrix where rows are samples and columns are features.

    threshold : float, default=0.95
        The fraction of total variance that should be explained by the
        selected components. Must be between 0 and 1.

    standardize : bool, default=True
        Whether to standardize the data (zero mean, unit variance) before
        applying PCA. Standardization is recommended when features have
        different scales.

    Returns
    -------
    n_components : int
        The number of principal components needed to explain the specified
        fraction of variance.

    Notes
    -----
    This method provides a linear estimate of dimensionality based on variance
    explained. It may overestimate the intrinsic dimension for nonlinear
    manifolds but is useful for understanding the effective linear dimension
    of the data.

    Examples
    --------
    >>> import numpy as np
    >>> from driada.dimensionality import pca_dimension
    >>> # Generate data with 3 effective dimensions
    >>> np.random.seed(42)  # For reproducible results
    >>> data = np.random.randn(1000, 10)
    >>> data[:, 3:] *= 0.1  # Make dimensions 4-10 have small variance
    >>> n_dim = pca_dimension(data, threshold=0.95)
    >>> print(f"Number of components for 95% variance: {n_dim}")
    Number of components for 95% variance: 10
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    data = np.asarray(data)

    if data.shape[0] < 2:
        raise ValueError("Need at least 2 samples to estimate dimensionality")

    # Standardize data if requested
    if standardize:
        # Compute mean and std, handling zero variance features
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)

        # Avoid division by zero for constant features
        data_std[data_std == 0] = 1.0

        data_standardized = (data - data_mean) / data_std
    else:
        data_standardized = data

    # Apply PCA
    pca = PCA()
    pca.fit(data_standardized)

    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find number of components exceeding threshold
    n_components = np.argmax(cumulative_variance >= threshold) + 1

    return n_components


def pca_dimension_profile(data, thresholds=None, standardize=True):
    """
    Compute PCA dimensionality estimates for multiple variance thresholds.

    This function provides a profile of how many components are needed
    for different levels of variance explained, which can help in
    understanding the distribution of variance across components.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data matrix.

    thresholds : array-like, optional
        Variance thresholds to evaluate. If None, uses
        [0.5, 0.8, 0.9, 0.95, 0.99].

    standardize : bool, default=True
        Whether to standardize the data before PCA.

    Returns
    -------
    profile : dict
        Dictionary containing:
        - 'thresholds': array of variance thresholds
        - 'n_components': array of components needed for each threshold
        - 'explained_variance_ratio': variance explained by each component
        - 'cumulative_variance': cumulative variance explained

    Examples
    --------
    >>> np.random.seed(42)  # For reproducible results
    >>> data = np.random.randn(1000, 20)
    >>> profile = pca_dimension_profile(data)
    >>> for thresh, n_comp in zip(profile['thresholds'], profile['n_components']):
    ...     print(f"{thresh*100:.0f}% variance: {n_comp} components")
    50% variance: 9 components
    80% variance: 16 components
    90% variance: 18 components
    95% variance: 19 components
    99% variance: 20 components
    """
    if thresholds is None:
        thresholds = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
    else:
        thresholds = np.asarray(thresholds)

    data = np.asarray(data)

    # Standardize if requested
    if standardize:
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1.0
        data_standardized = (data - data_mean) / data_std
    else:
        data_standardized = data

    # Apply PCA
    pca = PCA()
    pca.fit(data_standardized)

    # Compute cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find components needed for each threshold
    n_components = []
    for threshold in thresholds:
        if threshold > cumulative_variance[-1]:
            # Can't reach this threshold
            n_comp = len(cumulative_variance)
        else:
            n_comp = np.argmax(cumulative_variance >= threshold) + 1
        n_components.append(n_comp)

    return {
        "thresholds": thresholds,
        "n_components": np.array(n_components),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumulative_variance,
    }


def effective_rank(data, standardize=True):
    """
    Compute the effective rank (Roy & Vetterli, 2007) of the data matrix.

    The effective rank is a continuous measure of dimensionality based on
    the entropy of the normalized eigenvalue distribution.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data matrix.

    standardize : bool, default=True
        Whether to standardize the data before computation.

    Returns
    -------
    eff_rank : float
        The effective rank of the data matrix.

    References
    ----------
    Roy, O., & Vetterli, M. (2007). The effective rank: A measure of
    effective dimensionality. In 2007 15th European Signal Processing
    Conference (pp. 606-610). IEEE.
    
    Examples
    --------
    >>> import numpy as np
    >>> from driada.dimensionality import effective_rank
    >>> # Full rank matrix
    >>> np.random.seed(42)  # For reproducible results
    >>> data = np.random.randn(100, 10)
    >>> eff_r = effective_rank(data)
    >>> print(f"Effective rank: {eff_r:.2f}")
    Effective rank: 9.88
    """
    data = np.asarray(data)

    # Standardize if requested
    if standardize:
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1.0
        data_standardized = (data - data_mean) / data_std
    else:
        data_standardized = data

    # Compute singular values
    _, s, _ = np.linalg.svd(data_standardized, full_matrices=False)
    
    # Remove negligible singular values to avoid numerical issues
    tolerance = np.finfo(float).eps * max(data.shape) * s[0] if len(s) > 0 else 0
    s_positive = s[s > tolerance]
    
    if len(s_positive) == 0:
        return 1.0  # Degenerate case

    # Normalize to get probability distribution
    s_normalized = s_positive / np.sum(s_positive)

    # Compute entropy (base e)
    ent = entropy(s_normalized)

    # Convert to effective rank
    eff_rank = np.exp(ent)

    return eff_rank
