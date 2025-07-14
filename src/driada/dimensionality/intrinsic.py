"""
Intrinsic dimensionality estimation using nearest neighbor methods.

This module implements methods to estimate the intrinsic dimensionality of datasets
based on the statistics of nearest neighbor distances.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import pynndescent


def nn_dimension(data, k=2, graph_method='sklearn'):
    """
    Estimate intrinsic dimension using the k-NN algorithm.
    
    This method estimates the intrinsic dimensionality by analyzing the ratios
    of distances to the k-th and (k-1)-th nearest neighbors. For k=2, this
    is the TWO-NN algorithm [1].
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data matrix where rows are samples and columns are features.
    
    k : int, default=2
        The number of nearest neighbors to consider. k=2 gives the TWO-NN algorithm.
        Higher values can provide more robust estimates.
    
    graph_method : {'sklearn', 'pynndescent'}, default='sklearn'
        Method to use for nearest neighbor graph construction:
        - 'sklearn': Uses scikit-learn's NearestNeighbors (exact)
        - 'pynndescent': Uses PyNNDescent (approximate, faster for large datasets)
    
    Returns
    -------
    d : float
        The estimated intrinsic dimension of the dataset.
    
    Notes
    -----
    The method is based on the principle that in a d-dimensional manifold,
    the ratio of distances to successive nearest neighbors follows a specific
    distribution that depends on d.
    
    References
    ----------
    [1] Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017).
        Estimating the intrinsic dimension of datasets by a minimal 
        neighborhood information. Scientific Reports, 7(1), 12140.
        https://doi.org/10.1038/s41598-017-11873-y
    
    Examples
    --------
    >>> import numpy as np
    >>> from driada.dimensionality.intrinsic import nn_dimension
    >>> # Generate data on a 2D manifold embedded in 10D
    >>> theta = np.random.uniform(0, 2*np.pi, 1000)
    >>> r = np.random.uniform(0, 1, 1000)
    >>> x = r * np.cos(theta)
    >>> y = r * np.sin(theta)
    >>> data = np.column_stack([x, y] + [np.random.randn(1000)*0.01 for _ in range(8)])
    >>> d_est = nn_dimension(data, k=2)
    >>> print(f"Estimated dimension: {d_est:.2f}")  # Should be close to 2
    """
    data = np.asarray(data)
    n_samples = len(data)
    
    if k < 2:
        raise ValueError("k must be at least 2")
    
    if n_samples <= k:
        raise ValueError(f"Number of samples ({n_samples}) must be greater than k ({k})")
    
    # Compute nearest neighbor distances
    if graph_method == 'sklearn':
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nbrs.fit(data)
        distances, indices = nbrs.kneighbors(data)
    elif graph_method == 'pynndescent':
        index = pynndescent.NNDescent(
            data,
            metric='euclidean',
            n_neighbors=k+1
        )
        indices, distances = index.neighbor_graph
    else:
        raise ValueError(f"Unknown graph construction method: {graph_method}. "
                        "Choose from 'sklearn' or 'pynndescent'.")
    
    # Compute ratios of successive nearest neighbor distances
    # distances[:, 0] is the distance to self (0), so we start from index 1
    ratios = np.zeros((k-1, n_samples))
    for i in range(k-1):
        # Ratio of (i+2)-th neighbor distance to (i+1)-th neighbor distance
        ratios[i, :] = distances[:, i+2] / distances[:, i+1]
    
    # Maximum likelihood estimation for intrinsic dimension
    if k == 2:
        # TWO-NN estimator: equation (3) from [1]
        d = (n_samples - 1) / np.sum(np.log(ratios[0, :]))
    else:
        # Generalized k-NN estimator
        log_sum = 0
        for j in range(k-1):
            log_sum += (j + 1) * np.sum(np.log(ratios[j, :]))
        d = (n_samples * (k - 1) - 1) / log_sum
    
    return d


def correlation_dimension(data, r_min=None, r_max=None, n_bins=20):
    """
    Estimate the correlation dimension using the Grassberger-Procaccia algorithm.
    
    The correlation dimension measures how the number of neighbor pairs scales
    with distance, providing another estimate of intrinsic dimensionality.
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data matrix.
    
    r_min : float, optional
        Minimum distance to consider. If None, automatically determined.
    
    r_max : float, optional
        Maximum distance to consider. If None, automatically determined.
    
    n_bins : int, default=20
        Number of distance bins for the correlation sum.
    
    Returns
    -------
    dimension : float
        The estimated correlation dimension.
    
    References
    ----------
    Grassberger, P., & Procaccia, I. (1983). Characterization of strange 
    attractors. Physical Review Letters, 50(5), 346.
    """
    from scipy.spatial.distance import pdist
    
    # Compute pairwise distances
    distances = pdist(data)
    distances = distances[distances > 0]  # Remove zero distances
    
    if r_min is None:
        r_min = np.percentile(distances, 1)
    if r_max is None:
        r_max = np.percentile(distances, 50)
    
    # Create log-spaced bins
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_bins)
    
    # Compute correlation sum C(r) for each r
    correlation_sum = []
    for r in r_values:
        C_r = np.mean(distances < r)
        if C_r > 0:
            correlation_sum.append(C_r)
        else:
            correlation_sum.append(np.nan)
    
    correlation_sum = np.array(correlation_sum)
    valid = ~np.isnan(correlation_sum) & (correlation_sum > 0)
    
    if np.sum(valid) < 2:
        return np.nan
    
    # Fit line in log-log space
    log_r = np.log(r_values[valid])
    log_C = np.log(correlation_sum[valid])
    
    # Linear regression to find slope (dimension)
    coeffs = np.polyfit(log_r, log_C, 1)
    dimension = coeffs[0]
    
    return dimension