"""
Intrinsic dimensionality estimation using nearest neighbor methods.

This module implements methods to estimate the intrinsic dimensionality of datasets
based on the statistics of nearest neighbor distances.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import pynndescent


def nn_dimension(data=None, k=2, graph_method="sklearn", precomputed_graph=None):
    """
    Estimate intrinsic dimension using the k-NN algorithm.

    This method estimates the intrinsic dimensionality by analyzing the ratios
    of distances to successive nearest neighbors using maximum likelihood 
    estimation. When k=2, this implements the TWO-NN algorithm [1], where 
    "TWO" refers to using the 1st and 2nd nearest neighbors to compute 
    distance ratios.

    The algorithm works by:
    1. For each point, computing the ratio r_i = d_2 / d_1 (for k=2), where 
       d_1 and d_2 are distances to the 1st and 2nd nearest neighbors
    2. Using maximum likelihood to estimate dimension from the distribution 
       of these ratios, which follows a specific form dependent on the 
       intrinsic dimension
    3. For k>2, extending to use multiple successive neighbor ratios 
       (d_2/d_1, d_3/d_2, ..., d_k/d_{k-1}) for more robust estimation

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features), optional
        The input data matrix where rows are samples and columns are features.
        Either data or precomputed_graph must be provided.

    k : int, default=2
        The number of nearest neighbors to consider. k=2 gives the TWO-NN algorithm.
        Higher values can provide more robust estimates.

    graph_method : {'sklearn', 'pynndescent'}, default='sklearn'
        Method to use for nearest neighbor graph construction:
        - 'sklearn': Uses scikit-learn's NearestNeighbors (exact)
        - 'pynndescent': Uses PyNNDescent (approximate, faster for large datasets)
        Only used if data is provided.

    precomputed_graph : tuple of (indices, distances), optional
        Precomputed k-NN graph as a tuple of:
        - indices: array of shape (n_samples, k+1) with neighbor indices
        - distances: array of shape (n_samples, k+1) with neighbor distances
        The first column (index 0) should contain self-references with distance 0.
        Either data or precomputed_graph must be provided.

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
    >>> 
    >>> # Example 1: Clean low-dimensional data
    >>> np.random.seed(42)
    >>> # Generate 2D Swiss roll
    >>> n = 1000
    >>> t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))  
    >>> height = 30 * np.random.rand(n)
    >>> X = np.zeros((n, 3))
    >>> X[:, 0] = t * np.cos(t)
    >>> X[:, 1] = height
    >>> X[:, 2] = t * np.sin(t)
    >>> d_est = nn_dimension(X, k=10)
    >>> print(f"Swiss roll (true dim=2): {d_est:.2f}")
    Swiss roll (true dim=2): 1.94
    >>> 
    >>> # Example 2: Impact of ambient noise
    >>> # The TWO-NN estimator (k=2) can overestimate dimension when data
    >>> # has noise in ambient dimensions. This is because noise affects
    >>> # the ratios of nearest neighbor distances.
    >>> 
    >>> # Pure 2D data: accurate estimate
    >>> theta = np.random.uniform(0, 2*np.pi, 1000)
    >>> r = np.sqrt(np.random.uniform(0, 1, 1000))  # Uniform distribution on disk
    >>> data_2d = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    >>> d_est = nn_dimension(data_2d, k=2)
    >>> print(f"Pure 2D disk: {d_est:.2f}")
    Pure 2D disk: 2.01
    >>> 
    >>> # Same data with small noise in extra dimensions: overestimates
    >>> # This happens because even small noise changes neighbor relationships
    >>> noise = 0.01 * np.random.randn(1000, 8)
    >>> data_10d = np.hstack([data_2d, noise])
    >>> d_est = nn_dimension(data_10d, k=2)
    >>> print(f"2D disk in 10D with noise: {d_est:.2f}")  # Overestimates due to noise
    2D disk in 10D with noise: 4.73
    >>> 
    >>> # For noisy high-dimensional data:
    >>> # 1. Use higher k values (more robust but may underestimate)
    >>> # 2. Consider denoising or PCA preprocessing
    >>> # 3. Use alternative methods like correlation_dimension
    
    Raises
    ------
    ValueError
        If inputs are invalid, sample size too small, or duplicate points exist.    """
    # Validate inputs
    if data is None and precomputed_graph is None:
        raise ValueError("Either data or precomputed_graph must be provided")

    if data is not None and precomputed_graph is not None:
        raise ValueError("Provide either data or precomputed_graph, not both")

    if k < 2:
        raise ValueError("k must be at least 2")

    # Get nearest neighbor distances
    if precomputed_graph is not None:
        # Use precomputed graph
        indices, distances = precomputed_graph
        indices = np.asarray(indices)
        distances = np.asarray(distances)
        n_samples = len(indices)

        # Validate precomputed graph
        if indices.shape != distances.shape:
            raise ValueError("Indices and distances must have the same shape")

        if indices.shape[1] < k + 1:
            raise ValueError(
                f"Precomputed graph must have at least k+1={k+1} neighbors, "
                f"but has {indices.shape[1]}"
            )

        # Ensure we have enough columns
        distances = distances[:, : k + 1]
        indices = indices[:, : k + 1]

    else:
        # Compute from data
        data = np.asarray(data)
        n_samples = len(data)

        if n_samples <= k:
            raise ValueError(
                f"Number of samples ({n_samples}) must be greater than k ({k})"
            )

        # Compute nearest neighbor distances
        if graph_method == "sklearn":
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
            nbrs.fit(data)
            distances, indices = nbrs.kneighbors(data)
        elif graph_method == "pynndescent":
            index = pynndescent.NNDescent(data, metric="euclidean", n_neighbors=k + 1)
            indices, distances = index.neighbor_graph
        else:
            raise ValueError(
                f"Unknown graph construction method: {graph_method}. "
                "Choose from 'sklearn' or 'pynndescent'."
            )

    # Compute ratios of successive nearest neighbor distances
    # distances[:, 0] is the distance to self (0), so we start from index 1
    ratios = np.zeros((k - 1, n_samples))
    for i in range(k - 1):
        # Ratio of (i+2)-th neighbor distance to (i+1)-th neighbor distance
        # Check for zero distances to avoid division by zero
        denom = distances[:, i + 1]
        if np.any(denom == 0):
            n_zero = np.sum(denom == 0)
            raise ValueError(
                f"Found {n_zero} zero distances to {i+1}-th nearest neighbor. "
                "This typically indicates duplicate points in the dataset. "
                "Consider removing duplicates or adding small noise."
            )
        ratios[i, :] = distances[:, i + 2] / denom

    # Maximum likelihood estimation for intrinsic dimension
    if k == 2:
        # TWO-NN estimator: equation (3) from [1]
        d = (n_samples - 1) / np.sum(np.log(ratios[0, :]))
    else:
        # Generalized k-NN estimator
        log_sum = 0
        for j in range(k - 1):
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
        The input data matrix. For large datasets, consider subsampling as
        this method requires O(n²) memory.

    r_min : float, optional
        Minimum distance to consider. If None, uses 1st percentile of distances.

    r_max : float, optional
        Maximum distance to consider. If None, uses 50th percentile of distances.

    n_bins : int, default=20
        Number of distance bins for the correlation sum.

    Returns
    -------
    dimension : float
        The estimated correlation dimension. Returns NaN if estimation fails.
        
    Raises
    ------
    ValueError
        If data has fewer than 2 samples or invalid parameters.
        
    Notes
    -----
    This method has O(n²) memory complexity. For datasets with more than
    10,000 samples, consider using a subsample or alternative methods.

    References
    ----------
    Grassberger, P., & Procaccia, I. (1983). Characterization of strange
    attractors. Physical Review Letters, 50(5), 346.
    
    Examples
    --------
    >>> import numpy as np
    >>> from driada.dimensionality import correlation_dimension
    >>> 
    >>> # Generate points on a 2D circle embedded in 3D
    >>> t = np.linspace(0, 2*np.pi, 500)
    >>> data = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    >>> d_est = correlation_dimension(data, n_bins=15)
    >>> print(f"Correlation dimension: {d_est:.2f}")  # Should be close to 1
    Correlation dimension: 1.07
    >>> 
    >>> # For noisy data, specify scaling range
    >>> np.random.seed(42)  # For reproducible results
    >>> noisy_data = data + 0.01 * np.random.randn(*data.shape)
    >>> d_est = correlation_dimension(noisy_data, r_min=0.05, r_max=0.5)
    >>> print(f"Correlation dimension: {d_est:.2f}")
    Correlation dimension: 1.09
    """
    from scipy.spatial.distance import pdist
    
    data = np.asarray(data)
    n_samples = len(data)
    
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}")
    
    # Warn about memory usage for large datasets
    if n_samples > 10000:
        import warnings
        warnings.warn(
            f"Computing pairwise distances for {n_samples} samples requires "
            f"~{n_samples * (n_samples - 1) * 8 / 2e9:.1f} GB of memory. "
            "Consider using a subsample for large datasets.",
            RuntimeWarning
        )

    # Compute pairwise distances
    distances = pdist(data)
    distances = distances[distances > 0]  # Remove zero distances
    
    if len(distances) == 0:
        raise ValueError("All points are identical")

    if r_min is None:
        r_min = np.percentile(distances, 1)
    if r_max is None:
        r_max = np.percentile(distances, 50)
    
    if r_min <= 0:
        raise ValueError(f"r_min must be positive, got {r_min}")
    if r_max <= r_min:
        raise ValueError(f"r_max ({r_max}) must be greater than r_min ({r_min})")

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


def geodesic_dimension(
    data=None, graph=None, k=15, mode="full", factor=2, dim_step=0.1
):
    """
    Estimate intrinsic dimension using geodesic distances (De Granata et al. 2016).

    This method estimates the intrinsic dimensionality by analyzing the distribution
    of geodesic distances computed as shortest paths on a k-nearest neighbor graph.
    The method focuses on the behavior of the distance distribution around its maximum.

    Warning
    -------
    This method is sensitive to the k parameter. Small k values (k < 15) often lead
    to underestimation due to disconnected or sparse graphs that poorly approximate
    geodesic distances. For reliable estimates, use k ≥ 15-25, with larger values
    needed for complex manifolds like Swiss rolls or S-curves.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features), optional
        The input data matrix. Either data or graph must be provided.

    graph : sparse matrix of shape (n_samples, n_samples), optional
        Precomputed graph adjacency matrix with edge weights as distances.
        If provided, data is ignored. Either data or graph must be provided.

    k : int, default=15
        Number of nearest neighbors for graph construction (if data provided).

    mode : {'full', 'fast'}, default='full'
        Computation mode:
        - 'full': Use all pairwise geodesic distances
        - 'fast': Use a random subset (1/factor of points)

    factor : int, default=2
        Subsampling factor for 'fast' mode. Uses n/factor random points.

    dim_step : float, default=0.1
        Step size for dimension grid search. Smaller values give more precise
        estimates but take longer to compute.

    Returns
    -------
    dimension : float
        The estimated intrinsic dimension of the dataset/manifold.

    Notes
    -----
    This method implements the approach from:
    Granata, D., Carnevale, V. (2016). Accurate Estimation of the Intrinsic
    Dimension Using Graph Distances: Unraveling the Geometric Complexity of
    Datasets. Scientific Reports, 6, 31377.

    The method is particularly robust for complex manifold geometries and can
    handle small sample sizes effectively.

    References
    ----------
    [1] Granata, D., Carnevale, V. (2016). Accurate Estimation of the Intrinsic
        Dimension Using Graph Distances: Unraveling the Geometric Complexity of
        Datasets. Scientific Reports, 6, 31377.
        https://doi.org/10.1038/srep31377

    Examples
    --------
    >>> import numpy as np
    >>> from driada.dimensionality.intrinsic import geodesic_dimension
    >>> # Generate 2D Swiss roll
    >>> from sklearn.datasets import make_swiss_roll
    >>> np.random.seed(42)  # For reproducible results
    >>> data, _ = make_swiss_roll(n_samples=1000, noise=0.05, random_state=42)
    >>> d_est = geodesic_dimension(data, k=15)
    >>> print(f"Estimated dimension: {d_est:.2f}")  # Should be close to 2
    Estimated dimension: 1.90
    
    Raises
    ------
    ValueError
        If inputs are invalid or graph is too disconnected.
    RuntimeWarning
        If graph is disconnected (infinite distances found).    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import kneighbors_graph

    if data is None and graph is None:
        raise ValueError("Either data or graph must be provided")

    if data is not None and graph is not None:
        raise ValueError("Provide either data or graph, not both")

    # Construct graph if data provided
    if data is not None:
        data = np.asarray(data)
        n_samples = len(data)

        # Build k-NN graph with distances
        graph = kneighbors_graph(
            data, n_neighbors=k, mode="distance", include_self=False
        )
        # Make symmetric by taking minimum distance
        graph_min = graph.minimum(graph.T)
        graph = sp.csr_matrix(graph_min)

    else:
        # Use provided graph
        graph = sp.csr_matrix(graph)
        n_samples = graph.shape[0]

    # Subsample for fast mode
    if mode == "fast":
        indices = np.random.permutation(n_samples)[: n_samples // factor]
        dm = graph[indices, :][:, indices]
    else:
        dm = graph

    # Compute shortest paths (geodesic distances)
    spmatrix = shortest_path(dm, method="D", directed=False)
    all_dists = spmatrix.flatten()
    all_dists = all_dists[all_dists != 0]  # Remove self-distances

    # Check for disconnected graph
    n_infinite = np.sum(np.isinf(all_dists))
    if n_infinite > 0:
        import warnings

        warnings.warn(
            f"Graph appears to be disconnected. Found {n_infinite} infinite distances. "
            f"These will be excluded from analysis.",
            RuntimeWarning,
        )

    all_dists = all_dists[np.isfinite(all_dists)]  # Remove infinite distances

    # Check if we have enough finite distances
    if len(all_dists) < 100:
        raise ValueError(
            f"Not enough finite distances for analysis. Got {len(all_dists)} distances. "
            "Graph may be too disconnected."
        )

    # Analyze distance distribution
    nbins = 500
    hist, bin_edges = np.histogram(all_dists, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find maximum of distribution
    dmax_idx = np.argmax(hist)
    dmax = bin_centers[dmax_idx]
    
    if dmax == 0:
        raise ValueError(
            "Maximum of distance distribution is at 0. "
            "This may indicate all points are identical or graph is degenerate."
        )

    # Normalize distances by maximum
    hist_norm, bin_edges_norm = np.histogram(all_dists / dmax, bins=nbins, density=True)
    bin_centers_norm = (bin_edges_norm[:-1] + bin_edges_norm[1:]) / 2

    # Analyze left side of distribution near maximum
    std_norm = np.std(all_dists / dmax)
    mask = (
        (bin_centers_norm > 1 - 2 * std_norm)
        & (bin_centers_norm <= 1)
        & (hist_norm > 1e-6)
    )
    x_left = bin_centers_norm[mask]
    y_left = np.log(hist_norm[mask] / np.max(hist_norm))  # Natural log, not log10

    # Fit dimension by minimizing error against theoretical distribution
    def theoretical_dist(x, D):
        """Theoretical distribution for D-dimensional hypersphere.

        Based on Granata & Carnevale 2016 paper, the distribution of geodesic
        distances near the maximum follows: (D-1) * log(sin(x * pi/2))
        where x is the normalized distance and D is the intrinsic dimension.
        
        The paper shows that for a D-dimensional hypersphere, the probability
        distribution is proportional to sin^(D-1)(π*r/(2*r_max)), which gives
        log P(r) ~ (D-1) * log(sin(π*r/(2*r_max))) when taking the logarithm.
        
        Parameters
        ----------
        x : array_like
            Normalized distances (should be in range [0, 1]).
        D : float
            Candidate intrinsic dimension.
            
        Returns
        -------
        array_like
            Log probability values for the theoretical distribution.        """
        # Clip x to valid domain for sin
        x_clipped = np.clip(x, 1e-10, 1 - 1e-10)
        return (D - 1) * np.log(np.sin(x_clipped * np.pi / 2))

    # Grid search for best dimension
    dimensions = np.arange(1.0, 26.0, dim_step)
    errors = []

    for D in dimensions:
        y_theory = theoretical_dist(x_left, D)
        error = np.linalg.norm(y_theory - y_left) / np.sqrt(len(y_left))
        errors.append(error)

    # Find dimension with minimum error
    best_idx = np.argmin(errors)
    dimension = dimensions[best_idx]

    return dimension
