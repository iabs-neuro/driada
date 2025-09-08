"""K-nearest neighbor estimators for mutual information and entropy.

This module implements the Kraskov-Stögbauer-Grassberger (KSG) estimator
and related k-NN based information theoretic measures.

Important Notes:
    - The Local Non-uniformity Correction (LNC) can be unstable when k <= d
      (where k is the number of neighbors and d is the total dimensionality).
      In such cases, LNC is automatically disabled to prevent numerical errors.
    - For best results, use k > d, with k >= d+2 recommended.
    - Common good values: k=5 for d<=3, k=10 for d=4-6, k=20 for d>6.

Credits:
    Original implementation by Greg Ver Steeg
    http://www.isi.edu/~gregv/npeet.html
    
References:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information. Physical Review E, 69(6), 066138.
"""

import numpy as np
import numpy.linalg as la
from numpy import log
from sklearn.neighbors import BallTree, KDTree

from .info_utils import py_fast_digamma

DEFAULT_NN = 5
# UTILITY FUNCTIONS

# Alpha selection for LNC correction based on k and dimensionality
# Values from https://github.com/BiuBiuBiLL/NPEET_LNC/blob/master/alpha.xlsx

# Alpha lookup table: (k, d) -> alpha
# k: number of nearest neighbors
# d: dimensionality
ALPHA_LNC_TABLE = {
    # k=2
    (2, 3): 0.182224, (2, 4): 0.284370, (2, 5): 0.372004, (2, 6): 0.442894,
    (2, 7): 0.503244, (2, 8): 0.554523, (2, 9): 0.594569, (2, 10): 0.630903,
    (2, 11): 0.660295, (2, 12): 0.689290, (2, 13): 0.711052, (2, 14): 0.735075,
    (2, 15): 0.751908, (2, 16): 0.767809, (2, 17): 0.782448, (2, 18): 0.795362,
    (2, 19): 0.806728, (2, 20): 0.817252,
    # k=3
    (3, 4): 0.077830, (3, 5): 0.167277, (3, 6): 0.250141, (3, 7): 0.320280,
    (3, 8): 0.384474, (3, 9): 0.441996, (3, 10): 0.489972, (3, 11): 0.532178,
    (3, 12): 0.568561, (3, 13): 0.603990, (3, 14): 0.636593, (3, 15): 0.660156,
    (3, 16): 0.683954, (3, 17): 0.706157, (3, 18): 0.724844, (3, 19): 0.743606,
    (3, 20): 0.757283,
    # k=5
    (5, 6): 0.023953, (5, 7): 0.067077, (5, 8): 0.123341, (5, 9): 0.180215,
    (5, 10): 0.239442, (5, 11): 0.297637, (5, 12): 0.351355, (5, 13): 0.404194,
    (5, 14): 0.451739, (5, 15): 0.498458, (5, 16): 0.538889, (5, 17): 0.578158,
    (5, 18): 0.614937, (5, 19): 0.651598, (5, 20): 0.679500,
    # k=10
    (10, 11): 0.003734, (10, 12): 0.014748, (10, 13): 0.034749, (10, 14): 0.063109,
    (10, 15): 0.100471, (10, 16): 0.147694, (10, 17): 0.200196, (10, 18): 0.261374,
    (10, 19): 0.325363, (10, 20): 0.398082,
}


def get_lnc_alpha(k, d):
    """Get optimal alpha value for LNC correction based on k and dimensionality.
    
    Parameters
    ----------
    k : int
        Number of nearest neighbors. Must be positive.
    d : int
        Dimensionality of the data. Must be positive.
        
    Returns
    -------
    float
        Alpha value for LNC correction. Returns 0.25 as default if no 
        suitable value found in the lookup table.
        
    Raises
    ------
    ValueError
        If k or d are not positive integers.
        
    Notes
    -----
    Values are based on the lookup table from:
    https://github.com/BiuBiuBiLL/NPEET_LNC
    
    For (k, d) pairs not in the table:
    - If k is not in {2, 3, 5, 10}, uses nearest available k
    - If d is outside available range, uses nearest available d    """
    # Validate inputs
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if not isinstance(d, (int, np.integer)) or d <= 0:
        raise ValueError(f"d must be a positive integer, got {d}")
    
    # Get available k values
    available_k = sorted(set(k_val for k_val, _ in ALPHA_LNC_TABLE.keys()))
    
    # Find closest k
    if k in available_k:
        k_use = k
    else:
        # Find nearest k
        k_use = min(available_k, key=lambda x: abs(x - k))
    
    # Get available d values for this k
    available_d = sorted([d_val for k_val, d_val in ALPHA_LNC_TABLE.keys() if k_val == k_use])
    
    if not available_d:
        # No data for this k, use default
        return 0.25  # Default from original implementation
    
    # Find appropriate d
    if d <= available_d[0]:
        # Use smallest available d
        d_use = available_d[0]
    elif d >= available_d[-1]:
        # Use largest available d
        d_use = available_d[-1]
    elif d in available_d:
        # Exact match
        d_use = d
    else:
        # Interpolate between adjacent values
        d_lower = max(d_val for d_val in available_d if d_val < d)
        d_upper = min(d_val for d_val in available_d if d_val > d)
        
        alpha_lower = ALPHA_LNC_TABLE.get((k_use, d_lower), 0)
        alpha_upper = ALPHA_LNC_TABLE.get((k_use, d_upper), 0)
        
        # Linear interpolation
        weight = (d - d_lower) / (d_upper - d_lower)
        return alpha_lower + weight * (alpha_upper - alpha_lower)
    
    return ALPHA_LNC_TABLE.get((k_use, d_use), 0.25)


def add_noise(x, ampl=1e-10):
    """Add small random noise to data to break degeneracy.
    
    When multiple data points have identical values, k-nearest neighbor 
    algorithms can become unstable. Adding small random noise helps break
    these ties without significantly affecting the mutual information estimate.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input data array to add noise to. Must contain finite values.
    ampl : float, optional
        Amplitude of the noise to add. Default is 1e-10, which is small
        enough to not affect the MI estimate but large enough to break ties.
        Must be non-negative.
        
    Returns
    -------
    numpy.ndarray
        Input data with small random noise added. Same shape as input.
        Noise is uniformly distributed in [0, ampl).
        
    Raises
    ------
    ValueError
        If ampl is negative or if x contains non-finite values.
        
    Notes
    -----
    The noise is uniformly distributed in [0, ampl). This is a standard
    technique in KSG mutual information estimation to handle degenerate
    cases where many points have identical coordinates.
    
    Uses numpy's global random state. For reproducible results, set the
    random seed before calling this function.
    
    References
    ----------
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical Review E, 69(6), 066138.    """
    # Validate inputs
    x = np.asarray(x)
    if ampl < 0:
        raise ValueError(f"ampl must be non-negative, got {ampl}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    
    # small noise to break degeneracy, see doc.
    return x + ampl * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    """Query k-th nearest neighbor distances for each point.
    
    Parameters
    ----------
    tree : BallTree or KDTree
        Pre-built spatial index tree for efficient neighbor queries.
    x : ndarray of shape (n_samples, n_features)
        Query points to find neighbors for.
    k : int
        Which neighbor distance to return (k-th nearest). Must be positive.
        
    Returns
    -------
    ndarray of shape (n_samples,)
        Distance to the k-th nearest neighbor for each query point.
        Note: k+1 is queried to exclude the point itself.
        
    Raises
    ------
    ValueError
        If k is not a positive integer.    """
    # Validate inputs
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # return tree.query(x, k=k+1, breadth_first = False)[0][:, k]
    return tree.query(x, k=k + 1)[0][:, k]


def _count_neighbors_single(tree, x, radii, ind):
    """Count neighbors within radius for a single point (legacy function).
    
    Parameters
    ----------
    tree : BallTree or KDTree
        Spatial index tree.
    x : ndarray
        All data points.
    radii : ndarray
        Search radius for each point.
    ind : int
        Index of the point to query. Must be valid index into x.
        
    Returns
    -------
    int
        Number of neighbors within radius (excluding self).
        
    Raises
    ------
    ValueError
        If ind is not a valid index or if radii is too short.
        
    Notes
    -----
    This is a legacy single-point implementation. Use count_neighbors
    for efficient vectorized computation.    """
    # Validate inputs
    x = np.asarray(x)
    radii = np.asarray(radii)
    if not isinstance(ind, (int, np.integer)) or ind < 0 or ind >= len(x):
        raise ValueError(f"ind must be valid index in [0, {len(x)}), got {ind}")
    if ind >= len(radii):
        raise ValueError(f"radii must have at least {ind+1} elements")
    
    dists, indices = tree.query(
        x[ind : ind + 1], k=DEFAULT_NN, distance_upper_bound=radii[ind]
    )
    # Fixed: subtract 1 for self-exclusion, not 2
    return len(np.unique(indices[0])) - 1


def count_neighbors(tree, x, radii):
    """Count neighbors within given radius for each point.
    
    Parameters
    ----------
    tree : BallTree or KDTree
        Pre-built spatial index tree for efficient radius queries.
    x : ndarray of shape (n_samples, n_features)
        Query points.
    radii : ndarray of shape (n_samples,)
        Search radius for each query point.
        
    Returns
    -------
    ndarray of shape (n_samples,)
        Number of neighbors within radius for each point (including self).
        
    Raises
    ------
    ValueError
        If x and radii have different lengths.
        
    Notes
    -----
    Uses efficient vectorized radius query. The count includes the
    query point itself, so subtract 1 if self-exclusion is needed.    """
    # Validate inputs
    x = np.asarray(x)
    radii = np.asarray(radii)
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    if len(x) != len(radii):
        raise ValueError(f"x and radii must have same length, got {len(x)} and {len(radii)}")
    
    return tree.query_radius(x, radii, count_only=True)


def build_tree(points, lf=5):
    """Build spatial index tree for k-NN queries.
    
    Automatically selects between KDTree and BallTree based on
    dimensionality for optimal performance.
    
    Parameters
    ----------
    points : ndarray of shape (n_samples, n_features)
        Data points to index. Must have at least 1 sample and 1 feature.
    lf : int, optional
        Leaf size parameter for tree construction. Smaller values
        lead to deeper trees with faster queries but slower construction.
        Default is 5. Must be positive.
        
    Returns
    -------
    BallTree or KDTree
        Spatial index tree using Chebyshev (max) metric.
        
    Raises
    ------
    ValueError
        If points is not 2D, is empty, or if lf is not positive.
        
    Notes
    -----
    - Uses BallTree for high dimensions (>=20) as KDTree performance
      degrades exponentially with dimension.
    - Chebyshev metric is used for compatibility with KSG estimator.    """
    # Validate inputs
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError(f"points must be 2D array, got shape {points.shape}")
    if points.size == 0:
        raise ValueError("points cannot be empty")
    if not isinstance(lf, (int, np.integer)) or lf <= 0:
        raise ValueError(f"lf must be positive integer, got {lf}")
    
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")

    return KDTree(points, metric="chebyshev", leaf_size=lf)
    # return KDTree(points, leafsize = lf)
    # return KDTree(points, copy_data=True, leafsize = 5)


def avgdigamma(points, dvec, lf=30, tree=None):
    """Compute average digamma of neighbor counts within given radii.
    
    Used in KSG mutual information estimation to compute the average
    logarithmic correction term based on neighbor counts in marginal spaces.
    
    Parameters
    ----------
    points : ndarray of shape (n_samples, n_features)
        Data points in the marginal space.
    dvec : ndarray of shape (n_samples,)
        Distance to k-th neighbor in the joint space, used as radius
        for counting neighbors in this marginal space.
    lf : int, optional
        Leaf size for tree construction if tree not provided. Default is 30.
    tree : BallTree or KDTree, optional
        Pre-built tree. If None, a new tree is constructed.
        
    Returns
    -------
    float
        Average of digamma(neighbor_count) across all points.
        
    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid values.
    Exception
        If more than 1% of points have no neighbors within their radius,
        indicating potential issues with the data or parameters.
        
    Notes
    -----
    - Subtracts small epsilon (1e-15) from radii to handle boundary cases
    - Points with zero neighbors are assigned count of 0.5 to avoid
      numerical issues with digamma function    """
    # Validate inputs
    points = np.asarray(points)
    dvec = np.asarray(dvec)
    
    if points.ndim != 2:
        raise ValueError(f"points must be 2D array, got shape {points.shape}")
    if dvec.ndim != 1:
        raise ValueError(f"dvec must be 1D array, got shape {dvec.shape}")
    if len(points) != len(dvec):
        raise ValueError(f"points and dvec must have same length, got {len(points)} and {len(dvec)}")
    if not isinstance(lf, (int, np.integer)) or lf <= 0:
        raise ValueError(f"lf must be positive integer, got {lf}")
    
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    if tree is None:
        tree = build_tree(points, lf=lf)

    # Create copy to avoid modifying input
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    num_points = num_points.astype(float)

    zero_inds = np.where(num_points == 0)[0]
    if 1.0 * len(zero_inds) / len(num_points) > 0.01:
        raise Exception("No neighbours in more than 1% points, check input!")
    else:
        if len(zero_inds) != 0:
            num_points[zero_inds] = 0.5

    # inf_inds = np.where(digamma(num_points) == -np.inf)
    # print(num_points[inf_inds])

    digammas = list(map(py_fast_digamma, num_points))
    return np.mean(digammas)


# CONTINUOUS ESTIMATORS


def nonparam_entropy_c(x, k=DEFAULT_NN, base=np.e):
    """The classic Kozachenko-Leonenko k-nearest neighbor continuous entropy estimator.
    
    Estimates differential entropy for continuous variables using k-nearest
    neighbor distances. This is the foundation for KSG mutual information
    estimation.
    
    Parameters
    ----------
    x : array-like
        Continuous data, shape (n_samples,) for 1D or (n_samples, n_features)
        for multivariate. Each row is a sample, columns are features.
    k : int, default=5
        Number of nearest neighbors to use. Common values:
        - k = 4-5 for most applications (optimal bias-variance tradeoff)
        - k = 3-10 for low dimensions (d ≤ 3)
        - k = 10-20 for higher dimensions (d > 3)
        Must satisfy k < n_samples. Higher k reduces variance but increases bias.
    base : float, default=np.e
        Logarithm base for entropy calculation. Use np.e for nats,
        2 for bits, or 10 for dits.
        
    Returns
    -------
    float
        Differential entropy estimate in units determined by base.
        Can be negative for continuous distributions.
        
    Raises
    ------
    ValueError
        If x is empty, k is invalid, or base is not positive.
        
    Notes
    -----
    The Kozachenko-Leonenko estimator is:
    H(X) = ψ(n) - ψ(k) + d*log(2) + d*<log(ε_k)>
    
    where:
    - ψ is the digamma function
    - n is the number of samples
    - d is the dimensionality
    - ε_k is the distance to the k-th nearest neighbor
    - <·> denotes average over all samples
    
    Small noise is added to break ties for discrete-valued continuous data.
    
    References
    ----------
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical Review E, 69(6), 066138. (Recommends k ≈ 4)
    
    Examples
    --------
    >>> # Entropy of standard normal (theoretical: 0.5*log(2πe) ≈ 1.42 nats)
    >>> np.random.seed(42)
    >>> x = np.random.randn(10000)
    >>> h = nonparam_entropy_c(x)
    >>> print(f"Estimated entropy: {h:.3f} nats")
    Estimated entropy: 1.422 nats
    
    >>> # Entropy in bits
    >>> h_bits = nonparam_entropy_c(x, base=2)
    >>> print(f"h_bits = {h_bits:.3f} bits")
    h_bits = 2.051 bits
    """
    # Validate inputs
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("x cannot be empty")
    
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    n_elements, n_features = x.shape
    
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be positive integer, got {k}")
    if k >= n_elements:
        raise ValueError(f"k must be less than n_samples, got k={k} with n_samples={n_elements}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = py_fast_digamma(n_elements) - py_fast_digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


def nonparam_cond_entropy_cc(x, y, k=DEFAULT_NN, base=np.e):
    """The classic K-L k-nearest neighbor continuous entropy estimator for the
    entropy of X conditioned on Y.
    
    Computes H(X|Y) = H(X,Y) - H(Y) using the Kozachenko-Leonenko estimator.
    This measures the remaining uncertainty in X when Y is known.
    
    Parameters
    ----------
    x : array-like
        Variable whose conditional entropy is computed, shape (n_samples,) or 
        (n_samples, n_features_x). Will be reshaped to 2D if 1D.
    y : array-like  
        Conditioning variable, shape (n_samples,) or (n_samples, n_features_y).
        Must have same number of samples as x.
    k : int, default=5
        Number of nearest neighbors for the estimator. Higher k reduces variance
        but increases bias. Common values are 3-10.
    base : float, default=np.e
        Logarithm base for entropy calculation. Use np.e for nats, 2 for bits.
        
    Returns
    -------
    float
        Conditional entropy H(X|Y) in units determined by base parameter.
        Can be negative since this is differential entropy for continuous variables.
        Lower values (more negative) indicate Y provides more information about X.
        
    Notes
    -----
    Uses the chain rule: H(X|Y) = H(X,Y) - H(Y), computing each term with
    the KL entropy estimator. Small random noise is added to handle discrete
    or repeated values.
    
    See Also
    --------
    ~driada.information.ksg.nonparam_entropy_c : Computes unconditional entropy H(X)
    ~driada.information.ksg.nonparam_mi_cc : Computes mutual information I(X;Y)
    
    Raises
    ------
    ValueError
        If x or y are empty or have different numbers of samples.    """
    # Validate inputs
    x = np.asarray(x)
    y = np.asarray(y)
    
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y cannot be empty")
    
    # Reshape to ensure 2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same number of samples, got {len(x)} and {len(y)}")
    
    xy = np.c_[x, y]
    entropy_union_xy = nonparam_entropy_c(xy, k=k, base=base)
    entropy_y = nonparam_entropy_c(y, k=k, base=base)
    return entropy_union_xy - entropy_y


def nonparam_mi_cc(
    x,
    y,
    z=None,
    k=DEFAULT_NN,
    base=np.e,
    alpha="auto",
    lf=5,
    precomputed_tree_x=None,
    precomputed_tree_y=None,
):
    """Kraskov-Stögbauer-Grassberger (KSG) mutual information estimator.
    
    Estimates mutual information between continuous variables using k-nearest
    neighbors. Can compute conditional MI when z is provided: I(X;Y|Z).
    
    Parameters
    ----------
    x : array-like
        First variable, shape (n_samples,) or (n_samples, n_features_x).
    y : array-like
        Second variable, shape (n_samples,) or (n_samples, n_features_y).
        Must have same number of samples as x.
    z : array-like, optional
        Conditioning variable for conditional MI: I(X;Y|Z).
        Shape (n_samples,) or (n_samples, n_features_z).
    k : int, default=5
        Number of nearest neighbors. Common values:
        - k = 4-5 for most applications
        - Use larger k for higher dimensions
        Must satisfy k < n_samples.
    base : float, default=np.e
        Logarithm base. Use np.e for nats, 2 for bits, 10 for dits.
    alpha : float or "auto", default="auto"
        Local Non-uniformity Correction (LNC) parameter.
        - "auto": automatically selects optimal alpha
        - float: manual alpha value (0 disables correction)
        - Warning: LNC disabled when k ≤ dimensionality
    lf : int, default=5
        Leaf size for k-d tree construction. Smaller values may be
        faster for small datasets, larger values for big datasets.
    precomputed_tree_x : BallTree/KDTree, optional
        Pre-built tree for x to avoid recomputation in repeated calls.
    precomputed_tree_y : BallTree/KDTree, optional
        Pre-built tree for y to avoid recomputation in repeated calls.
        
    Returns
    -------
    float
        Mutual information estimate in units determined by base.
        Always non-negative (up to estimation error).
        
    Notes
    -----
    Uses the KSG estimator algorithm 1:
    I(X;Y) = ψ(k) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(n)
    
    where:
    - ψ is the digamma function
    - n_x, n_y are the number of neighbors in X, Y spaces
    - <·> denotes average over all samples
    - n is the total number of samples
    
    Small noise is added to continuous variables to break ties.
    
    References
    ----------
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical Review E, 69(6), 066138.
    
    Gao, W., et al. (2015). Estimating mutual information for discrete-continuous
    mixtures. NIPS 2017. (LNC correction)
    
    Examples
    --------
    >>> # MI between correlated Gaussians
    >>> np.random.seed(42)
    >>> x = np.random.randn(1000)
    >>> y = x + np.random.randn(1000) * 0.5
    >>> mi = nonparam_mi_cc(x, y)
    >>> print(f"MI = {mi:.3f} nats")
    MI = 0.760 nats
    
    >>> # Conditional MI: I(X;Y|Z)
    >>> z = np.random.randn(1000)
    >>> cmi = nonparam_mi_cc(x, y, z=z)
    >>> print(f"cmi = {cmi:.3f} nats")
    cmi = 0.756 nats
    
    Raises
    ------
    ValueError
        If arrays have different lengths, k is invalid, or base is not positive.    """

    # Validate inputs
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be positive integer, got {k}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    
    x = np.asarray(x)
    y = np.asarray(y) 
    
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y cannot be empty")
    
    if len(x) != len(y):
        raise ValueError(f"Arrays should have same length, got {len(x)} and {len(y)}")
    if k >= len(x):
        raise ValueError(f"k must be less than n_samples, got k={k} with n_samples={len(x)}")

    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)

    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)

    points = np.hstack(points)
    d = points.shape[1]  # Total dimensionality
    
    # Auto-select alpha if requested
    if alpha == "auto":
        alpha = get_lnc_alpha(k, d)
        
        # Disable LNC correction if k <= d to avoid instability
        if k <= d:
            import warnings
            warnings.warn(
                f"LNC correction disabled: k={k} <= dimensionality={d}. "
                f"LNC requires k > d for stability. Consider using k >= {d+1}.",
                UserWarning
            )
            alpha = 0

    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points, lf=lf)
    dvec = query_neighbors(tree, points, k)

    if z is None:
        a = avgdigamma(x, dvec, tree=precomputed_tree_x, lf=lf)
        b = avgdigamma(y, dvec, tree=precomputed_tree_y, lf=lf)
        c = py_fast_digamma(k)
        d = py_fast_digamma(len(x))

        # print(a, b, c, d)

        if isinstance(alpha, (int, float)) and alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = (
            avgdigamma(xz, dvec),
            avgdigamma(yz, dvec),
            avgdigamma(z, dvec),
            py_fast_digamma(k),
        )

    return (-a - b + c + d) / log(base)


def lnc_correction(tree, points, k, alpha):
    """Local Non-uniformity Correction for KSG mutual information estimator.
    
    Implements the Local Non-uniformity Correction (LNC) to improve KSG estimator
    accuracy when data exhibits local non-uniformity. The correction detects regions
    where k-nearest neighbors are aligned along lower-dimensional manifolds and
    adjusts the entropy estimate accordingly.
    
    Parameters
    ----------
    tree : sklearn.neighbors.KDTree or BallTree
        Pre-built tree structure for efficient nearest neighbor queries on the
        joint space data points.
    points : ndarray of shape (n_samples, n_features)
        The joint space data points (typically concatenated X and Y variables
        for mutual information estimation). Must have at least k+1 samples.
    k : int
        Number of nearest neighbors used in the KSG estimator. Should be > d
        where d is the dimensionality, with k >= d+2 recommended for stability.
        Must be positive.
    alpha : float
        Threshold parameter for detecting local non-uniformity. The correction
        is applied when Volume_PCA / Volume_axis < alpha. Typical values from
        the lookup table range from 0.18 to 1.8. Larger values make the 
        correction more aggressive (applied more often). Set to 0 to disable.
        Must be non-negative.
        
    Returns
    -------
    float
        The correction term to be added to the entropy/MI estimate. Always >= 0.
        
    Raises
    ------
    ValueError
        If inputs have invalid shapes or values.
        
    Notes
    -----
    The algorithm works by:
    1. For each point, finding its k nearest neighbors
    2. Computing PCA on these neighbors (after mean-centering)
    3. Comparing the volume of the PCA-aligned bounding box vs axis-aligned box
    4. If PCA box is significantly smaller (by factor alpha), a correction is applied
    
    This detects when neighbors lie on a lower-dimensional manifold, which would
    cause the standard KSG estimator to overestimate entropy.
    
    Warning: Can be unstable when k <= d. In practice, the correction is often
    disabled (alpha=0) when k is too small relative to dimensionality.
    
    See Also
    --------
    ~driada.information.ksg.get_lnc_alpha : Get optimal alpha value from lookup table
    ~driada.information.ksg.nonparam_mi_cc : Main MI estimator that uses this correction with alpha="auto"
    
    References
    ----------
    Gao et al. (2015). Efficient estimation of mutual information for strongly
    dependent variables. AISTATS.    """
    # Validate inputs
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError(f"points must be 2D array, got shape {points.shape}")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be positive integer, got {k}")
    if not isinstance(alpha, (int, float, np.number)) or alpha < 0:
        raise ValueError(f"alpha must be non-negative number, got {alpha}")
    if points.shape[0] <= k:
        raise ValueError(f"Need at least k+1 points, got {points.shape[0]} points with k={k}")
    
    # Early return if alpha is 0 (correction disabled)
    if alpha == 0:
        return 0.0
    
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k + 1, return_distance=False)[0]
        knn_points = points[knn]
        # Subtract mean of k-nearest neighbor points (fixed from subtracting first point)
        knn_points = knn_points - np.mean(knn_points, axis=0)
        # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        # Use eigh for symmetric matrix (more stable than eig)
        try:
            _, v = la.eigh(covr)
        except la.LinAlgError:
            # Skip this point if covariance is singular
            continue
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect)
    
    # Normalize by number of samples at the end (more efficient)
    return e / n_sample


def nonparam_mi_cd(x_continuous, y_discrete, k=DEFAULT_NN, base=np.e):
    """
    Mutual information between continuous and discrete variables using KSG estimator.
    
    Uses the mixed-type mutual information estimator from the KSG paper.
    
    Parameters
    ----------
    x_continuous : array_like
        Continuous variable data of shape (n_samples,) or (n_samples, n_features).
        Should contain finite values.
    y_discrete : array_like
        Discrete variable data of shape (n_samples,). Values should be discrete
        categories (integers or strings).
    k : int, optional
        Number of nearest neighbors to use. Default is 5. Must be positive.
    base : float, optional
        Logarithm base. Default is e (natural logarithm). Must be positive.
    
    Returns
    -------
    float
        Mutual information in units determined by base. Always non-negative.
        
    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid values.
        
    Notes
    -----
    Computes MI as I(X;Y) = H(X) - H(X|Y) where H(X|Y) is the weighted average
    of conditional entropies. Categories with fewer than k+1 samples are skipped,
    which may introduce bias for small sample sizes.    """
    # Validate inputs
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be positive integer, got {k}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    
    x_continuous = np.asarray(x_continuous)
    y_discrete = np.asarray(y_discrete)
    
    if x_continuous.size == 0 or y_discrete.size == 0:
        raise ValueError("x_continuous and y_discrete cannot be empty")
    
    if len(x_continuous.shape) == 1:
        x_continuous = x_continuous.reshape(-1, 1)
    
    if len(x_continuous) != len(y_discrete):
        raise ValueError(f"Arrays should have same length, got {len(x_continuous)} and {len(y_discrete)}")
    if k >= len(x_continuous):
        raise ValueError(f"k must be less than n_samples, got k={k} with n_samples={len(x_continuous)}")
    
    n_samples = len(x_continuous)
    
    # Add small noise to continuous variables to break ties
    x_continuous = add_noise(x_continuous)
    
    # Calculate H(X) - H(X|Y)
    # H(X) is the entropy of the continuous variable
    h_x = nonparam_entropy_c(x_continuous, k=k, base=base)
    
    # H(X|Y) is the conditional entropy
    h_x_given_y = 0.0
    unique_y = np.unique(y_discrete)
    
    for y_val in unique_y:
        mask = y_discrete == y_val
        p_y = np.sum(mask) / n_samples
        
        if p_y > 0:
            x_subset = x_continuous[mask]
            if len(x_subset) > k:
                h_x_y = nonparam_entropy_c(x_subset, k=min(k, len(x_subset)-1), base=base)
                h_x_given_y += p_y * h_x_y
    
    mi = h_x - h_x_given_y
    return max(0, mi)  # MI is non-negative


def nonparam_mi_dc(x_discrete, y_continuous, k=DEFAULT_NN, base=np.e):
    """
    Mutual information between discrete and continuous variables using KSG estimator.
    
    This is just the symmetric version of nonparam_mi_cd.
    
    Parameters
    ----------
    x_discrete : array_like
        Discrete variable data of shape (n_samples,). Values should be discrete
        categories (integers or strings).
    y_continuous : array_like
        Continuous variable data of shape (n_samples,) or (n_samples, n_features).
        Should contain finite values.
    k : int, optional
        Number of nearest neighbors to use. Default is 5. Must be positive.
    base : float, optional
        Logarithm base. Default is e (natural logarithm). Must be positive.
    
    Returns
    -------
    float
        Mutual information in units determined by base. Always non-negative.
        
    Notes
    -----
    MI is symmetric, so this function simply swaps the arguments and calls
    nonparam_mi_cd. See that function for implementation details.    """
    # MI is symmetric, so we can just swap the arguments
    return nonparam_mi_cd(y_continuous, x_discrete, k=k, base=base)


