import numpy as np
from numba import njit
from ..utils.jit import conditional_njit


@conditional_njit
def py_fast_digamma_arr(data):
    """Compute digamma function for an array of values using fast approximation.
    
    This is a JIT-compiled version that processes arrays efficiently.
    Uses a series expansion approximation that is accurate for x > 5.
    
    Parameters
    ----------
    data : ndarray
        Input array of positive values. All values must be > 0.
        
    Returns
    -------
    ndarray
        Array of digamma values corresponding to input data.
        
    Raises
    ------
    None
        Invalid inputs (x <= 0) return NaN instead of raising exceptions
        due to numba JIT compilation constraints.
        
    Notes
    -----
    The algorithm uses a recurrence relation to shift x to the range
    where the series expansion is accurate (x > 5), then applies
    an asymptotic expansion with correction terms.
    
    For x <= 0, the function returns NaN to avoid infinite loops.    """
    res = np.zeros(len(data))
    for i, x in enumerate(data):
        # Handle non-positive values to avoid infinite loop
        if x <= 0:
            res[i] = np.nan
            continue
        r = 0
        while x <= 5:
            r -= 1 / x
            x += 1
        f = 1 / (x * x)
        t = f * (
            -1 / 12.0
            + f
            * (
                1 / 120.0
                + f
                * (
                    -1 / 252.0
                    + f
                    * (
                        1 / 240.0
                        + f
                        * (
                            -1 / 132.0
                            + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0))
                        )
                    )
                )
            )
        )

        res[i] = r + np.log(x) - 0.5 / x + t

    return res


@conditional_njit
def py_fast_digamma(x):
    """Compute digamma function for a single value using fast approximation.
    
    This is a JIT-compiled scalar version of the digamma (psi) function.
    Uses a series expansion approximation that is accurate for x > 5.
    
    Parameters
    ----------
    x : float
        Input value. Must be positive (x > 0).
        
    Returns
    -------
    float
        The digamma function value psi(x).
        
    Raises
    ------
    None
        Invalid inputs (x <= 0) return NaN instead of raising exceptions
        due to numba JIT compilation constraints.
        
    Notes
    -----
    The digamma function is the logarithmic derivative of the gamma function:
    psi(x) = d/dx log(Gamma(x)) = Gamma'(x) / Gamma(x)
    
    The algorithm uses:
    1. Recurrence relation psi(x) = psi(x+1) - 1/x to shift to x > 5
    2. Asymptotic expansion for large x with Bernoulli number corrections
    
    For x <= 0, the function returns NaN to avoid infinite loops.    """
    # Handle non-positive values to avoid infinite loop
    if x <= 0:
        return np.nan
    
    r = 0
    x = float(x)  # Ensure float type
    while x <= 5:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    t = f * (
        -1 / 12.0
        + f
        * (
            1 / 120.0
            + f
            * (
                -1 / 252.0
                + f
                * (
                    1 / 240.0
                    + f
                    * (
                        -1 / 132.0
                        + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0))
                    )
                )
            )
        )
    )

    res = r + np.log(x) - 0.5 / x + t
    return res


def binary_mi_score(contingency):
    """Calculate mutual information for discrete variables from contingency table.
    
    Computes the mutual information between two discrete random variables
    based on their joint probability distribution represented as a contingency table.
    
    Parameters
    ----------
    contingency : ndarray of shape (n_classes_x, n_classes_y)
        Contingency table where element [i, j] contains the count of samples
        with x=i and y=j. Must contain non-negative values.
        
    Returns
    -------
    float
        Mutual information score in nats (natural log base).
        Returns 0.0 if either variable has only one class.
        
    Raises
    ------
    ValueError
        If contingency table has wrong dimensions or contains negative values.
    TypeError
        If contingency is not array-like or contains non-numeric values.
        
    Notes
    -----
    The mutual information is calculated as:
    MI(X,Y) = sum_ij P(x_i, y_j) * log(P(x_i, y_j) / (P(x_i) * P(y_j)))
    
    This implementation:
    - Handles sparse contingency tables efficiently by only computing over non-zero entries
    - Returns 0 for degenerate cases (single cluster)
    - Clips negative values due to numerical errors to 0    """
    # Input validation
    contingency = np.asarray(contingency)
    if contingency.ndim != 2:
        raise ValueError(f"Contingency table must be 2D, got {contingency.ndim}D")
    if np.any(contingency < 0):
        raise ValueError("Contingency table cannot contain negative values")
    
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a
    # single cluster, implies MI = 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - np.log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)
