"""JIT-compiled core functions for RSA.

These functions provide optimized implementations of computationally
intensive RSA operations.
"""

import numpy as np
from ..utils.jit import conditional_njit


@conditional_njit
def fast_correlation_distance(patterns):
    """
    Compute correlation distance matrix using JIT-optimized loops.
    
    This function computes pairwise correlation distances between patterns
    using explicit loops optimized for numba JIT compilation. It handles
    edge cases like zero-variance patterns and uses sample correlation
    (ddof=1) to match numpy.corrcoef behavior.

    Parameters
    ----------
    patterns : np.ndarray
        Pattern matrix of shape (n_items, n_features). Each row represents
        a pattern/item, each column a feature.

    Returns
    -------
    rdm : np.ndarray
        Correlation distance matrix (n_items, n_items). Values range from
        0 (identical patterns) to 2 (perfectly anti-correlated). Diagonal
        is always 0.
        
    Notes
    -----
    The function standardizes each pattern to zero mean and unit variance
    using sample standard deviation (n-1 denominator). For patterns with
    zero variance, correlation is undefined and distance is set to 0 if
    patterns are identical, 1 otherwise.
    
    Correlation values are clipped to [-1, 1] to handle numerical errors
    before computing distance as 1 - correlation.
    
    Examples
    --------
    >>> patterns = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
    >>> rdm = fast_correlation_distance(patterns)
    >>> # rdm[0,1] ≈ 0 (perfect correlation)
    >>> # rdm[0,2] = 1 (undefined correlation, different patterns)
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm : Higher-level function that uses this for correlation metric
    ~driada.rsa.core_jit.fast_euclidean_distance : Alternative distance metric
    ~driada.rsa.core_jit.fast_manhattan_distance : Alternative distance metric    """
    n_items, n_features = patterns.shape
    rdm = np.zeros((n_items, n_items))

    # Standardize patterns (mean=0, std=1) - manual computation for numba
    patterns_std = np.zeros_like(patterns, dtype=np.float64)
    for i in range(n_items):
        # Compute mean
        mean = 0.0
        for j in range(n_features):
            mean += patterns[i, j]
        mean /= n_features

        # Compute std with ddof=1 (sample std) to match numpy.corrcoef
        var = 0.0
        for j in range(n_features):
            diff = patterns[i, j] - mean
            var += diff * diff
        # Use n-1 for sample standard deviation
        if n_features > 1:
            std = np.sqrt(var / (n_features - 1))
        else:
            std = 0.0

        # Standardize
        if std > 0:
            for j in range(n_features):
                patterns_std[i, j] = (patterns[i, j] - mean) / std
        else:
            # If std is 0, all values are the same, set to 0
            for j in range(n_features):
                patterns_std[i, j] = 0.0

    # Compute correlation distances
    for i in range(n_items):
        for j in range(i + 1, n_items):
            # Handle edge case where both patterns have zero variance
            # Check if either pattern is all zeros (standardized)
            pattern_i_is_zero = True
            pattern_j_is_zero = True

            for k in range(n_features):
                if patterns_std[i, k] != 0.0:
                    pattern_i_is_zero = False
                if patterns_std[j, k] != 0.0:
                    pattern_j_is_zero = False

            if pattern_i_is_zero or pattern_j_is_zero:
                # If either pattern has zero variance, correlation is undefined
                # Set distance to 0 if patterns are identical, else 1
                patterns_equal = True
                for k in range(n_features):
                    if patterns[i, k] != patterns[j, k]:
                        patterns_equal = False
                        break
                dist = 0.0 if patterns_equal else 1.0
            else:
                # Normal correlation computation
                # When standardized with ddof=1, sum of squares = n-1
                # So correlation = dot product / (n-1)
                corr = 0.0
                for k in range(n_features):
                    corr += patterns_std[i, k] * patterns_std[j, k]
                
                # Divide by (n-1) to get correlation coefficient
                if n_features > 1:
                    corr /= (n_features - 1)
                else:
                    corr = 1.0  # Single feature case

                # Clip to [-1, 1] to handle numerical errors
                if corr > 1.0:
                    corr = 1.0
                elif corr < -1.0:
                    corr = -1.0

                # Distance = 1 - correlation
                dist = 1.0 - corr

            rdm[i, j] = dist
            rdm[j, i] = dist

    return rdm


@conditional_njit
def fast_average_patterns(data, labels, unique_labels):
    """
    Average patterns within conditions using fast JIT-compiled loops.

    This function computes the mean pattern for each unique condition
    label by averaging all timepoints that belong to that condition.
    Optimized for performance using explicit loops compatible with
    numba JIT compilation.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_features, n_timepoints)
    labels : np.ndarray
        Label for each timepoint
    unique_labels : np.ndarray
        Unique condition labels

    Returns
    -------
    patterns : np.ndarray
        Averaged patterns (n_conditions, n_features)
        
    Notes
    -----
    If no timepoints match a given label, that condition's pattern
    will be all zeros. This is intentional to maintain consistent
    output shape.
    
    Examples
    --------
    >>> data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> labels = np.array([0, 1, 0, 1])
    >>> unique = np.array([0, 1])
    >>> patterns = fast_average_patterns(data, labels, unique)
    >>> # patterns[0] = mean of columns 0,2 = [2, 6]
    >>> # patterns[1] = mean of columns 1,3 = [3, 7]
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm_from_timeseries_labels : Higher-level function that uses this
    ~driada.rsa.core.compute_rdm_from_trials : Alternative averaging approach for trial data    """
    n_features, n_timepoints = data.shape
    n_conditions = len(unique_labels)
    patterns = np.zeros((n_conditions, n_features))

    for c in range(n_conditions):
        label = unique_labels[c]
        count = 0
        for t in range(n_timepoints):
            if labels[t] == label:
                patterns[c] += data[:, t]
                count += 1
        if count > 0:
            patterns[c] /= count
        # If count == 0, pattern remains zeros (no data for this label)

    return patterns


@conditional_njit
def fast_euclidean_distance(patterns):
    """
    Compute Euclidean distance matrix using JIT-optimized loops.
    
    Computes pairwise Euclidean distances between all pattern pairs
    using explicit loops for numba JIT compilation compatibility.

    Parameters
    ----------
    patterns : np.ndarray
        Pattern matrix of shape (n_items, n_features). Each row is a
        pattern in n_features-dimensional space.

    Returns
    -------
    rdm : np.ndarray
        Symmetric Euclidean distance matrix (n_items, n_items) with
        zeros on diagonal. Values are non-negative.
        
    Notes
    -----
    Uses the standard Euclidean distance formula:
    d(i,j) = sqrt(sum((patterns[i,k] - patterns[j,k])^2))
    
    No overflow protection is implemented. For very large values,
    consider normalizing patterns first.
    
    Examples
    --------
    >>> patterns = np.array([[0, 0], [3, 4], [1, 0]])
    >>> rdm = fast_euclidean_distance(patterns)
    >>> # rdm[0,1] = 5.0 (distance from origin to (3,4))
    >>> # rdm[0,2] = 1.0 (distance from origin to (1,0))
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm : Higher-level function that uses this for euclidean metric
    ~driada.rsa.core_jit.fast_correlation_distance : Alternative distance metric
    ~driada.rsa.core_jit.fast_manhattan_distance : Alternative distance metric    """
    n_items, n_features = patterns.shape
    rdm = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(i + 1, n_items):
            dist = 0.0
            for k in range(n_features):
                diff = patterns[i, k] - patterns[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            rdm[i, j] = dist
            rdm[j, i] = dist

    return rdm


@conditional_njit
def fast_manhattan_distance(patterns):
    """
    Compute Manhattan distance matrix using JIT-optimized loops.
    
    Computes pairwise Manhattan (L1) distances between patterns using
    explicit loops for numba JIT compilation compatibility.

    Parameters
    ----------
    patterns : np.ndarray
        Pattern matrix of shape (n_items, n_features). Each row represents
        a pattern/item, each column a feature.

    Returns
    -------
    rdm : np.ndarray
        Symmetric Manhattan distance matrix (n_items, n_items) with zeros
        on diagonal. All values are non-negative.
        
    Notes
    -----
    Manhattan distance (also called L1 distance or taxicab distance) is
    the sum of absolute differences: d(i,j) = sum(|patterns[i,k] - patterns[j,k]|)
    
    This metric is more robust to outliers than Euclidean distance and
    often used for high-dimensional or sparse data.
    
    Examples
    --------
    >>> patterns = np.array([[0, 0], [3, 4], [1, 1]])
    >>> rdm = fast_manhattan_distance(patterns)
    >>> # rdm[0,1] = 7 (|0-3| + |0-4|)
    >>> # rdm[0,2] = 2 (|0-1| + |0-1|)
    >>> # rdm[1,2] = 5 (|3-1| + |4-1|)
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm : Higher-level function that uses this for manhattan metric
    ~driada.rsa.core_jit.fast_euclidean_distance : Alternative distance metric
    ~driada.rsa.core_jit.fast_correlation_distance : Alternative distance metric    """
    n_items, n_features = patterns.shape
    rdm = np.zeros((n_items, n_items))

    # Use explicit loops for JIT compatibility
    for i in range(n_items):
        for j in range(i + 1, n_items):
            dist = 0.0
            for k in range(n_features):
                dist += abs(patterns[i, k] - patterns[j, k])
            rdm[i, j] = dist
            rdm[j, i] = dist

    return rdm
