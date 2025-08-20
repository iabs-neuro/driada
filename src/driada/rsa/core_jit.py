"""JIT-compiled core functions for RSA.

These functions provide optimized implementations of computationally
intensive RSA operations.
"""

import numpy as np
from ..utils.jit import conditional_njit


@conditional_njit
def fast_correlation_distance(patterns):
    """
    Fast computation of correlation distance matrix using numba-compatible operations.

    Parameters
    ----------
    patterns : np.ndarray
        Pattern matrix of shape (n_items, n_features)

    Returns
    -------
    rdm : np.ndarray
        Correlation distance matrix (n_items, n_items)
    """
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
    Fast averaging of patterns within conditions.

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
    """
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

    return patterns


@conditional_njit
def fast_euclidean_distance(patterns):
    """
    Fast computation of Euclidean distance matrix using numba-compatible operations.

    Parameters
    ----------
    patterns : np.ndarray
        Pattern matrix of shape (n_items, n_features)

    Returns
    -------
    rdm : np.ndarray
        Euclidean distance matrix (n_items, n_items)
    """
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
    Fast computation of Manhattan distance matrix using explicit loops.

    Parameters
    ----------
    patterns : np.ndarray
        Pattern matrix of shape (n_items, n_features)

    Returns
    -------
    rdm : np.ndarray
        Manhattan distance matrix (n_items, n_items)
    """
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
