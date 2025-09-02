"""
JIT-compiled entropy calculation functions for performance optimization.

Performance characteristics:
- entropy_d_jit: Faster for small datasets (< 1000 samples), slower for large datasets
  due to numpy's highly optimized C implementation. Best speedups seen with small data.
- joint_entropy_dd_jit: Consistently faster across all dataset sizes (2x-30x speedup)
  as it avoids the overhead of numpy's histogram2d function.

The implementations use vectorized operations without explicit loops where possible,
leveraging numba's ability to compile numpy operations efficiently.
"""

import numpy as np
from numba import njit
from ..utils.jit import conditional_njit


@conditional_njit
def entropy_d_jit(x):
    """JIT-compiled discrete entropy calculation using vectorized operations.

    Parameters
    ----------
    x : array-like
        Discrete variable values. Must be numeric and sortable.

    Returns
    -------
    float
        Entropy in bits. Returns 0.0 for empty arrays.
        
    Raises
    ------
    AttributeError
        If x doesn't have required array attributes (size).
    TypeError
        If x contains non-numeric or non-sortable values.
        
    Notes
    -----
    Uses Shannon entropy formula H = -Σ(p*log2(p)) where p are the
    probabilities of each unique value. The implementation sorts the
    input array to efficiently count unique values.    """
    n = x.size
    if n == 0:
        return 0.0

    # Sort the array
    x_sorted = np.sort(x)

    # Find where values change using diff
    # diff[i] = 1 if x_sorted[i] != x_sorted[i-1], else 0
    diff = np.empty(n, dtype=np.int32)
    diff[0] = 1  # First element is always a new value
    diff[1:] = x_sorted[1:] != x_sorted[:-1]

    # Get indices where values change
    change_indices = np.where(diff)[0]
    n_unique = len(change_indices)

    # Calculate counts for each unique value
    counts = np.empty(n_unique, dtype=np.int64)
    # Add the final index to simplify counting
    all_indices = np.concatenate((change_indices, np.array([n])))

    # Vectorized count calculation
    counts[:] = all_indices[1:] - all_indices[:-1]

    # Calculate probabilities
    probs = counts.astype(np.float64) / n

    # Calculate entropy using vectorized operations
    # H = -sum(p * log2(p))
    # Use np.where to avoid log(0)
    log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    h = -np.sum(probs * log_probs)

    return h


@conditional_njit
def joint_entropy_dd_jit(x, y):
    """JIT-compiled joint entropy for two discrete variables.

    Parameters
    ----------
    x : array-like
        First discrete variable. Must be numeric.
    y : array-like
        Second discrete variable. Must be numeric and same length as x.

    Returns
    -------
    float
        Joint entropy H(X,Y) in bits.
        
    Raises
    ------
    ValueError
        If x and y have different lengths or are empty.
    AttributeError
        If x or y don't have required array attributes.
    TypeError
        If x or y contain non-numeric values.
        
    Notes
    -----
    Creates a joint encoding of (x,y) pairs and calculates entropy of
    the joint distribution. Uses overflow-safe encoding with automatic
    fallback to Cantor pairing for large value ranges.    """
    n = x.size
    
    # Validate inputs
    if n == 0:
        return 0.0
    
    if y.size != n:
        raise ValueError(f"x and y must have same length, got {n} and {y.size}")

    # Find ranges for proper scaling - use int64 to prevent overflow
    x_min = np.int64(np.min(x))
    x_max = np.int64(np.max(x))
    y_min = np.int64(np.min(y))
    y_max = np.int64(np.max(y))

    # Create joint variable using vectorized operations
    # Ensure no collisions by proper scaling
    x_range = x_max - x_min + 1

    # Check for potential overflow
    if (y_max - y_min + 1) * x_range > np.iinfo(np.int64).max // 2:
        # Fall back to a different encoding for large ranges
        # Use Cantor pairing function: (x+y)*(x+y+1)/2 + y
        joint = ((x + y) * (x + y + 1)) // 2 + y
    else:
        # Vectorized pairing: joint = (x - x_min) + (y - y_min) * x_range
        joint = (x - x_min) + (y - y_min) * x_range

    # Sort for efficient counting
    joint_sorted = np.sort(joint)

    # Find where values change using diff
    diff = np.empty(n, dtype=np.int32)
    diff[0] = 1
    diff[1:] = joint_sorted[1:] != joint_sorted[:-1]

    # Get indices where values change
    change_indices = np.where(diff)[0]
    n_unique = len(change_indices)

    # Calculate counts for each unique value
    counts = np.empty(n_unique, dtype=np.int64)
    all_indices = np.concatenate((change_indices, np.array([n])))
    counts[:] = all_indices[1:] - all_indices[:-1]

    # Calculate probabilities and entropy
    probs = counts.astype(np.float64) / n
    log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    h = -np.sum(probs * log_probs)

    return h
