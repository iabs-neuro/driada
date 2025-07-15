"""
JIT-compiled entropy calculation functions for performance optimization.
"""

import numpy as np
from numba import njit


@njit
def entropy_d_jit(x):
    """JIT-compiled discrete entropy calculation.
    
    Parameters
    ----------
    x : array-like
        Discrete variable values.
        
    Returns
    -------
    float
        Entropy in bits.
    """
    # Count occurrences efficiently
    unique_vals = np.unique(x)
    counts = np.zeros(unique_vals.size)
    
    for i in range(x.size):
        for j in range(unique_vals.size):
            if x[i] == unique_vals[j]:
                counts[j] += 1
                break
    
    # Calculate entropy
    p = counts / x.size
    h = 0.0
    for i in range(p.size):
        if p[i] > 0:
            h -= p[i] * np.log2(p[i])
    
    return h


@njit
def joint_entropy_dd_jit(x, y):
    """JIT-compiled joint entropy for two discrete variables.
    
    Parameters
    ----------
    x : array-like
        First discrete variable.
    y : array-like
        Second discrete variable.
        
    Returns
    -------
    float
        Joint entropy H(X,Y) in bits.
    """
    # Create joint distribution
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    joint_counts = np.zeros((unique_x.size, unique_y.size))
    
    # Count joint occurrences
    for i in range(x.size):
        x_idx = -1
        y_idx = -1
        
        # Find indices
        for j in range(unique_x.size):
            if x[i] == unique_x[j]:
                x_idx = j
                break
        
        for j in range(unique_y.size):
            if y[i] == unique_y[j]:
                y_idx = j
                break
                
        if x_idx >= 0 and y_idx >= 0:
            joint_counts[x_idx, y_idx] += 1
    
    # Calculate joint entropy
    total = x.size
    h = 0.0
    
    for i in range(unique_x.size):
        for j in range(unique_y.size):
            if joint_counts[i, j] > 0:
                p = joint_counts[i, j] / total
                h -= p * np.log2(p)
    
    return h