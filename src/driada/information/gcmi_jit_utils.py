"""
JIT-compiled copula transformation functions for GCMI.
"""

import numpy as np
from numba import njit
from scipy.special import ndtri


@njit
def ctransform_jit(x):
    """JIT-compiled copula transformation (empirical CDF).
    
    Fast implementation using direct ranking algorithm.
    
    Parameters
    ----------
    x : ndarray
        1D array of values to transform.
        
    Returns
    -------
    ndarray
        Copula-transformed values in (0, 1).
    """
    n = x.size
    ranks = np.empty(n)
    
    # Direct ranking - O(n^2) but fast for small n with JIT
    for i in range(n):
        rank = 1
        for j in range(n):
            if x[j] < x[i]:
                rank += 1
            elif x[j] == x[i] and j < i:
                rank += 1
        ranks[i] = rank
    
    # Convert to copula values
    return ranks / (n + 1)


@njit
def ctransform_2d_jit(x):
    """JIT-compiled copula transformation for 2D arrays.
    
    Transforms each row independently.
    
    Parameters
    ----------
    x : ndarray
        2D array where each row is transformed independently.
        
    Returns
    -------
    ndarray
        Copula-transformed array.
    """
    n_vars, n_samples = x.shape
    result = np.empty_like(x)
    
    for i in range(n_vars):
        result[i, :] = ctransform_jit(x[i, :])
    
    return result


@njit
def erfinv_approx(x):
    """Approximate inverse error function for JIT compilation.
    
    Uses Winitzki's approximation which is accurate to ~0.00035 for all x.
    
    Parameters
    ----------
    x : float or ndarray
        Values in (-1, 1).
        
    Returns
    -------
    float or ndarray
        Approximate inverse error function values.
    """
    a = 0.147  # Winitzki's constant
    
    # Compute the approximation
    ln_term = np.log((1 - x*x))
    inner = 2/(np.pi * a) + ln_term/2
    sqrt_term = np.sqrt(inner*inner - ln_term/a)
    
    sign = np.sign(x)
    result = sign * np.sqrt(sqrt_term - inner)
    
    return result


@njit  
def ndtri_approx(p):
    """Approximate inverse normal CDF for JIT compilation.
    
    Uses the relationship: ndtri(p) = sqrt(2) * erfinv(2*p - 1)
    
    Parameters
    ----------
    p : float or ndarray
        Probability values in (0, 1).
        
    Returns
    -------
    float or ndarray
        Approximate quantile values.
    """
    return np.sqrt(2.0) * erfinv_approx(2.0 * p - 1.0)


@njit
def copnorm_jit(x):
    """JIT-compiled copula normalization.
    
    Fast implementation using approximations suitable for JIT.
    
    Parameters
    ----------
    x : ndarray
        1D array to normalize.
        
    Returns  
    -------
    ndarray
        Standard normal samples with same empirical CDF as input.
    """
    cx = ctransform_jit(x)
    return ndtri_approx(cx)


@njit
def copnorm_2d_jit(x):
    """JIT-compiled copula normalization for 2D arrays.
    
    Parameters
    ----------
    x : ndarray
        2D array where each row is normalized independently.
        
    Returns
    -------
    ndarray
        Copula-normalized array.
    """
    n_vars, n_samples = x.shape
    result = np.empty_like(x)
    
    for i in range(n_vars):
        result[i, :] = copnorm_jit(x[i, :])
    
    return result