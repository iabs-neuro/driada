"""
Entropy calculation functions for discrete, continuous, and mixed variable types.

This module provides various entropy calculation methods including:
- Discrete entropy
- Joint entropy for discrete and mixed variables
- Conditional entropy for different variable type combinations
"""

import numpy as np
from .gcmi import ent_g
from .ksg import nonparam_entropy_c

# Import JIT versions if available
try:
    from .entropy_jit import entropy_d_jit, joint_entropy_dd_jit

    _JIT_AVAILABLE = True
except ImportError:
    _JIT_AVAILABLE = False

# Performance thresholds based on empirical measurements
ENTROPY_D_JIT_THRESHOLD = 1000  # Use JIT for arrays smaller than this
JOINT_ENTROPY_DD_ALWAYS_JIT = True  # Always use JIT for joint entropy


def entropy_d(x):
    """Calculate entropy for a discrete variable.

    Automatically selects between JIT-compiled and numpy implementations based
    on dataset size for optimal performance. JIT version is used for arrays
    smaller than ENTROPY_D_JIT_THRESHOLD (1000 elements).

    Parameters
    ----------
    x : array-like
        Discrete variable values. Should contain numeric values (integers or
        floats representing discrete states).

    Returns
    -------
    float
        Entropy in bits.

    Raises
    ------
    ValueError
        If input is not numeric.

    Examples
    --------
    >>> entropy_d([1, 1, 2, 2])  # uniform binary distribution
    1.0
    >>> entropy_d([1, 2, 3, 4])  # uniform 4-way distribution
    2.0

    Notes
    -----
    For small datasets (< 1000 elements), automatically uses JIT-compiled
    implementation if available. For larger datasets, uses optimized numpy
    implementation to avoid JIT compilation overhead."""
    x = np.asarray(x)

    # Verify input is numeric
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"Input must be numeric, got dtype: {x.dtype}")

    # Use JIT version for small datasets if available
    if _JIT_AVAILABLE and x.size < ENTROPY_D_JIT_THRESHOLD:
        return entropy_d_jit(x)

    # Use numpy implementation for large datasets
    unique_x, counts_x = np.unique(x, return_counts=True)
    p_x = counts_x / len(x)
    H_x = probs_to_entropy(p_x)
    return H_x


def probs_to_entropy(p):
    """Calculate entropy for a discrete probability distribution.

    Parameters
    ----------
    p : array-like
        Probability distribution. Will be automatically normalized to sum to 1.

    Returns
    -------
    float
        Entropy in bits.

    Examples
    --------
    >>> round(probs_to_entropy([0.5, 0.5]), 4)  # uniform binary
    1.0
    >>> abs(round(probs_to_entropy([1.0, 0.0]), 4))  # deterministic
    0.0
    >>> round(probs_to_entropy([0.25, 0.25, 0.25, 0.25]), 4)  # uniform 4-way
    2.0

    Notes
    -----
    Probabilities are automatically normalized to sum to 1. A small epsilon
    (1e-10) is added before taking logarithm to avoid numerical issues with
    log(0) and ensure numerical stability."""
    p = np.asarray(p)
    p = p / np.sum(p)  # Normalize to sum to 1
    return -np.sum(p * np.log2(p + 1e-10))  # Add small value to avoid log(0)


def joint_entropy_dd(x, y):
    """Calculate joint entropy for two discrete variables.

    Automatically uses JIT-compiled version which is consistently faster
    than the histogram2d approach across all dataset sizes.

    Parameters
    ----------
    x : array-like
        First discrete variable. Must have same length as y.
    y : array-like
        Second discrete variable. Must have same length as x.

    Returns
    -------
    float
        Joint entropy H(X,Y) in bits.

    Examples
    --------
    >>> joint_entropy_dd([1, 1, 2, 2], [1, 2, 1, 2])  # independent
    2.0
    >>> joint_entropy_dd([1, 1, 2, 2], [1, 1, 2, 2])  # perfectly dependent
    1.0

    Notes
    -----
    When JIT compilation is available, always uses the JIT version as it is
    consistently faster. Falls back to histogram2d-based implementation
    if JIT is not available."""
    x = np.asarray(x)
    y = np.asarray(y)

    # Use JIT version if available (always faster)
    if _JIT_AVAILABLE and JOINT_ENTROPY_DD_ALWAYS_JIT:
        return joint_entropy_dd_jit(x, y)

    # Fallback to histogram2d implementation
    joint_prob = np.histogram2d(x, y, bins=[np.unique(x).size, np.unique(y).size], density=True)[0]
    joint_prob /= np.sum(joint_prob)  # Normalize
    return probs_to_entropy(joint_prob.flatten())


def conditional_entropy_cdd(z, x, y, k=5, estimator="gcmi"):
    """Calculate conditional differential entropy for a continuous variable given two discrete variables.

    Computes H(Z|X,Y) where Z is continuous and X,Y are discrete. Two estimators
    are available: GCMI (fast, Gaussian assumption) and KSG (accurate, nonparametric).

    Parameters
    ----------
    z : array-like
        Continuous variable. Must have same length as x and y.
    x : array-like
        First discrete variable. Must have same length as z and y.
    y : array-like
        Second discrete variable. Must have same length as z and x.
    k : int, optional
        For KSG: number of nearest neighbors. For GCMI: minimum subset size
        threshold (partitions smaller than k are excluded). Default: 5.
    estimator : {'gcmi', 'ksg'}, optional
        Entropy estimation method:
        - 'gcmi': Fast, assumes Gaussian distribution
        - 'ksg': Accurate, nonparametric k-nearest neighbor approach
        Default: 'gcmi'.

    Returns
    -------
    float
        Conditional entropy H(Z|X,Y) in bits.

    Examples
    --------
    >>> z = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
    >>> x = [1, 1, 2, 2, 1, 2]
    >>> y = [1, 2, 1, 2, 1, 1]
    >>> result = conditional_entropy_cdd(z, x, y, k=3)
    >>> isinstance(result, float)
    True

    Notes
    -----
    GCMI estimator is faster but assumes data follows Gaussian distribution.
    KSG estimator is slower but works for arbitrary continuous distributions."""
    z = np.asarray(z)
    x = np.asarray(x)
    y = np.asarray(y)

    unique_x = np.unique(x)
    unique_y = np.unique(y)

    h_conditional = 0.0
    for ux in unique_x:
        for uy in unique_y:
            # Filter z based on x and y
            filtered_z = z[(x == ux) & (y == uy)]
            if len(filtered_z) > k:
                if estimator == "ksg":
                    # Use KSG estimator with k neighbors
                    entropy_val = nonparam_entropy_c(filtered_z.reshape(-1, 1), k=k)
                else:
                    # Use GCMI estimator (default)
                    entropy_val = ent_g(filtered_z.reshape(1, -1))

                h_conditional += entropy_val * (len(filtered_z) / len(z))

    return h_conditional


def conditional_entropy_cd(z, x, k=5, estimator="gcmi"):
    """Calculate conditional differential entropy for a continuous variable given a discrete variable.

    Computes H(Z|X) where Z is continuous and X is discrete. Two estimators
    are available: GCMI (fast, Gaussian assumption) and KSG (accurate, nonparametric).

    Parameters
    ----------
    z : array-like
        Continuous variable. Must have same length as x.
    x : array-like
        Discrete variable. Must have same length as z.
    k : int, optional
        For KSG: number of nearest neighbors. For GCMI: minimum subset size
        threshold (partitions smaller than k are excluded). Default: 5.
    estimator : {'gcmi', 'ksg'}, optional
        Entropy estimation method:
        - 'gcmi': Fast, assumes Gaussian distribution
        - 'ksg': Accurate, nonparametric k-nearest neighbor approach
        Default: 'gcmi'.

    Returns
    -------
    float
        Conditional entropy H(Z|X) in bits.

    Examples
    --------
    >>> z = [0.1, 0.2, 0.8, 0.9]
    >>> x = [1, 1, 2, 2]
    >>> result = conditional_entropy_cd(z, x, k=1)
    >>> isinstance(result, float)
    True

    Notes
    -----
    GCMI estimator is faster but assumes data follows Gaussian distribution.
    KSG estimator is slower but works for arbitrary continuous distributions."""
    z = np.asarray(z)
    x = np.asarray(x)

    unique_x = np.unique(x)
    h_conditional = 0.0

    for ux in unique_x:
        # Filter z based on x
        filtered_z = z[x == ux]
        if len(filtered_z) > k:
            if estimator == "ksg":
                # Use KSG estimator with k neighbors
                entropy_val = nonparam_entropy_c(filtered_z.reshape(-1, 1), k=k)
            else:
                # Use GCMI estimator (default)
                entropy_val = ent_g(filtered_z.reshape(1, -1))

            h_conditional += entropy_val * (len(filtered_z) / len(z))

    return h_conditional


def joint_entropy_cdd(x, y, z, k=5, estimator="gcmi"):
    """Calculate joint entropy for two discrete and one continuous variable.

    Computes H(X,Y,Z) where X,Y are discrete and Z is continuous using
    the chain rule: H(X,Y,Z) = H(X,Y) + H(Z|X,Y)

    Parameters
    ----------
    x : array-like
        First discrete variable. Must have same length as y and z.
    y : array-like
        Second discrete variable. Must have same length as x and z.
    z : array-like
        Continuous variable. Must have same length as x and y.
    k : int, optional
        For KSG: number of nearest neighbors. For GCMI: minimum subset size
        threshold. Default: 5.
    estimator : {'gcmi', 'ksg'}, optional
        Entropy estimation method for the continuous component:
        - 'gcmi': Fast, assumes Gaussian distribution
        - 'ksg': Accurate, nonparametric k-nearest neighbor approach
        Default: 'gcmi'.

    Returns
    -------
    float
        Joint entropy H(X,Y,Z) in bits.

    Examples
    --------
    >>> x = [1, 1, 2, 2]
    >>> y = [1, 2, 1, 2]
    >>> z = [0.1, 0.2, 0.8, 0.9]
    >>> result = joint_entropy_cdd(x, y, z, k=2)
    >>> isinstance(result, float)
    True

    Notes
    -----
    Discrete component H(X,Y) is computed exactly. Continuous component H(Z|X,Y)
    uses the specified estimator. Chain rule ensures mathematical correctness."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    H_xy = joint_entropy_dd(x, y)
    H_z_given_xy = conditional_entropy_cdd(z, x, y, k=k, estimator=estimator)
    H_xyz = H_xy + H_z_given_xy
    return H_xyz


def joint_entropy_cd(x, z, k=5, estimator="gcmi"):
    """Calculate joint entropy for one discrete and one continuous variable.

    Computes H(X,Z) where X is discrete and Z is continuous using
    the chain rule: H(X,Z) = H(X) + H(Z|X)

    Parameters
    ----------
    x : array-like
        Discrete variable. Must have same length as z.
    z : array-like
        Continuous variable. Must have same length as x.
    k : int, optional
        For KSG: number of nearest neighbors. For GCMI: minimum subset size
        threshold. Default: 5.
    estimator : {'gcmi', 'ksg'}, optional
        Entropy estimation method for the continuous component:
        - 'gcmi': Fast, assumes Gaussian distribution
        - 'ksg': Accurate, nonparametric k-nearest neighbor approach
        Default: 'gcmi'.

    Returns
    -------
    float
        Joint entropy H(X,Z) in bits.

    Examples
    --------
    >>> x = [1, 1, 2, 2]
    >>> z = [0.1, 0.2, 0.8, 0.9]
    >>> result = joint_entropy_cd(x, z, k=2)
    >>> isinstance(result, float)
    True

    Notes
    -----
    Discrete component H(X) is computed exactly. Continuous component H(Z|X)
    uses the specified estimator. Chain rule ensures mathematical correctness."""
    x = np.asarray(x)
    z = np.asarray(z)

    H_x = entropy_d(x)
    H_z_given_x = conditional_entropy_cd(z, x, k=k, estimator=estimator)
    H_xz = H_x + H_z_given_x
    return H_xz


def mi_cd_fft(
    copnorm_z: np.ndarray,
    discrete_x: np.ndarray,
    shifts: np.ndarray = None,
    biascorrect: bool = True,
) -> np.ndarray:
    """Compute MI(continuous; discrete) at multiple shifts using FFT.

    Uses FFT-based circular correlation to compute class-conditional
    statistics for all shifts in O(n log n) time, providing 100-2500x
    speedup over per-shift computation for typical INTENSE workloads.

    The key insight is that class-conditional sums are circular correlations:
        sum(z[x_k == class_i]) = (z ⊛ I_i)[k]
    where I_i is the indicator function for class i and ⊛ denotes circular
    correlation. FFT allows computing ALL shifts in one operation.

    Parameters
    ----------
    copnorm_z : array, shape (n,)
        Copula-normalized continuous variable.
    discrete_x : array, shape (n,)
        Discrete variable (integer labels 0 to Ym-1).
    shifts : array, shape (nsh,), optional
        Specific shift indices to return. If None, returns all n shifts.
    biascorrect : bool, default=True
        Apply Panzeri-Treves bias correction for finite sample effects.
        The correction is constant for all shifts since class sizes don't
        change under circular shifts.

    Returns
    -------
    mi_values : array
        MI values at each shift (in bits). Shape is (nsh,) if shifts provided,
        otherwise (n,).

    Notes
    -----
    Uses Gaussian entropy assumption: H = 0.5 * log(2*pi*e*var) / ln(2).
    Class sizes are constant under circular shifts, so complexity is O(n log n)
    regardless of the number of shifts requested.

    Classes with fewer than 2 samples are skipped to avoid division by zero.
    """
    from scipy.special import psi  # digamma function
    n = len(copnorm_z)
    z = np.asarray(copnorm_z)
    x = np.asarray(discrete_x).astype(int)
    Ym = int(np.max(x) + 1)  # Number of classes
    ln2 = np.log(2)

    # Precompute FFTs of z and z^2
    FFT_z = np.fft.fft(z)
    FFT_z2 = np.fft.fft(z**2)

    # Compute H(z) once - unconditional entropy
    # Uses variance (demeaned) for consistency with per-class conditional entropy
    var_z = np.var(z, ddof=1)
    if var_z < 1e-10:
        var_z = 1e-10
    H_z = 0.5 * np.log(2 * np.pi * np.e * var_z) / ln2

    # Apply bias correction to H(z) if requested
    # For univariate Gaussian: correction = dterm + psiterms where
    # dterm = (ln2 - log(n-1)) / 2, psiterms = psi((n-1)/2) / 2
    # Correction is in nats, so divide by ln2 to convert to bits
    if biascorrect and n > 2:
        dterm_z = (ln2 - np.log(n - 1.0)) / 2.0
        psiterm_z = psi((n - 1.0) / 2.0) / 2.0
        H_z = H_z - (dterm_z + psiterm_z) / ln2

    # Compute class-conditional entropies for all shifts
    H_cond = np.zeros(n)

    for yi in range(Ym):
        # Indicator function for class yi
        I_yi = (x == yi).astype(float)
        n_yi = np.sum(I_yi)

        if n_yi < 2:
            # Skip classes with too few samples
            continue

        # FFT of indicator function
        FFT_Iyi = np.fft.fft(I_yi)

        # Circular correlations -> sums for all shifts
        # sum(z[x_k == yi]) = IFFT(FFT(z) * conj(FFT(I_yi)))
        sum_z_yi = np.fft.ifft(FFT_z * np.conj(FFT_Iyi)).real
        sum_z2_yi = np.fft.ifft(FFT_z2 * np.conj(FFT_Iyi)).real

        # Variance for all shifts: var = E[z^2] - E[z]^2
        # Convert from population variance (ddof=0) to sample variance (ddof=1)
        # to match ent_g which uses np.cov (ddof=1 by default)
        mean_yi = sum_z_yi / n_yi
        var_yi_pop = sum_z2_yi / n_yi - mean_yi**2
        var_yi = var_yi_pop * n_yi / (n_yi - 1)  # Bessel's correction
        var_yi = np.maximum(var_yi, 1e-10)  # Numerical stability

        # Gaussian entropy for all shifts
        H_yi = 0.5 * np.log(2 * np.pi * np.e * var_yi) / ln2

        # Apply bias correction to H(z|x=yi) if requested
        # Correction is constant across shifts since n_yi doesn't change
        # Correction is in nats, so divide by ln2 to convert to bits
        if biascorrect and n_yi > 2:
            dterm_yi = (ln2 - np.log(n_yi - 1.0)) / 2.0
            psiterm_yi = psi((n_yi - 1.0) / 2.0) / 2.0
            H_yi = H_yi - (dterm_yi + psiterm_yi) / ln2

        # Weight by class probability (constant across shifts)
        p_yi = n_yi / n
        H_cond += p_yi * H_yi

    # MI for all shifts: MI = H(z) - H(z|x)
    MI_all = H_z - H_cond
    MI_all = np.maximum(0, MI_all)  # Ensure non-negative

    # Return specific shifts or all
    if shifts is None:
        return MI_all
    else:
        return MI_all[np.asarray(shifts).astype(int) % n]
