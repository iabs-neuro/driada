"""Shared utilities for FFT-accelerated mutual information computation.

This module provides common helper functions used by FFT-based MI estimators
in info_fft.py. All functions are prefixed with _ to indicate they are internal
utilities not intended for direct use by end users.

Author: DRIADA Development Team
"""

import numpy as np
from scipy.special import psi

# Regularization thresholds for numerical stability (dimension-specific)
# Used when computing Gaussian entropies from covariance determinants
REG_VARIANCE_THRESHOLD = 1e-10  # Variance (1D): less strict, single value
REG_DET_2D_THRESHOLD = 1e-15    # Determinant (2D): stricter, product of 2 terms
REG_DET_3D_THRESHOLD = 1e-20    # Determinant (3D+): strictest, product of 3+ terms


def _validate_and_reshape_mts(data, name="MTS", max_d=3):
    """Validate and reshape MTS data to (d, n) format.

    Parameters
    ----------
    data : ndarray
        Input data (1D or 2D)
    name : str, optional
        Variable name for error messages (default: "MTS")
    max_d : int, optional
        Maximum allowed dimensions (default: 3)

    Returns
    -------
    reshaped : ndarray, shape (d, n)
        Validated and reshaped data
    d : int
        Number of dimensions
    n : int
        Number of samples

    Raises
    ------
    ValueError
        If data has wrong shape, appears transposed, has too many dimensions,
        or has too few samples
    """
    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {data.ndim}D")

    d, n = data.shape

    # Validate shape is (d, n) not (n, d)
    if d > n:
        raise ValueError(
            f"{name} shape looks transposed: {data.shape}. "
            f"Expected shape (d, n) with d <= {max_d} and n > d."
        )

    if d > max_d:
        raise NotImplementedError(
            f"{name} has d={d} dimensions. Only d <= {max_d} is supported."
        )

    if n < 2:
        raise ValueError(f"{name} must have at least 2 samples, got {n}")

    return data, d, n


def _demean_data(data, axis=None):
    """Remove mean from data along specified axis.

    Parameters
    ----------
    data : ndarray
        Input data
    axis : int or None, optional
        Axis along which to compute mean
        If None: demean entire array (for 1D)
        If int: demean along that axis with keepdims

    Returns
    -------
    demeaned : ndarray
        Data with mean removed
    """
    if axis is None:
        return data - data.mean()
    else:
        return data - data.mean(axis=axis, keepdims=True)


def _compute_fft_cross_correlation(data1, data2, shifts_int=None, normalize_by=1.0):
    """Compute cross-correlation at specified shifts using FFT.

    Uses rfft/irfft for real inputs (50% faster than fft/ifft).

    Parameters
    ----------
    data1 : ndarray, shape (n,)
        First time series
    data2 : ndarray, shape (n,)
        Second time series
    shifts_int : ndarray or None, optional
        Shift indices to extract (if None, return all n shifts)
    normalize_by : float, optional
        Normalization factor (default: 1.0)

    Returns
    -------
    cross_corr : ndarray
        Cross-correlation at requested shifts
    """
    n = len(data1)
    fft1 = np.fft.rfft(data1)
    fft2 = np.fft.rfft(data2)
    cross_corr_full = np.fft.irfft(fft1 * np.conj(fft2), n=n)

    if shifts_int is not None:
        cross_corr = cross_corr_full[shifts_int % n]
    else:
        cross_corr = cross_corr_full

    return cross_corr / normalize_by


def _entropy_from_variance(var, regularize=REG_VARIANCE_THRESHOLD):
    """Compute Gaussian entropy from variance (1D case).

    H = 0.5 * log(var) in nats

    Parameters
    ----------
    var : float or ndarray
        Variance(s)
    regularize : float, optional
        Minimum variance threshold (default: REG_VARIANCE_THRESHOLD)

    Returns
    -------
    H : float or ndarray
        Entropy in nats
    """
    var_reg = np.maximum(var, regularize)
    return 0.5 * np.log(var_reg)


def _entropy_from_det_2x2(cov, regularize=REG_DET_2D_THRESHOLD):
    """Compute Gaussian entropy from 2×2 covariance matrix.

    det(Σ) = σ₀₀ * σ₁₁ - σ₀₁²
    H = 0.5 * log(det(Σ)) in nats

    Parameters
    ----------
    cov : ndarray, shape (2, 2) or (2, 2, nsh)
        Covariance matrix/matrices
    regularize : float, optional
        Minimum determinant threshold (default: REG_DET_2D_THRESHOLD)

    Returns
    -------
    H : float or ndarray, shape (nsh,)
        Entropy in nats
    """
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    det_reg = np.maximum(det, regularize)
    return 0.5 * np.log(det_reg)


def _entropy_from_det_3x3(cov, regularize=REG_DET_3D_THRESHOLD):
    """Compute Gaussian entropy from 3×3 covariance matrix using Sarrus rule.

    det(Σ) = c₀₀*c₁₁*c₂₂ + 2*c₀₁*c₀₂*c₁₂ - c₀₀*c₁₂² - c₁₁*c₀₂² - c₂₂*c₀₁²
    H = 0.5 * log(det(Σ)) in nats

    Parameters
    ----------
    cov : ndarray, shape (3, 3) or (3, 3, nsh)
        Covariance matrix/matrices
    regularize : float, optional
        Minimum determinant threshold (default: REG_DET_3D_THRESHOLD)

    Returns
    -------
    H : float or ndarray, shape (nsh,)
        Entropy in nats
    """
    c00, c11, c22 = cov[0, 0], cov[1, 1], cov[2, 2]
    c01, c02, c12 = cov[0, 1], cov[0, 2], cov[1, 2]

    det = (c00 * c11 * c22 + 2 * c01 * c02 * c12 -
           c00 * c12**2 - c11 * c02**2 - c22 * c01**2)

    det_reg = np.maximum(det, regularize)
    return 0.5 * np.log(det_reg)


def _apply_gaussian_bias_correction(H, n, d, ln2):
    """Apply Panzeri-Treves bias correction to Gaussian entropy estimate.

    Uses scipy.special.psi (digamma function) for numerical stability.
    This is the standard bias correction for Gaussian MI estimators.

    The correction formula is:
        correction = d * dterm + sum(psiterms)
    where:
        dterm = (ln2 - log(n-1)) / 2
        psiterms[i] = psi((n - i - 1) / 2) / 2 for i = 0, 1, ..., d-1

    Parameters
    ----------
    H : float or ndarray
        Uncorrected entropy estimate(s) in nats
    n : int
        Number of samples
    d : int
        Number of dimensions
    ln2 : float
        Natural log of 2 (for nats→bits conversion context)

    Returns
    -------
    H_corrected : float or ndarray
        Bias-corrected entropy in nats

    Notes
    -----
    The bias correction is always applied in nats, regardless of the
    final output unit (bits or nats).

    References
    ----------
    Panzeri & Treves (1996). "Analytical estimates of limited sampling
    biases in different information measures." Network: Computation in
    Neural Systems, 7(1), 87-107.
    """
    if n <= 2:
        return H

    # Determinant-based correction term
    dterm = (ln2 - np.log(n - 1.0)) / 2.0

    # Psi-based correction terms
    psiterms = np.zeros(d)
    for i in range(d):
        psiterms[i] = psi((n - i - 1.0) / 2.0) / 2.0

    # Total bias correction (in nats)
    correction = d * dterm + psiterms.sum()

    return H - correction
