"""FFT-accelerated mutual information computation for INTENSE.

This module contains all FFT-based MI estimators optimized for computing
mutual information across many shifts efficiently. These functions provide
10-1000x speedup over loop-based approaches for INTENSE shuffle testing.

Key optimizations implemented:
- rfft/irfft for real inputs (50% faster than fft/ifft)
- Memory-efficient shift extraction (100-1000x less memory)
- Unified bias correction using scipy.special.psi
- Dimension-specific regularization thresholds

Functions
---------
Univariate:
    compute_mi_batch_fft : MI between two 1D continuous variables at multiple shifts
    mi_cd_fft : MI between 1D continuous and discrete variables at multiple shifts
    compute_mi_gd_fft : Wrapper for mi_cd_fft (continuous-discrete MI)
    mi_dd_fft : MI between two discrete variables at multiple shifts
    compute_mi_dd_fft : Wrapper for mi_dd_fft (discrete-discrete MI)

Multivariate:
    compute_mi_mts_fft : MI between 1D and multi-dimensional (d≤3) continuous variables
    compute_mi_mts_mts_fft : MI between two multi-dimensional (d1,d2≤3) continuous variables
    compute_mi_mts_discrete_fft : MI between multi-dimensional continuous and discrete variables

Internal helpers (not for direct use):
    _compute_joint_entropy_3x3_mts : 3×3 determinant-based entropy
    _compute_joint_entropy_4x4_mts : 4×4 determinant-based entropy
    _compute_joint_entropy_mts_mts_block : Block determinant for MTS-MTS MI

Author: DRIADA Development Team
"""

import numpy as np
import warnings
from scipy.special import psi

from .gcmi import regularized_cholesky
from .info_utils import py_fast_digamma, py_fast_digamma_arr
from .info_fft_utils import (
    REG_VARIANCE_THRESHOLD,
    REG_DET_2D_THRESHOLD,
    REG_DET_3D_THRESHOLD,
)

ln2 = np.log(2)


def compute_mi_batch_fft(
    copnorm_x: np.ndarray,
    copnorm_y: np.ndarray,
    shifts: np.ndarray,
    biascorrect: bool = True,
) -> np.ndarray:
    """Compute MI at multiple shifts using FFT cross-correlation.

    For copula-normalized univariate Gaussian variables, MI can be computed from
    the correlation coefficient: MI = -0.5 * log2(1 - r^2). This function computes
    correlations at multiple shifts efficiently using FFT-based cross-correlation,
    providing 10-100x speedup over per-shift computation when nsh >> log(n).

    Parameters
    ----------
    copnorm_x : ndarray of shape (n,)
        First variable, already copula-normalized (rank-transformed to Gaussian).
    copnorm_y : ndarray of shape (n,)
        Second variable, already copula-normalized.
    shifts : ndarray of shape (nsh,)
        Shift indices (0 to n-1) to compute MI for. These are circular shifts
        applied to copnorm_y before computing correlation with copnorm_x.
    biascorrect : bool, default=True
        Apply Panzeri-Treves bias correction for finite sample effects.
        The correction is constant for all shifts since sample size n is fixed.

    Returns
    -------
    mi_values : ndarray of shape (nsh,)
        Mutual information values at each shift, in bits. Non-negative.

    Notes
    -----
    This function is mathematically equivalent to calling mi_gg(x, roll(y, s))
    for each shift s, but ~100x faster for large numbers of shifts.

    The FFT computes the circular cross-correlation for all n possible shifts
    in O(n log n) time. We then look up the specific shifts requested.

    Bias correction follows Panzeri & Treves (1996) for univariate Gaussian MI:
    bias = (psi((n-2)/2) - psi((n-1)/2)) / (2 * ln(2))

    Examples
    --------
    >>> x = copnorm(np.random.randn(1000))
    >>> y = copnorm(np.random.randn(1000))
    >>> shifts = np.array([0, 10, 50, 100])
    >>> mi_values = compute_mi_batch_fft(x, y, shifts)

    See Also
    --------
    mi_gg : Single-shift MI computation (slower for multiple shifts)
    get_1d_mi : High-level MI function with TimeSeries support
    """
    n = len(copnorm_x)
    ln2 = np.log(2)

    # Demean (copnorm should already be ~zero mean, but ensure for numerical stability)
    x = copnorm_x - copnorm_x.mean()
    y = copnorm_y - copnorm_y.mean()

    # Compute circular cross-correlation using FFT
    # Use n-point FFT for circular correlation (matches np.roll behavior)
    fft_x = np.fft.rfft(x)
    fft_y = np.fft.rfft(y)
    cross_power = fft_x * np.conj(fft_y)
    cross_corr = np.fft.irfft(cross_power, n=n)

    # Normalize to correlation coefficient: r = cov(x,y) / (std_x * std_y)
    # cross_corr[s] = sum_i x[i] * y[(i+s) mod n]
    # So r[s] = cross_corr[s] / ((n-1) * std_x * std_y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)

    # Handle edge case of zero variance
    if std_x < REG_VARIANCE_THRESHOLD or std_y < REG_VARIANCE_THRESHOLD:
        return np.zeros(len(shifts))

    r_all = cross_corr / ((n - 1) * std_x * std_y)

    # Look up correlations at specific shifts
    r = r_all[shifts]

    # Compute MI: -0.5 * log2(1 - r^2)
    # Clip r^2 to avoid log(0) for perfect correlation
    r_squared = np.clip(r ** 2, 0, 1 - 1e-10)
    mi = -0.5 * np.log(1 - r_squared) / ln2

    # Bias correction (constant for all shifts since n is fixed)
    # From mi_gg for univariate (Nvarx=1, Nvary=1, Nvarxy=2):
    # The net correction is: (psi((n-2)/2) - psi((n-1)/2)) / (2 * ln2)
    if biascorrect and n > 2:
        psi_1 = py_fast_digamma((n - 1) / 2.0)
        psi_2 = py_fast_digamma((n - 2) / 2.0)
        bias_correction = (psi_2 - psi_1) / (2.0 * ln2)
        mi = mi + bias_correction  # Note: correction is typically negative

    return np.maximum(0, mi)  # MI is non-negative




def compute_mi_gd_fft(
    copnorm_continuous: np.ndarray,
    discrete_labels: np.ndarray,
    shifts: np.ndarray,
    biascorrect: bool = True,
) -> np.ndarray:
    """Compute MI(continuous; discrete) at multiple shifts using FFT.

    FFT-accelerated version for INTENSE shuffle loops with discrete features.
    Uses circular correlation to compute class-conditional statistics for all
    shifts in O(n log n) time, providing ~100-400x speedup over per-shift
    computation.

    Parameters
    ----------
    copnorm_continuous : ndarray of shape (n,)
        Copula-normalized continuous variable.
    discrete_labels : ndarray of shape (n,)
        Discrete variable (integer labels 0 to Ym-1).
    shifts : ndarray of shape (nsh,)
        Shift indices to compute MI for. These are circular shifts
        applied to discrete_labels before computing MI.
    biascorrect : bool, default=True
        Apply Panzeri-Treves bias correction for finite sample effects.

    Returns
    -------
    mi_values : ndarray of shape (nsh,)
        Mutual information values at each shift, in bits. Non-negative.

    Notes
    -----
    This function uses the insight that class-conditional sums under circular
    shifts are circular correlations: sum(z[x_k == class_i]) = (z ⊛ I_i)[k].
    FFT allows computing sums for ALL shifts in one operation.

    Works with any number of discrete classes (binary or multi-class).

    See Also
    --------
    mi_cd_fft : Core implementation in entropy module
    compute_mi_batch_fft : FFT version for continuous-continuous MI
    """
    return mi_cd_fft(copnorm_continuous, discrete_labels, shifts, biascorrect=biascorrect)



def compute_mi_mts_fft(
    copnorm_z: np.ndarray,
    copnorm_x: np.ndarray,
    shifts: np.ndarray,
    biascorrect: bool = True,
) -> np.ndarray:
    """Compute MI(1D; dD) at multiple shifts using FFT for MultiTimeSeries.

    FFT-accelerated MI computation between a univariate TimeSeries and a
    d-dimensional MultiTimeSeries. Uses circular correlation to compute
    cross-covariances for all shifts in O(d * n log n) time, then evaluates
    MI via vectorized determinant computation.

    Parameters
    ----------
    copnorm_z : ndarray of shape (n,)
        Copula-normalized univariate variable (e.g., neural activity).
    copnorm_x : ndarray of shape (d, n)
        Copula-normalized multivariate variable (e.g., 2D position).
        Each row is one dimension.
    shifts : ndarray of shape (nsh,)
        Shift indices to compute MI for. These are circular shifts
        applied to copnorm_x before computing MI.
    biascorrect : bool, default=True
        Apply Panzeri-Treves bias correction for finite sample effects.

    Returns
    -------
    mi_values : ndarray of shape (nsh,)
        Mutual information values at each shift, in bits. Non-negative.

    Raises
    ------
    NotImplementedError
        If copnorm_x has more than 3 dimensions (d > 3).

    Notes
    -----
    For d=2 (3x3 determinant) and d=3 (4x4 determinant), uses closed-form
    vectorized determinant formulas for maximum performance.

    The FFT computes cross-covariances Cov(Z, Xi) for all n possible shifts
    in O(n log n) time per dimension. The invariant statistics (Var(Z),
    covariance within X) are computed once.

    Complexity: O(d * n log n) regardless of number of shifts, plus O(nsh)
    for the vectorized determinant computation.

    See Also
    --------
    compute_mi_batch_fft : FFT version for 1D-1D MI
    compute_mi_gd_fft : FFT version for discrete-continuous MI
    mi_gg : Reference implementation for multivariate Gaussian MI
    """
    n = len(copnorm_z)

    # Explicit shape validation to prevent transpose bugs
    if copnorm_x.ndim == 1:
        copnorm_x = copnorm_x.reshape(1, -1)
    elif copnorm_x.ndim == 2:
        # Validate shape is (d, n) not (n, d)
        if copnorm_x.shape[0] > copnorm_x.shape[1]:
            raise ValueError(
                f"MultiTimeSeries data shape looks transposed: {copnorm_x.shape}. "
                f"Expected shape (d, n) with d <= 3 and n > d."
            )
    else:
        raise ValueError(f"copnorm_x must be 1D or 2D, got {copnorm_x.ndim}D")

    d = copnorm_x.shape[0]  # dimensionality of MultiTimeSeries
    nsh = len(shifts)
    ln2 = np.log(2)

    # Validate minimum sample size for bias correction (ddof=1 requires n >= 2)
    if n < 2:
        raise ValueError(
            f"MultiTimeSeries FFT requires at least 2 samples for bias correction. Got n={n}."
        )

    if d > 3:
        raise NotImplementedError(
            f"FFT-accelerated MI for MultiTimeSeries with d > 3 is not implemented. "
            f"Got d={d}. Use engine='loop' for higher dimensions."
        )

    # Demean all variables (copnorm should be ~zero mean, but ensure stability)
    z = copnorm_z - copnorm_z.mean()
    x = copnorm_x - copnorm_x.mean(axis=1, keepdims=True)

    # --- Step 1: Compute shift-invariant statistics (once) ---

    # Variance of z (scalar)
    var_z = np.var(z, ddof=1)

    # Handle edge case of zero variance
    if var_z < REG_VARIANCE_THRESHOLD:
        return np.zeros(nsh)

    # Covariance matrix of x (d x d) - shift-invariant
    if d == 1:
        cov_xx = np.array([[np.var(x[0], ddof=1)]])
    else:
        cov_xx = np.cov(x)  # uses ddof=1 by default

    # H(Z) in nats (normalizations cancel for MI)
    H_Z = 0.5 * np.log(var_z)

    # H(X) via Cholesky - computed once
    from .gcmi import regularized_cholesky

    chol_xx = regularized_cholesky(cov_xx)
    H_X = np.sum(np.log(np.diag(chol_xx)))

    # --- Step 2: Compute shift-dependent cross-covariances via FFT ---

    # FFT of z
    fft_z = np.fft.rfft(z)

    # Extract specific shifts and allocate only what we need
    # This avoids allocating full (d, n) buffer when nsh << n
    shifts_int = shifts.astype(int) % n
    nsh = len(shifts)
    cov_zx = np.zeros((d, nsh))

    # Cross-covariance Cov(Z, Xi) for requested shifts only
    for i in range(d):
        fft_xi = np.fft.rfft(x[i])
        cross_corr = np.fft.irfft(fft_z * np.conj(fft_xi), n=n)
        # Convert to covariance: cross_corr[s] = sum_j z[j] * x[i,(j+s) mod n]
        # So cov[s] = cross_corr[s] / (n-1)
        # Direct indexing - extract only requested shifts
        cov_zx[i] = cross_corr[shifts_int] / (n - 1)

    # --- Step 3: Compute H(Z,X) for all shifts (vectorized determinants) ---

    if d == 1:
        # 2x2 case: det = var_z * var_x - cov_zx^2
        var_x = cov_xx[0, 0]
        det_joint = var_z * var_x - cov_zx[0, :] ** 2
        det_joint = np.maximum(det_joint, REG_DET_2D_THRESHOLD)
        H_ZX = 0.5 * np.log(det_joint)

    elif d == 2:
        # 3x3 case: closed-form determinant
        H_ZX = _compute_joint_entropy_3x3_mts(var_z, cov_xx, cov_zx)

    elif d == 3:
        # 4x4 case: closed-form determinant
        H_ZX = _compute_joint_entropy_4x4_mts(var_z, cov_xx, cov_zx)

    # --- Step 4: Compute MI = H(Z) + H(X) - H(Z,X) ---

    # Apply bias correction before combining
    if biascorrect and n > 2:
        # From mi_gg: bias correction terms
        # psiterms = psi((n - k) / 2) / 2 for k = 1, 2, ..., dim
        # dterm = (ln2 - log(n-1)) / 2
        dterm = (ln2 - np.log(n - 1.0)) / 2.0

        Nvarx = d
        Nvary = 1
        Nvarxy = d + 1

        psiterms = py_fast_digamma_arr((n - np.arange(1, Nvarxy + 1)) / 2.0) / 2.0

        H_Z = H_Z - Nvary * dterm - psiterms[:Nvary].sum()
        H_X = H_X - Nvarx * dterm - psiterms[:Nvarx].sum()
        H_ZX = H_ZX - Nvarxy * dterm - psiterms[:Nvarxy].sum()

    MI = (H_Z + H_X - H_ZX) / ln2

    return np.maximum(0, MI)




def _compute_joint_entropy_3x3_mts(
    var_z: float, cov_xx: np.ndarray, cov_zx: np.ndarray
) -> np.ndarray:
    """Vectorized 3x3 determinant for joint entropy H(Z,X) with d=2.

    Joint covariance matrix structure:
    [[var_z,    cov_zx[0], cov_zx[1]],
     [cov_zx[0], cov_xx[0,0], cov_xx[0,1]],
     [cov_zx[1], cov_xx[0,1], cov_xx[1,1]]]

    Parameters
    ----------
    var_z : float
        Variance of z (scalar).
    cov_xx : ndarray of shape (2, 2)
        Covariance matrix of x (shift-invariant).
    cov_zx : ndarray of shape (2, nsh)
        Cross-covariances Cov(Z, Xi) for each shift.

    Returns
    -------
    H_ZX : ndarray of shape (nsh,)
        Joint entropy H(Z,X) for each shift, in nats.
    """
    # Matrix elements (using cofactor expansion along first row)
    # Sigma = [[a, b, c], [b, d, e], [c, e, f]]
    a = var_z
    b = cov_zx[0, :]  # shape (nsh,)
    c = cov_zx[1, :]  # shape (nsh,)
    d_val = cov_xx[0, 0]
    e = cov_xx[0, 1]
    f = cov_xx[1, 1]

    # det = a*(d*f - e^2) - b*(b*f - c*e) + c*(b*e - c*d)
    # Simplify: det = a*det_xx - b^2*f + b*c*e + c*b*e - c^2*d
    #              = a*det_xx - f*b^2 - d*c^2 + 2*e*b*c
    det_xx = d_val * f - e * e  # scalar (shift-invariant)

    # Check for ill-conditioned covariance before using det_xx
    if det_xx < REG_DET_2D_THRESHOLD:
        raise ValueError(
            f"Covariance matrix is nearly singular (det_xx={det_xx:.2e}). "
            f"MultiTimeSeries dimensions may be linearly dependent."
        )

    det = a * det_xx - f * b**2 - d_val * c**2 + 2 * e * b * c

    # Check conditional covariance and regularize if needed
    if np.any(det < REG_DET_3D_THRESHOLD):
        import warnings
        warnings.warn(
            f"Conditional covariance determinant near zero (min={np.min(det):.2e}). "
            f"MI estimate may be unreliable. Applying regularization.",
            UserWarning
        )
        det = np.maximum(det, REG_DET_3D_THRESHOLD)

    return 0.5 * np.log(det)




def _compute_joint_entropy_4x4_mts(
    var_z: float, cov_xx: np.ndarray, cov_zx: np.ndarray
) -> np.ndarray:
    """Vectorized 4x4 determinant for joint entropy H(Z,X) with d=3.

    Joint covariance matrix structure:
    [[var_z,     cov_zx[0], cov_zx[1], cov_zx[2]],
     [cov_zx[0], c00,       c01,       c02      ],
     [cov_zx[1], c01,       c11,       c12      ],
     [cov_zx[2], c02,       c12,       c22      ]]

    Parameters
    ----------
    var_z : float
        Variance of z (scalar).
    cov_xx : ndarray of shape (3, 3)
        Covariance matrix of x (shift-invariant).
    cov_zx : ndarray of shape (3, nsh)
        Cross-covariances Cov(Z, Xi) for each shift.

    Returns
    -------
    H_ZX : ndarray of shape (nsh,)
        Joint entropy H(Z,X) for each shift, in nats.
    """
    # Extract elements for clarity
    a = var_z
    b = cov_zx[0, :]  # shape (nsh,)
    c = cov_zx[1, :]
    d = cov_zx[2, :]

    # Covariance matrix elements of X (shift-invariant)
    c00 = cov_xx[0, 0]
    c01 = cov_xx[0, 1]
    c02 = cov_xx[0, 2]
    c11 = cov_xx[1, 1]
    c12 = cov_xx[1, 2]
    c22 = cov_xx[2, 2]

    # Use cofactor expansion along first row:
    # det(A) = a * M00 - b * M01 + c * M02 - d * M03
    # where M_ij is the (i,j) minor

    # M00 = det of 3x3 submatrix excluding row 0 and col 0
    # [[c00, c01, c02], [c01, c11, c12], [c02, c12, c22]]
    M00 = c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (
        c01 * c12 - c11 * c02
    )

    # M01 = det of 3x3 submatrix excluding row 0 and col 1
    # [[b, c01, c02], [c, c11, c12], [d, c12, c22]]
    M01 = (
        b * (c11 * c22 - c12 * c12)
        - c01 * (c * c22 - c12 * d)
        + c02 * (c * c12 - c11 * d)
    )

    # M02 = det of 3x3 submatrix excluding row 0 and col 2
    # [[b, c00, c02], [c, c01, c12], [d, c02, c22]]
    M02 = (
        b * (c01 * c22 - c12 * c02)
        - c00 * (c * c22 - c12 * d)
        + c02 * (c * c02 - c01 * d)
    )

    # M03 = det of 3x3 submatrix excluding row 0 and col 3
    # [[b, c00, c01], [c, c01, c11], [d, c02, c12]]
    M03 = (
        b * (c01 * c12 - c11 * c02)
        - c00 * (c * c12 - c11 * d)
        + c01 * (c * c02 - c01 * d)
    )

    # Full determinant
    det = a * M00 - b * M01 + c * M02 - d * M03

    # Ensure positive for log
    det = np.maximum(det, REG_DET_3D_THRESHOLD)

    return 0.5 * np.log(det)




def _compute_joint_entropy_mts_mts_block(
    cov_x1x1: np.ndarray,
    cov_x2x2: np.ndarray,
    cov_x1x2: np.ndarray,
) -> np.ndarray:
    """Vectorized block determinant for MTS-MTS joint entropy.

    Uses Schur complement: det([[A,B],[C,D]]) = det(A) × det(D - C A⁻¹ B)

    Parameters
    ----------
    cov_x1x1 : ndarray of shape (d1, d1)
        Within-covariance of X1 (shift-invariant).
    cov_x2x2 : ndarray of shape (d2, d2)
        Within-covariance of X2 (shift-invariant).
    cov_x1x2 : ndarray of shape (d1, d2, nsh)
        Cross-covariances for all shifts.

    Returns
    -------
    H_X1X2 : ndarray of shape (nsh,)
        Joint entropy H(X1,X2) for each shift, in nats.
    """
    from .gcmi import regularized_cholesky

    d1, d2, nsh = cov_x1x2.shape

    # Compute invariant determinants and inverse
    det_x1 = np.linalg.det(cov_x1x1)

    # Validate conditioning
    if det_x1 < 1e-15:
        raise ValueError("Singular covariance matrix for X1")

    inv_x1x1 = np.linalg.inv(cov_x1x1)

    # Vectorized Schur complement computation
    # S[k] = cov_x2x2 - cov_x1x2[:,:,k].T @ inv_x1x1 @ cov_x1x2[:,:,k]

    # Compute inv_x1x1 @ cov_x1x2 for all shifts: (d1, d2, nsh)
    inv_cov = np.einsum('ij,jkl->ikl', inv_x1x1, cov_x1x2)

    # Compute cov_x1x2.T @ inv_cov for all shifts: (d2, d2, nsh)
    # cov_x1x2 is (d1, d2, nsh), we need transpose along first two dims
    quad_form = np.einsum('jil,jkl->ikl', cov_x1x2, inv_cov)

    # Schur complement: broadcast cov_x2x2 and subtract
    schur = cov_x2x2[:, :, np.newaxis] - quad_form  # (d2, d2, nsh)

    # Compute determinants via Cholesky (stable and efficient)
    det_schur = np.zeros(nsh)
    for k in range(nsh):
        try:
            chol = regularized_cholesky(schur[:, :, k])
            det_schur[k] = np.prod(np.diag(chol)) ** 2
        except np.linalg.LinAlgError:
            # Nearly singular - use stronger regularization
            schur_reg = schur[:, :, k] + np.eye(d2) * 1e-10
            chol = regularized_cholesky(schur_reg)
            det_schur[k] = np.prod(np.diag(chol)) ** 2

    # Full determinant via block formula
    det_joint = det_x1 * det_schur
    det_joint = np.maximum(det_joint, REG_DET_3D_THRESHOLD)

    return 0.5 * np.log(det_joint)




def compute_mi_mts_mts_fft(
    copnorm_x1: np.ndarray,
    copnorm_x2: np.ndarray,
    shifts: np.ndarray,
    biascorrect: bool = True,
) -> np.ndarray:
    """Compute MI(X1; X2) at multiple shifts using FFT for MTS-MTS pairs.

    FFT-accelerated MI computation between two multivariate time series.
    Uses circular correlation to compute cross-covariances for all shifts
    in O(d1 × d2 × n log n) time, then evaluates MI via block determinant
    formula (Schur complement).

    Parameters
    ----------
    copnorm_x1 : ndarray of shape (d1, n)
        First copula-normalized multivariate variable.
    copnorm_x2 : ndarray of shape (d2, n)
        Second copula-normalized multivariate variable.
    shifts : ndarray of shape (nsh,)
        Shift indices for circular shifts of copnorm_x2.
    biascorrect : bool, default=True
        Apply Panzeri-Treves bias correction.

    Returns
    -------
    mi_values : ndarray of shape (nsh,)
        Mutual information at each shift, in bits.

    Raises
    ------
    NotImplementedError
        If d1 + d2 > 6 (dimension limit for FFT acceleration).
    ValueError
        If inputs have incompatible shapes or singular covariances.

    Notes
    -----
    Uses block matrix determinant formula for all cases:
    det([[A,B],[C,D]]) = det(A) × det(D - C A⁻¹ B)

    The FFT computes cross-covariances Cov(X1_i, X2_j) for all n possible
    shifts in O(n log n) time per (i,j) pair. The invariant statistics
    (within-covariances) are computed once.

    Complexity: O(d1 × d2 × n log n + nsh × d³) where d = max(d1, d2).

    See Also
    --------
    compute_mi_mts_fft : FFT version for MTS + 1D TimeSeries
    compute_mi_batch_fft : FFT version for 1D-1D MI
    mi_gg : Reference implementation for multivariate Gaussian MI
    """
    from .gcmi import regularized_cholesky

    # Step 1: Input validation
    if copnorm_x1.ndim == 1:
        copnorm_x1 = copnorm_x1.reshape(1, -1)
    if copnorm_x2.ndim == 1:
        copnorm_x2 = copnorm_x2.reshape(1, -1)

    if copnorm_x1.ndim != 2 or copnorm_x2.ndim != 2:
        raise ValueError(
            f"copnorm_x1 and copnorm_x2 must be 1D or 2D arrays. "
            f"Got shapes: {copnorm_x1.shape}, {copnorm_x2.shape}"
        )

    d1, n1 = copnorm_x1.shape
    d2, n2 = copnorm_x2.shape

    # Check for transposed inputs
    if d1 > n1:
        raise ValueError(
            f"copnorm_x1 shape looks transposed: {copnorm_x1.shape}. "
            f"Expected shape (d1, n) with d1 <= n."
        )
    if d2 > n2:
        raise ValueError(
            f"copnorm_x2 shape looks transposed: {copnorm_x2.shape}. "
            f"Expected shape (d2, n) with d2 <= n."
        )

    if n1 != n2:
        raise ValueError(
            f"copnorm_x1 and copnorm_x2 must have same number of samples. "
            f"Got n1={n1}, n2={n2}"
        )

    n = n1
    nsh = len(shifts)
    ln2 = np.log(2)

    # Validate minimum sample size
    if n < 2:
        raise ValueError(
            f"MTS-MTS FFT requires at least 2 samples for bias correction. Got n={n}."
        )

    # Check dimension limit
    if d1 + d2 > 6:
        raise NotImplementedError(
            f"FFT-accelerated MI for MTS-MTS with d1+d2 > 6 is not implemented. "
            f"Got d1={d1}, d2={d2}, d1+d2={d1+d2}. Use engine='loop' for higher dimensions."
        )

    # Step 2: Handle special cases - delegate to existing MTS-1D function
    if d1 == 1:
        # X1 is effectively 1D, use existing MTS-1D function
        return compute_mi_mts_fft(copnorm_x1[0], copnorm_x2, shifts, biascorrect)
    if d2 == 1:
        # X2 is effectively 1D, use existing MTS-1D function
        # Note: MI is symmetric, but shifts need to be applied to X2
        return compute_mi_mts_fft(copnorm_x2[0], copnorm_x1, shifts, biascorrect)
    if d1 == 0 or d2 == 0:
        return np.zeros(nsh)

    # Step 3: Demean variables
    x1 = copnorm_x1 - copnorm_x1.mean(axis=1, keepdims=True)
    x2 = copnorm_x2 - copnorm_x2.mean(axis=1, keepdims=True)

    # Step 4: Compute shift-invariant statistics (once)
    cov_x1x1 = np.cov(x1)  # (d1, d1)
    cov_x2x2 = np.cov(x2)  # (d2, d2)

    # Ensure 2D for scalar case
    if d1 == 1:
        cov_x1x1 = cov_x1x1.reshape(1, 1)
    if d2 == 1:
        cov_x2x2 = cov_x2x2.reshape(1, 1)

    # Validate conditioning
    det_x1 = np.linalg.det(cov_x1x1)
    det_x2 = np.linalg.det(cov_x2x2)
    if det_x1 < 1e-15 or det_x2 < 1e-15:
        raise ValueError(
            f"Singular covariance matrix detected. "
            f"det(cov_x1x1)={det_x1:.2e}, det(cov_x2x2)={det_x2:.2e}"
        )

    # Compute entropies H(X1), H(X2) via Cholesky
    chol_x1 = regularized_cholesky(cov_x1x1)
    chol_x2 = regularized_cholesky(cov_x2x2)
    H_X1 = np.sum(np.log(np.diag(chol_x1)))
    H_X2 = np.sum(np.log(np.diag(chol_x2)))

    # Step 5: Compute cross-covariances via FFT (shift-dependent)
    # Extract specific shifts and allocate only what we need
    # This avoids allocating full (d1, d2, n) buffer when nsh << n
    shifts_int = shifts.astype(int) % n
    nsh = len(shifts)
    cov_x1x2 = np.zeros((d1, d2, nsh))

    for i in range(d1):
        fft_x1i = np.fft.rfft(x1[i])
        for j in range(d2):
            fft_x2j = np.fft.rfft(x2[j])
            # FFT cross-correlation: use fft_x1i * conj(fft_x2j) to match np.roll(x2, shift)
            # This computes sum_t x1[i,t] * x2[j,(t-shift) mod n] which matches np.roll(x2, shift)
            cross_corr = np.fft.irfft(fft_x1i * np.conj(fft_x2j), n=n)
            # Direct indexing - extract only requested shifts
            cov_x1x2[i, j, :] = cross_corr[shifts_int] / (n - 1)

    # Step 6: Compute H(X1, X2) using block determinant
    H_X1X2 = _compute_joint_entropy_mts_mts_block(
        cov_x1x1, cov_x2x2, cov_x1x2
    )

    # Step 7: Bias correction (following mi_gg pattern)
    if biascorrect and n > 2:
        dterm = (ln2 - np.log(n - 1.0)) / 2.0
        Nvarx = d1
        Nvary = d2
        Nvarxy = d1 + d2

        psiterms = py_fast_digamma_arr((n - np.arange(1, Nvarxy + 1)) / 2.0) / 2.0

        H_X1 = H_X1 - Nvarx * dterm - psiterms[:Nvarx].sum()
        H_X2 = H_X2 - Nvary * dterm - psiterms[:Nvary].sum()
        H_X1X2 = H_X1X2 - Nvarxy * dterm - psiterms[:Nvarxy].sum()

    # Step 8: Compute MI and convert to bits
    MI = (H_X1 + H_X2 - H_X1X2) / ln2

    return np.maximum(0, MI)




def compute_mi_mts_discrete_fft(
    copnorm_mts: np.ndarray,
    discrete_labels: np.ndarray,
    shifts: np.ndarray,
    biascorrect: bool = True,
) -> np.ndarray:
    """Compute MI(MultiTimeSeries; discrete) at multiple shifts using FFT.

    Implements FFT-accelerated mutual information computation for multivariate
    continuous timeseries (MTS) vs discrete features. Uses class-conditional
    FFT approach from mi_cd_fft() extended to multivariate case.

    The key insight is that class-conditional statistics are circular correlations:
        sum(X[discrete == c]) = IFFT(FFT(X) * conj(FFT(I_c)))
    where I_c is the indicator function for class c. FFT allows computing
    statistics for ALL shifts in O(d² × Ym × n log n) time instead of
    O(nsh × [n×d² + d³]) for the loop-based approach.

    Expected speedup: 5-50x for typical INTENSE workloads with d=2-3, Ym=2-5,
    nsh=100-1000.

    Parameters
    ----------
    copnorm_mts : array, shape (d, n)
        Copula-normalized MultiTimeSeries data. Each row is one dimension.
    discrete_labels : array, shape (n,)
        Discrete variable (integer labels 0 to Ym-1).
    shifts : array, shape (nsh,)
        Specific shift indices to compute MI for.
    biascorrect : bool, default=True
        Apply Panzeri-Treves bias correction for finite sample effects using
        scipy.special.psi (digamma function). Matches mi_cd_fft() in entropy.py.

    Returns
    -------
    mi_values : array, shape (nsh,)
        MI values at each requested shift (in bits).

    Raises
    ------
    ValueError
        If input shapes are invalid or transposed.
    NotImplementedError
        If d > 3 (use engine='loop' for higher dimensions).

    Notes
    -----
    Uses Gaussian entropy via Cholesky decomposition (regularized_cholesky from
    gcmi.py) for numerical stability. Matches implementation patterns from
    compute_mi_mts_mts_fft() and mi_cd_fft().

    Complexity: O(d² × Ym × n log n + d³ × n × Ym)
    - FFT of d MTS dimensions: O(d × n log n)
    - FFT of d² products: O(d² × n log n)
    - Per-class processing: Ym classes × O(d² × n)
    - Determinants: O(d³ × n × Ym) using closed-form vectorized formulas

    FFT wins over loop when: nsh > Ym × log(n)
    - For Ym=3, n=2000: threshold ≈ 33 shifts
    - Typical INTENSE: nsh=100-1000, so FFT always beneficial

    See Also
    --------
    compute_mi_mts_fft : FFT version for MTS + continuous univariate
    compute_mi_mts_mts_fft : FFT version for MTS + MTS
    mi_cd_fft : FFT version for continuous + discrete (1D case)
    mi_model_gd : Reference loop implementation
    """
    from scipy.special import psi  # digamma function

    # Input validation and reshaping
    if copnorm_mts.ndim == 1:
        copnorm_mts = copnorm_mts.reshape(1, -1)
    elif copnorm_mts.ndim == 2:
        # Validate shape is (d, n) not (n, d)
        if copnorm_mts.shape[0] > copnorm_mts.shape[1]:
            raise ValueError(
                f"MultiTimeSeries data shape looks transposed: {copnorm_mts.shape}. "
                f"Expected shape (d, n) with d <= 3 and n > d."
            )
    else:
        raise ValueError(f"copnorm_mts must be 1D or 2D, got {copnorm_mts.ndim}D")

    d, n = copnorm_mts.shape
    nsh = len(shifts)
    ln2 = np.log(2)

    # Validate discrete_labels
    if discrete_labels.ndim != 1:
        raise ValueError(f"discrete_labels must be 1D, got {discrete_labels.ndim}D")
    if len(discrete_labels) != n:
        raise ValueError(
            f"discrete_labels length {len(discrete_labels)} doesn't match n={n}"
        )

    # Validate minimum sample size
    if n < 2:
        raise ValueError(
            f"MTS-discrete FFT requires at least 2 samples. Got n={n}."
        )

    # Dimension limit check
    if d > 3:
        raise NotImplementedError(
            f"FFT-accelerated MI for MultiTimeSeries with d > 3 is not implemented. "
            f"Got d={d}. Use engine='loop' for higher dimensions."
        )

    # Demean all variables
    x = copnorm_mts - copnorm_mts.mean(axis=1, keepdims=True)
    y = discrete_labels.astype(int)
    Ym = int(np.max(y)) + 1
    MIN_SAMPLES = max(d + 1, 2)

    # --- Step 1: Compute H(X) once (shift-invariant) using regularized_cholesky ---

    from .gcmi import regularized_cholesky

    # Covariance matrix of x (d x d)
    if d == 1:
        cov_xx = np.array([[np.var(x[0], ddof=1)]])
    else:
        cov_xx = np.cov(x)  # uses ddof=1 by default

    # H(X) via Cholesky in nats
    chol_xx = regularized_cholesky(cov_xx)
    H_X = np.sum(np.log(np.diag(chol_xx)))

    # Bias correction for H(X)
    if biascorrect and n > 2:
        dterm = (ln2 - np.log(n - 1.0)) / 2.0
        psiterms = np.zeros(d)
        for i in range(d):
            psiterms[i] = psi((n - i - 1.0) / 2.0) / 2.0
        H_X = H_X - (d * dterm + psiterms.sum())

    # --- Step 2: Precompute FFTs for all dimensions and products ---
    # Using rfft for real input - 50% faster than fft

    FFT_x = [np.fft.rfft(x[i]) for i in range(d)]
    FFT_products = {}
    for i in range(d):
        for j in range(i, d):
            FFT_products[(i, j)] = np.fft.rfft(x[i] * x[j])

    # --- Step 3: For each class, compute conditional entropy across all shifts ---

    H_cond = np.zeros(n)

    for c in range(Ym):
        I_c = (y == c).astype(float)
        n_c = int(np.sum(I_c))

        if n_c < MIN_SAMPLES:
            import warnings

            warnings.warn(
                f"Class {c} has {n_c} samples (< {MIN_SAMPLES}), skipping",
                RuntimeWarning,
            )
            continue

        FFT_Ic = np.fft.rfft(I_c)

        # Class-conditional means for all shifts (d × n)
        mean_c = np.zeros((d, n))
        for i in range(d):
            mean_c[i] = np.fft.irfft(FFT_x[i] * np.conj(FFT_Ic), n=n) / n_c

        # Class-conditional second moments (d × d × n)
        E_xixj_c = np.zeros((d, d, n))
        for i in range(d):
            for j in range(i, d):
                E_xixj_c[i, j] = np.fft.irfft(
                    FFT_products[(i, j)] * np.conj(FFT_Ic), n=n
                ) / n_c
                if i != j:
                    E_xixj_c[j, i] = E_xixj_c[i, j]

        # Class-conditional covariances with Bessel correction (d × d × n)
        cov_c = np.zeros((d, d, n))
        for i in range(d):
            for j in range(d):
                cov_pop = E_xixj_c[i, j] - mean_c[i] * mean_c[j]
                cov_c[i, j] = cov_pop * n_c / (n_c - 1)  # Bessel correction

        # Compute entropies (dimension-specific for performance)
        if d == 1:
            var_c = np.maximum(cov_c[0, 0], 1e-10)
            H_c = 0.5 * np.log(var_c)  # in nats
        elif d == 2:
            det_c = cov_c[0, 0] * cov_c[1, 1] - cov_c[0, 1] ** 2
            det_c = np.maximum(det_c, 1e-15)
            H_c = 0.5 * np.log(det_c)  # in nats
        elif d == 3:
            # Sarrus rule (vectorized 3×3 determinant)
            c00, c11, c22 = cov_c[0, 0], cov_c[1, 1], cov_c[2, 2]
            c01, c02, c12 = cov_c[0, 1], cov_c[0, 2], cov_c[1, 2]
            det_c = (
                c00 * c11 * c22
                + 2 * c01 * c02 * c12
                - c00 * c12**2
                - c11 * c02**2
                - c22 * c01**2
            )
            det_c = np.maximum(det_c, REG_DET_3D_THRESHOLD)
            H_c = 0.5 * np.log(det_c)  # in nats

        # Bias correction per class (constant across shifts)
        if biascorrect and n_c > 2:
            dterm_c = (ln2 - np.log(n_c - 1.0)) / 2.0
            psiterms_c = np.zeros(d)
            for i in range(d):
                psiterms_c[i] = psi((n_c - i - 1.0) / 2.0) / 2.0
            H_c = H_c - (d * dterm_c + psiterms_c.sum())

        # Weight by class probability and accumulate
        p_c = n_c / n
        H_cond += p_c * H_c

    # --- Step 4: Compute MI and extract requested shifts ---

    MI_all = (H_X - H_cond) / ln2  # Convert to bits
    MI_all = np.maximum(0, MI_all)  # Ensure non-negative

    return MI_all[shifts.astype(int) % n]




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

    # Precompute FFTs of z and z^2 (using rfft for real input - 50% faster)
    FFT_z = np.fft.rfft(z)
    FFT_z2 = np.fft.rfft(z**2)

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

        # FFT of indicator function (using rfft for real input)
        FFT_Iyi = np.fft.rfft(I_yi)

        # Circular correlations -> sums for all shifts
        # sum(z[x_k == yi]) = IRFFT(RFFT(z) * conj(RFFT(I_yi)))
        sum_z_yi = np.fft.irfft(FFT_z * np.conj(FFT_Iyi), n=n)
        sum_z2_yi = np.fft.irfft(FFT_z2 * np.conj(FFT_Iyi), n=n)

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


def mi_dd_fft(
    discrete_x: np.ndarray,
    discrete_y: np.ndarray,
    shifts: np.ndarray = None,
    biascorrect: bool = True,
) -> np.ndarray:
    """Compute MI(discrete; discrete) at multiple shifts using FFT.

    Uses FFT-based circular correlation to compute contingency tables
    for all shifts in O(Ym * Yn * n log n) time, providing significant
    speedup over per-shift computation when nsh >> Ym * Yn * log(n).

    The key insight is that co-occurrence counts are circular correlations:
        count(x==i AND y_shifted==j) = (I_i ⊛ J_j)[shift]
    where I_i is the indicator for x==i, J_j for y==j, and ⊛ denotes
    circular correlation. FFT allows computing ALL shifts at once.

    Parameters
    ----------
    discrete_x : array, shape (n,)
        First discrete variable (integer labels 0 to Ym-1).
    discrete_y : array, shape (n,)
        Second discrete variable (integer labels 0 to Yn-1).
    shifts : array, shape (nsh,), optional
        Specific shift indices to return. If None, returns all n shifts.
    biascorrect : bool, default=True
        Apply Miller-Madow bias correction for finite sample effects.

    Returns
    -------
    mi_values : array
        MI values at each shift (in bits). Shape is (nsh,) if shifts provided,
        otherwise (n,).

    Notes
    -----
    Uses plugin estimator: MI = Σᵢⱼ P(i,j) * log2(P(i,j) / (P(i) * P(j)))
    where P(i,j) is the joint probability and P(i), P(j) are marginals.

    Complexity: O(Ym * Yn * n log n) for FFT computation plus O(nsh * Ym * Yn)
    for MI computation from contingency tables.

    This is faster than loop-based O(nsh * n) when:
        Ym * Yn * log(n) < nsh

    For typical discrete variables (Ym, Yn ≤ 5) and INTENSE shuffles (nsh ≥ 100),
    FFT is typically 10-100x faster.
    """
    n = len(discrete_x)
    x = np.asarray(discrete_x).astype(int)
    y = np.asarray(discrete_y).astype(int)
    Ym = int(np.max(x) + 1)  # Number of classes in x
    Yn = int(np.max(y) + 1)  # Number of classes in y
    ln2 = np.log(2)

    # If shifts not specified, compute all
    if shifts is None:
        target_shifts = np.arange(n)
    else:
        target_shifts = np.asarray(shifts).astype(int) % n

    nsh = len(target_shifts)

    # Step 1: Precompute FFTs of indicator functions
    # I_x[i] = FFT of (x == i)
    # I_y[j] = FFT of (y == j)
    I_x_fft = []
    I_y_fft = []

    for i in range(Ym):
        I_x_fft.append(np.fft.rfft((x == i).astype(float)))

    for j in range(Yn):
        I_y_fft.append(np.fft.rfft((y == j).astype(float)))

    # Marginal counts (constant across shifts)
    n_x = np.array([np.sum(x == i) for i in range(Ym)])
    n_y = np.array([np.sum(y == j) for j in range(Yn)])

    # Step 2: Compute contingency tables for target shifts via FFT
    # contingency[s, i, j] = count of (x==i AND y_shifted==j) at shift s
    contingency = np.zeros((nsh, Ym, Yn))

    for i in range(Ym):
        if n_x[i] == 0:
            continue
        for j in range(Yn):
            if n_y[j] == 0:
                continue
            # Circular correlation gives counts at ALL shifts
            cross_corr = np.fft.irfft(I_x_fft[i] * np.conj(I_y_fft[j]), n=n)
            # Extract only the shifts we need
            contingency[:, i, j] = cross_corr[target_shifts]

    # Round to integers (FFT may introduce small floating point errors)
    # Use 0.5 threshold for rounding to handle floating point noise
    contingency = np.round(contingency)

    # Step 3: Compute MI from contingency tables for each shift (VECTORIZED)
    # Joint probability P(i,j) for all shifts at once
    P_joint = contingency / n  # (nsh, Ym, Yn)

    # Marginal probabilities are constant across shifts for circular correlation:
    # - P(X=i) = count(x==i) / n  (x is fixed)
    # - P(Y=j) = count(y==j) / n  (circular shift preserves counts)
    P_x = n_x / n  # (Ym,)
    P_y = n_y / n  # (Yn,)

    # Broadcast marginals for vectorized computation
    # P_x_bc: (1, Ym, 1) and P_y_bc: (1, 1, Yn) -> product is (1, Ym, Yn)
    P_x_bc = P_x[np.newaxis, :, np.newaxis]
    P_y_bc = P_y[np.newaxis, np.newaxis, :]
    P_xy_independent = P_x_bc * P_y_bc  # (1, Ym, Yn), broadcast to (nsh, Ym, Yn)

    # MI = Σᵢⱼ P(i,j) * log2(P(i,j) / (P(i) * P(j)))
    # Handle zeros: 0 * log(0) = 0 by convention, and log(0/x) should be masked
    # Note: np.where evaluates both branches before selecting, so we need errstate
    # to cover the multiply as well (log_ratio may contain NaN/Inf from 0/0 or log(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = P_joint / P_xy_independent
        log_ratio = np.log2(ratio)
        # Mask invalid entries (where P_joint == 0 or P_xy_independent == 0)
        valid_mask = (P_joint > 0) & (P_xy_independent > 0)
        mi_terms = np.where(valid_mask, P_joint * log_ratio, 0.0)
    mi_values = mi_terms.sum(axis=(1, 2))

    # Apply bias correction if requested (Miller-Madow correction)
    # Bias ≈ (k - 1) / (2 * n * ln(2)) where k is number of non-zero cells
    if biascorrect and n > 1:
        # Vectorized: count non-zero cells for each shift
        nonzero_cells = np.sum(contingency > 0.5, axis=(1, 2))
        avg_nonzero = np.mean(nonzero_cells)
        bias = (avg_nonzero - 1) / (2 * n * ln2)
        mi_values = mi_values - bias

    # Ensure non-negative
    mi_values = np.maximum(0, mi_values)

    return mi_values


def compute_mi_dd_fft(
    discrete_x: np.ndarray,
    discrete_y: np.ndarray,
    shifts: np.ndarray,
    biascorrect: bool = False,
) -> np.ndarray:
    """Wrapper for mi_dd_fft for consistency with other compute_* functions.

    Parameters
    ----------
    discrete_x : array, shape (n,)
        First discrete variable (integer labels).
    discrete_y : array, shape (n,)
        Second discrete variable (integer labels).
    shifts : array, shape (nsh,)
        Shift indices to compute MI for.
    biascorrect : bool, default=False
        Whether to apply Miller-Madow bias correction. Default is False to match
        the loop-based implementation which uses sklearn's mutual_info_score
        (plugin estimator without bias correction).

    Returns
    -------
    mi_values : array, shape (nsh,)
        MI values at each shift (in bits).
    """
    return mi_dd_fft(discrete_x, discrete_y, shifts, biascorrect=biascorrect)

