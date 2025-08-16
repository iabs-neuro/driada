"""
JIT-compiled copula transformation functions for GCMI.
"""

import numpy as np
from numba import njit


@njit
def ctransform_jit(x):
    """JIT-compiled copula transformation (empirical CDF).

    Efficient O(n log n) implementation using sorting-based ranking.

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

    # Get sorting indices - O(n log n)
    sorted_indices = np.argsort(x)

    # Create ranks array
    ranks = np.empty(n, dtype=np.int64)

    # Assign ranks based on sorted order
    for i in range(n):
        ranks[sorted_indices[i]] = i + 1

    # Handle ties by using the original tie-breaking logic
    # For identical values, use index-based tie breaking
    sorted_values = x[sorted_indices]
    current_rank = 1

    for i in range(n):
        if i > 0 and sorted_values[i] == sorted_values[i - 1]:
            # Same value as previous - keep incrementing rank for each occurrence
            current_rank += 1
        else:
            # New value - start fresh rank
            current_rank = i + 1

        ranks[sorted_indices[i]] = current_rank

    # Convert to copula values
    return ranks.astype(np.float64) / (n + 1)


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
def ndtri_approx(p):
    """Approximate inverse normal CDF for JIT compilation.

    Uses a simpler but efficient approximation for JIT compilation.

    Parameters
    ----------
    p : float or ndarray
        Probability values in (0, 1).

    Returns
    -------
    float or ndarray
        Approximate quantile values.
    """
    # Handle array input
    if hasattr(p, "shape"):
        result = np.empty_like(p)
        for i in range(p.size):
            flat_p = p.flat[i]
            if flat_p <= 0.0:
                result.flat[i] = -np.inf
            elif flat_p >= 1.0:
                result.flat[i] = np.inf
            else:
                # Use Box-Muller-like transformation
                # This is a simplified approximation suitable for JIT
                if flat_p < 0.5:
                    # Use symmetry
                    q = flat_p
                    t = np.sqrt(-2.0 * np.log(q))
                    result.flat[i] = -(
                        t
                        - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                        / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
                    )
                else:
                    q = 1.0 - flat_p
                    t = np.sqrt(-2.0 * np.log(q))
                    result.flat[i] = t - (
                        2.515517 + 0.802853 * t + 0.010328 * t * t
                    ) / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
        return result
    else:
        # Scalar input
        if p <= 0.0:
            return -np.inf
        elif p >= 1.0:
            return np.inf
        else:
            if p < 0.5:
                q = p
                t = np.sqrt(-2.0 * np.log(q))
                return -(
                    t
                    - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                    / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
                )
            else:
                q = 1.0 - p
                t = np.sqrt(-2.0 * np.log(q))
                return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
                    1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
                )


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


@njit
def mi_gg_jit(x, y, biascorrect=True, demeaned=False):
    """JIT-compiled Gaussian mutual information between two variables.

    Parameters
    ----------
    x : ndarray
        First variable data (n_vars_x, n_samples).
    y : ndarray
        Second variable data (n_vars_y, n_samples).
    biascorrect : bool
        Apply bias correction.
    demeaned : bool
        Whether data is already demeaned.

    Returns
    -------
    float
        Mutual information in bits.
    """
    if x.shape[1] != y.shape[1]:
        raise ValueError("Number of samples must match")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx + Nvary

    if not demeaned:
        # Demean data - manual implementation for numba
        for i in range(x.shape[0]):
            x[i] = x[i] - np.mean(x[i])
        for i in range(y.shape[0]):
            y[i] = y[i] - np.mean(y[i])

    # Compute covariance matrices
    Cxx = np.dot(x, x.T) / (Ntrl - 1)
    Cyy = np.dot(y, y.T) / (Ntrl - 1)
    Cxy = np.dot(x, y.T) / (Ntrl - 1)
    Cyx = np.dot(y, x.T) / (Ntrl - 1)

    # Joint covariance
    C = np.empty((Nvarxy, Nvarxy))
    C[:Nvarx, :Nvarx] = Cxx
    C[:Nvarx, Nvarx:] = Cxy
    C[Nvarx:, :Nvarx] = Cyx
    C[Nvarx:, Nvarx:] = Cyy

    # Compute log determinants using Cholesky decomposition
    # Add small regularization to prevent numerical issues with identical data
    C += np.eye(C.shape[0]) * 1e-12
    Cxx += np.eye(Cxx.shape[0]) * 1e-12
    Cyy += np.eye(Cyy.shape[0]) * 1e-12

    chC = np.linalg.cholesky(C)
    chCxx = np.linalg.cholesky(Cxx)
    chCyy = np.linalg.cholesky(Cyy)

    # Sum of log diagonals
    HX = np.sum(np.log(np.diag(chCxx)))
    HY = np.sum(np.log(np.diag(chCyy)))
    HXY = np.sum(np.log(np.diag(chC)))

    ln2 = np.log(2.0)

    if biascorrect:
        # Bias correction terms
        psiterms = np.zeros(Nvarxy)
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0

        for i in range(Nvarxy):
            psiterms[i] = digamma_approx((Ntrl - i - 1.0) / 2.0) / 2.0

        HX = HX - Nvarx * dterm - np.sum(psiterms[:Nvarx])
        HY = HY - Nvary * dterm - np.sum(psiterms[:Nvary])
        HXY = HXY - Nvarxy * dterm - np.sum(psiterms)

    # MI in bits
    I = (HX + HY - HXY) / ln2
    return I


@njit
def cmi_ggg_jit(x, y, z, biascorrect=True, demeaned=False):
    """JIT-compiled conditional mutual information for Gaussian variables.

    Computes I(X;Y|Z) for continuous variables.

    Parameters
    ----------
    x : ndarray
        First variable (n_vars_x, n_samples).
    y : ndarray
        Second variable (n_vars_y, n_samples).
    z : ndarray
        Conditioning variable (n_vars_z, n_samples).
    biascorrect : bool
        Apply bias correction.
    demeaned : bool
        Whether data is already demeaned.

    Returns
    -------
    float
        Conditional mutual information in bits.
    """
    if x.shape[1] != y.shape[1] or x.shape[1] != z.shape[1]:
        raise ValueError("Number of samples must match")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarz = z.shape[0]
    Nvaryz = Nvary + Nvarz
    Nvarxz = Nvarx + Nvarz
    Nvarxyz = Nvarx + Nvary + Nvarz

    if not demeaned:
        # Demean data - manual implementation for numba
        for i in range(x.shape[0]):
            x[i] = x[i] - np.mean(x[i])
        for i in range(y.shape[0]):
            y[i] = y[i] - np.mean(y[i])
        for i in range(z.shape[0]):
            z[i] = z[i] - np.mean(z[i])

    # Compute all required covariance matrices
    Cxx = np.dot(x, x.T) / (Ntrl - 1)
    Cyy = np.dot(y, y.T) / (Ntrl - 1)
    Czz = np.dot(z, z.T) / (Ntrl - 1)
    Cxy = np.dot(x, y.T) / (Ntrl - 1)
    Cxz = np.dot(x, z.T) / (Ntrl - 1)
    Cyz = np.dot(y, z.T) / (Ntrl - 1)

    # Build joint covariance matrices
    # C(y,z)
    Cyz_joint = np.empty((Nvaryz, Nvaryz))
    Cyz_joint[:Nvary, :Nvary] = Cyy
    Cyz_joint[:Nvary, Nvary:] = Cyz
    Cyz_joint[Nvary:, :Nvary] = Cyz.T
    Cyz_joint[Nvary:, Nvary:] = Czz

    # C(x,z)
    Cxz_joint = np.empty((Nvarxz, Nvarxz))
    Cxz_joint[:Nvarx, :Nvarx] = Cxx
    Cxz_joint[:Nvarx, Nvarx:] = Cxz
    Cxz_joint[Nvarx:, :Nvarx] = Cxz.T
    Cxz_joint[Nvarx:, Nvarx:] = Czz

    # C(x,y,z)
    Cxyz = np.empty((Nvarxyz, Nvarxyz))
    Cxyz[:Nvarx, :Nvarx] = Cxx
    Cxyz[:Nvarx, Nvarx : Nvarx + Nvary] = Cxy
    Cxyz[:Nvarx, Nvarx + Nvary :] = Cxz
    Cxyz[Nvarx : Nvarx + Nvary, :Nvarx] = Cxy.T
    Cxyz[Nvarx : Nvarx + Nvary, Nvarx : Nvarx + Nvary] = Cyy
    Cxyz[Nvarx : Nvarx + Nvary, Nvarx + Nvary :] = Cyz
    Cxyz[Nvarx + Nvary :, :Nvarx] = Cxz.T
    Cxyz[Nvarx + Nvary :, Nvarx : Nvarx + Nvary] = Cyz.T
    Cxyz[Nvarx + Nvary :, Nvarx + Nvary :] = Czz

    # Compute log determinants
    # Add small regularization to prevent numerical issues with identical data
    Czz += np.eye(Czz.shape[0]) * 1e-12
    Cyz_joint += np.eye(Cyz_joint.shape[0]) * 1e-12
    Cxz_joint += np.eye(Cxz_joint.shape[0]) * 1e-12
    Cxyz += np.eye(Cxyz.shape[0]) * 1e-12

    chCz = np.linalg.cholesky(Czz)
    chCyz = np.linalg.cholesky(Cyz_joint)
    chCxz = np.linalg.cholesky(Cxz_joint)
    chCxyz = np.linalg.cholesky(Cxyz)

    HZ = np.sum(np.log(np.diag(chCz)))
    HYZ = np.sum(np.log(np.diag(chCyz)))
    HXZ = np.sum(np.log(np.diag(chCxz)))
    HXYZ = np.sum(np.log(np.diag(chCxyz)))

    ln2 = np.log(2.0)

    if biascorrect:
        # Bias correction
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0

        # Compute psi terms
        psiterms_z = np.zeros(Nvarz)
        psiterms_yz = np.zeros(Nvaryz)
        psiterms_xz = np.zeros(Nvarxz)
        psiterms_xyz = np.zeros(Nvarxyz)

        for i in range(Nvarz):
            psiterms_z[i] = digamma_approx((Ntrl - i - 1.0) / 2.0) / 2.0
        for i in range(Nvaryz):
            psiterms_yz[i] = digamma_approx((Ntrl - i - 1.0) / 2.0) / 2.0
        for i in range(Nvarxz):
            psiterms_xz[i] = digamma_approx((Ntrl - i - 1.0) / 2.0) / 2.0
        for i in range(Nvarxyz):
            psiterms_xyz[i] = digamma_approx((Ntrl - i - 1.0) / 2.0) / 2.0

        HZ = HZ - Nvarz * dterm - np.sum(psiterms_z)
        HYZ = HYZ - Nvaryz * dterm - np.sum(psiterms_yz)
        HXZ = HXZ - Nvarxz * dterm - np.sum(psiterms_xz)
        HXYZ = HXYZ - Nvarxyz * dterm - np.sum(psiterms_xyz)

    # CMI in bits: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    I = (HXZ + HYZ - HXYZ - HZ) / ln2
    return I


@njit
def digamma_approx(x):
    """Approximate digamma function for JIT compilation.

    Uses asymptotic expansion for x > 6 and recurrence for smaller values.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Approximate digamma value.
    """
    if x <= 0:
        return -np.inf

    # Use recurrence relation to get x > 6
    result = 0.0
    while x < 6:
        result -= 1.0 / x
        x += 1.0

    # Asymptotic expansion
    x_inv = 1.0 / x
    x_inv2 = x_inv * x_inv

    # psi(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)
    result += np.log(x) - 0.5 * x_inv - x_inv2 / 12.0 + x_inv2 * x_inv2 / 120.0

    return result


@njit
def gcmi_cc_jit(x, y):
    """JIT-compiled Gaussian-Copula MI between continuous variables.

    Full pipeline: copula transform -> normalize -> compute MI.

    Parameters
    ----------
    x : ndarray
        First variable (n_vars_x, n_samples).
    y : ndarray
        Second variable (n_vars_y, n_samples).

    Returns
    -------
    float
        GCMI in bits.
    """
    # Copula transform
    if x.ndim == 1:
        cx = np.empty((1, x.shape[0]))
        cx[0, :] = copnorm_jit(x)
    else:
        cx = copnorm_2d_jit(x)

    if y.ndim == 1:
        cy = np.empty((1, y.shape[0]))
        cy[0, :] = copnorm_jit(y)
    else:
        cy = copnorm_2d_jit(y)

    # Compute MI with bias correction
    return mi_gg_jit(cx, cy, biascorrect=True, demeaned=True)


@njit
def gccmi_ccd_jit(x, y, z, Zm):
    """JIT-compiled Gaussian-Copula CMI between 2 continuous variables 
    conditioned on a discrete variable.
    
    Parameters
    ----------
    x : ndarray
        First continuous variable (n_vars_x, n_samples).
    y : ndarray
        Second continuous variable (n_vars_y, n_samples).
    z : ndarray
        Discrete conditioning variable (n_samples,).
    Zm : int
        Number of discrete states (z values should be in [0, Zm-1]).
        
    Returns
    -------
    float
        Conditional mutual information in bits.
    """
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    
    # Compute marginal MI without conditioning
    Icond = np.zeros(Zm)
    Pz = np.zeros(Zm)
    
    # For each value of z
    for zi in range(Zm):
        # Find trials where z == zi
        idx = np.where(z == zi)[0]
        Pz[zi] = len(idx)
        
        if len(idx) > 0:
            # Extract conditional data
            x_cond = x[:, idx]
            y_cond = y[:, idx]
            
            # Copula transform conditional data
            if x_cond.shape[1] > 1:  # Need at least 2 samples for copula
                if Nvarx == 1:
                    cx_cond = np.empty((1, x_cond.shape[1]))
                    cx_cond[0, :] = copnorm_jit(x_cond[0, :])
                else:
                    cx_cond = copnorm_2d_jit(x_cond)
                    
                if Nvary == 1:
                    cy_cond = np.empty((1, y_cond.shape[1]))
                    cy_cond[0, :] = copnorm_jit(y_cond[0, :])
                else:
                    cy_cond = copnorm_2d_jit(y_cond)
                
                # Compute MI for this conditioning value
                Icond[zi] = mi_gg_jit(cx_cond, cy_cond, biascorrect=True, demeaned=True)
            else:
                # Not enough samples for this z value
                Icond[zi] = 0.0
    
    # Normalize probabilities
    Pz = Pz / Ntrl
    
    # Conditional mutual information
    CMI = np.sum(Pz * Icond)
    
    return CMI
