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
    Cxyz[:Nvarx, Nvarx:Nvarx+Nvary] = Cxy
    Cxyz[:Nvarx, Nvarx+Nvary:] = Cxz
    Cxyz[Nvarx:Nvarx+Nvary, :Nvarx] = Cxy.T
    Cxyz[Nvarx:Nvarx+Nvary, Nvarx:Nvarx+Nvary] = Cyy
    Cxyz[Nvarx:Nvarx+Nvary, Nvarx+Nvary:] = Cyz
    Cxyz[Nvarx+Nvary:, :Nvarx] = Cxz.T
    Cxyz[Nvarx+Nvary:, Nvarx:Nvarx+Nvary] = Cyz.T
    Cxyz[Nvarx+Nvary:, Nvarx+Nvary:] = Czz
    
    # Compute log determinants
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