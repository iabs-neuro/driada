"""Gaussian Copula Mutual Information (GCMI) implementation.

This module implements mutual information and conditional mutual information
estimators using the Gaussian copula approach.

Credits:
    Based on the original GCMI implementation by Robin Ince:
    https://github.com/robince/gcmi
    
    Reference:
    Ince, R.A.A., Giordano, B.L., Kayser, C., Rousselet, G.A., Gross, J. and 
    Schyns, P.G. (2017). A statistical framework for neuroimaging data analysis 
    based on mutual information estimated via a gaussian copula. Human Brain 
    Mapping, 38(3), 1541-1573.
"""

import numpy as np
from numba import njit
import warnings
from scipy.special import ndtri, psi

from .info_utils import py_fast_digamma_arr
from ..utils.jit import conditional_njit


def _prepare_for_jit(*arrays):
    """Prepare arrays for JIT compilation by ensuring they are contiguous and float type.
    
    This helper function ensures arrays meet the requirements for efficient JIT compilation:
    - C-contiguous memory layout for cache efficiency
    - Float32 or float64 dtype for numerical operations
    
    Parameters
    ----------
    *arrays
        Variable number of arrays to prepare. Can be any dtype or memory layout.
        
    Returns
    -------
    tuple or ndarray
        If multiple arrays: tuple of prepared arrays in the same order as input.
        If single array: the prepared array directly (not wrapped in tuple).
        Each output array is C-contiguous with float32 or float64 dtype.
        
    Notes
    -----
    Arrays that already meet the requirements are returned unchanged.
    Arrays with non-float dtypes are converted to float64.
    Non-contiguous arrays are made contiguous via copy.
    
    Examples
    --------
    >>> x = np.array([1, 2, 3], dtype=np.int32)
    >>> x_prep = _prepare_for_jit(x)
    >>> x_prep.dtype == np.float64
    True
    >>> x_prep.flags.c_contiguous
    True
    
    >>> # Multiple arrays
    >>> x = np.array([[1, 2], [3, 4]], order='F')  # Fortran order
    >>> y = np.array([1.0, 2.0])
    >>> x_prep, y_prep = _prepare_for_jit(x, y)
    >>> x_prep.flags.c_contiguous
    True    """
    prepared = []
    for arr in arrays:
        # Ensure contiguous
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        # Ensure float type
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float64)
        prepared.append(arr)
    
    return tuple(prepared) if len(prepared) > 1 else prepared[0]

# Import JIT versions if available
try:
    from .gcmi_jit_utils import (
        ctransform_jit,
        ctransform_2d_jit,
        copnorm_jit,
        copnorm_2d_jit,
        mi_gg_jit,
        cmi_ggg_jit,
        gcmi_cc_jit,
        gccmi_ccd_jit,
    )

    _JIT_AVAILABLE = True
except ImportError:
    _JIT_AVAILABLE = False


def ctransform(x):
    """Copula transformation (empirical CDF).
    
    Transforms data to uniform marginals on (0,1) using the empirical
    cumulative distribution function. This is the first step in copula-based
    mutual information estimation.
    
    Parameters
    ----------
    x : ndarray
        Input data, shape (n_samples,) for 1D or (n_features, n_samples)
        for multivariate data. The transformation is applied along the last
        axis (samples).
        
    Returns
    -------
    ndarray
        Copula-transformed data with same shape as input. Values are in
        the open interval (0, 1), representing empirical CDF values.
        
    Notes
    -----
    The transformation maps each value to its rank divided by (n+1), where
    n is the number of samples. This ensures values are strictly between
    0 and 1 (never exactly 0 or 1), which is important for subsequent
    inverse normal CDF transformation.
    
    Automatically uses JIT-compiled version when available for better
    performance on large datasets.
    
    Examples
    --------
    >>> x = np.array([3.2, 1.1, 4.5, 2.3, 1.1])
    >>> result = ctransform(x)
    >>> np.round(result, 3)
    array([[0.667, 0.167, 0.833, 0.5  , 0.333]])
    
    >>> # 2D multivariate case
    >>> x = np.array([[1, 3, 2], [4, 2, 3]])
    >>> ctransform(x)  # Each row transformed independently
    array([[0.25, 0.75, 0.5 ],
           [0.75, 0.25, 0.5 ]])
    
    See Also
    --------
    ~driada.information.gcmi.copnorm : Complete copula normalization to standard normal    """
    x = np.atleast_2d(x)

    # Use JIT version if available
    if _JIT_AVAILABLE:
        x = _prepare_for_jit(x)
        
        if x.shape[0] == 1:
            # 1D case
            return ctransform_jit(x.ravel()).reshape(1, -1)
        else:
            # 2D case
            return ctransform_2d_jit(x)

    # Fallback to original implementation
    xi = np.argsort(x)
    xr = np.argsort(xi)
    cx = (xr + 1).astype(float) / (xr.shape[-1] + 1)
    return cx


def copnorm(x):
    """Copula normalization to standard normal distribution.
    
    Transforms data to have standard normal marginals while preserving
    the copula (dependence structure). This is the key preprocessing step
    for Gaussian Copula Mutual Information (GCMI) estimation.
    
    Parameters
    ----------
    x : ndarray
        Input data, shape (n_samples,) for 1D or (n_features, n_samples)
        for multivariate data. The transformation is applied along the last
        axis (samples).
        
    Returns
    -------
    ndarray
        Copula-normalized data with same shape as input. Each marginal
        distribution is transformed to standard normal N(0,1).
        
    Notes
    -----
    The two-step process:
    1. Apply empirical CDF transform to get uniform marginals (ctransform)
    2. Apply inverse normal CDF to get standard normal marginals
    
    This transformation is:
    - Robust to outliers (rank-based)
    - Preserves all dependence relationships
    - Makes data amenable to Gaussian-based MI estimation
    
    Automatically uses JIT-compiled version when available for better
    performance.
    
    Examples
    --------
    >>> # Transform non-Gaussian data
    >>> x = np.random.exponential(size=1000)
    >>> x_norm = copnorm(x)
    >>> # x_norm now has standard normal marginal distribution
    >>> np.abs(np.mean(x_norm)) < 0.1  # Close to 0
    True
    >>> np.abs(np.std(x_norm) - 1) < 0.1  # Close to 1
    True
    
    See Also
    --------
    ~driada.information.gcmi.ctransform : First step of copula normalization
    ~driada.information.gcmi.gcmi_cc : Uses copnorm internally for MI estimation
    
    References
    ----------
    Ince, R. A., et al. (2017). A statistical framework for neuroimaging data
    analysis based on mutual information estimated via a Gaussian copula.
    Human Brain Mapping, 38(3), 1541-1573.    """
    x = np.atleast_2d(x)

    # Use JIT version if available
    if _JIT_AVAILABLE:
        x = _prepare_for_jit(x)
            
        if x.shape[0] == 1:
            # 1D case
            return copnorm_jit(x.ravel()).reshape(1, -1)
        else:
            # 2D case
            return copnorm_2d_jit(x)

    # Fallback to original implementation
    # cx = sp.stats.norm.ppf(ctransform(x))
    cx = ndtri(ctransform(x))
    return cx


@conditional_njit
def demean(x):
    """Demean each row of a 2D array.
    
    Subtracts the mean from each row independently, resulting in rows with
    zero mean. This is a common preprocessing step for covariance calculations.

    Parameters
    ----------
    x : ndarray, shape (n_features, n_samples)
        2D array where each row (feature) is demeaned independently.
        Must be 2-dimensional.

    Returns
    -------
    ndarray
        Array with same shape as input where each row has zero mean.
        The operation is performed in-place on a copy of the input.
        
    Notes
    -----
    This function is JIT-compiled with numba for performance.
    The demeaning is performed row-wise, treating each row as a separate
    signal or feature to be centered.
    
    Examples
    --------
    >>> x = np.array([[1.0, 2.0, 3.0], 
    ...               [4.0, 5.0, 6.0]])
    >>> demean(x)
    array([[-1.,  0.,  1.],
           [-1.,  0.,  1.]])
    
    >>> # Verify zero mean
    >>> demeaned = demean(x)
    >>> np.allclose(demeaned.mean(axis=1), 0)
    True    """
    # Get the number of rows
    num_rows = x.shape[0]

    # Create an output array with the same shape as input
    demeaned_x = np.empty_like(x)

    # Demean each row
    for i in range(num_rows):
        row_mean = np.mean(x[i])
        demeaned_x[i] = x[i] - row_mean

    return demeaned_x


@conditional_njit
def regularized_cholesky(C, regularization=1e-12):
    """Compute Cholesky decomposition with regularization for numerical stability.

    Adds diagonal regularization to prevent issues with near-singular
    covariance matrices. Uses adaptive regularization for severely ill-conditioned
    matrices based on determinant check.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        Symmetric positive semi-definite matrix (typically a covariance matrix).
        Should be square and symmetric.
    regularization : float, optional
        Base regularization parameter added to diagonal. Default is 1e-12.
        For severely ill-conditioned matrices, adaptive regularization may
        apply a larger value.

    Returns
    -------
    L : ndarray, shape (n, n)
        Lower triangular Cholesky factor such that C ≈ L @ L.T.
        
    Notes
    -----
    The function detects ill-conditioning by comparing the determinant to an
    expected scale based on the trace. For severely ill-conditioned matrices
    (det < expected_scale * 1e-8), it applies stronger adaptive regularization.
    
    The regularization modifies the input as: C_reg = C + reg * I, where I is
    the identity matrix and reg is the regularization parameter.
    
    This function is JIT-compiled with numba for performance.
    
    Examples
    --------
    >>> # Well-conditioned matrix
    >>> C = np.array([[2.0, 1.0], [1.0, 2.0]])
    >>> L = regularized_cholesky(C)
    >>> np.allclose(L @ L.T, C, rtol=1e-10)
    True
    
    >>> # Near-singular matrix
    >>> C = np.array([[1.0, 0.99999], [0.99999, 1.0]])
    >>> L = regularized_cholesky(C)  # Applies regularization
    >>> # Result is stable despite near-singularity    """
    # Check matrix conditioning using determinant
    det_C = np.linalg.det(C)
    trace_C = np.trace(C)

    # Adaptive regularization based on determinant relative to trace
    # For near-singular matrices, det << trace^n where n is matrix size
    n = C.shape[0]
    expected_det_scale = (trace_C / n) ** n

    if det_C > 0 and det_C < expected_det_scale * 1e-8:  # Severely ill-conditioned
        # Use stronger regularization proportional to trace
        adaptive_reg = trace_C * 1e-8 / n  # Scale by matrix size
        reg = max(regularization, adaptive_reg)
    else:
        reg = regularization

    # Apply regularization
    C_reg = C + np.eye(C.shape[0]) * reg
    return np.linalg.cholesky(C_reg)


@conditional_njit()
def ent_g(x, biascorrect=True):
    """
    Entropy of a Gaussian variable in bits.
    
    Computes the differential entropy of a (possibly multidimensional) 
    Gaussian variable using the covariance matrix. Supports any number
    of features/dimensions.
    
    Parameters
    ----------
    x : array_like, shape (n_features, n_samples) or (n_samples,)
        Data array. If 2D, rows are features/dimensions and columns are samples.
        If 1D with shape (n,), it is converted to shape (1, n) representing 
        a single feature with n samples.
        Can handle any number of features (no limit on n_features).
    biascorrect : bool, optional
        Whether to apply bias correction for finite samples. Default is True.
        
    Returns
    -------
    float
        Entropy in bits. For a d-dimensional Gaussian with covariance C:
        H = 0.5 * log(det(2πeC)) / log(2)
        
    Raises
    ------
    ValueError
        If x is a 3D or higher dimensional array (must be 1D or 2D).
        
    Notes
    -----
    The bias correction uses digamma functions to account for finite sample
    effects in covariance estimation. Data is demeaned before computation.
    The function supports high-dimensional data with many features.
    
    This function is JIT-compiled with numba for performance.
    
    Examples
    --------
    >>> # 1D Gaussian entropy
    >>> rng = np.random.RandomState(42)
    >>> x = rng.randn(1000)  # Standard normal
    >>> h = ent_g(x)
    >>> # Theoretical entropy of N(0,1) is 0.5*log2(2*pi*e) ≈ 2.047
    >>> abs(h - 2.047) < 0.1
    True
    
    >>> # 2D Gaussian entropy
    >>> x = rng.randn(2, 1000)  # 2D standard normal
    >>> h = ent_g(x)
    >>> # Theoretical entropy is d*0.5*log2(2*pi*e) ≈ 2*2.047
    >>> abs(h - 4.094) < 0.2
    True
    
    >>> # Effect of correlation on entropy
    >>> x = rng.randn(2, 1000)
    >>> x[1] = 0.9 * x[0] + 0.1 * rng.randn(1000)  # Correlated
    >>> h_corr = ent_g(x)
    >>> h_uncorr = ent_g(rng.randn(2, 1000))
    >>> h_corr < h_uncorr  # Correlation reduces entropy
    True
    
    See Also
    --------
    ~driada.information.gcmi.mi_gg : Uses entropy to compute mutual information
    
    References
    ----------
    Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be a 1D or 2D array with shape (n_features, n_samples)")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    # demean data
    x = demean(x)
    # covariance
    C = np.dot(x, x.T) / float(Ntrl - 1)
    chC = regularized_cholesky(C)

    # entropy in nats
    # Extract diagonal manually for Numba compatibility
    diag_sum = 0.0
    for i in range(chC.shape[0]):
        diag_sum += np.log(chC[i, i])
    HX = diag_sum + 0.5 * Nvarx * (np.log(2 * np.pi) + 1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = (
            py_fast_digamma_arr(
                (Ntrl - np.arange(1, Nvarx + 1, dtype=np.float64)) / 2.0
            )
            / 2.0
        )
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HX = HX - Nvarx * dterm - psiterms.sum()

    # convert to bits
    return HX / ln2


@conditional_njit()
def mi_gg(x, y, biascorrect=True, demeaned=False):
    """Mutual information (MI) between two Gaussian variables in bits.

    Computes mutual information between two (possibly multidimensional)
    Gaussian variables using the entropy relation: I(X;Y) = H(X) + H(Y) - H(X,Y).
    Assumes variables follow a multivariate Gaussian distribution.

    Parameters
    ----------
    x : ndarray
        First variable, shape (n_features_x, n_samples) or (n_samples,) for 1D.
        Columns correspond to samples, rows to dimensions/variables.
    y : ndarray
        Second variable, shape (n_features_y, n_samples) or (n_samples,) for 1D.
        Must have same number of samples as x.
    biascorrect : bool, default=True
        Whether to apply bias correction to the MI estimate. Uses psi function
        (digamma) correction for finite sample bias in entropy estimation.
    demeaned : bool, default=False
        Whether input data already has zero mean. Set True if data has been
        copula-normalized or otherwise centered.

    Returns
    -------
    float
        Mutual information in bits. Always non-negative, with 0 indicating
        independence.

    Raises
    ------
    ValueError
        If x and y have different number of samples, or if inputs are not
        1D or 2D arrays.

    Notes
    -----
    This function assumes data follows a multivariate Gaussian distribution.
    For non-Gaussian data, use copula normalization first (via ctransform).
    
    The bias correction uses the Miller-Madow correction generalized to
    multivariate Gaussians, improving accuracy for small sample sizes.

    Examples
    --------
    >>> # Independent Gaussian variables
    >>> rng = np.random.RandomState(42)
    >>> x = rng.randn(1, 1000)  # 1D Gaussian
    >>> y = rng.randn(1, 1000)  # Independent 1D Gaussian
    >>> mi = mi_gg(x, y)
    >>> mi < 0.05  # Should be near 0 for independent variables
    True
    
    >>> # Correlated Gaussian variables
    >>> x = rng.randn(1, 1000)
    >>> y = 0.7 * x + 0.3 * rng.randn(1, 1000)  # Correlated
    >>> mi = mi_gg(x, y)
    >>> mi > 0.5  # Significant mutual information
    True
    
    >>> # Multidimensional case
    >>> x = rng.randn(3, 1000)  # 3D Gaussian
    >>> y = rng.randn(2, 1000)  # 2D Gaussian
    >>> # Create correlation: y depends on x
    >>> y[0] = 0.5 * x[0] + 0.5 * rng.randn(1000)
    >>> mi = mi_gg(x, y)
    >>> mi > 0.2  # Detects the dependency
    True
    
    See Also
    --------
    ~driada.information.gcmi.gcmi_cc : Gaussian-copula MI for arbitrary continuous distributions
    ~driada.information.gcmi.ent_g : Gaussian entropy used in MI calculation
    
    References
    ----------
    Ince, R. A., et al. (2017). A statistical framework for neuroimaging data
    analysis based on mutual information estimated via a Gaussian copula.
    Human Brain Mapping, 38(3), 1541-1573.    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError(
            "x and y must be 1D or 2D arrays"
        )
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx + Nvary

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xy = np.vstack((x, y))

    if not demeaned:
        xy = demean(xy)

    Cxy = np.dot(xy, xy.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cx = Cxy[:Nvarx, :Nvarx]
    Cy = Cxy[Nvarx:, Nvarx:]

    chCxy = regularized_cholesky(Cxy)
    chCx = regularized_cholesky(Cx)
    chCy = regularized_cholesky(Cy)

    # entropies in nats
    # normalizations cancel for mutual information
    HX = np.sum(np.log(np.diag(chCx)))  # + 0.5*Nvarx*(np.log(2*np.pi)+1.0)
    HY = np.sum(np.log(np.diag(chCy)))  # + 0.5*Nvary*(np.log(2*np.pi)+1.0)
    HXY = np.sum(np.log(np.diag(chCxy)))  # + 0.5*Nvarxy*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = py_fast_digamma_arr((Ntrl - np.arange(1, Nvarxy + 1)) / 2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HX = HX - Nvarx * dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary * dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy * dterm - psiterms[:Nvarxy].sum()

    # MI in bits
    I = (HX + HY - HXY) / ln2
    return max(0.0, I)  # MI is non-negative


def mi_model_gd(x, y, Ym=None, biascorrect=True, demeaned=False):
    """
    Mutual information between a Gaussian and a discrete variable in bits.
    
    Computes MI between a (possibly multidimensional) Gaussian variable x and 
    a discrete variable y using ANOVA-style model comparison. For 1D x this 
    provides a lower bound to the mutual information.
    
    Note: Each discrete class must have at least 2 samples for covariance 
    estimation. Classes with fewer samples will be skipped with a warning.
    
    Parameters
    ----------
    x : array_like, shape (n_features, n_samples) or (n_samples,)
        Gaussian variable data. If 2D, rows are features and columns are samples.
        If 1D with shape (n,), converted to shape (1, n) representing a single 
        feature with n samples.
    y : array_like, shape (n_samples,)
        Discrete variable containing integer values in range [0, max(y)].
        Must be 1D array with same number of samples as x.
    Ym : int, optional
        Number of discrete states. If None (default), automatically computed as
        np.max(y) + 1. Useful if y doesn't contain all possible states.
    biascorrect : bool, optional
        Whether to apply bias correction for finite samples. Default is True.
    demeaned : bool, optional
        Whether input data x already has zero mean (e.g., if copula-normalized).
        Default is False.
        
    Returns
    -------
    float
        Mutual information I(X;Y) in bits.
        
    Raises
    ------
    ValueError
        If x is 3D or higher dimensional array.
        If y is not a 1D array.
        If number of samples don't match between x and y.
        If Ym is not an integer.
        
    Warnings
    --------
    RuntimeWarning
        If any class has fewer than 2 samples. These classes will be skipped
        in the MI calculation as covariance estimation requires at least 2 
        samples per class.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Continuous data (2 variables, 100 samples)
    >>> x = np.random.randn(2, 100)
    >>> # Discrete labels (3 classes)
    >>> y = np.random.randint(0, 3, 100)
    >>> mi = mi_model_gd(x, y, Ym=3)
    >>> isinstance(mi, float)
    True
    >>> # Automatic Ym detection
    >>> mi_auto = mi_model_gd(x, y)
    >>> isinstance(mi_auto, float)
    True
        
    See Also
    --------
    ~driada.information.gcmi.mi_mixture_gd : MI estimation using mixture of Gaussians model.
    ~driada.information.gcmi.ent_g : Gaussian entropy estimation    """

    x = np.atleast_2d(x)
    # y = np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be a 1D or 2D array with shape (n_features, n_samples)")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    
    # Handle Ym parameter
    if Ym is None:
        # Auto-detect number of states
        Ym = int(np.max(y) + 1)
    else:
        # Validate and convert user-provided value
        if int(Ym) != Ym:
            raise ValueError("Ym should be an integer")
        Ym = int(Ym)
    
    # Check for classes with too few samples
    import warnings
    for yi in range(Ym):
        n_samples = np.sum(y == yi)
        if n_samples == 0:
            warnings.warn(f"Class {yi} has no samples and will be skipped", RuntimeWarning)
        elif n_samples == 1:
            warnings.warn(f"Class {yi} has only 1 sample and will be skipped. At least 2 samples per class are required for covariance estimation.", RuntimeWarning)

    # Call JIT-compiled implementation
    return _mi_model_gd_jit(x, y, Ym, biascorrect, demeaned)


@conditional_njit()
def _mi_model_gd_jit(x, y, Ym, biascorrect, demeaned):
    """JIT-compiled implementation of mi_model_gd.
    
    Parameters
    ----------
    x : ndarray
        Continuous data array of shape (n_features, n_trials)
    y : ndarray
        Discrete labels array of shape (n_trials,)
    Ym : int
        Number of unique discrete values in y
    biascorrect : bool
        Whether to apply bias correction
    demeaned : bool
        Whether the data is already demeaned
    
    Returns
    -------
    float
        Mutual information in bits
    """
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    if y.size != Ntrl:
        raise ValueError("number of trials do not match")
    """
    if not demeaned:
        x = x - x.mean(axis=1)[:,np.newaxis]
    """
    # class-conditional entropies
    Ntrl_y = np.zeros(Ym)
    Hcond = np.zeros(Ym)
    c = 0.5 * (np.log(2.0 * np.pi) + 1)

    for yi in range(Ym):
        idx = y == yi
        xm = x[:, idx]
        Ntrl_y[yi] = xm.shape[1]
        
        # Skip classes with too few samples
        if Ntrl_y[yi] < 2:
            # Set weight to 0 and entropy to 0 for this class
            Hcond[yi] = 0.0
            continue
            
        xm = demean(xm)
        Cm = np.dot(xm, xm.T) / float(Ntrl_y[yi] - 1)
        chCm = regularized_cholesky(Cm)
        Hcond[yi] = np.sum(np.log(np.diag(chCm)))  # + c*Nvarx

    # class weights
    w = Ntrl_y / float(Ntrl)

    # unconditional entropy from unconditional Gaussian fit
    Cx = np.dot(x, x.T) / float(Ntrl - 1)
    chC = regularized_cholesky(Cx)
    Hunc = np.sum(np.log(np.diag(chC)))  # + c*Nvarx

    ln2 = np.log(2)
    if biascorrect:
        vars = np.arange(1, Nvarx + 1)

        psiterms = py_fast_digamma_arr((Ntrl - vars) / 2.0) / 2.0
        dterm = (ln2 - np.log(float(Ntrl - 1))) / 2.0
        Hunc = Hunc - Nvarx * dterm - psiterms.sum()

        # For bias correction, handle each class separately
        for yi in range(Ym):
            if Ntrl_y[yi] < 2:
                # Skip bias correction for classes with insufficient samples
                continue
            dterm_yi = (ln2 - np.log(float(Ntrl_y[yi] - 1))) / 2.0
            psiterm_yi = 0.0
            for vi in vars:
                if Ntrl_y[yi] > vi:
                    idx = Ntrl_y[yi] - vi
                    psiterm_yi += py_fast_digamma_arr(np.array([idx / 2.0]))[0]
            Hcond[yi] = Hcond[yi] - Nvarx * dterm_yi - (psiterm_yi / 2.0)

    # MI in bits
    I = (Hunc - np.sum(w * Hcond)) / ln2
    return max(0.0, I)  # MI is non-negative


def gcmi_cc(x, y):
    """Gaussian-Copula Mutual Information between two continuous variables.
    
    Main user-facing function for computing mutual information between
    continuous variables using the Gaussian Copula MI (GCMI) method.
    Handles arbitrary continuous distributions by transforming marginals
    to Gaussian via copula normalization.
    
    Parameters
    ----------
    x : ndarray
        First continuous variable, shape (n_features_x, n_samples) or (n_samples,) for 1D.
        If 2D, rows are features and columns are samples. If 1D, treated as
        single feature with multiple samples.
    y : ndarray
        Second continuous variable, shape (n_features_y, n_samples) or (n_samples,) for 1D.
        Must have same number of samples as x.
        
    Returns
    -------
    float
        Mutual information in bits. Always non-negative, with 0 indicating
        independence. Provides a lower bound to the true MI.
        
    Notes
    -----
    The GCMI method:
    1. Transforms each variable to standard normal marginals using the
       empirical CDF (copula transform)
    2. Computes MI under the Gaussian copula assumption
    3. Applies bias correction for finite samples
    
    This approach is:
    - Robust to outliers due to rank-based transform
    - Computationally efficient (no density estimation)
    - Provides MI lower bound (exact for jointly Gaussian data)
    - Suitable for continuous neural data (firing rates, LFP, etc.)
    
    For discrete variables, use gcmi_cd or gcmi_ccd instead.
    
    Examples
    --------
    >>> # Linear relationship with non-Gaussian marginals
    >>> rng = np.random.RandomState(42)
    >>> x = rng.exponential(size=1000)  # Non-Gaussian
    >>> y = 2 * x + rng.normal(0, 0.5, size=1000)
    >>> mi = gcmi_cc(x, y)
    >>> mi > 1.0  # Detects strong dependency
    True
    
    >>> # Monotonic nonlinear relationship
    >>> x = rng.exponential(size=1000)
    >>> y = np.log(x + 1) + rng.normal(0, 0.1, size=1000)
    >>> mi = gcmi_cc(x, y)
    >>> mi > 0.5  # Detects monotonic dependency
    True
    
    >>> # Multidimensional example
    >>> x = rng.randn(3, 1000)  # 3D variable
    >>> y = rng.randn(2, 1000)  # 2D variable
    >>> # Create dependency
    >>> y[0] = 0.5 * x[0] + 0.3 * x[1] + 0.2 * rng.randn(1000)
    >>> mi = gcmi_cc(x, y)
    >>> mi > 0.3  # Detects multivariate dependency
    True
    
    See Also
    --------
    ~driada.information.gcmi.mi_gg : MI for Gaussian variables (without copula transform)
    ~driada.information.gcmi.gccmi_ccd : Conditional MI with discrete conditioning variable
    
    References
    ----------
    Ince, R. A., et al. (2017). A statistical framework for neuroimaging data
    analysis based on mutual information estimated via a Gaussian copula.
    Human Brain Mapping, 38(3), 1541-1573.    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # Use JIT version if available and suitable
    if (
        _JIT_AVAILABLE
        and x.flags.c_contiguous
        and y.flags.c_contiguous
        and x.dtype in (np.float32, np.float64)
        and y.dtype in (np.float32, np.float64)
    ):
        return gcmi_cc_jit(x, y)

    """
    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break
    """

    # copula normalization
    cx = copnorm(x)
    cy = copnorm(y)
    # parametric Gaussian MI
    I = mi_gg(cx, cy, True, True)
    return max(0.0, I)  # MI is non-negative


# Note: All functions below have been integrated with numba JIT compilation
def cmi_ggg(x, y, z, biascorrect=True, demeaned=False):
    """
    Conditional mutual information between two Gaussian variables given a third.
    
    Computes CMI between two (possibly multidimensional) Gaussian variables 
    x and y, conditioned on a third variable z. Uses entropy decomposition:
    I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z).
    
    Parameters
    ----------
    x : array_like, shape (n_features_x, n_samples) or (n_samples,)
        First Gaussian variable. If 2D, rows are features and columns are samples.
        If 1D with shape (n,), converted to shape (1, n).
    y : array_like, shape (n_features_y, n_samples) or (n_samples,)
        Second Gaussian variable. Must have same number of samples as x.
    z : array_like, shape (n_features_z, n_samples) or (n_samples,)
        Conditioning Gaussian variable. Must have same number of samples as x and y.
    biascorrect : bool, optional
        Whether to apply bias correction for finite samples. Default is True.
    demeaned : bool, optional
        Whether input data already has zero mean (e.g., if copula-normalized).
        Default is False.
        
    Returns
    -------
    float
        Conditional mutual information I(X;Y|Z) in bits.
        
    Raises
    ------
    ValueError
        If x, y, or z are 3D or higher dimensional arrays.
        If number of samples don't match between x, y, and z.
    
    Notes
    -----
    Conditional mutual information measures the dependency between X and Y 
    after accounting for the influence of Z. If X and Y are independent 
    given Z, then I(X;Y|Z) = 0.
    
    This function assumes all variables follow a joint Gaussian distribution.
    For non-Gaussian data, apply copula normalization first.
    
    The entropy decomposition uses:
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    
    Examples
    --------
    >>> # Independent given Z
    >>> rng = np.random.RandomState(42)
    >>> z = rng.randn(1, 1000)
    >>> x = z + 0.5 * rng.randn(1, 1000)  # X depends on Z
    >>> y = z + 0.5 * rng.randn(1, 1000)  # Y depends on Z
    >>> # X and Y are correlated through Z
    >>> mi_uncond = mi_gg(x, y)
    >>> mi_uncond > 0.3  # Significant MI without conditioning
    True
    >>> # But independent given Z
    >>> cmi = cmi_ggg(x, y, z)
    >>> cmi < 0.05  # Near zero when conditioned on Z
    True
    
    >>> # Direct dependency not explained by Z
    >>> z = rng.randn(1, 1000)
    >>> x = rng.randn(1, 1000)
    >>> y = 0.5 * x + 0.3 * z + 0.2 * rng.randn(1, 1000)
    >>> cmi = cmi_ggg(x, y, z)
    >>> cmi > 0.3  # Still significant after conditioning
    True
    
    See Also
    --------
    ~driada.information.gcmi.mi_gg : Unconditional mutual information
    ~driada.information.gcmi.gccmi_ccd : CMI with discrete conditioning variable
    
    References
    ----------
    Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)

    # Use JIT version if available
    if _JIT_AVAILABLE:
        x, y, z = _prepare_for_jit(x, y, z)
        return cmi_ggg_jit(x, y, z, biascorrect, demeaned)

    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarz = z.shape[0]
    Nvaryz = Nvary + Nvarz
    Nvarxy = Nvarx + Nvary
    Nvarxz = Nvarx + Nvarz
    Nvarxyz = Nvarx + Nvaryz

    if y.shape[1] != Ntrl or z.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xyz = np.vstack((x, y, z))
    if not demeaned:
        xyz = xyz - xyz.mean(axis=1)[:, np.newaxis]
    Cxyz = np.dot(xyz, xyz.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cz = Cxyz[Nvarxy:, Nvarxy:]
    Cyz = Cxyz[Nvarx:, Nvarx:]
    Cxz = np.zeros((Nvarxz, Nvarxz))
    Cxz[:Nvarx, :Nvarx] = Cxyz[:Nvarx, :Nvarx]
    Cxz[:Nvarx, Nvarx:] = Cxyz[:Nvarx, Nvarxy:]
    Cxz[Nvarx:, :Nvarx] = Cxyz[Nvarxy:, :Nvarx]
    Cxz[Nvarx:, Nvarx:] = Cxyz[Nvarxy:, Nvarxy:]

    chCz = regularized_cholesky(Cz)
    chCxz = regularized_cholesky(Cxz)
    chCyz = regularized_cholesky(Cyz)
    chCxyz = regularized_cholesky(Cxyz)

    # entropies in nats
    # normalizations cancel for cmi
    HZ = np.sum(np.log(np.diagonal(chCz)))  # + 0.5*Nvarz*(np.log(2*np.pi)+1.0)
    HXZ = np.sum(np.log(np.diagonal(chCxz)))  # + 0.5*Nvarxz*(np.log(2*np.pi)+1.0)
    HYZ = np.sum(np.log(np.diagonal(chCyz)))  # + 0.5*Nvaryz*(np.log(2*np.pi)+1.0)
    HXYZ = np.sum(np.log(np.diagonal(chCxyz)))  # + 0.5*Nvarxyz*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = psi((Ntrl - np.arange(1, Nvarxyz + 1)).astype(float) / 2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HZ = HZ - Nvarz * dterm - psiterms[:Nvarz].sum()
        HXZ = HXZ - Nvarxz * dterm - psiterms[:Nvarxz].sum()
        HYZ = HYZ - Nvaryz * dterm - psiterms[:Nvaryz].sum()
        HXYZ = HXYZ - Nvarxyz * dterm - psiterms[:Nvarxyz].sum()

    # MI in bits
    I = (HXZ + HYZ - HXYZ - HZ) / ln2
    return max(0.0, I)  # CMI is non-negative


def gccmi_ccd(x, y, z, Zm=None):
    """Gaussian-Copula CMI between 2 continuous variables conditioned on a discrete variable.

    Calculates the conditional mutual information (CMI) between two continuous 
    variables x and y, conditioned on a discrete variable z, using a Gaussian 
    copula approach. This method can handle multivariate continuous variables.
    
    The Gaussian copula transforms the marginal distributions to standard Gaussian
    while preserving the dependence structure, allowing efficient estimation of
    mutual information.

    Parameters
    ----------
    x : array_like, shape (n_features_x, n_samples) or (n_samples,)
        First continuous variable. If multivariate, features are in rows and 
        samples in columns.
    y : array_like, shape (n_features_y, n_samples) or (n_samples,)
        Second continuous variable. If multivariate, features are in rows and 
        samples in columns.
    z : array_like, shape (n_samples,)
        Discrete conditioning variable. Must contain integer values in the range 
        [0, max(z)] (inclusive).
    Zm : int, optional
        Number of unique values in the discrete variable z. If None (default),
        it is automatically computed as len(np.unique(z)). Providing this value
        can be useful if you know z doesn't contain all possible values.

    Returns
    -------
    I : float
        Conditional mutual information I(X;Y|Z) in bits.

    Raises
    ------
    ValueError
        If x or y have more than 2 dimensions.
        If z is not a 1D array.
        If z does not contain integer values.
        If the number of samples doesn't match across inputs.

    Notes
    -----
    The function uses a Gaussian copula transformation to estimate CMI. For each
    value of the discrete conditioning variable z, it transforms the conditional
    distributions of x and y to Gaussian, then computes the CMI using entropy
    calculations on the Gaussian-transformed data.
    
    This is particularly useful for analyzing neural data where the conditioning
    variable might represent experimental conditions, stimulus types, or 
    behavioral states.

    Examples
    --------
    >>> # Continuous variables with dependency modulated by discrete state
    >>> rng = np.random.RandomState(42)
    >>> n_samples = 3000
    >>> z = rng.choice([0, 1, 2], size=n_samples)  # 3 discrete states
    >>> x = np.zeros(n_samples)
    >>> y = np.zeros(n_samples)
    >>> 
    >>> # Different relationships for each state
    >>> for state in [0, 1, 2]:
    ...     mask = z == state
    ...     n_state = mask.sum()
    ...     if state == 0:  # Independent in state 0
    ...         x[mask] = rng.randn(n_state)
    ...         y[mask] = rng.randn(n_state)
    ...     elif state == 1:  # Linear relationship in state 1
    ...         x[mask] = rng.randn(n_state)
    ...         y[mask] = 0.8 * x[mask] + 0.2 * rng.randn(n_state)
    ...     else:  # Nonlinear in state 2
    ...         x[mask] = rng.uniform(-2, 2, n_state)
    ...         y[mask] = x[mask]**2 + 0.5 * rng.randn(n_state)
    >>> 
    >>> # Reshape for function input
    >>> x = x.reshape(1, -1)
    >>> y = y.reshape(1, -1)
    >>> 
    >>> # CMI captures state-dependent relationships
    >>> cmi = gccmi_ccd(x, y, z)
    >>> cmi > 0.2  # Significant CMI due to state-dependent coupling
    True
    
    >>> # Compare with unconditional MI
    >>> mi_uncond = gcmi_cc(x, y)
    >>> cmi > mi_uncond * 0.8  # CMI captures most of the dependency
    True

    See Also
    --------
    ~driada.information.gcmi.gcmi_cc : Unconditional Gaussian-copula MI
    ~driada.information.gcmi.cmi_ggg : CMI for all-continuous variables
    ~driada.information.gcmi.gcmi_cd : MI between continuous and discrete variables
    
    References
    ----------
    Ince, R. A., et al. (2017). A statistical framework for neuroimaging data
    analysis based on mutual information estimated via a Gaussian copula.
    Human Brain Mapping, 38(3), 1541-1573.    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    if z.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(z.dtype, np.integer):
        raise ValueError("z should be an integer array")
    
    # Compute Zm if not provided
    if Zm is None:
        Zm = len(np.unique(z))
    elif not isinstance(Zm, int):
        raise ValueError("Zm should be an integer")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]

    if y.shape[1] != Ntrl or z.size != Ntrl:
        raise ValueError("number of trials do not match")

    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi, :]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi, :]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break

    # check values of discrete variable
    if z.min() != 0 or z.max() != (Zm - 1):
        raise ValueError("values of discrete variable z are out of bounds")
    
    # Use JIT version if available
    if _JIT_AVAILABLE:
        x, y = _prepare_for_jit(x, y)
        return gccmi_ccd_jit(x, y, z, Zm)

    # calculate gcmi for each z value
    Icond = np.zeros(Zm)
    Pz = np.zeros(Zm)
    cx = []
    cy = []
    for zi in range(Zm):
        idx = z == zi
        thsx = copnorm(x[:, idx])
        thsy = copnorm(y[:, idx])
        Pz[zi] = idx.sum()
        cx.append(thsx)
        cy.append(thsy)
        Icond[zi] = mi_gg(thsx, thsy, True, True)

    Pz = Pz / float(Ntrl)

    # conditional mutual information
    CMI = np.sum(Pz * Icond)
    # I = mi_gg(np.hstack(cx),np.hstack(cy),True,False)
    return max(0.0, CMI)  # CMI is non-negative
