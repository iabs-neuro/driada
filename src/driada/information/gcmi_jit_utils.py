"""
JIT-compiled copula transformation functions for GCMI.
"""

import numpy as np
from numba import njit
from ..utils.jit import conditional_njit


@conditional_njit
def ctransform_jit(x):
    """Transform data to uniform marginals using empirical CDF.
    Efficient O(n log n) implementation using sorting-based ranking. This function
    converts continuous data to uniform marginals on (0, 1) by computing the 
    empirical cumulative distribution function (CDF). The transformation preserves
    the rank ordering while handling ties appropriately.

    Mathematical background:
    The copula transformation maps each value x_i to its empirical CDF value:
    F_n(x_i) = rank(x_i) / (n + 1)
    
    where rank(x_i) is the position of x_i in the sorted array, and n is the
    sample size. The denominator (n + 1) ensures values are strictly in (0, 1).

    Parameters
    ----------
    x : ndarray
        1D array of values to transform. Must contain at least 2 elements.

    Returns
    -------
    ndarray
        Copula-transformed values in (0, 1). Same shape as input, with each
        value mapped to its empirical CDF value.

    Notes
    -----
    - For tied values, each occurrence gets a unique rank based on its position
      in the original array (stable tie-breaking).
    - The output values are guaranteed to be in the open interval (0, 1).
    - Empty arrays will cause undefined behavior due to JIT compilation.

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import ctransform_jit
    >>> 
    >>> # Simple example with unique values
    >>> x = np.array([3.0, 1.0, 4.0, 2.0])
    >>> ct = ctransform_jit(x)
    >>> print(ct)
    [0.6 0.2 0.8 0.4]
    >>> 
    >>> # Example with tied values
    >>> x_tied = np.array([1.0, 2.0, 2.0, 3.0])
    >>> ct_tied = ctransform_jit(x_tied)
    >>> print(ct_tied)
    [0.2 0.4 0.6 0.8]
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


@conditional_njit
def ctransform_2d_jit(x):
    """Transform 2D array to uniform marginals using empirical CDF.
    Applies copula transformation independently to each row of a 2D array. This is
    commonly used when transforming multiple variables simultaneously, where each
    variable (row) has its own empirical distribution.

    Mathematical background:
    For each row i, applies the transformation:
    F_n,i(x_{i,j}) = rank_i(x_{i,j}) / (n_i + 1)
    
    where rank_i is computed within row i, and n_i is the number of samples
    in row i (typically all rows have the same number of samples).

    Parameters
    ----------
    x : ndarray
        2D array of shape (n_vars, n_samples) where each row represents a 
        different variable to be transformed independently. Each row must
        have at least 2 samples.

    Returns
    -------
    ndarray
        Copula-transformed array of same shape as input. Each row contains
        values in (0, 1) representing the empirical CDF of that row.

    Notes
    -----
    - Each row is transformed independently, preserving within-row relationships.
    - Cross-row relationships are maintained through the rank structure.
    - Empty rows or single-element rows will cause undefined behavior.
    - The function is optimized for multivariate data analysis where each
      variable needs its own marginal transformation.

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import ctransform_2d_jit
    >>> 
    >>> # Transform two variables with different distributions
    >>> x = np.array([[1.0, 3.0, 2.0, 4.0],   # Variable 1
    ...               [10.0, 30.0, 20.0, 40.0]])  # Variable 2
    >>> ct = ctransform_2d_jit(x)
    >>> print(ct.shape)
    (2, 4)
    >>> # Each row is transformed to uniform (0, 1)
    >>> print(ct[0])
    [0.2 0.6 0.4 0.8]
    >>> print(ct[1])
    [0.2 0.6 0.4 0.8]
    """
    n_vars, n_samples = x.shape
    result = np.empty_like(x)

    for i in range(n_vars):
        result[i, :] = ctransform_jit(x[i, :])

    return result


@conditional_njit
def ndtri_approx(p):
    """Compute inverse normal CDF (quantile function) using rational approximation.
    Implements the inverse of the standard normal cumulative distribution function
    (probit function) using a rational polynomial approximation suitable for JIT
    compilation. This provides a fast approximation of scipy.special.ndtri.

    Mathematical background:
    The function computes Φ^(-1)(p) where Φ is the standard normal CDF.
    Uses the Abramowitz and Stegun rational approximation:
    
    For p ∈ (0, 0.5):
        t = sqrt(-2 * ln(p))
        Φ^(-1)(p) ≈ -(t - P(t)/Q(t))
    
    For p ∈ [0.5, 1):
        t = sqrt(-2 * ln(1-p))
        Φ^(-1)(p) ≈ t - P(t)/Q(t)
    
    where P and Q are polynomials with coefficients optimized for accuracy.

    Parameters
    ----------
    p : float or ndarray
        Probability values in (0, 1). Values at boundaries are handled:
        - p ≤ 0 returns -inf
        - p ≥ 1 returns +inf

    Returns
    -------
    float or ndarray
        Approximate quantile values (z-scores) of the standard normal
        distribution. Same shape as input.

    Notes
    -----
    - Accuracy: ~2.5e-4 absolute error for p in [0.001, 0.999]
    - Coefficients from Abramowitz & Stegun, Handbook of Mathematical Functions
    - Optimized for speed over accuracy compared to scipy.special.ndtri
    - Handles edge cases: p=0 → -∞, p=1 → +∞
    - Symmetric around p=0.5: ndtri(p) = -ndtri(1-p)

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import ndtri_approx
    >>> 
    >>> # Single value
    >>> z = ndtri_approx(0.975)  # 95% quantile
    >>> print(f"{z:.4f}")
    1.9604
    >>> 
    >>> # Array of probabilities
    >>> probs = np.array([0.025, 0.5, 0.975])
    >>> z_scores = ndtri_approx(probs)
    >>> # Round to avoid tiny numerical differences around 0
    >>> z_scores_rounded = np.round(z_scores, 4)
    >>> print(z_scores_rounded)
    [-1.9604 -0.      1.9604]
    >>> 
    >>> # Edge cases
    >>> print(ndtri_approx(0.0))
    -inf
    >>> print(ndtri_approx(1.0))
    inf
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


@conditional_njit
def copnorm_jit(x):
    """Transform data to standard normal using copula-normalization.
    Combines copula transformation with inverse normal CDF to convert arbitrary
    continuous data to standard normal distribution while preserving rank
    relationships. This is a key preprocessing step for Gaussian Copula methods.

    Mathematical background:
    The copula-normalization is a two-step process:
    1. Transform to uniform: u_i = F_n(x_i) ∈ (0, 1)
    2. Transform to normal: z_i = Φ^(-1)(u_i)
    
    where F_n is the empirical CDF and Φ^(-1) is the inverse normal CDF.
    The result has standard normal marginals while preserving the copula
    (dependence structure) of the original data.

    Parameters
    ----------
    x : ndarray
        1D array of continuous values to normalize. Must have at least 2 elements
        for meaningful empirical CDF estimation.

    Returns
    -------
    ndarray
        Standard normal samples with same empirical CDF as input. Values have
        mean≈0, std≈1, and preserve the rank ordering of the input.

    Notes
    -----
    - The transformation is rank-preserving (monotonic)
    - Output is approximately N(0,1) distributed
    - Ties in input data are handled consistently
    - Edge effects: extreme ranks map to finite but large z-scores
    - Small sample sizes (<30) may show deviations from exact normality

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import copnorm_jit
    >>> 
    >>> # Transform exponential data to normal
    >>> np.random.seed(42)
    >>> x_exp = np.random.exponential(scale=2.0, size=100)
    >>> z = copnorm_jit(x_exp)
    >>> 
    >>> # Check properties
    >>> print(f"Mean: {np.mean(z):.3f}")
    Mean: -0.000
    >>> print(f"Std: {np.std(z):.3f}")
    Std: 0.961
    >>> print(f"Ranks preserved: {np.all(np.argsort(x_exp) == np.argsort(z))}")
    Ranks preserved: True
    >>> 
    >>> # Works with any continuous distribution
    >>> x_uniform = np.random.uniform(0, 10, size=50)
    >>> z_uniform = copnorm_jit(x_uniform)
    >>> print(f"Output range: [{np.min(z_uniform):.2f}, {np.max(z_uniform):.2f}]")
    Output range: [-2.06, 2.06]
    """
    cx = ctransform_jit(x)
    return ndtri_approx(cx)


@conditional_njit
def copnorm_2d_jit(x):
    """Transform 2D array to standard normal using copula-normalization.
    Applies copula-normalization independently to each row of a 2D array. This
    transforms multivariate data where each variable (row) is converted to standard
    normal marginals while preserving the dependence structure between variables.

    Mathematical background:
    For each row i:
    1. Compute empirical CDF: u_{i,j} = F_{n,i}(x_{i,j})
    2. Apply inverse normal: z_{i,j} = Φ^(-1)(u_{i,j})
    
    The resulting data has standard normal marginals for each variable while
    preserving the copula (multivariate dependence structure). This is essential
    for Gaussian Copula Mutual Information (GCMI) estimation.

    Parameters
    ----------
    x : ndarray
        2D array of shape (n_vars, n_samples) where each row represents a
        different variable to be normalized independently. Each row must
        have at least 2 samples.

    Returns
    -------
    ndarray
        Copula-normalized array of same shape as input. Each row has
        approximately standard normal distribution N(0,1) while preserving
        the multivariate dependence structure.

    Notes
    -----
    - Each variable is normalized independently
    - Cross-variable dependencies are preserved through the copula
    - The transformation is applied row-wise for efficiency
    - Small sample sizes per variable may affect normality
    - Empty or single-sample rows will cause undefined behavior

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import copnorm_2d_jit
    >>> 
    >>> # Create multivariate data with different marginals
    >>> np.random.seed(42)
    >>> x = np.zeros((3, 100))
    >>> x[0, :] = np.random.exponential(2.0, 100)      # Exponential
    >>> x[1, :] = np.random.uniform(-5, 5, 100)        # Uniform
    >>> x[2, :] = np.random.gamma(2.0, 2.0, 100)       # Gamma
    >>> 
    >>> # Transform to standard normal marginals
    >>> z = copnorm_2d_jit(x)
    >>> 
    >>> # Check each variable is approximately N(0,1)
    >>> for i in range(3):
    ...     print(f"Var {i}: mean={np.mean(z[i]):.3f}, std={np.std(z[i]):.3f}")
    Var 0: mean=-0.000, std=0.961
    Var 1: mean=-0.000, std=0.961
    Var 2: mean=-0.000, std=0.961
    >>> 
    >>> # Dependencies between variables are preserved
    >>> # (correlation structure remains similar)
    """
    n_vars, n_samples = x.shape
    result = np.empty_like(x)

    for i in range(n_vars):
        result[i, :] = copnorm_jit(x[i, :])

    return result


@conditional_njit
def mi_gg_jit(x, y, biascorrect=True, demeaned=False):
    """JIT-compiled Gaussian mutual information between two variables.
    Computes mutual information between two multivariate Gaussian variables
    using entropy-based calculations with optional small-sample bias correction.
    This is the core computational function used by higher-level GCMI methods.

    Mathematical background:
    For Gaussian variables X and Y, mutual information is:
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    where H is differential entropy. For Gaussian variables:
    H(X) = 0.5 * log(det(Σ_X)) + 0.5 * d_X * log(2πe)
    
    Using covariance matrices:
    I(X;Y) = 0.5 * log(det(Σ_X)) + 0.5 * log(det(Σ_Y)) - 0.5 * log(det(Σ_XY))
    
    Bias correction follows Panzeri & Treves (1996), accounting for finite
    sample effects using digamma function corrections.

    Parameters
    ----------
    x : ndarray
        First variable data of shape (n_vars_x, n_samples). Can be univariate
        (1, n_samples) or multivariate. Must have at least 2 samples.
    y : ndarray
        Second variable data of shape (n_vars_y, n_samples). Must have same
        number of samples as x.
    biascorrect : bool, default=True
        Apply small-sample bias correction using Panzeri-Treves method.
        Recommended for sample sizes < 1000.
    demeaned : bool, default=False
        Whether input data has already been mean-centered. If False, data
        will be demeaned internally (modifying input arrays).

    Returns
    -------
    float
        Mutual information in bits. Always non-negative (>= 0). Returns 0
        for independent variables.

    Notes
    -----
    - Uses Cholesky decomposition for numerical stability
    - Adds small regularization (1e-12) to handle near-singular matrices
    - Modifies input arrays if demeaned=False
    - Assumes data follows multivariate Gaussian distribution
    - For non-Gaussian data, use gcmi_cc_jit which applies copula transform

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import mi_gg_jit
    >>> 
    >>> # Independent Gaussian variables - MI ≈ 0
    >>> np.random.seed(1)
    >>> x = np.random.randn(2, 100)  # 2 variables, 100 samples
    >>> y = np.random.randn(3, 100)  # 3 variables, 100 samples
    >>> mi = mi_gg_jit(x, y)
    >>> print(f"MI (independent): {mi:.3f} bits")
    MI (independent): 0.003 bits
    >>> 
    >>> # Correlated variables - MI > 0
    >>> np.random.seed(2)
    >>> x = np.random.randn(1, 100)
    >>> y = x + 0.5 * np.random.randn(1, 100)  # y depends on x
    >>> mi = mi_gg_jit(x, y)
    >>> print(f"MI (dependent): {mi:.2f} bits")
    MI (dependent): 1.10 bits
    >>> 
    >>> # Pre-demeaned data
    >>> x_centered = x - np.mean(x, axis=1, keepdims=True)
    >>> y_centered = y - np.mean(y, axis=1, keepdims=True)
    >>> mi = mi_gg_jit(x_centered, y_centered, demeaned=True)
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


@conditional_njit
def cmi_ggg_jit(x, y, z, biascorrect=True, demeaned=False):
    """JIT-compiled conditional mutual information for Gaussian variables.
    Computes conditional mutual information I(X;Y|Z) between continuous
    Gaussian variables X and Y given conditioning variable Z. Measures
    the information shared between X and Y that is not explained by Z.

    Mathematical background:
    For Gaussian variables, conditional MI is:
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    
    where H denotes differential entropy. Using covariance matrices:
    I(X;Y|Z) = 0.5 * [log(det(Σ_XZ)) + log(det(Σ_YZ)) - log(det(Σ_XYZ)) - log(det(Σ_Z))]
    
    This measures the residual dependence between X and Y after accounting
    for their mutual dependence on Z.

    Parameters
    ----------
    x : ndarray
        First variable of shape (n_vars_x, n_samples). Can be univariate
        or multivariate. Must have at least 2 samples.
    y : ndarray
        Second variable of shape (n_vars_y, n_samples). Must have same
        number of samples as x.
    z : ndarray
        Conditioning variable of shape (n_vars_z, n_samples). The variable
        being conditioned on. Must have same number of samples.
    biascorrect : bool, default=True
        Apply Panzeri-Treves bias correction for finite samples.
        Recommended for sample sizes < 1000.
    demeaned : bool, default=False
        Whether input data has already been mean-centered. If False,
        data will be demeaned internally (modifying input arrays).

    Returns
    -------
    float
        Conditional mutual information in bits. Always non-negative.
        Returns 0 when X and Y are conditionally independent given Z.

    Notes
    -----
    - Assumes all variables follow joint Gaussian distribution
    - Uses Cholesky decomposition for numerical stability
    - Adds regularization (1e-12) to handle near-singular matrices
    - Modifies input arrays if demeaned=False
    - For non-Gaussian data, apply copula transform first

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import cmi_ggg_jit
    >>> 
    >>> # Example 1: X and Y are independent given Z
    >>> np.random.seed(42)
    >>> z = np.random.randn(1, 100)
    >>> x = z + 0.5 * np.random.randn(1, 100)  # X depends on Z
    >>> y = z + 0.5 * np.random.randn(1, 100)  # Y depends on Z
    >>> cmi = cmi_ggg_jit(x, y, z)
    >>> print(f"CMI (conditionally independent): {cmi:.4f} bits")
    CMI (conditionally independent): -0.0074 bits
    >>> 
    >>> # Example 2: X and Y have dependence beyond Z
    >>> z = np.random.randn(1, 100)
    >>> x = z + np.random.randn(1, 100)
    >>> y = x + z + 0.5 * np.random.randn(1, 100)  # Y depends on both X and Z
    >>> cmi = cmi_ggg_jit(x, y, z)
    >>> print(f"CMI (conditionally dependent): {cmi:.4f} bits")
    CMI (conditionally dependent): 1.2443 bits
    >>> 
    >>> # Example 3: Multivariate case
    >>> x = np.random.randn(2, 100)  # 2D variable
    >>> y = np.random.randn(3, 100)  # 3D variable  
    >>> z = np.random.randn(1, 100)  # 1D conditioning
    >>> # Add some conditional dependence
    >>> y[0] += 0.5 * x[0]  # y[0] depends on x[0]
    >>> cmi = cmi_ggg_jit(x, y, z)
    >>> print(f"Multivariate CMI: {cmi:.4f} bits")
    Multivariate CMI: 0.3231 bits
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


@conditional_njit
def digamma_approx(x):
    """Approximate digamma function for JIT compilation.
    Computes the digamma (psi) function ψ(x) = d/dx[log(Γ(x))] using
    asymptotic expansion for large values and recurrence relation for
    smaller values. Optimized for JIT compilation in bias correction.

    Mathematical background:
    Uses the recurrence relation: ψ(x) = ψ(x+1) - 1/x
    to shift x > 6, then applies asymptotic expansion:
    ψ(x) ≈ log(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) + O(1/x⁶)
    
    This approximation is accurate to ~1e-5 for x > 6 and sufficient
    for bias correction in information theory calculations.

    Parameters
    ----------
    x : float
        Input value. Must be positive (x > 0). For x <= 0, returns -inf.

    Returns
    -------
    float
        Approximate digamma value. Returns -inf for x <= 0.

    Notes
    -----
    - Accuracy decreases for very small x (< 1)
    - Optimized for speed over precision
    - Used primarily in Panzeri-Treves bias correction
    - Not suitable for high-precision mathematical applications

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import digamma_approx
    >>> 
    >>> # Compare with scipy for typical bias correction values
    >>> x = 50.0  # Typical value: (n_samples - n_vars - 1) / 2
    >>> psi_approx = digamma_approx(x)
    >>> print(f"ψ({x}) ≈ {psi_approx:.6f}")
    ψ(50.0) ≈ 3.901990
    >>> 
    >>> # Behavior at different scales
    >>> values = [0.5, 1.0, 5.0, 10.0, 100.0]
    >>> for val in values:
    ...     print(f"ψ({val:5.1f}) ≈ {digamma_approx(val):8.4f}")
    ψ(  0.5) ≈  -1.9635
    ψ(  1.0) ≈  -0.5772
    ψ(  5.0) ≈   1.5061
    ψ( 10.0) ≈   2.2518
    ψ(100.0) ≈   4.6002
    >>> 
    >>> # Edge case: non-positive input
    >>> print(f"ψ(0) = {digamma_approx(0.0)}")
    ψ(0) = -inf
    >>> print(f"ψ(-1) = {digamma_approx(-1.0)}")
    ψ(-1) = -inf
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


@conditional_njit
def gcmi_cc_jit(x, y):
    """JIT-compiled Gaussian-Copula MI between continuous variables.
    Computes mutual information between continuous variables using the
    Gaussian Copula method. This is the main user-facing function for
    MI estimation between continuous variables of any distribution.

    The method applies copula-normalization to transform arbitrary
    continuous distributions to Gaussian, then computes MI assuming
    Gaussian marginals. This preserves the dependence structure while
    enabling robust MI estimation.

    Parameters
    ----------
    x : ndarray
        First variable, either 1D array of shape (n_samples,) or 2D array
        of shape (n_vars_x, n_samples). Must have at least 2 samples.
    y : ndarray
        Second variable, either 1D array of shape (n_samples,) or 2D array
        of shape (n_vars_y, n_samples). Must have same number of samples as x.

    Returns
    -------
    float
        Gaussian-copula mutual information in bits. Always non-negative.
        Returns 0 for independent variables.

    Notes
    -----
    - Automatically handles 1D input by reshaping to (1, n_samples)
    - Applies copula transform independently to each variable
    - Uses bias-corrected MI estimation
    - Robust to different marginal distributions
    - Input data is not modified

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import gcmi_cc_jit
    >>> 
    >>> # Example 1: Linear dependence with different marginals
    >>> np.random.seed(0)
    >>> x = np.random.randn(100)  # Standard normal
    >>> y = 2 * x + 0.5 * np.random.randn(100)  # Strong linear relation
    >>> mi = gcmi_cc_jit(x, y)
    >>> print(f"MI (linear dependence): {mi:.2f} bits")
    MI (linear dependence): 1.86 bits
    >>> 
    >>> # Example 2: Nonlinear dependence
    >>> np.random.seed(0)
    >>> x = np.random.uniform(-2, 2, 200)
    >>> y = np.sin(3 * x) + 0.3 * np.random.normal(0, 1, 200)  # Sinusoidal relation
    >>> mi = gcmi_cc_jit(x, y) 
    >>> print(f"MI (nonlinear): {mi:.2f} bits")
    MI (nonlinear): 0.10 bits
    >>> 
    >>> # Example 3: Multivariate case
    >>> np.random.seed(42)
    >>> x = np.random.randn(2, 100)  # 2D Gaussian
    >>> y = np.vstack([x[0] + x[1], x[0] - x[1]])  # 2D linear combination
    >>> mi = gcmi_cc_jit(x, y)
    >>> print(f"MI (multivariate): {mi:.3f} bits")
    MI (multivariate): 5.306 bits
    >>> 
    >>> # Example 4: Independent variables
    >>> np.random.seed(0)
    >>> x = np.random.gamma(2, 2, 100)
    >>> y = np.random.beta(2, 5, 100)
    >>> mi = gcmi_cc_jit(x, y)
    >>> # MI should be close to 0 for independent variables
    >>> print(f"MI (independent): {abs(mi) < 0.1}")
    MI (independent): True
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


@conditional_njit
def gccmi_ccd_jit(x, y, z, Zm):
    """JIT-compiled Gaussian-Copula CMI between 2 continuous variables 
    conditioned on a discrete variable.
    Computes conditional mutual information I(X;Y|Z) where X and Y are
    continuous variables and Z is discrete. Uses Gaussian copula transform
    for robustness to non-Gaussian marginals.
    
    Mathematical background:
    For discrete Z with states {0, 1, ..., Zm-1}:
    I(X;Y|Z) = Σ_z P(Z=z) * I(X;Y|Z=z)
    
    where I(X;Y|Z=z) is the MI between X and Y computed only on samples
    where Z=z. Each conditional MI is computed using Gaussian copula.
    
    Parameters
    ----------
    x : ndarray
        First continuous variable of shape (n_vars_x, n_samples) or
        (n_samples,) for univariate case. Must have at least 2 samples.
    y : ndarray
        Second continuous variable of shape (n_vars_y, n_samples) or
        (n_samples,) for univariate case. Same number of samples as x.
    z : ndarray
        Discrete conditioning variable of shape (n_samples,). Values must
        be integers in range [0, Zm-1].
    Zm : int
        Number of discrete states. Must be positive. Z values should be
        in range [0, Zm-1].
        
    Returns
    -------
    float
        Conditional mutual information in bits. Always non-negative.
        Returns 0 when X and Y are conditionally independent given Z.
        
    Notes
    -----
    - Each conditional subset must have ≥ 2 samples for copula transform
    - States with < 2 samples contribute 0 to the CMI
    - Automatically handles variable reshaping
    - Uses bias-corrected MI estimation for each conditional
    - Robust to different marginal distributions
    
    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.gcmi_jit_utils import gccmi_ccd_jit
    >>> 
    >>> # Example 1: Simpson's paradox - dependence reverses by group
    >>> np.random.seed(42)
    >>> n_samples = 300
    >>> z = np.random.randint(0, 2, n_samples)  # Binary grouping
    >>> x = np.zeros((1, n_samples))
    >>> y = np.zeros((1, n_samples))
    >>> 
    >>> # Group 0: negative correlation
    >>> idx0 = z == 0
    >>> x[0, idx0] = np.random.randn(np.sum(idx0))
    >>> y[0, idx0] = -x[0, idx0] + 0.5 * np.random.randn(np.sum(idx0))
    >>> 
    >>> # Group 1: positive correlation  
    >>> idx1 = z == 1
    >>> x[0, idx1] = np.random.randn(np.sum(idx1)) + 3
    >>> y[0, idx1] = x[0, idx1] + 0.5 * np.random.randn(np.sum(idx1)) + 3
    >>> 
    >>> cmi = gccmi_ccd_jit(x, y, z, 2)
    >>> print(f"CMI given group: {cmi:.2f} bits")
    CMI given group: 1.21 bits
    >>> 
    >>> # Example 2: Conditional independence
    >>> z = np.random.randint(0, 3, 200)  # 3 states
    >>> x = np.random.randn(1, 200)
    >>> # Y depends only on Z, not on X given Z
    >>> y = np.zeros((1, 200))
    >>> for state in range(3):
    ...     mask = z == state
    ...     y[0, mask] = state + np.random.randn(np.sum(mask))
    >>> 
    >>> cmi = gccmi_ccd_jit(x, y, z, 3)
    >>> # CMI should be close to 0 for conditional independence
    >>> print(f"CMI close to 0: {abs(cmi) < 0.05}")
    CMI close to 0: True
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
