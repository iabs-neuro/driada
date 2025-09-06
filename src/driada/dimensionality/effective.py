from .utils import correct_cov_spectrum, eigh
import numpy as np
from scipy.stats import entropy
import warnings
from ..utils.data import correlation_matrix

DATA_SHAPE_THR = (
    0.01  # if n/t in multivariate time series data is more than DATA_SHAPE_THR,
)
# cov/corr spectrum may be significantly distorted, correction is recommended


def _eff_dim(corr_eigs, q=2):
    """Compute effective dimension using Renyi entropy.
    
    Parameters
    ----------
    corr_eigs : array-like
        Eigenvalues of correlation/covariance matrix. Must be non-negative.
    q : float, default=2
        Order of Renyi entropy. Common values:
        - q=1: Shannon entropy (standard entropy)
        - q=2: Quadratic entropy (collision entropy)
        - q=inf: Min-entropy
        
    Returns
    -------
    float
        Effective dimension based on entropy of eigenvalue distribution.
        
    Raises
    ------
    ValueError
        If q < 0 or if eigenvalues are invalid.
        
    Notes
    -----
    The effective dimension quantifies the spread of variance across
    principal components. Higher values indicate more distributed variance.
    
    References
    ----------
    Rényi, A. (1961). On measures of entropy and information. Proceedings
    of the Fourth Berkeley Symposium on Mathematical Statistics and
    Probability, Volume 1: Contributions to the Theory of Statistics.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Example with eigenvalues from PCA
    >>> eigenvalues = np.array([5.0, 3.0, 1.0, 0.5, 0.3, 0.2])
    >>> eff_d = _eff_dim(eigenvalues, q=2)
    >>> print(f"Effective dimension (q=2): {eff_d:.2f}")
    Effective dimension (q=2): 2.83
    """
    if q < 0:
        raise ValueError("Renyi entropy is undefined for q<0")
    
    corr_eigs = np.asarray(corr_eigs)
    
    # Validate eigenvalues
    if len(corr_eigs) == 0:
        raise ValueError("Empty eigenvalue array")
    
    if np.any(corr_eigs < -1e-10):
        raise ValueError("Eigenvalues must be non-negative")
    
    # Clean up numerical noise
    corr_eigs = np.maximum(corr_eigs, 0)
    eig_sum = np.sum(corr_eigs)
    
    if eig_sum == 0:
        raise ValueError("All eigenvalues are zero")

    if q == 1:
        # Shannon entropy
        norm_corr_eigs = corr_eigs / eig_sum
        # Remove zero eigenvalues to avoid log(0)
        norm_corr_eigs = norm_corr_eigs[norm_corr_eigs > 0]
        return 2 ** entropy(norm_corr_eigs, base=2)

    elif q == 2:
        # Quadratic entropy
        return (eig_sum ** 2) / np.sum(corr_eigs ** 2)

    elif q == np.inf:
        # Min-entropy
        return eig_sum / np.max(corr_eigs)

    else:
        # General Renyi entropy
        norm_corr_eigs = corr_eigs / eig_sum
        # Remove zeros to avoid 0^q issues
        norm_corr_eigs = norm_corr_eigs[norm_corr_eigs > 0]
        return np.exp((1.0 / (1.0 - q)) * np.log(np.sum(norm_corr_eigs ** q)))


def eff_dim(data, enable_correction, q=2, **correction_kwargs):
    """Compute effective dimension of multivariate data.
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input data matrix where rows are samples and columns are features.
        This follows the standard scikit-learn convention.
    enable_correction : bool
        Whether to apply finite-sample spectrum correction to eigenvalues.
    q : float, default=2
        Order of Renyi entropy (see _eff_dim for details).
    **correction_kwargs
        Additional arguments for spectrum correction (see correct_cov_spectrum).
        
    Returns
    -------
    float
        Effective dimension of the data.
        
    Raises
    ------
    ValueError
        If data has invalid shape or if computation fails.
        
    Notes
    -----
    When n_features/n_samples > 0.01, spectrum correction is recommended
    to account for finite-sample biases in eigenvalue estimation.
    
    The data should be organized with samples as rows and features as
    columns, following the standard machine learning convention. For neural
    data, this means timepoints as rows and neurons as columns.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Generate data with 3 effective dimensions
    >>> n_samples, n_features = 1000, 50
    >>> np.random.seed(42)  # For reproducible results
    >>> latent = np.random.randn(n_samples, 3)
    >>> mixing = np.random.randn(3, n_features)
    >>> data = latent @ mixing + 0.1 * np.random.randn(n_samples, n_features)
    >>> 
    >>> # Compute effective dimension
    >>> eff_d = eff_dim(data, enable_correction=False, q=2)
    >>> print(f"Effective dimension: {eff_d:.2f}")  # Should be close to 3
    Effective dimension: 3.00
    """
    data = np.asarray(data)
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    
    # Transpose to get (n_variables, n_observations) for internal computation
    data_t = data.T
    n, t = data_t.shape
    
    if t < 2:
        raise ValueError(f"Need at least 2 samples, got {t}")
    
    if 1.0 * n / t > DATA_SHAPE_THR and not enable_correction:
        warnings.warn(
            f"n_features/n_samples ratio is {1.0*n/t:.3f}, which is bigger than {DATA_SHAPE_THR}. "
            "Spectrum correction is recommended (set enable_correction=True)"
        )

    cmat = correlation_matrix(data_t)
    
    # Check for NaN in correlation matrix
    if np.any(np.isnan(cmat)):
        raise ValueError("Correlation matrix contains NaN values. "
                        "Check for constant features or insufficient data.")
    
    if enable_correction:
        corrected_eigs = correct_cov_spectrum(n, t, cmat, **correction_kwargs)
        final_eigs = corrected_eigs[-1]
    else:
        final_eigs = eigh(cmat, eigvals_only=True)

    return _eff_dim(final_eigs, q=q)
