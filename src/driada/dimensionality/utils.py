from scipy.stats import pearsonr, norm
from scipy.linalg import eigh
import numpy as np
import warnings


def res_var_metric(all_dists, emb_dists):
    """Compute residual variance metric (1 - R²) between distance arrays.
    
    Parameters
    ----------
    all_dists : array-like
        Original distances between points.
    emb_dists : array-like  
        Embedding distances between corresponding points.
        
    Returns
    -------
    float
        Residual variance metric (1 - R²) where R is Pearson correlation.
        Values close to 0 indicate good preservation of distances.
        
    Raises
    ------
    ValueError
        If arrays have different lengths or contain invalid values.
        
    Notes
    -----
    This metric quantifies how much variance in the original distances
    is NOT explained by the embedding distances.    """
    all_dists = np.asarray(all_dists)
    emb_dists = np.asarray(emb_dists)
    
    if len(all_dists) != len(emb_dists):
        raise ValueError(f"Distance arrays must have same length: "
                        f"{len(all_dists)} vs {len(emb_dists)}")
    
    if len(all_dists) < 2:
        raise ValueError("Need at least 2 distance pairs")
    
    # Check for constant arrays
    if np.std(all_dists) == 0 or np.std(emb_dists) == 0:
        return np.nan
        
    r, p_value = pearsonr(all_dists, emb_dists)
    return 1 - r**2


def correct_cov_spectrum(
    N, T, cmat, correction_iters=10, ensemble_size=1, min_eigenvalue=1e-10
):
    """
    Correct the spectrum of a covariance/correlation matrix for finite-sample bias.

    This function implements an iterative algorithm to correct eigenvalue biases
    that arise when estimating covariance matrices from finite samples.

    Parameters
    ----------
    N : int
        Number of variables (neurons).
    T : int
        Number of time points (observations).
    cmat : ndarray of shape (N, N)
        Covariance or correlation matrix to correct.
    correction_iters : int, default=10
        Number of correction iterations to perform.
    ensemble_size : int, default=1
        Size of the ensemble for phase 1 averaging.
    min_eigenvalue : float, default=1e-10
        Minimum eigenvalue threshold to avoid numerical issues.

    Returns
    -------
    corrected_eigs : list of ndarray
        List containing eigenvalue arrays for each iteration, including
        the initial eigenvalues at index 0.
        
    Warns
    -----
    UserWarning
        If significant negative eigenvalues are found in the input matrix.
        
    Notes
    -----
    The algorithm performs iterative bias correction using random matrix theory.
    Each iteration consists of two phases:
    1. Ensemble averaging to estimate bias ratios
    2. Eigenvalue update using the bias estimates
    
    References
    ----------
    Duan, J., Popescu, I., & Matzinger, H. (2022). Recover the spectrum of 
    covariance matrix: a non-asymptotic iterative method. arXiv preprint 
    arXiv:2201.00230.    """
    eigs = eigh(cmat, eigvals_only=True)

    # Check for negative eigenvalues and clip them
    if np.any(eigs < 0):
        neg_fraction = np.sum(eigs < 0) / len(eigs)
        min_eig = np.min(eigs)
        if min_eig < -1e-6:  # Significant negative eigenvalue
            warnings.warn(
                f"Found significant negative eigenvalues (min={min_eig:.2e}). "
                f"{neg_fraction:.1%} of eigenvalues are negative. "
                "This may indicate numerical precision issues with the correlation matrix."
            )
        eigs = np.maximum(eigs, min_eigenvalue)

    init_eigs = eigs.copy()
    iter_eigs = eigs.copy()
    corrected_eigs = [init_eigs]

    for i in range(correction_iters):
        all_ratios = np.zeros((ensemble_size, N))

        # phase 1
        for j in range(ensemble_size):
            M = norm.rvs(size=(N, T))
            # Ensure eigenvalues are non-negative before taking sqrt
            iter_eigs = np.maximum(iter_eigs, min_eigenvalue)
            L = np.diag(np.sqrt(iter_eigs))
            M2 = L @ M @ M.T @ L / T
            ps_eigs = eigh(M2)[0]
            # Clip ps_eigs to avoid division issues
            ps_eigs = np.maximum(ps_eigs, min_eigenvalue)
            all_ratios[j, :] = np.divide(ps_eigs, iter_eigs)

        s1 = np.sum(all_ratios, axis=0)
        s2 = np.sum(np.square(all_ratios), axis=0)
        S = np.diag(np.divide(s1, s2))

        iter_eigs = eigh(np.diag(init_eigs) @ S, eigvals_only=True)

        # phase 2
        M = norm.rvs(size=(N, T))
        # Ensure eigenvalues are non-negative before taking sqrt
        iter_eigs = np.maximum(iter_eigs, min_eigenvalue)
        L = np.diag(np.sqrt(iter_eigs))
        W = L @ M @ M.T @ L / T
        _, V = eigh(W)
        upd_eigs = np.diagonal(V @ np.diag(init_eigs) @ V.T)
        # Ensure updated eigenvalues are non-negative
        upd_eigs = np.maximum(upd_eigs, min_eigenvalue)

        corrected_eigs.append(upd_eigs)
        iter_eigs = upd_eigs

    return corrected_eigs
