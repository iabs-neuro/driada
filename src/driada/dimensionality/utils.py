from scipy.spatial.distance import pdist, cdist
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, norm
from scipy.linalg import eigh
import numpy as np
import warnings


def res_var_metric(all_dists, emb_dists):
    m = 1 - (pearsonr(all_dists, emb_dists)[0])**2
    return m


def correct_cov_spectrum(N, T, cmat, correction_iters=10, ensemble_size=1, min_eigenvalue=1e-10):
    """
    Correct the spectrum of a covariance/correlation matrix.
    
    Parameters
    ----------
    N : int
        Number of variables (neurons).
    T : int
        Number of time points.
    cmat : ndarray
        Covariance or correlation matrix.
    correction_iters : int, optional
        Number of correction iterations. Default is 10.
    ensemble_size : int, optional
        Size of the ensemble for phase 1. Default is 1.
    min_eigenvalue : float, optional
        Minimum eigenvalue threshold to avoid numerical issues. Default is 1e-10.
        
    Returns
    -------
    corrected_eigs : list
        List of eigenvalue arrays for each iteration.
    """
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
            M = norm.rvs(size=(N,T))
            # Ensure eigenvalues are non-negative before taking sqrt
            iter_eigs = np.maximum(iter_eigs, min_eigenvalue)
            L = np.diag(np.sqrt(iter_eigs))
            M2 = L@M@M.T@L/T
            ps_eigs = eigh(M2)[0]
            # Clip ps_eigs to avoid division issues
            ps_eigs = np.maximum(ps_eigs, min_eigenvalue)
            all_ratios[j,:] = np.divide(ps_eigs, iter_eigs)

        s1 = np.sum(all_ratios, axis=0)
        s2 = np.sum(np.square(all_ratios), axis=0)
        S = np.diag(np.divide(s1, s2))

        iter_eigs = eigh(np.diag(init_eigs)@S, eigvals_only=True)

        # phase 2
        M = norm.rvs(size=(N,T))
        # Ensure eigenvalues are non-negative before taking sqrt
        iter_eigs = np.maximum(iter_eigs, min_eigenvalue)
        L = np.diag(np.sqrt(iter_eigs))
        W = L@M@M.T@L/T
        _, V = eigh(W)
        upd_eigs = np.diagonal(V@np.diag(init_eigs)@V.T)
        # Ensure updated eigenvalues are non-negative
        upd_eigs = np.maximum(upd_eigs, min_eigenvalue)

        corrected_eigs.append(upd_eigs)
        iter_eigs = upd_eigs

    return corrected_eigs