from scipy.spatial.distance import pdist, cdist
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, norm
from scipy.linalg import eigh
import numpy as np


def res_var_metric(all_dists, emb_dists):
    m = 1 - (pearsonr(all_dists, emb_dists)[0])**2
    return m


def correct_cov_spectrum(N, T, cmat, correction_iters=10, ensemble_size=1):
    eigs = eigh(cmat, eigvals_only=True)

    init_eigs = eigs.copy()
    iter_eigs = eigs.copy()
    corrected_eigs = [init_eigs]

    for i in range(correction_iters):
        all_ratios = np.zeros((ensemble_size, N))

        # phase 1
        for j in range(ensemble_size):
            M = norm.rvs(size=(N,T))
            L = np.diag(np.sqrt(iter_eigs))
            M2 = L@M@M.T@L/T
            ps_eigs = eigh(M2)[0]
            all_ratios[j,:] = np.divide(ps_eigs, iter_eigs)

        s1 = np.sum(all_ratios, axis=0)
        s2 = np.sum(np.square(all_ratios), axis=0)
        S = np.diag(np.divide(s1, s2))

        iter_eigs = eigh(np.diag(init_eigs)@S, eigvals_only=True)

        # phase 2
        M = norm.rvs(size=(N,T))
        L = np.diag(np.sqrt(iter_eigs))
        W = L@M@M.T@L/T
        _, V = eigh(W)
        upd_eigs = np.diagonal(V@np.diag(init_eigs)@V.T)

        corrected_eigs.append(upd_eigs)
        iter_eigs = upd_eigs

    return corrected_eigs