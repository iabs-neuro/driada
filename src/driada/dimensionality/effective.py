from .utils import *
from scipy.stats import entropy
import warnings
from ..utils.data import correlation_matrix

DATA_SHAPE_THR = 0.01  # if n/t in multivariate time series data is more than DATA_SHAPE_THR,
                      # cov/corr spectrum may be significantly distorted, correction is recommended


def _eff_dim(corr_eigs, q=2):
    if q < 0:
        raise ValueError('Renyi entropy is undefined for q<0')

    norm_corr_eigs = corr_eigs / np.sum(corr_eigs)
    if q == 1:
        # standard entropy
        return 2**entropy(norm_corr_eigs, base=2)

    elif q == 2:
        # quadratic entropy
        return (sum(corr_eigs)**2)/sum([e**2 for e in corr_eigs])

    elif q == np.inf:
        # min-entropy
        return np.sum(corr_eigs)/np.max(corr_eigs)

    else:
        return 1.0/(1.0 - q)*np.log(np.sum([p**q for p in norm_corr_eigs]))


def eff_dim(data, enable_correction, q=2, **correction_kwargs):
    n, t = data.shape
    if 1.0*n/t > DATA_SHAPE_THR and not enable_correction:
        warnings.warn(f'fN/T is {1.0*n/t}, which is bigger than {DATA_SHAPE_THR}. Spectrum correction is recommended')

    cmat = correlation_matrix(data)
    if enable_correction:
        corrected_eigs = correct_cov_spectrum(n, t, cmat, **correction_kwargs)
        final_eigs = corrected_eigs[-1]
    else:
        final_eigs = eigh(cmat, eigvals_only=True)

    return _eff_dim(final_eigs, q=q)



