import numpy as np
import numba as nb
from numba import njit
from scipy.special import ndtri, psi, digamma

from .info_utils import py_fast_digamma

#TODO: credits to original GCMI

def ctransform(x):
    """Copula transformation (empirical CDF)
    cx = ctransform(x) returns the empirical CDF value along the first
    axis of x. Data is ranked and scaled within [0 1] (open interval).
    """

    xi = np.argsort(np.atleast_2d(x))
    xr = np.argsort(xi)
    cx = (xr + 1).astype(float) / (xr.shape[-1] + 1)
    return cx


def copnorm(x):
    """Copula normalization

    cx = copnorm(x) returns standard normal samples with the same empirical
    CDF value as the input. Operates along the last axis.
    """
    # cx = sp.stats.norm.ppf(ctransform(x))
    cx = ndtri(ctransform(x))
    return cx


@njit()
def demean(x):
    return x - np.full(len(x), x.mean())


@njit()
def ent_g(x, biascorrect=True):
    """Entropy of a Gaussian variable in bits
    H = ent_g(x) returns the entropy of a (possibly
    multidimensional) Gaussian variable x with bias correction.
    Columns of x correspond to samples, rows to dimensions/variables.
    (Samples last axis)
    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    # demean data
    x = demean(x)
    # covariance
    C = np.dot(x, x.T) / float(Ntrl - 1)
    chC = np.linalg.cholesky(C)

    # entropy in nats
    HX = np.sum(np.log(np.diagonal(chC))) + 0.5 * Nvarx * (np.log(2 * np.pi) + 1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = py_fast_digamma((Ntrl - np.arange(1, Nvarx + 1).astype(float)) / 2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HX = HX - Nvarx * dterm - psiterms.sum()

    # convert to bits
    return HX / ln2


@njit()
def mi_gg(x, y, biascorrect=True, demeaned=False):
    """Mutual information (MI) between two Gaussian variables in bits

    I = mi_gg(x,y) returns the MI between two (possibly multidimensional)
    Gassian variables, x and y, with bias correction.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis)

    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether th
    input data already has zero mean (true if it has been copula-normalized)
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
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

    chCxy = np.linalg.cholesky(Cxy)
    chCx = np.linalg.cholesky(Cx)
    chCy = np.linalg.cholesky(Cy)

    # entropies in nats
    # normalizations cancel for mutual information
    HX = np.sum(np.log(np.diag(chCx)))  # + 0.5*Nvarx*(np.log(2*np.pi)+1.0)
    HY = np.sum(np.log(np.diag(chCy)))  # + 0.5*Nvary*(np.log(2*np.pi)+1.0)
    HXY = np.sum(np.log(np.diag(chCxy)))  # + 0.5*Nvarxy*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = py_fast_digamma((Ntrl - np.arange(1, Nvarxy + 1)) / 2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HX = HX - Nvarx * dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary * dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy * dterm - psiterms[:Nvarxy].sum()

    # MI in bits
    I = (HX + HY - HXY) / ln2
    return I


@njit()
def mi_model_gd(x, y, Ym, biascorrect=True, demeaned=False):
    """Mutual information (MI) between a Gaussian and a discrete variable in bits
    based on ANOVA style model comparison.
    I = mi_model_gd(x,y,Ym) returns the MI between the (possibly multidimensional)
    Gaussian variable x and the discrete variable y.
    For 1D x this is a lower bound to the mutual information.
    Columns of x correspond to samples, rows to dimensions/variables.
    (Samples last axis)
    y should contain integer values in the range [0 Ym-1] (inclusive).
    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the estimated MI.
    demeaned : false / true option (default false) which specifies whether the
    input data already has zero mean (true if it has been copula-normalized)
    See also: mi_mixture_gd
    """

    x = np.atleast_2d(x)
    # y = np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    '''
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y should be an integer array")
    '''
    if int(Ym) != Ym:
        raise ValueError("Ym should be an integer")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    if y.size != Ntrl:
        raise ValueError("number of trials do not match")
    '''
    if not demeaned:
        x = x - x.mean(axis=1)[:,np.newaxis]
    '''
    # class-conditional entropies
    Ntrl_y = np.zeros(Ym)
    Hcond = np.zeros(Ym)
    c = 0.5 * (np.log(2.0 * np.pi) + 1)

    for yi in range(Ym):
        idx = y == yi
        xm = x[:, idx]
        Ntrl_y[yi] = xm.shape[1]
        xm = demean(xm)
        Cm = np.dot(xm, xm.T) / float(Ntrl_y[yi] - 1)
        chCm = np.linalg.cholesky(Cm)
        Hcond[yi] = np.sum(np.log(np.diag(chCm)))  # + c*Nvarx

    # class weights
    w = Ntrl_y / float(Ntrl)

    # unconditional entropy from unconditional Gaussian fit
    Cx = np.dot(x, x.T) / float(Ntrl - 1)
    chC = np.linalg.cholesky(Cx)
    Hunc = np.sum(np.log(np.diag(chC)))  # + c*Nvarx

    ln2 = np.log(2)
    if biascorrect:
        vars = np.arange(1, Nvarx + 1)

        psiterms = py_fast_digamma((Ntrl - vars) / 2.0) / 2.0
        dterm = (ln2 - np.log(float(Ntrl - 1))) / 2.0
        Hunc = Hunc - Nvarx * dterm - psiterms.sum()

        dterm = (ln2 - np.log((Ntrl_y - 1))) / 2.0
        psiterms = np.zeros(Ym)
        for vi in vars:
            idx = Ntrl_y - vi
            psiterms = psiterms + py_fast_digamma(idx / 2.0)
        Hcond = Hcond - Nvarx * dterm - (psiterms / 2.0)

    # MI in bits
    I = (Hunc - np.sum(w * Hcond)) / ln2
    return I


def gcmi_cc(x, y):
    """Gaussian-Copula Mutual Information between two continuous variables.
    I = gcmi_cc(x,y) returns the MI between two (possibly multidimensional)
    continuous variables, x and y, estimated via a Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis)
    This provides a lower bound to the true MI value.
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    '''
    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break
    '''

    # copula normalization
    cx = copnorm(x)
    cy = copnorm(y)
    # parametric Gaussian MI
    I = mi_gg(cx, cy, True, True)
    return I
