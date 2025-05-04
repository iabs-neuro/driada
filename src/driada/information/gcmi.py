import numpy as np
import numba as nb
from numba import njit
import warnings
from scipy.special import ndtri, psi, digamma

from .info_utils import py_fast_digamma_arr

#TODO: credits to original GCMI: https://github.com/robince/gcmi

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


@njit
def demean(x):
    # Get the number of rows
    num_rows = x.shape[0]

    # Create an output array with the same shape as input
    demeaned_x= np.empty_like(x)

    # Demean each row
    for i in range(num_rows):
        row_mean = np.mean(x[i])
        demeaned_x[i] = x[i] - row_mean

    return demeaned_x


#@njit()
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
        psiterms = py_fast_digamma_arr((Ntrl - np.arange(1, Nvarx + 1).astype(float)) / 2.0) / 2.0
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
    bias correction should be applied to the estimated MI.
    demeaned : false / true option (default false) which specifies whether th
    input data already has zero mean (true if it has been copula-normalized)
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 3 or y.ndim > 3:
        raise ValueError("x and y must be at most 3d")
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
        psiterms = py_fast_digamma_arr((Ntrl - np.arange(1, Nvarxy + 1)) / 2.0) / 2.0
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

        psiterms = py_fast_digamma_arr((Ntrl - vars) / 2.0) / 2.0
        dterm = (ln2 - np.log(float(Ntrl - 1))) / 2.0
        Hunc = Hunc - Nvarx * dterm - psiterms.sum()

        dterm = (ln2 - np.log((Ntrl_y - 1))) / 2.0
        psiterms = np.zeros(Ym)
        for vi in vars:
            idx = Ntrl_y - vi
            psiterms = psiterms + py_fast_digamma_arr(idx / 2.0)
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

# TODO: integrate into numba everything below this line
def cmi_ggg(x, y, z, biascorrect=True, demeaned=False):
    """Conditional Mutual information (CMI) between two Gaussian variables
    conditioned on a third

    I = cmi_ggg(x,y,z) returns the CMI between two (possibly multidimensional)
    Gassian variables, x and y, conditioned on a third, z, with bias correction.
    If x / y / z are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis)

    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether the
    input data already has zero mean (true if it has been copula-normalized)

    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)
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
    xyz = np.vstack((x,y,z))
    if not demeaned:
        xyz = xyz - xyz.mean(axis=1)[:,np.newaxis]
    Cxyz = np.dot(xyz,xyz.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cz = Cxyz[Nvarxy:,Nvarxy:]
    Cyz = Cxyz[Nvarx:,Nvarx:]
    Cxz = np.zeros((Nvarxz,Nvarxz))
    Cxz[:Nvarx,:Nvarx] = Cxyz[:Nvarx,:Nvarx]
    Cxz[:Nvarx,Nvarx:] = Cxyz[:Nvarx,Nvarxy:]
    Cxz[Nvarx:,:Nvarx] = Cxyz[Nvarxy:,:Nvarx]
    Cxz[Nvarx:,Nvarx:] = Cxyz[Nvarxy:,Nvarxy:]

    chCz = np.linalg.cholesky(Cz)
    chCxz = np.linalg.cholesky(Cxz)
    chCyz = np.linalg.cholesky(Cyz)
    chCxyz = np.linalg.cholesky(Cxyz)

    # entropies in nats
    # normalizations cancel for cmi
    HZ = np.sum(np.log(np.diagonal(chCz))) # + 0.5*Nvarz*(np.log(2*np.pi)+1.0)
    HXZ = np.sum(np.log(np.diagonal(chCxz))) # + 0.5*Nvarxz*(np.log(2*np.pi)+1.0)
    HYZ = np.sum(np.log(np.diagonal(chCyz))) # + 0.5*Nvaryz*(np.log(2*np.pi)+1.0)
    HXYZ = np.sum(np.log(np.diagonal(chCxyz))) # + 0.5*Nvarxyz*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = psi((Ntrl - np.arange(1,Nvarxyz+1)).astype(np.float)/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HZ = HZ - Nvarz*dterm - psiterms[:Nvarz].sum()
        HXZ = HXZ - Nvarxz*dterm - psiterms[:Nvarxz].sum()
        HYZ = HYZ - Nvaryz*dterm - psiterms[:Nvaryz].sum()
        HXYZ = HXYZ - Nvarxyz*dterm - psiterms[:Nvarxyz].sum()

    # MI in bits
    I = (HXZ + HYZ - HXYZ - HZ) / ln2
    return I


def gccmi_ccd(x,y,z,Zm):
    """Gaussian-Copula CMI between 2 continuous variables conditioned on a discrete variable.

    I = gccmi_ccd(x,y,z,Zm) returns the CMI between two (possibly multidimensional)
    continuous variables, x and y, conditioned on a third discrete variable z, estimated
    via a Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis)
    z should contain integer values in the range [0 Zm-1] (inclusive).

    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    if z.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(z.dtype, np.integer):
        raise ValueError("z should be an integer array")
    if not isinstance(Zm, int):
        raise ValueError("Zm should be an integer")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]

    if y.shape[1] != Ntrl or z.size != Ntrl:
        raise ValueError("number of trials do not match")

    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break

    # check values of discrete variable
    if z.min()!=0 or z.max()!=(Zm-1):
        raise ValueError("values of discrete variable z are out of bounds")

    # calculate gcmi for each z value
    Icond = np.zeros(Zm)
    Pz = np.zeros(Zm)
    cx = []
    cy = []
    for zi in range(Zm):
        idx = z==zi
        thsx = copnorm(x[:,idx])
        thsy = copnorm(y[:,idx])
        Pz[zi] = idx.sum()
        cx.append(thsx)
        cy.append(thsy)
        Icond[zi] = mi_gg(thsx,thsy,True,True)

    Pz = Pz / float(Ntrl)

    # conditional mutual information
    CMI = np.sum(Pz*Icond)
    #I = mi_gg(np.hstack(cx),np.hstack(cy),True,False)
    return CMI
