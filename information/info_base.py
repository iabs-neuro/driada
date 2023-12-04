import warnings

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .ksg import *
from .gcmi import *
from ..signal.sig_base import TimeSeries


def get_1d_mi(ts1, ts2, shift=0, ds=1, k=DEFAULT_NN, estimator='gcmi'):
    """Computes mutual information between two 1d variables efficiently

    Parameters
    ----------
    ts1: TimeSeries instance or numpy array
    ts2: TimeSeries instance or numpy array
    shift: int
        ts2 will be roll-moved by the number 'shift' after downsampling by 'ds' factor
    ds: int
        downsampling constant (take every 'ds'-th point)
    k: int
        number of neighbors for ksg estimator
    estimator: str
        Estimation method. Should be 'ksg' (accurate but slow) and 'gcmi' (ultra-fast, but estimates the lower bound on MI).
        In most cases 'gcmi' should be preferred.

    Returns
    -------
    mi: mutual information (or its lower bound in case of 'gcmi' estimator) between ts1 and (possibly) shifted ts2

    """
    if not isinstance(ts1, TimeSeries):
        ts1 = TimeSeries(ts1)
    if not isinstance(ts2, TimeSeries):
        ts2 = TimeSeries(ts2)

    x = ts1.scdata[::ds].reshape(-1, 1)
    y = ts2.scdata[::ds]
    if shift != 0:
        y = np.roll(y, shift)

    if estimator == 'ksg':
        if not ts1.discrete and not ts2.discrete:
            mi = nonparam_mi_cc_mod(ts1.scdata, y, k=k,
                                    precomputed_tree_x=ts1.get_kdtree(),
                                    precomputed_tree_y=ts2.get_kdtree())

        elif ts1.discrete and ts2.discrete:
            mi = mutual_info_classif(x, y, discrete_features=True, n_neighbors=k)[0]

        # TODO: refactor using ksg functions
        elif ts1.discrete and not ts2.discrete:
            mi = mutual_info_regression(x, y, discrete_features=False, n_neighbors=k)[0]

        elif not ts1.discrete and ts2.discrete:
            mi = mutual_info_classif(x, y, discrete_features=True, n_neighbors=k)[0]

        return mi

    elif estimator == 'gcmi':
        if not ts1.discrete and not ts2.discrete:
            ny1 = ts1.copula_normal_data[::ds]
            ny2 = np.roll(ts2.copula_normal_data[::ds], shift)
            mi = mi_gg(ny1, ny2, True, True)

        elif ts1.discrete and ts2.discrete:
            mi = mutual_info_classif(x, y, discrete_features=True, n_neighbors=k)[0]

        elif ts1.discrete and not ts2.discrete:
            ny1 = ts1.scdata.astype(int)[::ds]
            ny2 = np.roll(ts2.copula_normal_data[::ds], shift)
            mi = mi_model_gd(ny2, ny1, np.max(ny1), biascorrect=True, demeaned=True)

        elif not ts1.discrete and ts2.discrete:
            ny1 = ts1.copula_normal_data[::ds]
            ny2 = np.roll(ts2.scdata.astype(int)[::ds], shift)
            mi = mi_model_gd(ny1, ny2, np.max(ny2), biascorrect=True, demeaned=True)

        if mi < 0:
            mi = 0

        return mi


def get_tdmi(data, min_shift=1, max_shift=100, nn=DEFAULT_NN):
    ts = TimeSeries(data, discrete=False)
    tdmi = [get_1d_mi(ts, ts, shift=shift, k=nn) for shift in range(min_shift, max_shift)]

    return tdmi


def get_multi_mi(tslist, ts2, shift=0, ds=1, k=DEFAULT_NN, estimator='gcmi'):

    if ~np.all([ts.discrete for ts in tslist]) and not ts2.discrete:
        nylist = [ts.copula_normal_data[::ds] for ts in tslist]
        ny1 = np.vstack(nylist)
        ny2 = np.roll(ts2.copula_normal_data, shift)[::ds]
        mi = mi_gg(ny1, ny2, True, True)
    else:
        raise ValueError('Multidimensional MI only implemented for continuous data!')

    if mi < 0:
        mi = 0

    return mi