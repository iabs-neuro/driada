import warnings

import numpy as np

from .ksg import *
from .gcmi import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import scipy
from scipy.stats import entropy, differential_entropy


class TimeSeries():

    @staticmethod
    def define_ts_type(ts):
        if len(ts) < 100:
            warnings.warn('Time series is too short for accurate type (discrete/continuous) determination')
        unique_vals = np.unique(ts)
        sc1 = len(unique_vals) / len(ts)
        hist = np.histogram(ts, bins=len(ts))[0]
        ent = entropy(hist)
        maxent = entropy(np.ones(len(ts)))
        sc2 = ent / maxent

        if sc1 > 0.70 and sc2 > 0.70:
            return False  # both scores are high - the variable is most probably continuous
        elif sc1 < 0.25 and sc2 < 0.25:
            return True  # both scores are low - the variable is most probably discrete
        else:
            raise ValueError(f'Unable to determine time series type automatically: score 1 = {sc1}, score 2 = {sc2}')

    def _check_input(self):
        pass

    def __init__(self, data, discrete=None):
        self.data = data
        if discrete is None:
            #warnings.warn('Time series type not specified and will be inferred automatically')
            self.discrete = TimeSeries.define_ts_type(data)
        else:
            self.discrete = discrete

        scaler = MinMaxScaler()
        self.scdata = scaler.fit_transform(self.data.reshape(-1, 1)).reshape(1, -1)[0]
        self.copula_normal_data = None
        if not self.discrete:
            self.copula_normal_data = copnorm(self.data).ravel()

        self.entropy = dict()
        self.kdtree = None
        self.kdtree_query = None

    def get_kdtree(self):
        if self.kdtree is None:
            tree = self._compute_kdtree()
            self.kdtree = tree

        return self.kdtree

    def _compute_kdtree(self):
        d = self.scdata.reshape(self.scdata.shape[0], -1)
        return build_tree(d)

    def get_kdtree_query(self, k=DEFAULT_NN):
        if self.kdtree_query is None:
            q = self._compute_kdtree_query(k=k)
            self.kdtree_query = q

        return self.kdtree_query

    def _compute_kdtree_query(self, k=DEFAULT_NN):
        tree = self.get_kdtree()
        return tree.query(self.scdata, k=k + 1)

    def get_entropy(self, ds=1):
        if ds not in self.entropy.keys():
            self._compute_entropy(ds=ds)
        return self.entropy[ds]

    def _compute_entropy(self, ds=1):
        if self.discrete:
            counts = []
            for val in np.unique(self.data[::ds]):
                counts.append(len(np.where(self.data[::ds] == val)[0]))

            self.entropy[ds] = scipy.stats.entropy(counts, base=np.e)

        else:
            self.entropy[ds] = get_tdmi(self.scdata[::ds], min_shift=1, max_shift=2)[0]
            #raise AttributeError('Entropy for continuous variables is not yet implemented'

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


def get_tdmi(data, min_shift = 1, max_shift = 100, nn = DEFAULT_NN):
    ts = TimeSeries(data, discrete = False)
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

    return mi