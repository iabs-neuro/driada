import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy, differential_entropy
from ..information.info_base import get_tdmi
from ..information.ksg import build_tree
from ..information.gcmi import copnorm

DEFAULT_NN = 5

# TODO: add @property decorators to properly set getter-setter functionality

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

            self.entropy[ds] = entropy(counts, base=np.e)

        else:
            # TODO: refactor entropy calculation
            self.entropy[ds] = get_tdmi(self.scdata[::ds], min_shift=1, max_shift=2)[0]
            #raise AttributeError('Entropy for continuous variables is not yet implemented'
