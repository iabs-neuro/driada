# @title Data class { form-width: "200px" }

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from .dr_base import *
from ..utils.data import correlation_matrix, to_numpy_array, rescale
from .embedding import Embedding
from .graph import ProximityGraph

# TODO: refactor this
def check_data_for_errors(d):
    sums = np.sum(np.abs(d), axis=0)
    if len(sums.nonzero()[1]) != d.shape[1]:
        bad_points = np.where(sums == 0)[1]
        print('zero points:', bad_points)
        print(d.A[:, bad_points[0]])
        raise Exception('Data contains zero points!')


class MVData(object):
    '''
    Main class for multivariate data storage & processing
    '''

    def __init__(self,
                 data,
                 labels=None,
                 distmat=None,
                 rescale_rows=False,
                 data_name=None,
                 downsampling=None):

        if downsampling is None:
            self.ds = 1
        else:
            self.ds = int(downsampling)

        self.data = to_numpy_array(data)[:, ::self.ds]

        # TODO: add support for various preprocessing methods (wvt, med_filt, etc.)
        self.rescale_rows = rescale_rows
        if self.rescale_rows:
            for i, row in enumerate(self.data):
                self.data[i] = rescale(row)

        self.data_name = data_name
        self.n_dim = self.data.shape[0]
        self.n_points = self.data.shape[1]

        if labels is None:
            self.labels = np.zeros(self.n_points)
        else:
            self.labels = to_numpy_array(labels)

        self.distmat = distmat

    def median_filter(self, window):
        from scipy.signal import medfilt
        d = self.data.A

        new_d = medfilt(d, window)

        self.data = sp.csr_matrix(new_d)

    def corr_mat(self):
        cm = correlation_matrix(self.data)
        return cm

    def get_distmat(self, method='euclidean'):
        raise NotImplementedError('To be implemented soon')

    def get_embedding(self, e_params, g_params=None, m_params=None, kwargs=None):
        method = e_params['e_method']
        method_name = e_params['e_method_name']

        if method_name not in EMBEDDING_CONSTRUCTION_METHODS:
            raise Exception('Unknown embedding construction method!')

        graph = None
        if method.requires_graph:
            if g_params is None:
                raise ValueError(f'Method {method_name} requires proximity graph, but '
                                 f'graph params were not provided')
            if g_params['weighted'] and m_params is None:
                raise ValueError(f'Method {method_name} requires weights for proximity graph, but '
                                 f'metric params were not provided')

            graph = self.get_proximity_graph(m_params, g_params)

        if method.requires_distmat and self.distmat is None:
            raise Exception(f'No distmat provided for {method_name} method.'
                            f' Try constructing it first with get_distmat() method')

        emb = Embedding(self.data, self.distmat, self.labels, e_params, g=graph)
        emb.build(kwargs=kwargs)

        return emb

    def get_proximity_graph(self, m_params, g_params):
        if g_params['g_method_name'] not in GRAPH_CONSTRUCTION_METHODS:
            raise Exception('Unknown graph construction method!')

        graph = ProximityGraph(self.data, m_params, g_params)
        # print('Graph succesfully constructed')
        return graph

    def draw_vector(self, num):
        data = self.data[:, num]
        plt.matshow(data.reshape(1, self.n_dim))
        plt.matshow(self.data)

    def draw_row(self, num):
        data = self.data[num, :]
        plt.figure(figsize=(12, 10))
        plt.plot(data)
