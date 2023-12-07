# @title Data class { form-width: "200px" }

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.stats as st

from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

from .dr_base import *
from .graph import ProximityGraph
from .embedding import Embedding

def check_data_for_errors(d):
    sums = np.sum(np.abs(d), axis=0).A
    if len(sums.nonzero()[1]) != d.shape[1]:
        bad_points = np.where(sums == 0)[1]
        print('zero points:', bad_points)
        print(d.A[:, bad_points[0]])
        raise Exception('Data contains zero points!')


def check_data_type(d):
    possible_types = (sp.csr.csr_matrix, np.ndarray)
    if not sp.issparse(d):
        data = sp.csr_matrix(d)
    else:
        data = d

    if isinstance(data, possible_types):
        check_data_for_errors(data)
        return data
    else:
        raise Exception("Wrong data type!")


class Data(object):
    '''
    Main class for data storage & processing
    '''

    def __init__(self, data, labels=None, distmat=None, standartize=True, data_name=None):

        self.data = check_data_type(data)
        self.data = self.data.todense().astype(float)

        self.std = standartize
        self.data_name = data_name

        self.dim = self.data.shape[0]
        self.npoints = self.data.shape[1]

        self.compression = 0

        if labels is None:
            self.labels = np.array([0] * self.npoints)
        else:
            try:
                labels = labels.A
            except AttributeError:
                self.labels = labels

        self.distmat = distmat

    def median_filter(self, window, plot=0):
        from scipy.signal import medfilt
        d = self.data.A

        new_d = medfilt(d, window)

        if plot:
            plt.figure(figsize=(12, 10))
            plt.plot(d[8])
            plt.plot(new_d[8], c='r')

        self.data = sp.csr_matrix(new_d)

    def corrmat(self):
        ncells = self.dim
        neuro = self.data.A
        corrmat = np.zeros((ncells, ncells))
        for i in range(ncells):
            for j in range(ncells):
                corrmat[i, j] = st.pearsonr(neuro[i, :], neuro[j, :])[0]

        return corrmat

    def get_embedding(self, m_params, g_params, e_params):
        method = e_params['e_method']
        method_name = e_params['e_method_name']

        if method_name not in EMBEDDING_CONSTRUCTION_METHODS:
            raise Exception('Unknown embedding construction method!')

        graph = None
        if method.requires_graph:
            graph = self.get_graph(m_params, g_params)

        if method.requires_distmat and self.distmat is None:
            raise Exception('No distmat provided for {} method'.format(method_name))

        emb = Embedding(self.data, self.distmat, self.labels, e_params, g=graph)
        emb.build()

        return emb

    def get_graph(self, m_params, g_params):
        if g_params['g_method_name'] not in GRAPH_CONSTRUCTION_METHODS:
            raise Exception('Unknown graph construction method!')

        graph = ProximityGraph(self.data, m_params, g_params, distmat=self.distmat)
        graph.build()
        # print('Graph succesfully constructed')
        return graph

    def draw_vector(self, num):
        data = self.data.A[:, num]
        plt.matshow(data.reshape(1, self.dim))
        plt.matshow(self.data.A[:, :1000])

    def draw_row(self, num):
        data = self.data.A[num, :]
        plt.figure(figsize=(12, 10))
        plt.plot(data)
        # plt.matshow(data.reshape(1,self.npoints))
        # plt.matshow(self.data.A[:, :1000])
