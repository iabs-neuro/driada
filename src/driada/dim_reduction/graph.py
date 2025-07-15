# @title Graph class { form-width: "200px" }

import pynndescent
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

from scipy.optimize import curve_fit

from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_symmetric
from scipy.sparse.csgraph import shortest_path
from umap.umap_ import fuzzy_simplicial_set
from .dr_base import *

from ..network.net_base import Network


class ProximityGraph(Network):
    '''
    Graph built on data points which represents the underlying manifold
    '''

    def __init__(self, d, m_params, g_params, create_nx_graph=False):
        self.all_metric_params = m_param_filter(m_params)
        self.metric = m_params['metric_name']
        self.metric_args = {key: self.all_metric_params[key] for key in self.all_metric_params.keys()
                            if key not in ['metric_name', 'sigma']}

        all_params = g_param_filter(g_params)
        for key in all_params:
            setattr(self, key, g_params[key])

        self.data = d

        self.construct_adjacency()
        # TODO: add graph_preprocessing to changeable graph params
        super(ProximityGraph, self).__init__(adj=self.adj,
                                             preprocessing='giant_cc',
                                             create_nx_graph=create_nx_graph,
                                             directed=False,
                                             weighted=all_params['weighted'])

        node_mapping = self._init_to_final_node_mapping
        original_n = self.data.shape[1]  # Data is (features, samples)
        lost_nodes = set(range(original_n)) - set(list(node_mapping.keys()))
        if len(lost_nodes) > 0:
            print(f'{len(lost_nodes)} nodes lost after giant component creation!')

            if len(lost_nodes) >= self.max_deleted_nodes * original_n:
                raise Exception(f'more than {self.max_deleted_nodes * 100} % of nodes discarded during gc creation!')
            else:
                self.lost_nodes = lost_nodes
                connected = [i for i in range(self.n) if i not in self.lost_nodes]
                self.bin_adj = self.bin_adj[connected, :].tocsc()[:, connected].tocsr()
                self.neigh_distmat = self.neigh_distmat[connected, :].tocsc()[:, connected].tocsr()

        self._checkpoint()

        #arr = np.array(range(self.n)).reshape(-1, 1)
        #self.timediff = cdist(arr, arr, 'cityblock')
        #self.norm_timediff = self.timediff / (self.n / 3)

    def distances_to_affinities(self):
        if self.neigh_distmat is None:
            raise Exception('distances between nearest neighbors not available')

        if not self.weighted:
            raise Exception('no need to construct affinities for binary graph weights')

        if self.dist_to_aff == 'hk':
            sigma = self.all_metric_params['sigma']
            self.adj = self.neigh_distmat.copy()
            sqdist_matrix = self.neigh_distmat.multiply(self.neigh_distmat)
            mean_sqdist = sqdist_matrix.sum() / sqdist_matrix.nnz
            self.adj.data = np.exp(-sqdist_matrix.data / (1.0 * sigma * mean_sqdist))

    def construct_adjacency(self):
        construct_fn = getattr(self, 'create_' + self.g_method_name + '_graph_')
        construct_fn()

    def _checkpoint(self):
        if self.adj is not None:
            if not sp.issparse(self.adj):
                # check for sparsity violation
                raise Exception('Adjacency matrix is not sparse!')
            self.adj = check_symmetric(self.adj, raise_exception=True)
            self.bin_adj = check_symmetric(self.bin_adj, raise_exception=True)
            self.neigh_distmat = check_symmetric(self.neigh_distmat, raise_exception=True)
            print('Adjacency symmetry confirmed')

        else:
            raise Exception('Adjacency matrix is not constructed, checkpoint routines are unavailable')

    def create_umap_graph_(self):
        RAND = np.random.RandomState(42)
        adj, _, _, dists = fuzzy_simplicial_set(self.data.T, self.nn, metric=self.metric,
                                                metric_kwds=self.metric_args,
                                                random_state=RAND,
                                                return_dists=True)

        self.adj = sp.csr_matrix(adj)
        #print('WARNING: distmat is not yet implemented in this branch')

    def create_knn_graph_(self):
        if self.metric in named_distances:
            curr_metric = self.metric
        else:
            curr_metric = globals()[self.metric]

        N = self.data.shape[1]
        index = pynndescent.NNDescent(self.data.T,
                                      metric=curr_metric,
                                      metric_kwds=self.metric_args,
                                      n_neighbors=self.nn + 1,
                                      diversify_prob=1.0,
                                      pruning_degree_multiplier=1.5)

        neighs, dists = index.neighbor_graph

        neigh_cols = neighs[:, 1:].flatten()
        dist_vals = dists[:, 1:].flatten()
        neigh_rows = np.repeat(np.arange(N), self.nn)

        all_neigh_cols = np.concatenate((neigh_cols, neigh_rows))
        all_neigh_rows = np.concatenate((neigh_rows, neigh_cols))
        all_dist_vals = np.concatenate((dist_vals, dist_vals))

        self.bin_adj = sp.csr_matrix((np.array([True for _ in range(2 * N * self.nn)]),
                                      (all_neigh_rows, all_neigh_cols)))

        self.neigh_distmat = sp.csr_matrix((all_dist_vals, (all_neigh_rows, all_neigh_cols)))

        if self.weighted:
            self.distances_to_affinities()
        else:
            self.adj = self.bin_adj.copy()

    def create_auto_knn_graph_(self):
        A = kneighbors_graph(self.data.T, self.nn, mode='connectivity', include_self=False)
        A.setdiag(0)
        A = A.astype(bool, casting='unsafe', copy=True)
        A = A + A.T
        self.adj = A
        #print('WARNING: distmat is not yet implemented in this branch')

    def calculate_indim(self, mode, factor=2):

        def normalizing_const(dim):
            if dim == 0:
                return 1
            if dim == 1:
                return 2
            elif dim == 2:
                return np.pi / 2
            else:
                return (dim - 1.0) / dim * normalizing_const(dim - 2)

        def func(x, a):
            # C = normalizing_const(a)
            return a * (np.log10(np.sin(1.0 * x * np.pi / 2)))  # + np.log10(C)
            # return a*(np.log10(np.sin(1.0*x/xmax*np.pi/2))) - np.log10(valmax)

        def func2(x, a):
            return -a / 2. * (x - max(x)) ** 2

        if self.g_method not in ['w_knn', 'w_auto_knn']:
            raise Exception('Distance matrix construction missed!')

        print('Calculating graph internal dimension...')

        if mode == 'fast':
            distmat = sp.csr_matrix(self.sqdist_matrix)
            distmat.data = np.sqrt(self.sqdist_matrix.data)
            indices = list(npr.permutation(npr.choice(self.n, size=self.n // factor, replace=False)))
            dm = distmat[indices, :][:, indices]

        elif mode == 'full':
            dm = self.sqdist_matrix
            dm.data = np.sqrt(self.sqdist_matrix.data)

        print('Shortest path computation started, distance matrix size: ', dm.shape)
        spmatrix = shortest_path(dm, method='D', directed=False)
        all_dists = spmatrix.flatten()
        all_dists = all_dists[all_dists != 0]

        nbins = 500
        pre_hist = np.histogram(all_dists, bins=nbins, density=True)
        # plt.hist(all_dists, bins = 500)
        dx = pre_hist[1][1] - pre_hist[1][0]
        dmax_bin = np.argmax(pre_hist[0])
        dmax = pre_hist[1][dmax_bin] + dx / 2.0

        hist = np.histogram(all_dists / dmax, bins=nbins, density=True)
        distr_x = hist[1][:-1] + dx / 2
        distr_y = hist[0] / max(hist[0][0:nbins])
        avg = np.mean(all_dists)
        std = np.std(all_dists / dmax)

        res = []
        # print(distr_x)
        # print(distr_y)
        left_distr_x = distr_x[(distr_x > 1 - 2. * std) & (distr_x <= 1) & (distr_y > 1e-6)]
        left_distr_y = np.log10(distr_y[(distr_x[:] > 1 - 2. * std) & (distr_x[:] <= 1)])  # & (distr_y[:]>1e-6)])

        for D in [0.1 * x for x in range(10, 260)]:
            y = func(left_distr_x, D - 1)
            res.append(np.linalg.norm(y - left_distr_y) / np.sqrt(len(y)))

        plot = 0
        if plot:
            fig = plt.figure(2, figsize=(12, 10))
            plt.plot(np.linspace(0, len(res) / 10.0, num=len(res)), res)

        Dmin = 0.1 * (np.argmax(-np.array(res)) + 1)
        print('Dmin = ', Dmin)
        fit = curve_fit(func2, left_distr_x, left_distr_y)
        # print(fit)
        a = (fit[0][0])
        # print('Dfit = ', Dfit)

        plot = 0
        if plot:
            fig = plt.figure(1, figsize=(12, 10))
            ax = fig.add_subplot(111)
            plt.hist(all_dists / dmax, bins=nbins, histtype='stepfilled', density=True, log=True)

        alpha = 2.0
        R = np.sqrt(2 * a)
        print('R = ', R)
        Dpr = 1 - alpha ** 2 / (2 * np.log(np.cos(alpha * np.pi / 2.0 / R)))
        print('D_calc = ', Dpr)

        return Dmin, Dpr

    def scaling(self):
        mat = self.adj.astype(bool).A.astype(int)
        diagsums = []
        for i in range(self.n - 1):
            diagsums.append(np.trace(mat, offset=i, dtype=None, out=None) / (self.n - i))

        return diagsums
