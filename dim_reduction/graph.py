# @title Graph class { form-width: "200px" }

import networkx as nx
import pynndescent
import scipy.sparse as sp
import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy.spatial.distance import cdist

from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_symmetric
import numpy.random as rand
from scipy.sparse.csgraph import shortest_path
from umap.umap_ import fuzzy_simplicial_set
from .dr_base import *


def take_giant_component_from_matrix(mat, weighted, gtype, error_if_isolates_exist, max_del=0.05):
    # this function preserves graph type: nx.Graph --> nx.Graph; nx.DiGraph --> nx.DiGraph
    # IMPORTANT: for an undirected graph, its largest connected component is returned.
    # For a directed graph, its largest strongly connected component is returned.

    # remove selfloops:
    a = sp.csr_matrix(mat)

    if weighted:
        a.setdiag(0)
    else:
        a.setdiag(False)

    a.eliminate_zeros()

    n_prev = a.shape[0]
    N_init = n_prev
    n_new = 0

    while n_new != n_prev:
        # remove nodes with both out- and in-degrees equal to zero:
        if weighted:
            indegrees = np.array(a.astype(bool).astype(int).sum(axis=1))[0]  # .flatten().ravel()
            outdegrees = np.array(a.astype(bool).astype(int).sum(axis=0))[0]  # .flatten().ravel()
        else:
            indegrees = np.array(a.astype(int).sum(axis=1))[0]  # .flatten().ravel()
            outdegrees = np.array(a.astype(int).sum(axis=0))[0]  # .flatten().ravel()

        indices = np.where(indegrees + outdegrees > 0)[0]

        cleared_matrix = a[indices, :].tocsc()[:, indices].tocsr()

        n_prev = n_new
        n_new = cleared_matrix.shape[0]
        a = cleared_matrix

    if error_if_isolates_exist and n_new != N_init:
        raise Exception('Error: isolated nodes found in graph!')

    if N_init != n_new:
        print('Isolates removed, %s nodes discarded' % (N_init - n_new))

    if n_new <= (1 - max_del) * N_init:
        raise Exception('more than {}% of nodes discarded as isolates!'.format(max_del * 100))

    G = nx.from_scipy_sparse_matrix(cleared_matrix, create_using=gtype)

    if nx.is_directed(G):
        strongly_connected_components = sorted(nx.strongly_connected_components(G),
                                               key=len, reverse=True)

        gcc = strongly_connected_components[0]

    else:
        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        gcc = connected_components[0]

    gc = nx.subgraph(G, gcc)  # cleared version of a graph

    gc_adj = nx.adjacency_matrix(gc)
    is_weighted = not ((gc_adj != gc_adj.astype(bool).astype(int)).nnz == 0)

    if is_weighted != weighted:
        raise Exception('Error in network construction, check weights')

    gc_nodes = gc.number_of_nodes()
    lost_nodes = list(set(list(G.nodes)).difference(set(list(gc.nodes))))

    return gc_adj, lost_nodes


class ProximityGraph():
    '''
    Graph built on data points which represents the underlying manifold
    '''

    def __init__(self, d, m_params, g_params, distmat=None):

        self.all_metric_params = m_param_filter(m_params)
        self.metric = m_params['metric_name']
        self.metric_args = {key: self.all_metric_params[key] for key in self.all_metric_params.keys() \
                            if key not in ['metric_name', 'sigma']}

        self.all_params = g_param_filter(g_params)
        for key in g_params:
            setattr(self, key, g_params[key])

        self.data = d

        self.adj = None
        self.bin_adj = None
        self.neigh_distmat = None
        self.distmat = distmat
        self.indim = -1
        self.K = self.data.shape[1]
        self.lost_nodes = None

        arr = np.array(range(self.K)).reshape(-1, 1)
        self.timediff = cdist(arr, arr, 'cityblock')
        self.norm_timediff = self.timediff / (self.K / 3)

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

    def build(self):
        fn = getattr(self, 'create_' + self.g_method_name + '_graph_')
        fn()

        if self.distmat is not None:
            check_symmetric(self.distmat, raise_exception=True)

        if self.adj is not None:
            self.take_graph_giant_component()
            self.adj = check_symmetric(self.adj, raise_exception=True)
            self.bin_adj = check_symmetric(self.bin_adj, raise_exception=True)
            self.neigh_distmat = check_symmetric(self.neigh_distmat, raise_exception=True)
            print('Adjacency symmetry confirmed')

        else:
            raise Exception('Adjacency matrix is not constructed, postprocessing routines unavailable')

    def take_graph_giant_component(self):

        a = self.adj
        if not sp.issparse(a):
            # check for sparsity violation
            raise Exception('Input is not sparse!')

        weighted = self.weighted

        '''
        TODO:
        fix the following part
        '''
        if self.g_method_name in ['knn', 'auto_knn', 'eknn', 'umap', 'tsne']:
            error_if_isolates_exist = 0
        elif self.g_method_name == 'eps':
            error_if_isolates_exist = 0

        gtype = nx.Graph()
        gc_adj, lost_nodes = take_giant_component_from_matrix(self.adj, weighted, gtype,
                                                              error_if_isolates_exist,
                                                              max_del=self.max_deleted_nodes)

        if len(lost_nodes) > 0:
            print('WARNING: %d nodes lost after giant component creation!' % len(lost_nodes))

            if len(lost_nodes) >= self.max_deleted_nodes * self.K:
                raise Exception(
                    'more than {} % of nodes discarded during gc creation!'.format(self.max_deleted_nodes * 100))

            else:
                if self.K != gc_adj.shape[0]:
                    self.adj = gc_adj
                    self.K = gc_adj.shape[0]
                    self.lost_nodes = lost_nodes

                    connected = [i for i in range(self.K) if i not in self.lost_nodes]
                    self.bin_adj = self.bin_adj[connected, :].tocsc()[:, connected].tocsr()
                    self.neigh_distmat = self.neigh_distmat[connected, :].tocsc()[:, connected].tocsr()
                    if self.distmat is not None:
                        self.distmat = self.distmat[connected, :].tocsc()[:, connected].tocsr()

    def is_connected(self):
        check_symmetric(self.adj)
        A = self.adj
        K = A.shape[0]  # number of nodes, each representing a data point
        A = A.astype(bool).astype(float)

        diags = A.sum(axis=1).flatten()
        nz = np.count_nonzero(diags)

        if nz != K:
            print('Graph has', str(K - nz), 'separate nodes!')
            raise Exception('Graph is not connected!')

        with scipy.errstate(divide='ignore'):
            invdiags = 1.0 / (diags)

        invD = scipy.sparse.spdiags(invdiags, [0], K, K, format='csr')
        X = A.dot(invD)

        eigvals, eigvecs = eigs(X, k=10, which='LR', maxiter=K * 100)
        eigvals = np.asarray([np.round(np.real(x), 6) for x in eigvals])
        n_comp = np.count_nonzero(eigvals == 1.0)

        if n_comp > 1:
            print('n_comp=', n_comp)
            raise Exception('Graph is not connected!')

        elif n_comp == 1:
            return True

        else:
            print(eigvals)
            raise Exception('Spectrum calculation error while checking integrity!')

    def create_umap_graph_(self):
        RAND = np.random.RandomState(42)
        adj, _, _, dists = fuzzy_simplicial_set(self.data.A.T, self.nn, metric=self.metric,
                                                metric_kwds=self.metric_args,
                                                random_state=RAND,
                                                return_dists=True)

        self.adj = sp.csr_matrix(adj)
        print('WARNING: distmat is not yet implemented in this branch')

    def create_knn_graph_(self):
        if self.metric in named_distances:
            curr_metric = self.metric
        else:
            curr_metric = globals()[self.metric]

        index = pynndescent.NNDescent(self.data.A.T,
                                      metric=curr_metric,
                                      metric_kwds=self.metric_args,
                                      n_neighbors=self.nn + 1,
                                      diversify_prob=1.0,
                                      pruning_degree_multiplier=1.5,
                                      max_candidates=40)

        neighs, dists = index.neighbor_graph

        neigh_cols = neighs[:, 1:].flatten()
        # print('set', set(np.arange(all_neighs_cols.shape[0])) - set(all_neighs_cols))
        dist_vals = dists[:, 1:].flatten()
        neigh_rows = np.repeat(np.arange(self.K), self.nn)

        all_neigh_cols = np.concatenate((neigh_cols, neigh_rows))
        all_neigh_rows = np.concatenate((neigh_rows, neigh_cols))
        all_dist_vals = np.concatenate((dist_vals, dist_vals))

        self.bin_adj = sp.csr_matrix((np.array([True for _ in range(2 * self.K * self.nn)]),
                                      (all_neigh_rows, all_neigh_cols))
                                     )

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
        print('WARNING: distmat is not yet implemented in this branch')

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

        if self.all_params['g_method'] not in ['w_knn', 'w_auto_knn']:
            raise Exception('Distance matrix construction missed!')
        print('Calculating graph internal dimension...')

        if mode == 'fast':
            distmat = sp.csr_matrix(self.sqdist_matrix)
            distmat.data = np.sqrt(self.sqdist_matrix.data)
            indices = list(rand.permutation(rand.choice(self.K, size=self.K // factor, replace=False)))
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
        for i in range(self.K - 1):
            diagsums.append(np.trace(mat, offset=i, dtype=None, out=None) / (self.K - i))

        return diagsums
