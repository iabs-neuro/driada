
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

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
        print(d.todense()[:, bad_points[0]])
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
        d = self.data.todense()

        new_d = medfilt(d, window)

        self.data = sp.csr_matrix(new_d)

    def corr_mat(self):
        cm = correlation_matrix(self.data)
        return cm

    def get_distmat(self, m_params=None):
        """Compute pairwise distance matrix.
        
        Parameters
        ----------
        m_params : dict or str, optional
            If dict: metric parameters with 'metric_name' key and optional metric-specific params
            If str: metric name directly
            If None: defaults to 'euclidean'
            
        Returns
        -------
        np.ndarray
            Distance matrix of shape (n_samples, n_samples)
        """
        from scipy.spatial.distance import pdist, squareform
        
        # Handle different input types
        if m_params is None:
            metric = 'euclidean'
            metric_kwargs = {}
        elif isinstance(m_params, str):
            metric = m_params
            metric_kwargs = {}
        elif isinstance(m_params, dict):
            metric = m_params.get('metric_name', 'euclidean')
            # Convert l2 to euclidean for scipy
            if metric == 'l2':
                metric = 'euclidean'
            # Extract additional parameters for the metric
            metric_kwargs = {k: v for k, v in m_params.items() if k not in ['metric_name', 'sigma']}
            # For minkowski distance, 'p' parameter is needed
            if metric == 'minkowski' and 'p' in m_params:
                metric_kwargs['p'] = m_params['p']
        else:
            metric = 'euclidean'
            metric_kwargs = {}
            
        # Compute distance matrix
        if metric_kwargs:
            distances = pdist(self.data.T, metric=metric, **metric_kwargs)
        else:
            distances = pdist(self.data.T, metric=metric)
            
        self.distmat = squareform(distances)
        return self.distmat

    def get_embedding(self, e_params=None, g_params=None, m_params=None, kwargs=None, 
                      method=None, **method_kwargs):
        """Get embedding using specified method.
        
        Parameters
        ----------
        e_params : dict, optional
            Embedding parameters (legacy format)
        g_params : dict, optional
            Graph parameters (legacy format)
        m_params : dict, optional
            Metric parameters (legacy format)
        kwargs : dict, optional
            Additional kwargs for the embedding method
        method : str, optional
            Method name for simplified API (e.g., 'pca', 'umap')
        **method_kwargs
            Additional parameters when using simplified API
            
        Returns
        -------
        Embedding
            The computed embedding
            
        Examples
        --------
        # Legacy format (still supported)
        >>> emb = mvdata.get_embedding(e_params, g_params, m_params)
        
        # New simplified format
        >>> emb = mvdata.get_embedding(method='pca', dim=3)
        >>> emb = mvdata.get_embedding(method='umap', n_components=2, n_neighbors=30)
        """
        # Handle new simplified API
        if method is not None:
            # Merge with defaults
            from .dr_base import merge_params_with_defaults
            params = merge_params_with_defaults(method, method_kwargs)
            e_params = params['e_params']
            g_params = params['g_params']
            m_params = params['m_params']
        elif e_params is None:
            raise ValueError("Either 'method' or 'e_params' must be provided")
        
        # Legacy compatibility: ensure e_method is set
        if 'e_method' not in e_params or e_params['e_method'] is None:
            method_name = e_params.get('e_method_name')
            if method_name and method_name in METHODS_DICT:
                e_params['e_method'] = METHODS_DICT[method_name]
        
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
