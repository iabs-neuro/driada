import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from .dr_base import (
    METHODS_DICT,
    EMBEDDING_CONSTRUCTION_METHODS,
    GRAPH_CONSTRUCTION_METHODS,
    merge_params_with_defaults,
)
from ..utils.data import correlation_matrix, to_numpy_array, rescale
from .embedding import Embedding
from .graph import ProximityGraph


def check_data_for_errors(d, verbose=True):
    """Check data matrix for zero columns which can cause issues in DR methods.
    
    Parameters
    ----------
    d : np.ndarray or scipy.sparse matrix
        Data matrix with shape (n_features, n_samples)
    verbose : bool, default=True
        Whether to print information about zero points
        
    Raises
    ------
    ValueError
        If data contains columns with all zeros
    """
    # Handle both dense and sparse matrices
    if sp.issparse(d):
        # For sparse matrices, use efficient column sum
        sums = np.asarray(np.abs(d).sum(axis=0)).flatten()
    else:
        # For dense arrays
        sums = np.sum(np.abs(d), axis=0)
    
    # Find zero columns
    zero_cols = np.where(sums == 0)[0]
    
    if len(zero_cols) > 0:
        if verbose:
            print(f"Found {len(zero_cols)} zero columns at indices: {zero_cols[:10]}")
            if len(zero_cols) > 10:
                print(f"... and {len(zero_cols) - 10} more")
            # Show first zero column if sparse
            if sp.issparse(d):
                print(f"Example zero column (index {zero_cols[0]}): {d[:, zero_cols[0]].toarray().flatten()}")
            else:
                print(f"Example zero column (index {zero_cols[0]}): {d[:, zero_cols[0]]}")
        
        raise ValueError(
            f"Data contains {len(zero_cols)} zero columns (all values are 0). "
            f"This can cause issues in dimensionality reduction. "
            f"Consider removing these columns or adding small noise."
        )


class MVData(object):
    """
    Main class for multivariate data storage & processing
    
    Parameters
    ----------
    data : array-like
        Data matrix with shape (n_features, n_samples)
    labels : array-like, optional
        Labels for each sample
    distmat : array-like, optional
        Precomputed distance matrix
    rescale_rows : bool, default=False
        Whether to rescale each row to [0, 1]
    data_name : str, optional
        Name for the dataset
    downsampling : int, optional
        Downsampling factor
    verbose : bool, default=False
        Whether to print progress messages
    allow_zero_columns : bool, default=False
        Whether to allow columns with all zero values. If False, raises ValueError
        when zero columns are detected.
    """

    def __init__(
        self,
        data,
        labels=None,
        distmat=None,
        rescale_rows=False,
        data_name=None,
        downsampling=None,
        verbose=False,
        allow_zero_columns=False,
    ):

        if downsampling is None:
            self.ds = 1
        else:
            self.ds = int(downsampling)

        self.data = to_numpy_array(data)[:, :: self.ds]
        
        # Check for zero columns that could cause issues
        if not allow_zero_columns:
            check_data_for_errors(self.data, verbose=verbose)

        # Note: Preprocessing methods (gaussian, savgol, wavelet) are available via
        # TimeSeries/MultiTimeSeries.filter() before creating MVData objects
        self.rescale_rows = rescale_rows
        if self.rescale_rows:
            for i, row in enumerate(self.data):
                self.data[i] = rescale(row)

        self.data_name = data_name
        self.n_dim = self.data.shape[0]
        self.n_points = self.data.shape[1]
        self.verbose = verbose

        if labels is None:
            self.labels = np.zeros(self.n_points)
        else:
            self.labels = to_numpy_array(labels)

        self.distmat = distmat

    def median_filter(self, window):
        from scipy.signal import medfilt

        # Handle both sparse and dense data
        if sp.issparse(self.data):
            d = self.data.todense()
        else:
            d = self.data

        new_d = medfilt(d, window)

        # Convert back to the original format
        if sp.issparse(self.data):
            self.data = sp.csr_matrix(new_d)
        else:
            self.data = new_d

    def corr_mat(self, axis=0):
        """Compute correlation matrix.

        Parameters
        ----------
        axis : int, default 0
            Axis along which to compute correlations:
            - 0: correlations between rows (features)
            - 1: correlations between columns (samples/timepoints)

        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        if axis == 0:
            cm = correlation_matrix(self.data)
        else:  # axis == 1
            # Transpose to compute correlations between columns
            cm = correlation_matrix(self.data.T)
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
            metric = "euclidean"
            metric_kwargs = {}
        elif isinstance(m_params, str):
            metric = m_params
            metric_kwargs = {}
        elif isinstance(m_params, dict):
            metric = m_params.get("metric_name", "euclidean")
            # Convert l2 to euclidean for scipy
            if metric == "l2":
                metric = "euclidean"
            # Extract additional parameters for the metric
            metric_kwargs = {
                k: v for k, v in m_params.items() if k not in ["metric_name", "sigma"]
            }
            # For minkowski distance, 'p' parameter is needed
            if metric == "minkowski" and "p" in m_params:
                metric_kwargs["p"] = m_params["p"]
        else:
            metric = "euclidean"
            metric_kwargs = {}

        # Compute distance matrix
        if metric_kwargs:
            distances = pdist(self.data.T, metric=metric, **metric_kwargs)
        else:
            distances = pdist(self.data.T, metric=metric)

        self.distmat = squareform(distances)
        return self.distmat

    def get_embedding(
        self,
        e_params=None,
        g_params=None,
        m_params=None,
        kwargs=None,
        method=None,
        **method_kwargs,
    ):
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
            params = merge_params_with_defaults(method, method_kwargs)
            e_params = params["e_params"]
            g_params = params["g_params"]
            m_params = params["m_params"]
        elif e_params is None:
            raise ValueError("Either 'method' or 'e_params' must be provided")

        # Legacy compatibility: ensure e_method is set
        if "e_method" not in e_params or e_params["e_method"] is None:
            method_name = e_params.get("e_method_name")
            if method_name and method_name in METHODS_DICT:
                e_params["e_method"] = METHODS_DICT[method_name]

        method = e_params["e_method"]
        method_name = e_params["e_method_name"]

        if method_name not in EMBEDDING_CONSTRUCTION_METHODS:
            raise Exception("Unknown embedding construction method!")

        graph = None
        if method.requires_graph:
            if g_params is None:
                raise ValueError(
                    f"Method {method_name} requires proximity graph, but "
                    f"graph params were not provided"
                )
            if g_params["weighted"] and m_params is None:
                raise ValueError(
                    f"Method {method_name} requires weights for proximity graph, but "
                    f"metric params were not provided"
                )

            graph = self.get_proximity_graph(m_params, g_params)

        if method.requires_distmat and self.distmat is None:
            raise Exception(
                f"No distmat provided for {method_name} method."
                f" Try constructing it first with get_distmat() method"
            )

        emb = Embedding(self.data, self.distmat, self.labels, e_params, g=graph)

        # For neural network methods, extract NN-specific params from e_params to pass as kwargs
        if method.nn_based:
            nn_kwargs = kwargs or {}
            # Extract neural network specific parameters from e_params
            nn_params = [
                "epochs",
                "lr",
                "batch_size",
                "seed",
                "verbose",
                "feature_dropout",
                "enc_kwargs",
                "dec_kwargs",
                "kld_weight",
                "inter_dim",
                "train_size",
                "add_corr_loss",
                "corr_hyperweight",
                "add_mi_loss",
                "mi_hyperweight",
                "minimize_mi_data",
                "log_every",
                "device",
                "continue_learning",
                # flexible_ae specific parameters
                "architecture",
                "loss_components",
                "logger",
            ]
            for param in nn_params:
                if param in e_params:
                    nn_kwargs[param] = e_params[param]
            emb.build(kwargs=nn_kwargs)
        else:
            # Extract verbose for non-neural network methods
            build_kwargs = kwargs or {}
            if 'verbose' in e_params:
                build_kwargs['verbose'] = e_params['verbose']
            emb.build(kwargs=build_kwargs)

        return emb

    def get_proximity_graph(self, m_params, g_params):
        if g_params["g_method_name"] not in GRAPH_CONSTRUCTION_METHODS:
            raise Exception("Unknown graph construction method!")

        graph = ProximityGraph(self.data, m_params, g_params, verbose=self.verbose)
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
