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
    """
    Graph built on data points which represents the underlying manifold
    """

    def __init__(self, d, m_params, g_params, create_nx_graph=False):
        self.all_metric_params = m_param_filter(m_params)
        self.metric = m_params["metric_name"]
        self.metric_args = {
            key: self.all_metric_params[key]
            for key in self.all_metric_params.keys()
            if key not in ["metric_name", "sigma"]
        }

        all_params = g_param_filter(g_params)
        for key in all_params:
            setattr(self, key, g_params[key])

        self.data = d

        self.construct_adjacency()
        # TODO: add graph_preprocessing to changeable graph params
        super(ProximityGraph, self).__init__(
            adj=self.adj,
            preprocessing="giant_cc",
            create_nx_graph=create_nx_graph,
            directed=False,
            weighted=all_params["weighted"],
        )

        node_mapping = self._init_to_final_node_mapping
        original_n = self.data.shape[1]  # Data is (features, samples)
        lost_nodes = set(range(original_n)) - set(list(node_mapping.keys()))
        if len(lost_nodes) > 0:
            print(f"{len(lost_nodes)} nodes lost after giant component creation!")

            if len(lost_nodes) >= self.max_deleted_nodes * original_n:
                raise Exception(
                    f"more than {self.max_deleted_nodes * 100} % of nodes discarded during gc creation!"
                )
            else:
                self.lost_nodes = lost_nodes
                connected = [i for i in range(original_n) if i not in self.lost_nodes]
                self.bin_adj = self.bin_adj[connected, :].tocsc()[:, connected].tocsr()
                self.neigh_distmat = (
                    self.neigh_distmat[connected, :].tocsc()[:, connected].tocsr()
                )

                # Update k-NN arrays if they exist
                if hasattr(self, "knn_indices") and self.knn_indices is not None:
                    self.knn_indices = self.knn_indices[connected]
                if hasattr(self, "knn_distances") and self.knn_distances is not None:
                    self.knn_distances = self.knn_distances[connected]

        self._checkpoint()

        # arr = np.array(range(self.n)).reshape(-1, 1)
        # self.timediff = cdist(arr, arr, 'cityblock')
        # self.norm_timediff = self.timediff / (self.n / 3)

    def distances_to_affinities(self):
        if self.neigh_distmat is None:
            raise Exception("distances between nearest neighbors not available")

        if not self.weighted:
            raise Exception("no need to construct affinities for binary graph weights")

        if self.dist_to_aff == "hk":
            sigma = self.all_metric_params["sigma"]
            self.adj = self.neigh_distmat.copy()
            sqdist_matrix = self.neigh_distmat.multiply(self.neigh_distmat)
            mean_sqdist = sqdist_matrix.sum() / sqdist_matrix.nnz
            self.adj.data = np.exp(-sqdist_matrix.data / (1.0 * sigma * mean_sqdist))
            # Ensure symmetry after transformation
            self.adj = (self.adj + self.adj.T) / 2.0

    def construct_adjacency(self):
        construct_fn = getattr(self, "create_" + self.g_method_name + "_graph_")
        construct_fn()

    def _checkpoint(self):
        if self.adj is not None:
            if not sp.issparse(self.adj):
                # check for sparsity violation
                raise Exception("Adjacency matrix is not sparse!")
            self.adj = check_symmetric(self.adj, raise_exception=True)
            self.bin_adj = check_symmetric(self.bin_adj, raise_exception=True)
            self.neigh_distmat = check_symmetric(
                self.neigh_distmat, raise_exception=True
            )
            print("Adjacency symmetry confirmed")

        else:
            raise Exception(
                "Adjacency matrix is not constructed, checkpoint routines are unavailable"
            )

    def create_umap_graph_(self):
        RAND = np.random.RandomState(42)
        adj, _, _, dists = fuzzy_simplicial_set(
            self.data.T,
            self.nn,
            metric=self.metric,
            metric_kwds=self.metric_args,
            random_state=RAND,
            return_dists=True,
        )

        self.adj = sp.csr_matrix(adj)
        self.bin_adj = (self.adj > 0).astype(int)
        self.neigh_distmat = (
            sp.csr_matrix(dists) if dists is not None else sp.csr_matrix(self.adj.shape)
        )

        # Initialize k-NN attributes to None (not directly available for UMAP)
        self.knn_indices = None
        self.knn_distances = None

    def create_knn_graph_(self):
        if callable(self.metric):
            # Custom metric function passed directly
            curr_metric = self.metric
        elif self.metric in named_distances:
            # Built-in metric name
            curr_metric = self.metric
        else:
            # Try to find custom metric in globals (backward compatibility)
            try:
                curr_metric = globals()[self.metric]
                if not callable(curr_metric):
                    raise ValueError(
                        f"Global '{self.metric}' is not a callable metric function"
                    )
            except KeyError:
                raise ValueError(
                    f"Unknown metric '{self.metric}'. Must be one of {list(named_distances.keys())}, "
                    f"a callable function, or a global function name."
                )

        N = self.data.shape[1]
        index = pynndescent.NNDescent(
            self.data.T,
            metric=curr_metric,
            metric_kwds=self.metric_args,
            n_neighbors=self.nn + 1,
            diversify_prob=1.0,
            pruning_degree_multiplier=1.5,
        )

        neighs, dists = index.neighbor_graph

        # Save the k-NN graph for potential use in intrinsic dimension estimation
        self.knn_indices = neighs
        self.knn_distances = dists

        neigh_cols = neighs[:, 1:].flatten()
        dist_vals = dists[:, 1:].flatten()
        neigh_rows = np.repeat(np.arange(N), self.nn)

        all_neigh_cols = np.concatenate((neigh_cols, neigh_rows))
        all_neigh_rows = np.concatenate((neigh_rows, neigh_cols))
        all_dist_vals = np.concatenate((dist_vals, dist_vals))

        self.bin_adj = sp.csr_matrix(
            (
                np.array([True for _ in range(2 * N * self.nn)]),
                (all_neigh_rows, all_neigh_cols),
            )
        )

        self.neigh_distmat = sp.csr_matrix(
            (all_dist_vals, (all_neigh_rows, all_neigh_cols))
        )

        if self.weighted:
            self.distances_to_affinities()
        else:
            self.adj = self.bin_adj.copy()

    def create_auto_knn_graph_(self):
        A = kneighbors_graph(
            self.data.T, self.nn, mode="connectivity", include_self=False
        )
        A.setdiag(0)
        A = A.astype(bool, casting="unsafe", copy=True)
        A = A + A.T
        self.adj = A
        self.bin_adj = A.copy()
        self.neigh_distmat = sp.csr_matrix(A.shape)

        # Initialize k-NN attributes to None (not available for this method)
        self.knn_indices = None
        self.knn_distances = None

    def create_eps_graph_(self):
        """Create epsilon-ball graph where edges connect points within distance eps."""
        from sklearn.neighbors import radius_neighbors_graph

        # Use radius_neighbors_graph to create epsilon-ball graph
        # eps is the maximum distance between points
        # mode='connectivity' gives binary adjacency matrix
        # include_self=False excludes self-loops
        A = radius_neighbors_graph(
            self.data.T,
            self.eps,
            mode="connectivity",
            metric=self.metric,
            metric_params=self.metric_args,
            include_self=False,
        )

        # Ensure symmetry
        A = A + A.T
        A = A.astype(bool, casting="unsafe", copy=True)

        # Check if graph is too sparse or too dense
        nnz_ratio = A.nnz / (self.data.shape[1] * (self.data.shape[1] - 1))
        if nnz_ratio < self.eps_min:
            raise ValueError(
                f"Epsilon graph too sparse (density={nnz_ratio:.4f} < eps_min={self.eps_min}). "
                f"Consider increasing eps parameter."
            )
        if nnz_ratio > 0.5:
            print(
                f"WARNING: Epsilon graph is dense (density={nnz_ratio:.4f}). "
                f"Consider decreasing eps parameter."
            )

        self.adj = A
        self.bin_adj = A.copy()

        # For weighted graphs, compute distance matrix
        if self.weighted:
            # Get distance matrix for connected pairs
            D = radius_neighbors_graph(
                self.data.T,
                self.eps,
                mode="distance",
                metric=self.metric,
                metric_params=self.metric_args,
                include_self=False,
            )
            D = D + D.T
            self.neigh_distmat = D
            self.distances_to_affinities()
        else:
            # For unweighted graphs, create sparse zero matrix for distances
            self.neigh_distmat = sp.csr_matrix(A.shape)

        # Initialize k-NN attributes to None (not available for epsilon-ball graphs)
        self.knn_indices = None
        self.knn_distances = None

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
            return -a / 2.0 * (x - max(x)) ** 2

        if self.g_method_name not in ["knn"]:
            raise Exception(
                "Distance matrix construction missed! Only knn graph method is supported for intrinsic dimension calculation."
            )

        print("Calculating graph internal dimension...")

        if mode == "fast":
            # Use neigh_distmat which contains the distances
            distmat = self.neigh_distmat.copy()
            indices = list(
                npr.permutation(
                    npr.choice(self.n, size=self.n // factor, replace=False)
                )
            )
            dm = distmat[indices, :][:, indices]

        elif mode == "full":
            # Use neigh_distmat for full mode too
            dm = self.neigh_distmat.copy()

        print("Shortest path computation started, distance matrix size: ", dm.shape)
        spmatrix = shortest_path(dm, method="D", directed=False)
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
        # avg = np.mean(all_dists)  # Not currently used but might be useful for future analysis
        std = np.std(all_dists / dmax)

        res = []
        # print(distr_x)
        # print(distr_y)
        # Create consistent mask for both x and y
        mask = (distr_x > 1 - 2.0 * std) & (distr_x <= 1) & (distr_y > 1e-6)
        left_distr_x = distr_x[mask]
        left_distr_y = np.log10(distr_y[mask])

        for D in [0.1 * x for x in range(10, 260)]:
            y = func(left_distr_x, D - 1)
            res.append(np.linalg.norm(y - left_distr_y) / np.sqrt(len(y)))

        plot = 0
        if plot:
            fig = plt.figure(2, figsize=(12, 10))
            plt.plot(np.linspace(0, len(res) / 10.0, num=len(res)), res)

        Dmin = 0.1 * (np.argmax(-np.array(res)) + 1)
        print("Dmin = ", Dmin)
        fit = curve_fit(func2, left_distr_x, left_distr_y)
        # print(fit)
        a = fit[0][0]
        # print('Dfit = ', Dfit)

        plot = 0
        if plot:
            fig = plt.figure(1, figsize=(12, 10))
            ax = fig.add_subplot(111)
            ax.hist(
                all_dists / dmax,
                bins=nbins,
                histtype="stepfilled",
                density=True,
                log=True,
            )

        alpha = 2.0
        R = np.sqrt(2 * a)
        print("R = ", R)
        Dpr = 1 - alpha**2 / (2 * np.log(np.cos(alpha * np.pi / 2.0 / R)))
        print("D_calc = ", Dpr)

        return Dmin, Dpr

    def scaling(self):
        # Convert sparse matrix to dense for trace calculation
        # .A is deprecated, use .toarray() instead
        mat = self.adj.astype(bool).toarray().astype(int)
        diagsums = []
        for i in range(self.n - 1):
            diagsums.append(
                np.trace(mat, offset=i, dtype=None, out=None) / (self.n - i)
            )

        return diagsums

    def get_int_dim(
        self, method="geodesic", force_recompute=False, logger=None, **kwargs
    ):
        """
        Estimate intrinsic dimension using graph-based methods.

        This method estimates the intrinsic dimensionality using either
        geodesic distances on the graph or k-NN method with precomputed
        k-NN information. Results are cached to avoid recomputation.

        Parameters
        ----------
        method : {'geodesic', 'nn'}, default='geodesic'
            The method to use for intrinsic dimension estimation:
            - 'geodesic': Uses geodesic distances (shortest paths) on the graph
            - 'nn': k-nearest neighbor based estimation using saved k-NN data

        force_recompute : bool, default=False
            If True, recompute the dimension even if it has been computed before.
            If False, return cached result if available.

        logger : logging.Logger, optional
            Logger instance for logging messages. If None, creates a default logger.

        **kwargs : dict
            Additional parameters passed to the dimension estimation method:
            - For 'geodesic': mode ('full'/'fast'), factor (subsampling factor)

        Returns
        -------
        dimension : float
            The estimated intrinsic dimension of the dataset/manifold.

        Raises
        ------
        ValueError
            If an unknown method is specified or required data is not available.

        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.graph import ProximityGraph
        >>> # Generate Swiss roll data
        >>> from sklearn.datasets import make_swiss_roll
        >>> data, _ = make_swiss_roll(n_samples=500)
        >>> # Create proximity graph
        >>> m_params = {'metric_name': 'euclidean', 'sigma': 1.0}
        >>> g_params = {'g_method_name': 'knn', 'nn': 15, 'weighted': True,
        ...             'dist_to_aff': 'hk', 'max_deleted_nodes': 0.5}
        >>> graph = ProximityGraph(data.T, m_params, g_params)
        >>> # Estimate dimension using geodesic method
        >>> dim_geo = graph.get_int_dim(method='geodesic')
        >>> # Get cached result (fast)
        >>> dim_geo_cached = graph.get_int_dim(method='geodesic')
        >>> # Force recomputation
        >>> dim_geo_new = graph.get_int_dim(method='geodesic', force_recompute=True)
        >>> # Or using nn method (only if k-NN graph was used)
        >>> dim_nn = graph.get_int_dim(method='nn')
        >>> # Access all computed dimensions
        >>> print(graph.intrinsic_dimensions)
        """
        import logging
        from ..dimensionality import geodesic_dimension, nn_dimension

        # Setup logger
        if logger is None:
            logger = logging.getLogger(f"{self.__class__.__name__}.get_int_dim")

        # Initialize cache if not exists
        if not hasattr(self, "intrinsic_dimensions"):
            self.intrinsic_dimensions = {}

        # Validate method
        valid_methods = {"geodesic", "nn"}
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Choose from {valid_methods}")

        # Create cache key for method with parameters
        cache_key = method
        if method == "geodesic":
            mode = kwargs.get("mode", "full")
            factor = kwargs.get("factor", 2)
            cache_key = f"{method}_{mode}_f{factor}"

        # Check cache unless force_recompute
        if not force_recompute and cache_key in self.intrinsic_dimensions:
            logger.info(
                f"Returning cached intrinsic dimension for {cache_key}: "
                f"{self.intrinsic_dimensions[cache_key]:.3f}"
            )
            return self.intrinsic_dimensions[cache_key]

        logger.info(f"Computing intrinsic dimension using {method} method")

        if method == "geodesic":
            # Extract parameters
            mode = kwargs.get("mode", "full")
            factor = kwargs.get("factor", 2)

            # Use the adjacency matrix with distances if available
            if hasattr(self, "neigh_distmat") and self.neigh_distmat is not None:
                # Use distance matrix if available
                graph_for_geodesic = self.neigh_distmat
            else:
                # Fall back to binary adjacency
                graph_for_geodesic = self.adj

            logger.debug(
                f"Using graph with shape {graph_for_geodesic.shape}, "
                f"nnz={graph_for_geodesic.nnz}, mode={mode}, factor={factor}"
            )

            dimension = geodesic_dimension(
                graph=graph_for_geodesic, mode=mode, factor=factor
            )

        elif method == "nn":
            # Use k-NN method with saved k-NN data
            # Check if we have saved k-NN data
            if (
                hasattr(self, "knn_indices")
                and self.knn_indices is not None
                and hasattr(self, "knn_distances")
                and self.knn_distances is not None
            ):

                # Use the k value from graph construction
                k = self.nn if hasattr(self, "nn") else self.knn_indices.shape[1] - 1

                logger.debug(f"Using saved k-NN data with k={k}")

                # Use precomputed graph
                dimension = nn_dimension(
                    precomputed_graph=(self.knn_indices, self.knn_distances), k=k
                )
            else:
                raise ValueError(
                    f"nn method requires k-NN graph data which is not available. "
                    f"The graph was created with method '{self.g_method_name}' which "
                    f"does not provide k-NN neighbor information."
                )

        # Cache the result
        self.intrinsic_dimensions[cache_key] = dimension

        logger.info(f"Estimated intrinsic dimension: {dimension:.3f}")

        return dimension
