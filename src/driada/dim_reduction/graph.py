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
from pynndescent.distances import named_distances
from .dr_base import m_param_filter, g_param_filter

from ..network.net_base import Network


class ProximityGraph(Network):
    """Proximity graph for manifold learning and dimensionality reduction.
    
    Constructs a graph where nodes are data points and edges connect nearby points,
    capturing the local geometry of the underlying manifold. Supports multiple 
    graph construction methods including k-NN, UMAP fuzzy topology, and epsilon-ball.
    
    The graph can be used for manifold learning algorithms, intrinsic dimension
    estimation, and as input to spectral dimensionality reduction methods.
    
    Parameters
    ----------
    d : ndarray
        Data matrix of shape (n_features, n_samples). Each column is a data point.
    m_params : dict
        Metric parameters dictionary. Required key:
        - 'metric_name' : str or callable
            Distance metric name from pynndescent.distances.named_distances
            ('euclidean', 'cosine', 'manhattan', etc.), 'hyperbolic', 
            or a callable custom distance function
        Optional keys (filtered by m_param_filter):
        - 'sigma' : float, bandwidth for heat kernel affinity transformation
        - 'p' : float, parameter for minkowski metric
        - Additional metric-specific parameters passed to distance function
    g_params : dict
        Graph construction parameters. Required key:
        - 'g_method_name' : str, graph construction method 
            Options: 'knn', 'umap', 'auto_knn', 'eps'
        Method-specific keys (filtered by g_param_filter):
        - 'nn' : int, number of nearest neighbors (for knn/umap/auto_knn)
        - 'eps' : float, epsilon radius (for eps method)
        - 'min_density' : float, minimum graph density threshold (for eps method)
        General optional keys:
        - 'weighted' : bool, whether to create weighted edges
        - 'dist_to_aff' : str, distance to affinity conversion ('hk' for heat kernel)
        - 'max_deleted_nodes' : float, maximum fraction of nodes that can be 
            deleted during giant component extraction (raises exception if exceeded)
        - 'graph_preprocessing' : str, preprocessing method (default: 'giant_cc')
    create_nx_graph : bool, default=False
        Whether to create NetworkX graph representation (passed to Network parent).
    verbose : bool, default=False
        Whether to print progress messages.
        
    Attributes
    ----------
    data : ndarray
        The input data matrix (n_features, n_samples).
    metric : str
        The distance metric name extracted from m_params.
    metric_args : dict
        Filtered metric parameters excluding 'metric_name' and 'sigma'.
    all_metric_params : dict
        All filtered metric parameters from m_param_filter.
    adj : sparse matrix
        Weighted adjacency matrix. For weighted graphs with dist_to_aff='hk',
        contains affinities exp(-d²/(sigma*mean_squared_dist)). For unweighted
        graphs, same as bin_adj.
    bin_adj : sparse matrix
        Binary adjacency matrix indicating connections.
    neigh_distmat : sparse matrix
        Sparse matrix of distances between connected neighbors. For unweighted
        graphs or methods without distance computation, contains zeros in sparse format.
    knn_indices : ndarray or None
        k-NN indices array of shape (n_nodes, k+1) including self (only for 'knn' method).
    knn_distances : ndarray or None
        k-NN distances array of shape (n_nodes, k+1) including self (only for 'knn' method).
    lost_nodes : set
        Set of node indices removed during giant component preprocessing. Empty set
        if no preprocessing or no nodes lost.
    intrinsic_dimensions : dict
        Cached intrinsic dimension estimates. Keys are method names with parameters.
    
    Plus all attributes from g_params set via setattr.
        
    Methods
    -------
    construct_adjacency()
        Dynamically call graph construction method based on g_method_name.
    distances_to_affinities()
        Convert neigh_distmat to affinity weights using heat kernel (only 'hk' implemented).
    create_knn_graph_()
        Create k-NN graph using pynndescent. Saves knn_indices and knn_distances.
    create_umap_graph_()  
        Create graph using UMAP's fuzzy_simplicial_set. Sets knn arrays to None.
    create_auto_knn_graph_()
        Create unweighted k-NN graph using sklearn. Sets knn arrays to None.
    create_eps_graph_()
        Create epsilon-ball graph. Checks density against min_density. Sets knn arrays to None.
    get_int_dim(method='geodesic', force_recompute=False, logger=None, **kwargs)
        Estimate intrinsic dimension. Methods: 'geodesic' (uses neigh_distmat or adj),
        'nn' (requires saved knn data from 'knn' method). Results are cached.
    scaling()
        Compute diagonal sums of adjacency matrix powers for graph scaling analysis.
    _checkpoint()
        Verify graph sparsity and symmetry. Called after construction.
        
    Raises
    ------
    Exception
        If more than max_deleted_nodes fraction of nodes are lost during preprocessing.
    Exception
        If adjacency matrix is not sparse or not symmetric.
    ValueError
        If unknown metric or graph method specified.
        
    Notes
    -----
    - Inherits from Network class, gaining spectral analysis capabilities
    - All graphs are enforced to be symmetric/undirected via check_symmetric
    - Giant connected component extraction is default preprocessing
    - For weighted graphs with 'hk', distances are converted to similarities
    - Node indices are remapped after giant component extraction
    
    Examples
    --------
    >>> import numpy as np
    >>> from driada.dim_reduction.graph import ProximityGraph
    >>> # Generate sample data
    >>> np.random.seed(42)  # For reproducible results
    >>> data = np.random.randn(3, 100)  # 100 points in 3D
    >>> # Define metric parameters
    >>> m_params = {'metric_name': 'euclidean', 'sigma': 1.0}
    >>> # Define graph parameters for k-NN graph
    >>> g_params = {
    ...     'g_method_name': 'knn',
    ...     'nn': 15,
    ...     'weighted': True,
    ...     'dist_to_aff': 'hk',
    ...     'max_deleted_nodes': 0.1
    ... }
    >>> # Create proximity graph
    >>> graph = ProximityGraph(data, m_params, g_params)
    >>> edges = graph.adj.nnz//2
    >>> print(f"Graph has {graph.n} nodes and {edges} edges")  # doctest: +ELLIPSIS
    Graph has 100 nodes and ... edges
    """

    def __init__(self, d, m_params, g_params, create_nx_graph=False, verbose=False):
        """Initialize proximity graph from data.
        
        Parameters
        ----------
        d : ndarray of shape (n_features, n_samples)
            Data matrix where each column is a sample.
        m_params : dict
            Metric parameters. Must contain 'metric_name'.
        g_params : dict  
            Graph parameters. Must contain 'g_method_name'.
        create_nx_graph : bool, default=False
            Whether to create NetworkX representation.
        verbose : bool, default=False
            Whether to print progress messages.
            
        Raises
        ------
        ValueError
            If data is not 2D or is empty.
            If required parameters are missing.
            
        Notes
        -----
        lost_nodes attribute is always set (even if empty).
        Data shape assumes (features, samples) format.        """
        # Validate input data
        d = np.asarray(d)
        if d.ndim != 2:
            raise ValueError(f"Data must be 2D array, got {d.ndim}D")
        if d.size == 0:
            raise ValueError("Data array is empty")
            
        self.all_metric_params = m_param_filter(m_params)
        self.metric = m_params["metric_name"]
        self.metric_args = {
            key: self.all_metric_params[key]
            for key in self.all_metric_params.keys()
            if key not in ["metric_name", "sigma"]
        }

        all_params = g_param_filter(g_params)
        
        # Safe attribute setting - only set allowed attributes
        allowed_attrs = {'g_method_name', 'nn', 'eps', 'min_density', 'perplexity',
                        'max_deleted_nodes', 'weighted', 'dist_to_aff', 'graph_preprocessing'}
        for key in all_params:
            if key in allowed_attrs:
                setattr(self, key, g_params[key])
            else:
                if self.verbose:
                    print(f"Warning: Ignoring unknown parameter '{key}'")
                    
        # Set default for max_deleted_nodes if not provided
        if not hasattr(self, 'max_deleted_nodes'):
            self.max_deleted_nodes = 0.5

        self.data = d
        self.verbose = verbose

        self.construct_adjacency()
        # Get graph preprocessing parameter with default value
        graph_preprocessing = all_params.get("graph_preprocessing", "giant_cc")
        super(ProximityGraph, self).__init__(
            adj=self.adj,
            preprocessing=graph_preprocessing,
            create_nx_graph=create_nx_graph,
            directed=False,
            weighted=all_params["weighted"],
        )

        node_mapping = self._init_to_final_node_mapping
        original_n = self.data.shape[1]  # Data is (features, samples)
        lost_nodes = set(range(original_n)) - set(node_mapping.values())
        self.lost_nodes = lost_nodes  # Always set the attribute
        if len(lost_nodes) > 0:
            if self.verbose:
                print(f"{len(lost_nodes)} nodes lost after giant component creation!")

            if len(lost_nodes) >= self.max_deleted_nodes * original_n:
                raise Exception(
                    f"more than {self.max_deleted_nodes * 100} % of nodes discarded during gc creation!"
                )
            else:
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
        """Convert distance matrix to affinity matrix.
        
        Transforms distances between neighbors into similarity weights.
        The transformation method is specified by self.dist_to_aff parameter.
        
        Currently implemented methods:
        - 'hk': Heat kernel with adaptive bandwidth
          w_ij = exp(-d_ij^2 / (sigma * mean_squared_distance))
        
        Raises
        ------
        RuntimeError
            If no distance matrix is available or graph is not weighted.
        ValueError
            If sigma is not positive.
            
        Notes
        -----
        Only applies to weighted graphs. The resulting affinity matrix
        is symmetrized to ensure undirected graph structure.
        Mean distance uses only non-zero entries.        """
        if self.neigh_distmat is None:
            raise RuntimeError("distances between nearest neighbors not available")

        if not self.weighted:
            raise RuntimeError("no need to construct affinities for binary graph weights")

        if self.dist_to_aff == "hk":
            sigma = self.all_metric_params.get("sigma", 1.0)
            if sigma <= 0:
                raise ValueError(f"Sigma must be positive, got {sigma}")
                
            self.adj = self.neigh_distmat.copy()
            sqdist_matrix = self.neigh_distmat.multiply(self.neigh_distmat)
            mean_sqdist = sqdist_matrix.sum() / sqdist_matrix.nnz
            
            if mean_sqdist == 0:
                raise ValueError("All distances are zero - cannot compute meaningful affinities")
            
            self.adj.data = np.exp(-sqdist_matrix.data / (sigma * mean_sqdist))
                
            # Ensure symmetry after transformation
            self.adj = (self.adj + self.adj.T) / 2.0

    def construct_adjacency(self):
        """Construct the adjacency matrix using the specified graph method.
        
        Dynamically calls the appropriate graph construction method based on
        self.g_method_name (e.g., 'knn', 'umap', 'auto_knn', 'eps').
        
        Raises
        ------
        ValueError
            If g_method_name contains invalid characters.
        AttributeError
            If the specified graph construction method doesn't exist.        """
        # Validate method name
        if not self.g_method_name.replace('_', '').isalnum():
            raise ValueError(f"Invalid g_method_name: {self.g_method_name}")
            
        method_name = "create_" + self.g_method_name + "_graph_"
        if not hasattr(self, method_name):
            raise AttributeError(f"Unknown graph construction method: {self.g_method_name}")
            
        construct_fn = getattr(self, method_name)
        construct_fn()

    def _checkpoint(self):
        """Verify graph construction integrity.
        
        Checks that adjacency matrices are sparse and symmetric.
        Called after graph construction to ensure valid graph structure.
        
        Raises
        ------
        RuntimeError
            If adjacency is not sparse or not symmetric.
            
        Notes
        -----
        Validates that matrices are symmetric within tolerance.        """
        if self.adj is not None:
            if not sp.issparse(self.adj):
                # check for sparsity violation
                raise RuntimeError("Adjacency matrix is not sparse!")
                
            # Check symmetry - this will raise if not symmetric
            try:
                check_symmetric(self.adj, raise_exception=True)
            except Exception as e:
                raise RuntimeError(f"Adjacency matrix not symmetric: {e}")
                
            if hasattr(self, 'bin_adj') and self.bin_adj is not None:
                try:
                    check_symmetric(self.bin_adj, raise_exception=True)
                except Exception as e:
                    raise RuntimeError(f"Binary adjacency not symmetric: {e}")
                    
            if hasattr(self, 'neigh_distmat') and self.neigh_distmat is not None:
                try:
                    check_symmetric(self.neigh_distmat, raise_exception=True)
                except Exception as e:
                    raise RuntimeError(f"Neighbor distances not symmetric: {e}")
                    
            if self.verbose:
                print("Adjacency symmetry confirmed")

        else:
            raise RuntimeError(
                "Adjacency matrix is not constructed, checkpoint routines are unavailable"
            )

    def create_umap_graph_(self):
        """Create graph using UMAP's fuzzy simplicial set construction.
        
        Uses UMAP's algorithm to build a fuzzy topological representation
        that captures both local and global structure of the data manifold.
        
        Notes
        -----
        The resulting graph has weighted edges representing fuzzy set membership.
        Sets self.adj (weighted), self.bin_adj (binary), and self.neigh_distmat.
        Uses fixed seed=42 for reproducibility.
        Data is transposed to match UMAP's expected (n_samples, n_features) format.        """
        # TODO: Make seed configurable via g_params
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
        """Create k-nearest neighbors graph using pynndescent.
        
        Constructs a symmetric k-NN graph where each point is connected to its
        k nearest neighbors. Uses approximate nearest neighbor search for efficiency.
        
        Raises
        ------
        ValueError
            If nn exceeds number of samples minus 1.
            If metric is unknown or invalid.
            
        Notes
        -----
        - Supports custom metrics via callable functions or named distances
        - Stores k-NN indices and distances for potential reuse
        - Creates both weighted and binary adjacency matrices
        - Uses diversify_prob=1.0 and pruning_degree_multiplier=1.5
        - Data is transposed to match pynndescent expected format
        - Excludes self-connections (uses nn+1 neighbors then skips first)        """
        N = self.data.shape[1]
        if self.nn >= N:
            raise ValueError(f"nn ({self.nn}) must be less than number of samples ({N})")
            
        if callable(self.metric):
            # Custom metric function passed directly
            curr_metric = self.metric
        elif self.metric in named_distances:
            # Built-in metric name
            curr_metric = self.metric
        else:
            # Safer approach - check if metric is a known string
            raise ValueError(
                f"Unknown metric '{self.metric}'. Must be one of {list(named_distances.keys())} "
                f"or a callable function."
            )

        index = pynndescent.NNDescent(
            self.data.T,  # Transpose for (n_samples, n_features)
            metric=curr_metric,
            metric_kwds=self.metric_args,
            n_neighbors=self.nn + 1,  # +1 to exclude self
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

        # Create initial sparse matrix
        self.neigh_distmat = sp.csr_matrix(
            (dist_vals, (neigh_rows, neigh_cols)), shape=(N, N)
        )
        
        # Symmetrize by taking minimum (same as sklearn's approach)
        # This avoids doubling distances when edges appear in both directions
        self.neigh_distmat = self.neigh_distmat.minimum(self.neigh_distmat.T)
        
        # Create binary adjacency from distance matrix
        self.bin_adj = (self.neigh_distmat > 0).astype(int)

        if self.weighted:
            self.distances_to_affinities()
        else:
            self.adj = self.bin_adj.copy()

    def create_auto_knn_graph_(self):
        """Create k-NN graph using scikit-learn's implementation.
        
        A simpler alternative to pynndescent that uses sklearn's 
        kneighbors_graph. Creates an unweighted, symmetric graph.
        
        Raises
        ------
        ValueError
            If nn exceeds number of samples minus 1.
            
        Notes
        -----
        - Always creates unweighted graphs (connectivity mode)
        - Uses 'auto' algorithm selection in sklearn
        - Sets diagonal to 0 to exclude self-connections
        - Does not store knn_indices or knn_distances
        - Data is transposed to match sklearn expected format
        - Symmetrizes by A = A + A.T        """
        N = self.data.shape[1]
        if self.nn >= N:
            raise ValueError(f"nn ({self.nn}) must be less than number of samples ({N})")
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
        """Create epsilon-ball graph where edges connect points within distance eps.
        
        Constructs a graph by connecting all pairs of points whose distance is
        less than or equal to the epsilon threshold. Uses sklearn's 
        radius_neighbors_graph for efficient computation.
        
        The resulting graph density is checked against self.min_density to ensure
        sufficient connectivity. Issues a warning if density exceeds 0.5.
        
        Raises
        ------
        ValueError
            If eps is not positive.
            If min_density is not in (0, 1].
            If graph density is below self.min_density threshold.
            
        Notes
        -----
        Sets self.adj (weighted or binary), self.bin_adj (binary), and 
        self.neigh_distmat (distances for weighted graphs, zero matrix otherwise).
        For weighted graphs, distances are converted to affinities.
        Does not store knn_indices or knn_distances.
        Data is transposed to match sklearn expected format.        """
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if not 0 < self.min_density <= 1:
            raise ValueError(f"min_density must be in (0, 1], got {self.min_density}")
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
        if nnz_ratio < self.min_density:
            raise ValueError(
                f"Epsilon graph too sparse (density={nnz_ratio:.4f} < min_density={self.min_density}). "
                f"Consider increasing eps parameter."
            )
        if nnz_ratio > 0.5:
            if self.verbose:
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


    def scaling(self):
        """Compute normalized diagonal sums of the binary adjacency matrix.
        
        Analyzes graph structure by computing the average connectivity at different
        node separations. For each diagonal offset i, computes the fraction of
        possible edges that exist between nodes separated by i positions.
        
        Returns
        -------
        list of float
            List of length n-1 where element i is the average value along the 
            i-th super-diagonal of the binary adjacency matrix, normalized by
            the diagonal length (n-i).
            
        Notes
        -----
        This provides a simple graph structure summary. The first elements 
        indicate local connectivity while later elements show longer-range
        connections. Regular lattices show characteristic patterns.
        Uses sparse matrix operations to avoid memory issues with large graphs.        """
        # Work with sparse binary adjacency matrix
        mat = self.bin_adj.astype(bool)
        diagsums = []
        
        # Extract diagonal sums using sparse operations
        for i in range(self.n - 1):
            # Get indices for the i-th superdiagonal
            diag_length = self.n - i
            row_indices = np.arange(diag_length)
            col_indices = row_indices + i
            
            # Sum the values along this diagonal
            diag_sum = 0
            for j in range(diag_length):
                if mat[row_indices[j], col_indices[j]]:
                    diag_sum += 1
                    
            diagsums.append(diag_sum / diag_length)

        return diagsums

    def get_int_dim(
        self, method="geodesic", force_recompute=False, logger=None, **kwargs
    ):
        """Estimate intrinsic dimension using graph-based methods.

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

        **kwargs
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
        >>> np.random.seed(42)  # For reproducible results
        >>> data, _ = make_swiss_roll(n_samples=500, random_state=42)
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
        >>> dims = sorted(graph.intrinsic_dimensions.keys())
        >>> print(dims)
        ['geodesic_full_f2', 'nn']
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
            
            # Warn about weighted graphs with transformations
            if self.weighted and hasattr(self, "dist_to_aff") and self.dist_to_aff is not None:
                import warnings
                warnings.warn(
                    "Geodesic dimension estimation on weighted graphs with distance-to-affinity "
                    "transformations (e.g., heat kernel) may give different results than expected. "
                    "Consider using an unweighted graph or raw distances for more consistent results.",
                    RuntimeWarning
                )

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
