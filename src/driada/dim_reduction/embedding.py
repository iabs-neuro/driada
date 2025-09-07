import numpy as np
from scipy.sparse.linalg import eigs
import umap.umap_ as umap
from pydiffmap import diffusion_map as dm
from scipy.sparse.csgraph import shortest_path

from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding, Isomap, LocallyLinearEmbedding, TSNE
from sklearn.model_selection import train_test_split

# from sklearn.cluster.spectral import discretize

# warnings.filterwarnings("ignore")

from .dr_base import e_param_filter
from .graph import ProximityGraph

try:
    from .neural import AE, VAE, NeuroDataset
except ImportError:
    # torch is optional dependency
    pass
from .mvu import MaximumVarianceUnfolding

from ..network.matrix_utils import get_inv_sqrt_diag_matrix
from ..utils.data import check_positive, check_nonnegative


class Embedding:
    """Low-dimensional representation of high-dimensional data.
    
    This is an internal class typically created by MVData.get_embedding().
    It provides a unified interface for various dimensionality reduction methods
    including linear (PCA), non-linear manifold learning (Isomap, LLE, UMAP),
    spectral methods (Laplacian Eigenmaps, Diffusion Maps), and neural
    network-based approaches (autoencoders, VAEs).
    
    Parameters
    ----------
    init_data : ndarray
        Input data matrix of shape (n_features, n_samples).
    init_distmat : ndarray or None
        Precomputed distance matrix of shape (n_samples, n_samples).
        Used by methods like MDS when available.
    labels : array-like
        Labels for data points, used for visualization and evaluation.
    params : dict
        Filtered embedding parameters from e_param_filter(). Must contain:
        - 'e_method' : DRMethod object from METHODS_DICT
        - 'e_method_name' : str, method name (e.g., 'pca', 'umap')
        - 'dim' : int, target embedding dimension
        May also contain method-specific keys:
        - 'min_dist' : float, for UMAP only
        - 'dm_alpha' : float, for dmaps/auto_dmaps only
        - 'dm_t' : int, for dmaps/auto_dmaps only
    g : ProximityGraph, optional
        Precomputed proximity graph. Required for graph-based methods
        (LE, Isomap, LLE, etc.). If None, must be created before building.
        
    Attributes
    ----------
    graph : ProximityGraph
        The proximity graph used by graph-based methods.
    coords : ndarray or None
        Embedding coordinates of shape (dim, n_samples). None until build() is called.
    labels : array-like
        Data point labels.
    nclasses : int
        Number of unique classes in labels.
    init_data : ndarray
        Original high-dimensional data.
    init_distmat : ndarray or None
        Precomputed distance matrix if provided.
    all_params : dict
        Filtered parameters from e_param_filter.
    transformation_matrix : ndarray or None
        For linear methods (PCA), the transformation matrix to project new data.
    nnmodel : nn.Module or None
        For neural network methods, the trained model.
    nn_loss : float or None
        For neural network methods, the final loss value.
    reducer_ : object
        The underlying reducer object (sklearn model, etc.) for potential reuse.
        
    Plus all parameters from params dict set as attributes via setattr.
        
    Methods
    -------
    build(kwargs=None)
        Build the embedding using the specified method.
    create_pca_embedding_(verbose=True)
        Linear projection via Principal Component Analysis.
    create_mds_embedding_()
        Classical multidimensional scaling preserving distances.
    create_isomap_embedding_()
        Non-linear embedding preserving geodesic distances.
    create_lle_embedding_()
        Locally Linear Embedding assuming local linearity.
    create_hlle_embedding_()
        Hessian LLE with better curvature preservation.
    create_mvu_embedding_()
        Maximum Variance Unfolding via semidefinite programming.
    create_le_embedding_()
        Laplacian Eigenmaps using graph spectral decomposition.
    create_auto_le_embedding_()
        Laplacian Eigenmaps using sklearn implementation.
    create_dmaps_embedding_()
        Diffusion Maps with manual implementation.
    create_auto_dmaps_embedding_()
        Diffusion Maps using pydiffmap library.
    create_tsne_embedding_()
        t-SNE for visualization with local structure preservation.
    create_umap_embedding_()
        UMAP balancing local and global structure.
    create_ae_embedding_(**kwargs)
        Autoencoder with optional correlation/MI losses (deprecated).
    create_vae_embedding_(**kwargs)
        Variational autoencoder (deprecated).
    create_flexible_ae_embedding_(**kwargs)
        Flexible autoencoder with modular loss composition.
    continue_learning(add_epochs, kwargs={})
        Continue training neural network models.
    to_mvdata()
        Convert embedding to MVData for further processing.
        
    Notes
    -----
    - Graph-based methods require a ProximityGraph to be provided or created
    - Neural methods (AE/VAE) require PyTorch to be installed
    - Some methods (MVU) require additional dependencies (cvxpy)
    - Coordinates are stored as (dim, n_samples) for consistency
    - The 'auto_' prefix indicates methods using external libraries
    
    Examples
    --------
    Direct instantiation (internal use):
    
    >>> from driada.dim_reduction.dr_base import METHODS_DICT
    >>> import numpy as np
    >>> data = np.random.randn(100, 500)  # 100 features, 500 samples
    >>> labels = np.random.randint(0, 3, 500)
    >>> # Parameters after e_param_filter
    >>> params = {
    ...     'e_method': METHODS_DICT['pca'],
    ...     'e_method_name': 'pca',
    ...     'dim': 2
    ... }
    >>> embedding = Embedding(data, None, labels, params)
    >>> embedding.build(kwargs={'verbose': False})
    >>> embedding.coords.shape
    (2, 500)
    
    Typical usage via MVData (recommended):
    
    >>> from driada.dim_reduction.data import MVData
    >>> mvdata = MVData(data, labels=labels)
    >>> # PCA embedding
    >>> embedding = mvdata.get_embedding(method='pca', dim=2, verbose=False)
    >>> embedding.coords.shape
    (2, 500)
    >>> # UMAP with custom parameters
    >>> embedding = mvdata.get_embedding(
    ...     method='umap', 
    ...     dim=2,
    ...     n_neighbors=15,
    ...     min_dist=0.1
    ... )
    """

    def __init__(self, init_data, init_distmat, labels, params, g=None):
        """Initialize Embedding object.
        
        Parameters
        ----------
        init_data : array-like
            Initial data matrix with shape (n_features, n_samples)
        init_distmat : array-like or None
            Pre-computed distance matrix, optional
        labels : array-like
            Labels for each data point
        params : dict
            Dictionary of embedding parameters, must include 'e_method', 
            'e_method_name', and method-specific parameters
        g : ProximityGraph, optional
            Pre-computed proximity graph. If None, may be computed later
            depending on the embedding method
        
        Raises
        ------
        TypeError
            If g is provided but not a ProximityGraph instance.
        AttributeError
            If required params keys are missing during attribute access.
            
        Notes
        -----
        All keys in params dict are set as instance attributes via setattr.
        The params are filtered through e_param_filter before use.
        
        Examples
        --------
        >>> # Typically created via MVData.get_embedding(), not directly
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> data = np.random.randn(100, 50)  # 100 features, 50 samples
        >>> labels = np.repeat([0, 1], 25)
        >>> params = {'e_method': METHODS_DICT['pca'], 'e_method_name': 'pca', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)
        """
        if g is not None:
            if isinstance(g, ProximityGraph):
                self.graph = g
            else:
                raise TypeError("Graph must be ProximityGraph instance")

        self.all_params = e_param_filter(params)
        for key in params:
            setattr(self, key, params[key])

        self.init_data = init_data
        self.init_distmat = init_distmat

        if self.e_method.is_linear:
            self.transformation_matrix = None

        self.labels = labels
        self.coords = None

        try:
            self.nclasses = len(set(self.labels))
        except (TypeError, AttributeError):
            # Fixed: was returning array instead of count
            self.nclasses = len(np.unique(self.labels))

        if self.e_method.nn_based:
            self.nnmodel = None

    def build(self, kwargs=None):
        """Build the embedding using the specified method.
        
        Dynamically calls the appropriate embedding creation method based on
        the embedding method name (e.g., 'pca' calls create_pca_embedding_).
        
        Parameters
        ----------
        kwargs : dict, optional
            Additional keyword arguments passed to the specific embedding method.
            For neural network methods (AE/VAE), this includes training parameters.
            
        Returns
        -------
        None
            Modifies self.coords in-place with embedding coordinates.
            
        Raises
        ------
        AttributeError
            If e_method_name is invalid or corresponding method not found.
            If self.graph is None for methods requiring it.
        Exception
            If the graph is disconnected and the method cannot handle it.
            
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> data = np.random.randn(20, 100)  # 20 features, 100 samples
        >>> labels = np.zeros(100)
        >>> params = {'e_method': METHODS_DICT['pca'], 'e_method_name': 'pca', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)  
        >>> emb.build(kwargs={'verbose': False})  # For PCA, no graph needed
        >>> emb.coords.shape
        (2, 100)
        """
        if kwargs is None:
            kwargs = dict()
        fn = getattr(self, "create_" + self.e_method_name + "_embedding_")

        if self.e_method.requires_graph:
            if (
                not self.graph.is_connected()
                and not self.e_method.handles_disconnected_graphs
            ):
                raise Exception("Graph is not connected!")

        fn(**kwargs)

    def create_pca_embedding_(self, verbose=True):
        """Create PCA (Principal Component Analysis) embedding.
        
        Linear dimensionality reduction using orthogonal transformation to
        convert data into linearly uncorrelated components ordered by variance.
        
        Parameters
        ----------
        verbose : bool, default=True
            Whether to print progress messages.
            
        Returns
        -------
        None
            Sets self.coords to shape (dim, n_samples).
            
        Raises
        ------
        AttributeError
            If self.dim is not set.
        ValueError
            If PCA fails (e.g., dim > n_features).
            
        Notes
        -----
        Sets self.coords to shape (dim, n_samples) and stores the PCA object
        in self.reducer_ for potential reuse or analysis.
        
        Data is transposed before fitting since init_data is (n_features, n_samples)
        while sklearn expects (n_samples, n_features).
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> data = np.random.randn(10, 500)  # 10 features, 500 samples
        >>> labels = np.random.randint(0, 3, 500)
        >>> params = {'e_method': METHODS_DICT['pca'], 'e_method_name': 'pca', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)
        >>> emb.create_pca_embedding_(verbose=False)
        >>> emb.coords.shape
        (2, 500)
        """
        if verbose:
            print("Calculating PCA embedding...")

        pca = PCA(n_components=self.dim)
        self.coords = pca.fit_transform(self.init_data.T).T
        self.reducer_ = pca
        # print(pca.explained_variance_ratio_)

    def create_isomap_embedding_(self):
        """Create Isomap embedding using geodesic distances.
        
        Non-linear dimensionality reduction through isometric mapping.
        Preserves geodesic distances between all points by first computing
        shortest paths on the neighborhood graph, then applying MDS.
        
        Returns
        -------
        None
            Sets self.coords to shape (dim, n_samples).
            
        Raises
        ------
        AttributeError
            If self.graph, self.dim, or self.nn not set.
        MemoryError
            If converting sparse matrix to dense fails.
            
        Notes
        -----
        Requires a proximity graph. Uses Dijkstra's algorithm to compute
        shortest paths, then applies classical MDS to the geodesic distance matrix.
        
        Warning: Converts sparse adjacency to dense matrix which may
        use excessive memory for large datasets.
        
        The Isomap object is stored in self.reducer_ for potential reuse.
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> from driada.dim_reduction.graph import ProximityGraph
        >>> data = np.random.randn(10, 500)  # 10 features, 500 samples
        >>> labels = np.random.randint(0, 3, 500)
        >>> params = {'e_method': METHODS_DICT['isomap'], 'e_method_name': 'isomap', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)
        >>> m_params = {'metric_name': 'euclidean'}
        >>> g_params = {'g_method_name': 'knn', 'nn': 10, 'weighted': False}
        >>> emb.graph = ProximityGraph(data, m_params, g_params)
        >>> emb.create_isomap_embedding_()
        >>> emb.coords.shape[0]  # Number of dimensions
        2
        """
        # Validate graph exists
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph not built. Call build_graph() first.")
            
        # Validate parameters
        check_positive(dim=self.dim, nn=self.graph.nn)
        
        A = self.graph.adj
        isomap_reducer = Isomap(
            n_components=self.dim, n_neighbors=self.graph.nn, metric="precomputed"
        )
        # self.coords = sp.csr_matrix(map.fit_transform(self.graph.data.A.T).T)
        spmatrix = shortest_path(A.todense(), method="D", directed=False)
        self.coords = isomap_reducer.fit_transform(spmatrix).T
        self.reducer_ = isomap_reducer

    def create_mds_embedding_(self):
        """Create MDS (Multi-Dimensional Scaling) embedding.
        
        Classical MDS finds a low-dimensional representation that preserves
        pairwise distances between points. Works with either a pre-computed
        distance matrix or calculates distances from the data.
        
        Returns
        -------
        None
            Sets self.coords to shape (dim, n_samples).
            
        Raises
        ------
        ImportError
            If sklearn.manifold.MDS not available.
        AttributeError
            If self.dim not set.
            
        Notes
        -----
        Sets self.coords to shape (dim, n_samples) containing the MDS coordinates.
        If init_distmat is provided, uses it as a precomputed distance matrix.
        Otherwise, computes Euclidean distances from init_data.
        
        The algorithm minimizes the stress function:
        stress = sum((d_ij - ||x_i - x_j||)^2)
        where d_ij are the input distances.
        
        Uses fixed random_state=42 for reproducibility - to be changed in the future.
        The MDS object is stored in self.reducer_ for potential reuse.
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> from scipy.spatial.distance import pdist, squareform
        >>> data = np.random.randn(10, 500)  # 10 features, 500 samples
        >>> labels = np.random.randint(0, 3, 500)
        >>> distmat = squareform(pdist(data.T))
        >>> params = {'e_method': METHODS_DICT['mds'], 'e_method_name': 'mds', 'dim': 2}
        >>> emb = Embedding(data, distmat, labels, params)
        >>> emb.create_mds_embedding_()
        >>> emb.coords.shape
        (2, 500)
        """
        from sklearn.manifold import MDS

        # MDS typically uses a distance matrix
        if hasattr(self, "init_distmat") and self.init_distmat is not None:
            # Use provided distance matrix
            mds = MDS(
                n_components=self.dim, dissimilarity="precomputed", random_state=42
            )
            self.coords = mds.fit_transform(self.init_distmat).T
        else:
            # Compute from data
            mds = MDS(n_components=self.dim, random_state=42)
            self.coords = mds.fit_transform(self.init_data.T).T

        self.reducer_ = mds

    def create_mvu_embedding_(self):
        """Create Maximum Variance Unfolding (MVU) embedding.
        
        Non-linear dimensionality reduction that "unfolds" a manifold by
        maximizing variance while preserving local distances. Solves a
        semidefinite programming problem to find the optimal embedding.
        
        Sets self.coords to shape (dim, n_samples) embedding coordinates
        and self.reducer_ to the fitted MVU object.
            
        Raises
        ------
        ImportError
            If cvxpy is not installed.
        AttributeError
            If self.graph or required attributes are not set.
        ValueError
            If self.dim is not positive or if solver fails to converge.
            
        Notes
        -----
        Requires cvxpy for convex optimization. Uses the SCS solver with
        reasonable defaults for most datasets. The embedding preserves
        local neighborhood structure while maximizing global variance.
        
        The optimization problem maximizes tr(Y'Y) subject to:
        - Local distance preservation: ||y_i - y_j||² = ||x_i - x_j||² for neighbors
        - Centering: ∑y_i = 0
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> from driada.dim_reduction.graph import ProximityGraph
        >>> data = np.random.randn(10, 500)  # 10 features, 500 samples
        >>> labels = np.random.randint(0, 3, 500)
        >>> params = {'e_method': METHODS_DICT['mvu'], 'e_method_name': 'mvu', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)
        >>> m_params = {'metric_name': 'euclidean'}
        >>> g_params = {'g_method_name': 'knn', 'nn': 10, 'weighted': False}
        >>> emb.graph = ProximityGraph(data, m_params, g_params)
        >>> emb.create_mvu_embedding_()
        >>> emb.coords.shape
        (2, 500)
        """
        # Validate required attributes
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph must be created before MVU embedding. Call create_graph() first.")
        if not hasattr(self, 'dim'):
            raise AttributeError("Embedding dimension 'dim' not set.")
        check_positive(dim=self.dim)
            
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError(
                "cvxpy is required for MVU but not installed. "
                "Install it with: pip install cvxpy or conda install -c conda-forge cvxpy"
            )
            
        mvu = MaximumVarianceUnfolding(
            equation="berkley",
            solver=cp.SCS,
            solver_tol=1e-2,
            eig_tol=1.0e-10,
            solver_iters=2500,
            warm_start=False,
            seed=None,
        )

        self.coords = mvu.fit_transform(
            self.graph.data.T, self.dim, self.graph.nn
        ).T
        self.reducer_ = mvu

    def create_lle_embedding_(self):
        """Create Locally Linear Embedding (LLE).
        
        Non-linear dimensionality reduction that assumes data lies on a
        locally linear manifold. Each point is reconstructed from its
        neighbors, and these weights are preserved in lower dimensions.
        
        Sets self.coords to shape (dim, n_samples) embedding coordinates
        and self.reducer_ to the fitted sklearn LLE object.
        
        Raises
        ------
        AttributeError
            If self.graph or required attributes are not set.
        ValueError
            If self.dim is not positive or n_neighbors is invalid.
            
        Notes
        -----
        Uses sklearn's LocallyLinearEmbedding implementation. The algorithm:
        1. Finds k nearest neighbors for each point
        2. Computes weights to reconstruct each point from neighbors
        3. Finds low-d embedding preserving reconstruction weights
        
        Time complexity is O(DNlog(k)N + DNk³) for N points in D dimensions.
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> from driada.dim_reduction.graph import ProximityGraph
        >>> data = np.random.randn(10, 500)  # 10 features, 500 samples
        >>> labels = np.random.randint(0, 3, 500)
        >>> params = {'e_method': METHODS_DICT['lle'], 'e_method_name': 'lle', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)
        >>> m_params = {'metric_name': 'euclidean'}
        >>> g_params = {'g_method_name': 'knn', 'nn': 10, 'weighted': False}
        >>> emb.graph = ProximityGraph(data, m_params, g_params)
        >>> emb.create_lle_embedding_()
        >>> emb.coords.shape
        (2, 500)
        
        References
        ----------
        Roweis & Saul (2000). Nonlinear dimensionality reduction by 
        locally linear embedding. Science 290:2323-2326.        """
        # Validate required attributes
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph must be created before LLE embedding. Call create_graph() first.")
        if not hasattr(self, 'dim'):
            raise AttributeError("Embedding dimension 'dim' not set.")
        check_positive(dim=self.dim)
        check_positive(n_neighbors=self.graph.nn)
        
        lle = LocallyLinearEmbedding(n_components=self.dim, n_neighbors=self.graph.nn)
        self.coords = lle.fit_transform(self.graph.data.T).T
        self.reducer_ = lle

    def create_hlle_embedding_(self):
        """Create Hessian Locally Linear Embedding (HLLE).
        
        Modified version of LLE that uses Hessian-based regularization to
        better preserve local geometric structure. Particularly effective
        for manifolds with varying curvature.
        
        Sets self.coords to shape (dim, n_samples) embedding coordinates
        and self.reducer_ to the fitted sklearn HLLE object.
        
        Raises
        ------
        AttributeError
            If self.graph or required attributes are not set.
        ValueError
            If self.dim is not positive or if the constraint
            n_neighbors > n_components * (n_components + 3) / 2 is not satisfied.
            
        Notes
        -----
        Requires n_neighbors > n_components * (n_components + 3) / 2 for the
        Hessian estimation. More computationally intensive than standard LLE
        but often produces better embeddings for complex manifolds.
        
        The algorithm estimates the Hessian at each point to capture local
        curvature, then finds an embedding that preserves this structure.
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> from driada.dim_reduction.graph import ProximityGraph
        >>> data = np.random.randn(10, 500)  # 10 features, 500 samples
        >>> labels = np.random.randint(0, 3, 500)
        >>> params = {'e_method': METHODS_DICT['hlle'], 'e_method_name': 'hlle', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)
        >>> # For 2D embedding, need at least 6 neighbors
        >>> m_params = {'metric_name': 'euclidean'}
        >>> g_params = {'g_method_name': 'knn', 'nn': 10, 'weighted': False}
        >>> emb.graph = ProximityGraph(data, m_params, g_params)
        >>> emb.create_hlle_embedding_()  # doctest: +SKIP
        
        References
        ----------
        Donoho, D. & Grimes, C. (2003). Hessian eigenmaps: Locally linear
        embedding techniques for high-dimensional data. PNAS.        """
        # Validate required attributes
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph must be created before HLLE embedding. Call create_graph() first.")
        if not hasattr(self, 'dim'):
            raise AttributeError("Embedding dimension 'dim' not set.")
        check_positive(dim=self.dim)
        check_positive(n_neighbors=self.graph.nn)
        
        # Check HLLE-specific constraint
        min_neighbors = int(self.dim * (self.dim + 3) / 2) + 1
        if self.graph.nn <= min_neighbors:
            raise ValueError(
                f"HLLE requires n_neighbors > {min_neighbors} for {self.dim}D embedding, "
                f"but got n_neighbors={self.graph.nn}. Increase nn parameter."
            )
        
        hlle = LocallyLinearEmbedding(
            n_components=self.dim, n_neighbors=self.graph.nn, method="hessian"
        )
        self.coords = hlle.fit_transform(self.graph.data.T).T
        self.reducer_ = hlle

    def create_le_embedding_(self):
        """Create Laplacian Eigenmaps (LE) embedding.
        
        Spectral embedding method that uses eigenvectors of the graph Laplacian
        to embed nodes while preserving local neighborhood structure.
        Particularly effective for data lying on low-dimensional manifolds.
        
        Sets self.coords to shape (dim, n_samples) containing the embedding.
        
        Raises
        ------
        AttributeError
            If self.graph or required attributes are not set.
        ValueError
            If self.dim is not positive or if the graph is disconnected
            (multiple eigenvalues equal to 1).
            
        Notes
        -----
        Uses the transition matrix eigenvectors (more stable than Laplacian).
        Normalizes eigenvectors by node degree to ensure proper embedding.
        
        The embedding minimizes: sum_ij W_ij ||y_i - y_j||^2
        subject to orthogonality constraints.
        
        The algorithm:
        1. Computes transition matrix P = D^(-1)A
        2. Finds top eigenvectors of P (excluding trivial)
        3. Normalizes by degree for proper embedding
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> from driada.dim_reduction.graph import ProximityGraph
        >>> data = np.random.randn(10, 500)  # 10 features, 500 samples
        >>> labels = np.random.randint(0, 3, 500)
        >>> params = {'e_method': METHODS_DICT['le'], 'e_method_name': 'le', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)
        >>> m_params = {'metric_name': 'euclidean'}
        >>> g_params = {'g_method_name': 'knn', 'nn': 10, 'weighted': False}
        >>> emb.graph = ProximityGraph(data, m_params, g_params)
        >>> emb.create_le_embedding_()
        >>> emb.coords.shape[0]  # Number of dimensions
        2
            
        References
        ----------
        Belkin, M. & Niyogi, P. (2003). Laplacian eigenmaps for
        dimensionality reduction and data representation. Neural Computation.        """
        # Validate required attributes
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph must be created before LE embedding. Call create_graph() first.")
        if not hasattr(self, 'dim'):
            raise AttributeError("Embedding dimension 'dim' not set.")
        check_positive(dim=self.dim)
        
        A = self.graph.adj
        dim = self.dim
        n = self.graph.n

        DH = get_inv_sqrt_diag_matrix(A)
        P = self.graph.get_matrix("trans")

        start_v = np.ones(n)
        # LR mode is much more stable, this is why we use P matrix largest eigenvalues
        eigvals, eigvecs = eigs(P, k=dim + 1, which="LR", v0=start_v, maxiter=n * 1000)
        # eigvals, vecs = eigs(nL, k = dim2 + 1, which = 'SM')

        eigvals = np.asarray([np.round(np.real(x), 6) for x in eigvals])

        # Check for disconnected graph (multiple eigenvalues close to 1)
        if np.sum(np.abs(eigvals - 1.0) < 1e-10) > 1:
            raise ValueError(
                "Graph appears to be disconnected (multiple eigenvalues ≈ 1). "
                "Laplacian Eigenmaps requires a connected graph. "
                "Consider increasing nn parameter or using a different method."
            )
        else:
            vecs = np.real(eigvecs.T[1:])
            vec_norms = np.array([np.real(sum([x * x for x in v])) for v in vecs])
            vecs = vecs / vec_norms[:, np.newaxis]
            vecs = vecs.dot(DH.toarray())
            self.coords = vecs

    def create_auto_le_embedding_(self):
        """Create Laplacian Eigenmaps embedding using sklearn's implementation.
        
        Alternative implementation using sklearn's spectral_embedding function.
        More robust and handles edge cases better than the manual implementation.
        
        Sets self.coords to shape (dim, n_samples).
        
        Raises
        ------
        AttributeError
            If self.graph or required attributes are not set.
        ValueError
            If self.dim is not positive.
        MemoryError
            If the graph is too large to convert to dense format.
            
        Notes
        -----
        Uses normalized Laplacian by default for better numerical stability.
        Automatically handles disconnected graphs by dropping the first
        eigenvector.
        
        Warning: Converts sparse adjacency matrix to dense format, which may
        cause memory issues for graphs with more than ~10,000 nodes.
        
        This is the recommended method for most use cases unless you need
        fine control over the eigendecomposition process.
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.dr_base import METHODS_DICT
        >>> from driada.dim_reduction.graph import ProximityGraph
        >>> data = np.random.randn(10, 500)  # 10 features, 500 samples
        >>> labels = np.random.randint(0, 3, 500)
        >>> params = {'e_method': METHODS_DICT['auto_le'], 'e_method_name': 'auto_le', 'dim': 2}
        >>> emb = Embedding(data, None, labels, params)
        >>> m_params = {'metric_name': 'euclidean'}
        >>> g_params = {'g_method_name': 'knn', 'nn': 10, 'weighted': False}
        >>> emb.graph = ProximityGraph(data, m_params, g_params)
        >>> emb.create_auto_le_embedding_()
        >>> emb.coords.shape[0]  # Number of dimensions
        2
        """
        # Validate required attributes
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph must be created before auto LE embedding. Call create_graph() first.")
        if not hasattr(self, 'dim'):
            raise AttributeError("Embedding dimension 'dim' not set.")
        check_positive(dim=self.dim)
        
        A = self.graph.adj
        dim = self.dim

        # Warn about memory usage for large graphs
        if A.shape[0] > 10000:
            import warnings
            warnings.warn(
                f"Converting sparse matrix with {A.shape[0]} nodes to dense format. "
                "This may use significant memory. Consider using create_le_embedding_() instead.",
                MemoryWarning
            )

        A = A.asfptype()
        # Convert to numpy array instead of matrix to avoid sklearn compatibility issues
        vecs = spectral_embedding(
            np.asarray(A.todense()),
            n_components=dim,
            eigen_solver=None,
            random_state=None,
            eigen_tol=0.0,
            norm_laplacian=True,
            drop_first=True,
        ).T

        self.coords = vecs

    def create_dmaps_embedding_(self):
        """Create diffusion maps embedding.
        
        Implements the standard diffusion maps algorithm with alpha normalization
        for anisotropic diffusion and diffusion time parameter t.
        
        Raises
        ------
        AttributeError
            If graph is not built or required attributes are missing.
        ValueError
            If dim is invalid, graph has isolated nodes, or eigendecomposition fails.
        
        Notes
        -----
        The algorithm performs the following steps:
        1. Apply alpha normalization to adjacency matrix
        2. Create Markov transition matrix
        3. Compute eigendecomposition
        4. Scale eigenvectors by eigenvalues^t
        
        Future enhancement: Variable bandwidth diffusion maps
        - Berry & Harlim (2016): "Variable bandwidth diffusion kernels"
        - DOI: https://doi.org/10.1016/j.acha.2015.01.001
        - Would allow adaptive kernel bandwidth based on local density
        
        References
        ----------
        - Coifman & Lafon (2006): Diffusion maps
        - DOI: https://doi.org/10.1016/j.acha.2006.04.006        """
        import numpy as np
        from scipy.sparse import csr_matrix, diags
        from scipy.sparse.linalg import eigsh
        
        # Validate graph exists
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph not built. Call build_graph() first.")
        
        # Get and validate parameters
        alpha = self.dm_alpha if hasattr(self, "dm_alpha") else 0.5
        t = self.dm_t if hasattr(self, "dm_t") else 1  # Diffusion time parameter
        check_positive(dim=self.dim, t=t)
        check_nonnegative(alpha=alpha)
        
        # Check dimension validity
        n_samples = self.graph.adj.shape[0]
        if self.dim >= n_samples:
            raise ValueError(f"dim ({self.dim}) must be less than n_samples ({n_samples})")
        
        # Get affinity matrix from graph
        W = self.graph.adj.astype(float)
        
        # Ensure the matrix is symmetric
        W = (W + W.T) / 2
        
        # Apply alpha normalization (anisotropic diffusion)
        # First compute the degree matrix
        D = np.asarray(W.sum(axis=1)).flatten()
        
        # Check for isolated nodes
        if np.any(D == 0):
            raise ValueError("Graph contains isolated nodes with zero degree")
        
        D_alpha = D ** alpha
        
        # Normalize by D^alpha from both sides
        D_alpha_inv = 1.0 / D_alpha
        W_alpha = diags(D_alpha_inv) @ W @ diags(D_alpha_inv)
        
        # Compute new degree matrix for normalized kernel
        D_alpha_norm = np.asarray(W_alpha.sum(axis=1)).flatten()
        
        # Check for numerical issues
        if np.any(D_alpha_norm == 0):
            raise ValueError("Alpha normalization resulted in zero row sums")
            
        D_alpha_norm_inv = 1.0 / D_alpha_norm
        
        # Create Markov transition matrix
        P = diags(D_alpha_norm_inv) @ W_alpha
        
        # Compute eigendecomposition
        # We need the largest eigenvalues (close to 1)
        try:
            # For sparse matrices, use eigsh
            if hasattr(P, 'toarray'):
                eigenvalues, eigenvectors = eigsh(P, k=self.dim+1, which='LM', tol=1e-6)
            else:
                # For dense matrices, convert to sparse first
                P_sparse = csr_matrix(P)
                eigenvalues, eigenvectors = eigsh(P_sparse, k=self.dim+1, which='LM', tol=1e-6)
        except Exception:
            # Fallback to dense computation
            P_dense = P.toarray() if hasattr(P, 'toarray') else P
            eigenvalues_all, eigenvectors_all = np.linalg.eig(P_dense)
            # Sort by magnitude
            idx = np.abs(eigenvalues_all).argsort()[::-1]
            eigenvalues = eigenvalues_all[idx[:self.dim+1]]
            eigenvectors = eigenvectors_all[:, idx[:self.dim+1]]
        
        # Sort eigenvalues/vectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Remove the first eigenvector (constant, eigenvalue=1)
        eigenvalues = eigenvalues[1:self.dim+1]
        eigenvectors = eigenvectors[:, 1:self.dim+1]
        
        # Apply diffusion time scaling: lambda^t
        eigenvalues_t = eigenvalues ** t
        
        # Scale eigenvectors by eigenvalues^t
        # For very small t or when eigenvalues are close to 1, this preserves more variance
        self.coords = (eigenvectors * eigenvalues_t).T
        
        # Store additional info
        self.reducer_ = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'alpha': alpha,
            't': t
        }

    def create_auto_dmaps_embedding_(self):
        """Create diffusion maps embedding using pydiffmap library.
        
        Alternative implementation using the pydiffmap library which provides
        automatic bandwidth selection via the Berry-Harlim-Gao (BGH) method.
        More sophisticated than the manual implementation.
        
        Raises
        ------
        AttributeError
            If graph is not built.
        ValueError
            If parameters are invalid or pydiffmap fails.
        ImportError
            If pydiffmap is not installed.
        
        Notes
        -----
        Sets self.coords to shape (dim, n_samples).
        Uses epsilon='bgh' for automatic bandwidth selection based on
        local geometry. The alpha parameter controls the degree of
        density normalization (alpha=0: no normalization, alpha=1: full
        Fokker-Planck normalization).
        
        This method is preferred when you want automatic parameter tuning
        and don't need fine control over the diffusion process.        """
        # Validate graph exists
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph not built. Call build_graph() first.")
            
        # Validate parameters
        check_positive(dim=self.dim, nn=self.graph.nn)
        alpha = self.dm_alpha if hasattr(self, "dm_alpha") else 1
        check_nonnegative(alpha=alpha)
        
        dim = self.dim
        nn = self.graph.nn
        metric = self.graph.metric
        metric_args = self.graph.metric_args

        try:
            from pydiffmap import diffusion_map as dm
        except ImportError:
            raise ImportError("pydiffmap not installed. Install with: pip install pydiffmap")
            
        try:
            mydmap = dm.DiffusionMap.from_sklearn(
                n_evecs=dim,
                k=nn,
                epsilon="bgh",
                metric=metric,
                metric_params=metric_args,
                alpha=alpha,
            )

            dmap = mydmap.fit_transform(self.init_data.T)

            self.coords = dmap.T
            self.reducer_ = dmap
        except Exception as e:
            raise ValueError(f"Diffusion maps computation failed: {str(e)}")

    def create_tsne_embedding_(self):
        """Create t-SNE (t-distributed Stochastic Neighbor Embedding).
        
        Non-linear dimensionality reduction that converts similarities between
        data points to joint probabilities and minimizes KL divergence between
        high-dimensional and low-dimensional distributions.
        
        Raises
        ------
        ValueError
            If dim is invalid or t-SNE computation fails.
        ImportError
            If scikit-learn is not installed.
        
        Notes
        -----
        Sets self.coords to shape (dim, n_samples).
        Particularly effective for visualization (dim=2 or 3).
        Non-parametric: cannot embed new points without refitting.
        Stochastic: different runs may produce different results.
        
        The perplexity parameter (related to number of neighbors) is
        automatically set by sklearn based on dataset size.
        
        References
        ----------
        van der Maaten, L. & Hinton, G. (2008). Visualizing data using
        t-SNE. Journal of Machine Learning Research.        """
        # Validate parameters
        check_positive(dim=self.dim)
        
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError("scikit-learn not installed")
            
        # Use self.verbose if available, otherwise silent
        verbose = getattr(self, 'verbose', 0)
        
        try:
            model = TSNE(n_components=self.dim, verbose=verbose)
            self.coords = model.fit_transform(self.init_data.T).T
            self.reducer_ = model
        except Exception as e:
            raise ValueError(f"t-SNE computation failed: {str(e)}")

    def create_umap_embedding_(self):
        """Create UMAP (Uniform Manifold Approximation and Projection) embedding.
        
        State-of-the-art manifold learning technique based on Riemannian
        geometry and algebraic topology. Preserves both local and global
        structure better than t-SNE.
        
        Notes
        -----
        Sets self.coords to shape (dim, n_samples).
        The min_dist parameter controls how tightly points are packed
        (smaller values = tighter packing).
        Unlike t-SNE, UMAP can transform new points after fitting.
        
        Advantages over t-SNE:
        - Preserves more global structure
        - Faster for large datasets
        - Supports supervised/semi-supervised modes
        - Can embed new points
        
        References
        ----------
        McInnes, L., Healy, J., & Melville, J. (2018).
        UMAP: Uniform Manifold Approximation and Projection for
        Dimension Reduction. arXiv:1802.03426.        """
        # Validate graph and parameters
        if not hasattr(self, 'graph') or self.graph is None:
            raise AttributeError("Graph not built. Call build_graph() first.")
        if not hasattr(self, 'min_dist'):
            raise AttributeError("min_dist attribute not set")
            
        check_positive(dim=self.dim, nn=self.graph.nn)
        check_nonnegative(min_dist=self.min_dist)
        
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn not installed. Install with: pip install umap-learn")

        try:
            reducer = umap.UMAP(
                n_neighbors=self.graph.nn, 
                n_components=self.dim, 
                min_dist=self.min_dist
            )
            # Use init_data, not graph.data
            self.coords = reducer.fit_transform(self.init_data.T).T
            self.reducer_ = reducer
        except Exception as e:
            raise ValueError(f"UMAP computation failed: {str(e)}")

    def _prepare_data_loaders(self, batch_size, train_size, seed):
        """Prepare train and test data loaders for neural network methods.
        
        Parameters
        ----------
        batch_size : int
            Batch size for data loaders. Must be positive.
        train_size : float
            Proportion of data to use for training. Must be in range (0, 1).
        seed : int
            Random seed for reproducible splits. Must be non-negative.
            
        Returns
        -------
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader  
            Test data loader
        device : torch.device
            Device to use for training (cuda if available, else cpu)
            
        Raises
        ------
        ValueError
            If parameters are invalid.
        ImportError
            If PyTorch is not installed.
            
        Notes
        -----
        Uses sklearn's train_test_split with shuffle=False to preserve
        temporal order in the data split.        """
        # Validate parameters
        check_positive(batch_size=batch_size)
        check_nonnegative(seed=seed)
        if not 0 < train_size < 1:
            raise ValueError(f"train_size must be in range (0, 1), got {train_size}")
            
        try:
            import torch
            from torch.utils.data import DataLoader
            from sklearn.model_selection import train_test_split
        except ImportError as e:
            if 'torch' in str(e):
                raise ImportError(
                    "PyTorch is required for autoencoder methods. "
                    "Please install it with: pip install torch"
                )
            else:
                raise ImportError("scikit-learn is required for data splitting")
            
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Split data into train and test sets
        data_T = self.init_data.T  # Transpose to (n_samples, n_features)
        indices = np.arange(data_T.shape[0])
        
        # Use sklearn's train_test_split without shuffling to preserve temporal order
        train_indices, test_indices = train_test_split(
            indices, 
            train_size=train_size, 
            random_state=seed,
            shuffle=False
        )
        
        # Create datasets with the split indices
        train_dataset = NeuroDataset(self.init_data[:, train_indices])
        test_dataset = NeuroDataset(self.init_data[:, test_indices])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return train_loader, test_loader, device

    def create_ae_embedding_(
        self,
        continue_learning=0,
        epochs=50,
        lr=1e-3,
        seed=42,
        batch_size=32,
        enc_kwargs=None,
        dec_kwargs=None,
        feature_dropout=0.2,
        train_size=0.8,
        inter_dim=100,
        verbose=True,
        add_corr_loss=False,
        corr_hyperweight=0,
        add_mi_loss=False,
        mi_hyperweight=0,
        minimize_mi_data=None,
        log_every=1,
        device=None,
    ):
        """Create autoencoder embedding.
        
        .. deprecated:: 
            This method is deprecated. Use `create_flexible_ae_embedding_` instead 
            for more flexibility and advanced loss functions.
            
            To recreate the same functionality with the new method:
            
            # Basic AE (no additional losses)
            create_flexible_ae_embedding_(architecture="ae", ...)
            
            # AE with correlation loss
            create_flexible_ae_embedding_(
                architecture="ae",
                loss_components=[
                    {"name": "reconstruction", "weight": 1.0},
                    {"name": "correlation", "weight": corr_hyperweight}
                ],
                ...
            )
            
            # AE with MI/orthogonality loss
            create_flexible_ae_embedding_(
                architecture="ae", 
                loss_components=[
                    {"name": "reconstruction", "weight": 1.0},
                    {"name": "orthogonality", "weight": mi_hyperweight, 
                     "external_data": minimize_mi_data}
                ],
                ...
            )
        
        Parameters
        ----------
        continue_learning : int, default=0
            Whether to continue training existing model.
        epochs : int, default=50
            Number of training epochs.
        lr : float, default=1e-3
            Learning rate.
        seed : int, default=42
            Random seed.
        batch_size : int, default=32
            Batch size for training.
        enc_kwargs : dict, optional
            Encoder configuration.
        dec_kwargs : dict, optional  
            Decoder configuration.
        feature_dropout : float, default=0.2
            Feature dropout rate.
        train_size : float, default=0.8
            Training set fraction.
        inter_dim : int, default=100
            Hidden layer dimension.
        verbose : bool, default=True
            Print training progress.
        add_corr_loss : bool, default=False
            Add correlation loss to encourage decorrelated latent features.
        corr_hyperweight : float, default=0
            Weight for correlation loss.
        add_mi_loss : bool, default=False
            Add MI-based orthogonality loss.
        mi_hyperweight : float, default=0
            Weight for MI loss.
        minimize_mi_data : np.ndarray, optional
            External data to minimize correlation with (for MI loss).
        log_every : int, default=1
            Logging frequency.
        device : torch.device, optional
            Device to run on.
            
        Raises
        ------
        ValueError
            If parameters are invalid.
        ImportError
            If PyTorch is not installed.
            
        Notes
        -----
        This method is deprecated in favor of create_flexible_ae_embedding_
        which provides more flexibility and advanced loss functions.        """

        # ---------------------------------------------------------------------------
        # Validate parameters
        check_positive(epochs=epochs, lr=lr, inter_dim=inter_dim, log_every=log_every)
        check_nonnegative(corr_hyperweight=corr_hyperweight, mi_hyperweight=mi_hyperweight)
        
        # Import torch dependencies (optional dependency)
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError(
                "PyTorch is required for autoencoder methods. "
                "Please install it with: pip install torch"
            )

        # Get data loaders and device
        train_loader, test_loader, device_to_use = self._prepare_data_loaders(
            batch_size=batch_size,
            train_size=train_size,
            seed=seed
        )
        
        if device is None:
            device = device_to_use
        
        # Set random seed again for model initialization
        torch.manual_seed(seed)
        if verbose:
            print("device:", device)

        if not continue_learning:
            # create a model from `AE` autoencoder class
            # load it to the specified device, either gpu or cpu
            # Ensure kwargs are dictionaries, not None
            if enc_kwargs is None:
                enc_kwargs = {}
            if dec_kwargs is None:
                dec_kwargs = {}
            model = AE(
                orig_dim=self.init_data.shape[0],
                inter_dim=inter_dim,
                code_dim=self.dim,
                enc_kwargs=enc_kwargs,
                dec_kwargs=dec_kwargs,
                device=device,
            )

            model = model.to(device)
        else:
            model = self.nnmodel

        # create an optimizer object
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # mean-squared error loss
        criterion = nn.MSELoss()

        def correlation_loss(data):
            """Compute correlation loss to encourage decorrelated features.
            
            Parameters
            ----------
            data : torch.Tensor
                Latent representations of shape (n_features, n_samples).
                
            Returns
            -------
            torch.Tensor
                Average pairwise correlation magnitude as scalar loss.
                
            Notes
            -----
            Computes average absolute correlation between all pairs of features,
            excluding self-correlations (diagonal).            """
            corr = torch.corrcoef(data)
            nv = corr.shape[0]
            closs = torch.abs(
                (torch.sum(torch.abs(corr)) - 1 * nv) / (nv**2 - nv)
            )  # average pairwise correlation amplitude
            return closs

        def data_orthogonality_loss(data, ortdata):
            """Compute orthogonality loss between embeddings and external data.
            
            Encourages embeddings to be uncorrelated with external data by
            penalizing correlation coefficients between all variable pairs.
            
            Parameters
            ----------
            data : torch.Tensor
                Embedding latent representations of shape (n_features, n_samples).
            ortdata : torch.Tensor
                External data to minimize correlation with, shape (n_features2, n_samples).
                
            Returns
            -------
            torch.Tensor
                Average absolute correlation between data and ortdata as scalar loss.
                
            Notes
            -----
            This is a workaround for mutual information estimation. The loss
            computes average magnitude of correlations between all pairs of
            variables from data and ortdata.            """
            n1, n2 = data.shape[0], ortdata.shape[1]
            fulldata = torch.cat((data, ortdata), dim=0)
            corr = torch.corrcoef(fulldata)
            nvar = n1 * n2
            closs = torch.abs(
                (torch.sum(torch.abs(corr))) / nvar
            )  # average pairwise correlation amplitude
            return closs

        # ---------------------------------------------------------------------------
        f_dropout = nn.Dropout(feature_dropout)

        best_test_epoch = -1
        best_test_loss = 1e10
        best_test_model = None

        for epoch in range(epochs):
            loss = 0
            for batch_features, indices in train_loader:
                batch_features = batch_features.to(device)
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
                noisy_batch_features = (
                    f_dropout(torch.ones(batch_features.shape).to(device))
                    * batch_features
                )
                outputs = model(noisy_batch_features.float())
                code = model.encoder(noisy_batch_features.float()).T

                """
                # ==================== MINE experiment ========================

                from torch_mist.estimators import mine
                from torch_mist.utils.train import train_mi_estimator
                from torch_mist.utils import evaluate_mi

                minimize_mi_data = torch.tensor(minimize_mi_data[:, _]).float().to(device)
                print(minimize_mi_data.shape, code.shape)

                estimator = mine(
                    x_dim=minimize_mi_data.shape[0],
                    y_dim=code.shape[0],
                    hidden_dims=[64, 32],
                )


                # Train it on the given samples
                train_log = train_mi_estimator(
                    estimator=estimator,
                    train_data=(minimize_mi_data.T, code.T),
                    batch_size=batch_size,
                    max_iterations=1000,
                    device=device,
                    fast_train=True,
                    max_epochs=10
                )

                # Evaluate the estimator on the entirety of the data
                estimated_mi = evaluate_mi(
                    estimator=estimator,
                    data=(minimize_mi_data.T.to(device), code.T.to(device)),
                    batch_size=batch_size
                )

                print(f"Mutual information estimated value: {estimated_mi} nats")
                # ==================== MINE experiment ========================
                """
                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features.float())

                if add_mi_loss:
                    ortdata = (
                        torch.tensor(minimize_mi_data[:, indices]).float().to(device)
                    )
                    train_loss += mi_hyperweight * data_orthogonality_loss(
                        code, ortdata
                    )

                if add_corr_loss:
                    train_loss += corr_hyperweight * correlation_loss(code)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            # compute the epoch training loss
            loss = loss / len(train_loader)

            # display the epoch training loss
            if (epoch + 1) % log_every == 0:

                # compute loss on test part
                tloss = 0
                for batch_features, indices in test_loader:
                    batch_features = batch_features.to(device)
                    # compute reconstructions
                    noisy_batch_features = (
                        f_dropout(torch.ones(batch_features.shape).to(device))
                        * batch_features
                    )
                    outputs = model(noisy_batch_features.float())

                    # compute test reconstruction loss
                    test_loss = criterion(outputs, batch_features.float())

                    if add_mi_loss:
                        ortdata = (
                            torch.tensor(minimize_mi_data[:, indices])
                            .float()
                            .to(device)
                        )
                        train_loss += mi_hyperweight * data_orthogonality_loss(
                            code, ortdata
                        )

                    if add_corr_loss:
                        code = model.encoder(noisy_batch_features.float()).T
                        train_loss += corr_hyperweight * correlation_loss(code)

                    tloss += test_loss.item()

                # compute the epoch test loss
                tloss = tloss / len(test_loader)
                if tloss < best_test_loss:
                    best_test_loss = tloss
                    best_test_epoch = epoch + 1
                    best_test_model = model
                if verbose:
                    print(
                        f"epoch : {epoch + 1}/{epochs}, train loss = {loss:.8f}, test loss = {tloss:.8f}"
                    )

        if verbose:
            if best_test_epoch != epochs + 1:
                print(f"best model: epoch {best_test_epoch}")

        self.nnmodel = best_test_model
        input_ = torch.tensor(self.init_data.T).float().to(device)
        self.coords = model.get_code_embedding(input_)
        self.nn_loss = tloss

    # -------------------------------------

    def create_vae_embedding_(
        self,
        continue_learning=0,
        epochs=50,
        lr=1e-3,
        seed=42,
        batch_size=32,
        enc_kwargs=None,
        dec_kwargs=None,
        feature_dropout=0.2,
        kld_weight=1,
        train_size=0.8,
        inter_dim=128,
        verbose=True,
        log_every=10,
        **kwargs,
    ):
        """Create variational autoencoder embedding.
        
        .. deprecated::
            This method is deprecated. Use `create_flexible_ae_embedding_` instead
            for more flexibility and advanced loss functions.
            
            To recreate the same functionality with the new method:
            
            # Standard VAE
            create_flexible_ae_embedding_(
                architecture="vae",
                loss_components=[
                    {"name": "reconstruction", "weight": 1.0},
                    {"name": "beta_vae", "weight": 1.0, "beta": kld_weight}
                ],
                ...
            )
            
            # For advanced disentanglement methods, use:
            # - TC-VAE: {"name": "tc_vae", "alpha": 1.0, "beta": 5.0, "gamma": 1.0}
            # - Factor-VAE: {"name": "factor_vae", "gamma": 10.0}
        
        Parameters
        ----------
        continue_learning : int, default=0
            Whether to continue training existing model.
        epochs : int, default=50
            Number of training epochs.
        lr : float, default=1e-3
            Learning rate.
        seed : int, default=42
            Random seed.
        batch_size : int, default=32
            Batch size for training.
        enc_kwargs : dict, optional
            Encoder configuration.
        dec_kwargs : dict, optional
            Decoder configuration.
        feature_dropout : float, default=0.2
            Feature dropout rate.
        kld_weight : float, default=1
            Weight for KL divergence loss term.
        train_size : float, default=0.8
            Training set fraction.
        inter_dim : int, default=128
            Hidden layer dimension.
        verbose : bool, default=True
            Print training progress.
        log_every : int, default=10
            Logging frequency.
        **kwargs
            Additional keyword arguments.
            
        Raises
        ------
        ValueError
            If parameters are invalid.
        ImportError
            If PyTorch is not installed.
            
        Notes
        -----
        This method is deprecated in favor of create_flexible_ae_embedding_
        which provides more flexibility and advanced loss functions.        """

        # ---------------------------------------------------------------------------
        # Validate parameters
        check_positive(epochs=epochs, lr=lr, inter_dim=inter_dim, log_every=log_every)
        check_nonnegative(kld_weight=kld_weight)
        
        # Import torch dependencies (optional dependency)
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError(
                "PyTorch is required for autoencoder methods. "
                "Please install it with: pip install torch"
            )
        
        # Get data loaders and device
        train_loader, test_loader, device = self._prepare_data_loaders(
            batch_size=batch_size,
            train_size=train_size,
            seed=seed
        )
        
        # Set random seed again for model initialization
        torch.manual_seed(seed)

        if not continue_learning:
            # create a model from `VAE` autoencoder class
            # load it to the specified device, either gpu or cpu
            model = VAE(
                orig_dim=len(self.init_data),
                inter_dim=inter_dim,
                code_dim=self.dim,
                enc_kwargs=enc_kwargs,
                dec_kwargs=dec_kwargs,
                device=device,
            )
            model = model.to(device)
        else:
            model = self.nnmodel

        # create an optimizer object
        # Adam optimizer with learning rate lr
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # BCE error loss
        # criterion = nn.BCELoss(reduction='sum')
        criterion = nn.MSELoss()

        # ---------------------------------------------------------------------------
        f_dropout = nn.Dropout(feature_dropout)
        
        best_test_epoch = -1
        best_test_loss = 1e10
        best_test_model = None

        for epoch in range(epochs):
            loss = 0
            loss1 = 0
            loss2 = 0
            for batch_features, indices in train_loader:  # NeuroDataset returns (sample, idx)
                batch_features = batch_features.to(device)
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
                data = (
                    f_dropout(torch.ones(batch_features.shape).to(device))
                    * batch_features
                )
                data = data.to(device).float()  # Ensure float32
                reconstruction, mu, logvar = model(data)

                # compute training reconstruction loss
                mse_loss = criterion(reconstruction, data)
                kld_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )  # * train_dataset.__len__()/batch_size
                train_loss = mse_loss + kld_weight * kld_loss

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
                loss1 += mse_loss.item()
                loss2 += kld_loss.item()

            # compute the epoch training loss
            loss = loss / len(train_loader)
            loss1 = loss1 / len(train_loader)
            loss2 = loss2 / len(train_loader)

            # display the epoch training loss
            # display the epoch training loss
            if (epoch + 1) % log_every == 0:
                # compute loss on test part
                tloss = 0
                for batch_features, indices in test_loader:  # NeuroDataset returns (sample, idx)
                    data = (
                        f_dropout(torch.ones(batch_features.shape).to(device))
                        * batch_features
                    )
                    data = data.to(device).float()  # Ensure float32
                    reconstruction, mu, logvar = model(data)

                    # compute training reconstruction loss
                    mse_loss = criterion(reconstruction, data)
                    kld_loss = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    )  # * train_dataset.__len__()/batch_size
                    test_loss = mse_loss + kld_weight * kld_loss
                    tloss += test_loss.item()

                # compute the epoch test loss
                tloss = tloss / len(test_loader)
                
                # Track best model
                if tloss < best_test_loss:
                    best_test_loss = tloss
                    best_test_epoch = epoch + 1
                    # Deep copy the model state
                    best_test_model = model.state_dict().copy()
                    
                if verbose:
                    print(
                        f"epoch : {epoch + 1}/{epochs}, train loss = {loss:.8f}, test loss = {tloss:.8f}"
                    )
        
        if verbose:
            if best_test_epoch != epochs:
                print(f"best model: epoch {best_test_epoch}")
        
        # Load best model weights
        if best_test_model is not None:
            model.load_state_dict(best_test_model)
            
        self.nnmodel = model
        input_ = torch.tensor(self.init_data.T).float().to(device)
        self.coords = model.get_code_embedding(input_)

        # -------------------------------------

    def create_flexible_ae_embedding_(
        self,
        architecture="ae",  # "ae" or "vae"
        continue_learning=0,
        epochs=50,
        lr=1e-3,
        seed=42,
        batch_size=32,
        enc_kwargs=None,
        dec_kwargs=None,
        feature_dropout=0.2,
        train_size=0.8,
        inter_dim=100,
        verbose=True,
        loss_components=None,
        log_every=1,
        device=None,
        logger=None,
    ):
        """Create flexible autoencoder embedding with modular loss composition.
        
        Parameters
        ----------
        architecture : str, default="ae"
            Architecture type: "ae" for standard autoencoder, "vae" for variational.
        continue_learning : int, default=0
            Whether to continue training existing model.
        epochs : int, default=50
            Number of training epochs.
        lr : float, default=1e-3
            Learning rate.
        seed : int, default=42
            Random seed for reproducibility.
        batch_size : int, default=32
            Batch size for training.
        enc_kwargs : dict, optional
            Encoder configuration (e.g., dropout).
        dec_kwargs : dict, optional
            Decoder configuration.
        feature_dropout : float, default=0.2
            Dropout rate for input features during training.
        train_size : float, default=0.8
            Fraction of data for training.
        inter_dim : int, default=100
            Hidden layer dimension.
        verbose : bool, default=True
            Whether to print training progress.
        loss_components : list of dict, optional
            Loss component configurations. Each dict should contain:
            - "name": str, the loss type
            - "weight": float, the loss weight
            - Additional parameters specific to each loss type
            If None, uses standard reconstruction loss for AE or
            reconstruction + KLD for VAE.
        log_every : int, default=1
            Log frequency (epochs).
        device : torch.device, optional
            Device to run on.
        logger : logging.Logger, optional
            Logger instance.
            
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.data import MVData
        >>> data = np.random.randn(50, 500)  # 50 features, 500 samples
        >>> mvdata = MVData(data)
        >>> 
        >>> # Standard autoencoder with correlation loss
        >>> emb = mvdata.get_embedding(
        ...     method="flexible_ae",
        ...     architecture="ae",
        ...     dim=10,
        ...     loss_components=[
        ...         {"name": "reconstruction", "weight": 1.0},
        ...         {"name": "correlation", "weight": 0.1}
        ...     ],
        ...     epochs=5,  # Quick test
        ...     verbose=False
        ... )
        
        >>> # β-VAE for disentanglement
        >>> emb = mvdata.get_embedding(
        ...     method="flexible_ae",
        ...     architecture="vae",
        ...     dim=10,
        ...     loss_components=[
        ...         {"name": "reconstruction", "weight": 1.0},
        ...         {"name": "beta_vae", "weight": 1.0, "beta": 4.0}
        ...     ],
        ...     epochs=5,  # Quick test
        ...     verbose=False
        ... )
        
        >>> # Recreate deprecated 'ae' method with correlation loss
        >>> # Old: method="ae", add_corr_loss=True, corr_hyperweight=0.1
        >>> # New:
        >>> emb = mvdata.get_embedding(
        ...     method="flexible_ae",
        ...     architecture="ae",
        ...     dim=10,
        ...     loss_components=[
        ...         {"name": "reconstruction", "weight": 1.0},
        ...         {"name": "correlation", "weight": 0.1}
        ...     ],
        ...     epochs=5,  # Quick test
        ...     verbose=False
        ... )
        
        >>> # Recreate deprecated 'ae' method with MI loss
        >>> # Old: method="ae", add_mi_loss=True, mi_hyperweight=0.1, minimize_mi_data=data
        >>> # New:
        >>> emb = mvdata.get_embedding(
        ...     method="flexible_ae",
        ...     architecture="ae",
        ...     dim=10,
        ...     loss_components=[
        ...         {"name": "reconstruction", "weight": 1.0},
        ...         {"name": "orthogonality", "weight": 0.1, "external_data": data}
        ...     ],
        ...     epochs=5,  # Quick test
        ...     verbose=False
        ... )
        
        >>> # Recreate deprecated 'vae' method
        >>> # Old: method="vae", kld_weight=0.1
        >>> # New:
        >>> emb = mvdata.get_embedding(
        ...     method="flexible_ae",
        ...     architecture="vae",
        ...     dim=10,
        ...     loss_components=[
        ...         {"name": "reconstruction", "weight": 1.0},
        ...         {"name": "beta_vae", "weight": 1.0, "beta": 0.1}
        ...     ],
        ...     epochs=5,  # Quick test
        ...     verbose=False
        ... )
        
        Raises
        ------
        ValueError
            If parameters are invalid or architecture not in ["ae", "vae"].
        ImportError
            If PyTorch is not installed.
            
        Notes
        -----
        This method provides a flexible framework for various autoencoder
        architectures with modular loss composition. It replaces the
        deprecated create_ae_embedding_ and create_vae_embedding_ methods.        """
        # Validate parameters
        check_positive(epochs=epochs, lr=lr, inter_dim=inter_dim, log_every=log_every)
        if architecture not in ["ae", "vae"]:
            raise ValueError(f"architecture must be 'ae' or 'vae', got '{architecture}'")
            
        # Import torch dependencies
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError(
                "PyTorch is required for autoencoder methods. "
                "Please install it with: pip install torch"
            )
        
        from .flexible_ae import ModularAutoencoder, FlexibleVAE
        
        # Get data loaders and device
        train_loader, test_loader, device_to_use = self._prepare_data_loaders(
            batch_size=batch_size,
            train_size=train_size,
            seed=seed
        )
        
        if device is None:
            device = device_to_use
        
        # Set random seed for model initialization
        torch.manual_seed(seed)
        if verbose:
            print("device:", device)
        
        # Default loss components if none specified
        if loss_components is None:
            loss_components = []
            
            # Always add reconstruction loss
            loss_components.append({
                "name": "reconstruction",
                "weight": 1.0,
                "loss_type": "mse"
            })
            
            # For VAE, add standard KLD loss
            if architecture == "vae":
                loss_components.append({
                    "name": "beta_vae",
                    "weight": 1.0,
                    "beta": 1.0
                })
        
        if not continue_learning:
            # Create model based on architecture
            if architecture == "ae":
                model = ModularAutoencoder(
                    input_dim=self.init_data.shape[0],
                    latent_dim=self.dim,
                    hidden_dim=inter_dim,
                    encoder_config=enc_kwargs,
                    decoder_config=dec_kwargs,
                    loss_components=loss_components,
                    device=device,
                    logger=logger
                )
            elif architecture == "vae":
                model = FlexibleVAE(
                    input_dim=self.init_data.shape[0],
                    latent_dim=self.dim,
                    hidden_dim=inter_dim,
                    encoder_config=enc_kwargs,
                    decoder_config=dec_kwargs,
                    loss_components=loss_components,
                    device=device,
                    logger=logger
                )
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            model = model.to(device)
        else:
            model = self.nnmodel
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Feature dropout
        f_dropout = nn.Dropout(feature_dropout)
        
        best_test_epoch = -1
        best_test_loss = 1e10
        best_test_model = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_features, indices in train_loader:
                batch_features = batch_features.to(device)
                optimizer.zero_grad()
                
                # Apply feature dropout
                noisy_features = f_dropout(torch.ones_like(batch_features)) * batch_features
                noisy_features = noisy_features.float()
                
                # Compute loss
                total_loss, loss_dict = model.compute_loss(
                    noisy_features,
                    indices=indices
                )
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(loss_dict)
            
            # Average training losses
            avg_train_losses = {}
            for key in train_losses[0].keys():
                avg_train_losses[key] = np.mean([d[key] for d in train_losses])
            
            # Validation phase
            if (epoch + 1) % log_every == 0:
                model.eval()
                test_losses = []
                
                with torch.no_grad():
                    for batch_features, indices in test_loader:
                        batch_features = batch_features.to(device)
                        
                        # Apply feature dropout for consistency
                        noisy_features = f_dropout(torch.ones_like(batch_features)) * batch_features
                        noisy_features = noisy_features.float()
                        
                        # Compute loss
                        total_loss, loss_dict = model.compute_loss(
                            noisy_features,
                            indices=indices
                        )
                        
                        test_losses.append(loss_dict)
                
                # Average test losses
                avg_test_losses = {}
                for key in test_losses[0].keys():
                    avg_test_losses[key] = np.mean([d[key] for d in test_losses])
                
                test_loss = avg_test_losses["total_loss"]
                
                # Track best model
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_epoch = epoch + 1
                    best_test_model = model.state_dict().copy()
                
                if verbose:
                    train_loss = avg_train_losses["total_loss"]
                    print(
                        f"epoch : {epoch + 1}/{epochs}, "
                        f"train loss = {train_loss:.8f}, "
                        f"test loss = {test_loss:.8f}"
                    )
                    
                    # Print individual loss components
                    if len(model.losses) > 1:
                        loss_str = ", ".join([
                            f"{k}: {v:.6f}" 
                            for k, v in avg_test_losses.items() 
                            if k != "total_loss" and not k.endswith("_weighted")
                        ])
                        print(f"  Components: {loss_str}")
        
        if verbose and best_test_epoch != epochs:
            print(f"best model: epoch {best_test_epoch}")
        
        # Load best model
        if best_test_model is not None:
            model.load_state_dict(best_test_model)
        
        # Store model and extract embeddings
        self.nnmodel = model
        input_ = torch.tensor(self.init_data.T).float().to(device)
        self.coords = model.get_latent_representation(input_)
        self.nn_loss = best_test_loss

    def continue_learning(self, add_epochs, **kwargs):
        """Continue training an existing autoencoder model.
        
        Allows resuming training of a previously trained autoencoder or VAE
        for additional epochs with potentially different parameters.
        
        Parameters
        ----------
        add_epochs : int
            Number of additional epochs to train. Must be positive.
        **kwargs
            Additional keyword arguments to pass to the training method.
            These override the original training parameters.
            
        Raises
        ------
        ValueError
            If add_epochs is not positive or method is not DL-based.
        AttributeError
            If no model has been trained yet.
            
        Notes
        -----
        This method requires that an autoencoder model was previously
        trained using one of the DL-based methods (ae, vae, flexible_ae).        """
        # Validate parameters
        check_positive(add_epochs=add_epochs)
        
        # Check if this is a DL-based method
        if self.all_params["e_method_name"] not in ["ae", "vae", "flexible_ae"]:
            raise ValueError("continue_learning only works with DL-based methods (ae, vae, flexible_ae)")
            
        # Check if model exists
        if not hasattr(self, 'nnmodel') or self.nnmodel is None:
            raise AttributeError("No model to continue training. Train a model first.")

        # Get the appropriate training method
        fn = getattr(self, "create_" + self.all_params["e_method_name"] + "_embedding_")
        # Continue training with additional epochs
        fn(continue_learning=1, epochs=add_epochs, **kwargs)

    def to_mvdata(self):
        """Convert embedding coordinates to MVData for further processing.

        This allows embeddings to be used as input for additional dimensionality
        reduction or analysis steps, enabling recursive embedding pipelines.

        Label Handling
        --------------
        Graph-based dimensionality reduction methods (LLE, Laplacian Eigenmaps, 
        Isomap, etc.) may remove disconnected nodes during preprocessing, resulting
        in fewer points in the embedding than in the original data. This method
        handles labels in the following way:
        
        1. If all points are preserved: Labels are passed through unchanged.
        2. If nodes were filtered and a node mapping exists: Labels are filtered
           to match only the kept nodes, preserving the correspondence.
        3. If nodes were filtered but no mapping is available: Labels are set to
           None to avoid misalignment between data points and labels.

        Returns
        -------
        MVData
            An MVData object containing the embedding coordinates as data.
            Labels will be:
            - Original labels if all points preserved
            - Filtered labels matching kept nodes if mapping available
            - None if nodes were removed but mapping unavailable

        Raises
        ------
        ValueError
            If embedding has not been built yet.
            
        Examples
        --------
        >>> import numpy as np
        >>> from driada.dim_reduction.data import MVData
        >>> np.random.seed(42)
        >>> high_dim_data = np.random.randn(20, 500)  # 20 features, 500 samples
        >>> labels = np.random.choice(['A', 'B', 'C', 'D'], size=500)
        >>> mvdata = MVData(high_dim_data, labels=labels)
        >>> # PCA preserves all points
        >>> pca_emb = mvdata.get_embedding(method='pca', dim=2, verbose=False)
        >>> pca_mvdata = pca_emb.to_mvdata()
        >>> assert len(pca_mvdata.labels) == 500  # All labels preserved
        
        >>> # LLE might remove disconnected nodes
        >>> lle_emb = mvdata.get_embedding(method='lle', dim=2, nn=2)
        >>> lle_mvdata = lle_emb.to_mvdata()
        >>> # Labels either filtered to match remaining nodes or None
        >>> assert lle_mvdata.labels is None or len(lle_mvdata.labels) == lle_mvdata.n_points
        """
        # Import here to avoid circular dependency
        from .data import MVData

        if self.coords is None:
            raise ValueError("Embedding has not been built yet. Call build() first.")

        # Handle case where graph preprocessing removed nodes
        labels_to_use = self.labels
        
        # Check if the number of points in coords differs from labels
        if self.labels is not None and self.coords.shape[1] != len(self.labels):
            # If we have a graph with node mapping, use it
            if hasattr(self, 'graph') and self.graph is not None:
                if hasattr(self.graph, '_init_to_final_node_mapping') and self.graph._init_to_final_node_mapping:
                    # Filter labels to match the nodes that remain after preprocessing
                    node_mapping = self.graph._init_to_final_node_mapping
                    # Get the original indices that were kept
                    kept_indices = sorted(node_mapping.keys())
                    labels_to_use = self.labels[kept_indices]
                else:
                    # No explicit mapping, but coords has fewer points
                    # This can happen with graph methods that remove disconnected components
                    # In this case, we can't recover which labels to keep, so set to None
                    labels_to_use = None
            else:
                # No graph, but mismatch in sizes - set labels to None
                labels_to_use = None
        
        # Create MVData with embedding coordinates
        # coords shape is (embedding_dim, n_points), which matches MVData format
        return MVData(
            data=self.coords,
            labels=labels_to_use,
            distmat=None,  # Distance matrix would need to be recomputed for embedding space
            rescale_rows=False,  # Embeddings are already scaled appropriately
            data_name=f"{self.e_method_name}_embedding",
        )
