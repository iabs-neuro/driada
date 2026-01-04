from .graph_utils import (
    get_giant_cc_from_graph,
    get_giant_scc_from_graph,
    remove_selfloops_from_graph,
    remove_isolates_and_selfloops_from_graph,
)
from .randomization import (
    random_rewiring_IOM_preserving,
    adj_random_rewiring_iom_preserving,
    random_rewiring_complete_graph,
    random_rewiring_dense_graph,
)
from .spectral import spectral_entropy, free_entropy, q_entropy
from .matrix_utils import (
    get_giant_cc_from_adj,
    get_giant_scc_from_adj,
    remove_selfloops_from_adj,
    remove_isolates_from_adj,
    get_ccs_from_adj,
    get_laplacian,
    get_norm_laplacian,
    get_rw_laplacian,
    get_trans_matrix,
    get_inv_sqrt_diag_matrix,
)

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy import linalg as la
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import logging
from typing import Optional

UNDIR_MATRIX_TYPES = ["adj", "trans", "lap", "nlap", "rwlap"]
DIR_MATRIX_TYPES = ["adj", "lap_out", "lap_in"]
MATRIX_TYPES = UNDIR_MATRIX_TYPES + DIR_MATRIX_TYPES
SUPPORTED_GRAPH_TYPES = [nx.Graph, nx.DiGraph]


def check_matrix_type(mode, is_directed):
    """Validate matrix type for directed/undirected networks.

    Ensures the requested matrix type is compatible with the network's
    directionality.

    Parameters
    ----------
    mode : str
        Matrix type to validate. Options include:
        - 'adj': adjacency matrix
        - 'lap': Laplacian matrix
        - 'nlap': normalized Laplacian
        - 'trans': transition matrix
        - 'rwlap': random walk Laplacian
        - 'lap_out', 'lap_in': directed Laplacians
    is_directed : bool
        Whether the network is directed.

    Raises
    ------
    ValueError
        If mode is not recognized or incompatible with network type.

    Notes
    -----
    Directed networks support: 'adj', 'lap_out', 'lap_in'.
    Undirected networks support: 'adj', 'trans', 'lap', 'nlap', 'rwlap'.    """
    if mode not in MATRIX_TYPES:
        raise ValueError(
            f"Matrix type {mode} is not in allowed matrix types: {MATRIX_TYPES}"
        )

    if is_directed and mode not in DIR_MATRIX_TYPES:
        raise ValueError(
            f"Matrix type {mode} is not allowed for directed networks."
            f"Supported options are: {DIR_MATRIX_TYPES}"
        )

    if not is_directed and mode not in UNDIR_MATRIX_TYPES:
        raise ValueError(
            f"Matrix type {mode} is not allowed for undirected networks."
            f"Supported options are: {UNDIR_MATRIX_TYPES}"
        )


def check_adjacency(a):
    """Validate that matrix is square (valid adjacency matrix).

    Parameters
    ----------
    a : numpy.ndarray or scipy.sparse matrix
        Matrix to validate.

    Raises
    ------
    ValueError
        If matrix is not square.
        
    Notes
    -----
    This function only checks if the matrix is square. It does not
    validate other adjacency matrix properties like non-negativity
    or symmetry.    """
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency matrix must be square. Got shape {a.shape}")


def check_directed(directed, real_world):
    """Validate directionality parameter.

    Parameters
    ----------
    directed : float
        Directionality value (0.0 for undirected, 1.0 for directed,
        fractional for partially directed).
    real_world : bool
        Whether this is a real-world network (must be fully directed
        or undirected).

    Raises
    ------
    Exception
        If directed value is invalid or fractional for real networks.

    Notes
    -----
    Real-world networks must have directed in {0, 1}.
    Synthetic networks can have 0 <= directed <= 1.    """
    if real_world:
        if directed not in [0, 1]:
            raise ValueError("Fractional direction is not valid for a real network")
    elif directed < 0 or directed > 1:
        raise ValueError(f'Wrong "directed" parameter value: {directed}')


def check_weights_and_directions(a, weighted, directed):
    """Verify adjacency matrix properties match specified parameters.

    Checks if the actual matrix properties (weighted/directed) match
    the declared parameters.

    Parameters
    ----------
    a : scipy.sparse matrix or numpy.ndarray
        Adjacency matrix to check.
    weighted : bool
        Whether the network is expected to be weighted.
    directed : bool or float
        Whether the network is expected to be directed.

    Raises
    ------
    ValueError
        If actual matrix properties don't match declared parameters.

    Notes
    -----
    - Directed: matrix is not symmetric (A != A^T)
    - Weighted: matrix has non-binary values
    - For sparse matrices, checks are done without converting to dense    """
    # Check if matrix is symmetric (directed property)
    if sp.issparse(a):
        is_directed = (a != a.T).nnz > 0
    else:
        is_directed = not np.allclose(a, a.T)
    
    # Check if matrix is weighted
    if sp.issparse(a):
        # For sparse, check if all non-zero values are 0 or 1
        is_weighted = not np.all(np.isin(a.data, [0, 1]))
    else:
        is_weighted = not np.allclose(a, a.astype(bool).astype(int))

    symm_text = "asymmetric" if is_directed else "symmetric"
    if is_directed != bool(directed):
        raise ValueError(
            f'Error in network construction: "directed" set to {directed},'
            f" but the adjacency matrix is {symm_text}"
        )

    w_text = "weighted" if is_weighted else "not weighted"
    if is_weighted != bool(weighted):
        raise ValueError(
            f'Error in network construction: "weighted" set to {weighted},'
            f" but the adjacency matrix is {w_text}"
        )


def calculate_directionality_fraction(adj):
    """Calculate the fraction of directed edges in an adjacency matrix.
    
    A fully symmetric matrix has directionality = 0.0
    A fully asymmetric matrix has directionality = 1.0
    A partially directed matrix has 0.0 < directionality < 1.0
    
    Parameters
    ----------
    adj : scipy.sparse matrix or numpy array
        Adjacency matrix. For weighted networks, edges are considered
        symmetric only if weights are equal.
        
    Returns
    -------
    float
        Fraction of directed edges (0.0 to 1.0)
        
    Notes
    -----
    For weighted networks, this function correctly checks weight equality.
    An edge (i,j) with weight w1 and edge (j,i) with weight w2 are only
    considered symmetric if w1 == w2.
    
    The function works efficiently with sparse matrices without converting
    to dense format when possible.    """
    # For sparse matrices, work directly with COO format
    if sp.issparse(adj):
        # Convert to COO format for easy access to (row, col, data)
        adj_coo = adj.tocoo()
        
        # Remove diagonal entries
        mask = adj_coo.row != adj_coo.col
        rows = adj_coo.row[mask]
        cols = adj_coo.col[mask]
        data = adj_coo.data[mask]
        
        total_edges = len(rows)
        if total_edges == 0:
            return 0.0
        
        # Create a dictionary for fast lookup of edge weights
        edge_dict = {}
        for r, c, d in zip(rows, cols, data):
            edge_dict[(r, c)] = d
        
        # Count symmetric edges (with equal weights)
        symmetric_edges = 0
        for r, c, d in zip(rows, cols, data):
            if r < c:  # Only check each pair once
                if (c, r) in edge_dict and np.allclose(edge_dict[(c, r)], d):
                    symmetric_edges += 2  # Count both directions
    else:
        # For dense matrices
        A_no_diag = adj.copy()
        np.fill_diagonal(A_no_diag, 0)
        
        # Count total edges (non-zero entries)
        total_edges = np.count_nonzero(A_no_diag)
        
        if total_edges == 0:
            return 0.0
        
        # Count symmetric edges with equal weights
        symmetric_edges = 0
        rows, cols = np.nonzero(A_no_diag)
        
        for i, j in zip(rows, cols):
            if i < j:  # Only count each pair once
                # Check if reciprocal edge exists with same weight
                if A_no_diag[j, i] != 0 and np.allclose(A_no_diag[i, j], A_no_diag[j, i]):
                    symmetric_edges += 2
    
    # Directed edges are those that don't have a reciprocal edge with same weight
    directed_edges = total_edges - symmetric_edges
    
    # Return fraction of directed edges
    return directed_edges / total_edges


def select_construction_pipeline(a, graph):
    """Select construction pipeline and determine directionality.
    
    Determines whether to initialize from adjacency matrix or NetworkX graph,
    and calculates the fraction of directed edges for auto-detection of
    network directionality.
    
    Parameters
    ----------
    a : scipy.sparse matrix or None
        Adjacency matrix. If provided, graph must be None.
    graph : networkx.Graph/DiGraph or None
        NetworkX graph object. If provided, a must be None.
        
    Returns
    -------
    tuple
        (pipeline, directed_fraction) where:
        - pipeline: 'adj' or 'graph' indicating initialization method
        - directed_fraction: float in [0.0, 1.0], fraction of asymmetric edges
        
    Raises
    ------
    ValueError
        If both a and graph are None, or if both are provided.
    TypeError
        If graph is not a supported NetworkX type.
        
    Notes
    -----
    The directed_fraction is always calculated and never None. For undirected
    NetworkX graphs, it returns 0.0. For directed graphs and adjacency matrices,
    it calculates the actual fraction of asymmetric edges.    """
    directed_fraction = None
    
    if a is None:
        if graph is None:
            raise ValueError('Either "adj" or "graph" argument must be non-empty')
        else:
            if not np.any(
                [isinstance(graph, gtype) for gtype in SUPPORTED_GRAPH_TYPES]
            ):
                raise TypeError(
                    f"graph should have one of supported graph types: {SUPPORTED_GRAPH_TYPES}"
                )
            else:
                pipeline = "graph"
                # Calculate directionality for NetworkX graphs
                if nx.is_directed(graph):
                    # For directed graphs, calculate the fraction of asymmetric edges
                    adj_from_graph = nx.adjacency_matrix(graph)
                    directed_fraction = calculate_directionality_fraction(adj_from_graph)
                else:
                    directed_fraction = 0.0

    else:
        if graph is None:
            pipeline = "adj"
            # Calculate directionality from adjacency matrix
            directed_fraction = calculate_directionality_fraction(a)
        else:
            raise ValueError('Either "adj" or "graph" should be given, not both')

    return pipeline, directed_fraction


class Network:
    """Network analysis class with focus on spectral graph theory.
    
    This class provides a comprehensive interface for analyzing networks using
    spectral methods. It supports both directed and undirected, weighted and
    unweighted networks, and can be initialized from adjacency matrices or
    NetworkX graphs.
    
    Parameters
    ----------
    adj : scipy.sparse matrix, optional
        Sparse adjacency matrix. Either adj or graph must be provided.
    graph : networkx.Graph or networkx.DiGraph, optional
        NetworkX graph object. Either adj or graph must be provided.
    preprocessing : str, default='giant_cc'
        Preprocessing method to apply:
        - None: No preprocessing (may cause connectivity issues)
        - 'remove_isolates': Remove isolated nodes and self-loops
        - 'giant_cc': Extract giant connected component (undirected)
        - 'giant_scc': Extract giant strongly connected component (directed)
    name : str, default=''
        Name identifier for the network.
    pos : dict, optional
        Node positions as {node: (x, y)} or {node: (x, y, z)}.
    verbose : bool, default=False
        Whether to print progress messages.
    create_nx_graph : bool, default=True
        Whether to create NetworkX graph from adjacency matrix.
        Set to False for large networks to save memory.
    node_attrs : dict, optional
        Node attributes as {node: attribute_dict}.
    logger : logging.Logger, optional
        Logger instance for debugging. If None, creates default logger.
    **network_args : dict
        Additional network parameters:
        - directed : bool, optional (auto-detected if not specified)
        - weighted : bool, optional (auto-detected if not specified)
        - real_world : bool, default=True
        
    Attributes
    ----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix (always stored as sparse).
    graph : networkx.Graph or networkx.DiGraph or None
        NetworkX graph object (None if create_nx_graph=False).
    n : int
        Number of nodes in the network.
    n_cc : int or None
        Number of connected components (set to 1 after giant_cc preprocessing).
    n_scc : int or None
        Number of strongly connected components (set to 1 after giant_scc).
    directed : bool
        Whether the network is directed.
    weighted : bool
        Whether the network is weighted.
    real_world : bool
        Whether this is a real-world network (affects certain analyses).
    name : str
        Network name identifier.
    pos : dict or None
        Node positions if provided.
    node_attrs : dict or None
        Node attributes if provided.
    verbose : bool
        Verbosity flag.
    logger : logging.Logger
        Logger instance.
    init_method : str
        Initialization method used ('adj' or 'graph').
    network_params : dict
        Original network parameters passed to __init__.
    create_nx_graph : bool
        Whether NetworkX graph was created from adjacency matrix.
    outdeg : ndarray
        Out-degree sequence (automatically computed).
    indeg : ndarray
        In-degree sequence (automatically computed).
    deg : ndarray
        Total degree sequence - undirected view (automatically computed).
    scaled_outdeg : ndarray
        Out-degree normalized to [0, 1] (automatically computed).
    scaled_indeg : ndarray
        In-degree normalized to [0, 1] (automatically computed).
    lem_emb : ndarray or None
        Laplacian eigenmaps embedding if computed.
    estrada_communicability : float or None
        Estrada communicability index if computed.
    estrada_bipartivity : float or None
        Estrada bipartivity index if computed.
    _calculated_directionality : float or None
        Fraction of directed edges detected during initialization (private).
    _init_to_final_node_mapping : dict or None
        Mapping from initial to final node indices after preprocessing 
        (private, only when using adjacency without creating graph).
    
    Dynamic Matrix Attributes
    ------------------------
    For each matrix type in ['adj', 'trans', 'lap', 'nlap', 'rwlap'] (undirected)
    or ['adj', 'lap_out', 'lap_in'] (directed), the following attributes are
    dynamically created and initially set to None:
    
    {matrix_type} : scipy.sparse matrix or None
        The matrix itself (computed on demand).
    {matrix_type}_spectrum : ndarray or None
        Eigenvalues of the matrix.
    {matrix_type}_eigenvectors : ndarray or None
        Eigenvectors of the matrix.
    {matrix_type}_zvalues : ndarray or None
        IPR z-values for eigenvector localization.
    {matrix_type}_ipr : ndarray or None
        Inverse participation ratio values.
        
    Methods
    -------
    is_connected()
        Check if the network is connected.
    randomize(rmode='shuffle')
        Create randomized version preserving certain properties.
    get_node_degrees()
        Calculate and store degree sequences.
    get_degree_distr(mode='all')
        Get degree distribution statistics.
    get_matrix(mode)
        Get or compute specified matrix type.
    get_spectrum(mode)
        Get eigenvalues for specified matrix.
    get_eigenvectors(mode)
        Get eigenvectors for specified matrix.
    get_ipr(mode)
        Get inverse participation ratio for specified matrix.
    get_z_values(mode)
        Get IPR z-values for specified matrix.
    diagonalize(mode='lap', verbose=None)
        Compute eigendecomposition for specified matrix.
    partial_diagonalize(spectrum_params)
        Compute partial eigendecomposition.
    calculate_gromov_hyperbolicity(num_samples=100000, return_list=False)
        Calculate Gromov hyperbolicity of the network.
    calculate_z_values(mode='lap')
        Calculate IPR z-values for eigenvector localization.
    calculate_ipr(mode='adj')
        Calculate inverse participation ratio.
    calculate_thermodynamic_entropy(tlist, verbose=False, norm=False)
        Calculate von Neumann entropy at different temperatures.
    calculate_free_entropy(tlist, norm=False)
        Calculate free entropy.
    calculate_q_entropy(q, tlist, norm=False)
        Calculate q-entropy (Tsallis entropy).
    calculate_estrada_communicability()
        Calculate Estrada communicability index.
    get_estrada_bipartivity_index()
        Calculate Estrada bipartivity index.
    localization_signatures(mode='lap')
        Analyze eigenvector localization.
    construct_lem_embedding(dim)
        Construct Laplacian eigenmaps embedding.
        
    Notes
    -----
    The class uses lazy evaluation for matrix computations - matrices and their
    spectral properties are only computed when first requested. The directionality
    is automatically detected if not specified, using a fractional approach that
    can identify partially directed networks.
    
    The `diagonalize` method may raise exceptions if:
    - The network has isolated nodes when n_cc=1 is expected
    - The network has multiple components when only one is expected
    - Complex eigenvalues/eigenvectors are found for undirected networks
    
    Degree sequences (outdeg, indeg, deg, scaled_outdeg, scaled_indeg) are
    automatically computed during initialization via get_node_degrees().
    
    Matrix types available:
    - 'adj': Adjacency matrix
    - 'trans': Transition matrix (undirected only)
    - 'lap': Laplacian matrix
    - 'nlap': Normalized Laplacian (undirected only)
    - 'rwlap': Random walk Laplacian (undirected only)
    - 'lap_out': Out-Laplacian (directed only)
    - 'lap_in': In-Laplacian (directed only)    """

    def __init__(
        self,
        adj=None,
        graph=None,
        preprocessing="giant_cc",
        name="",
        pos=None,
        verbose=False,
        create_nx_graph=True,
        node_attrs=None,
        logger: Optional[logging.Logger] = None,
        **network_args,
    ):
        """Initialize a Network object from adjacency matrix or NetworkX graph.
        
        Creates a network representation with automatic detection of directed/weighted
        properties if not specified. Supports preprocessing to extract connected
        components.
        
        Parameters
        ----------
        adj : scipy.sparse matrix or None
            Adjacency matrix. Mutually exclusive with graph parameter.
        graph : networkx.Graph/DiGraph or None
            NetworkX graph object. Mutually exclusive with adj parameter.
        preprocessing : str or None, default="giant_cc"
            Preprocessing method:
            
            * None: No preprocessing
            * "remove_isolates": Remove isolated nodes
            * "giant_cc": Extract giant connected component
            * "giant_scc": Extract giant strongly connected component (directed only)
        name : str, default=""
            Name identifier for the network.
        pos : dict or None
            Node positions as {node: (x, y)} dictionary.
        verbose : bool, default=False
            Whether to print progress messages.
        create_nx_graph : bool, default=True
            Whether to create NetworkX graph from adjacency matrix.
        node_attrs : dict or None
            Node attributes as {node: {attr: value}} nested dictionary.
        logger : logging.Logger or None
            Logger instance for logging messages.
        **network_args
            Additional network parameters:
            
            * directed : bool or None (auto-detected if None)
            * weighted : bool or None (auto-detected if None)
            * real_world : bool (affects directionality validation)
            
        Attributes
        ----------
        adj : scipy.sparse matrix
            Adjacency matrix in sparse format.
        graph : networkx.Graph/DiGraph or None
            NetworkX graph representation.
        directed : bool
            Whether the network is directed.
        weighted : bool  
            Whether the network has weighted edges.
        n : int
            Number of nodes.
        outdeg : np.ndarray
            Out-degree sequence (set by get_node_degrees()).
        indeg : np.ndarray
            In-degree sequence (set by get_node_degrees()).
        init_method : str
            Initialization method used ('adj' or 'graph').
        _calculated_directionality : float
            Fraction of asymmetric edges (0.0 to 1.0).
        network_params : dict
            Original network parameters passed to constructor.
        real_world : bool
            Whether this is a real-world network.
        _init_to_final_node_mapping : dict
            Mapping from initial to final node indices after preprocessing.
        lem_emb : None
            Placeholder for Laplacian eigenmaps embedding.
        trans, lap, nlap, rwlap, lap_out, lap_in : None or array
            Various matrix representations (initialized to None).
        n_cc : int or None
            Number of connected components (set during some preprocessing).
        n_scc : int or None
            Number of strongly connected components (set during some preprocessing).
            
        Raises
        ------
        ValueError
            If both adj and graph are provided or neither is provided.
            
        Notes
        -----
        The constructor automatically computes node degrees after initialization.
        For large sparse matrices, auto-detection of directed/weighted properties
        uses efficient sparse operations to avoid memory issues.        """
        self.name = name
        self.verbose = verbose
        self.network_params = network_args
        self.create_nx_graph = create_nx_graph
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.init_method, self._calculated_directionality = select_construction_pipeline(adj, graph)

        self.directed = network_args.get("directed")
        if self.directed is None:
            # Use the calculated directionality fraction
            if self._calculated_directionality is not None:
                # Convert float directionality to boolean
                self.directed = bool(self._calculated_directionality > 0)
            else:
                # Fallback to binary detection (for backward compatibility)
                if self.init_method == "adj":
                    # Use sparse operations to avoid memory issues
                    if sp.issparse(adj):
                        self.directed = (adj != adj.T).nnz > 0
                    else:
                        self.directed = not np.allclose(adj, adj.T)
                elif self.init_method == "graph":
                    self.directed = nx.is_directed(graph)

        self.weighted = network_args.get("weighted")
        if self.weighted is None:
            if self.init_method == "adj":
                # Check if the adjacency matrix has only binary values (0 or 1)
                if sp.issparse(adj):
                    # Check if all non-zero values are 1
                    self.weighted = not np.all(np.isin(adj.data, [0, 1]))
                else:
                    self.weighted = not np.allclose(
                        adj, adj.astype(bool).astype(int)
                    )
            elif self.init_method == "graph":
                self.weighted = nx.is_weighted(graph)

        self.real_world = network_args.get("real_world")
        if self.real_world is None:
            self.real_world = True

        check_directed(self.directed, self.real_world)

        # set empty attributes for different matrix and data types
        valid_mtypes = DIR_MATRIX_TYPES if self.directed else UNDIR_MATRIX_TYPES
        for mt in valid_mtypes:
            setattr(self, mt, None)
            setattr(self, mt + "_spectrum", None)
            setattr(self, mt + "_eigenvectors", None)
            setattr(self, mt + "_zvalues", None)
            setattr(self, mt + "_ipr", None)

        # initialize adjacency matrix and (probably) associated graph
        if self.init_method == "adj":
            # initialize Network object from sparse matrix
            self._preprocess_adj_and_data(
                a=adj,
                pos=pos,
                node_attrs=node_attrs,
                preprocessing=preprocessing,
                create_graph=create_nx_graph,
            )

        if self.init_method == "graph":
            # initialize Network object from NetworkX graph or digraph
            self._preprocess_graph_and_data(
                graph=graph, pos=pos, node_attrs=node_attrs, preprocessing=preprocessing
            )

        # each network object has out- and in-degree sequences from its initialization
        self.get_node_degrees()

        self.lem_emb = None

    def _preprocess_graph_and_data(
        self, graph=None, preprocessing=None, pos=None, node_attrs=None
    ):
        """Preprocess graph by removing isolated nodes or extracting components.
        
        Applies preprocessing to NetworkX graph and filters node positions/attributes
        to match the processed graph. Sets instance attributes based on the result.
        
        Parameters
        ----------
        graph : networkx.Graph or DiGraph
            Input graph to preprocess.
        preprocessing : str or None
            Preprocessing method: None, "remove_isolates", "giant_cc", "giant_scc".
        pos : dict or None
            Node positions to filter.
        node_attrs : dict or None
            Node attributes to filter.
            
        Warning
        -------
        This method assumes node labels remain the same after preprocessing.
        If preprocessing relabels nodes, pos and node_attrs filtering may fail
        with KeyError.
        
        Notes
        -----
        Sets the following instance attributes:
        - self.graph: Processed NetworkX graph
        - self.adj: Adjacency matrix from processed graph
        - self.n: Number of nodes
        - self.pos: Filtered node positions
        - self.node_attrs: Filtered node attributes
        - self.n_cc: Number of connected components (only for 'giant_cc' mode)
        - self.n_scc: Number of strongly connected components (only for 'giant_scc' mode)
        
        When verbose=True, prints information about removed nodes and edges.        """
        if preprocessing is None:
            if self.verbose:
                print(
                    "No preprocessing specified, this may lead to unexpected errors in graph connectivity!"
                )
            fgraph = remove_selfloops_from_graph(graph)

        elif preprocessing == "remove_isolates":
            fgraph = remove_isolates_and_selfloops_from_graph(graph)

        elif preprocessing == "giant_cc":
            g_ = remove_selfloops_from_graph(graph)
            fgraph = get_giant_cc_from_graph(g_)
            self.n_cc = 1

        elif preprocessing == "giant_scc":
            g_ = remove_selfloops_from_graph(graph)
            fgraph = get_giant_scc_from_graph(g_)
            self.n_cc = 1
            self.n_scc = 1

        else:
            raise ValueError("Wrong preprocessing type!")

        lost_nodes = graph.number_of_nodes() - fgraph.number_of_nodes()
        lost_edges = graph.number_of_edges() - fgraph.number_of_edges()
        if lost_nodes + lost_edges != 0 and self.verbose:
            print(f"{lost_nodes} nodes and {lost_edges} edges removed ")

        # add node positions if provided
        if pos is not None:
            self.pos = {
                node: pos[node] for node in graph.nodes() if node in fgraph.nodes()
            }
        else:
            self.pos = None

        if node_attrs is not None:
            self.node_attrs = {
                node: node_attrs[node]
                for node in graph.nodes()
                if node in fgraph.nodes()
            }
        else:
            self.node_attrs = None

        self.graph = fgraph
        self.adj = nx.adjacency_matrix(fgraph)
        self.n = nx.number_of_nodes(self.graph)

    def _preprocess_adj_and_data(
        self, a=None, preprocessing=None, pos=None, node_attrs=None, create_graph=True
    ):
        """Preprocess adjacency matrix by extracting connected components.
        
        Applies preprocessing to adjacency matrix, creates node mapping for removed
        nodes, and filters positions/attributes accordingly.
        
        Parameters
        ----------
        a : scipy.sparse matrix
            Input adjacency matrix.
        preprocessing : str or None
            Preprocessing method: None, "remove_isolates", "giant_cc", "giant_scc".
        pos : dict or None
            Node positions to filter.
        node_attrs : dict or None
            Node attributes to filter.
        create_graph : bool, default=True
            Whether to delegate to graph-based preprocessing.
            
        Warning
        -------
        Edge count reporting for undirected networks assumes perfect symmetry.
        Creates identity mapping even when no nodes are removed (memory overhead).
        
        Notes
        -----
        If create_graph=True, delegates to _preprocess_graph_and_data which may
        lose the original sparse matrix format. Otherwise processes the adjacency
        matrix directly and maintains sparsity.
        
        Sets additional attributes:
        - self._init_to_final_node_mapping: Maps original to final node indices
        - self.graph: Set to None when create_graph=False
        - self.n_cc/n_scc: Set conditionally based on preprocessing mode
        
        When verbose=True, prints information about removed nodes and edges.        """
        # if NetworkX graph should be created, we revert to graph-based initialization for simplicity
        if create_graph:
            gtype = nx.DiGraph if self.directed else nx.Graph
            graph = nx.from_scipy_sparse_array(a, create_using=gtype)
            self._preprocess_graph_and_data(
                graph=graph, pos=pos, node_attrs=node_attrs, preprocessing=preprocessing
            )

            return

        if preprocessing is None:
            if self.verbose:
                print(
                    "No preprocessing specified, this may lead to unexpected errors in graph connectivity!"
                )
            fadj = remove_selfloops_from_adj(a)
            nodes_range = range(fadj.shape[0])
            node_mapping = dict(
                zip(nodes_range, nodes_range)
            )  # no nodes have been deleted

        elif preprocessing == "remove_isolates":
            a_ = remove_selfloops_from_adj(a)
            fadj, node_mapping = remove_isolates_from_adj(a_)

        elif preprocessing == "giant_cc":
            a_ = remove_selfloops_from_adj(a)
            fadj, node_mapping = get_giant_cc_from_adj(a_)
            self.n_cc = 1

        elif preprocessing == "giant_scc":
            a_ = remove_selfloops_from_adj(a)
            fadj, node_mapping = get_giant_scc_from_adj(a_)
            self.n_cc = 1
            self.n_scc = 1

        else:
            raise ValueError("Wrong preprocessing type!")

        lost_nodes = a.shape[0] - fadj.shape[0]
        lost_edges = a.nnz - fadj.nnz
        if not self.directed:
            lost_edges = lost_edges // 2

        if lost_nodes + lost_edges != 0 and self.verbose:
            print(f"{lost_nodes} nodes and {lost_edges} edges removed")

        # add node positions if provided
        if pos is not None:
            self.pos = {
                node: pos[node] for node in range(a.shape[0]) if node in node_mapping
            }
        else:
            self.pos = None

        if node_attrs is not None:
            self.node_attrs = {
                node: node_attrs[node]
                for node in range(a.shape[0])
                if node in node_mapping
            }
        else:
            self.node_attrs = None

        self.graph = None
        self.adj = fadj
        self._init_to_final_node_mapping = node_mapping
        self.n = self.adj.shape[0]

    def is_connected(self):
        """Check if the network is connected.
        
        A network is connected if there is a path between every pair of nodes.
        For directed networks, checks weak connectivity (ignoring edge direction).
        
        Returns
        -------
        bool
            True if the network has only one connected component, False otherwise.        """
        ccs = list(get_ccs_from_adj(self.adj))
        return len(ccs) == 1

    def randomize(self, rmode="shuffle"):
        """Create a randomized version of the network.
        
        Different randomization methods preserve different network properties.
        
        Parameters
        ----------
        rmode : str, default='shuffle'
            Randomization method:
            - 'shuffle': Complete edge shuffling (random graph with same density)
            - 'graph_iom': In-degree, Out-degree, and Mutual degree preserving (via NetworkX)
            - 'adj_iom': In-degree, Out-degree, and Mutual degree preserving (via adjacency matrix)
            - 'complete': Randomize as complete graph
            
        Returns
        -------
        Network
            New Network object with randomized edges.
            
        Notes
        -----
        IOM methods preserve in-degree, out-degree, and mutual (reciprocal) degree
        sequences for each node. This maintains the local connectivity patterns
        while randomizing the global structure.
        Shuffle method only preserves edge density.        """
        if rmode == "graph_iom":
            if self.graph is None:
                raise ValueError("NetworkX graph not available. Use 'adj_iom' mode instead.")
            if self.directed:
                g = nx.DiGraph(self.graph)
            else:
                g = nx.Graph(self.graph)

            new_graph = random_rewiring_IOM_preserving(g, r=2)
            rand_adj = nx.adjacency_matrix(new_graph)

        elif rmode == "adj_iom":
            rand_adj = adj_random_rewiring_iom_preserving(
                self.adj, is_weighted=self.weighted, r=2, enable_progressbar=False
            )

        elif rmode == "complete":
            rand_adj = random_rewiring_complete_graph(self.adj)

        elif rmode == "shuffle":
            rand_adj = random_rewiring_dense_graph(self.adj)

        else:
            raise ValueError("Unknown randomization method")

        rand_net = Network(
            adj=sp.csr_matrix(rand_adj),
            name=self.name + f" {rmode} rand",
            pos=self.pos,
            directed=self.directed,
            weighted=self.weighted,
            real_world=False,
            verbose=False,
            **self.network_params,
        )

        return rand_net

    def get_node_degrees(self):
        """Calculate degree sequences for all nodes.
        
        Computes in-degree, out-degree, and total degree for each node.
        Also creates scaled versions normalized to [0, 1].
        
        Sets Attributes
        ---------------
        outdeg : np.ndarray
            Out-degree for each node.
        indeg : np.ndarray
            In-degree for each node.
        deg : np.ndarray
            Total degree (considering undirected edges).
        scaled_outdeg : np.ndarray
            Out-degrees scaled to [0, 1].
        scaled_indeg : np.ndarray
            In-degrees scaled to [0, 1].        """
        # convert sparse matrix to 0-1 format and sum over specific axis
        self.outdeg = np.array(self.adj.astype(bool).astype(int).sum(axis=0)).ravel()
        self.indeg = np.array(self.adj.astype(bool).astype(int).sum(axis=1)).ravel()
        self.deg = np.array(
            (self.adj + self.adj.T).astype(bool).astype(int).sum(axis=1)
        ).ravel()

        min_out = min(self.outdeg)
        min_in = min(self.indeg)
        max_out = max(self.outdeg)
        max_in = max(self.indeg)

        if max_out != min_out:
            self.scaled_outdeg = np.array(
                [1.0 * (deg - min_out) / (max_out - min_out) for deg in self.outdeg]
            )
        else:
            self.scaled_outdeg = np.ones(len(self.outdeg))

        if min_in != max_in:
            self.scaled_indeg = np.array(
                [1.0 * (deg - min_in) / (max_in - min_in) for deg in self.indeg]
            )
        else:
            self.scaled_indeg = np.ones(len(self.indeg))

    def get_degree_distr(self, mode="all"):
        """Get degree distribution of the network.
        
        Parameters
        ----------
        mode : str, default='all'
            Type of degree distribution:
            - 'all': Total degree
            - 'in': In-degree only
            - 'out': Out-degree only
            
        Returns
        -------
        np.ndarray
            Normalized histogram of degree values.
            
        Raises
        ------
        ValueError
            If mode is not recognized.        """
        if mode == "all":
            deg = self.deg
        elif mode == "out":
            deg = self.outdeg
        elif mode == "in":
            deg = self.indeg
        else:
            raise ValueError("Wrong mode for degree distribution.")

        if max(deg) == min(deg):
            # All nodes have same degree
            return np.array([1.0])
        hist, bins = np.histogram(deg, bins=max(deg) - min(deg), density=True)
        return hist

    def get_matrix(self, mode):
        """Get a specific matrix representation of the network.
        
        Computes and caches various matrix representations lazily.
        
        Parameters
        ----------
        mode : str
            Matrix type to retrieve:
            - 'adj': Adjacency matrix
            - 'lap'/'lap_out': Laplacian matrix
            - 'nlap': Normalized Laplacian
            - 'rwlap': Random walk Laplacian
            - 'trans': Transition matrix
            
        Returns
        -------
        scipy.sparse matrix
            The requested matrix representation.
            
        Raises
        ------
        ValueError
            If matrix type is not recognized or not applicable.
            
        Notes
        -----
        Matrices are computed lazily and cached as instance attributes.        """
        check_matrix_type(mode, self.directed)
        matrix = getattr(self, mode)
        if matrix is None:
            if mode == "lap" or mode == "lap_out":
                matrix = get_laplacian(self.adj)
            elif mode == "nlap":
                matrix = get_norm_laplacian(self.adj)
            elif mode == "rwlap":
                matrix = get_rw_laplacian(self.adj)
            elif mode == "trans":
                matrix = get_trans_matrix(self.adj)
            elif mode == "adj":
                matrix = self.adj
            else:
                raise ValueError(f"Wrong matrix type: {mode}")

            setattr(self, mode, matrix)

        matrix = getattr(self, mode)
        return matrix

    def get_spectrum(self, mode):
        """Get eigenvalues of a specific matrix representation.
        
        Computes eigendecomposition if not already cached.
        
        Parameters
        ----------
        mode : str
            Matrix type (see get_matrix for options).
            
        Returns
        -------
        np.ndarray
            Eigenvalues of the specified matrix, sorted in ascending order.
            
        Notes
        -----
        Calls diagonalize() if spectrum not already computed.        """
        check_matrix_type(mode, self.directed)
        spectrum = getattr(self, mode + "_spectrum")
        if spectrum is None:
            self.diagonalize(mode=mode)
            spectrum = getattr(self, mode + "_spectrum")

        return spectrum

    def get_eigenvectors(self, mode):
        """Get eigenvectors of a specific matrix representation.
        
        Computes eigendecomposition if not already cached.
        
        Parameters
        ----------
        mode : str
            Matrix type (see get_matrix for options).
            
        Returns
        -------
        np.ndarray
            Eigenvectors of the specified matrix (as columns).
            Column order matches eigenvalue ordering (ascending).
            Shape is (n_nodes, n_nodes).
            
        Notes
        -----
        Calls diagonalize() if eigenvectors not already computed.        """
        check_matrix_type(mode, self.directed)
        eigenvectors = getattr(self, mode + "_eigenvectors")
        if eigenvectors is None:
            self.diagonalize(mode=mode)
            eigenvectors = getattr(self, mode + "_eigenvectors")

        return eigenvectors

    def get_ipr(self, mode):
        """Get Inverse Participation Ratio (IPR) for eigenvectors.
        
        IPR measures localization of eigenvectors. Higher values indicate
        more localized eigenvectors.
        
        Parameters
        ----------
        mode : str
            Matrix type (see get_matrix for options).
            
        Returns
        -------
        np.ndarray
            IPR values for each eigenvector.
            
        Notes
        -----
        IPR = sum(|v_i|^4) / (sum(|v_i|^2))^2
        For normalized eigenvectors, this simplifies to IPR = sum(|v_i|^4).
        Range: 1/n <= IPR <= 1, where n is the number of nodes.        """
        check_matrix_type(mode, self.directed)
        ipr = getattr(self, mode + "_ipr")
        if ipr is None:
            self.calculate_ipr(mode=mode)
            ipr = getattr(self, mode + "_ipr")

        return ipr

    def get_z_values(self, mode):
        """Get eigenvalue spacing ratios for the specified matrix mode.
        
        Computes the ratio z_i = (λ_nn - λ_i) / (λ_nnn - λ_i) where λ_nn is the
        nearest neighbor eigenvalue and λ_nnn is the next-nearest neighbor.
        These ratios are useful for analyzing spectral statistics without
        requiring eigenvalue unfolding procedures.
        
        Parameters
        ----------
        mode : str
            Matrix mode to use:
            - Undirected: 'adj', 'trans', 'lap', 'nlap', 'rwlap'
            - Directed: 'adj', 'lap_out', 'lap_in'
            
        Returns
        -------
        dict
            Dictionary mapping eigenvalues to their spacing ratios (z-values).
            
        Notes
        -----
        The spacing ratio is a standard measure in random matrix theory for
        characterizing level statistics. It avoids the need for eigenvalue
        unfolding that is required for simple spacing distributions.
        
        If z-values haven't been calculated yet, this method will trigger
        their calculation via calculate_z_values().
        
        References
        ----------
        Atas, Y. Y., et al. (2013). Distribution of the ratio of consecutive
        level spacings in random matrix ensembles. Physical Review Letters,
        110(8), 084101.
        
        Sá, L., Ribeiro, P., & Prosen, T. (2020). Complex spacing ratios:
        A signature of dissipative quantum chaos. Physical Review X, 10(2),
        021019. https://link.aps.org/doi/10.1103/PhysRevX.10.021019        """
        check_matrix_type(mode, self.directed)
        zvals = getattr(self, mode + "_zvalues")
        if zvals is None:
            self.calculate_z_values(mode=mode)
            zvals = getattr(self, mode + "_zvalues")

        return zvals

    def partial_diagonalize(self, spectrum_params):
        """Partial diagonalization for large matrices.

        Parameters
        ----------
        spectrum_params : dict
            Parameters including 'neigs' (number of eigenvalues)

        Returns
        -------
        eigenvalues, eigenvectors : array-like
            Partial spectrum and eigenvectors
            
        Raises
        ------
        NotImplementedError
            This function is not yet implemented.        """
        raise NotImplementedError("Partial diagonalization is not yet implemented")

    def diagonalize(self, mode="lap", verbose=None):
        """Compute eigenvalues and eigenvectors of the specified matrix.
        
        Performs full eigendecomposition of the network matrix specified by mode.
        Results are cached as attributes for later retrieval.
        
        Parameters
        ----------
        mode : str, optional
            Matrix mode to diagonalize:
            - Undirected: 'adj', 'trans', 'lap', 'nlap', 'rwlap'
            - Directed: 'adj', 'lap_out', 'lap_in'
            Default is 'lap'.
        verbose : bool or None, optional
            Whether to print progress messages. If None, uses self.verbose.
            Default is None.
            
        Notes
        -----
        After diagonalization, eigenvalues and eigenvectors are stored as:
        - self.{mode}_spectrum: eigenvalues (sorted)
        - self.{mode}_eigenvectors: right eigenvectors (as columns)
        
        The method uses scipy.linalg.eigh for symmetric matrices and
        scipy.linalg.eig for non-symmetric matrices. Complex eigenvalues
        and eigenvectors are allowed for directed networks but will raise
        an error for undirected networks.        """
        if verbose is None:
            verbose = self.verbose

        check_matrix_type(mode, self.directed)
        if verbose:
            print("Preparing for diagonalizing...")

        outdeg = self.outdeg
        indeg = self.indeg
        deg = self.deg

        A = self.adj.astype(float)
        n = self.n

        if n != np.count_nonzero(outdeg) and verbose:
            self.logger.warning(
                f"{n - np.count_nonzero(outdeg)} nodes without out-edges"
            )
        if n != np.count_nonzero(indeg) and verbose:
            self.logger.warning(f"{n - np.count_nonzero(indeg)} nodes without in-edges")

        nz = np.count_nonzero(deg)
        if nz != n and self.n_cc == 1:
            self.logger.error(f"Graph has {n - nz} isolated nodes!")
            raise Exception(f"Graph has {n - nz} isolated nodes!")

        if not self.weighted and not self.directed and not np.allclose(outdeg, indeg):
            raise Exception("out- and in- degrees do not coincide in boolean")

        matrix = self.get_matrix(mode)

        if verbose:
            print("Performing diagonalization...")

        # Check symmetry properly without converting to dense
        if sp.issparse(matrix):
            # Check if A - A.T has any non-zero elements
            diff = matrix - matrix.T
            matrix_is_symmetric = diff.nnz == 0 or np.allclose(diff.data, 0)
        else:
            matrix_is_symmetric = np.allclose(matrix, matrix.T)
        if matrix_is_symmetric:
            raw_eigs, right_eigvecs = la.eigh(matrix.todense())
        else:
            raw_eigs, right_eigvecs = la.eig(matrix.todense(), right=True)

        raw_eigs = np.around(raw_eigs, decimals=12)
        sorted_eigs = np.sort(raw_eigs)

        if "lap" in mode:
            n_comp = len(raw_eigs[np.abs(raw_eigs) == 0])
            if n_comp != 1 and not self.weighted and self.n_cc == 1:
                print("eigenvalues:", sorted_eigs)
                raise Exception("Graph has %d components!" % n_comp)

        setattr(self, mode, matrix)

        if np.allclose(np.imag(sorted_eigs), np.zeros(len(sorted_eigs)), atol=1e-12):
            sorted_eigs = np.real(sorted_eigs)
        else:
            if not self.directed:
                raise ValueError("Complex eigenvalues found in non-directed network!")

        setattr(self, mode + "_spectrum", sorted_eigs)

        # Sort eigenvectors to match eigenvalue order
        sort_indices = np.argsort(raw_eigs)
        sorted_eigenvectors = right_eigvecs[:, sort_indices]
        if np.allclose(
            np.imag(sorted_eigenvectors), np.zeros(sorted_eigenvectors.shape), atol=1e-8
        ):
            sorted_eigenvectors = np.real(sorted_eigenvectors)
        else:
            if not self.directed:
                raise ValueError("Complex eigenvectors found in non-directed network!")

        setattr(self, mode + "_eigenvectors", sorted_eigenvectors)

        if verbose:
            print("Diagonalizing finished")

    def calculate_gromov_hyperbolicity(self, num_samples=100000, return_list=False):
        """Calculate Gromov hyperbolicity of the network.
        
        Gromov hyperbolicity measures how "tree-like" a graph is. A tree has 
        hyperbolicity 0, while graphs with many cycles have higher values.
        
        This implementation uses random sampling of 4-tuples of nodes to estimate
        the hyperbolicity efficiently for large graphs.
        
        Parameters
        ----------
        num_samples : int, default=100000
            Number of random 4-tuples to sample
        return_list : bool, default=False
            If True, return the list of all hyperbolicity values.
            If False, return the average.
            
        Returns
        -------
        float or list
            Average hyperbolicity value, or list of all values if return_list=True
            
        References
        ----------
        - Chalopin, J., Chepoi, V., Dragan, F.F., Ducoffe, G., Mohammed, A., 
          Vaxès, Y. (2018). "Fast approximation and exact computation of negative 
          curvature parameters of graphs" arXiv:1803.06324
          
        Notes
        -----
        Warning: For large graphs, this precomputes ALL shortest paths which
        requires O(n²) memory. Use with caution on graphs with >10k nodes.        """
        import random
        
        # Ensure we have a NetworkX graph
        if not hasattr(self, 'graph') or self.graph is None:
            raise ValueError("NetworkX graph not available. Initialize Network with create_nx_graph=True")
            
        G = self.graph
        
        # Check if graph has enough nodes
        if len(G) < 4:
            raise ValueError(f"Graph must have at least 4 nodes for hyperbolicity calculation, got {len(G)}")
        
        # Relabel nodes to 0, 1, 2, ... for efficient indexing
        rG = nx.relabel_nodes(G, {list(G.nodes)[i]: i for i in range(len(G.nodes))})
        nds = list(rG.nodes)
        
        # Precompute all shortest paths
        spmat = np.zeros((len(nds), len(nds)))
        gen = nx.all_pairs_dijkstra_path_length(rG, weight=None)
        
        for i in range(len(nds)):
            n0, n0dict = next(gen)
            spmat[n0, np.array(list(n0dict.keys()))] = np.array(list(n0dict.values()))
        
        hsum = 0
        hvals = []
        
        # Sample random 4-tuples and compute hyperbolicity
        for i in range(num_samples):
            # Sample 4 distinct nodes
            node_tuple = random.sample(nds, 4)
            n0, n1, n2, n3 = node_tuple
            
            # Get distances between all pairs
            d01 = spmat[n0, n1]
            d23 = spmat[n2, n3]
            d02 = spmat[n0, n2]
            d13 = spmat[n1, n3]
            d03 = spmat[n0, n3]
            d12 = spmat[n1, n2]
            
            # Compute the three sums
            s = [d01 + d23, d02 + d13, d03 + d12]
            s.sort()
            
            # Hyperbolicity is half the difference between the two largest sums
            h = (s[-1] - s[-2]) / 2
            hsum += h
            hvals.append(h)
        
        if return_list:
            return hvals
        else:
            return hsum / num_samples

    def calculate_z_values(self, mode="lap"):
        """Calculate eigenvalue spacing ratios (z-values) for spectral analysis.
        
        Internal method that computes the spacing ratios z_i = (λ_nn - λ_i) / (λ_nnn - λ_i)
        where λ_nn and λ_nnn are the nearest and next-nearest neighbor eigenvalues
        in the complex plane. These ratios characterize level statistics without
        requiring eigenvalue unfolding.
        
        Parameters
        ----------
        mode : str, default="lap"
            Matrix mode for eigenvalue calculation. See get_matrix() for options.
            
        Raises
        ------
        ValueError
            If fewer than 3 unique eigenvalues exist (minimum needed for ratios).
            
        Notes
        -----
        Results are stored in self.<mode>_zvalues as a dictionary mapping
        eigenvalues to their z-values. Duplicate eigenvalues are removed
        before calculation. Uses k-d tree search in 2D (real, imaginary)
        space for efficient nearest neighbor finding.
        
        The spacing ratio avoids the need for eigenvalue unfolding that is
        required for simple spacing distributions, making it particularly
        useful for analyzing spectra with non-uniform density.
        
        References
        ----------
        Atas, Y. Y., et al. (2013). Distribution of the ratio of consecutive
        level spacings in random matrix ensembles. Physical Review Letters,
        110(8), 084101.
        
        Sá, L., Ribeiro, P., & Prosen, T. (2020). Complex spacing ratios:
        A signature of dissipative quantum chaos. Physical Review X, 10(2),
        021019. https://link.aps.org/doi/10.1103/PhysRevX.10.021019
        
        See Also
        --------
        ~driada.network.net_base.get_z_values : Public interface to retrieve calculated z-values
        """
        spectrum = self.get_spectrum(mode)
        seigs = sorted(list(set(spectrum)), key=np.abs)
        if len(seigs) != len(spectrum) and self.verbose:
            print(
                "WARNING:", len(spectrum) - len(seigs), "repeated eigenvalues discarded"
            )

        if len(seigs) < 3:
            raise ValueError(
                f"Cannot compute z-values: need at least 3 eigenvalues, got {len(seigs)}"
            )

        if self.verbose:
            print("Computing nearest neighbours...")

        X = np.array([[np.real(x), np.imag(x)] for x in seigs])
        nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(X)
        distances, indices = nbrs.kneighbors(X)
        nnbs = np.array([seigs[x] for x in indices[:, 1]])
        nnnbs = np.array([seigs[x] for x in indices[:, 2]])
        # Vectorized computation with division by zero protection
        with np.errstate(divide='ignore', invalid='ignore'):
            zlist = (nnbs - seigs) / (nnnbs - seigs)
            # Replace any inf/nan values with 0
            zlist = np.nan_to_num(zlist, nan=0.0, posinf=0.0, neginf=0.0)
        zdict = dict(zip(seigs, zlist))

        setattr(self, mode + "_zvalues", zdict)

    def calculate_ipr(self, mode="adj"):
        """Calculate Inverse Participation Ratio (IPR) for eigenvectors.
        
        Internal method that computes the IPR for each eigenvector of the specified
        matrix. IPR quantifies the degree of localization of eigenvectors - higher
        values indicate more localized (less spread out) eigenvectors.
        
        Parameters
        ----------
        mode : str, default="adj"
            Matrix mode for eigenvector calculation. See get_matrix() for options.
            
        Notes
        -----
        For each eigenvector v_i with components v_{i,j}, the IPR is:
        IPR_i = sum_j |v_{i,j}|^4
        
        The IPR ranges from 1/N (completely delocalized) to 1 (completely localized
        on a single node), where N is the number of nodes.
        
        Results are stored in self.<mode>_ipr as an array of IPR values for each
        eigenvector. Also computes eigenvector Shannon entropy but does not store it.
        
        The IPR is widely used in physics to characterize Anderson localization
        and in network science to study eigenvector localization properties.
        
        See Also
        --------
        ~driada.network.net_base.get_ipr : Public interface to retrieve calculated IPR values
        ~driada.network.net_base.get_eigenvectors : Retrieves the eigenvectors used in calculation
        
        Notes
        -----
        Also computes Shannon entropy of eigenvectors internally but does not
        store it. This may be changed in future versions.
        """
        eigenvectors = self.get_eigenvectors(mode)
        nvecs = eigenvectors.shape[1]
        ipr = np.zeros(nvecs)
        eig_entropy = np.zeros(nvecs)

        for i in range(nvecs):
            # Vectorized computation for better performance
            ipr[i] = np.sum(np.abs(eigenvectors[:, i]) ** 4)
            # entropy[i] = -np.log(ipr[i]) # erdos entropy (deprecated)
            # Vectorized entropy computation (still not stored)
            eig_entropy[i] = entropy(np.abs(eigenvectors[:, i]) ** 2)

        setattr(self, mode + "_ipr", ipr)

    def _get_lap_spectrum(self, norm=False):
        """Get Laplacian spectrum for directed or undirected networks.
        
        Parameters
        ----------
        norm : bool, default=False
            If True, returns normalized Laplacian spectrum.
            
        Returns
        -------
        np.ndarray
            Eigenvalues of the appropriate Laplacian matrix.
            
        Raises
        ------
        NotImplementedError
            If normalized Laplacian requested for directed network.        """
        if not self.directed:
            if norm:
                spectrum = self.get_spectrum("nlap")  # could be rwlap as well
            else:
                spectrum = self.get_spectrum("lap")
        else:
            if norm:
                raise NotImplementedError(
                    "Normalized Laplacian not implemented for directed networks"
                )
            else:
                spectrum = self.get_spectrum("lap_out")

        return spectrum

    def calculate_thermodynamic_entropy(self, tlist, verbose=False, norm=False):
        """Calculate von Neumann entropy at different temperatures.
        
        Computes the von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ) for density matrices
        ρ = exp(-tL)/Z constructed from the graph Laplacian spectrum at various
        inverse temperatures t.
        
        Parameters
        ----------
        tlist : array-like
            List of inverse temperature values (t = β). Higher values correspond
            to lower temperatures in the thermodynamic analogy.
        verbose : bool, default=False
            If True, prints intermediate calculation steps.
        norm : bool, default=False
            If True, uses normalized Laplacian spectrum. If False, uses
            unnormalized Laplacian. Not supported for directed graphs.
            
        Returns
        -------
        list of float
            Von Neumann entropy values S(ρ) for each temperature in tlist.
            Values are in bits (using log₂).
            
        Notes
        -----
        The density matrix ρ = exp(-tL)/Z represents a thermal state where:
        - L is the graph Laplacian (encoding network structure)
        - t is inverse temperature (time parameter in diffusion context)
        - Z = Tr(exp(-tL)) is the partition function
        
        The entropy quantifies the "quantumness" or disorder in the network's
        spectral properties. It interpolates between:
        - t → 0: S → log₂(N) (maximum entropy, uniform distribution)
        - t → ∞: S → 0 (minimum entropy, ground state dominates)
        
        References
        ----------
        De Domenico, M., & Biamonte, J. (2016). Spectral entropies as
        information-theoretic tools for complex network comparison.
        Physical Review X, 6(4), 041062.
        
        See Also
        --------
        ~driada.network.net_base.calculate_free_entropy : Free entropy (log partition function)
        ~driada.network.net_base.calculate_q_entropy : Generalized Rényi entropy        """
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [spectral_entropy(spectrum, t, verbose=verbose) for t in tlist]
        return res

    def calculate_free_entropy(self, tlist, norm=False):
        """Calculate free entropy (log partition function) at different temperatures.
        
        Computes the free entropy F = log₂(Z) where Z = Tr(exp(-tL)) is the
        partition function derived from the graph Laplacian spectrum.
        
        Parameters
        ----------
        tlist : array-like
            List of inverse temperature values (t = β). Higher values correspond
            to lower temperatures.
        norm : bool, default=False
            If True, uses normalized Laplacian spectrum. If False, uses
            unnormalized Laplacian. Not supported for directed graphs.
            
        Returns
        -------
        list of float
            Free entropy values F(t) = log₂(Z(t)) for each temperature.
            
        Notes
        -----
        The free entropy F = log₂(Z) is the logarithm of the partition function
        in bits. This is also known as the Massieu function in statistical physics.
        Note: This differs from the Helmholtz free energy F_Helmholtz = -kT ln(Z).
        
        Our measure F = log₂(Z) = ln(Z)/ln(2) represents the effective number of 
        accessible states on a logarithmic scale:
        
        - Low t (high T): Many states accessible, large F
        - High t (low T): Few states accessible, small F
        
        The partition function Z = sum_i exp(-tλ_i) sums over all eigenvalues λ_i
        of the Laplacian, encoding the network's spectral properties.
        
        References
        ----------
        Ghavasieh, A., et al. (2021). Multiscale statistical physics of the
        pan-viral interactome unravels the systemic nature of SARS-CoV-2
        infections. Communications Physics, 4(1), 83.
        https://www.nature.com/articles/s42005-021-00582-8
        
        See Also
        --------
        ~driada.network.net_base.calculate_thermodynamic_entropy : Von Neumann entropy
        ~driada.network.net_base.calculate_q_entropy : Generalized Rényi entropy        """
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [free_entropy(spectrum, t) for t in tlist]
        return res

    def calculate_q_entropy(self, q, tlist, norm=False):
        """Calculate Rényi q-entropy at different temperatures.
        
        Computes the Rényi entropy of order q for density matrices derived from
        the graph Laplacian spectrum. Generalizes von Neumann entropy (q=1) to a
        family of entropy measures with different sensitivities to the probability
        distribution.
        
        Parameters
        ----------
        q : float
            Order parameter for Rényi entropy. Must be positive.
            Special cases:
            - q → 0: Hartley entropy (log of support size)
            - q = 1: Von Neumann entropy (Shannon entropy)
            - q = 2: Collision entropy (related to purity)
            - q → ∞: Min-entropy (negative log of max probability)
        tlist : array-like
            List of inverse temperature values (t = β).
        norm : bool, default=False
            If True, uses normalized Laplacian spectrum. If False, uses
            unnormalized Laplacian. Not supported for directed graphs.
            
        Returns
        -------
        list of float
            Rényi q-entropy values S_q(ρ) for each temperature in bits.
            
        Raises
        ------
        Exception
            If q ≤ 0 or if complex entropy values are detected.
            
        Notes
        -----
        The Rényi q-entropy is defined as:
        - For q ≠ 1: S_q(ρ) = (1/(1-q)) log(Tr(ρ^q))
        - For q = 1: S_1(ρ) = -Tr(ρ log ρ) (von Neumann entropy)
        
        Different q values emphasize different parts of the spectrum:
        - Small q: Sensitive to small probabilities (rare events)
        - Large q: Sensitive to large probabilities (typical events)
        
        References
        ----------
        De Domenico, M., & Biamonte, J. (2016). Spectral entropies as
        information-theoretic tools for complex network comparison.
        Physical Review X, 6(4), 041062.
        
        See Also
        --------
        ~driada.network.net_base.calculate_thermodynamic_entropy : Von Neumann entropy (q=1 case)
        ~driada.network.net_base.calculate_free_entropy : Free entropy (log partition function)        """
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [q_entropy(spectrum, t, q=q) for t in tlist]
        return res

    def calculate_estrada_communicability(self):
        """Calculate Estrada communicability index of the network.
        
        The Estrada communicability is the sum of exponentials of the
        adjacency matrix eigenvalues: G = sum(exp(λᵢ)). It measures the
        ease of communication across the entire network.
        
        Returns
        -------
        float
            Estrada communicability index G.
            
        Notes
        -----
        Higher values indicate better overall network communicability.
        This metric is sensitive to the number of closed walks of all lengths.
        
        References
        ----------
        Estrada, E., & Hatano, N. (2008). Communicability in complex networks.
        Physical Review E, 77(3), 036111.        """
        adj_spectrum = self.get_spectrum("adj")
        self.estrada_communicability = sum([np.exp(e) for e in adj_spectrum])
        return self.estrada_communicability

    def get_estrada_bipartivity_index(self):
        """Calculate Estrada bipartivity index of the network.
        
        The bipartivity index measures how close a network is to being bipartite.
        It is computed as the ratio of sums of hyperbolic functions of eigenvalues:
        β = (sum(cosh(λᵢ)) - sum(sinh(λᵢ))) / (sum(cosh(λᵢ)) + sum(sinh(λᵢ)))
        
        Returns
        -------
        float
            Bipartivity index between 0 and 1, where 1 indicates perfect bipartivity.
            
        Notes
        -----
        For bipartite graphs, all eigenvalues come in ±λ pairs, making sinh terms cancel.
        The index equals 1 for perfectly bipartite graphs and decreases with deviation.
        
        References
        ----------
        Estrada, E., & Rodríguez-Velázquez, J. A. (2005). Spectral measures of
        bipartivity in complex networks. Physical Review E, 72(4), 046105.        """
        adj_spectrum = self.get_spectrum("adj")
        esi1 = sum([np.exp(-e) for e in adj_spectrum])
        esi2 = sum([np.exp(e) for e in adj_spectrum])
        self.estrada_bipartivity = esi1 / esi2
        return self.estrada_bipartivity

    def localization_signatures(self, mode="lap"):
        """Compute statistical signatures of eigenstate localization.
        
        Analyzes the z-values (eigenvalue spacing ratios) to extract signatures
        that distinguish between localized and delocalized eigenstates.
        
        Parameters
        ----------
        mode : str, default="lap"
            Matrix type to analyze. Must be a valid matrix mode.
            
        Returns
        -------
        tuple of (float, float)
            - mean_cos_phi: Average of cos(arg(z)) over all z-values
            - mean_inv_r_squared: Average of 1/|z|² over all z-values
            
        Notes
        -----
        These signatures help identify Anderson localization transitions.
        Localized states typically show different statistical properties
        in their eigenvalue spacing ratios compared to extended states.
        
        Requires prior calculation of z-values via calculate_z_values().        """
        zvals = self.get_z_values(mode)

        mean_cos_phi = np.mean(np.array([np.cos(np.angle(x)) for x in zvals]))
        rvals = [1.0 / (np.abs(z)) ** 2 for z in zvals]
        mean_inv_r_sq = np.mean(np.array(rvals))

        if self.verbose:
            self.logger.info(f"mean cos phi complex: {mean_cos_phi}")
            self.logger.info(f"mean 1/r^2 real: {mean_inv_r_sq}")

        return mean_inv_r_sq, mean_cos_phi

    def construct_lem_embedding(self, dim):
        """Construct Laplacian Eigenmaps (LEM) embedding of the network.
        
        Computes a low-dimensional embedding that preserves local network structure
        by using the eigenvectors of the graph Laplacian corresponding to the
        smallest non-zero eigenvalues.
        
        Parameters
        ----------
        dim : int
            Target embedding dimension. Must be less than number of nodes.
            
        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_nodes, dim) where each row is the
            low-dimensional representation of a node.
            
        Notes
        -----
        The embedding minimizes the weighted sum of squared distances between
        connected nodes. The first eigenvector (constant) is excluded as it
        doesn't provide discriminative information.
        
        For disconnected graphs, only the giant component is embedded.
        
        This method modifies self.lem_emb to store the embedding and
        self.lem_eigvals to store the selected eigenvalues.
        
        References
        ----------
        Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality
        reduction and data representation. Neural computation, 15(6), 1373-1396.        """
        if self.directed:
            raise Exception("LEM embedding is not implemented for directed graphs")

        A = self.adj
        A = A.astype(float)
        self.logger.info("Performing spectral decomposition...")
        K = A.shape[0]
        NL = get_norm_laplacian(A)
        DH = get_inv_sqrt_diag_matrix(A)

        # For Laplacian Eigenmaps, we need the SMALLEST eigenvalues
        # Request dim+1 to include the zero eigenvalue
        start_v = np.ones(K)
        
        # Use SM (Smallest Magnitude) instead of LR (Largest Real)
        eigvals, eigvecs = eigs(NL, k=min(dim + 1, K-1), which="SM", v0=start_v, maxiter=K * 1000)
        eigvals = np.asarray([np.round(np.real(x), 6) for x in eigvals])
        
        # Sort by eigenvalue (smallest first)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Check connectivity: should have exactly one zero eigenvalue
        n_zero_eigvals = np.sum(np.abs(eigvals) < 1e-6)
        if n_zero_eigvals != 1:
            raise Exception(
                f"Error while LEM embedding construction: graph is not connected! "
                f"Found {n_zero_eigvals} zero eigenvalues, expected 1."
            )
        
        # Select the eigenvectors corresponding to the smallest non-zero eigenvalues
        # Skip the first (zero) eigenvalue and its constant eigenvector
        vecs = eigvecs[:, 1 : dim + 1]  # shape: (n_nodes, dim)
        
        # Normalize eigenvectors
        vec_norms = np.sqrt(np.sum(np.abs(vecs) ** 2, axis=0))
        vecs = vecs / vec_norms
        
        # Apply D^{-1/2} transformation
        # explanation: https://jlmelville.github.io/smallvis/spectral.html
        vecs = DH.dot(vecs)  # DH is (n_nodes, n_nodes), vecs is (n_nodes, dim)

        self.lem_emb = vecs.T  # Store as (dim, n_nodes) for backward compatibility
