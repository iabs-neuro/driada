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
    Undirected networks support: 'adj', 'trans', 'lap', 'nlap', 'rwlap'.
    """
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
    Exception
        If matrix is not square.
    """
    if a.shape[0] != a.shape[1]:
        raise Exception("Non-square adjacency matrix!")


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
    Synthetic networks can have 0 <= directed <= 1.
    """
    if real_world:
        if directed not in [0, 1, 0.0, 1.0]:
            raise Exception("Fractional direction is not valid for a real network")
    elif directed < 0 or directed > 1:
        raise Exception(f'Wrong "directed" parameter value: {directed}')


def check_weights_and_directions(a, weighted, directed):
    """Verify adjacency matrix properties match specified parameters.

    Checks if the actual matrix properties (weighted/directed) match
    the declared parameters.

    Parameters
    ----------
    a : scipy.sparse matrix
        Adjacency matrix to check.
    weighted : bool
        Whether the network is expected to be weighted.
    directed : bool or float
        Whether the network is expected to be directed.

    Raises
    ------
    Exception
        If actual matrix properties don't match declared parameters.

    Notes
    -----
    - Directed: matrix is not symmetric (A != A^T)
    - Weighted: matrix has non-binary values
    """
    is_directed = not np.allclose(a.toarray(), a.toarray().T)
    is_weighted = not np.allclose(a.toarray(), a.astype(bool).astype(int).toarray())

    symm_text = "asymmetric" if is_directed else "symmetric"
    if is_directed != bool(directed):
        raise Exception(
            f'Error in network construction: "directed" set to {directed},'
            f" but the adjacency matrix is {symm_text}"
        )

    w_text = "weighted" if is_weighted else "not weighted"
    if is_weighted != bool(weighted):
        raise Exception(
            f'Error in network construction: "weighted" set to {weighted},'
            f" but the adjacency matrix {w_text}"
        )


def calculate_directionality_fraction(adj):
    """Calculate the fraction of directed edges in an adjacency matrix.
    
    A fully symmetric matrix has directionality = 0.0
    A fully asymmetric matrix has directionality = 1.0
    A partially directed matrix has 0.0 < directionality < 1.0
    
    Parameters
    ----------
    adj : scipy.sparse matrix or numpy array
        Adjacency matrix
        
    Returns
    -------
    float
        Fraction of directed edges (0.0 to 1.0)
    """
    # Convert to dense for easier calculation
    if sp.issparse(adj):
        A = adj.todense()
    else:
        A = adj
        
    # Remove diagonal to focus on edges
    A_no_diag = A.copy()
    np.fill_diagonal(A_no_diag, 0)
    
    # Count total edges (non-zero entries)
    total_edges = np.count_nonzero(A_no_diag)
    
    if total_edges == 0:
        return 0.0  # Empty graph is considered undirected
    
    # Count symmetric edges (edges that exist in both directions)
    # For each edge (i,j), check if (j,i) also exists
    symmetric_edges = 0
    rows, cols = np.nonzero(A_no_diag)
    
    for i, j in zip(rows, cols):
        if i < j and A_no_diag[j, i] != 0:  # Only count each symmetric pair once
            symmetric_edges += 2  # Count both (i,j) and (j,i)
    
    # Directed edges are those that don't have a reciprocal edge
    directed_edges = total_edges - symmetric_edges
    
    # Return fraction of directed edges
    return directed_edges / total_edges


def select_construction_pipeline(a, graph):
    """Select construction pipeline and determine directionality.
    
    This function now supports partial directions by calculating
    the exact fraction of directed edges in the input.
    
    Parameters
    ----------
    a : scipy.sparse matrix or None
        Adjacency matrix
    graph : networkx.Graph/DiGraph or None
        NetworkX graph object
        
    Returns
    -------
    tuple
        (pipeline, directed_fraction) where:
        - pipeline is 'adj' or 'graph'
        - directed_fraction is float between 0.0 and 1.0 or None
    """
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
    """
    An object for network analysis with the focus on spectral graph theory
    """

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
                self.directed = self._calculated_directionality
            else:
                # Fallback to binary detection (for backward compatibility)
                if self.init_method == "adj":
                    self.directed = not np.allclose(adj.toarray(), adj.toarray().T)
                elif self.init_method == "graph":
                    self.directed = nx.is_directed(graph)

        self.weighted = network_args.get("weighted")
        if self.weighted is None:
            if self.init_method == "adj":
                self.weighted = not np.allclose(
                    adj.toarray(), adj.toarray().astype(bool).astype(int)
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
            True if the network has only one connected component, False otherwise.
        """
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
        Shuffle method only preserves edge density.
        """
        if rmode == "graph_iom":
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
            In-degrees scaled to [0, 1].
        """
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
            If mode is not recognized.
        """
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
        Exception
            If matrix type is not recognized or not applicable.
        """
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
                raise Exception(f"Wrong matrix type: {mode}")

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
            Eigenvalues of the specified matrix.
        """
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
        """
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
        """
        check_matrix_type(mode, self.directed)
        ipr = getattr(self, mode + "_ipr")
        if ipr is None:
            self.calculate_ipr(mode=mode)
            ipr = getattr(self, mode + "_ipr")

        return ipr

    def get_z_values(self, mode):
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
        """
        raise NotImplementedError("Partial diagonalization is not yet implemented")

    def diagonalize(self, mode="lap", verbose=None):
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

        matrix_is_symmetric = np.allclose(matrix.data, matrix.T.data)
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

        sorted_eigenvectors = right_eigvecs[
            np.ix_(range(len(sorted_eigs)), np.argsort(raw_eigs))
        ]
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
        """
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
        nnbs = [seigs[x] for x in indices[:, 1]]
        nnnbs = [seigs[x] for x in indices[:, 2]]
        """
        nndist = np.array([(nnbs[i] - eigs[i]) for i in range(len(eigs))])
        nnndist = np.array([(nnnbs[i] - eigs[i]) for i in range(len(eigs))])
        """
        zlist = np.array(
            [(nnbs[i] - seigs[i]) / (nnnbs[i] - seigs[i]) for i in range(len(seigs))]
        )
        zdict = dict(zip(seigs, zlist))

        setattr(self, mode + "_zvalues", zdict)

    def calculate_ipr(self, mode="adj"):
        eigenvectors = self.get_eigenvectors(mode)
        nvecs = eigenvectors.shape[1]
        ipr = np.zeros(nvecs)
        eig_entropy = np.zeros(nvecs)

        for i in range(nvecs):
            ipr[i] = sum([np.abs(v) ** 4 for v in eigenvectors[:, i]])
            # entropy[i] = -np.log(ipr[i]) # erdos entropy (deprecated)
            eig_entropy[i] = entropy(
                np.array([np.abs(v) ** 2 for v in eigenvectors[:, i]])
            )

        setattr(self, mode + "_ipr", ipr)
        # self.eigenvector_entropy = eig_entropy / np.log(self.n)

    def _get_lap_spectrum(self, norm=False):
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
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [spectral_entropy(spectrum, t, verbose=verbose) for t in tlist]
        return res

    def calculate_free_entropy(self, tlist, norm=False):
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [free_entropy(spectrum, t) for t in tlist]
        return res

    def calculate_q_entropy(self, q, tlist, norm=False):
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [q_entropy(spectrum, t, q=q) for t in tlist]
        return res

    def calculate_estrada_communicability(self):
        adj_spectrum = self.get_spectrum("adj")
        self.estrada_communicability = sum([np.exp(e) for e in adj_spectrum])
        return self.estrada_communicability

    def get_estrada_bipartivity_index(self):
        adj_spectrum = self.get_spectrum("adj")
        esi1 = sum([np.exp(-e) for e in adj_spectrum])
        esi2 = sum([np.exp(e) for e in adj_spectrum])
        self.estrada_bipartivity = esi1 / esi2
        return self.estrada_bipartivity

    def localization_signatures(self, mode="lap"):
        zvals = self.get_z_values(mode)

        mean_cos_phi = np.mean(np.array([np.cos(np.angle(x)) for x in zvals]))
        rvals = [1.0 / (np.abs(z)) ** 2 for z in zvals]
        mean_inv_r_sq = np.mean(np.array(rvals))

        if self.verbose:
            self.logger.info(f"mean cos phi complex: {mean_cos_phi}")
            self.logger.info(f"mean 1/r^2 real: {mean_inv_r_sq}")

        return mean_inv_r_sq, mean_cos_phi

    def construct_lem_embedding(self, dim):
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
