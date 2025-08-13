"""
Graph randomization algorithms for network analysis.

This module provides various methods for randomizing graphs while preserving
certain structural properties like in-out motifs (IOM), degree sequences, etc.
"""

import copy
import logging
from typing import Optional, Union, Tuple, List
import warnings
import random

import numpy as np
import scipy.sparse as sp
import networkx as nx
import tqdm

from .graph_utils import *
from .matrix_utils import *


def _validate_adjacency_matrix(
    a: Union[np.ndarray, sp.spmatrix], allow_dense: bool = True
) -> None:
    """Validate input adjacency matrix.

    Parameters
    ----------
    a : array-like
        Adjacency matrix to validate
    allow_dense : bool
        Whether to allow dense matrices

    Raises
    ------
    ValueError
        If matrix is invalid
    """
    if not (isinstance(a, np.ndarray) or sp.issparse(a)):
        raise ValueError("Input must be a numpy array or scipy sparse matrix")

    if a.ndim != 2:
        raise ValueError(f"Adjacency matrix must be 2D, got shape {a.shape}")

    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {a.shape}")

    if not allow_dense and not sp.issparse(a):
        raise ValueError("This function requires a sparse matrix input")


def adj_random_rewiring_iom_preserving(
    a: Union[np.ndarray, sp.spmatrix],
    is_weighted: bool,
    r: int = 10,
    logger: Optional[logging.Logger] = None,
    enable_progressbar: bool = True,
    random_state: Optional[int] = None,
) -> sp.csr_matrix:
    """
    Perform In-Out-Motif (IOM) preserving graph randomization on adjacency matrix.

    This method rewires edges while preserving the in-out motif structure of the graph.
    It handles symmetric (bidirectional) and non-symmetric (unidirectional) components
    separately to maintain the graph's motif patterns.

    Parameters
    ----------
    a : array-like
        Input adjacency matrix (sparse or dense)
    is_weighted : bool
        Whether the graph is weighted
    r : int, default=10
        Number of rewiring iterations per edge (higher values = more randomization)
    logger : logging.Logger, optional
        Logger for tracking progress
    enable_progressbar : bool, default=True
        Whether to show progress bar
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    scipy.sparse.csr_matrix
        Randomized adjacency matrix

    References
    ----------
    Sporns, O., & Kötter, R. (2004). Motifs in brain networks.
    PLoS biology, 2(11), e369.
    """
    # Setup
    logger = logger or logging.getLogger(__name__)
    _validate_adjacency_matrix(a)

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    logger.info(f"Starting IOM-preserving randomization with r={r}")

    # Process symmetric component (bidirectional edges)
    s = symmetric_component(a, is_weighted)
    rs = turn_to_partially_directed(s, directed=1.0, weighted=is_weighted)
    # rs is already dense matrix
    rows, cols = rs.nonzero()
    edgeset = set(zip(rows, cols))
    upper = list(edgeset)
    source_nodes = [e[0] for e in upper]
    target_nodes = [e[1] for e in upper]

    double_edges = len(upper)
    total_double_rewires = double_edges * r

    logger.debug(f"Processing {double_edges} bidirectional edges")

    # Progress tracking for double edges
    if enable_progressbar:
        pbar = tqdm.tqdm(
            total=total_double_rewires, desc="Rewiring bidirectional edges"
        )

    i = 0
    while i < total_double_rewires:
        good_choice = 0
        attempts = 0
        max_attempts = 1000  # Prevent infinite loops

        while not good_choice and attempts < max_attempts:
            ind1, ind2 = np.random.choice(double_edges, 2)
            n1, n3 = source_nodes[ind1], source_nodes[ind2]
            n2, n4 = target_nodes[ind1], target_nodes[ind2]

            if len(set([n1, n2, n3, n4])) == 4:
                good_choice = 1
            attempts += 1

        if attempts >= max_attempts:
            logger.warning(
                f"Max attempts reached for finding valid edge pair at iteration {i}"
            )
            break

        w1 = s[n1, n2]
        w2 = s[n2, n1]
        w3 = s[n3, n4]
        w4 = s[n4, n3]

        # Check if rewiring is valid (no existing edges)
        if s[n1, n3] + s[n1, n4] + s[n2, n3] + s[n2, n4] == 0:
            s[n1, n4] = w1
            s[n4, n1] = w2
            s[n2, n3] = w3
            s[n3, n2] = w4

            s[n1, n2] = 0
            s[n2, n1] = 0
            s[n3, n4] = 0
            s[n4, n3] = 0

            target_nodes[ind1], target_nodes[ind2] = n4, n2

            i += 1
            if enable_progressbar:
                pbar.update(1)

    if enable_progressbar:
        pbar.close()

    # Process non-symmetric component (unidirectional edges)
    ns = non_symmetric_component(a, is_weighted)
    rows, cols = ns.nonzero()
    edges = list(set(zip(rows, cols)))
    source_nodes = [e[0] for e in edges]
    target_nodes = [e[1] for e in edges]
    single_edges = len(edges)
    total_single_rewires = single_edges * r

    logger.debug(f"Processing {single_edges} unidirectional edges")

    # Progress tracking for single edges
    if enable_progressbar:
        pbar = tqdm.tqdm(
            total=total_single_rewires, desc="Rewiring unidirectional edges"
        )

    i = 0
    while i < total_single_rewires:
        good_choice = 0
        attempts = 0
        max_attempts = 1000

        while not good_choice and attempts < max_attempts:
            ind1, ind2 = np.random.choice(single_edges, 2)
            n1, n3 = source_nodes[ind1], source_nodes[ind2]
            n2, n4 = target_nodes[ind1], target_nodes[ind2]

            if len(set([n1, n2, n3, n4])) == 4:
                good_choice = 1
            attempts += 1

        if attempts >= max_attempts:
            logger.warning(
                f"Max attempts reached for finding valid edge pair at iteration {i}"
            )
            break

        w1 = ns[n1, n2]
        w2 = ns[n3, n4]

        # Check all possible conflicts
        checklist = [
            ns[n1, n3],
            ns[n1, n4],
            ns[n2, n3],
            ns[n2, n4],
            ns[n3, n1],
            ns[n4, n1],
            ns[n3, n2],
            ns[n4, n2],
            s[n3, n1],
            s[n4, n1],
            s[n3, n2],
            s[n4, n2],
        ]

        if checklist.count(0) == 12:  # No conflicting edges
            ns[n1, n4] = w1
            ns[n3, n2] = w2

            ns[n1, n2] = 0
            ns[n3, n4] = 0

            i += 1
            target_nodes[ind1], target_nodes[ind2] = n4, n2

            if enable_progressbar:
                pbar.update(1)

    if enable_progressbar:
        pbar.close()

    # Combine components
    res = s + ns
    if not is_weighted:
        res = res.astype(bool)

    logger.info("IOM-preserving randomization completed")
    return sp.csr_array(res)


def random_rewiring_complete_graph(
    a: Union[np.ndarray, sp.spmatrix],
    p: float = 1.0,
    logger: Optional[logging.Logger] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Randomize edge weights in a complete graph.

    This function shuffles edge weights while preserving the complete graph structure.
    It's useful for testing null models where connectivity is fixed but weights vary.

    Parameters
    ----------
    a : array-like
        Complete adjacency matrix
    p : float, default=1.0
        Proportion of edges to shuffle (0 to 1)
    logger : logging.Logger, optional
        Logger for tracking progress
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    numpy.ndarray
        Randomized adjacency matrix

    Raises
    ------
    ValueError
        If graph is not complete
    """
    logger = logger or logging.getLogger(__name__)
    _validate_adjacency_matrix(a)

    if random_state is not None:
        np.random.seed(random_state)

    # Convert to dense if sparse
    if sp.issparse(a):
        a = a.toarray()

    n = a.shape[0]
    n_edges = len(np.nonzero(a)[0])
    n_expected = n**2 - n

    if n_edges < n_expected:
        raise ValueError(
            f"Graph is not complete: has {n_edges} edges, expected {n_expected}"
        )

    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    logger.info(f"Randomizing complete graph with p={p}")
    symmetric = np.allclose(a, a.T)

    all_x_inds, all_y_inds = np.nonzero(a)
    vals = a[all_x_inds, all_y_inds]
    shuffled_positions = np.random.choice(
        np.arange((n**2 - n)), replace=False, size=int(p * (n**2 - n))
    )

    shuffled_x_inds = all_x_inds[shuffled_positions]
    shuffled_y_inds = all_y_inds[shuffled_positions]
    shuffled_vals = vals[shuffled_positions]
    np.random.shuffle(shuffled_vals)  # shuffling in-place

    stable_part = a.copy()
    if symmetric:
        stable_part[shuffled_x_inds, shuffled_y_inds] = 0
        stable_part[shuffled_y_inds, shuffled_x_inds] = 0
    else:
        stable_part[shuffled_x_inds, shuffled_y_inds] = 0

    if symmetric:
        shuffled_half = np.zeros(a.shape)
        shuffled_half[shuffled_x_inds, shuffled_y_inds] = shuffled_vals
        shuffled_part = (shuffled_half + shuffled_half.T) / 2.0
    else:
        shuffled_part = np.zeros(a.shape)
        shuffled_part[shuffled_x_inds, shuffled_y_inds] = shuffled_vals

    rewired = stable_part + shuffled_part

    return rewired


def random_rewiring_dense_graph(
    a: Union[np.ndarray, sp.spmatrix],
    logger: Optional[logging.Logger] = None,
    random_state: Optional[int] = None,
    gap_fill_weight: float = 0.0001,
) -> np.ndarray:
    """
    Randomize edge weights for dense (nearly complete) graphs.

    This function handles graphs that are almost complete by using a "gap filling"
    technique. It adds small weights to missing edges, performs randomization,
    then removes them.

    Parameters
    ----------
    a : array-like
        Dense adjacency matrix
    logger : logging.Logger, optional
        Logger for tracking progress
    random_state : int, optional
        Random seed for reproducibility
    gap_fill_weight : float, default=0.0001
        Small weight to temporarily fill gaps

    Returns
    -------
    numpy.ndarray
        Randomized adjacency matrix
    """
    logger = logger or logging.getLogger(__name__)
    _validate_adjacency_matrix(a)

    if random_state is not None:
        np.random.seed(random_state)

    # Convert to dense
    if isinstance(a, np.ndarray):
        afull = a.copy()
        nelem = len(np.nonzero(a)[0])
    else:
        afull = a.toarray()
        nelem = a.nnz

    n = a.shape[0]
    max_edges = n**2 - n

    if nelem != max_edges:
        logger.debug(
            f"Graph is not complete ({nelem}/{max_edges} edges), using gap filling"
        )
        temp = gap_fill_weight
    else:
        temp = 0

    # Fill gaps with small weights
    ps_a = afull + np.full(a.shape, temp) - np.eye(n) * temp

    # Extract upper triangular values (exploit symmetry)
    vals = ps_a[np.nonzero(np.triu(ps_a))]
    np.random.shuffle(vals)

    # Reconstruct symmetric matrix
    tri = np.zeros(a.shape)
    tri[np.triu_indices(n, 1)] = vals
    rewired = tri + tri.T

    # Remove gap filling weights
    res = rewired - np.full(a.shape, temp) + np.eye(n) * temp

    logger.info("Dense graph randomization completed")
    return res


def get_single_double_edges_lists(
    g: nx.Graph,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Separate edges into single (unidirectional) and double (bidirectional) lists.

    Parameters
    ----------
    g : networkx.Graph or DiGraph
        Input graph

    Returns
    -------
    tuple of lists
        (single_edges, double_edges) where each is a list of (source, target) tuples
    """
    single_edges = []
    double_edges = []
    h = nx.to_undirected(g).copy()

    for e in h.edges():
        if g.has_edge(e[1], e[0]):
            if g.has_edge(e[0], e[1]):
                double_edges.append((e[0], e[1]))
            else:
                single_edges.append((e[1], e[0]))
        else:
            single_edges.append((e[0], e[1]))

    return single_edges, double_edges


def random_rewiring_IOM_preserving(G: nx.Graph, r: int = 10) -> nx.Graph:
    """
    DEPRECATED: Use adj_random_rewiring_iom_preserving() instead.

    Legacy NetworkX-based implementation of IOM-preserving randomization.
    This function is slower and less efficient than the adjacency matrix version.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    r : int, default=10
        Number of rewiring iterations per edge

    Returns
    -------
    networkx.Graph
        Randomized graph

    Warnings
    --------
    This function is deprecated and will be removed in v2.0.
    Use adj_random_rewiring_iom_preserving() with nx.adjacency_matrix() instead.
    """
    warnings.warn(
        "random_rewiring_IOM_preserving is deprecated and will be removed in v2.0. "
        "Use adj_random_rewiring_iom_preserving() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    [L1, L2] = get_single_double_edges_lists(G)
    Number_of_single_edges = len(L1)
    Number_of_double_edges = len(L2)
    Number_of_rewired_1_edge_pairs = Number_of_single_edges * r
    Number_of_rewired_2_edge_pairs = Number_of_double_edges * r
    # Number_of_rewired_2_edge_pairs = 20
    i = 0
    count = 0
    previous_text = ""

    # Legacy implementation - not optimized
    while i < Number_of_rewired_2_edge_pairs:
        Edge_index_1 = random.randint(0, Number_of_double_edges - 1)
        Edge_index_2 = random.randint(0, Number_of_double_edges - 1)
        Edge_1 = L2[Edge_index_1]
        Edge_2 = L2[Edge_index_2]
        [Node_A, Node_B] = Edge_1
        [Node_C, Node_D] = Edge_2
        while (
            (Node_A == Node_C)
            or (Node_A == Node_D)
            or (Node_B == Node_C)
            or (Node_B == Node_D)
        ):
            Edge_index_1 = random.randint(0, Number_of_double_edges - 1)
            Edge_index_2 = random.randint(0, Number_of_double_edges - 1)
            Edge_1 = L2[Edge_index_1]
            Edge_2 = L2[Edge_index_2]
            [Node_A, Node_B] = Edge_1
            [Node_C, Node_D] = Edge_2

        # print ('Edges:',Node_A, Node_B, ';',Node_C, Node_D)
        # print G.has_edge(Node_A, Node_B), G.has_edge(Node_B, Node_A), G.has_edge(Node_C, Node_D), G.has_edge(Node_D, Node_C)
        if (
            G.has_edge(Node_A, Node_D) == 0
            and G.has_edge(Node_D, Node_A) == 0
            and G.has_edge(Node_C, Node_B) == 0
            and G.has_edge(Node_B, Node_C) == 0
        ):
            # try:
            try:
                w_ab = G.get_edge_data(Node_A, Node_B)["weight"]
            except (KeyError, TypeError, AttributeError):
                pass  # Edge may not have weight attribute
            G.remove_edge(Node_A, Node_B)
            G.remove_edge(Node_B, Node_A)
            """
            except nx.NetworkXError:
                pass
                #print('fuck')
            """
            try:
                try:
                    w_cd = G.get_edge_data(Node_C, Node_D)["weight"]
                except (KeyError, TypeError, AttributeError):
                    pass  # Edge may not have weight attribute
                G.remove_edge(Node_C, Node_D)
                G.remove_edge(Node_D, Node_C)
            except nx.NetworkXError:
                pass
                # print('fuck')

            try:
                G.add_edge(Node_A, Node_D, weight=w_ab)
                G.add_edge(Node_D, Node_A, weight=w_ab)
            except (NameError, UnboundLocalError):
                G.add_edge(Node_A, Node_D)
                G.add_edge(Node_D, Node_A)

            try:
                G.add_edge(Node_C, Node_B, weight=w_cd)
                G.add_edge(Node_B, Node_C, weight=w_cd)
            except (NameError, UnboundLocalError):
                G.add_edge(Node_C, Node_B)
                G.add_edge(Node_B, Node_C)

            # print L2[Edge_index_1]
            L2[Edge_index_1] = (Node_A, Node_D)
            # print L2[Edge_index_1]
            # L2[Edge_index_1+1] = (Node_D, Node_A)
            L2[Edge_index_2] = (Node_C, Node_B)
            # L2[Edge_index_2+1] = (Node_B, Node_C)
            i += 1

        if (i != 0) and (i % (Number_of_double_edges // 1)) == 0:
            text = str(round(100.0 * i / Number_of_rewired_2_edge_pairs, 0)) + "%"
            if text != previous_text:
                # print text
                pass
            previous_text = text

    i = 0
    # Process single connections
    while i < Number_of_rewired_1_edge_pairs:
        Edge_index_1 = random.randint(0, Number_of_single_edges - 1)
        Edge_index_2 = random.randint(0, Number_of_single_edges - 1)
        Edge_1 = L1[Edge_index_1]
        Edge_2 = L1[Edge_index_2]
        [Node_A, Node_B] = Edge_1
        [Node_C, Node_D] = Edge_2
        while (
            (Node_A == Node_C)
            or (Node_A == Node_D)
            or (Node_B == Node_C)
            or (Node_B == Node_D)
        ):
            Edge_index_1 = random.randint(0, Number_of_single_edges - 1)
            Edge_index_2 = random.randint(0, Number_of_single_edges - 1)
            Edge_1 = L1[Edge_index_1]
            Edge_2 = L1[Edge_index_2]
            [Node_A, Node_B] = Edge_1
            [Node_C, Node_D] = Edge_2

        if (
            G.has_edge(Node_A, Node_D) == 0
            and G.has_edge(Node_D, Node_A) == 0
            and G.has_edge(Node_C, Node_B) == 0
            and G.has_edge(Node_B, Node_C) == 0
        ):
            try:
                try:
                    w_ab = G.get_edge_data(Node_A, Node_B)["weight"]
                except (KeyError, TypeError, AttributeError):
                    pass  # Edge may not have weight attribute
                G.remove_edge(Node_A, Node_B)

            except nx.NetworkXError:
                pass  # Error handled

            try:
                try:
                    w_cd = G.get_edge_data(Node_C, Node_D)["weight"]
                except (KeyError, TypeError, AttributeError):
                    pass  # Edge may not have weight attribute
                G.remove_edge(Node_C, Node_D)

            except nx.NetworkXError:
                pass  # Error handled

            try:
                G.add_edge(Node_A, Node_D, weight=w_ab)
            except (NameError, UnboundLocalError):
                G.add_edge(Node_A, Node_D)

            try:
                G.add_edge(Node_C, Node_B, weight=w_cd)
            except (NameError, UnboundLocalError):
                G.add_edge(Node_C, Node_B)

            L1[Edge_index_1] = (Node_A, Node_D)
            L1[Edge_index_2] = (Node_C, Node_B)
            i += 1

        if (i != 0) and (i % (Number_of_single_edges // 1)) == 0:
            text = str(round(100.0 * i / Number_of_rewired_1_edge_pairs, 0)) + "%"
            if text != previous_text:
                # print text
                pass
            previous_text = text

    G_rewired = copy.deepcopy(G)

    return G_rewired


def randomize_graph(
    adjacency: Union[np.ndarray, sp.spmatrix],
    method: str = "iom",
    iterations: Optional[int] = None,
    preserve_weights: bool = True,
    p: float = 1.0,
    logger: Optional[logging.Logger] = None,
    enable_progressbar: bool = True,
    random_state: Optional[int] = None,
    **kwargs,
) -> Union[np.ndarray, sp.spmatrix]:
    """
    Unified interface for graph randomization.

    This function provides a single entry point for all graph randomization methods,
    making it easier to switch between different algorithms.

    Parameters
    ----------
    adjacency : array-like
        Input adjacency matrix (sparse or dense)
    method : str, default='iom'
        Randomization method to use:
        - 'iom': In-Out-Motif preserving (for general graphs)
        - 'complete': Complete graph randomization (weights only)
        - 'dense': Dense graph randomization (with gap filling)
    iterations : int, optional
        Number of rewiring iterations (method-specific)
        Default: 10 * number of edges for 'iom'
    preserve_weights : bool, default=True
        Whether to preserve edge weights
    p : float, default=1.0
        Proportion of edges to shuffle (for 'complete' method)
    logger : logging.Logger, optional
        Logger for tracking progress
    enable_progressbar : bool, default=True
        Whether to show progress bar
    random_state : int, optional
        Random seed for reproducibility
    **kwargs
        Additional method-specific parameters

    Returns
    -------
    array-like
        Randomized adjacency matrix (same format as input)

    Raises
    ------
    ValueError
        If unknown method is specified

    Examples
    --------
    >>> import numpy as np
    >>> from driada.network.randomization import randomize_graph
    >>>
    >>> # Create a simple adjacency matrix
    >>> adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>>
    >>> # Randomize using IOM preservation
    >>> rand_adj = randomize_graph(adj, method='iom', random_state=42)
    >>>
    >>> # Randomize complete graph weights
    >>> rand_adj = randomize_graph(adj, method='complete', p=0.5)
    """
    logger = logger or logging.getLogger(__name__)

    if method == "iom":
        r = iterations or 10
        return adj_random_rewiring_iom_preserving(
            adjacency,
            is_weighted=preserve_weights,
            r=r,
            logger=logger,
            enable_progressbar=enable_progressbar,
            random_state=random_state,
        )

    elif method == "complete":
        return random_rewiring_complete_graph(
            adjacency, p=p, logger=logger, random_state=random_state
        )

    elif method == "dense":
        gap_fill_weight = kwargs.get("gap_fill_weight", 0.0001)
        return random_rewiring_dense_graph(
            adjacency,
            logger=logger,
            random_state=random_state,
            gap_fill_weight=gap_fill_weight,
        )

    else:
        raise ValueError(
            f"Unknown randomization method: {method}. "
            f"Choose from: 'iom', 'complete', 'dense'"
        )
