"""
Common graph fixtures for network tests.

This module provides standardized graph fixtures to ensure consistent
testing across all network modules.
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx


def create_standard_graph(
    n=10, density=0.3, weighted=False, directed=True, random_state=42
):
    """
    Create a standard test graph with consistent properties.

    Parameters
    ----------
    n : int, default=10
        Number of nodes
    density : float, default=0.3
        Approximate edge density (0 to 1)
    weighted : bool, default=False
        Whether to create weighted edges
    directed : bool, default=True
        Whether the graph is directed
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacency matrix
    """
    np.random.seed(random_state)

    # Create base structure with guaranteed connectivity
    adj = np.zeros((n, n))

    # Create a ring to ensure connectivity
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        if not directed:
            adj[(i + 1) % n, i] = 1

    # Add random edges to reach desired density
    n_edges_target = int(n * (n - 1) * density)
    if not directed:
        n_edges_target //= 2

    n_edges_current = np.count_nonzero(adj)

    while n_edges_current < n_edges_target:
        i, j = np.random.randint(0, n, 2)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1
            if not directed:
                adj[j, i] = 1
            n_edges_current = np.count_nonzero(adj)

    if weighted:
        # Convert to weights
        weights = np.random.rand(n, n) * 0.8 + 0.2  # Weights between 0.2 and 1.0
        adj = adj * weights
        if not directed:
            adj = (adj + adj.T) / 2

    return sp.csr_matrix(adj)


def create_complete_graph(n=10, weighted=True, random_state=42):
    """
    Create a complete graph for testing.

    Parameters
    ----------
    n : int, default=10
        Number of nodes
    weighted : bool, default=True
        Whether to create weighted edges
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    numpy.ndarray
        Dense adjacency matrix
    """
    np.random.seed(random_state)

    adj = np.ones((n, n)) - np.eye(n)

    if weighted:
        weights = np.random.rand(n, n) * 0.8 + 0.2
        weights = (weights + weights.T) / 2  # Make symmetric
        np.fill_diagonal(weights, 0)
        adj = adj * weights

    return adj


def create_dense_graph(n=10, completeness=0.8, weighted=True, random_state=42):
    """
    Create a dense (but not complete) graph for testing.

    Parameters
    ----------
    n : int, default=10
        Number of nodes
    completeness : float, default=0.8
        Fraction of possible edges to include
    weighted : bool, default=True
        Whether to create weighted edges
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    numpy.ndarray
        Dense adjacency matrix
    """
    np.random.seed(random_state)

    # Start with complete graph
    adj = create_complete_graph(n, weighted=False, random_state=random_state)

    # Remove some edges
    n_edges_total = n * (n - 1)
    n_edges_keep = int(n_edges_total * completeness)
    n_edges_remove = n_edges_total - n_edges_keep

    # Get all possible edges (only upper triangle for symmetric graph)
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_edges_remove = min(n_edges_remove // 2, len(edges))  # Divide by 2 since we'll remove symmetrically
    edges_to_remove = np.random.choice(len(edges), n_edges_remove, replace=False)

    for idx in edges_to_remove:
        i, j = edges[idx]
        adj[i, j] = 0
        adj[j, i] = 0  # Make symmetric

    if weighted:
        weights = np.random.rand(n, n) * 0.8 + 0.2
        weights = (weights + weights.T) / 2  # Make symmetric
        adj = adj * weights
        np.fill_diagonal(adj, 0)

    return adj


def create_bipartite_graph(n1=5, n2=5, density=0.5, weighted=False, random_state=42):
    """
    Create a bipartite graph for testing edge cases.

    Parameters
    ----------
    n1 : int, default=5
        Number of nodes in first partition
    n2 : int, default=5
        Number of nodes in second partition
    density : float, default=0.5
        Edge density between partitions
    weighted : bool, default=False
        Whether to create weighted edges
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacency matrix
    """
    np.random.seed(random_state)

    n = n1 + n2
    adj = np.zeros((n, n))

    # Add edges only between partitions
    n_edges_target = int(n1 * n2 * density)
    n_edges_current = 0

    while n_edges_current < n_edges_target:
        i = np.random.randint(0, n1)
        j = np.random.randint(n1, n)
        if adj[i, j] == 0:
            adj[i, j] = 1
            adj[j, i] = 1  # Make symmetric
            n_edges_current += 1

    if weighted:
        weights = np.random.rand(n, n) * 0.8 + 0.2
        weights = (weights + weights.T) / 2
        adj = adj * weights

    return sp.csr_matrix(adj)


def create_small_test_graph():
    """Create a small graph for quick validation tests."""
    adj = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    return sp.csr_matrix(adj)


def create_networkx_graph(n=10, graph_type="erdos_renyi", random_state=42):
    """
    Create a NetworkX graph for legacy function testing.

    Parameters
    ----------
    n : int, default=10
        Number of nodes
    graph_type : str, default='erdos_renyi'
        Type of graph to create
    random_state : int, default=42
        Random seed

    Returns
    -------
    networkx.Graph or DiGraph
        NetworkX graph object
    """
    np.random.seed(random_state)

    if graph_type == "erdos_renyi":
        G = nx.erdos_renyi_graph(n, 0.3, seed=random_state)
    elif graph_type == "barabasi":
        G = nx.barabasi_albert_graph(n, 3, seed=random_state)
    elif graph_type == "cycle":
        G = nx.cycle_graph(n)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return G
