"""Population-level recurrence graph construction."""

import numpy as np
import scipy.sparse as sp


def population_recurrence_graph(recurrence_graphs, method='joint', threshold=1.0,
                                 binarize_threshold=None):
    """Combine per-neuron recurrence graphs into a population graph.

    Parameters
    ----------
    recurrence_graphs : list of RecurrenceGraph
        Individual recurrence graphs. All must have the same number of nodes.
    method : {'joint', 'mean'}
        Combination strategy:
        - 'joint': Thresholded element-wise AND (JRP). A point (i,j) is kept
          if at least ``threshold`` fraction of neurons recur there.
          threshold=1.0 is strict AND, threshold=0.5 is majority voting.
        - 'mean': Average recurrence matrices. Values in [0, 1].
          Optionally binarize with ``binarize_threshold``.
    threshold : float, default=1.0
        For method='joint': fraction of neurons that must recur.
    binarize_threshold : float, optional
        For method='mean': if given, binarize the averaged matrix.

    Returns
    -------
    RecurrenceGraph
        Population-level recurrence graph (via from_adjacency).

    Raises
    ------
    ValueError
        If graphs have different sizes or method is unknown.
    """
    if not recurrence_graphs:
        raise ValueError("recurrence_graphs must be non-empty")

    valid_methods = ('joint', 'mean')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {valid_methods}.")

    sizes = [rg.adj.shape[0] for rg in recurrence_graphs]
    if len(set(sizes)) > 1:
        raise ValueError(
            f"All recurrence graphs must have same number of nodes, got {set(sizes)}"
        )

    n_graphs = len(recurrence_graphs)
    n = sizes[0]

    # Sum all adjacency matrices
    summed = sp.csr_matrix((n, n), dtype=float)
    for rg in recurrence_graphs:
        summed = summed + rg.adj.astype(float)

    if method == 'joint':
        min_count = threshold * n_graphs
        summed_coo = summed.tocoo()
        mask = summed_coo.data >= min_count
        adj = sp.csr_matrix(
            (np.ones(mask.sum()), (summed_coo.row[mask], summed_coo.col[mask])),
            shape=(n, n),
        )
    else:  # mean
        adj = summed / n_graphs
        if binarize_threshold is not None:
            adj_coo = adj.tocoo()
            mask = adj_coo.data >= binarize_threshold
            adj = sp.csr_matrix(
                (np.ones(mask.sum()), (adj_coo.row[mask], adj_coo.col[mask])),
                shape=(n, n),
            )

    from .recurrence_graph import RecurrenceGraph
    return RecurrenceGraph.from_adjacency(adj)
