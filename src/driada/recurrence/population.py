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


def pairwise_jaccard_sparse(matrices, trim_to_min=False):
    """Compute pairwise Jaccard similarity for sparse binary matrices.

    Uses sparse matrix multiplication to compute all pairwise intersection
    counts in a single operation, avoiding O(N²) Python loops.

    Parameters
    ----------
    matrices : list of scipy.sparse matrices or RecurrenceGraph objects
        Square binary adjacency matrices.  RecurrenceGraph objects are
        accepted (uses ``.adj``).
    trim_to_min : bool, default=False
        If True, matrices with different sizes (e.g., from per-neuron
        embeddings with different tau/dim) are trimmed to the smallest
        common size before comparison.  If False (default), different
        sizes raise ValueError.

    Returns
    -------
    ndarray of shape (N, N)
        Symmetric matrix of Jaccard indices. Diagonal is 1.0 for
        non-empty matrices, 0.0 for empty ones.

    Raises
    ------
    ValueError
        If list is empty, or if matrices have different shapes and
        *trim_to_min* is False.
    """
    if not matrices:
        raise ValueError("matrices must be non-empty")

    # Accept RecurrenceGraph objects
    adjs = []
    for m in matrices:
        adjs.append(m.adj if hasattr(m, 'adj') else m)
    matrices = adjs

    sizes = [m.shape[0] for m in matrices]
    if len(set(sizes)) > 1:
        if not trim_to_min:
            raise ValueError(
                f"All matrices must have the same shape, got sizes "
                f"{sorted(set(sizes))}. Pass trim_to_min=True to trim "
                f"to the smallest common size."
            )
        min_n = min(sizes)
        matrices = [
            m[:min_n, :min_n] if m.shape[0] > min_n else m
            for m in matrices
        ]

    n_matrices = len(matrices)
    n = matrices[0].shape[0]
    n_sq = n * n

    # Flatten each (n, n) matrix into a row of length n², stack into (N, n²)
    rows_list = []
    for m in matrices:
        coo = sp.coo_matrix(m)
        linear_idx = coo.row.astype(np.int64) * n + coo.col.astype(np.int64)
        rows_list.append(sp.csr_matrix(
            (np.ones(len(linear_idx)), (np.zeros(len(linear_idx), dtype=int), linear_idx)),
            shape=(1, n_sq),
        ))

    X = sp.vstack(rows_list, format='csr')

    # X @ X.T gives all pairwise intersection counts
    intersections = (X @ X.T).toarray().astype(float)
    sizes = np.diag(intersections).copy()

    # Jaccard = intersection / (|A| + |B| - intersection)
    unions = sizes[:, None] + sizes[None, :] - intersections
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard = np.where(unions > 0, intersections / unions, 0.0)

    return jaccard
