"""Population-level recurrence graph construction."""

import warnings

import numpy as np
import scipy.sparse as sp


def _reconcile_graph_sizes(items, trim='adaptive', **kwargs):
    """Reconcile different-sized recurrence graphs or sparse matrices.

    Parameters
    ----------
    items : list of RecurrenceGraph or scipy.sparse matrices
        Items to reconcile.  RecurrenceGraph objects are identified by
        having an ``.adj`` attribute.
    trim : {None, 'min', 'adaptive'}
        Strategy for handling different sizes:

        - ``None``: raise ValueError if sizes differ.
        - ``'min'``: trim all items to the smallest common size.
        - ``'adaptive'``: compute per-item embedding loss (= max_size - size),
          remove items whose loss exceeds ``mean + tail_sigma * std``,
          then trim the rest to the new minimum.  Issues a warning when
          items are removed.
    **kwargs
        Extra parameters for the ``'adaptive'`` strategy:

        - ``tail_sigma`` (float, default 3.0): number of standard deviations
          for the outlier threshold.

    Returns
    -------
    items : list
        Reconciled items (same type as input, trimmed copies where needed).
    kept_mask : ndarray of bool
        Boolean mask of length ``len(original_items)``.  True for items
        that were kept (all True when no outliers are removed).
    """
    if not items:
        raise ValueError("items must be non-empty")

    is_rg = [hasattr(it, 'adj') for it in items]
    sizes = np.array([
        it.adj.shape[0] if has_adj else it.shape[0]
        for it, has_adj in zip(items, is_rg)
    ])
    n_items = len(items)
    kept_mask = np.ones(n_items, dtype=bool)

    if len(set(sizes)) == 1:
        return items, kept_mask

    # --- Strategy dispatch ---
    if trim is None:
        raise ValueError(
            f"Items have different sizes ({sorted(set(sizes))}). "
            f"Pass trim='min' or trim='adaptive' to handle this."
        )

    if trim == 'adaptive':
        tail_sigma = kwargs.get('tail_sigma', 3.0)
        max_size = sizes.max()
        losses = max_size - sizes  # 0 for the longest, positive for shorter
        mean_loss = losses.mean()
        std_loss = losses.std()
        if std_loss > 0:
            threshold = mean_loss + tail_sigma * std_loss
            kept_mask = losses <= threshold
            n_removed = n_items - kept_mask.sum()
            if n_removed > 0:
                warnings.warn(
                    f"Adaptive trim: removed {n_removed}/{n_items} items "
                    f"with embedding loss > {threshold:.0f} "
                    f"(mean={mean_loss:.0f}, std={std_loss:.0f}, "
                    f"sigma={tail_sigma}). "
                    f"Removed sizes: {sorted(sizes[~kept_mask])}",
                    stacklevel=3,
                )
            items = [it for it, keep in zip(items, kept_mask) if keep]
            is_rg = [r for r, keep in zip(is_rg, kept_mask) if keep]
            sizes = sizes[kept_mask]

    # trim='min' or post-adaptive: trim all to min size
    min_n = sizes.min()
    from .recurrence_graph import RecurrenceGraph

    trimmed = []
    for it, has_adj in zip(items, is_rg):
        if has_adj:
            if it.adj.shape[0] > min_n:
                trimmed.append(RecurrenceGraph.from_adjacency(
                    it.adj[:min_n, :min_n]))
            else:
                trimmed.append(it)
        else:
            if it.shape[0] > min_n:
                trimmed.append(it[:min_n, :min_n])
            else:
                trimmed.append(it)

    return trimmed, kept_mask


def population_recurrence_graph(recurrence_graphs, method='joint', threshold=1.0,
                                 binarize_threshold=None, trim='adaptive',
                                 **trim_kwargs):
    """Combine per-neuron recurrence graphs into a population graph.

    Parameters
    ----------
    recurrence_graphs : list of RecurrenceGraph
        Individual recurrence graphs.
    method : {'joint', 'mean'}
        Combination strategy:

        - ``'joint'``: Thresholded element-wise AND (JRP).  A point (i,j)
          is kept if at least ``threshold`` fraction of neurons recur there.
          ``threshold=1.0`` is strict AND, ``threshold=0.5`` is majority voting.
        - ``'mean'``: Average recurrence matrices.  Values in [0, 1].
          Optionally binarize with ``binarize_threshold``.
    threshold : float, default=1.0
        For method='joint': fraction of neurons that must recur.
    binarize_threshold : float, optional
        For method='mean': if given, binarize the averaged matrix.
    trim : {None, 'min', 'adaptive'}, default='adaptive'
        How to handle graphs of different sizes.  See
        :func:`_reconcile_graph_sizes` for details.
    **trim_kwargs
        Extra parameters passed to :func:`_reconcile_graph_sizes`
        (e.g., ``tail_sigma=3.0`` for adaptive mode).

    Returns
    -------
    RecurrenceGraph
        Population-level recurrence graph (via ``from_adjacency``).

    Raises
    ------
    ValueError
        If graphs have different sizes and ``trim`` is None,
        or method is unknown.
    """
    if not recurrence_graphs:
        raise ValueError("recurrence_graphs must be non-empty")

    valid_methods = ('joint', 'mean')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {valid_methods}.")

    recurrence_graphs, _ = _reconcile_graph_sizes(
        recurrence_graphs, trim=trim, **trim_kwargs)

    n_graphs = len(recurrence_graphs)
    n = recurrence_graphs[0].adj.shape[0]

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


def pairwise_jaccard_sparse(matrices, trim='adaptive', trim_to_min=None,
                            **trim_kwargs):
    """Compute pairwise Jaccard similarity for sparse binary matrices.

    Uses sparse matrix multiplication to compute all pairwise intersection
    counts in a single operation, avoiding O(N^2) Python loops.

    Parameters
    ----------
    matrices : list of scipy.sparse matrices or RecurrenceGraph objects
        Square binary adjacency matrices.  RecurrenceGraph objects are
        accepted (uses ``.adj``).
    trim : {None, 'min', 'adaptive'}, default='adaptive'
        How to handle matrices of different sizes.  See
        :func:`_reconcile_graph_sizes` for details.
    trim_to_min : bool, optional
        **Deprecated in 1.2, will be removed in 2.0.**
        Use ``trim='min'`` instead.  If passed, overrides ``trim``.
    **trim_kwargs
        Extra parameters passed to :func:`_reconcile_graph_sizes`
        (e.g., ``tail_sigma=3.0`` for adaptive mode).

    Returns
    -------
    jaccard : ndarray of shape (N, N)
        Symmetric matrix of Jaccard indices.  Diagonal is 1.0 for
        non-empty matrices, 0.0 for empty ones.  When ``trim='adaptive'``
        removes items, N < len(input).
    kept_mask : ndarray of bool
        Boolean mask of length ``len(input)``.  True for items that were
        kept after reconciliation.

    Raises
    ------
    ValueError
        If list is empty, or if matrices have different shapes and
        ``trim`` is None.
    """
    if not matrices:
        raise ValueError("matrices must be non-empty")

    # Deprecated trim_to_min alias
    if trim_to_min is not None:
        warnings.warn(
            "trim_to_min is deprecated since 1.2 and will be removed in 2.0. "
            "Use trim='min' or trim='adaptive' instead.",
            FutureWarning,
            stacklevel=2,
        )
        trim = 'min' if trim_to_min else None

    # Accept RecurrenceGraph objects
    adjs = [m.adj if hasattr(m, 'adj') else m for m in matrices]

    adjs, kept_mask = _reconcile_graph_sizes(adjs, trim=trim, **trim_kwargs)

    n_matrices = len(adjs)
    n = adjs[0].shape[0]
    n_sq = n * n

    # Flatten each (n, n) matrix into a row of length n^2, stack into (N, n^2)
    rows_list = []
    for m in adjs:
        coo = sp.coo_matrix(m)
        linear_idx = coo.row.astype(np.int64) * n + coo.col.astype(np.int64)
        rows_list.append(sp.csr_matrix(
            (np.ones(len(linear_idx)),
             (np.zeros(len(linear_idx), dtype=int), linear_idx)),
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

    return jaccard, kept_mask
