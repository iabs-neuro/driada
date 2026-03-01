"""Recurrence analysis wrappers for TimeSeries and MultiTimeSeries.

Functions here are delegated to by thin methods on TimeSeries and
MultiTimeSeries in info_base.py.
"""

import numpy as np


def estimate_tau(ts, max_shift=100, method='first_minimum', estimator='gcmi', **kw):
    """Estimate optimal embedding delay for a TimeSeries.

    Caches result as ts._recurrence_tau.
    """
    if ts.discrete:
        raise ValueError("Recurrence analysis requires continuous time series")

    cache_key = (max_shift, method, estimator, tuple(sorted(kw.items())))
    if hasattr(ts, '_recurrence_tau') and ts._recurrence_tau is not None:
        cached_key, cached_val = ts._recurrence_tau
        if cached_key == cache_key:
            return cached_val

    from driada.recurrence.embedding import estimate_tau as _estimate_tau
    tau = _estimate_tau(ts.data, max_shift=max_shift, method=method,
                        estimator=estimator, **kw)
    ts._recurrence_tau = (cache_key, tau)
    return tau


def estimate_embedding_dim(ts, tau=None, max_dim=10, **kw):
    """Estimate embedding dimension for a TimeSeries.

    If tau is None, calls estimate_tau() first. Caches result.
    """
    if ts.discrete:
        raise ValueError("Recurrence analysis requires continuous time series")

    if tau is None:
        tau = estimate_tau(ts)

    cache_key = (tau, max_dim, tuple(sorted(kw.items())))
    if hasattr(ts, '_recurrence_embedding_dim') and ts._recurrence_embedding_dim is not None:
        cached_key, cached_val = ts._recurrence_embedding_dim
        if cached_key == cache_key:
            return cached_val

    from driada.recurrence.embedding import estimate_embedding_dim as _estimate_dim
    m = _estimate_dim(ts.data, tau=tau, max_dim=max_dim, **kw)
    ts._recurrence_embedding_dim = (cache_key, m)
    return m


def takens_embedding(ts, tau=None, m=None):
    """Compute Takens delay embedding for a TimeSeries.

    Auto-estimates tau and m if None.
    """
    if ts.discrete:
        raise ValueError("Recurrence analysis requires continuous time series")

    if tau is None:
        tau = estimate_tau(ts)
    if m is None:
        m = estimate_embedding_dim(ts, tau=tau)

    from driada.recurrence.embedding import takens_embedding as _embed
    return _embed(ts.data, tau=tau, m=m)


def recurrence_graph(ts, tau=None, m=None, method='knn', k=5,
                     epsilon=None, metric='euclidean', theiler_window='auto'):
    """Build RecurrenceGraph for a TimeSeries. Caches result."""
    if ts.discrete:
        raise ValueError("Recurrence analysis requires continuous time series")

    if tau is None:
        tau = estimate_tau(ts)
    if m is None:
        m = estimate_embedding_dim(ts, tau=tau)

    if theiler_window == 'auto':
        theiler_window = tau * (m - 1) + 1

    cache_key = (tau, m, method, k, epsilon, metric, theiler_window)
    if hasattr(ts, '_recurrence_graph_cache') and ts._recurrence_graph_cache is not None:
        cached_key, cached_val = ts._recurrence_graph_cache
        if cached_key == cache_key:
            return cached_val

    emb = takens_embedding(ts, tau=tau, m=m)

    from driada.recurrence.recurrence_graph import RecurrenceGraph
    rg = RecurrenceGraph(
        emb, method=method, k=k, epsilon=epsilon, metric=metric,
        theiler_window=theiler_window,
    )
    ts._recurrence_graph_cache = (cache_key, rg)
    return rg


def rqa(ts, tau=None, m=None, l_min=2, v_min=2, **rg_kwargs):
    """Compute RQA measures for a TimeSeries."""
    rg = recurrence_graph(ts, tau=tau, m=m, **rg_kwargs)
    return rg.rqa(l_min=l_min, v_min=v_min)


def population_recurrence_graph(mts, tau=None, m=None, method='joint',
                                 threshold=1.0, binarize_threshold=None,
                                 rg_method='knn', k=5, epsilon=None,
                                 metric='euclidean', theiler_window='auto',
                                 n_jobs=-1, verbose=False):
    """Build population recurrence graph from a MultiTimeSeries.

    Constructs per-component recurrence graphs (parallelized), then combines.
    Individual graphs are cached on each ts_list entry.
    """
    if mts.discrete:
        raise ValueError("Recurrence analysis requires continuous time series")

    from driada.recurrence.population import population_recurrence_graph as _pop_rg
    from driada.utils.parallel import parallel_executor
    from joblib import delayed

    def _build_one(ts_component):
        return recurrence_graph(
            ts_component, tau=tau, m=m, method=rg_method, k=k,
            epsilon=epsilon, metric=metric, theiler_window=theiler_window,
        )

    # Force threading backend: NumPy/scipy release GIL, and threading
    # allows cache writes to propagate back to ts_list entries (shared memory).
    # Loky/multiprocessing serialize copies, losing cache updates.
    with parallel_executor(n_jobs, verbose=verbose, backend='threading') as parallel:
        graphs = parallel(
            delayed(_build_one)(ts_comp) for ts_comp in mts.ts_list
        )

    return _pop_rg(graphs, method=method, threshold=threshold,
                   binarize_threshold=binarize_threshold)


def visibility_graph(ts, method='horizontal', directed=False):
    """Build VisibilityGraph for a TimeSeries. Caches result."""
    if ts.discrete:
        raise ValueError("Visibility graph requires continuous time series")

    cache_key = (method, directed)
    if hasattr(ts, '_vg_cache') and ts._vg_cache is not None:
        cached_key, cached_val = ts._vg_cache
        if cached_key == cache_key:
            return cached_val

    from driada.recurrence.visibility import VisibilityGraph
    vg = VisibilityGraph(ts.data, method=method, directed=directed)
    ts._vg_cache = (cache_key, vg)
    return vg


def ordinal_partition_network(ts, d=None, tau=None):
    """Build OrdinalPartitionNetwork for a TimeSeries. Caches result.

    Auto-estimates tau and d if None. d is capped at 7.
    """
    if ts.discrete:
        raise ValueError("OPN requires continuous time series")

    if tau is None:
        tau = estimate_tau(ts)
    if d is None:
        d = min(estimate_embedding_dim(ts, tau=tau), 7)

    cache_key = (d, tau)
    if hasattr(ts, '_opn_cache') and ts._opn_cache is not None:
        cached_key, cached_val = ts._opn_cache
        if cached_key == cache_key:
            return cached_val

    from driada.recurrence.opn import OrdinalPartitionNetwork
    opn = OrdinalPartitionNetwork(ts.data, d=d, tau=tau)
    ts._opn_cache = (cache_key, opn)
    return opn


def permutation_entropy(ts, d=None, tau=None):
    """Compute permutation entropy for a TimeSeries."""
    opn = ordinal_partition_network(ts, d=d, tau=tau)
    return opn.permutation_entropy
