"""Visibility graph from 1D time series."""

import numpy as np
import scipy.sparse as sp
from ..network.net_base import Network


def _build_hvg(data):
    """Build Horizontal Visibility Graph adjacency. O(N) via stack.

    Two points i, j are connected if all intermediate values
    x_k < min(x_i, x_j) for i < k < j (Luque et al. 2009).

    Uses a monotone stack: process points left to right, maintaining
    a stack of indices with non-increasing values. Each point connects
    to the stack top (nearest left point >= it) and to each popped
    element (which it dominates).
    """
    n = len(data)
    rows, cols = [], []

    stack = []
    for i in range(n):
        while stack and data[stack[-1]] < data[i]:
            j = stack.pop()
            rows.append(j)
            cols.append(i)
        if stack:
            rows.append(stack[-1])
            cols.append(i)
        stack.append(i)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    ones = np.ones(len(rows), dtype=np.float64)

    # Symmetrize: add both (i,j) and (j,i)
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_data = np.concatenate([ones, ones])

    adj = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(n, n))
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    return adj


def _build_nvg(data):
    """Build Natural Visibility Graph adjacency. O(N^2).

    Two points i, j are connected if for all k with i < k < j:
        (x_k - x_i) / (k - i) < (x_j - x_i) / (j - i)
    """
    n = len(data)
    rows, cols = [], []

    for i in range(n - 1):
        for j in range(i + 1, n):
            slope_ij = (data[j] - data[i]) / (j - i)
            visible = True
            for k in range(i + 1, j):
                slope_ik = (data[k] - data[i]) / (k - i)
                if slope_ik >= slope_ij:
                    visible = False
                    break
            if visible:
                rows.append(i)
                cols.append(j)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    ones = np.ones(len(rows), dtype=np.float64)

    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_data = np.concatenate([ones, ones])

    adj = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(n, n))
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    return adj


class VisibilityGraph(Network):
    """Visibility graph from 1D time series.

    Two time points are connected if no intermediate value blocks
    the line of sight between them (NVG), or if all intermediate
    values are below both endpoints (HVG).

    Parameters
    ----------
    data : array-like, 1D
        Raw time series values.
    method : {'horizontal', 'natural'}, default='horizontal'
        'horizontal' (HVG): O(N) via monotone stack. Default.
        'natural' (NVG): O(N^2) pairwise line-of-sight.
    directed : bool, default=False
        If True, edges point forward in time only.
    """

    def __init__(self, data, method='horizontal', directed=False):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(
                f"Data must be 1D time series, got {data.ndim}D."
            )

        if method == 'horizontal':
            adj = _build_hvg(data)
        elif method == 'natural':
            adj = _build_nvg(data)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'horizontal' or 'natural'."
            )

        if directed:
            adj = sp.triu(adj, k=1, format='csr')

        Network.__init__(
            self, adj=adj, preprocessing=None,
            create_nx_graph=False, directed=directed,
        )
        self._data = data
        self._method = method

    @property
    def timeseries_data(self):
        """Original time series data."""
        return self._data

    @property
    def vg_method(self):
        """Construction method ('horizontal' or 'natural')."""
        return self._method
