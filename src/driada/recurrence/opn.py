"""Ordinal partition network from 1D time series."""

import math
import numpy as np
import scipy.sparse as sp
from ..network.net_base import Network
from .embedding import takens_embedding


_MAX_D = 7


def _encode_patterns(ranks, d):
    """Encode ordinal rank patterns as Lehmer codes.

    Parameters
    ----------
    ranks : ndarray of shape (d, N_embedded)
        Argsort of each embedded vector (column-wise ranks).
    d : int
        Embedding dimension.

    Returns
    -------
    pattern_ids : ndarray of shape (N_embedded,), dtype int32
        Integer ID for each ordinal pattern. Range [0, d!).
    """
    n = ranks.shape[1]
    ids = np.zeros(n, dtype=np.int64)
    for i in range(d):
        # Lehmer code: count how many elements after position i
        # have a smaller rank value
        count = np.zeros(n, dtype=np.int64)
        for j in range(i + 1, d):
            count += (ranks[j] < ranks[i]).astype(np.int64)
        ids = ids * (d - i) + count
    return ids.astype(np.int32)


def _build_transition_matrix(pattern_ids, d):
    """Build directed weighted transition matrix between ordinal patterns.

    Parameters
    ----------
    pattern_ids : 1D int array of length N_embedded
        Pattern IDs from _encode_patterns.
    d : int
        Embedding dimension.

    Returns
    -------
    adj : sparse CSR matrix of shape (n_states, n_states)
        Row-normalized transition probabilities.
    """
    n_states = math.factorial(d)
    n = len(pattern_ids)

    if n < 2:
        return sp.csr_matrix((n_states, n_states))

    # Count transitions
    src = pattern_ids[:-1]
    dst = pattern_ids[1:]
    data = np.ones(len(src), dtype=np.float64)

    adj = sp.coo_matrix((data, (src, dst)), shape=(n_states, n_states))
    adj = adj.tocsr()

    # Remove self-loops before normalizing — Network base class strips
    # diagonal entries, so we exclude them here to keep row sums = 1.
    adj.setdiag(0)
    adj.eliminate_zeros()

    # Row-normalize to get transition probabilities
    row_sums = np.array(adj.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    diag_inv = sp.diags(1.0 / row_sums)
    adj = diag_inv @ adj

    return adj


class OrdinalPartitionNetwork(Network):
    """Ordinal partition network from 1D time series.

    Embeds with Takens delay embedding, ranks elements to get
    ordinal patterns. Nodes = unique patterns, directed weighted
    edges = transition probabilities between consecutive patterns.

    Parameters
    ----------
    data : array-like, 1D
        Raw time series values.
    d : int
        Embedding dimension (pattern length). Must be <= 7.
    tau : int
        Embedding delay.
    """

    def __init__(self, data, d, tau):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(
                f"Data must be 1D time series, got {data.ndim}D."
            )
        if d > _MAX_D:
            raise ValueError(
                f"Embedding dimension d={d} too large (d! = {math.factorial(d)} "
                f"nodes). Must be <= {_MAX_D}."
            )
        if d < 2:
            raise ValueError(f"Embedding dimension d must be >= 2, got {d}.")

        emb = takens_embedding(data, tau=tau, m=d)
        ranks = np.argsort(emb, axis=0)
        pattern_ids = _encode_patterns(ranks, d)
        adj = _build_transition_matrix(pattern_ids, d)

        Network.__init__(
            self, adj=adj, preprocessing=None,
            create_nx_graph=False, directed=True,
        )
        self._data = data
        self.d = d
        self.tau = tau
        self._pattern_ids = pattern_ids

    @property
    def permutation_entropy(self):
        """Permutation entropy normalized to [0, 1].

        Shannon entropy of pattern visit frequencies divided by log2(d!).
        High (~1) = complex/random. Low (~0) = regular/periodic.
        """
        n_states = math.factorial(self.d)
        freqs = np.bincount(self._pattern_ids, minlength=n_states)
        freqs = freqs[freqs > 0].astype(np.float64)
        freqs = freqs / freqs.sum()
        h = -np.sum(freqs * np.log2(freqs))
        h_max = np.log2(n_states)
        if h_max == 0:
            return 0.0
        return h / h_max

    @property
    def missing_patterns(self):
        """Number of d! ordinal patterns never visited."""
        n_states = math.factorial(self.d)
        return n_states - len(np.unique(self._pattern_ids))

    @property
    def timeseries_data(self):
        """Original time series data."""
        return self._data
