"""Recurrence graph from delay-embedded time series."""

import numpy as np
import scipy.sparse as sp
from ..dim_reduction.graph import ProximityGraph


class RecurrenceGraph(ProximityGraph):
    """Recurrence graph constructed from delay-embedded time series data.

    Extends ProximityGraph (-> Network) with Theiler window removal and
    recurrence quantification analysis (RQA). Inherits spectral analysis,
    entropy, degree distribution, and randomization from Network.

    Parameters
    ----------
    data : ndarray of shape (m, N_embedded)
        Pre-embedded data matrix. Use ``takens_embedding()`` first.
    method : {'knn', 'eps'}, default='knn'
        Graph construction method.
    k : int, default=5
        Number of neighbors for k-NN.
    epsilon : float, optional
        Radius for epsilon-ball.
    metric : str, default='euclidean'
        Distance metric.
    theiler_window : int or None, optional
        Remove entries where |i-j| < theiler_window.
    verbose : bool, default=False
        Print progress.

    Raises
    ------
    ValueError
        If data is not 2D.
    """

    def __init__(
        self,
        data,
        method='knn',
        k=5,
        epsilon=None,
        metric='euclidean',
        theiler_window=None,
        verbose=False,
    ):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(
                f"Data must be 2D array of shape (m, N_embedded), got {data.ndim}D. "
                "Use takens_embedding() to embed a 1D time series first."
            )

        m_params = {'metric_name': metric}

        if method == 'knn':
            g_params = {
                'g_method_name': 'knn',
                'nn': k,
                'weighted': False,
                'graph_preprocessing': None,
                'max_deleted_nodes': 0.0,
            }
        elif method == 'eps':
            if epsilon is None:
                raise ValueError("epsilon must be specified for method='eps'")
            g_params = {
                'g_method_name': 'eps',
                'eps': epsilon,
                'weighted': False,
                'graph_preprocessing': None,
                'min_density': 1e-6,
                'max_deleted_nodes': 0.0,
            }
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'knn' or 'eps'.")

        # Store Theiler window BEFORE calling super().__init__(), because
        # super().__init__() calls self.construct_adjacency() via dynamic
        # dispatch — our override (see below) reads _theiler_window to apply
        # the band removal at the right time in the initialization sequence.
        self._theiler_window = theiler_window

        super().__init__(data, m_params, g_params, create_nx_graph=False, verbose=verbose)

        self._rqa_cache = None

    def construct_adjacency(self):
        """Build adjacency matrix with Theiler window applied before Network init.

        Overrides ProximityGraph.construct_adjacency() to inject Theiler window
        removal at the correct point in the initialization sequence.

        ProximityGraph.__init__ runs these steps in order:
          1. self.construct_adjacency()   — builds adj, bin_adj
          2. Network.__init__(adj=...)    — computes degree sequences, etc.
          3. self._checkpoint()           — validates symmetry

        By overriding step 1 to also apply the Theiler window, Network.__init__
        at step 2 sees the already-filtered adjacency. This ensures degree
        sequences, spectral analysis, and all other Network properties are
        computed on the correct matrix — no stale state from post-init mutation.
        """
        super().construct_adjacency()
        if self._theiler_window is not None and self._theiler_window > 0:
            self._apply_theiler_window(self._theiler_window)

    def _apply_theiler_window(self, window):
        """Remove recurrence points within |i-j| < window from adjacency."""
        adj_coo = self.adj.tocoo()
        mask = np.abs(adj_coo.row - adj_coo.col) >= window
        self.adj = sp.csr_matrix(
            (adj_coo.data[mask], (adj_coo.row[mask], adj_coo.col[mask])),
            shape=self.adj.shape,
        )
        if hasattr(self, 'bin_adj') and self.bin_adj is not None:
            bin_coo = self.bin_adj.tocoo()
            mask_bin = np.abs(bin_coo.row - bin_coo.col) >= window
            self.bin_adj = sp.csr_matrix(
                (bin_coo.data[mask_bin], (bin_coo.row[mask_bin], bin_coo.col[mask_bin])),
                shape=self.bin_adj.shape,
            )

    @property
    def theiler_window(self):
        """Theiler window applied to this recurrence graph."""
        return self._theiler_window

    @property
    def recurrence_rate(self):
        """Fraction of recurrence points in the matrix."""
        n = self.adj.shape[0]
        total = n * n
        if self._theiler_window is not None and self._theiler_window > 0:
            band_size = 0
            for d in range(self._theiler_window):
                band_size += 2 * (n - d) if d > 0 else n
            total -= band_size
        if total <= 0:
            return 0.0
        return self.adj.nnz / total

    @classmethod
    def from_adjacency(cls, adj, theiler_window=None):
        """Create RecurrenceGraph from pre-built adjacency matrix.

        Bypasses ProximityGraph construction. Used for population
        recurrence graphs or loaded matrices.

        Parameters
        ----------
        adj : scipy.sparse matrix
            Square adjacency matrix.
        theiler_window : int or None
            Theiler window to record (not applied).

        Returns
        -------
        RecurrenceGraph
        """
        instance = object.__new__(cls)
        instance._theiler_window = theiler_window
        instance._rqa_cache = None
        from ..network.net_base import Network
        Network.__init__(
            instance, adj=adj, preprocessing=None,
            create_nx_graph=False, directed=False,
        )
        instance.data = None
        instance.lost_nodes = set()
        instance.bin_adj = adj.copy()
        instance.neigh_distmat = None
        instance.knn_indices = None
        instance.knn_distances = None
        instance.metric = None
        return instance

    def rqa(self, l_min=2, v_min=2):
        """Compute recurrence quantification analysis measures.

        Parameters
        ----------
        l_min : int, default=2
            Minimum diagonal line length.
        v_min : int, default=2
            Minimum vertical line length.

        Returns
        -------
        dict
            RQA measures. See ``compute_rqa`` for keys.
        """
        if self._rqa_cache is not None:
            cached_params, cached_result = self._rqa_cache
            if cached_params == (l_min, v_min):
                return cached_result

        from .rqa import compute_rqa
        result = compute_rqa(self.adj, l_min=l_min, v_min=v_min)
        result['RR'] = self.recurrence_rate
        self._rqa_cache = ((l_min, v_min), result)
        return result
