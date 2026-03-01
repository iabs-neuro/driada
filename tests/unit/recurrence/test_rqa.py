"""Tests for RQA computation."""

import numpy as np
import pytest
import scipy.sparse as sp
from driada.recurrence.rqa import compute_rqa


def _make_diagonal_matrix(n, diag_length):
    """Create sparse matrix with a single diagonal line of given length."""
    rows = list(range(diag_length))
    cols = list(range(diag_length))
    data = [1] * diag_length
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _make_vertical_matrix(n, vert_length, col=0):
    """Create sparse matrix with a vertical line of given length."""
    rows = list(range(vert_length))
    cols = [col] * vert_length
    data = [1] * vert_length
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


class TestComputeRQA:
    """Tests for compute_rqa()."""

    def test_single_diagonal(self):
        """Matrix with one diagonal line of length 5."""
        adj = _make_diagonal_matrix(20, 5)
        rqa = compute_rqa(adj, l_min=2, v_min=2)
        assert rqa['DET'] == 1.0  # All points are in diagonal line
        assert rqa['L_max'] == 5
        assert rqa['L_mean'] == 5.0

    def test_single_vertical(self):
        """Matrix with one vertical line of length 5."""
        adj = _make_vertical_matrix(20, 5)
        rqa = compute_rqa(adj, l_min=2, v_min=2)
        assert rqa['LAM'] == 1.0  # All points are in vertical line
        assert rqa['TT'] == 5.0

    def test_identity_matrix(self):
        """Identity matrix: all isolated points on main diagonal -> one long diagonal line."""
        adj = sp.eye(20, format='csr')
        rqa = compute_rqa(adj, l_min=2, v_min=2)
        # Main diagonal is one long line of length 20
        assert rqa['DET'] == 1.0
        assert rqa['L_max'] == 20

    def test_empty_matrix(self):
        """Empty matrix should return zero measures."""
        adj = sp.csr_matrix((20, 20))
        rqa = compute_rqa(adj, l_min=2, v_min=2)
        assert rqa['DET'] == 0.0
        assert rqa['LAM'] == 0.0
        assert rqa['L_max'] == 0
        assert rqa['TT'] == 0.0

    def test_periodic_signal_high_det(self):
        """Periodic signal should have high DET.

        Uses k=30 so that enough recurrence points appear on off-diagonal
        bands to form long diagonal lines.  With k=5 the k-NN graph of a
        low-dimensional attractor produces very few diagonals.
        """
        from driada.recurrence import takens_embedding, RecurrenceGraph
        t = np.arange(500)
        data = np.sin(2 * np.pi * t / 40)
        emb = takens_embedding(data, tau=10, m=3)
        rg = RecurrenceGraph(emb, method='knn', k=30, theiler_window=2)
        rqa = rg.rqa()
        assert rqa['DET'] > 0.6

    def test_noise_low_det(self):
        """White noise should have low DET."""
        from driada.recurrence import takens_embedding, RecurrenceGraph
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        emb = takens_embedding(data, tau=1, m=3)
        rg = RecurrenceGraph(emb, method='knn', k=5, theiler_window=3)
        rqa = rg.rqa()
        assert rqa['DET'] < 0.5, f"Noise DET should be low, got {rqa['DET']:.3f}"

    def test_all_keys_present(self):
        """All standard RQA keys must be present."""
        adj = _make_diagonal_matrix(20, 10)
        rqa = compute_rqa(adj, l_min=2, v_min=2)
        expected_keys = {'DET', 'L_mean', 'L_max', 'DIV', 'ENTR', 'LAM', 'TT',
                         'diagonal_lines', 'vertical_lines'}
        assert expected_keys.issubset(rqa.keys())

    def test_l_min_filtering(self):
        """Lines shorter than l_min should be excluded from DET."""
        # Create matrix with two diagonal lines: length 3 and length 1
        rows = [0, 1, 2, 10]
        cols = [0, 1, 2, 10]
        adj = sp.csr_matrix(([1, 1, 1, 1], (rows, cols)), shape=(20, 20))
        rqa = compute_rqa(adj, l_min=2)
        # Only the length-3 line counts for DET
        assert rqa['DET'] == 3 / 4  # 3 of 4 points in lines >= 2
