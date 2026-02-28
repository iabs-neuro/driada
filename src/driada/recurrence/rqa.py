"""Recurrence Quantification Analysis (RQA) for sparse recurrence matrices."""

import numpy as np
import scipy.sparse as sp

try:
    from ._numba_rqa import (
        scan_diagonal_lines_numba,
        scan_vertical_lines_numba,
        HAS_NUMBA,
    )
except ImportError:
    HAS_NUMBA = False


def compute_rqa(adj, l_min=2, v_min=2):
    """Compute RQA measures from a sparse recurrence matrix.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Binary recurrence matrix.
    l_min : int, default=2
        Minimum diagonal line length.
    v_min : int, default=2
        Minimum vertical line length.

    Returns
    -------
    dict
        RQA measures: DET, L_mean, L_max, DIV, ENTR, LAM, TT,
        diagonal_lines, vertical_lines.
    """
    adj = sp.coo_matrix(adj)
    n_recurrence = adj.nnz

    diagonal_lines = _scan_diagonal_lines(adj)
    vertical_lines = _scan_vertical_lines(adj)

    diag_filtered = [l for l in diagonal_lines if l >= l_min]
    vert_filtered = [l for l in vertical_lines if l >= v_min]

    diag_points = sum(diag_filtered)
    det = diag_points / n_recurrence if n_recurrence > 0 else 0.0

    l_max = max(diagonal_lines) if diagonal_lines else 0
    l_mean = float(np.mean(diag_filtered)) if diag_filtered else 0.0
    div = 1.0 / l_max if l_max > 0 else 0.0

    if diag_filtered:
        lengths, counts = np.unique(diag_filtered, return_counts=True)
        probs = counts / counts.sum()
        entr = -np.sum(probs * np.log(probs))
    else:
        entr = 0.0

    vert_points = sum(vert_filtered)
    lam = vert_points / n_recurrence if n_recurrence > 0 else 0.0

    tt = float(np.mean(vert_filtered)) if vert_filtered else 0.0

    return {
        'DET': det,
        'L_mean': l_mean,
        'L_max': l_max,
        'DIV': div,
        'ENTR': entr,
        'LAM': lam,
        'TT': tt,
        'diagonal_lines': diagonal_lines,
        'vertical_lines': vertical_lines,
    }


def _scan_diagonal_lines(adj_coo):
    """Extract diagonal line lengths from sparse COO matrix."""
    if adj_coo.nnz == 0:
        return []

    if HAS_NUMBA and adj_coo.nnz > 1000:
        lines, n = scan_diagonal_lines_numba(
            adj_coo.row.astype(np.int32),
            adj_coo.col.astype(np.int32),
            adj_coo.shape[0],
        )
        return lines[:n].tolist()

    row = adj_coo.row
    col = adj_coo.col
    diag_idx = col - row

    unique_diags = np.unique(diag_idx)
    all_lines = []

    for d in unique_diags:
        mask = diag_idx == d
        positions = np.sort(row[mask])
        all_lines.extend(_runs_of_consecutive(positions))

    return all_lines


def _scan_vertical_lines(adj_coo):
    """Extract vertical line lengths from sparse matrix."""
    if adj_coo.nnz == 0:
        return []

    adj_csc = sp.csc_matrix(adj_coo)

    if HAS_NUMBA and adj_coo.nnz > 1000:
        lines, n = scan_vertical_lines_numba(
            adj_csc.indptr.astype(np.int32),
            adj_csc.indices.astype(np.int32),
            adj_csc.shape[1],
        )
        return lines[:n].tolist()

    all_lines = []

    for j in range(adj_csc.shape[1]):
        col_rows = adj_csc.indices[adj_csc.indptr[j]:adj_csc.indptr[j + 1]]
        if len(col_rows) > 0:
            all_lines.extend(_runs_of_consecutive(col_rows))

    return all_lines


def _runs_of_consecutive(sorted_positions):
    """Find lengths of runs of consecutive integers."""
    if len(sorted_positions) == 0:
        return []

    diffs = np.diff(sorted_positions)
    breaks = np.where(diffs != 1)[0]

    lengths = []
    start = 0
    for b in breaks:
        lengths.append(b - start + 1)
        start = b + 1
    lengths.append(len(sorted_positions) - start)

    return lengths
