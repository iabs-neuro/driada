"""Optional Numba-accelerated RQA scanning functions."""

try:
    import numba
    import numpy as np

    @numba.njit
    def scan_diagonal_lines_numba(rows, cols, n):
        """Numba-accelerated diagonal line scanning.

        Parameters
        ----------
        rows : int32 array — COO row indices
        cols : int32 array — COO col indices
        n : int — matrix dimension

        Returns
        -------
        lines : int32 array — line lengths (padded with zeros)
        n_lines : int — number of valid entries
        """
        n_entries = len(rows)
        diag_idx = cols - rows
        positions = rows

        # Sort by (diag_idx, position)
        order = np.argsort(diag_idx * n + positions)

        lines = np.zeros(n_entries, dtype=numba.int32)
        n_lines = 0
        run_len = 1

        for i in range(1, n_entries):
            prev = order[i - 1]
            curr = order[i]
            same_diag = diag_idx[prev] == diag_idx[curr]
            consecutive = positions[curr] == positions[prev] + 1

            if same_diag and consecutive:
                run_len += 1
            else:
                lines[n_lines] = run_len
                n_lines += 1
                run_len = 1

        lines[n_lines] = run_len
        n_lines += 1

        return lines, n_lines

    @numba.njit
    def scan_vertical_lines_numba(indptr, indices, n_cols):
        """Numba-accelerated vertical line scanning from CSC format."""
        lines = np.zeros(len(indices), dtype=numba.int32)
        n_lines = 0

        for j in range(n_cols):
            start = indptr[j]
            end = indptr[j + 1]
            if start == end:
                continue

            run_len = 1
            for i in range(start + 1, end):
                if indices[i] == indices[i - 1] + 1:
                    run_len += 1
                else:
                    lines[n_lines] = run_len
                    n_lines += 1
                    run_len = 1

            lines[n_lines] = run_len
            n_lines += 1

        return lines, n_lines

    HAS_NUMBA = True

except ImportError:
    HAS_NUMBA = False
