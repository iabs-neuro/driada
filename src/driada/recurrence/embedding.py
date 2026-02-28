"""Takens delay embedding and parameter estimation for recurrence analysis."""

import numpy as np


def takens_embedding(data, tau, m):
    """Construct time-delay embedding matrix.

    Given a 1D time series x of length N, constructs the matrix::

        [[x[0],        x[1],        ..., x[N_e-1]       ],
         [x[tau],      x[tau+1],    ..., x[tau+N_e-1]   ],
         ...
         [x[(m-1)*tau], ...              x[N-1]          ]]

    where N_e = N - (m-1)*tau.

    Parameters
    ----------
    data : array-like, 1D
        Time series of length N.
    tau : int
        Time delay in samples. Must be >= 1.
    m : int
        Embedding dimension. Must be >= 2.

    Returns
    -------
    ndarray of shape (m, N_embedded)
        Delay-embedded matrix. Rows are embedding dimensions,
        columns are time points. Matches ProximityGraph (features, samples) convention.

    Raises
    ------
    ValueError
        If data is not 1D, or series is too short for given tau and m.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"Data must be 1D array, got {data.ndim}D with shape {data.shape}")

    n = len(data)
    n_embedded = n - (m - 1) * tau
    if n_embedded < 1:
        raise ValueError(
            f"Time series of length {n} is too short for tau={tau}, m={m}. "
            f"Need at least {(m - 1) * tau + 1} points."
        )

    indices = np.arange(n_embedded)[np.newaxis, :] + (np.arange(m) * tau)[:, np.newaxis]
    return data[indices]
