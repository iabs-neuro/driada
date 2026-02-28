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


def estimate_tau(data, max_shift=100, method='first_minimum', estimator='gcmi',
                 **estimator_kwargs):
    """Estimate optimal time delay for Takens embedding.

    Parameters
    ----------
    data : array-like, 1D
        Time series.
    max_shift : int, optional
        Maximum lag to evaluate. Default: 100.
    method : {'first_minimum', 'exponential_fit'}
        - 'first_minimum': First local minimum of time-delayed mutual information.
        - 'exponential_fit': Fit exponential decay to TDMI, return -1/slope.
    estimator : {'gcmi', 'ksg'}, optional
        MI estimator passed to get_tdmi(). Default: 'gcmi'.
    **estimator_kwargs
        Additional kwargs passed to get_tdmi() (e.g., nn=5 for KSG).

    Returns
    -------
    int
        Optimal time delay in samples. Always >= 1.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    from driada.information.info_base import get_tdmi

    valid_methods = ('first_minimum', 'exponential_fit')
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {valid_methods}."
        )

    # get_tdmi max_shift is exclusive, so pass max_shift+1 to include max_shift
    tdmi = get_tdmi(data, min_shift=1, max_shift=max_shift + 1,
                    estimator=estimator, **estimator_kwargs)
    tdmi = np.array(tdmi)

    if method == 'first_minimum':
        return _tau_first_minimum(tdmi)
    else:
        return _tau_exponential_fit(tdmi)


def _tau_first_minimum(tdmi):
    """Find first local minimum of TDMI curve.

    Parameters
    ----------
    tdmi : ndarray
        Time-delayed MI values. tdmi[0] corresponds to shift=1.

    Returns
    -------
    int
        Time delay at the first local minimum (>= 1).
    """
    if len(tdmi) < 2:
        return 1

    for i in range(1, len(tdmi) - 1):
        if tdmi[i] < tdmi[i - 1] and tdmi[i] <= tdmi[i + 1]:
            return i + 1  # +1 because tdmi[0] corresponds to shift=1
    # No minimum found — return shift where TDMI drops to 1/e of initial
    threshold = tdmi[0] / np.e
    below = np.where(tdmi < threshold)[0]
    if len(below) > 0:
        return int(below[0]) + 1
    return max(1, len(tdmi) // 3)


def _tau_exponential_fit(tdmi):
    """Estimate tau from TDMI via exponential-decay analogy.

    For a pure exponential I(s) = I(0)*exp(-s/tau), the integral of the
    normalised curve from 0 to infinity equals tau.  We approximate this
    by trapezoidal integration of the normalised TDMI over its initial
    monotonically-decreasing segment.

    If log-linear fitting is feasible (>= 3 positive values in the
    decreasing segment), uses ``-1/slope`` from a least-squares fit;
    otherwise falls back to the integral approach.

    Parameters
    ----------
    tdmi : ndarray
        Time-delayed MI values. tdmi[0] corresponds to shift=1.

    Returns
    -------
    int
        Time delay estimated from exponential fit (>= 1).
    """
    tdmi = np.array(tdmi, dtype=float)
    if len(tdmi) == 0 or tdmi[0] <= 0:
        return 1

    # Find the initial monotonically decreasing segment
    dec_end = len(tdmi)
    for i in range(1, len(tdmi)):
        if tdmi[i] >= tdmi[i - 1]:
            dec_end = i
            break

    # Normalise and integrate (area-under-curve = tau for true exponential)
    segment = tdmi[:dec_end]
    normalised = segment / tdmi[0]
    shifts = np.arange(1, dec_end + 1)
    # Prepend the implicit point (shift=0, normalised=1.0)
    all_shifts = np.concatenate([[0], shifts])
    all_norm = np.concatenate([[1.0], normalised])
    tau = int(np.round(np.trapezoid(all_norm, all_shifts)))
    return max(1, tau)


def estimate_embedding_dim(data, tau, max_dim=10, r_tol=10.0, a_tol=2.0,
                           fnn_threshold=0.01):
    """Estimate embedding dimension via false nearest neighbors (FNN).

    For each candidate dimension *m* (from 2 to *max_dim*), embeds the time
    series, finds each point's nearest neighbor, and checks whether that
    neighbor is still close in dimension *m* + 1.  "False" neighbors appear
    close only because the attractor is projected into too few dimensions.

    Parameters
    ----------
    data : array-like, 1D
        Time series.
    tau : int
        Time delay for embedding.
    max_dim : int, optional
        Maximum dimension to test.  Default: 10.
    r_tol : float, optional
        Distance-ratio threshold (Kennel *et al.* 1992).  Default: 10.0.
    a_tol : float, optional
        Absolute threshold relative to attractor size.  Default: 2.0.
    fnn_threshold : float, optional
        FNN fraction below which the dimension is accepted.
        Default: 0.01 (1 %).

    Returns
    -------
    int
        Estimated embedding dimension.  Minimum 2.
    """
    from scipy.spatial import cKDTree

    data = np.asarray(data, dtype=float).ravel()
    attractor_size = np.std(data)
    if attractor_size == 0:
        return 2

    # Minimum meaningful distance: pairs closer than this are considered
    # true (deterministic) neighbors and excluded from the FNN ratio test.
    dist_tol = attractor_size * 1e-8

    for m in range(2, max_dim + 1):
        emb_m = takens_embedding(data, tau, m).T       # (N_embedded, m)
        emb_m1 = takens_embedding(data, tau, m + 1).T  # (N_embedded_m1, m+1)

        n_m1 = emb_m1.shape[0]
        emb_m_trimmed = emb_m[:n_m1]

        tree = cKDTree(emb_m_trimmed)
        dists, indices = tree.query(emb_m_trimmed, k=2)

        nn_dists_m = dists[:, 1]       # distance to nearest neighbor
        nn_indices = indices[:, 1]     # index of nearest neighbor

        # Euclidean distance in m+1 dimensions
        nn_dists_m1 = np.linalg.norm(
            emb_m1 - emb_m1[nn_indices], axis=1
        )

        # Points with near-zero NN distance are deterministic repeats
        # (e.g. periodic signals); exclude them from the ratio criterion.
        valid = nn_dists_m > dist_tol
        if not np.any(valid):
            # All neighbors are true neighbors — no false neighbors at dim m
            return m

        ratio = np.zeros(n_m1)
        ratio[valid] = (
            np.abs(nn_dists_m1[valid] - nn_dists_m[valid]) / nn_dists_m[valid]
        )

        criterion1 = ratio > r_tol
        criterion2 = (nn_dists_m1 / attractor_size) > a_tol
        fnn_fraction = np.sum(criterion1 | criterion2) / n_m1

        if fnn_fraction < fnn_threshold:
            return m

    return max_dim
