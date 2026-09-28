"""Calcium signal preprocessing utilities.

This module provides preprocessing functions for calcium imaging signals,
including noise addition for numerical stability and negative value clipping.
"""

import numpy as np
from ..utils.jit import conditional_njit


# Statistical constants
MAD_SCALE_FACTOR = 1.4826  # Scaling factor for MAD → std consistency (normal distribution)
# This is 1 / (sqrt(2) * erfcinv(1.5))


def calcium_preprocessing(ca, seed=None):
    """Preprocess calcium signal for spike reconstruction.

    Applies preprocessing steps:
    - Converts to float64 for numerical stability
    - Clips negative values to 0 (calcium cannot be negative)
    - Adds tiny noise to prevent numerical singularities

    Parameters
    ----------
    ca : array-like
        Raw calcium signal. Must be 1D.
    seed : int, numpy.random.SeedSequence or numpy.random.Generator, optional
        Seed for the noise, passed to ``numpy.random.default_rng``. With the
        same seed the output is identical between calls and processes. If
        None, fresh entropy is used and the noise differs between calls.

    Returns
    -------
    ndarray
        Preprocessed calcium signal as float64 array.

    Raises
    ------
    ValueError
        If ca is empty.

    Notes
    -----
    The small noise (1e-8 scale) prevents division by zero and other
    numerical issues in downstream spike reconstruction algorithms. It also
    breaks ties between equal values (e.g. clipped zeros), so it changes the
    ranks used by copula-based mutual information; pass a seed when results
    must be reproducible.

    The noise is drawn outside the Numba-compiled part: Numba keeps its own
    random state, which ``np.random.seed`` called from Python does not reach.

    Examples
    --------
    >>> ca = np.array([1.0, -0.5, 2.0, 0.5])
    >>> processed = calcium_preprocessing(ca, seed=42)
    >>> processed[1]  # Negative value clipped to ~0
    0.0...
    >>> (processed > 0).all()  # All values positive after noise
    True
    """
    ca = np.asarray(ca)
    if ca.size == 0:
        raise ValueError("Calcium signal cannot be empty")
    ca = _calcium_preprocessing_jit(ca)
    ca += np.random.default_rng(seed).random(len(ca)) * 1e-08
    return ca


def _calcium_preprocessing_jit(ca):
    """JIT-compiled core computation for calcium_preprocessing.

    Applies numerical preprocessing to calcium signal for stability.
    This is the performance-critical inner loop separated for JIT compilation.

    Parameters
    ----------
    ca : ndarray
        Calcium signal array. Will be converted to float64.

    Returns
    -------
    ndarray
        Signal as float64 with negative values clipped to 0. The tie-breaking
        noise is added by the caller, outside the compiled code.
    """
    ca = ca.astype(np.float64)
    ca[ca < 0] = 0
    return ca


# Apply JIT compilation decorator
_calcium_preprocessing_jit = conditional_njit(_calcium_preprocessing_jit)
