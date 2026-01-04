"""
Utility functions for synthetic data generation.
"""

import numpy as np
from driada.utils.data import check_positive


def get_effective_decay_time(decay_time, duration, verbose=True):
    """
    Calculate effective decay time for experiments to ensure valid shuffle positions.

    For short duration experiments, the decay time is reduced to prevent
    the shuffle mask from excluding all timepoints. This ensures that calcium
    signals can be properly analyzed even in brief experimental recordings.

    Parameters
    ----------
    decay_time : float
        Original calcium decay time in seconds. Must be positive.
    duration : float
        Experiment duration in seconds. Must be positive.
    verbose : bool, optional
        Whether to print adjustment messages. Default is True.

    Returns
    -------
    float
        Effective decay time to use. Will be min(decay_time, 0.5) for
        experiments ≤30s, otherwise returns original decay_time.

    Raises
    ------
    ValueError
        If decay_time or duration are non-positive, NaN, or infinity.
    TypeError
        If inputs are not numeric.

    Notes
    -----
    The 30-second threshold and 0.5-second maximum are empirically
    determined values that work well for most calcium imaging experiments.    """
    # Input validation
    check_positive(decay_time=decay_time, duration=duration)
    
    # Constants for shuffle exclusion prevention
    SHORT_DURATION_THRESHOLD = 30.0  # seconds
    MAX_DECAY_TIME_SHORT = 0.5  # seconds
    
    # For short durations, use smaller decay_time to reduce shuffle exclusion
    if duration <= SHORT_DURATION_THRESHOLD:
        effective_decay_time = min(decay_time, MAX_DECAY_TIME_SHORT)
        if verbose and effective_decay_time < decay_time:
            print(
                f"  Using reduced decay time {effective_decay_time}s for short duration experiment"
            )
    else:
        effective_decay_time = decay_time

    return effective_decay_time
