"""
Utility functions for synthetic data generation.
"""


def get_effective_decay_time(decay_time, duration, verbose=True):
    """
    Calculate effective decay time for experiments to ensure valid shuffle positions.

    For short duration experiments, the decay time is reduced to prevent
    the shuffle mask from excluding all timepoints.

    Parameters
    ----------
    decay_time : float
        Original calcium decay time in seconds.
    duration : float
        Experiment duration in seconds.
    verbose : bool
        Whether to print adjustment messages.

    Returns
    -------
    float
        Effective decay time to use.
    """
    # For short durations, use smaller decay_time to reduce shuffle exclusion
    if duration <= 30:  # 30 seconds or less
        effective_decay_time = min(decay_time, 0.5)  # Max 0.5s for short tests
        if verbose and effective_decay_time < decay_time:
            print(
                f"  Using reduced decay time {effective_decay_time}s for short duration experiment"
            )
    else:
        effective_decay_time = decay_time

    return effective_decay_time
