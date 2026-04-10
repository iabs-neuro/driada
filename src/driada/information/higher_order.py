"""High-order information measures on MultiTimeSeries objects.

This module provides object-aware wrappers around the low-level
Gaussian-copula implementations in gcmi.py (tc_gg, dtc_gg, o_info_gg).
An estimator parameter is accepted for API symmetry with other DRIADA
functions, but only 'gcmi' is implemented in v1.
"""
from .info_base import MultiTimeSeries
from .gcmi import tc_gg, dtc_gg, o_info_gg


def _extract_data(mts, estimator):
    """Extract input data for higher-order functions based on estimator choice.

    Parameters
    ----------
    mts : MultiTimeSeries
    estimator : {'gcmi', 'ksg'}

    Returns
    -------
    ndarray
        Data suitable for the requested low-level estimator.

    Raises
    ------
    TypeError
        If mts is not a MultiTimeSeries.
    ValueError
        If copula_normal_data is None (discrete MTS or after clear_caches).
    NotImplementedError
        If estimator is 'ksg'.
    ValueError
        If estimator is unknown.
    """
    if not isinstance(mts, MultiTimeSeries):
        raise TypeError(
            f"Expected MultiTimeSeries, got {type(mts).__name__}"
        )

    if estimator == "gcmi":
        if mts.copula_normal_data is None:
            raise ValueError(
                "MultiTimeSeries has no copula_normal_data. This happens "
                "either because the MultiTimeSeries was constructed from "
                "discrete TimeSeries (GCMI O-information requires continuous "
                "signals), or because clear_caches() was called. "
                "Rebuild the MultiTimeSeries from continuous components."
            )
        return mts.copula_normal_data

    elif estimator == "ksg":
        raise NotImplementedError(
            "KSG O-information is not implemented. The Kozachenko-Leonenko "
            "entropy estimator is unstable for populations larger than "
            "~10 variables due to the curse of dimensionality. "
            "Use estimator='gcmi' for multivariate neural data."
        )

    else:
        raise ValueError(
            f"Unknown estimator: {estimator!r}. Use 'gcmi' or 'ksg'."
        )


def total_correlation(mts, estimator="gcmi"):
    """Total correlation on a MultiTimeSeries.

    Parameters
    ----------
    mts : MultiTimeSeries
    estimator : {'gcmi', 'ksg'}, default 'gcmi'
        'ksg' raises NotImplementedError.

    Returns
    -------
    tc : float
    """
    x = _extract_data(mts, estimator)
    return tc_gg(x)


def dual_total_correlation(mts, estimator="gcmi"):
    """Dual total correlation on a MultiTimeSeries.

    Parameters
    ----------
    mts : MultiTimeSeries
    estimator : {'gcmi', 'ksg'}, default 'gcmi'
        'ksg' raises NotImplementedError.

    Returns
    -------
    dtc : float
    """
    x = _extract_data(mts, estimator)
    return dtc_gg(x)


def o_information(mts, estimator="gcmi", return_components=False):
    """O-information on a MultiTimeSeries.

    Parameters
    ----------
    mts : MultiTimeSeries
        Population activity.
    estimator : {'gcmi', 'ksg'}, default 'gcmi'
        'ksg' raises NotImplementedError.
    return_components : bool, default False
        If True, return dict with keys 'omega', 'tc', 'dtc'.

    Returns
    -------
    float or dict
        Omega value, or dict with omega, tc, dtc if return_components=True.
    """
    x = _extract_data(mts, estimator)
    result = o_info_gg(x, return_components=return_components)
    if return_components:
        omega, tc, dtc = result
        return {"omega": omega, "tc": tc, "dtc": dtc}
    return result
