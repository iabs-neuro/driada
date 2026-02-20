"""Calcium indicator kinetics and signal reconstruction.

This module provides functions for modeling calcium indicator dynamics using
double-exponential kernels and reconstructing calcium signals from spike trains.

The core model uses separate rise and decay time constants to capture the
asymmetric temporal dynamics of genetically-encoded calcium indicators (GECIs)
like GCaMP.
"""

import numpy as np
from scipy.signal import fftconvolve

from ..utils.data import check_positive
from ..utils.jit import conditional_njit


# Default kinetics parameters (in seconds)
DEFAULT_T_RISE = 0.25  # Rise time constant (seconds)
DEFAULT_T_OFF = 2.0  # Decay time constant (seconds)

# Kernel generation constants
KERNEL_LENGTH_FRAMES = 500  # Minimum kernel length; actual length is max(500, 5 × t_off)


def spike_form(t, t_rise, t_off):
    """Calculate normalized calcium response kernel shape.

    Computes the double-exponential kernel used to model calcium
    indicator dynamics with separate rise and decay time constants.

    Parameters
    ----------
    t : array-like
        Time points (in frames). Must be non-negative.
    t_rise : float
        Rise time constant (in frames). Must be positive.
    t_off : float
        Decay time constant (in frames). Must be positive.

    Returns
    -------
    ndarray
        Normalized kernel values with peak = 1.

    Raises
    ------
    ValueError
        If t_rise or t_off are not positive.

    Notes
    -----
    The kernel has the form: (1 - exp(-t/τ_rise)) * exp(-t/τ_off)
    normalized to have maximum value of 1.

    Examples
    --------
    >>> t = np.linspace(0, 100, 100)
    >>> kernel = spike_form(t, t_rise=5, t_off=20)
    >>> kernel.max()  # Should be 1.0
    1.0
    """
    check_positive(t_rise=t_rise, t_off=t_off)
    return _spike_form_jit(t, t_rise, t_off)


def _spike_form_jit(t, t_rise, t_off):
    """JIT-compiled core computation for spike_form.

    Computes normalized double-exponential calcium response kernel.
    This is the performance-critical inner loop separated for JIT compilation.

    Parameters
    ----------
    t : ndarray
        Time points in frames. Assumed to be non-negative.
    t_rise : float
        Rise time constant in frames. Assumed positive.
    t_off : float
        Decay time constant in frames. Assumed positive.

    Returns
    -------
    ndarray
        Normalized kernel with maximum value of 1.

    Raises
    ------
    ValueError
        If computed kernel has zero maximum (numerical issue).

    Notes
    -----
    Input validation is performed in the wrapper function spike_form().
    JIT compilation provides significant speedup for large arrays.
    """
    form = (1 - np.exp(-t / t_rise)) * np.exp(-t / t_off)
    max_val = np.max(form)
    if max_val == 0:
        raise ValueError("Kernel form has zero maximum")
    return form / max_val


# Apply JIT compilation decorator
_spike_form_jit = conditional_njit(_spike_form_jit)


def get_restored_calcium(sp, t_rise, t_off):
    """Reconstruct calcium signal from spike train.

    Convolves spike train with double-exponential kernel to simulate
    calcium indicator dynamics. The output has the same length as the
    input spike train by truncating the convolution tail.

    Parameters
    ----------
    sp : array-like
        Spike train. Can be binary (0/1) or amplitude-weighted (float).
        For best reconstruction fidelity, use neuron.asp.data with
        amplitude information. Must be 1D array.
    t_rise : float
        Rise time constant (in frames). Must be positive.
    t_off : float
        Decay time constant (in frames). Must be positive.

    Returns
    -------
    ndarray
        Reconstructed calcium signal with same length as sp.

    Raises
    ------
    ValueError
        If t_rise or t_off are not positive, or if sp is empty.

    Notes
    -----
    Uses FFT-based convolution for optimal performance. Kernel length is
    adaptive: max(500, 5 × t_off frames) to ensure complete kernel capture
    for all decay time constants.

    The convolution naturally handles amplitude-weighted spikes, where
    each spike value represents event strength in dF/F0 units.

    Examples
    --------
    >>> sp = np.zeros(1000)
    >>> sp[100] = 1.0  # Single spike
    >>> ca = get_restored_calcium(sp, t_rise=5, t_off=20)
    >>> ca.shape
    (1000,)
    """
    sp = np.asarray(sp)
    if sp.size == 0:
        raise ValueError("Spike train cannot be empty")
    check_positive(t_rise=t_rise, t_off=t_off)

    # Adaptive kernel length: 5× decay time for complete kernel, minimum 500 frames
    # Safety check: cap at 2000 frames to prevent memory issues from bad t_off
    # (t_off > 400 frames or ~8s @ 20Hz is suspicious for typical indicators)
    kernel_length = max(KERNEL_LENGTH_FRAMES, int(5 * t_off))
    if kernel_length > 2000:
        import warnings

        warnings.warn(
            f"Kernel length {kernel_length} (from t_off={t_off:.1f} frames) capped at 2000. "
            f"This may indicate incorrect t_off measurement. Typical calcium indicators have "
            f"t_off < 200 frames (~8-10s @ 20Hz).",
            UserWarning,
        )
        kernel_length = 2000

    x = np.arange(kernel_length)
    spform = spike_form(x, t_rise, t_off)
    conv = fftconvolve(sp, spform, mode="full")
    return conv[: len(sp)]


def ca_mse_error(t_off, ca, spk, t_rise):
    """Calculate RMSE between observed calcium and reconstructed from spikes.

    This function is designed to be used with scipy.optimize.minimize,
    hence the parameter order with t_off first.

    Parameters
    ----------
    t_off : float
        Decay time constant (in frames). Must be positive.
    ca : array-like
        Observed calcium signal. Must be 1D.
    spk : array-like
        Spike train. Must be 1D with same length as ca.
    t_rise : float
        Rise time constant (in frames). Must be positive.

    Returns
    -------
    float
        Root mean square error between observed and reconstructed calcium.

    Raises
    ------
    ValueError
        If arrays have different lengths or time constants are invalid.

    Notes
    -----
    Parameter order (t_off first) is optimized for scipy.optimize.minimize
    where t_off is the parameter being optimized.

    Examples
    --------
    >>> ca = np.random.random(1000)
    >>> spk = np.zeros(1000)
    >>> spk[100:900:100] = 1
    >>> error = ca_mse_error(t_off=20, ca=ca, spk=spk, t_rise=5)
    >>> error > 0
    True
    """
    # scipy.optimize.minimize passes parameters as 1-element ndarrays;
    # extract scalar to avoid NumPy deprecation (ndim>0 to scalar).
    t_off = float(np.asarray(t_off).ravel()[0])
    ca = np.asarray(ca)
    spk = np.asarray(spk)
    if len(ca) != len(spk):
        raise ValueError(f"ca and spk must have same length: {len(ca)} vs {len(spk)}")
    check_positive(t_rise=t_rise, t_off=t_off)
    re_ca = get_restored_calcium(spk, t_rise, t_off)
    return np.sqrt(np.sum((ca - re_ca) ** 2) / len(ca))
