"""JIT compilation utilities for DRIADA.

Provides conditional JIT compilation based on environment settings.
"""

import os

# Check if Numba should be disabled
DRIADA_DISABLE_NUMBA = os.getenv("DRIADA_DISABLE_NUMBA", "False").lower() in (
    "true",
    "1",
    "yes",
)

# Try to import numba
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Define dummy decorators
    def njit(*args, **kwargs):
        """Dummy njit decorator when numba is not available.
        
        This function acts as a pass-through decorator that returns
        the original function unchanged when numba is not installed
        or disabled.
        
        Parameters
        ----------
        *args
            Positional arguments (function to decorate if called directly)
        **kwargs
            Keyword arguments (ignored)
            
        Returns
        -------
        function or decorator
            Original function or decorator that returns original function
        """
        def decorator(func):
            return func

        return decorator if not args else args[0]

    prange = range


def conditional_njit(*args, **kwargs):
    """
    Conditionally apply numba JIT compilation based on environment settings.

    If DRIADA_DISABLE_NUMBA environment variable is set to true, or if numba
    is not available, this returns the original function without JIT compilation.

    Parameters
    ----------
    *args, **kwargs
        Arguments passed to numba.njit

    Returns
    -------
    decorator or function
        JIT-compiled function if enabled, otherwise original function

    Examples
    --------
    >>> @conditional_njit
    ... def fast_computation(x):
    ...     return x ** 2

    >>> # With parallel=True
    >>> @conditional_njit(parallel=True)
    ... def parallel_computation(x):
    ...     return x ** 2
    """
    if DRIADA_DISABLE_NUMBA or not NUMBA_AVAILABLE:
        # Return identity decorator
        def decorator(func):
            return func

        return decorator if not args else args[0]
    else:
        # Use actual njit
        return njit(*args, **kwargs)


def is_jit_enabled():
    """Check if JIT compilation is enabled."""
    return NUMBA_AVAILABLE and not DRIADA_DISABLE_NUMBA


def jit_info():
    """Print information about JIT compilation status."""
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"JIT disabled by environment: {DRIADA_DISABLE_NUMBA}")
    print(f"JIT enabled: {is_jit_enabled()}")
    if NUMBA_AVAILABLE:
        import numba

        print(f"Numba version: {numba.__version__}")
