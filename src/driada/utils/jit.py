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
    """Check if JIT compilation is enabled.
    
    Determines whether Numba JIT compilation is available and active
    based on installation status and environment settings.
    
    Returns
    -------
    bool
        True if both conditions are met:
        - Numba is installed and importable
        - DRIADA_DISABLE_NUMBA environment variable is not set to '1'
        
    Examples
    --------
    >>> is_jit_enabled()
    True  # If Numba is installed and not disabled
    
    >>> # Disable JIT via environment
    >>> import os
    >>> os.environ['DRIADA_DISABLE_NUMBA'] = '1'
    >>> is_jit_enabled()
    False
    
    Notes
    -----
    JIT compilation significantly speeds up numerical computations but
    may cause issues during debugging. Use DRIADA_DISABLE_NUMBA=1 to
    disable JIT when debugging or if encountering Numba-related errors.
    
    See Also
    --------
    jit_info : Print detailed JIT status information
    conditional_jit : Decorator that respects JIT settings
    """
    return NUMBA_AVAILABLE and not DRIADA_DISABLE_NUMBA


def jit_info():
    """Print information about JIT compilation status.
    
    Displays comprehensive information about the JIT compilation
    environment, including Numba availability, version, and current
    configuration settings.
    
    Prints
    ------
    - Whether Numba is installed
    - If JIT is disabled via environment variable
    - Overall JIT enabled status
    - Numba version (if available)
    
    Examples
    --------
    >>> jit_info()
    Numba available: True
    JIT disabled by environment: False
    JIT enabled: True
    Numba version: 0.57.0
    
    >>> # With JIT disabled
    >>> os.environ['DRIADA_DISABLE_NUMBA'] = '1'
    >>> jit_info()
    Numba available: True
    JIT disabled by environment: True
    JIT enabled: False
    Numba version: 0.57.0
    
    Notes
    -----
    Useful for debugging performance issues or verifying that JIT
    compilation is working as expected in your environment.
    
    See Also
    --------
    is_jit_enabled : Check JIT status programmatically
    """
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"JIT disabled by environment: {DRIADA_DISABLE_NUMBA}")
    print(f"JIT enabled: {is_jit_enabled()}")
    if NUMBA_AVAILABLE:
        import numba

        print(f"Numba version: {numba.__version__}")
