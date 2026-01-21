"""Parallel execution utilities for DRIADA.

Provides centralized parallel execution configuration that respects
the global driada.PARALLEL_BACKEND setting.
"""

from contextlib import contextmanager
from joblib import Parallel, delayed, parallel_config


def get_parallel_backend():
    """Get the current parallel backend setting.

    Returns
    -------
    str
        Current backend: 'loky', 'threading', or 'multiprocessing'.
    """
    import driada
    return driada.PARALLEL_BACKEND


@contextmanager
def parallel_executor(
    n_jobs,
    verbose=False,
    backend=None,
    pre_dispatch=None,
    idle_worker_timeout=None,
):
    """Context manager for parallel execution with backend-specific config.

    Provides centralized configuration for all parallel calls across DRIADA,
    with backend-specific optimizations:
    - loky: Aggressive idle_worker_timeout (60s) to prevent worker accumulation
    - threading: Conservative pre_dispatch to limit memory buildup
    - multiprocessing: Default settings

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs. Use -1 for all available cores.
    verbose : bool, default=False
        Whether to print configuration details.
    backend : str, optional
        Override the global driada.PARALLEL_BACKEND setting for this call.
        Options: 'loky', 'threading', 'multiprocessing'.
    pre_dispatch : str or int, optional
        Override the backend-specific pre_dispatch setting.
        Controls how many tasks are dispatched ahead of time.
        Examples: 'n_jobs', '2*n_jobs', 'all', or an integer.
    idle_worker_timeout : int, optional
        Override idle worker timeout for loky backend (seconds).
        Workers idle longer than this are terminated. Default: 60 for loky.

    Yields
    ------
    Parallel
        Configured joblib Parallel executor.

    Examples
    --------
    Basic usage with global backend:

    >>> from driada.utils.parallel import parallel_executor
    >>> from joblib import delayed
    >>> def process(x):
    ...     return x * 2
    >>> with parallel_executor(n_jobs=2) as parallel:
    ...     results = parallel(delayed(process)(i) for i in range(10))

    Override backend for specific call:

    >>> with parallel_executor(n_jobs=4, backend='threading') as parallel:
    ...     results = parallel(delayed(process)(i) for i in range(10))

    Custom pre_dispatch for memory-intensive tasks:

    >>> with parallel_executor(n_jobs=4, pre_dispatch='n_jobs') as parallel:
    ...     results = parallel(delayed(heavy_task)(i) for i in range(100))

    Notes
    -----
    Threading backend is recommended for:
    - Windows systems (avoids DLL loading issues with PyTorch/ssqueezepy)
    - Code with unpicklable objects (no serialization needed)
    - NumPy-heavy operations (releases GIL during FFT/BLAS)

    Loky backend is recommended for:
    - CPU-bound Python code
    - Long-running tasks where true parallelism matters
    - Linux/macOS systems without pickling constraints

    To change the backend globally:

    >>> import driada
    >>> driada.set_parallel_backend('threading')
    """
    import driada

    # Use provided backend or fall back to global setting
    if backend is None:
        backend = driada.PARALLEL_BACKEND

    # Backend-specific parallel_config settings
    config = {'backend': backend}
    parallel_kwargs = {'n_jobs': n_jobs, 'backend': backend}

    # Determine pre_dispatch: use provided value or backend-specific default
    if pre_dispatch is not None:
        parallel_kwargs['pre_dispatch'] = pre_dispatch
    elif backend == 'threading':
        # Threading backend: conservative pre_dispatch to limit memory pressure
        parallel_kwargs['pre_dispatch'] = 'n_jobs'
    else:
        # loky/multiprocessing: more aggressive pre_dispatch
        parallel_kwargs['pre_dispatch'] = '2*n_jobs'

    # Determine idle_worker_timeout for loky backend
    if backend == 'loky':
        if idle_worker_timeout is not None:
            config['idle_worker_timeout'] = idle_worker_timeout
        else:
            # Default: aggressive cleanup after 60s (vs joblib default 300s)
            config['idle_worker_timeout'] = 60

    if verbose:
        timeout_info = ""
        if 'idle_worker_timeout' in config:
            timeout_info = f", idle_timeout={config['idle_worker_timeout']}s"
        print(f"Parallel config: backend={backend}{timeout_info}, pre_dispatch={parallel_kwargs['pre_dispatch']}")

    with parallel_config(**config):
        yield Parallel(**parallel_kwargs)


# Re-export delayed for convenience
__all__ = ['parallel_executor', 'get_parallel_backend', 'delayed']
