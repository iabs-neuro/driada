import numpy as np
from ..utils.jit import conditional_njit, is_jit_enabled
from ..utils.data import check_positive

# Import numba types only if JIT is enabled
if is_jit_enabled():
    from numba.experimental import jitclass
    from numba import float32, boolean
    from numba import types, typed
else:
    # Provide dummy implementations
    jitclass = lambda spec: lambda cls: cls
    types = None
    typed = None


# Define spec only when numba is available
if is_jit_enabled():
    spec = [
        ("indices", types.ListType(types.float64)),
        ("ampls", types.ListType(types.float64)),
        ("birth_scale", float32),
        ("scales", types.ListType(types.float64)),
        ("wvt_times", types.ListType(types.float64)),
        ("terminated", boolean),
        ("end_scale", float32),
        ("length", types.int64),
        ("max_scale", float32),
        ("max_ampl", float32),
        ("start", float32),
        ("end", float32),
        ("duration", float32),
    ]
else:
    spec = None


@conditional_njit()
def maxpos_numba(x):
    """Find the index of the maximum value in a list or array.
    
    Parameters
    ----------
    x : list or array-like
        The sequence to search for maximum value. Must be non-empty.
        
    Returns
    -------
    int
        Index of the first occurrence of the maximum value.
        
    Raises
    ------
    ValueError
        If x is empty.
        
    Notes
    -----
    This function is JIT-compiled when numba is available for performance.
    Handles both regular Python lists and numba typed lists.    """
    if not x:
        raise ValueError("Cannot find maximum of empty sequence")
        
    m = max(x)
    # Handle both list and typed.List
    if hasattr(x, 'index'):
        return x.index(m)
    else:
        # Fallback for regular lists
        for i, val in enumerate(x):
            if val == m:
                return i
        return 0


# Conditional compilation for Ridge class
if is_jit_enabled():
    @jitclass(spec)
    class Ridge(object):
        """Container for wavelet ridge information during ridge detection.
        
        A ridge represents a connected path through wavelet transform scales
        where significant coefficients are found. This class tracks the evolution
        of a ridge as it is being constructed.
        
        Attributes
        ----------
        indices : list of float
            Time indices along the ridge.
        ampls : list of float
            Amplitudes (wavelet coefficients) along the ridge.
        birth_scale : float
            Scale at which the ridge was born (started).
        scales : list of float
            Wavelet scales along the ridge.
        wvt_times : list of float
            Wavelet times corresponding to the ridge points.
        terminated : bool
            Whether the ridge has been terminated.
        end_scale : float
            Scale at which the ridge ended (-1 if not terminated).
        length : int
            Number of points in the ridge (-1 if not terminated).
        max_scale : float
            Scale at which maximum amplitude occurs (-1 if not terminated).
        max_ampl : float
            Maximum amplitude along the ridge (-1 if not terminated).
        start : float
            Starting time index of the ridge (-1 if not terminated).
        end : float
            Ending time index of the ridge (-1 if not terminated).
        duration : float
            Time duration of the ridge (-1 if not terminated).
            
        Notes
        -----
        This class uses numba's JIT compilation when available for performance.
        The class is compiled with specific type annotations for optimal speed.        """
        def __init__(self, start_index, ampl, start_scale, wvt_time):
            """Initialize a new Ridge object.
            
            Parameters
            ----------
            start_index : float
                Time index where the ridge begins.
            ampl : float
                Initial amplitude (wavelet coefficient) of the ridge.
            start_scale : float
                Wavelet scale where the ridge begins.
            wvt_time : float
                Wavelet time corresponding to the start point.
                
            Notes
            -----
            All summary statistics (length, max_ampl, etc.) are initialized
            to -1 and will be computed when terminate() is called.            """
            self.indices = typed.List.empty_list(types.float64)
            self.indices.append(start_index)

            self.ampls = typed.List.empty_list(types.float64)
            self.ampls.append(ampl)

            self.birth_scale = start_scale

            self.scales = typed.List.empty_list(types.float64)
            self.scales.append(start_scale)

            self.wvt_times = typed.List.empty_list(types.float64)
            self.wvt_times.append(wvt_time)

            self.terminated = False

            self.end_scale = -1
            self.length = -1
            self.max_scale = -1
            self.max_ampl = -1
            self.start = -1
            self.end = -1
            self.duration = -1

        def extend(self, index, ampl, scale, wvt_time):
            """Extend the ridge with a new point.
            
            Adds a new data point to the ridge, updating all tracked attributes.
            Can only be called on non-terminated ridges.
            
            Parameters
            ----------
            index : float
                Time index of the new point.
            ampl : float
                Amplitude (wavelet coefficient) at this point.
            scale : float
                Wavelet scale at this point.
            wvt_time : float
                Wavelet time corresponding to this point.
                
            Raises
            ------
            ValueError
                If the ridge has already been terminated.
                
            Notes
            -----
            The order of appending is important for maintaining consistency
            across all ridge attributes.            """
            if not self.terminated:
                self.scales.append(scale)
                self.ampls.append(ampl)
                self.indices.append(index)
                self.wvt_times.append(wvt_time)
            else:
                raise ValueError("Ridge is terminated")

        def tip(self):
            """Get the time index of the ridge's current endpoint.
            
            Returns
            -------
            float
                The time index of the last point in the ridge.
                
            Raises
            ------
            IndexError
                If the ridge is empty (should not occur in normal use).
                
            Notes
            -----
            This is typically used during ridge construction to determine
            where to look for the next potential ridge point.            """
            return self.indices[-1]

        def terminate(self):
            """Finalize the ridge and compute summary statistics.
            
            Marks the ridge as terminated and calculates various summary
            attributes including length, maximum amplitude, duration, etc.
            Can only be called once per ridge.
            
            Notes
            -----
            If already terminated, this method does nothing (silent no-op).
            After termination, the ridge can no longer be extended.
            The following attributes are computed:
            - end_scale: Final scale value
            - length: Number of points in the ridge
            - max_scale: Scale at maximum amplitude
            - max_ampl: Maximum amplitude value
            - start: Starting time index
            - end: Ending time index
            - duration: Time span of the ridge            """
            if self.terminated:
                pass
            else:
                self.end_scale = self.scales[-1]
                self.length = len(self.scales)
                self.max_scale = self.scales[maxpos_numba(self.ampls)]
                self.max_ampl = max(self.ampls)
                self.start = min(self.indices)
                self.end = max(self.indices)
                self.duration = self.end - self.start
                self.terminated = True
else:
    # Pure Python implementation
    class Ridge(object):
        """Container for wavelet ridge information during ridge detection.
        
        A ridge represents a connected path through wavelet transform scales
        where significant coefficients are found. This class tracks the evolution
        of a ridge as it is being constructed.
        
        Attributes
        ----------
        indices : list of float
            Time indices along the ridge.
        ampls : list of float
            Amplitudes (wavelet coefficients) along the ridge.
        birth_scale : float
            Scale at which the ridge was born (started).
        scales : list of float
            Wavelet scales along the ridge.
        wvt_times : list of float
            Wavelet times corresponding to the ridge points.
        terminated : bool
            Whether the ridge has been terminated.
        end_scale : float
            Scale at which the ridge ended (-1 if not terminated).
        length : int
            Number of points in the ridge (-1 if not terminated).
        max_scale : float
            Scale at which maximum amplitude occurs (-1 if not terminated).
        max_ampl : float
            Maximum amplitude along the ridge (-1 if not terminated).
        start : float
            Starting time index of the ridge (-1 if not terminated).
        end : float
            Ending time index of the ridge (-1 if not terminated).
        duration : float
            Time duration of the ridge (-1 if not terminated).
            
        Notes
        -----
        This is the pure Python implementation used when numba is not available.        """
        def __init__(self, start_index, ampl, start_scale, wvt_time):
            """Initialize a new Ridge object.
            
            Parameters
            ----------
            start_index : float
                Time index where the ridge begins.
            ampl : float
                Initial amplitude (wavelet coefficient) of the ridge.
            start_scale : float
                Wavelet scale where the ridge begins.
            wvt_time : float
                Wavelet time corresponding to the start point.
                
            Notes
            -----
            All summary statistics (length, max_ampl, etc.) are initialized
            to -1 and will be computed when terminate() is called.            """
            self.indices = [start_index]
            self.ampls = [ampl]
            self.birth_scale = start_scale
            self.scales = [start_scale]
            self.wvt_times = [wvt_time]
            self.terminated = False
            self.end_scale = -1
            self.length = -1
            self.max_scale = -1
            self.max_ampl = -1
            self.start = -1
            self.end = -1
            self.duration = -1

        def extend(self, index, ampl, scale, wvt_time):
            """Extend the ridge with a new point.
            
            Adds a new data point to the ridge, updating all tracked attributes.
            Can only be called on non-terminated ridges.
            
            Parameters
            ----------
            index : float
                Time index of the new point.
            ampl : float
                Amplitude (wavelet coefficient) at this point.
            scale : float
                Wavelet scale at this point.
            wvt_time : float
                Wavelet time corresponding to this point.
                
            Raises
            ------
            ValueError
                If the ridge has already been terminated.
                
            Notes
            -----
            The order of appending is important for maintaining consistency
            across all ridge attributes.            """
            if not self.terminated:
                self.scales.append(scale)
                self.ampls.append(ampl)
                self.indices.append(index)
                self.wvt_times.append(wvt_time)
            else:
                raise ValueError("Ridge is terminated")

        def tip(self):
            """Get the time index of the ridge's current endpoint.
            
            Returns
            -------
            float
                The time index of the last point in the ridge.
                
            Raises
            ------
            IndexError
                If the ridge is empty (should not occur in normal use).
                
            Notes
            -----
            This is typically used during ridge construction to determine
            where to look for the next potential ridge point.            """
            return self.indices[-1]

        def terminate(self):
            """Finalize the ridge and compute summary statistics.
            
            Marks the ridge as terminated and calculates various summary
            attributes including length, maximum amplitude, duration, etc.
            Can only be called once per ridge.
            
            Notes
            -----
            If already terminated, this method does nothing (silent no-op).
            After termination, the ridge can no longer be extended.
            The following attributes are computed:
            - end_scale: Final scale value
            - length: Number of points in the ridge
            - max_scale: Scale at maximum amplitude
            - max_ampl: Maximum amplitude value
            - start: Starting time index
            - end: Ending time index
            - duration: Time span of the ridge            """
            if self.terminated:
                pass
            else:
                self.end_scale = self.scales[-1]
                self.length = len(self.scales)
                self.max_scale = self.scales[maxpos_numba(self.ampls)]
                self.max_ampl = max(self.ampls)
                self.start = min(self.indices)
                self.end = max(self.indices)
                self.duration = self.end - self.start
                self.terminated = True


class RidgeInfoContainer(object):
    """Container for finalized ridge information after ridge detection.
    
    This class stores the complete information about a detected ridge
    in a format suitable for further analysis. Unlike the Ridge class,
    this uses numpy arrays for efficient computation.
    
    Parameters
    ----------
    indices : list or array-like
        Time indices along the ridge. Must be non-empty.
    ampls : list or array-like
        Amplitudes (wavelet coefficients) along the ridge. Must be non-empty
        and same length as indices.
    scales : list or array-like
        Wavelet scales along the ridge. Must be non-empty and same length as indices.
    wvt_times : list or array-like
        Wavelet times corresponding to the ridge points. Must be non-empty
        and same length as indices.
        
    Attributes
    ----------
    indices : numpy.ndarray
        Time indices along the ridge.
    ampls : numpy.ndarray
        Amplitudes along the ridge.
    scales : numpy.ndarray
        Wavelet scales along the ridge.
    wvt_times : numpy.ndarray
        Wavelet times along the ridge.
    birth_scale : float
        Scale at which the ridge was born.
    end_scale : float
        Scale at which the ridge ended.
    length : int
        Number of points in the ridge.
    max_scale : float
        Scale at which maximum amplitude occurs.
    max_ampl : float
        Maximum amplitude along the ridge.
    start : float
        Starting time index of the ridge.
    end : float
        Ending time index of the ridge.
    duration : float
        Time duration of the ridge.
        
    Raises
    ------
    ValueError
        If any input array is empty.
        If input arrays have different lengths.
        
    Examples
    --------
    >>> indices = [10, 11, 12, 13]
    >>> ampls = [0.5, 0.8, 0.9, 0.6]
    >>> scales = [1.0, 1.1, 1.2, 1.3]
    >>> wvt_times = [0.1, 0.11, 0.12, 0.13]
    >>> ridge_info = RidgeInfoContainer(indices, ampls, scales, wvt_times)
    >>> ridge_info.max_ampl
    0.9
    >>> ridge_info.duration
    3

    Notes
    -----
    All input arrays are converted to numpy arrays. The class assumes
    the ridge data comes from a terminated Ridge object.    """
    def __init__(self, indices, ampls, scales, wvt_times):
        """Initialize RidgeInfoContainer with ridge data.
        
        Parameters
        ----------
        indices : array-like
            Indices of the ridge points in the wavelet transform.
        ampls : array-like
            Amplitudes at the ridge points.
        scales : array-like
            Scales at the ridge points.
        wvt_times : array-like
            Time points corresponding to the ridge in wavelet space.
        
        Raises
        ------
        ValueError
            If any input array is empty.
        """
        # Convert to arrays
        self.indices = np.array(indices)
        self.ampls = np.array(ampls)
        self.scales = np.array(scales)
        self.wvt_times = np.array(wvt_times)
        
        # Validate non-empty
        if self.indices.size == 0:
            raise ValueError("indices cannot be empty")
        if self.ampls.size == 0:
            raise ValueError("ampls cannot be empty")
        if self.scales.size == 0:
            raise ValueError("scales cannot be empty")
        if self.wvt_times.size == 0:
            raise ValueError("wvt_times cannot be empty")
            
        # Validate consistent lengths
        lengths = [self.indices.size, self.ampls.size, self.scales.size, self.wvt_times.size]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(f"All input arrays must have the same length, got {lengths}")

        self.birth_scale = self.scales[0]
        self.end_scale = self.scales[-1]
        self.length = len(self.scales)
        self.max_scale = self.scales[np.argmax(self.ampls)]
        self.max_ampl = np.max(self.ampls)
        self.start = min(self.indices)
        self.end = max(self.indices)
        self.duration = self.end - self.start


def ridges_to_containers(ridges):
    """Convert a list of Ridge objects to RidgeInfoContainer objects.
    
    This function transforms the Ridge objects (which may be JIT-compiled)
    into RidgeInfoContainer objects that use numpy arrays for efficient
    further processing and analysis.
    
    Parameters
    ----------
    ridges : list of Ridge
        List of Ridge objects to convert. Each ridge should be terminated.
        
    Returns
    -------
    list of RidgeInfoContainer
        List of RidgeInfoContainer objects with the same ridge information
        stored in numpy arrays. Empty list if input is empty.
        
    Raises
    ------
    AttributeError
        If any ridge object lacks required attributes.
    ValueError
        If RidgeInfoContainer initialization fails (e.g., empty ridge data).
        
    Examples
    --------
    >>> # Assuming we have detected ridges
    >>> ridge = Ridge(10, 0.5, 1.0, 0.1)
    >>> ridge.extend(11, 0.8, 1.1, 0.11)
    >>> ridge.terminate()
    >>> containers = ridges_to_containers([ridge])
    >>> len(containers)
    1
    >>> containers[0].max_ampl
    0.8
    
    Notes
    -----
    The function assumes all ridges have been properly terminated before
    conversion. Unterminated ridges may have incomplete attribute data.    """
    rcs = [
        RidgeInfoContainer(ridge.indices, ridge.ampls, ridge.scales, ridge.wvt_times)
        for ridge in ridges
    ]
    return rcs
