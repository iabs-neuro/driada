from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics.cluster import mutual_info_score
import scipy

from .ksg import (
    build_tree,
    nonparam_entropy_c,
    nonparam_mi_cc,
    nonparam_mi_cd,
    nonparam_mi_dc,
)
from .gcmi import (
    copnorm,
    ent_g,
    mi_gg,
    mi_model_gd,
    cmi_ggg,
    gccmi_ccd,
)
from .info_utils import binary_mi_score
from ..utils.data import correlation_matrix
from .entropy import entropy_d, joint_entropy_dd, joint_entropy_cd, joint_entropy_cdd
from ..dim_reduction.data import MVData
from .time_series_types import (
    analyze_time_series_type,
    is_discrete_time_series,
    TimeSeriesType,
)

import numpy as np
import warnings
from typing import Optional
from sklearn.preprocessing import MinMaxScaler

from ..utils.data import to_numpy_array


DEFAULT_NN = 5

# FUTURE: add @property decorators to properly set getter-setter functionality


class TimeSeries:
    """Single time series with automatic type detection and analysis capabilities.
    
    Represents a univariate time series with automatic detection of whether it's
    discrete (categorical, binary, count) or continuous (linear, circular). 
    Provides methods for entropy calculation, mutual information estimation,
    filtering, and complexity analysis.
    
    Parameters
    ----------
    data : array-like
        1D time series data.
    discrete : bool, optional
        If provided, overrides automatic type detection. Legacy parameter for
        backward compatibility.
    ts_type : TimeSeriesType or str, optional
        Explicit type specification. Can be a TimeSeriesType object or string:
        - 'binary': discrete binary data
        - 'categorical': discrete categorical data  
        - 'count': discrete monotonic count data
        - 'timeline': discrete regularly spaced values
        - 'linear': continuous linear data
        - 'circular': continuous circular/angular data
        - 'ambiguous': ambiguous discrete data
    shuffle_mask : array-like, optional
        Boolean mask indicating valid positions for shuffling operations.
        Used in significance testing.
    name : str, optional
        Name of the time series (provides context for type detection).
        
    Attributes
    ----------
    data : ndarray
        The time series data as 1D numpy array.
    discrete : bool
        Whether the time series is discrete.
    type_info : TimeSeriesType
        Detailed type information including subtype and confidence.
    scdata : ndarray
        Min-max scaled data to [0,1].
    data_scale : float
        Scaling factor used by MinMaxScaler.
    copula_normal_data : ndarray or None
        Copula-normalized data (continuous only).
    int_data : ndarray or None
        Integer representation (discrete only).
    is_binary : bool
        True if discrete with exactly 2 unique values, False otherwise.
    bool_data : ndarray or None
        Boolean representation (binary discrete only).
    shuffle_mask : ndarray
        Boolean mask for valid shuffle positions.
    entropy : dict
        Cached entropy values for different downsampling factors.
    kdtree : KDTree or None
        Cached KD-tree for k-NN searches.
    kdtree_query : tuple or None
        Cached k-NN query results.
        
    Methods
    -------
    get_entropy(ds=1)
        Compute entropy with optional downsampling.
    get_kdtree()
        Get or build KD-tree for k-NN operations.
    get_kdtree_query(k=5)
        Get k-nearest neighbors.
    filter(method='gaussian', **kwargs)
        Apply signal filtering and return new TimeSeries.
    approximate_entropy(m=2, r=None)
        Calculate approximate entropy (continuous only).
    define_ts_type(ts)
        Legacy static method for type detection (deprecated).
        
    Notes
    -----
    - Type detection uses statistical heuristics and can be overridden
    - Discrete data is stored as integers for efficient computation
    - Continuous data is copula-normalized for GCMI estimation
    - Entropy is computed in bits for consistency
    
    Warning
    -------
    This class is NOT thread-safe. The following attributes are lazily computed
    and cached, which can cause race conditions with concurrent access:
    - entropy dict (via get_entropy)
    - kdtree (via get_kdtree)
    - kdtree_query dict (via get_kdtree_query)
    
    For concurrent usage, ensure proper synchronization or use separate
    TimeSeries instances per thread.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Continuous time series
    >>> ts = TimeSeries(np.random.randn(1000))
    >>> print(ts.discrete)
    False
    >>> entropy = ts.get_entropy()
    >>> 
    >>> # Binary discrete time series
    >>> ts_binary = TimeSeries([0, 1, 0, 1, 1, 0], ts_type='binary')
    >>> print(ts_binary.is_binary)
    True
    >>> 
    >>> # With shuffle mask
    >>> data = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> mask = np.ones(1000, dtype=bool)
    >>> mask[:100] = False  # Invalid positions at start
    >>> ts_masked = TimeSeries(data, shuffle_mask=mask)    """
    @staticmethod
    def define_ts_type(ts):
        """Legacy method for backward compatibility. Use is_discrete_time_series instead.
        
        .. deprecated:: 2.0
           Use :func:`driada.information.time_series_types.is_discrete_time_series` instead.
           
        Attempts to determine if a time series is discrete or continuous using
        simple heuristics based on unique value ratio. This method is maintained
        only for backward compatibility and may be removed in future versions.
        
        Parameters
        ----------
        ts : array-like
            Time series data to analyze.
            
        Returns
        -------
        bool
            True if likely discrete, False if likely continuous.
            
        Warns
        -----
        DeprecationWarning
            Always emitted when this method is called.
        UserWarning
            If time series is too short (<100 samples) or type is ambiguous.
            
        Notes
        -----
        The legacy heuristic uses uniqueness ratio (n_unique/n_samples):
        - < 0.25: classified as discrete
        - > 0.70: classified as continuous  
        - Between: ambiguous, defaults to continuous
        
        The new type detection system is much more sophisticated and accurate.        """
        warnings.warn(
            "TimeSeries.define_ts_type is deprecated. "
            "Use driada.information.time_series_types.is_discrete_time_series instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Use new detection system but handle errors gracefully for backward compatibility
        try:
            return is_discrete_time_series(ts)
        except (ValueError, TypeError, AttributeError):
            # Fallback to legacy logic if new system fails
            if len(ts) < 100:
                warnings.warn(
                    "Time series is too short for accurate type determination"
                )

            unique_vals = np.unique(ts)
            sc1 = len(unique_vals) / len(ts)
            if sc1 < 0.25:
                return True  # Likely discrete
            elif sc1 > 0.7:
                return False  # Likely continuous
            else:
                # Ambiguous - default to continuous
                warnings.warn(
                    f"Ambiguous time series type (uniqueness ratio: {sc1:.2f}). Defaulting to continuous."
                )
                return False

    def _check_input(self):
        """Validate time series data.
        
        Checks for:
        - Valid data type and shape
        - No NaN or infinite values
        - Minimum length requirements
        - Data consistency
        
        Raises
        ------
        ValueError
            If data validation fails        """
        # Check data is numpy array
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Time series data must be a numpy array")
            
        # Check 1D
        if self.data.ndim != 1:
            raise ValueError(f"Time series must be 1D, got shape {self.data.shape}")
            
        # Check length
        if len(self.data) < 2:
            raise ValueError("Time series must have at least 2 points")
            
        # Check for NaN or infinite values
        if np.any(np.isnan(self.data)):
            raise ValueError("Time series contains NaN values")
        if np.any(np.isinf(self.data)):
            raise ValueError("Time series contains infinite values")
            
        # Check shuffle mask if provided
        if hasattr(self, 'shuffle_mask') and self.shuffle_mask is not None:
            if len(self.shuffle_mask) != len(self.data):
                raise ValueError("Shuffle mask must have same length as data")
            if not np.all((self.shuffle_mask == 0) | (self.shuffle_mask == 1)):
                raise ValueError("Shuffle mask must contain only 0s and 1s")

    def _create_type_from_string(self, type_str):
        """Create TimeSeriesType from string shortcut.
        
        Converts user-friendly string shortcuts into proper TimeSeriesType
        objects for manual type specification.
        
        Parameters
        ----------
        type_str : str
            Type shortcut string. Supported values:
            - 'discrete', 'd': generic discrete type
            - 'continuous', 'c': generic continuous type
            - 'binary', 'spike': binary discrete (0/1)
            - 'count': count data (non-negative integers)
            - 'categorical', 'cat': categorical discrete
            - 'circular', 'phase', 'angle': circular continuous
            - 'timeline', 'time': timeline/timestamp data
            
        Returns
        -------
        TimeSeriesType
            Configured type object with appropriate primary type and subtype.
            
        Raises
        ------
        ValueError
            If type_str is not recognized.
            
        Examples
        --------
        >>> data = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        >>> ts = TimeSeries(data, ts_type='spike')
        >>> ts.type_info.primary_type
        'discrete'
        >>> ts.type_info.subtype
        'binary'
        """
        type_str = type_str.lower()

        # Map string to primary type and subtype
        type_map = {
            # Full names
            "binary": ("discrete", "binary"),
            "categorical": ("discrete", "categorical"),
            "count": ("discrete", "count"),
            "timeline": ("discrete", "timeline"),
            "linear": ("continuous", "linear"),
            "circular": ("continuous", "circular"),
            "ambiguous": ("ambiguous", None),  # Primary type with no subtype
            # Generic types
            "discrete": ("discrete", None),
            "d": ("discrete", None),
            "continuous": ("continuous", "linear"),  # Default continuous to linear
            "c": ("continuous", "linear"),
            # Shortcuts
            "spike": ("discrete", "binary"),
            "cat": ("discrete", "categorical"),
            "phase": ("continuous", "circular"),
            "angle": ("continuous", "circular"),
            "time": ("discrete", "timeline"),
        }

        if type_str not in type_map:
            raise ValueError(
                f"Unknown type string '{type_str}'. Must be one of: {', '.join(type_map.keys())}"
            )

        primary_type, subtype = type_map[type_str]

        # Create appropriate metadata
        n_unique = len(np.unique(self.data))
        n_samples = len(self.data)

        # Special handling for circular
        is_circular = subtype == "circular"
        circular_period = None
        if is_circular:
            # Try to guess period from data range
            data_range = np.ptp(self.data)
            if 6 < data_range < 7:  # Likely radians
                circular_period = 2 * np.pi
            elif 350 < data_range < 370:  # Likely degrees
                circular_period = 360

        return TimeSeriesType(
            primary_type=primary_type,
            subtype=subtype,
            confidence=1.0,  # User specified
            is_circular=is_circular,
            circular_period=circular_period,
            periodicity=None,
            metadata={
                "n_unique": n_unique,
                "n_samples": n_samples,
                "user_specified": True,
            },
        )

    def __init__(self, data, discrete=None, ts_type=None, shuffle_mask=None, name=None):
        """
        Initialize TimeSeries object.

        Parameters
        ----------
        data : array-like
            Time series data
        discrete : bool, optional
            Legacy parameter for backward compatibility. If provided, overrides auto-detection.
        ts_type : TimeSeriesType or str, optional
            Full type specification or a string matching existing subtypes:
            - 'binary': discrete binary data
            - 'categorical': discrete categorical data
            - 'count': discrete monotonic count data
            - 'timeline': discrete regularly spaced values
            - 'linear': continuous linear data
            - 'circular': continuous circular/angular data
            - 'ambiguous': ambiguous discrete data
        shuffle_mask : array-like, optional
            Mask for valid shuffling positions
        name : str, optional
            Name of the time series (used for context in type detection)
            
        Raises
        ------
        ValueError
            If data is not a numpy array, not 1D, has less than 2 points,
            contains NaN or infinite values, or if shuffle_mask has invalid
            length or values.        """
        self.data = to_numpy_array(data)
        self.name = name
        self.shuffle_mask = shuffle_mask
        
        # Validate input data
        self._check_input()

        # Handle type specification
        if isinstance(ts_type, str):
            # String shortcut provided - create type from it
            self.type_info = self._create_type_from_string(ts_type)
            if self.type_info.is_ambiguous:
                warnings.warn(
                    f"Time series type is ambiguous (confidence: {self.type_info.confidence:.2f}). "
                    "Defaulting to continuous behavior. Consider specifying type explicitly."
                )
                self.discrete = False
            else:
                self.discrete = self.type_info.is_discrete
        elif ts_type is not None:
            # User provided full type specification
            self.type_info = ts_type
            if ts_type.is_ambiguous:
                warnings.warn(
                    f"Time series type is ambiguous (confidence: {ts_type.confidence:.2f}). "
                    "Defaulting to continuous behavior. Consider specifying type explicitly."
                )
                self.discrete = False
            else:
                self.discrete = ts_type.is_discrete
        elif discrete is not None:
            # Legacy discrete parameter - create minimal type info
            self.discrete = discrete
            n_unique = len(np.unique(self.data))
            self.type_info = TimeSeriesType(
                primary_type="discrete" if discrete else "continuous",
                subtype="binary" if discrete and n_unique == 2 else None,
                confidence=1.0,  # User specified
                is_circular=False,
                circular_period=None,
                periodicity=None,
                metadata={"n_unique": n_unique, "n_samples": len(self.data)},
            )
        else:
            # Auto-detect using new comprehensive system
            self.type_info = analyze_time_series_type(self.data, name=self.name)
            if self.type_info.is_ambiguous:
                warnings.warn(
                    f"Time series type is ambiguous (confidence: {self.type_info.confidence:.2f}). "
                    f"Detected scores: discrete={self.type_info.metadata.get('discrete_score', 'N/A'):.2f}, "
                    f"continuous={self.type_info.metadata.get('continuous_score', 'N/A'):.2f}. "
                    "Defaulting to continuous behavior. Consider specifying type explicitly."
                )
                self.discrete = False
            else:
                self.discrete = self.type_info.is_discrete

        scaler = MinMaxScaler()
        self.scdata = scaler.fit_transform(self.data.reshape(-1, 1)).reshape(1, -1)[0]
        self.data_scale = scaler.scale_
        self.copula_normal_data = None

        if self.discrete:
            self.int_data = np.round(self.data).astype(int)
            # Use type info for binary detection
            if (
                self.type_info.subtype == "binary"
                or len(set(self.data.astype(int))) == 2
            ):
                self.is_binary = True
                self.bool_data = self.int_data.astype(bool)
            else:
                self.is_binary = False

        else:
            self.copula_normal_data = copnorm(self.data).ravel()
            # Continuous time series are never binary
            self.is_binary = False

        self.entropy = dict()  # supports various downsampling constants
        self.kdtree = None
        self.kdtree_query = dict()  # Cache for different k values

        if shuffle_mask is None:
            # which shuffles are valid
            self.shuffle_mask = np.ones(len(self.data)).astype(bool)
        else:
            self.shuffle_mask = shuffle_mask.astype(bool)

    def get_kdtree(self):
        """Get or build KDTree for efficient nearest neighbor queries.
        
        Lazily constructs a KDTree from the time series data on first access
        and caches it for subsequent calls. The tree is built using the
        reshaped data (flattened to 2D).
        
        Returns
        -------
        sklearn.neighbors.KDTree
            KDTree structure for the time series data. Built using the
            build_tree function from ksg module.
            
        Notes
        -----
        The KDTree is cached in self.kdtree after first construction.
        Data is reshaped to (n_samples, -1) before tree construction
        to handle multi-dimensional time series.
        
        See Also
        --------
        ~driada.information.info_base.get_kdtree_query : Query the KDTree for k-nearest neighbors.        """
        if self.kdtree is None:
            tree = self._compute_kdtree()
            self.kdtree = tree

        return self.kdtree

    def _compute_kdtree(self):
        """Build KDTree structure from time series data.
        
        Internal method that constructs a KDTree for efficient nearest
        neighbor queries. Reshapes 1D time series to 2D as required by
        the KDTree implementation.
        
        Returns
        -------
        sklearn.neighbors.KDTree
            KDTree built from the reshaped time series data.
            
        Notes
        -----
        Called lazily by get_kdtree() and cached for efficiency.
        Uses build_tree from ksg module which wraps sklearn's KDTree.        """
        d = self.data.reshape(self.data.shape[0], -1)
        return build_tree(d)

    def get_kdtree_query(self, k=DEFAULT_NN):
        """Query KDTree for k-nearest neighbors of each point.
        
        Performs k-nearest neighbor search for all points in the time series
        data. Results are cached for efficiency when called multiple times
        with the same k value.
        
        Parameters
        ----------
        k : int, default=5
            Number of nearest neighbors to find for each point. The default
            value is DEFAULT_NN (5). Note that the query returns k+1 neighbors
            since each point is its own nearest neighbor.
            
        Returns
        -------
        tuple of (distances, indices)
            distances : ndarray of shape (n_samples, k+1)
                Distances to the k+1 nearest neighbors for each point.
            indices : ndarray of shape (n_samples, k+1)
                Indices of the k+1 nearest neighbors for each point.
                
        Notes
        -----
        The query includes each point as its own nearest neighbor (distance 0),
        so k+1 neighbors are returned. Results are cached for each k value
        in self.kdtree_query dictionary.
        
        See Also
        --------
        ~driada.information.info_base.get_kdtree : Build or retrieve the KDTree structure.        """
        if k not in self.kdtree_query:
            q = self._compute_kdtree_query(k=k)
            self.kdtree_query[k] = q

        return self.kdtree_query[k]

    def _compute_kdtree_query(self, k=DEFAULT_NN):
        """Query KDTree for k nearest neighbors of each point.
        
        Internal method that performs k-NN search on the time series data.
        Finds k+1 neighbors since each point is its own nearest neighbor.
        
        Parameters
        ----------
        k : int, default=DEFAULT_NN
            Number of nearest neighbors to find (excluding self).
            The actual query uses k+1 to include the point itself.
            
        Returns
        -------
        tuple of (distances, indices)
            distances : ndarray of shape (n_samples, k+1)
                Distances to the k+1 nearest neighbors.
            indices : ndarray of shape (n_samples, k+1)
                Indices of the k+1 nearest neighbors.
                
        Notes
        -----
        Triggers KDTree construction via get_kdtree() if not already built.
        Results are cached by get_kdtree_query() for each k value.        """
        tree = self.get_kdtree()
        # Reshape data for query - KDTree expects 2D
        d = self.data.reshape(self.data.shape[0], -1)
        return tree.query(d, k=k + 1)

    def get_entropy(self, ds=1):
        """Calculate entropy of the time series.
        
        Computes Shannon entropy for the time series data, using appropriate
        methods for discrete vs continuous variables. Results are cached by
        downsampling factor.
        
        Parameters
        ----------
        ds : int, default=1
            Downsampling factor. Data is subsampled by taking every ds-th
            sample before entropy calculation. Must be positive integer.
            
        Returns
        -------
        float
            Entropy value in bits. For discrete variables, uses discrete 
            entropy calculation. For continuous variables, uses non-parametric 
            KSG entropy estimation.
            
        Raises
        ------
        ValueError
            If ds is not a positive integer or if ds >= len(data).
            
        Notes
        -----
        - For discrete data: Uses entropy_d which directly returns bits.
        - For continuous data: Uses nonparam_entropy_c with base=2 to get bits.
        - Results are cached in self.entropy dict keyed by ds value.
        - Downsampling is applied as data[::ds] for both discrete and continuous data.
        
        See Also
        --------
        ~driada.information.entropy.entropy_d : Discrete entropy calculation.
        ~driada.information.ksg.nonparam_entropy_c : Continuous entropy estimation using KSG method.        """
        # Validate downsampling factor
        if not isinstance(ds, int) or ds < 1:
            raise ValueError(f"Downsampling factor ds must be a positive integer, got {ds}")
        if ds >= len(self.data):
            raise ValueError(f"Downsampling factor ds={ds} must be less than data length={len(self.data)}")
            
        if ds not in self.entropy.keys():
            self._compute_entropy(ds=ds)
        return self.entropy[ds]

    def _compute_entropy(self, ds=1):
        """Compute and cache entropy for given downsampling factor.
        
        Internal method that calculates Shannon entropy using appropriate
        estimators based on data type. Results are stored in self.entropy
        dictionary for efficiency.
        
        Parameters
        ----------
        ds : int, default=1
            Downsampling factor. Data is subsampled as data[::ds]
            before entropy calculation.
            
        Notes
        -----
        - For discrete data: Uses entropy_d which returns bits directly
        - For continuous data: Uses nonparam_entropy_c with base=2 for bits
        - Results cached in self.entropy[ds] to avoid recomputation
        - Called by get_entropy() when cached value not available
        
        Side Effects
        ------------
        Modifies self.entropy dictionary by adding entry for key ds.        """
        if self.discrete:
            # Use entropy_d with int_data for efficient computation
            # entropy_d already returns bits, no conversion needed
            self.entropy[ds] = entropy_d(self.int_data[::ds])

        else:
            # Apply downsampling to continuous data as well
            downsampled_data = self.data[::ds] if ds > 1 else self.data
            # Use base=2 to get entropy directly in bits
            self.entropy[ds] = nonparam_entropy_c(downsampled_data, base=2)

    def filter(self, method="gaussian", **kwargs):
        """
        Apply filtering to the time series and return a new filtered TimeSeries.

        Parameters
        ----------
        method : str
            Filtering method: 'gaussian', 'savgol', 'wavelet', or 'none'
        **kwargs
            Method-specific parameters:
            - gaussian: sigma (default: 1.0)
            - savgol: window_length (default: 5), polyorder (default: 2)
            - wavelet: wavelet (default: 'db4'), level (default: None)

        Returns
        -------
        TimeSeries
            New TimeSeries object with filtered data        """
        from ..utils.signals import filter_1d_timeseries

        if method == "none":
            return TimeSeries(
                self.data.copy(),
                ts_type=self.type_info,
                shuffle_mask=self.shuffle_mask.copy(),
            )

        if self.discrete:
            warnings.warn(
                "Filtering discrete time series may produce unexpected results"
            )

        # Apply filtering to 1D time series
        filtered_data = filter_1d_timeseries(self.data, method=method, **kwargs)

        # Create new TimeSeries with filtered data, preserving full type information
        return TimeSeries(
            filtered_data, ts_type=self.type_info, shuffle_mask=self.shuffle_mask.copy()
        )

    def approximate_entropy(self, m: int = 2, r: Optional[float] = None) -> float:
        """
        Calculate approximate entropy (ApEn) of the time series.

        Approximate entropy is a regularity statistic that quantifies the
        unpredictability of fluctuations in a time series. A time series
        containing many repetitive patterns has a relatively small ApEn;
        a less predictable process has a higher ApEn.

        Parameters
        ----------
        m : int, optional
            Pattern length. Common values are 1 or 2. Default is 2.
        r : float, optional
            Tolerance threshold for pattern matching. If None, defaults to
            0.2 times the standard deviation of the data.

        Returns
        -------
        float
            The approximate entropy value. Higher values indicate more
            randomness/complexity.

        Raises
        ------
        ValueError
            If called on a discrete TimeSeries.

        Notes
        -----
        This method is only valid for continuous time series. For discrete
        time series, consider using other complexity measures.

        Examples
        --------
        >>> np.random.seed(42)
        >>> ts = TimeSeries(np.random.randn(1000), discrete=False)
        >>> apen = ts.approximate_entropy(m=2)
        >>> print(f"Approximate entropy: {apen:.3f}")
        Approximate entropy: 1.637
        """
        if self.discrete:
            raise ValueError(
                "approximate_entropy is only valid for continuous time series"
            )

        # Use lazy import to avoid circular imports
        from ..utils.signals import approximate_entropy

        # Default r to 0.2 * std if not provided
        if r is None:
            r = 0.2 * np.std(self.data)

        return approximate_entropy(self.data, m=m, r=r)


class MultiTimeSeries(MVData):
    """Multiple aligned time series with dimensionality reduction capabilities.
    
    Represents a collection of aligned univariate time series, all of the same
    type (either all continuous or all discrete). Inherits from MVData to enable
    direct application of dimensionality reduction methods.
    
    Parameters
    ----------
    data_or_tslist : ndarray or list of TimeSeries
        Either:
        - 2D numpy array of shape (n_series, n_timepoints)
        - List of TimeSeries objects (must all be same type)
    labels : array-like, optional
        Labels for dimensionality reduction visualization.
    distmat : array-like, optional
        Precomputed distance matrix.
    rescale_rows : bool, default=False
        Whether to rescale each time series to [0, 1].
    data_name : str, optional
        Name for the dataset.
    downsampling : int, optional
        Downsampling factor.
    discrete : bool, optional
        Required when passing numpy array. Specifies if all series are discrete.
    shuffle_mask : array-like, optional
        Boolean mask for valid shuffle positions. If not provided, combines
        individual TimeSeries masks using AND operation.
    allow_zero_columns : bool, default=False
        Whether to allow time points where all series are zero.
        
    Attributes
    ----------
    data : ndarray
        Stacked time series data of shape (n_series, n_timepoints).
    discrete : bool
        Whether all time series are discrete.
    scdata : ndarray
        Min-max scaled data for each series.
    copula_normal_data : ndarray or None
        Copula-normalized data (continuous only).
    int_data : ndarray or None
        Integer representation (discrete only).
    shuffle_mask : ndarray
        Combined shuffle mask for all series.
    entropy : dict
        Cached joint entropy values for different downsampling factors.
    shape : tuple
        Shape of the data array.
        
    Plus all attributes from MVData parent class.
        
    Methods
    -------
    get_entropy(ds=1)
        Compute joint entropy with optional downsampling.
    filter(method='gaussian', **kwargs)
        Apply filtering to all series and return new MultiTimeSeries.
        
    Plus all methods from MVData parent class (get_embedding, etc.).
        
    Raises
    ------
    ValueError
        If mixing continuous and discrete TimeSeries.
        If TimeSeries have different lengths.
        If combined shuffle_mask has no valid positions.
        If data is not 2D when providing numpy array.
        If discrete parameter is not specified when providing numpy array.
        
    Notes
    -----
    - All component time series must be of the same type (continuous/discrete)
    - Shuffle masks are combined restrictively (AND operation)
    - Joint entropy is computed for up to 2 discrete variables
    - Inherits dimensionality reduction capabilities from MVData
    
    Examples
    --------
    >>> import numpy as np
    >>> # From numpy array
    >>> data = np.random.randn(10, 1000)  # 10 time series, 1000 points each
    >>> mts = MultiTimeSeries(data, discrete=False)
    >>> 
    >>> # From list of TimeSeries
    >>> ts_list = [TimeSeries(np.random.randn(1000)) for _ in range(5)]
    >>> mts = MultiTimeSeries(ts_list)
    >>> 
    >>> # Apply dimensionality reduction
    >>> embedding = mts.get_embedding(method='pca', dim=2)  # doctest: +ELLIPSIS
    Calculating PCA embedding...
    >>> 
    >>> # Filter all time series
    >>> mts_filtered = mts.filter(method='gaussian', sigma=2.0)    """

    def __init__(
        self,
        data_or_tslist,
        labels=None,
        distmat=None,
        rescale_rows=False,
        data_name=None,
        downsampling=None,
        discrete=None,
        shuffle_mask=None,
        allow_zero_columns=False,
    ):
        """Initialize MultiTimeSeries object.
        
        Parameters
        ----------
        data_or_tslist : ndarray or list of TimeSeries
            Either a 2D numpy array of shape (n_series, n_timepoints) or
            a list of TimeSeries objects (must all be same type)
        labels : array-like, optional
            Labels for dimensionality reduction visualization
        distmat : array-like, optional
            Precomputed distance matrix
        rescale_rows : bool, default=False
            Whether to rescale each time series to [0, 1]
        data_name : str, optional
            Name for the dataset
        downsampling : int, optional
            Downsampling factor
        discrete : bool, optional
            Required when passing numpy array. Specifies if all series are discrete
        shuffle_mask : array-like, optional
            Boolean mask for valid shuffle positions
        allow_zero_columns : bool, default=False
            Whether to allow time points where all series are zero
        
        Notes
        -----
        This method handles both numpy array and list of TimeSeries inputs,
        validates compatibility, combines data and metadata, and initializes
        the parent MVData class for dimensionality reduction capabilities.
        """
        # Handle both numpy array and list of TimeSeries inputs
        if isinstance(data_or_tslist, np.ndarray):
            # Direct numpy array input: each row is a time series
            if data_or_tslist.ndim != 2:
                raise ValueError(
                    "When providing numpy array, it must be 2D with shape (n_series, n_timepoints)"
                )
            if discrete is None:
                raise ValueError(
                    "When providing numpy array, 'discrete' parameter must be specified"
                )

            # Set discrete flag early for numpy array input
            self.discrete = discrete

            # Create TimeSeries objects from numpy array rows for processing
            tslist = [
                TimeSeries(data_or_tslist[i, :], discrete=discrete)
                for i in range(data_or_tslist.shape[0])
            ]
            data = data_or_tslist

            # Store provided shuffle_mask for later use (after combining with TimeSeries masks)
            self._provided_shuffle_mask = shuffle_mask
        else:
            # List of TimeSeries objects
            tslist = data_or_tslist
            self._check_input(tslist)
            # Stack data from all TimeSeries
            data = np.vstack([ts.data for ts in tslist])

            # Store provided shuffle_mask for later use
            self._provided_shuffle_mask = shuffle_mask

        # Store allow_zero_columns for later use (e.g., in filter method)
        self.allow_zero_columns = allow_zero_columns
        
        # Initialize MVData parent class
        super().__init__(
            data,
            labels=labels,
            distmat=distmat,
            rescale_rows=rescale_rows,
            data_name=data_name,
            downsampling=downsampling,
            allow_zero_columns=allow_zero_columns,
        )

        # Additional MultiTimeSeries specific attributes
        self.scdata = np.vstack([ts.scdata for ts in tslist])

        # Handle copula normal data for continuous components
        if not self.discrete:
            self.copula_normal_data = np.vstack(
                [ts.copula_normal_data for ts in tslist]
            )
        else:
            # For discrete MultiTimeSeries, store integer data
            self.int_data = np.vstack([ts.int_data for ts in tslist])
            self.copula_normal_data = None

        # Combine shuffle masks
        if (
            hasattr(self, "_provided_shuffle_mask")
            and self._provided_shuffle_mask is not None
        ):
            # If shuffle_mask was provided explicitly, use it
            self.shuffle_mask = self._provided_shuffle_mask
            if not np.any(self.shuffle_mask):
                warnings.warn(
                    "Provided shuffle_mask has no valid positions for shuffling!"
                )
        else:
            # Otherwise, combine individual TimeSeries masks restrictively
            shuffle_masks = np.vstack([ts.shuffle_mask for ts in tslist])
            # Restrictive combination: ALL masks must allow shuffling at a position
            self.shuffle_mask = np.all(shuffle_masks, axis=0)

            # Check if the combined mask is problematic
            valid_positions = np.sum(self.shuffle_mask)
            total_positions = len(self.shuffle_mask)

            if valid_positions == 0:
                raise ValueError(
                    "Combined shuffle_mask has NO valid positions for shuffling! "
                    "This typically happens when combining many neurons with restrictive individual masks. "
                    "Consider providing an explicit shuffle_mask parameter to MultiTimeSeries."
                )
            elif valid_positions < 0.1 * total_positions:
                warnings.warn(
                    f"Combined shuffle_mask is extremely restrictive: only {valid_positions}/{total_positions} "
                    f"({100*valid_positions/total_positions:.1f}%) positions are valid for shuffling. "
                    f"This may cause issues with shuffle-based significance testing."
                )

        self.entropy = dict()  # supports various downsampling constants

    @property
    def shape(self):
        """Return shape of the data for compatibility with numpy-like access.
        
        Returns
        -------
        tuple of int
            Shape as (n_variables, n_timepoints).        """
        return self.data.shape

    def _check_input(self, tslist):
        """Validate list of TimeSeries for MultiTimeSeries construction.
        
        Internal method that ensures all TimeSeries components are valid
        and compatible for creating a MultiTimeSeries object.
        
        Parameters
        ----------
        tslist : list of TimeSeries
            List of TimeSeries objects to validate.
            
        Raises
        ------
        ValueError
            If any element is not a TimeSeries instance.
            If TimeSeries have different lengths.
            If mixing discrete and continuous TimeSeries.
            
        Side Effects
        ------------
        Sets self.discrete based on the type of component TimeSeries.
        
        Notes
        -----
        All components must be either all discrete or all continuous.
        The discrete/continuous nature is determined from the first element
        and verified against all others.        """
        is_ts = np.array([isinstance(ts, TimeSeries) for ts in tslist])
        if not np.all(is_ts):
            raise ValueError("Input to MultiTimeSeries must be iterable of TimeSeries")

        # Check all TimeSeries have same length
        lengths = np.array([len(ts.data) for ts in tslist])
        if not np.all(lengths == lengths[0]):
            raise ValueError("All TimeSeries must have the same length")

        # Check all TimeSeries have same discrete/continuous type
        is_discrete = np.array([ts.discrete for ts in tslist])
        if not (np.all(is_discrete) or np.all(~is_discrete)):
            raise ValueError(
                "All components of MultiTimeSeries must be either continuous or discrete (no mixing)"
            )

        # Set discrete flag based on components
        self.discrete = is_discrete[0]

    def get_entropy(self, ds=1):
        """Calculate joint entropy of the multivariate time series.
        
        Computes joint Shannon entropy for multivariate data, using appropriate
        methods based on whether the variables are discrete or continuous.
        Results are cached by downsampling factor.
        
        Parameters
        ----------
        ds : int, default=1
            Downsampling factor. Data is subsampled by taking every ds-th
            sample before entropy calculation. Must be positive integer.
            
        Returns
        -------
        float
            Joint entropy value in bits. The method used depends on data type:
            - Discrete data: Uses entropy_d for single variable, joint_entropy_dd 
              for 2 variables. More than 2 discrete variables not yet supported.
            - Continuous data: Uses Gaussian copula entropy estimation (ent_g)
              for any number of variables.
            
        Raises
        ------
        NotImplementedError
            If attempting to compute joint entropy for more than 2 discrete
            variables.
            
        Notes
        -----
        - Results are cached in self.entropy dict keyed by ds value.
        - Downsampling is applied as data[:, ::ds] before computation.
        - For continuous multivariate data, uses copula-based entropy estimation
          which is more efficient than KSG for multiple variables.
        
        See Also
        --------
        ~driada.information.entropy.entropy_d : Single discrete variable entropy.
        ~driada.information.entropy.joint_entropy_dd : Joint entropy for 2 discrete variables.
        ~driada.information.gcmi.ent_g : Gaussian copula entropy for continuous multivariate data.        """
        if ds not in self.entropy.keys():
            self._compute_entropy(ds=ds)
        return self.entropy[ds]

    def _compute_entropy(self, ds=1):
        """Compute and cache joint entropy for multivariate time series.
        
        Internal method that calculates joint Shannon entropy using
        appropriate estimators based on data type and dimensionality.
        
        Parameters
        ----------
        ds : int, default=1
            Downsampling factor. Data is subsampled as data[:, ::ds]
            before entropy calculation.
            
        Raises
        ------
        NotImplementedError
            If attempting joint entropy for more than 2 discrete variables.
            
        Notes
        -----
        - Discrete data: Uses entropy_d (1 var) or joint_entropy_dd (2 vars)
        - Continuous data: Uses ent_g (Gaussian copula entropy) for any dimension
        - Results cached in self.entropy[ds] to avoid recomputation
        - Called by get_entropy() when cached value not available
        
        Side Effects
        ------------
        Modifies self.entropy dictionary by adding entry for key ds.
        May import joint_entropy_dd lazily for 2-variable discrete case.        """
        if self.discrete:
            # All components are discrete - compute joint entropy
            if self.n_dim == 1:
                # Single variable - use regular discrete entropy
                self.entropy[ds] = entropy_d(self.int_data[0, ::ds])
            elif self.n_dim == 2:
                # Two variables - use joint_entropy_dd
                from .entropy import joint_entropy_dd

                self.entropy[ds] = joint_entropy_dd(
                    self.int_data[0, ::ds], self.int_data[1, ::ds]
                )
            else:
                # Multiple discrete variables not yet supported
                raise NotImplementedError(
                    f"Joint entropy for {self.n_dim} discrete variables is not yet implemented"
                )
        else:
            # All continuous - use existing continuous entropy
            self.entropy[ds] = ent_g(self.data[:, ::ds])

    def filter(self, method="gaussian", **kwargs):
        """
        Apply filtering to all time series components and return a new filtered MultiTimeSeries.

        Parameters
        ----------
        method : str
            Filtering method: 'gaussian', 'savgol', 'wavelet', or 'none'
        **kwargs
            Method-specific parameters (see TimeSeries.filter for details)

        Returns
        -------
        MultiTimeSeries
            New MultiTimeSeries object with all components filtered        """
        from ..utils.signals import filter_signals

        if method == "none":
            # Return a copy without filtering
            return MultiTimeSeries(
                self.data.copy(),
                labels=self.labels.copy() if self.labels is not None else None,
                rescale_rows=False,
                data_name=self.data_name,
                discrete=self.discrete,
                shuffle_mask=self.shuffle_mask.copy(),
                allow_zero_columns=self.allow_zero_columns,  # Inherit from original
            )

        if self.discrete:
            warnings.warn(
                "Filtering discrete MultiTimeSeries may produce unexpected results"
            )

        # Apply filtering to all time series at once
        filtered_data = filter_signals(self.data, method=method, **kwargs)

        # Create new MultiTimeSeries from filtered data
        return MultiTimeSeries(
            filtered_data,
            labels=self.labels.copy() if self.labels is not None else None,
            rescale_rows=False,
            data_name=self.data_name,
            discrete=self.discrete,
            shuffle_mask=self.shuffle_mask.copy(),
            allow_zero_columns=self.allow_zero_columns,  # Inherit from original
        )


def get_stats_function(sname):
    """Get a statistical function from scipy.stats by name.
    
    Parameters
    ----------
    sname : str
        Name of the function in scipy.stats module (e.g., 'pearsonr', 'spearmanr').
        
    Returns
    -------
    callable
        The requested function from scipy.stats.
        
    Raises
    ------
    ValueError
        If the function name is not found in scipy.stats.
        
    Examples
    --------
    >>> func = get_stats_function('pearsonr')
    >>> r, p = func([1, 2, 3], [2, 4, 6])    """
    try:
        return getattr(scipy.stats, sname)
    except AttributeError:
        raise ValueError(f"Metric '{sname}' not found in scipy.stats")


def calc_signal_ratio(binary_ts, continuous_ts):
    """Calculate signal-to-baseline ratio for binary-gated continuous signal.
    
    Computes the ratio of the average continuous signal value when the binary
    signal is ON (1) versus when it is OFF (0). Useful for quantifying
    the modulation of a continuous signal by a binary event.
    
    Parameters
    ----------
    binary_ts : ndarray
        Binary time series with values 0 and 1, indicating OFF and ON states.
    continuous_ts : ndarray
        Continuous signal values. Must have same length as binary_ts.
        
    Returns
    -------
    float
        Ratio of average signal when ON to average signal when OFF.
        Returns np.inf if baseline (OFF) is zero but ON signal is non-zero.
        Returns np.nan if both ON and OFF averages are zero.
        
    Examples
    --------
    >>> binary = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    >>> continuous = np.array([1, 1, 5, 6, 2, 1, 4, 5])
    >>> ratio = calc_signal_ratio(binary, continuous)
    >>> print(f"Signal is {ratio:.1f}x higher when ON")
    Signal is 4.0x higher when ON
    """
    # Calculate average of continuous_ts when binary_ts is 1 or 0
    avg_on = np.mean(continuous_ts[binary_ts == 1])
    avg_off = np.mean(continuous_ts[binary_ts == 0])

    # Calculate ratio (handle division by zero)
    if avg_off == 0:
        return np.inf if avg_on != 0 else np.nan

    return avg_on / avg_off


def get_sim(
    x, y, metric, shift=0, ds=1, k=5, estimator="gcmi", check_for_coincidence=False
):
    """Computes similarity between two (possibly multidimensional) variables efficiently

    Parameters
    ----------
    x : TimeSeries, MultiTimeSeries, or numpy.ndarray
        First time series. If numpy array, will be converted to TimeSeries (1D) 
        or MultiTimeSeries (2D+).
    y : TimeSeries, MultiTimeSeries, or numpy.ndarray
        Second time series. If numpy array, will be converted to TimeSeries (1D) 
        or MultiTimeSeries (2D+).
    metric : str
        Similarity metric to compute. Options include:
        - 'mi': Mutual information (supports multivariate data)
        - 'spearman', 'pearson', 'kendall': Correlation coefficients (univariate only)
        - 'av': Activity ratio (requires one binary and one continuous variable)
        - 'fast_pearsonr': Fast Pearson correlation (univariate only)
        - Any scipy.stats correlation function name (univariate only)
    shift : int, optional
        Time shift to apply to y before computing similarity. Positive values 
        shift y forward in time. Default is 0.
    ds : int, optional
        Downsampling factor. Only every ds-th sample is used. Default is 1.
    k : int, optional
        Number of nearest neighbors for KSG mutual information estimator. Only 
        used when metric='mi' and estimator='ksg'. Default is 5.
    estimator : {'gcmi', 'ksg'}, optional
        Estimator to use for mutual information calculation. Only used when 
        metric='mi'. Default is 'gcmi'.
    check_for_coincidence : bool, optional
        Whether to check if x and y contain identical data (which would result 
        in infinite MI). Only used for MI calculation. Default is False.

    Returns
    -------
    similarity : float
        Similarity value between x and (possibly shifted) y. The interpretation 
        depends on the metric:
        - MI: Non-negative value in bits
        - Correlations: Value between -1 and 1
        - Activity ratio: Non-negative ratio
        
    Raises
    ------
    ValueError
        If metric is not supported for the given variable types (e.g., 'av' requires
        one binary and one continuous variable).
        If trying to use correlation metrics with multivariate data.
    Exception
        If multidimensional inputs are not provided as MultiTimeSeries.    """

    def _check_input(ts):
        """Convert array input to TimeSeries/MultiTimeSeries if needed.
        
        Internal helper for get_sim to ensure inputs are proper time series objects.
        
        Parameters
        ----------
        ts : TimeSeries, MultiTimeSeries, or array-like
            Input to validate/convert.
            
        Returns
        -------
        TimeSeries or MultiTimeSeries
            Validated time series object.
            
        Raises
        ------
        Exception
            If multidimensional array not provided as MultiTimeSeries.        """
        if not isinstance(ts, TimeSeries) and not isinstance(ts, MultiTimeSeries):
            if np.ndim(ts) == 1:
                ts = TimeSeries(ts)
            else:
                raise Exception(
                    "Multidimensional inputs must be provided as MultiTimeSeries"
                )
        return ts

    ts1 = _check_input(x)
    ts2 = _check_input(y)

    if metric == "mi":
        me = get_mi(
            ts1,
            ts2,
            shift=shift,
            ds=ds,
            k=k,
            estimator=estimator,
            check_for_coincidence=check_for_coincidence,
        )

    else:
        if isinstance(ts1, TimeSeries) and isinstance(ts2, TimeSeries):
            if not ts1.discrete and not ts2.discrete:
                if metric == "fast_pearsonr":
                    x = ts1.data[::ds]
                    y = np.roll(ts2.data[::ds], shift)
                    me = correlation_matrix(np.vstack([x, y]))[0, 1]
                else:
                    metric_func = get_stats_function(metric)
                    me = metric_func(ts1.data[::ds], np.roll(ts2.data[::ds], shift))[0]

            if ts1.discrete and not ts2.discrete:
                if metric == "av":
                    if ts1.is_binary:
                        me = calc_signal_ratio(
                            ts1.data[::ds], np.roll(ts2.data[::ds], shift)
                        )
                    else:
                        raise ValueError(
                            f"First TimeSeries (ts1) must be binary for metric='{metric}', "
                            f"but has {len(np.unique(ts1.int_data))} unique values"
                        )
                else:
                    raise ValueError(
                        f"Only 'av' and 'mi' metrics are supported for discrete-continuous pairs. "
                        f"Got metric='{metric}' with discrete ts1 and continuous ts2"
                    )

            if ts2.discrete and not ts1.discrete:
                if metric == "av":
                    if ts2.is_binary:
                        me = calc_signal_ratio(
                            ts2.data[::ds], np.roll(ts1.data[::ds], shift)
                        )
                    else:
                        raise ValueError(
                            f"Second TimeSeries (ts2) must be binary for metric='{metric}', "
                            f"but has {len(np.unique(ts2.int_data))} unique values"
                        )
                else:
                    raise ValueError(
                        f"Only 'av' and 'mi' metrics are supported for continuous-discrete pairs. "
                        f"Got metric='{metric}' with continuous ts1 and discrete ts2"
                    )

            if ts2.discrete and ts1.discrete:
                raise ValueError(
                    f"Metric={metric} is not supported for two discrete ts"
                )

        else:
            raise Exception(
                "Metrics except 'mi' are not supported for multi-dimensional data"
            )

    return me


def get_mi(x, y, shift=0, ds=1, k=5, estimator="gcmi", check_for_coincidence=False):
    """Compute mutual information between two (possibly multidimensional) variables.
    
    Efficiently calculates mutual information (MI) between continuous, discrete,
    or mixed-type variables. Supports both univariate and multivariate inputs,
    with time-shifted analysis capabilities for temporal dependencies.
    
    Parameters
    ----------
    x : TimeSeries, MultiTimeSeries, or array-like
        First variable. Can be:
        - TimeSeries: univariate time series (continuous or discrete)
        - MultiTimeSeries: multivariate time series
        - array-like: converted to TimeSeries internally
    y : TimeSeries, MultiTimeSeries, or array-like
        Second variable. Must have same length as x.
    shift : int, default=0
        Number of samples to shift y after downsampling. Positive values
        shift y forward in time (y leads x). Used for time-delayed MI.
    ds : int, default=1
        Downsampling factor. Takes every ds-th sample to reduce computation.
        Note: for GCMI with ds>1, copula transform is applied before downsampling
        which may affect accuracy for large ds or non-smooth signals.
    k : int, default=5
        Number of nearest neighbors for KSG estimator. Common values:
        - k=4-5: optimal for most applications
        - k=3-10: for low dimensions (d≤3)
        - k=10-20: for higher dimensions
    estimator : {'gcmi', 'ksg'}, default='gcmi'
        MI estimation method:
        - 'gcmi': Gaussian Copula MI (fast, gives lower bound)
        - 'ksg': Kraskov-Stögbauer-Grassberger (slower, more accurate)
    check_for_coincidence : bool, default=False
        If True, checks for MI(X,X) computation and handles appropriately:
        - For discrete single TimeSeries: returns H(X) (well-defined)
        - For continuous variables: raises ValueError (MI would be infinite)
        - For discrete MultiTimeSeries: raises NotImplementedError (not yet supported)
        Set to False to bypass this check (use with caution).
        
    Returns
    -------
    float
        Mutual information in bits. Always non-negative (clipped at 0).
        For GCMI, this is a lower bound on the true MI.
        
    Notes
    -----
    The function automatically handles different variable type combinations:
    - Continuous-Continuous: Uses GCMI or KSG as specified
    - Discrete-Discrete: Uses exact MI computation (same for both estimators)
    - Mixed (Continuous-Discrete): Uses appropriate mixed estimators
    - Multivariate: Supported for continuous variables only
    
    For discrete-discrete MI, the estimator parameter is ignored since MI
    can be computed exactly from the joint probability distribution.
    
    GCMI is recommended for most applications as it's much faster and provides
    a useful lower bound. KSG is more accurate but computationally expensive,
    especially for large datasets.
    
    Examples
    --------
    >>> # Simple correlation detection
    >>> np.random.seed(42)
    >>> x = np.random.randn(1000)
    >>> y = x + np.random.randn(1000) * 0.5
    >>> mi = get_mi(x, y)
    >>> print(f"MI = {mi:.3f} bits")
    MI = 1.114 bits
    
    >>> # Time-delayed mutual information
    >>> ts1 = TimeSeries(np.sin(np.linspace(0, 10*np.pi, 1000)))
    >>> ts2 = TimeSeries(np.sin(np.linspace(0, 10*np.pi, 1000) + np.pi/4))
    >>> mi_delay = get_mi(ts1, ts2, shift=25)  # Check 25-sample delay
    
    >>> # Multivariate MI
    >>> mts1 = MultiTimeSeries(np.random.randn(3, 1000), discrete=False)
    >>> mts2 = MultiTimeSeries(np.random.randn(2, 1000), discrete=False)
    >>> mi_multi = get_mi(mts1, mts2)
    
    See Also
    --------
    ~driada.information.info_base.get_1d_mi : MI for univariate time series (called internally)
    ~driada.information.info_base.get_multi_mi : MI between multiple and single time series
    ~driada.information.info_base.get_tdmi : Time-delayed MI for finding optimal embedding delays
    ~driada.information.info_base.conditional_mi : Conditional mutual information I(X;Y|Z)    """

    def _check_input(ts):
        """Convert array input to TimeSeries/MultiTimeSeries if needed.
        
        Internal helper for get_mi to ensure inputs are proper time series objects.
        
        Parameters
        ----------
        ts : TimeSeries, MultiTimeSeries, or array-like
            Input to validate/convert.
            
        Returns
        -------
        TimeSeries or MultiTimeSeries
            Validated time series object.
            
        Raises
        ------
        Exception
            If multidimensional array not provided as MultiTimeSeries.        """
        if not isinstance(ts, TimeSeries) and not isinstance(ts, MultiTimeSeries):
            if np.ndim(ts) == 1:
                ts = TimeSeries(ts)
            else:
                raise Exception(
                    "Multidimensional inputs must be provided as MultiTimeSeries"
                )
        return ts

    def multi_single_mi(mts, ts, shift=0, ds=1, k=5, estimator="gcmi"):
        """
        Calculate mutual information between a MultiTimeSeries and a single TimeSeries.
        
        Computes MI(X;Y) where X is multivariate (MultiTimeSeries) and Y is univariate
        (TimeSeries). Supports both continuous and discrete Y variables.
        
        Parameters
        ----------
        mts : MultiTimeSeries
            Multivariate time series (X). Must be continuous.
        ts : TimeSeries
            Univariate time series (Y). Can be continuous or discrete.
        shift : int, optional
            Time shift to apply to ts before computing MI. Positive values shift
            ts forward in time. Default is 0.
        ds : int, optional
            Downsampling factor. Only every ds-th sample is used. Default is 1.
        k : int, optional
            Number of nearest neighbors for KSG estimator (not currently supported
            for multivariate data). Default is 5.
        estimator : {'gcmi', 'ksg'}, optional
            MI estimator to use. Currently only 'gcmi' is supported for multivariate
            data. Default is 'gcmi'.
            
        Returns
        -------
        mi : float
            Mutual information MI(X;Y) in bits.
            
        Raises
        ------
        NotImplementedError
            If estimator='ksg' is requested (not supported for multivariate).
            
        Warns
        -----
        UserWarning
            If the MultiTimeSeries contains the same data as the TimeSeries,
            which would result in infinite MI. Returns 0 in this case.
            
        Notes
        -----
        Uses Gaussian copula MI (GCMI) estimation for efficiency. The data is
        transformed to a Gaussian copula representation before MI calculation.
        For discrete Y, uses the mixed continuous-discrete GCMI estimator.        """
        if estimator == "ksg":
            raise NotImplementedError("KSG estimator is not supported for dim>1 yet")

        # Safety check: if single TimeSeries data is contained in MultiTimeSeries
        # This should not happen due to aggregate_multiple_ts adding noise, but add as safety net
        if not ts.discrete and shift == 0:
            # Check if any row of the MultiTimeSeries matches the single TimeSeries
            for i in range(mts.data.shape[0]):
                if np.allclose(
                    mts.data[i, ::ds], ts.data[::ds], rtol=1e-10, atol=1e-10
                ):
                    warnings.warn(
                        "MI computation between MultiTimeSeries containing identical data detected, returning 0"
                    )
                    return 0.0

        if ts.discrete:
            ny1 = mts.copula_normal_data[:, ::ds]
            ny2 = np.roll(ts.int_data[::ds], shift)
            # Ensure ny1 is contiguous for better performance with Numba
            if not ny1.flags["C_CONTIGUOUS"]:
                ny1 = np.ascontiguousarray(ny1)
            # Fix: mi_model_gd expects Ym to be the number of discrete states (max + 1)
            Ym = int(np.max(ny2) + 1)
            mi = mi_model_gd(ny1, ny2, Ym=Ym, biascorrect=True, demeaned=True)

        else:
            ny1 = mts.copula_normal_data[:, ::ds]
            ny2 = np.roll(ts.copula_normal_data[::ds], shift)
            mi = mi_gg(ny1, ny2, True, True)

        return mi

    def multi_multi_mi(
        mts1, mts2, shift=0, ds=1, k=5, estimator="gcmi", check_for_coincidence=False
    ):
        """
        Calculate mutual information between two MultiTimeSeries.
        
        Computes MI(X;Y) where both X and Y are multivariate (MultiTimeSeries).
        Currently only supports continuous variables.
        
        Parameters
        ----------
        mts1 : MultiTimeSeries
            First multivariate time series (X). Must be continuous.
        mts2 : MultiTimeSeries
            Second multivariate time series (Y). Must be continuous.
        shift : int, optional
            Time shift to apply to mts2 before computing MI. Positive values shift
            mts2 forward in time. Default is 0.
        ds : int, optional
            Downsampling factor. Only every ds-th sample is used. Default is 1.
        k : int, optional
            Number of nearest neighbors for KSG estimator (not currently supported
            for multivariate data). Default is 5.
        estimator : {'gcmi', 'ksg'}, optional
            MI estimator to use. Currently only 'gcmi' is supported for multivariate
            data. Default is 'gcmi'.
        check_for_coincidence : bool, optional
            Whether to check if mts1 and mts2 contain identical data (which would
            result in infinite MI). Default is False.
            
        Returns
        -------
        mi : float
            Mutual information MI(X;Y) in bits.
            
        Raises
        ------
        NotImplementedError
            If estimator='ksg' is requested (not supported for multivariate).
            
        Warns
        -----
        UserWarning
            If check_for_coincidence=True and the two MultiTimeSeries contain
            identical data with shift=0. Returns 0 in this case.
            
        Notes
        -----
        Uses Gaussian copula MI (GCMI) estimation for efficiency. The data is
        transformed to a Gaussian copula representation before MI calculation.
        The GCMI method is particularly well-suited for high-dimensional data
        as it avoids the curse of dimensionality associated with kernel-based
        estimators.        """
        if estimator == "ksg":
            raise NotImplementedError("KSG estimator is not supported for dim>1 yet")

        if check_for_coincidence:
            if np.allclose(mts1.data, mts2.data) and shift == 0:
                if mts1.discrete and mts2.discrete:
                    # For discrete multivariate variables, MI(X,X) = H(X) is well-defined
                    # but not yet implemented for multivariate discrete entropy
                    raise NotImplementedError(
                        "MI(X,X) for discrete MultiTimeSeries requires multivariate discrete entropy "
                        "calculation, which is not yet implemented. This will be added in a future version."
                    )
                else:
                    # For continuous variables, MI(X,X) is infinite
                    raise ValueError(
                        "MI(X,X) for continuous variables is infinite and cannot be computed. "
                        "See https://math.stackexchange.com/questions/2809880/"
                    )

        if mts1.discrete or mts2.discrete:
            raise NotImplementedError(
                "MI computation between MultiTimeSeries\
             is currently supported for continuous data only"
            )

        else:
            ny1 = mts1.copula_normal_data[:, ::ds]
            ny2 = np.roll(mts2.copula_normal_data[:, ::ds], shift, axis=1)
            mi = mi_gg(ny1, ny2, True, True)

        return mi

    ts1 = _check_input(x)
    ts2 = _check_input(y)

    if isinstance(ts1, TimeSeries) and isinstance(ts2, TimeSeries):
        mi = get_1d_mi(
            x,
            y,
            shift=shift,
            ds=ds,
            k=k,
            estimator=estimator,
            check_for_coincidence=check_for_coincidence,
        )

    if isinstance(ts1, MultiTimeSeries) and isinstance(ts2, TimeSeries):
        mi = multi_single_mi(ts1, ts2, shift=shift, ds=ds, k=k, estimator=estimator)

    if isinstance(ts2, MultiTimeSeries) and isinstance(ts1, TimeSeries):
        mi = multi_single_mi(ts2, ts1, shift=shift, ds=ds, k=k, estimator=estimator)

    if isinstance(ts1, MultiTimeSeries) and isinstance(ts2, MultiTimeSeries):
        mi = multi_multi_mi(
            ts1,
            ts2,
            shift=shift,
            ds=ds,
            k=k,
            estimator=estimator,
            check_for_coincidence=check_for_coincidence,
        )
        # raise NotImplementedError('MI computation between two MultiTimeSeries is not supported yet')

    if mi < 0:
        mi = 0.0

    return mi


def get_1d_mi(
    ts1, ts2, shift=0, ds=1, k=5, estimator="gcmi", check_for_coincidence=True
):
    """Computes mutual information between two 1d variables efficiently

    Parameters
    ----------
    ts1 : TimeSeries/MultiTimeSeries instance or numpy array
        First time series or variable
    ts2 : TimeSeries/MultiTimeSeries instance or numpy array
        Second time series or variable
    shift : int, default=0
        ts2 will be roll-moved by the number 'shift' after downsampling by 'ds' factor
    ds : int, default=1
        downsampling constant (take every 'ds'-th point)
    k : int, default=5
        number of neighbors for ksg estimator
    estimator : str, default='gcmi'
        Estimation method. Should be 'ksg' (accurate but slow) and 'gcmi' (fast, but estimates the lower bound on MI).
        In most cases 'gcmi' should be preferred.
        
        Note on downsampling with GCMI: For performance reasons, when ds > 1, the copula transformation
        is applied to the full data before downsampling. This is an approximation that works well for
        small downsampling factors (ds ≤ 5) and smooth signals, but may introduce inaccuracies for
        large downsampling factors or highly variable signals.
    check_for_coincidence : bool, default=True
        If True, checks for MI(X,X) computation at zero shift:
        - For discrete variables: returns H(X)
        - For continuous variables: raises ValueError (MI is infinite)
        Default: True.

    Returns
    -------
    mi: float
        Mutual information in bits (or its lower bound in case of 'gcmi' estimator) 
        between ts1 and (possibly) shifted ts2. Both estimators return values in bits.    """

    def _check_input(ts):
        """Convert array input to TimeSeries/MultiTimeSeries if needed.
        
        Internal helper for get_1d_mi to ensure inputs are proper time series objects.
        
        Parameters
        ----------
        ts : TimeSeries, MultiTimeSeries, or array-like
            Input to validate/convert.
            
        Returns
        -------
        TimeSeries or MultiTimeSeries
            Validated time series object.
            
        Raises
        ------
        Exception
            If multidimensional array not provided as MultiTimeSeries.        """
        if not isinstance(ts, TimeSeries) and not isinstance(ts, MultiTimeSeries):
            if np.ndim(ts) == 1:
                ts = TimeSeries(ts)
            else:
                raise Exception(
                    "Multidimensional inputs must be provided as MultiTimeSeries"
                )
        return ts

    ts1 = _check_input(ts1)
    ts2 = _check_input(ts2)

    if check_for_coincidence and ts1.data.shape == ts2.data.shape:
        if np.allclose(ts1.data, ts2.data) and shift == 0:
            if ts1.discrete:
                # For discrete variables, MI(X,X) = H(X) is well-defined
                return ts1.get_entropy()
            else:
                # For continuous variables, MI(X,X) is infinite
                raise ValueError(
                    "MI(X,X) for continuous variables is infinite and cannot be computed. "
                    "See https://math.stackexchange.com/questions/2809880/"
                )

    if estimator == "ksg":
        x = ts1.data[::ds].reshape(-1, 1)
        y = ts2.data[::ds]
        if shift != 0:
            y = np.roll(y, shift)

        if not ts1.discrete and not ts2.discrete:
            # FUTURE: Implement downsampled KD-trees for better performance
            # Currently, precomputed trees cannot be used with downsampling as they
            # are built on full data while MI is computed on downsampled data
            mi = nonparam_mi_cc(
                x,  # Use the reshaped x, not ts1.data[::ds]
                y,
                k=k,
                base=2,  # Use base=2 to get MI in bits
                precomputed_tree_x=None if ds > 1 else ts1.get_kdtree(),
                precomputed_tree_y=None if ds > 1 else ts2.get_kdtree(),
            )

        elif ts1.discrete and ts2.discrete:
            # Both discrete - use exact computation
            x1_discrete = ts1.int_data[::ds]
            y2_discrete = ts2.int_data[::ds]
            if shift != 0:
                y2_discrete = np.roll(y2_discrete, shift)
            # For discrete-discrete MI, we don't need KSG estimator
            # Use exact plugin estimator from mutual_info_score
            # Note: mutual_info_score returns nats, convert to bits
            mi = mutual_info_score(x1_discrete, y2_discrete) / np.log(2)

        elif ts1.discrete and not ts2.discrete:
            # X is discrete, Y is continuous
            x1_discrete = ts1.int_data[::ds]
            if shift != 0:
                x1_discrete = np.roll(x1_discrete, shift)
            mi = nonparam_mi_dc(x1_discrete, y, k=k, base=2)  # Use base=2 for bits

        elif not ts1.discrete and ts2.discrete:
            # X is continuous, Y is discrete
            y2_discrete = ts2.int_data[::ds]
            if shift != 0:
                y2_discrete = np.roll(y2_discrete, shift)
            mi = nonparam_mi_cd(x, y2_discrete, k=k, base=2)  # Use base=2 for bits

        return mi

    elif estimator == "gcmi":
        if not ts1.discrete and not ts2.discrete:
            ny1 = ts1.copula_normal_data[::ds]
            ny2 = np.roll(ts2.copula_normal_data[::ds], shift)
            mi = mi_gg(ny1, ny2, True, True)

        elif ts1.discrete and ts2.discrete:
            # if features are binary:
            if ts1.is_binary and ts2.is_binary:
                ny1 = ts1.bool_data[::ds]
                ny2 = np.roll(ts2.bool_data[::ds], shift)

                contingency = np.zeros((2, 2))
                contingency[0, 0] = (ny1 & ny2).sum()
                contingency[0, 1] = (~ny1 & ny2).sum()
                contingency[1, 0] = (ny1 & ~ny2).sum()
                contingency[1, 1] = (~ny1 & ~ny2).sum()

                mi = binary_mi_score(contingency)

            else:
                ny1 = ts1.int_data[::ds]  # .reshape(-1, 1)
                ny2 = np.roll(ts2.int_data[::ds], shift)
                # Note: mutual_info_score returns nats, convert to bits
                mi = mutual_info_score(ny1, ny2) / np.log(2)
                # Ensure float type for consistency
                mi = float(mi)

        elif ts1.discrete and not ts2.discrete:
            ny1 = ts1.int_data[::ds]
            ny2 = np.roll(ts2.copula_normal_data[::ds], shift)
            # Ensure ny2 is contiguous for better performance with Numba
            if not ny2.flags["C_CONTIGUOUS"]:
                ny2 = np.ascontiguousarray(ny2)
            # Fix: mi_model_gd expects Ym to be the number of discrete states (max + 1)
            Ym = int(np.max(ny1) + 1)
            mi = mi_model_gd(ny2, ny1, Ym=Ym, biascorrect=True, demeaned=True)

        elif not ts1.discrete and ts2.discrete:
            ny1 = ts1.copula_normal_data[::ds]
            ny2 = np.roll(ts2.int_data[::ds], shift)
            # Ensure ny1 is contiguous for better performance with Numba
            if not ny1.flags["C_CONTIGUOUS"]:
                ny1 = np.ascontiguousarray(ny1)
            # Ensure ny2 is contiguous for better performance with Numba
            if not ny2.flags["C_CONTIGUOUS"]:
                ny2 = np.ascontiguousarray(ny2)
            # Fix: mi_model_gd expects Ym to be the number of discrete states (max + 1)
            Ym = int(np.max(ny2) + 1)
            mi = mi_model_gd(ny1, ny2, Ym=Ym, biascorrect=True, demeaned=True)

        if mi < 0:
            mi = 0.0

        return mi


def get_tdmi(data, min_shift=1, max_shift=100, nn=DEFAULT_NN, estimator='gcmi'):
    """Compute time-delayed mutual information (TDMI) for a time series.
    
    Calculates mutual information between a time series and delayed versions
    of itself across a range of time lags. Useful for detecting temporal
    dependencies and optimal embedding delays.
    
    Parameters
    ----------
    data : array-like
        1D time series data.
    min_shift : int, optional
        Minimum time lag to compute. Default: 1.
    max_shift : int, optional
        Maximum time lag to compute (exclusive). Default: 100.
    nn : int, optional
        Number of nearest neighbors for KSG MI estimation. Only used when
        estimator='ksg'. Default: 5.
    estimator : {'gcmi', 'ksg'}, optional
        MI estimator to use. 'gcmi' is faster but provides a lower bound,
        'ksg' is more accurate but slower. Default: 'gcmi'.
        
    Returns
    -------
    list of float
        TDMI values in bits for each time lag from min_shift to max_shift-1.
        
    Notes
    -----
    - The first minimum in TDMI often indicates optimal embedding delay
    - High TDMI at specific lags indicates periodic structure
    - All values are returned in bits for consistency
    - For long time series, 'gcmi' is recommended for speed
    - For precise embedding delay detection, 'ksg' may be more accurate
    
    Examples
    --------
    >>> data = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> tdmi = get_tdmi(data, min_shift=1, max_shift=50)
    >>> optimal_delay = np.argmin(tdmi) + 1  # First minimum
    >>> 
    >>> # Using KSG for more accuracy
    >>> tdmi_ksg = get_tdmi(data, min_shift=1, max_shift=50, estimator='ksg')    """
    ts = TimeSeries(data, discrete=False)
    tdmi = [
        get_1d_mi(ts, ts, shift=shift, k=nn, estimator=estimator) for shift in range(min_shift, max_shift)
    ]

    return tdmi


def get_multi_mi(tslist, ts2, shift=0, ds=1, k=DEFAULT_NN, estimator="gcmi"):
    """Compute mutual information between multiple time series and a single time series.
    
    Parameters
    ----------
    tslist : list of TimeSeries
        List of TimeSeries objects (multivariate X)
    ts2 : TimeSeries
        Single TimeSeries object (Y)
    shift : int, optional
        Number of samples to shift ts2. Default: 0
    ds : int, optional
        Downsampling factor. Default: 1
    k : int, optional
        Number of neighbors for KSG estimator. Default: 5
    estimator : str, optional
        Estimation method. 'gcmi' (fast, lower bound) or 'ksg' (slower, more accurate).
        Default: 'gcmi'
        
        Note on downsampling with GCMI: For performance reasons, when ds > 1, the copula transformation
        is applied to the full data before downsampling. This is an approximation that works well for
        small downsampling factors (ds ≤ 5) and smooth signals, but may introduce inaccuracies for
        large downsampling factors or highly variable signals.
    
    Returns
    -------
    float
        Mutual information I(X;Y) in bits where X is the multivariate input from tslist    """
    
    # Check if all variables are continuous
    all_continuous = all(not ts.discrete for ts in tslist) and not ts2.discrete
    
    if estimator == "gcmi":
        if all_continuous:
            nylist = [ts.copula_normal_data[::ds] for ts in tslist]
            ny1 = np.vstack(nylist)
            ny2 = ts2.copula_normal_data[::ds]
            if shift != 0:
                ny2 = np.roll(ny2, shift)
            mi = mi_gg(ny1, ny2, True, True)
        else:
            raise ValueError("GCMI estimator for multidimensional MI only supports continuous data!")
            
    elif estimator == "ksg":
        if all_continuous:
            # Stack time series data into multivariate array
            x_data = np.column_stack([ts.data[::ds] for ts in tslist])
            y_data = ts2.data[::ds]
            if shift != 0:
                y_data = np.roll(y_data, shift)
            
            # Use existing KSG function which handles multidimensional inputs
            mi = nonparam_mi_cc(x_data, y_data.reshape(-1, 1), k=k, base=2)
        else:
            raise ValueError("KSG estimator for multidimensional MI currently only supports continuous data!")
    else:
        raise ValueError(f"Unknown estimator: {estimator}. Use 'gcmi' or 'ksg'.")

    if mi < 0:
        mi = 0.0

    return mi


def aggregate_multiple_ts(*ts_args, noise=1e-5):
    """Aggregate multiple continuous TimeSeries into a single MultiTimeSeries.

    Adds small noise to break degeneracy and creates a MultiTimeSeries from
    the input TimeSeries objects.

    Parameters
    ----------
    *ts_args : TimeSeries
        Variable number of TimeSeries objects to aggregate.
    noise : float, default=1e-5
        Amount of noise to add to break degeneracy.

    Returns
    -------
    MultiTimeSeries
        Aggregated multi-dimensional time series.

    Raises
    ------
    ValueError
        If any input TimeSeries is discrete.

    Examples
    --------
    >>> ts1 = TimeSeries(np.random.randn(100), discrete=False)
    >>> ts2 = TimeSeries(np.random.randn(100), discrete=False)
    >>> mts = aggregate_multiple_ts(ts1, ts2)    """
    # add small noise to break degeneracy
    mod_tslist = []
    for ts in ts_args:
        if ts.discrete:
            raise ValueError("this is not applicable to discrete TimeSeries")
        mod_ts = TimeSeries(
            ts.data + np.random.random(size=len(ts.data)) * noise, discrete=False
        )
        mod_tslist.append(mod_ts)

    mts = MultiTimeSeries(mod_tslist)  # add last two TS into a single 2-d MTS
    return mts


def conditional_mi(ts1, ts2, ts3, ds=1, k=5):
    """Calculate conditional mutual information I(X;Y|Z).

    Computes the conditional mutual information between ts1 (X) and ts2 (Y)
    given ts3 (Z) for various combinations of continuous and discrete variables.

    Parameters
    ----------
    ts1 : TimeSeries
        First variable (X). Must be continuous.
    ts2 : TimeSeries
        Second variable (Y). Can be continuous or discrete.
    ts3 : TimeSeries
        Conditioning variable (Z). Can be continuous or discrete.
    ds : int, optional
        Downsampling factor. Default: 1.
    k : int, optional
        Number of neighbors for entropy estimation. Default: 5.

    Returns
    -------
    float
        Conditional mutual information I(X;Y|Z) in bits.

    Raises
    ------
    ValueError
        If ts1 is discrete (only continuous X is currently supported).

    Notes
    -----
    Supports four cases:
    - CCC: All continuous - uses Gaussian copula
    - CCD: X,Y continuous, Z discrete - uses Gaussian copula per Z value
    - CDC: X,Z continuous, Y discrete - uses chain rule identity
    - CDD: X continuous, Y,Z discrete - uses entropy decomposition

    For the CDD case, GCMI estimator has limitations due to uncontrollable
    biases (copula transform does not conserve entropy). See
    https://doi.org/10.1002/hbm.23471 for details.
    
    Error Handling
    --------------
    Conditional MI can be negative due to estimation biases, especially with
    finite samples. This function uses adaptive thresholds:
    - Small negatives (< 1% of entropy scale): Silently clipped to 0
    - Moderate negatives (1-10% of scale): Clipped with warning
    - Large negatives (> 10% of scale): Raises ValueError
    
    The CDD case is particularly prone to negative biases due to mixed
    estimators and receives more lenient treatment.    """
    if ts1.discrete:
        raise ValueError(
            "conditional MI(X,Y|Z) is currently implemented for continuous X only"
        )

    # print(ts1.discrete, ts2.discrete, ts3.discrete)
    if not ts2.discrete and not ts3.discrete:
        # CCC: All continuous
        g1 = ts1.copula_normal_data[::ds]
        g2 = ts2.copula_normal_data[::ds]
        g3 = ts3.copula_normal_data[::ds]
        cmi = cmi_ggg(g1, g2, g3, biascorrect=True, demeaned=True)

    elif not ts2.discrete and ts3.discrete:
        # CCD: X,Y continuous, Z discrete
        cmi = gccmi_ccd(
            ts1.data[::ds],
            ts2.data[::ds],
            ts3.int_data[::ds],
        )

    elif ts2.discrete and not ts3.discrete:
        # CDC: X,Z continuous, Y discrete
        # Use entropy-based identity: I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
        # This avoids mixing different MI estimators that cause bias inconsistency

        # H(X|Z) for continuous X,Z using GCMI
        x_data = ts1.data[::ds].reshape(1, -1)
        z_data = ts3.data[::ds].reshape(1, -1)

        # Joint data for H(X,Z) and marginal H(Z)
        xz_joint = np.vstack([x_data, z_data])
        H_xz = ent_g(xz_joint, biascorrect=True)
        H_z = ent_g(z_data, biascorrect=True)
        H_x_given_z = H_xz - H_z

        # H(X|Y,Z) - conditional entropy of X given both Y (discrete) and Z (continuous)
        unique_y_vals = np.unique(ts2.int_data[::ds])
        H_x_given_yz = 0.0

        for y_val in unique_y_vals:
            # Find indices where Y = y_val
            y_mask = ts2.int_data[::ds] == y_val
            n_y = np.sum(y_mask)

            if n_y > 2:  # Need sufficient samples for entropy estimation
                # Extract X,Z values for this Y group
                x_subset = x_data[:, y_mask]
                z_subset = z_data[:, y_mask]

                # Joint entropy H(X,Z|Y=y_val)
                xz_subset = np.vstack([x_subset, z_subset])
                H_xz_given_y = ent_g(xz_subset, biascorrect=True)

                # Marginal entropy H(Z|Y=y_val)
                H_z_given_y = ent_g(z_subset, biascorrect=True)

                # Conditional entropy H(X|Z,Y=y_val) = H(X,Z|Y=y_val) - H(Z|Y=y_val)
                H_x_given_z_y = H_xz_given_y - H_z_given_y

                # Weight by probability P(Y=y_val)
                p_y = n_y / len(ts2.int_data[::ds])
                H_x_given_yz += p_y * H_x_given_z_y

        # Final CMI calculation
        cmi = H_x_given_z - H_x_given_yz

        # Ensure CMI >= 0 due to information theory constraint
        # Small negative values are due to numerical precision and estimation noise
        if cmi < 0:
            # Use relative threshold: 1% of the entropy scale involved
            entropy_scale = max(H_x_given_z, H_x_given_yz, 0.1)  # Minimum scale of 0.1 bits
            threshold = 0.01 * entropy_scale
            
            if abs(cmi) < threshold:
                cmi = 0.0
            else:
                warnings.warn(
                    f"Conditional MI is negative ({cmi:.4f}) beyond expected numerical noise "
                    f"(threshold: {threshold:.4f}). This may indicate insufficient data or "
                    f"numerical issues in entropy estimation. Clipping to 0."
                )
                cmi = 0.0

    else:
        # CDD: X continuous, Y,Z discrete
        # Here we use the identity I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        # Implementation verified: Uses the mathematically correct identity
        # I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        # Note that GCMI estimator is poorly applicable here because of the uncontrollable biases:
        # GCMI correctly estimates the lower bound on MI, but copula transform does not conserve the entropy
        # See  https://doi.org/10.1002/hbm.23471 for further details
        # Therefore, joint entropy estimation relies on ksg estimator instead
        # Note: Original code used copula_normal_data, but our entropy functions expect raw data
        # Using data instead of copula_normal_data for consistency with entropy functions
        H_xz = joint_entropy_cd(ts3.int_data[::ds], ts1.data[::ds], k=k)
        H_yz = joint_entropy_dd(ts2.int_data[::ds], ts3.int_data[::ds])
        H_xyz = joint_entropy_cdd(
            ts2.int_data[::ds], ts3.int_data[::ds], ts1.data[::ds], k=k
        )
        H_z = entropy_d(ts3.int_data[::ds])
        # print('entropies:', H_xz, H_yz, H_xyz, H_z)
        cmi = H_xz + H_yz - H_xyz - H_z

        # Ensure CMI >= 0 due to information theory constraint
        # Small negative values are due to numerical precision and estimation noise
        if cmi < 0:
            # Use relative threshold: 1% of the entropy scale involved
            # For CDD case, entropy scale can be larger due to joint entropy calculations
            entropy_scale = max(H_xz, H_yz, H_xyz, H_z, 0.1)  # Minimum scale of 0.1 bits
            threshold = 0.01 * entropy_scale
            
            # CDD case is known to have larger biases due to mixed estimators
            # (see comment above about GCMI biases). Use a more lenient threshold.
            error_threshold = 0.1 * entropy_scale  # 10% is clearly problematic
            
            if abs(cmi) < threshold:
                cmi = 0.0
            elif abs(cmi) < error_threshold:
                warnings.warn(
                    f"Conditional MI is negative ({cmi:.4f}) beyond expected numerical noise "
                    f"(threshold: {threshold:.4f}). This is common for CDD case due to mixed "
                    f"estimator biases. Clipping to 0."
                )
                cmi = 0.0
            else:
                raise ValueError(
                    f"Conditional MI is significantly negative ({cmi:.4f}, threshold: {error_threshold:.4f}). "
                    f"This indicates serious numerical issues in entropy estimation. "
                    f"Consider using more data or adjusting k parameter."
                )

    return cmi


def interaction_information(ts1, ts2, ts3, ds=1, k=5):
    """Calculate three-way interaction information II(X;Y;Z).

    The interaction information quantifies the amount of information
    that is shared among all three variables. It can be positive (synergy)
    or negative (redundancy).

    Parameters
    ----------
    ts1 : TimeSeries
        First variable (X). Must be continuous.
    ts2 : TimeSeries
        Second variable (Y). Can be continuous or discrete.
    ts3 : TimeSeries
        Third variable (Z). Can be continuous or discrete.
    ds : int, optional
        Downsampling factor. Default: 1.
    k : int, optional
        Number of neighbors for entropy estimation. Default: 5.

    Returns
    -------
    float
        Interaction information II(X;Y;Z) in bits.
        - II < 0: Redundancy (Y and Z provide overlapping information about X)
        - II > 0: Synergy (Y and Z together provide more information than separately)

    Notes
    -----
    The interaction information is computed using Williams & Beer convention:
    II(X;Y;Z) = I(X;Y|Z) - I(X;Y) = I(X;Z|Y) - I(X;Z)

    This implementation assumes X is the target variable (e.g., neural activity)
    and Y, Z are predictor variables (e.g., behavioral features).    """
    # Compute pairwise mutual information
    mi_xy = get_mi(ts1, ts2, ds=ds)
    mi_xz = get_mi(ts1, ts3, ds=ds)

    # Compute conditional mutual information
    cmi_xy_given_z = conditional_mi(ts1, ts2, ts3, ds=ds, k=k)
    cmi_xz_given_y = conditional_mi(ts1, ts3, ts2, ds=ds, k=k)

    # Compute interaction information (should be the same from both formulas)
    # Using Williams & Beer convention: II = I(X;Y|Z) - I(X;Y)
    # This gives negative II for redundancy and positive II for synergy
    ii_1 = cmi_xy_given_z - mi_xy
    ii_2 = cmi_xz_given_y - mi_xz

    # Average for numerical stability
    ii = (ii_1 + ii_2) / 2.0

    return ii


# DOC_VERIFIED for interaction_information
