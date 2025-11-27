import numpy as np
import warnings
import tqdm
import pickle
import logging
from typing import Optional, Union, List

from ..information.info_base import TimeSeries
from ..information.info_base import MultiTimeSeries
from .neuron import (
    DEFAULT_MIN_BEHAVIOUR_TIME,
    DEFAULT_T_OFF,
    DEFAULT_FPS,
    MIN_CA_SHIFT,
    MIN_CA_SHIFT_SEC,
    Neuron,
)
from ..utils.data import get_hash, populate_nested_dict
from ..information.info_base import get_1d_mi
from ..intense.intense_base import get_multicomp_correction_thr

STATS_VARS = [
    "data_hash",
    "opt_delay",
    "pre_pval",
    "pre_rval",
    "pval",
    "rval",
    "me",
    "rel_me_beh",
    "rel_me_ca",
]
SIGNIFICANCE_VARS = [
    "stage1",
    "shuffles1",
    "stage2",
    "shuffles2",
    "final_p_thr",
    "multicomp_corr",
    "pairwise_pval_thr",
]
DEFAULT_STATS = dict(zip(STATS_VARS, [None for _ in STATS_VARS]))
DEFAULT_SIGNIFICANCE = dict(zip(SIGNIFICANCE_VARS, [None for _ in SIGNIFICANCE_VARS]))


def check_dynamic_features(dynamic_features):
    """Validate that all dynamic features have the same length.

    Parameters
    ----------
    dynamic_features : dict
        Dictionary mapping feature names (str) to feature data. Supported data types:
        - TimeSeries: Length determined by len(data) attribute
        - MultiTimeSeries: Length determined by n_points attribute  
        - numpy.ndarray: Length is last dimension (shape[-1]). Must be at least 1D.
        Empty dict is allowed and will return without error.

    Returns
    -------
    None
        Function returns None. Validation is done via exceptions.

    Raises
    ------
    ValueError
        If features have different lengths. Error message includes detailed
        listing of each feature name and its length in timepoints.
    TypeError
        If a feature has an unsupported type (not TimeSeries, MultiTimeSeries, 
        or numpy array), or if a numpy array is 0-dimensional (scalar).
        Error message includes the problematic feature name and its type.

    Notes
    -----
    For numpy arrays, the last dimension is always interpreted as the time
    dimension, consistent with the shape convention (n_features, n_timepoints).    """
    if not dynamic_features:
        return  # Handle empty features gracefully

    dfeat_lengths = {}
    for feat_id, current_ts in dynamic_features.items():
        # Only accept specific types - reject everything else
        if isinstance(current_ts, TimeSeries):
            len_ts = len(current_ts.data)
        elif isinstance(current_ts, MultiTimeSeries):
            # MultiTimeSeries inherits from MVData which has n_points attribute
            len_ts = current_ts.n_points
        elif isinstance(current_ts, np.ndarray):
            # Handle raw numpy arrays - last dimension is time
            if current_ts.ndim == 0:
                raise TypeError(
                    f"Feature '{feat_id}' is a scalar numpy array. "
                    f"Expected TimeSeries, MultiTimeSeries, or numpy array with at least 1 dimension."
                )
            len_ts = current_ts.shape[-1]
        else:
            # Reject all other types
            raise TypeError(
                f"Feature '{feat_id}' has unsupported type: {type(current_ts).__name__}. "
                f"Expected TimeSeries, MultiTimeSeries, or numpy array only."
            )

        dfeat_lengths[feat_id] = len_ts

    # Check all features have same length
    unique_lengths = set(dfeat_lengths.values())
    if len(unique_lengths) != 1:
        # Create informative error message
        length_info = [
            f"  {feat}: {length} timepoints" for feat, length in dfeat_lengths.items()
        ]
        raise ValueError(
            "Dynamic features have different lengths:\n" + "\n".join(length_info)
        )


class Experiment:
    """Base class for calcium imaging and spike train experiments.
    
    This class provides a unified interface for analyzing neural activity data
    (calcium imaging or spike trains) in relation to various experimental features
    (behavioral variables, stimuli, etc.). It handles data organization, feature
    extraction, mutual information analysis, and statistical significance testing.
    
    Parameters
    ----------
    signature : str
        Unique identifier for the experiment.
    calcium : numpy.ndarray
        Calcium imaging data of shape (n_neurons, n_timepoints). Required parameter.
    spikes : numpy.ndarray or None
        Spike train data. Can be provided directly or reconstructed from calcium.
    exp_identificators : dict
        Experiment metadata and identifiers. Each key-value pair becomes an 
        attribute of the object (e.g., exp_identificators={'mouse_id': 'M1'} 
        creates self.mouse_id = 'M1').
    static_features : dict
        Time-invariant features (e.g., cell types, anatomical properties).
        Should include 'fps', 't_rise_sec', 't_off_sec' or defaults will be used.
        The dict is stored as self.static_features, and each key also becomes 
        an individual attribute (e.g., static_features={'fps': 20} creates both
        self.static_features['fps'] and self.fps = 20).
    dynamic_features : dict
        Time-varying features (e.g., behavior, stimuli). Keys are feature names,
        values are TimeSeries, MultiTimeSeries, or numpy arrays. All features
        must have the same length (number of timepoints).
    **kwargs : dict
        Additional parameters including:
        - optimize_kinetics : bool or str, optimize kinetics per neuron (default: False).
          If True, uses 'lbfgs' method. Can specify 'lbfgs' or 'grid' explicitly.
        - reconstruct_spikes : str or bool, spike reconstruction method (default: 'wavelet')
        - bad_frames_mask : array-like, boolean mask where True indicates bad frames
        - spike_kwargs : dict, parameters for spike reconstruction
        - verbose : bool, print progress messages (default: True)
        
    Attributes
    ----------
    signature : str
        Experiment identifier.
    neurons : list
        List of Neuron objects, indexed by cell ID (0-based). Access with
        self.neurons[cell_id].
    n_cells : int
        Number of neurons in the experiment.
    n_frames : int
        Number of time points (frames) in the experiment.
    calcium : MultiTimeSeries
        Calcium imaging data as MultiTimeSeries object.
    spikes : MultiTimeSeries
        Spike train data as MultiTimeSeries object.
    static_features : dict
        Time-invariant experimental features as originally provided.
    dynamic_features : dict
        Time-varying experimental features as TimeSeries/MultiTimeSeries objects.
    stats_tables : dict
        Nested dict storing mutual information statistics. Structure:
        stats_tables[mode][feat_id][cell_id] = stats_dict.
    significance_tables : dict
        Nested dict storing statistical significance data. Structure:
        significance_tables[mode][feat_id][cell_id] = sig_dict.
    embeddings : dict
        Stored dimensionality reduction results by data type and method.
        Structure: embeddings[data_type][method_name] = embedding_data.
    verbose : bool
        Whether to print progress messages.
    spike_reconstruction_method : str or None
        Method used for spike reconstruction if applicable.
    filtered_flag : bool
        Whether bad frames were filtered out.
    selectivity_tables_initialized : bool
        Whether selectivity tables have been initialized.
    exp_identificators : dict
        Original experiment identifiers dictionary.
    _data_hashes : dict
        Private attribute storing hash representations for caching.
    _rdm_cache : dict
        Private attribute caching representational dissimilarity matrices.
    
    Methods
    -------
    check_ds(ds)
        Validate downsampling rate for behavioral analysis.
    get_neuron_feature_pair_stats(cell_id, feat_id, mode='calcium')
        Get selectivity statistics for a neuron-feature pair.
    get_neuron_feature_pair_significance(cell_id, feat_id, mode='calcium')
        Get statistical significance data for a neuron-feature pair.
    update_neuron_feature_pair_stats(stats, cell_id, feat_id, mode='calcium', ...)
        Update statistics for a neuron-feature pair.
    update_neuron_feature_pair_significance(sig, cell_id, feat_id, mode='calcium')
        Update significance data for a neuron-feature pair.
    get_multicell_shuffled_calcium(cbunch=None, method='roll_based', **kwargs)
        Get shuffled calcium data for specified neurons.
    get_multicell_shuffled_spikes(cbunch=None, method='isi_based', **kwargs)
        Get shuffled spike data for specified neurons.
    get_stats_slice(cell_ids, feat_ids, mode='calcium', vars=None)
        Extract statistics for multiple neuron-feature pairs.
    get_significance_slice(cell_ids, feat_ids, mode='calcium', vars=None)
        Extract significance data for multiple neuron-feature pairs.
    get_feature_entropy(feat_id, ds=1)
        Calculate Shannon entropy of a feature.
    get_significant_neurons(min_nspec=1, cbunch=None, fbunch=None, mode='calcium', ...)
        Find neurons with significant selectivity to one or more features.
    store_embedding(embedding, method_name, data_type='calcium', metadata=None)
        Store dimensionality reduction results.
    get_embedding(method_name, data_type='calcium')
        Retrieve stored dimensionality reduction embedding.
    compute_rdm(items, activity_type='calcium', metric='correlation', **kwargs)
        Compute representational dissimilarity matrix.
    clear_rdm_cache()
        Clear the RDM computation cache.
    
    Notes
    -----
    The class supports both calcium imaging and spike train analysis through
    the 'mode' parameter in various methods. Results are cached using hash-based
    lookups to avoid redundant computations. Statistical significance is determined
    using the INTENSE algorithm with two-stage hypothesis testing.
    
    Spike reconstruction is performed automatically if spikes are not provided
    and reconstruct_spikes is not False or None. The 'wavelet' method is recommended
    for calcium imaging data. Set reconstruct_spikes to False or None to disable
    spike reconstruction entirely.
    
    Individual static and dynamic features can be accessed as attributes. For
    example, if 'position' is a dynamic feature, access it via self.position.
    Protected attribute names will have an underscore prefix if conflicts occur.    """

    def __init__(
        self,
        signature,
        calcium,
        spikes,
        exp_identificators,
        static_features,
        dynamic_features,
        **kwargs,
    ):
        """Initialize experiment with neural data and behavioral features.
        
        Creates an experiment object that integrates calcium imaging data, spike trains,
        and behavioral features for neural population analysis. Handles data validation,
        spike reconstruction, and sets up internal data structures for statistical analysis.
        
        Parameters
        ----------
        signature : str
            Unique identifier for the experiment (e.g., 'mouse123_session1').
        calcium : array-like
            Calcium imaging data with shape (n_cells, n_frames). Required.
            Each row is a neuron's calcium trace over time.
        spikes : array-like or None
            Spike train data with same shape as calcium. If None and
            reconstruct_spikes is specified in kwargs, spikes will be
            reconstructed from calcium data.
        exp_identificators : dict or None
            Metadata about the experiment (e.g., subject_id, session_date).
            Keys become attributes of the experiment object.
        static_features : dict or None
            Time-invariant features. Expected keys include:
            - 'fps': sampling rate (frames per second)
            - 't_rise_sec': calcium rise time in seconds
            - 't_off_sec': calcium decay time in seconds
            - Other experiment-wide parameters
        dynamic_features : dict or None
            Time-varying behavioral features (e.g., position, speed).
            Values should be array-like with time dimension matching n_frames.
            Keys become accessible via self.dynamic_features.
        **kwargs
            Additional parameters:
            - optimize_kinetics (bool or str): Optimize kinetics per neuron. Default False.
              If True, uses 'lbfgs' method. Can specify 'lbfgs' or 'grid' explicitly.
            - reconstruct_spikes (str, False, or None): Method for spike reconstruction.
              Options: 'wavelet' (default), 'threshold', False, or None.
              If False or None, spike reconstruction is disabled.
              Only used if spikes is None.
            - bad_frames_mask (array-like): Boolean mask of frames to exclude.
            - spike_kwargs (dict): Parameters for spike reconstruction method.
            - verbose (bool): Print progress messages. Default True.
            
        Raises
        ------
        ValueError
            If calcium is None, if data shapes are inconsistent, if feature
            names conflict with protected attributes, or if data appears
            transposed (n_cells > n_frames).
        TypeError
            If dynamic features have incompatible types.
            
        Warnings
        --------
        UserWarning
            If both spikes and reconstruct_spikes are provided (spikes will
            be overwritten), or if static feature names conflict with
            existing attributes (will be prefixed with underscore).
            
        Notes
        -----
        Protected attribute names that cannot be used as feature names:
        'spikes', 'calcium', 'neurons', 'n_cells', 'n_frames', 'static_features',
        'dynamic_features', 'downsampling', 'significance_tables', 'stats_tables',
        '_data_hashes', 'embeddings', '_rdm_cache', 'intense_results'
        
        The initialization process:
        1. Validates and stores basic data (calcium required)
        2. Handles spike data or reconstruction
        3. Creates Neuron objects for each cell
        4. Processes static and dynamic features
        5. Builds internal data structures for caching
        6. Validates data consistency
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.information.info_base import TimeSeries
        >>> 
        >>> # Basic initialization with calcium data only
        >>> calcium_data = np.random.randn(10, 1000)  # 10 neurons, 1000 timepoints
        >>> exp = Experiment('exp001', calcium_data, None, {}, {}, {}, 
        ...                  reconstruct_spikes=None, verbose=False)
        
        >>> # With spikes and behavioral features
        >>> spike_data = np.random.poisson(0.05, (10, 1000))  # spike trains
        >>> speed_trace = TimeSeries(np.random.rand(1000))    # behavioral data
        >>> exp = Experiment(
        ...     'exp002',
        ...     calcium_data,
        ...     spike_data,
        ...     {},
        ...     {'fps': 30.0},
        ...     {'speed': speed_trace},
        ...     verbose=False
        ... )
        
        >>> # Without spike reconstruction (faster for doctests)
        >>> exp = Experiment(
        ...     'exp003',
        ...     calcium_data,
        ...     None,
        ...     {},
        ...     {'fps': 30.0},
        ...     {},
        ...     reconstruct_spikes=None,
        ...     verbose=False
        ... )
        """
        optimize_kinetics = kwargs.get("optimize_kinetics", False)
        reconstruct_spikes = kwargs.get("reconstruct_spikes", "wavelet")
        bad_frames_mask = kwargs.get("bad_frames_mask", None)
        spike_kwargs = kwargs.get("spike_kwargs", None)
        self.verbose = kwargs.get("verbose", True)

        check_dynamic_features(dynamic_features)
        self.exp_identificators = exp_identificators
        self.signature = signature

        for idx in exp_identificators:
            setattr(self, idx, exp_identificators[idx])

        if calcium is None:
            raise ValueError(
                "Calcium data is required. Please provide a numpy array with shape (n_neurons, n_timepoints)."
            )

        # Handle spike reconstruction based on reconstruct_spikes parameter
        if reconstruct_spikes is None or reconstruct_spikes is False:
            # No spike reconstruction requested
            if spikes is None and self.verbose:
                warnings.warn(
                    "No spike data provided, spikes reconstruction from Ca2+ data disabled"
                )
        else:
            # Spike reconstruction requested
            if spikes is not None:
                warnings.warn(
                    f"Spike data will be overridden by reconstructed spikes from Ca2+ data with method={reconstruct_spikes}"
                )

            # Store the reconstruction method for potential future use
            self.spike_reconstruction_method = reconstruct_spikes

            # Reconstruct spikes
            spikes = self._reconstruct_spikes(
                calcium, reconstruct_spikes, static_features.get("fps"), spike_kwargs
            )

        self.filtered_flag = False
        if bad_frames_mask is not None:
            calcium, spikes, dynamic_features = self._trim_data(
                calcium, spikes, dynamic_features, bad_frames_mask
            )
        else:
            for feat_id in dynamic_features.copy():
                feat_data = dynamic_features[feat_id]
                # Skip if already a TimeSeries or MultiTimeSeries
                if not isinstance(feat_data, (TimeSeries, MultiTimeSeries)):
                    # Convert numpy arrays based on dimensionality
                    if isinstance(feat_data, np.ndarray):
                        if feat_data.ndim == 1:
                            # 1D array -> TimeSeries
                            dynamic_features[feat_id] = TimeSeries(feat_data)
                        elif feat_data.ndim == 2:
                            # 2D array -> MultiTimeSeries (each row is a component)
                            ts_list = [
                                TimeSeries(feat_data[i, :], discrete=False)
                                for i in range(feat_data.shape[0])
                            ]
                            dynamic_features[feat_id] = MultiTimeSeries(ts_list)
                        else:
                            raise ValueError(
                                f"Feature {feat_id} has unsupported dimensionality: {feat_data.ndim}D"
                            )
                    else:
                        # Assume it's 1D data if not numpy array
                        dynamic_features[feat_id] = TimeSeries(feat_data)

        self.n_cells = calcium.shape[0]
        self.n_frames = calcium.shape[1]

        self.neurons = []

        if self.verbose:
            print("Building neurons...")
        for i in tqdm.tqdm(np.arange(self.n_cells), position=0, leave=True, disable=not self.verbose):
            cell = Neuron(
                str(i),
                calcium[i, :],
                spikes[i, :] if spikes is not None else None,
                default_t_rise=static_features.get("t_rise_sec"),
                default_t_off=static_features.get("t_off_sec"),
                fps=static_features.get("fps"),
                optimize_kinetics=optimize_kinetics,
            )

            self.neurons.append(cell)

        # Now create MultiTimeSeries from neurons to preserve their shuffle masks
        calcium_ts_list = [neuron.ca for neuron in self.neurons]
        spikes_ts_list = [
            (
                neuron.sp
                if neuron.sp is not None
                else TimeSeries(np.zeros(self.n_frames), discrete=True)
            )
            for neuron in self.neurons
        ]

        # Create MultiTimeSeries from the TimeSeries objects in neurons
        # This preserves the individual shuffle masks created by each Neuron
        self.calcium = MultiTimeSeries(calcium_ts_list)
        # Allow zero columns for spikes since many neurons might not spike
        self.spikes = MultiTimeSeries(spikes_ts_list, allow_zero_columns=True)

        self.dynamic_features = dynamic_features

        # Protected attributes that should not be overwritten
        protected_attrs = {
            "neurons",
            "calcium",
            "spikes",
            "n_cells",
            "n_frames",
            "stats_tables",
            "significance_tables",
            "embeddings",
            "_rdm_cache",
            "_data_hashes",
            "signature",
            "exp_identificators",
            "static_features",
            "dynamic_features",
        }

        # Check for protected attributes in dynamic features and remove them
        conflicting_features = []
        for feat_id in dynamic_features:
            if isinstance(feat_id, str) and feat_id in protected_attrs:
                conflicting_features.append(feat_id)

        if conflicting_features:
            raise ValueError(
                f"Dynamic feature names conflict with protected attributes: {conflicting_features}. "
                f"Protected attributes are: {sorted(protected_attrs)}"
            )

        # Store static_features as an attribute for consistency
        self.static_features = static_features

        # Set dynamic features as attributes
        for feat_id in dynamic_features:
            if isinstance(feat_id, str):
                if hasattr(self, feat_id):
                    warnings.warn(
                        f"Feature name '{feat_id}' overwrites existing attribute."
                    )
                setattr(self, feat_id, dynamic_features[feat_id])
            # Skip tuples (multifeatures) as they can't be attribute names

        # Also set static features as individual attributes for backward compatibility
        for sfeat_name in static_features:
            if sfeat_name in protected_attrs:
                warnings.warn(
                    f"Static feature name '{sfeat_name}' conflicts with protected attribute. "
                    f"Access via attribute name with underscore: _{sfeat_name}"
                )
                setattr(self, f"_{sfeat_name}", static_features[sfeat_name])
            else:
                setattr(self, sfeat_name, static_features[sfeat_name])

        # for selectivity data from INTENSE
        self.stats_tables = {}
        self.significance_tables = {}
        self.selectivity_tables_initialized = False

        # for dimensionality reduction embeddings
        self.embeddings = {"calcium": {}, "spikes": {}}

        # Cache for RDM computations
        self._rdm_cache = {}

        if self.verbose:
            print("Building data hashes...")
        self._build_data_hashes(mode="calcium")
        # Only build spike hashes if we have actual spike data (not just zeros)
        # Check if any neuron has non-None spikes
        has_spikes = any(neuron.sp is not None for neuron in self.neurons)
        if has_spikes:
            self._build_data_hashes(mode="spikes")

        if self.verbose:
            print("Final checkpoint...")
        self._checkpoint()
        # self._load_precomputed_data(**kwargs)

        if self.verbose:
            print(
                f'Experiment "{self.signature}" constructed successfully with {self.n_cells} neurons and {len(self.dynamic_features)} features'
            )

    def check_ds(self, ds):
        """Check if downsampling rate is appropriate for behavior analysis.
        
        Validates that the downsampling rate won't cause time gaps larger than
        the minimum behavior time interval (0.25 seconds), which could lead to 
        missed behavioral events.
        
        Parameters
        ----------
        ds : int
            Downsampling factor. The data will be sampled every ds frames.
            Must be a positive integer (>= 1).
            
        Returns
        -------
        None
            
        Raises
        ------
        ValueError
            If fps (frames per second) is not set for this experiment, or if
            ds is less than 1. Error messages include relevant context.
        TypeError
            If ds is not an integer.
            
        Warnings
        --------
        UserWarning
            Issued if the time gap created by downsampling exceeds
            DEFAULT_MIN_BEHAVIOUR_TIME (0.25 seconds). The warning includes
            the current threshold, downsampling factor, and resulting time gap.
        
        Notes
        -----
        The time gap is calculated as (1/fps) * ds seconds. For example,
        with fps=20 and ds=10, the time gap would be 0.5 seconds, which
        exceeds the 0.25 second threshold and triggers a warning.        """
        # Validate ds parameter
        if not isinstance(ds, (int, np.integer)):
            raise TypeError(f"Downsampling factor ds must be an integer, got {type(ds).__name__}")
        if ds < 1:
            raise ValueError(f"Downsampling factor ds must be >= 1, got {ds}")
            
        if not hasattr(self, "fps"):
            raise ValueError(f"fps not set for {self.signature}")

        time_step = 1.0 / self.fps
        if time_step * ds > DEFAULT_MIN_BEHAVIOUR_TIME:
            warnings.warn(
                f"Downsampling constant is too high: some behaviour acts may be skipped. "
                f"Current minimal behaviour time interval is set to {DEFAULT_MIN_BEHAVIOUR_TIME} sec, "
                f"downsampling {ds} will create time gaps of {time_step*ds:.3f} sec",
                UserWarning
            )

    def _set_selectivity_tables(self, mode, fbunch=None, cbunch=None):
        """Create or reset selectivity statistics tables for the specified mode.
        
        Creates nested dictionaries for storing mutual information statistics.
        Overwrites any existing tables for the given mode.
        
        Parameters
        ----------
        mode : str
            Table identifier (typically 'calcium' or 'spikes'). Not validated.
        fbunch : None, str, or iterable of str, optional
            Feature(s) to include. If None, includes all dynamic features.
        cbunch : None, int, or iterable of int, optional
            Cell ID(s) to include. If None, includes all cells.
            
        Notes
        -----
        Creates two nested dictionaries {feature: {cell: dict}}:
        - self.stats_tables[mode]: MI statistics (initialized from DEFAULT_STATS)
        - self.significance_tables[mode]: Significance data (from DEFAULT_SIGNIFICANCE)
        
        Sets self.selectivity_tables_initialized to True.
        
        Warning: Overwrites existing tables without preserving data.        """
        # neuron-feature pair statistics
        stats_table = self._populate_cell_feat_dict(
            DEFAULT_STATS, fbunch=fbunch, cbunch=cbunch
        )

        # neuron-feature pair significance-related data
        significance_table = self._populate_cell_feat_dict(
            DEFAULT_SIGNIFICANCE, fbunch=fbunch, cbunch=cbunch
        )
        self.stats_tables[mode] = stats_table
        self.significance_tables[mode] = significance_table
        self.selectivity_tables_initialized = True

    def _build_pair_hash(self, cell_id, feat_id, mode="calcium"):
        """Build a unique hash representation of neuron-feature pair data.
        
        Creates a hash tuple that uniquely identifies the combination of neural
        activity data and feature data for caching computations.
        
        Parameters
        ----------
        cell_id : int
            Neuron index. Must exist in self.neurons list.
        feat_id : str or iterable of str
            Feature name(s). Single-element iterables are converted to strings.
            All features must exist in self.dynamic_features.
        mode : {'calcium', 'spikes'}, optional
            Type of neural activity data. Default is 'calcium'.
            
        Returns
        -------
        tuple
            Hash tuple containing:
            - Single feature: (activity_hash, feature_hash)
            - Multiple features: (activity_hash, feature1_hash, feature2_hash, ...)
            - Empty iterable: (activity_hash,)
            
        Raises
        ------
        ValueError
            If mode is not 'calcium' or 'spikes'.
        KeyError
            If cell_id not in self.neurons or feat_id not in self.dynamic_features.
        AttributeError
            If neuron lacks .ca/.sp attributes or feature lacks .data attribute.
            
        Notes
        -----
        Uses SHA256 hashing on raw numpy arrays. Multiple features are sorted
        alphabetically before hashing to ensure order-independent results.        """
        if mode == "calcium":
            act = self.neurons[cell_id].ca.data
        elif mode == "spikes":
            act = self.neurons[cell_id].sp.data
        else:
            raise ValueError('"mode" can be either "calcium" or "spikes"')

        act_hash = get_hash(act)

        if (not isinstance(feat_id, str)) and len(feat_id) == 1:
            feat_id = feat_id[0]

        if isinstance(feat_id, str):
            dyn_data = self.dynamic_features[feat_id].data
            dyn_data_hash = get_hash(dyn_data)
            pair_hash = (act_hash, dyn_data_hash)

        else:
            ordered_fnames = tuple(sorted(list(feat_id)))
            list_of_hashes = [act_hash]
            for fname in ordered_fnames:
                dyn_data = self.dynamic_features[fname].data
                dyn_data_hash = get_hash(dyn_data)
                list_of_hashes.append(dyn_data_hash)

            pair_hash = tuple(list_of_hashes)

        return pair_hash

    def _build_data_hashes(self, mode="calcium"):
        """Build hash representations for all neuron-feature pairs.
        
        Pre-computes SHA256 hashes for all combinations of neurons and features
        to enable efficient caching of mutual information calculations.
        
        Parameters
        ----------
        mode : {'calcium', 'spikes'}, optional
            Type of neural activity data. Default is 'calcium'. Not validated.
            
        Notes
        -----
        Creates nested dictionary structure:
        self._data_hashes[mode][feat_id][cell_id] = hash_tuple
        
        Where hash_tuple is from _build_pair_hash(cell_id, feat_id, mode).
        
        Warning: Calling this method multiple times for the same mode will
        completely recreate all hashes, overwriting existing data.        """
        # Create default hashes structure for this mode only
        if not hasattr(self, "_data_hashes"):
            self._data_hashes = {}

        # Create independent dictionary for each mode to avoid aliasing
        mode_hashes = {
            dfeat: dict(zip(range(self.n_cells), [None for _ in range(self.n_cells)]))
            for dfeat in self.dynamic_features.keys()
        }
        self._data_hashes[mode] = mode_hashes

        # Populate hashes for this mode
        for feat_id in self.dynamic_features:
            for cell_id in range(self.n_cells):
                self._data_hashes[mode][feat_id][cell_id] = self._build_pair_hash(
                    cell_id, feat_id, mode=mode
                )

    def _trim_data(
        self, calcium, spikes, dynamic_features, bad_frames_mask, force_filter=False
    ):
        """Filter out bad frames from all data arrays.
        
        Removes frames marked as bad in bad_frames_mask from calcium, spikes,
        and all dynamic features, maintaining temporal alignment.
        
        Parameters
        ----------
        calcium : numpy.ndarray
            Calcium data, shape (n_neurons, n_frames).
        spikes : numpy.ndarray or None
            Spike data, same shape as calcium. Can be None.
        dynamic_features : dict
            Time-varying features as arrays, TimeSeries, or MultiTimeSeries.
        bad_frames_mask : array-like of bool
            Boolean mask where True indicates bad frames to remove.
        force_filter : bool, optional
            Force re-filtering even if already filtered. Default False.
            
        Returns
        -------
        tuple
            (filtered_calcium, filtered_spikes, filtered_dynamic_features)
            
        Raises
        ------
        AttributeError
            If data already filtered and force_filter is False.
            
        Side Effects
        ------------
        Sets self.filtered_flag = True and self.bad_frames_mask.
        
        Notes
        -----
        For multi-dimensional arrays, assumes time is the second dimension.
        For 1D arrays or unknown types, assumes time is the last dimension.        """

        if not force_filter and self.filtered_flag:
            raise AttributeError(
                'Data is already filtered, if you want to force filtering it again, set "force_filter = True"'
            )

        f_calcium = calcium[:, ~bad_frames_mask]
        if spikes is not None:
            f_spikes = spikes[:, ~bad_frames_mask]
        else:
            f_spikes = None

        f_dynamic_features = {}
        for feat_id in dynamic_features:
            current_ts = dynamic_features[feat_id]
            if isinstance(current_ts, TimeSeries):
                f_ts = TimeSeries(
                    current_ts.data[~bad_frames_mask], discrete=current_ts.discrete
                )
            elif isinstance(current_ts, MultiTimeSeries):
                # Handle MultiTimeSeries by trimming each component
                filtered_components = []
                for i in range(current_ts.n_dim):
                    component_data = current_ts.data[i, ~bad_frames_mask]
                    filtered_components.append(
                        TimeSeries(component_data, discrete=current_ts.discrete)
                    )
                f_ts = MultiTimeSeries(filtered_components)
            elif isinstance(current_ts, np.ndarray):
                # Handle raw arrays
                if current_ts.ndim == 1:
                    f_ts = TimeSeries(current_ts[~bad_frames_mask])
                else:
                    # Multi-dimensional array
                    f_ts = current_ts[:, ~bad_frames_mask]
            else:
                # Fallback for other types
                f_ts = TimeSeries(current_ts[~bad_frames_mask])

            f_dynamic_features[feat_id] = f_ts

        self.filtered_flag = True
        self.bad_frames_mask = bad_frames_mask

        return f_calcium, f_spikes, f_dynamic_features

    def _checkpoint(self):
        """Validate experiment data integrity and consistency.
        
        Performs comprehensive checks to ensure the experiment data is properly
        formatted and meets minimum requirements for analysis.
        
        Raises
        ------
        ValueError
            If any of the following conditions are met:
            - Signal is too short for shuffle mask creation
            - Number of cells exceeds number of time frames (likely transposed)
            - Feature shapes are inconsistent with experiment duration
            
        Notes
        -----
        Checks include:
        - Minimum signal length based on decay time and shuffle requirements
        - Proper data orientation (neurons × timepoints)
        - Consistency of all feature dimensions with n_frames        """
        # Check minimal length for proper shuffle mask creation
        t_off_sec = getattr(self, "t_off_sec", DEFAULT_T_OFF)
        fps = getattr(self, "fps", DEFAULT_FPS)
        t_off_frames = t_off_sec * fps
        # FPS-adaptive: MIN_CA_SHIFT_SEC seconds worth of frames
        min_ca_shift_frames = int(MIN_CA_SHIFT_SEC * fps)
        min_required_frames = (
            int(t_off_frames * min_ca_shift_frames * 2) + 10
        )  # Need space for both ends + some valid positions

        if self.n_frames < min_required_frames:
            raise ValueError(
                f"Signal too short: {self.n_frames} frames. "
                f"Minimum {min_required_frames} frames required for proper shuffle mask creation "
                f"(based on t_off={t_off_sec}s, fps={fps}Hz)."
            )

        if self.n_cells > self.n_frames:
            raise ValueError(
                f"Number of cells ({self.n_cells}) > number of time frames ({self.n_frames}). "
                f"Data appears to be transposed. Expected shape: (n_neurons, n_timepoints)."
            )

        for dfeat in ["calcium", "spikes"]:
            if self.n_frames not in getattr(self, dfeat).shape:
                raise ValueError(
                    f'"{dfeat}" feature has inappropriate shape: {getattr(self, dfeat).data.shape}'
                    f"inconsistent with data length {self.n_frames}"
                )

        for dfeat in self.dynamic_features.keys():
            if isinstance(dfeat, str):
                if self.n_frames not in getattr(self, dfeat).data.shape:
                    raise ValueError(
                        f'"{dfeat}" feature has inappropriate shape: {getattr(self, dfeat).data.shape}'
                        f"inconsistent with data length {self.n_frames}"
                    )
            else:
                # For tuple features (multifeatures), check the underlying data
                feat_data = self.dynamic_features[dfeat]
                if (
                    hasattr(feat_data, "data")
                    and self.n_frames not in feat_data.data.shape
                ):
                    raise ValueError(
                        f'"{dfeat}" feature has inappropriate shape: {feat_data.data.shape}'
                        f"inconsistent with data length {self.n_frames}"
                    )

    def _populate_cell_feat_dict(self, content, fbunch=None, cbunch=None):
        """Create nested dictionary structure for cell-feature pairs.
        
        Builds a two-level dictionary where the outer level contains features
        and the inner level contains cells, with each entry initialized to
        the specified content value.
        
        Parameters
        ----------
        content : any
            Default value to populate in each cell-feature entry.
        fbunch : None, str, or iterable of str, optional
            Feature(s) to include. If None, includes all features.
        cbunch : None, int, or iterable of int, optional
            Cell ID(s) to include. If None, includes all cells.
            
        Returns
        -------
        dict
            Nested dictionary with structure: {feature_id: {cell_id: content}}        """
        cell_ids = self._process_cbunch(cbunch)
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True)
        nested_dict = populate_nested_dict(content, feat_ids, cell_ids)
        return nested_dict

    def _process_cbunch(self, cbunch):
        """Convert cell specification to list of cell IDs.
        
        Parameters
        ----------
        cbunch : None, int, or iterable of int
            Cell specification:
            - None: returns all cell IDs
            - int: returns single cell ID in a list
            - iterable: returns list of specified cell IDs
            
        Returns
        -------
        list of int
            List of cell IDs to process.        """
        if isinstance(cbunch, int):
            cell_ids = [cbunch]
        elif cbunch is None:
            cell_ids = list(np.arange(self.n_cells))
        else:
            cell_ids = list(cbunch)

        return cell_ids

    def _process_fbunch(self, fbunch, allow_multifeatures=False, mode="calcium"):
        """Convert feature specification to list of feature IDs.
        
        Parameters
        ----------
        fbunch : None, str, iterable of str, or iterable of tuples
            Feature specification:
            - None: returns all feature IDs
            - str: returns single feature ID in a list
            - iterable of str: returns list of feature IDs
            - iterable of tuples: multi-feature combinations (if allowed)
        allow_multifeatures : bool, optional
            Whether to allow multi-feature tuples. Default is False.
        mode : {'calcium', 'spikes'}, optional
            Activity mode for filtering relevant features. Default is 'calcium'.
            
        Returns
        -------
        list
            List of feature IDs or feature ID tuples to process.
            
        Raises
        ------
        ValueError
            If multi-features are provided but not allowed.        """
        if isinstance(fbunch, str):
            feat_ids = [fbunch]

        elif fbunch is None:  # default set of features
            if allow_multifeatures:
                try:
                    # stats table contains up-to-date set of features, including multifeatures
                    feat_ids = list(self.stats_tables[mode].keys())
                except KeyError:
                    # if stats is not available, take pre-defined full set of features
                    feat_ids = list(self.dynamic_features.keys())
            else:
                feat_ids = list(self.dynamic_features.keys())

        else:
            feat_ids = []

            # check for multifeatures
            for fname in fbunch:
                if isinstance(fname, str):
                    if fname in self.dynamic_features:
                        feat_ids.append(fname)
                else:
                    if allow_multifeatures:
                        feat_ids.append(tuple(sorted(list(fname))))
                    else:
                        raise ValueError(
                            'Multifeature detected in "allow_multifeatures=False" mode'
                        )

        return feat_ids

    def _process_sbunch(self, sbunch, significance_mode=False):
        """Process statistics bunch specification into filtered list.
        
        Converts input formats for specifying statistics types into a 
        standardized list, filtering out any invalid entries.
        
        Parameters
        ----------
        sbunch : None, str, or iterable of str
            Statistics specification. If None, returns all valid stats.
            Invalid entries in iterables are silently filtered out.
        significance_mode : bool, optional
            If True, uses SIGNIFICANCE_VARS, else uses STATS_VARS. Default False.
            
        Returns
        -------
        list of str
            Valid statistics variable names only.
            
        Notes
        -----
        Single strings are returned as-is without validation.
        Iterables have invalid entries filtered out silently.        """
        if significance_mode:
            default_list = SIGNIFICANCE_VARS
        else:
            default_list = STATS_VARS

        if isinstance(sbunch, str):
            return [sbunch]

        elif sbunch is None:
            return default_list

        else:
            return [st for st in sbunch if st in default_list]

    def _add_single_feature_to_data_hashes(self, feat_id, mode="calcium"):
        """Add hash mapping for a single feature.
        
        Creates hash representations for all neuron-feature pairs for the
        specified feature and adds them to the data hashes structure.
        
        Parameters
        ----------
        feat_id : str
            Feature identifier. Must exist in self.dynamic_features.
        mode : {'calcium', 'spikes'}, optional
            Neural activity type. Default 'calcium'.
            
        Side Effects
        ------------
        Modifies self._data_hashes[mode][feat_id] in place.
        
        Notes
        -----
        Only adds if feature not already in data hashes.
        No validation performed on feat_id existence.        """
        if feat_id not in self._data_hashes[mode]:
            self._data_hashes[mode][feat_id] = {}
            # Add hashes for all cells
            for cell_id in range(self.n_cells):
                pair_hash = self._build_pair_hash(cell_id, feat_id, mode=mode)
                self._data_hashes[mode][feat_id][cell_id] = pair_hash

    def _add_single_feature_to_stats(self, feat_id, mode="calcium"):
        """Add empty stats and significance tables for a single feature.
        
        Initializes the statistics and significance tracking structures for
        all neurons for the specified feature.
        
        Parameters
        ----------
        feat_id : str
            Feature identifier to add tables for.
        mode : {'calcium', 'spikes'}, optional
            Neural activity type. Default 'calcium'.
            
        Side Effects
        ------------
        Modifies self.stats_tables[mode][feat_id] and
        self.significance_tables[mode][feat_id] in place.
        
        Notes
        -----
        Only adds if feature not already in stats tables.
        Creates deep copies of DEFAULT_STATS and DEFAULT_SIGNIFICANCE
        for each cell to prevent aliasing.        """
        if feat_id not in self.stats_tables[mode]:
            # Initialize stats for all cells
            self.stats_tables[mode][feat_id] = {}
            self.significance_tables[mode][feat_id] = {}
            
            for cell_id in range(self.n_cells):
                self.stats_tables[mode][feat_id][cell_id] = DEFAULT_STATS.copy()
                self.significance_tables[mode][feat_id][cell_id] = DEFAULT_SIGNIFICANCE.copy()

    def _add_multifeature_to_data_hashes(self, feat_id, mode="calcium"):
        """Add hash mapping for a multi-feature combination.
        
        Creates hash representations for the specified multi-feature combination
        across all neurons for joint mutual information calculations.
        
        .. deprecated:: 
            The multifeature mechanism using tuples is marked for deprecation.
            Future versions will use a different approach for joint MI.
        
        Parameters
        ----------
        feat_id : list or tuple of str
            Multiple feature names (at least 2). Must not be a string or
            single-element collection.
        mode : {'calcium', 'spikes'}, optional
            Neural activity type. Default 'calcium'.
            
        Side Effects
        ------------
        Modifies self._data_hashes[mode] by adding sorted tuple key.
        
        Raises
        ------
        ValueError
            If feat_id is a string or single-element collection.
            
        Notes
        -----
        Multi-features are stored as sorted tuples. Existing entries ignored.
        The mode parameter is now properly passed to _build_pair_hash.        """
        if isinstance(feat_id, str):
            raise ValueError("This method is for multifeature update only. Use _add_single_feature_to_data_hashes for single features.")
        
        if len(feat_id) == 1:
            raise ValueError(
                f"Single-element list {feat_id} provided. "
                "Use _add_single_feature_to_data_hashes for single features or provide multiple features."
            )

        ordered_fnames = tuple(sorted(list(feat_id)))
        if ordered_fnames not in self._data_hashes[mode]:
            all_hashes = [
                self._build_pair_hash(cell_id, ordered_fnames, mode=mode)
                for cell_id in range(self.n_cells)
            ]
            new_dict = {ordered_fnames: dict(zip(range(self.n_cells), all_hashes))}
            self._data_hashes[mode].update(new_dict)

    def _add_multifeature_to_stats(self, feat_id, mode="calcium"):
        """Add empty stats and significance tables for a multi-feature combination.
        
        Initializes the statistics and significance tracking structures for
        all neurons for the specified multi-feature combination.
        
        .. deprecated:: 
            The multifeature mechanism using tuples is marked for deprecation.
            Future versions will use a different approach for joint MI.
        
        Parameters
        ----------
        feat_id : list or tuple of str
            Multiple feature names (at least 2). Must not be a string or
            single-element collection.
        mode : {'calcium', 'spikes'}, optional
            Neural activity type. Default 'calcium'.
            
        Side Effects
        ------------
        - Modifies self.stats_tables[mode] by adding sorted tuple key
        - Modifies self.significance_tables[mode] by adding sorted tuple key
        - Prints to stdout if self.verbose=True and feature is new
        
        Raises
        ------
        ValueError
            If feat_id is a string or single-element collection.
            
        Notes
        -----
        Multi-features are normalized to sorted tuples. Only prints verbose 
        message for new features using the sorted tuple representation.        """
        if isinstance(feat_id, str):
            raise ValueError("This method is for multifeature update only. Use _add_single_feature_to_stats for single features.")
        
        if len(feat_id) == 1:
            raise ValueError(
                f"Single-element list {feat_id} provided. "
                "Use _add_single_feature_to_stats for single features or provide multiple features."
            )

        ordered_fnames = tuple(sorted(list(feat_id)))
        if ordered_fnames not in self.stats_tables[mode]:
            if self.verbose:
                print(f"Multifeature {ordered_fnames} is new, it will be added to stats table")
            self.stats_tables[mode][ordered_fnames] = {
                cell_id: DEFAULT_STATS.copy() for cell_id in range(self.n_cells)
            }

            self.significance_tables[mode][ordered_fnames] = {
                cell_id: DEFAULT_SIGNIFICANCE.copy()
                for cell_id in range(self.n_cells)
            }

    def _check_stats_relevance(self, cell_id, feat_id, mode="calcium"):
        """Check if stats exist and are current, adding new features if needed.
        
        Verifies if statistics for a neuron-feature pair exist and match current
        data hashes. Can add new features to tables with deprecation warnings.
        
        .. note::
            This method has side effects - it can add features to stats_tables,
            significance_tables, and _data_hashes. This behavior will be removed
            after the tuple multifeature mechanism is fully deprecated.
        
        Parameters
        ----------
        cell_id : int
            Neuron index.
        feat_id : str or tuple of str
            Single feature or tuple of features for joint MI.
        mode : {'calcium', 'spikes'}, optional
            Neural activity type. Default 'calcium'.
            
        Returns
        -------
        bool
            True if stats exist/were added and hashes match, False if data changed.
            
        Side Effects
        ------------
        May add features to stats_tables, significance_tables, and _data_hashes.
        Issues deprecation warnings for dynamic feature additions.
        
        Raises
        ------
        ValueError
            If single feature not in self.dynamic_features.        """

        if not isinstance(feat_id, str):
            feat_id = tuple(sorted(list(feat_id)))

        if feat_id not in self.stats_tables[mode]:
            # DEPRECATED: Dynamic stats table updates are discouraged due to statistical validity concerns
            # Multiple comparison correction depends on the total number of hypotheses tested
            import warnings
            
            if isinstance(feat_id, tuple):
                warnings.warn(
                    "DEPRECATED: Adding tuple features to stats table after initialization is discouraged. "
                    "This can invalidate multiple comparison corrections. "
                    "Consider using use_precomputed_stats=False for proper statistical analysis.",
                    DeprecationWarning,
                    stacklevel=2
                )
                # Still allow it for backward compatibility
                self._add_multifeature_to_stats(feat_id, mode=mode)
                if feat_id not in self._data_hashes[mode]:
                    self._add_multifeature_to_data_hashes(feat_id, mode=mode)
                return True
            else:
                # Single feature case - this shouldn't happen if experiment is properly initialized
                if feat_id not in self.dynamic_features:
                    raise ValueError(
                        f"Feature {feat_id} is not present in dynamic_features. "
                        "Check the feature name."
                    )
                warnings.warn(
                    f"Feature {feat_id} was not included in initial stats computation. "
                    "This suggests the experiment was not properly initialized with all features. "
                    "Adding it now may invalidate multiple comparison corrections.",
                    UserWarning,
                    stacklevel=2
                )
                # Add new feature to stats table
                self._add_single_feature_to_stats(feat_id, mode=mode)
                # Also ensure it's in data hashes
                if feat_id not in self._data_hashes[mode]:
                    self._add_single_feature_to_data_hashes(feat_id, mode=mode)
                return True

        pair_hash = self._data_hashes[mode][feat_id][cell_id]
        existing_hash = self.stats_tables[mode][feat_id][cell_id]["data_hash"]

        # if (stats does not exist yet) or (stats exists and data is the same):
        if (existing_hash is None) or (pair_hash == existing_hash):
            return True

        else:
            if self.verbose:
                print(
                    f"Looks like the data for the pair (cell {cell_id}, feature {feat_id}) "
                    "has been changed since the last calculation)"
                )

            return False

    def _update_stats_and_significance(
        self, stats, mode, cell_id, feat_id, stage2_only
    ):
        """
        Updates stats table and linked significance table to erase irrelevant data properly.
        
        Parameters
        ----------
        stats : dict
            Statistics dictionary to update the table with
        mode : str
            Mode of data processing (e.g., 'calcium')
        cell_id : int
            Cell identifier
        feat_id : str or tuple
            Feature identifier or tuple of feature identifiers
        stage2_only : bool
            If True, only update stage 2 significance data
        """
        # update statistics
        self.stats_tables[mode][feat_id][cell_id].update(stats)
        if not stage2_only:
            # erase significance data completely since stats for stage 1 has been modified
            self.significance_tables[mode][feat_id][cell_id].update(
                DEFAULT_SIGNIFICANCE.copy()
            )
        else:
            # erase significance data for stage 2 since stats for stage 2 has been modified
            self.significance_tables[mode][feat_id][cell_id].update(
                {"stage2": None, "shuffles2": None}
            )

    def update_neuron_feature_pair_stats(
        self,
        stats,
        cell_id,
        feat_id,
        mode="calcium",
        force_update=False,
        stage2_only=False,
    ):
        """
        Updates calcium-feature pair statistics.
        
        feat_id should be a string or an iterable of strings (in case of joint MI calculation).
        This function allows multifeatures.
        
        Parameters
        ----------
        stats : dict
            Statistics dictionary containing the updates
        cell_id : int
            Cell identifier
        feat_id : str or iterable of str
            Feature identifier(s). Can be a string or iterable of strings for joint MI calculation
        mode : str, optional
            Data processing mode. Default is "calcium"
        force_update : bool, optional
            If True, force update even if data hashes match. Default is False
        stage2_only : bool, optional
            If True, only update stage 2 statistics. Default is False
        """

        if not isinstance(feat_id, str):
            self._add_multifeature_to_data_hashes(feat_id, mode=mode)
            self._add_multifeature_to_stats(feat_id, mode=mode)

        if self._check_stats_relevance(cell_id, feat_id, mode=mode):
            self._update_stats_and_significance(
                stats, mode, cell_id, feat_id, stage2_only=stage2_only
            )

        else:
            if not force_update:
                if self.verbose:
                    print('To forcefully update the stats, set "force_update=True"')
            else:
                self._update_stats_and_significance(
                    stats, mode, cell_id, feat_id, stage2_only=stage2_only
                )

    def update_neuron_feature_pair_significance(
        self, sig, cell_id, feat_id, mode="calcium"
    ):
        """
        Updates calcium-feature pair significance data.
        
        feat_id should be a string or an iterable of strings (in case of joint MI calculation).
        This function allows multifeatures.
        
        Parameters
        ----------
        sig : dict
            Significance data to update
        cell_id : int
            Cell identifier
        feat_id : str or iterable of str
            Feature identifier(s). Can be a string or iterable of strings for joint MI calculation
        mode : str, optional
            Data processing mode. Default is "calcium"
        """
        if not isinstance(feat_id, str):
            self._add_multifeature_to_data_hashes(feat_id, mode=mode)
            self._add_multifeature_to_stats(feat_id, mode=mode)

        if self._check_stats_relevance(cell_id, feat_id, mode=mode):
            self.significance_tables[mode][feat_id][cell_id].update(sig)

        else:
            raise ValueError(
                "Can not update significance table until the collision between actual data hashes and "
                "saved stats data hashes is resolved. Use update_neuron_feature_pair_stats"
                'with "force_update=True" to forcefully rewrite statistics'
            )

    def get_neuron_feature_pair_stats(self, cell_id, feat_id, mode="calcium"):
        """Get selectivity statistics for a neuron-feature pair.
        
        Retrieves pre-computed statistics measuring the relationship between
        neural activity and behavioral/experimental features. Supports both 
        single features and multi-feature analysis.
        
        Parameters
        ----------
        cell_id : int
            Neuron/cell identifier.
        feat_id : str or tuple of str
            Feature identifier(s). Can be a single feature name or tuple
            of feature names for joint analysis.
        mode : {'calcium', 'spikes'}, optional
            Type of neural activity. Default is 'calcium'.
            
        Returns
        -------
        dict or None
            Dictionary containing various statistical measures of the
            neuron-feature relationship. Returns None if statistics 
            have not been computed or if data has changed since computation.
            
        Notes
        -----
        Statistics must be pre-computed using the selectivity analysis pipeline.
        The method checks data integrity using hashes to ensure statistics
        are up-to-date with the current data.
        
        .. note::
            Despite the 'get' name, this method can trigger side effects via
            _check_stats_relevance which may add new features to tables.
            This behavior will be removed after tuple multifeature deprecation.        """
        stats = None
        if self._check_stats_relevance(cell_id, feat_id, mode=mode):
            stats = self.stats_tables[mode][feat_id][cell_id]
        else:
            if self.verbose:
                print("Consider recalculating stats")

        return stats

    def get_neuron_feature_pair_significance(self, cell_id, feat_id, mode="calcium"):
        """Get statistical significance data for a neuron-feature pair.
        
        Retrieves significance testing results for the neuron-feature
        relationship, typically from shuffle-based permutation tests.
        
        Parameters
        ----------
        cell_id : int
            Neuron/cell identifier.
        feat_id : str or tuple of str
            Feature identifier(s). Can be a single feature name or tuple
            of feature names for joint analysis.
        mode : {'calcium', 'spikes'}, optional
            Type of neural activity. Default is 'calcium'.
            
        Returns
        -------
        dict or None
            Dictionary containing significance test results including
            p-values, shuffle distributions, and multiple comparison
            corrections. Returns None if significance has not been computed
            or if underlying statistics are outdated.
            
        Notes
        -----
        Significance testing typically uses shuffle tests where temporal
        relationships are destroyed while preserving marginal distributions.
        Results include both single-stage and two-stage testing procedures.
        
        .. note::
            Despite the 'get' name, this method can trigger side effects via
            _check_stats_relevance which may add new features to tables.
            This behavior will be removed after tuple multifeature deprecation.        """
        sig = None
        if self._check_stats_relevance(cell_id, feat_id, mode=mode):
            sig = self.significance_tables[mode][feat_id][cell_id]
        else:
            if self.verbose:
                print("Consider recalculating stats")

        return sig

    def get_multicell_shuffled_calcium(
        self, cbunch=None, method="roll_based", return_array=True, **kwargs
    ):
        """
        Get shuffled calcium data for multiple cells.

        Parameters
        ----------
        cbunch : int, list, or None
            Cell indices. If None, all cells are used.
        method : {'roll_based', 'waveform_based', 'chunks_based'}, default='roll_based'
            Shuffling method to use
        return_array : bool, default=True
            If True, return numpy array. If False, return MultiTimeSeries object.
        **kwargs
            Additional parameters passed to the shuffling method

        Returns
        -------
        np.ndarray or MultiTimeSeries
            If return_array=True: Shuffled calcium data with shape (n_cells, n_frames)
            If return_array=False: MultiTimeSeries object containing shuffled data        """
        # Validate method
        valid_methods = ["roll_based", "waveform_based", "chunks_based"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid shuffling method '{method}'. Must be one of: {valid_methods}"
            )

        cell_list = self._process_cbunch(cbunch)

        # Validate cell indices
        if any(idx < 0 or idx >= self.n_cells for idx in cell_list):
            raise ValueError(
                f"Invalid cell indices. Must be between 0 and {self.n_cells-1}"
            )

        if return_array:
            agg_sh_data = np.zeros((len(cell_list), self.n_frames))
            for i, cell_idx in enumerate(cell_list):
                cell = self.neurons[cell_idx]
                sh_data = cell.get_shuffled_calcium(method=method, return_array=True, **kwargs)
                agg_sh_data[i, :] = sh_data
            return agg_sh_data
        else:
            # Return MultiTimeSeries object
            ts_list = []
            for cell_idx in cell_list:
                cell = self.neurons[cell_idx]
                sh_ts = cell.get_shuffled_calcium(method=method, return_array=False, **kwargs)
                ts_list.append(sh_ts)
            
            # Create MultiTimeSeries from list of TimeSeries
            from ..information.info_base import MultiTimeSeries
            return MultiTimeSeries(ts_list)

    def get_multicell_shuffled_spikes(
        self, cbunch=None, method="isi_based", return_array=True, **kwargs
    ):
        """
        Get shuffled spike data for multiple cells.

        Parameters
        ----------
        cbunch : int, list, or None
            Cell indices. If None, all cells are used.
        method : {'isi_based'}, default='isi_based'
            Shuffling method. Currently only 'isi_based' is supported for spikes.
        return_array : bool, default=True
            If True, return numpy array. If False, return MultiTimeSeries object.
        **kwargs
            Additional parameters passed to the shuffling method

        Returns
        -------
        np.ndarray or MultiTimeSeries
            If return_array=True: Shuffled spike data with shape (n_cells, n_frames)
            If return_array=False: MultiTimeSeries object containing shuffled spike data        """
        # Check if spikes data is meaningful (not all zeros)
        if not np.any(self.spikes.data):
            raise AttributeError(
                "Unable to shuffle spikes without meaningful spikes data"
            )

        # Validate method
        valid_methods = ["isi_based"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid spike shuffling method '{method}'. Must be one of: {valid_methods}"
            )

        cell_list = self._process_cbunch(cbunch)

        # Validate cell indices
        if any(idx < 0 or idx >= self.n_cells for idx in cell_list):
            raise ValueError(
                f"Invalid cell indices. Must be between 0 and {self.n_cells-1}"
            )

        if return_array:
            agg_sh_data = np.zeros((len(cell_list), self.n_frames))
            for i, cell_idx in enumerate(cell_list):
                cell = self.neurons[cell_idx]
                sh_data = cell.get_shuffled_spikes(method=method, return_array=True, **kwargs)
                agg_sh_data[i, :] = sh_data
            return agg_sh_data
        else:
            # Return MultiTimeSeries object
            ts_list = []
            for cell_idx in cell_list:
                cell = self.neurons[cell_idx]
                sh_ts = cell.get_shuffled_spikes(method=method, return_array=False, **kwargs)
                ts_list.append(sh_ts)
            
            # Create MultiTimeSeries from list of TimeSeries
            from ..information.info_base import MultiTimeSeries
            return MultiTimeSeries(ts_list)

    def get_stats_slice(
        self,
        table_to_scan=None,
        cbunch=None,
        fbunch=None,
        sbunch=None,
        significance_mode=False,
        mode="calcium",
    ):
        """
        Returns slice of accumulated statistics data (or significance data if "significance_mode=True").
        
        Parameters
        ----------
        table_to_scan : dict, optional
            Specific table to scan. If None, uses default stats or significance table
        cbunch : int, list, or None, optional
            Cell identifiers to include. If None, includes all cells
        fbunch : str, list, or None, optional
            Feature identifiers to include. If None, includes all features
        sbunch : str, list, or None, optional
            Statistics keys to include. If None, includes all statistics
        significance_mode : bool, optional
            If True, returns significance data instead of statistics. Default is False
        mode : str, optional
            Data processing mode. Default is "calcium"
            
        Returns
        -------
        dict
            Nested dictionary with structure table[feat_id][cell_id][stat_key]
        """
        cell_ids = self._process_cbunch(cbunch)
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True, mode=mode)
        slist = self._process_sbunch(sbunch, significance_mode=significance_mode)

        if table_to_scan is None:
            if significance_mode:
                full_table = self.significance_tables[mode]
            else:
                full_table = self.stats_tables[mode]
        else:
            full_table = table_to_scan

        out_table = self._populate_cell_feat_dict(dict(), fbunch=fbunch, cbunch=cbunch)
        for feat_id in feat_ids:
            for cell_id in cell_ids:
                out_table[feat_id][cell_id] = {
                    s: full_table[feat_id][cell_id][s] for s in slist
                }

        return out_table

    def get_significance_slice(
        self, cbunch=None, fbunch=None, sbunch=None, mode="calcium"
    ):
        """Extract significance test results for selected cells and features.
        
        Convenience method that retrieves statistical significance data
        (p-values, test statistics, etc.) for specific cell-feature combinations.
        This is equivalent to calling get_stats_slice with significance_mode=True.
        
        Parameters
        ----------
        cbunch : int, list of int, or None, optional
            Cell indices to include. None means all cells.
        fbunch : int, str, list, or None, optional
            Feature indices/names to include. None means all features.
        sbunch : str, list of str, or None, optional
            Significance measures to extract (e.g., 'pval', 'qval', 'statistic').
            None means all available measures.
        mode : {'calcium', 'spikes'}, default='calcium'
            Which data type's significance tables to use.
            
        Returns
        -------
        dict
            Nested dictionary with structure: {feature: {cell: {measure: value}}}.
            Contains only the requested significance test results.
            
        See Also
        --------
        ~driada.experiment.exp_base.get_stats_slice : More general method for extracting any statistics
        ~driada.experiment.exp_base._update_stats_and_significance : Internal method that computes significance
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.experiment.exp_base import Experiment
        >>> from driada.information.info_base import TimeSeries
        >>> 
        >>> # Create an example experiment with selectivity data
        >>> calcium_data = np.random.randn(20, 1000)
        >>> running_speed = TimeSeries(np.random.rand(1000))
        >>> exp = Experiment(
        ...     'example',
        ...     calcium_data,
        ...     None,
        ...     {},
        ...     {'fps': 30.0},
        ...     {'running_speed': running_speed},
        ...     verbose=False
        ... )
        >>> 
        >>> # Initialize stats tables
        >>> exp._set_selectivity_tables('calcium')
        >>> 
        >>> # Get p-values for cells 0-5 and feature 'running_speed'
        >>> sig_data = exp.get_significance_slice(
        ...     cbunch=[0, 1, 2, 3, 4, 5],
        ...     fbunch=['running_speed'],
        ...     sbunch=['pval']
        ... )
        >>> # Returns nested dict structure
        >>> sorted(sig_data.keys())
        ['running_speed']
        >>> sorted(sig_data['running_speed'].keys())
        [0, 1, 2, 3, 4, 5]
        """
        return self.get_stats_slice(
            cbunch=cbunch,
            fbunch=fbunch,
            sbunch=sbunch,
            significance_mode=True,
            mode=mode,
        )

    def get_feature_entropy(self, feat_id, ds=1):
        """
        Calculates entropy of a single dynamic feature or joint entropy of multiple features.

        Parameters
        ----------
        feat_id : str or tuple
            - str: Name of a single feature (TimeSeries or MultiTimeSeries)
            - tuple: Names of exactly 2 features for joint entropy calculation
        ds : int
            Downsampling factor

        Returns
        -------
        float
            Entropy value in bits (or nats for continuous variables)

        Notes
        -----
        - Single features use their native get_entropy() method
        - Tuples calculate joint entropy for exactly 2 variables
        - Joint entropy of 3+ variables is not supported (use MultiTimeSeries instead)
        - Continuous variables may return negative entropy values        """
        if isinstance(feat_id, str):
            # Single feature - use its get_entropy method
            fts = self.dynamic_features[feat_id]

            # Check for continuous components and warn
            if isinstance(fts, TimeSeries) and not fts.discrete:
                warnings.warn(
                    f"Feature '{feat_id}' is continuous. Entropy may be negative. "
                    "Consider using mutual information or other measures for continuous variables."
                )
            elif isinstance(fts, MultiTimeSeries):
                # MultiTimeSeries with continuous components
                if not fts.discrete:
                    warnings.warn(
                        f"Feature '{feat_id}' contains continuous components. "
                        "Differential entropy may be negative and depends on the scale/units."
                    )

            return fts.get_entropy(ds=ds)

        elif isinstance(feat_id, (tuple, list)):
            # Joint entropy of multiple features
            if len(feat_id) != 2:
                raise ValueError(
                    f"Joint entropy is only supported for exactly 2 variables, got {len(feat_id)}. "
                    f"For {len(feat_id)} variables, create a MultiTimeSeries instead."
                )

            # Get the two features
            feat1_name, feat2_name = feat_id
            feat1 = self.dynamic_features[feat1_name]
            feat2 = self.dynamic_features[feat2_name]

            # Check for continuous components
            has_continuous = False
            if isinstance(feat1, TimeSeries) and not feat1.discrete:
                has_continuous = True
            elif isinstance(feat1, MultiTimeSeries) and not feat1.discrete:
                has_continuous = True
            if isinstance(feat2, TimeSeries) and not feat2.discrete:
                has_continuous = True
            elif isinstance(feat2, MultiTimeSeries) and not feat2.discrete:
                has_continuous = True

            if has_continuous:
                warnings.warn(
                    "One or both features contain continuous components. "
                    "Joint differential entropy may be negative and is scale-dependent."
                )

            # For joint entropy of 2 variables, use H(X,Y) = H(X) + H(Y) - MI(X,Y)
            h_x = feat1.get_entropy(ds=ds)
            h_y = feat2.get_entropy(ds=ds)
            mi_xy = get_1d_mi(feat1, feat2, ds=ds)

            return h_x + h_y - mi_xy

        else:
            raise TypeError(
                f"feat_id must be str or tuple of 2 feature names, got {type(feat_id)}"
            )

    def _reconstruct_spikes(self, calcium, method, fps, spike_kwargs=None):
        """
        Reconstruct spikes from calcium signals using specified method.

        Parameters
        ----------
        calcium : np.ndarray
            Calcium traces, shape (n_neurons, n_timepoints)
        method : str or callable
            Reconstruction method: 'wavelet' or a callable function
        fps : float
            Sampling rate in frames per second
        spike_kwargs : dict, optional
            Method-specific parameters

        Returns
        -------
        spikes : np.ndarray
            Reconstructed spike trains        """
        from .spike_reconstruction import reconstruct_spikes

        # Convert calcium to MultiTimeSeries if needed
        if isinstance(calcium, np.ndarray):
            # Create temporary MultiTimeSeries from numpy array
            from ..information.info_base import TimeSeries, MultiTimeSeries

            # Calcium data is always continuous, so explicitly set discrete=False
            ts_list = [
                TimeSeries(calcium[i, :], discrete=False)
                for i in range(calcium.shape[0])
            ]
            calcium_mts = MultiTimeSeries(ts_list)
        else:
            calcium_mts = calcium

        # Call the unified reconstruction function
        spikes_mts, metadata = reconstruct_spikes(
            calcium_mts, method=method, fps=fps, params=spike_kwargs
        )

        # Store metadata
        self._reconstruction_metadata = metadata

        # Return numpy array for backward compatibility
        return spikes_mts.data

    def get_significant_neurons(
        self, min_nspec=1, cbunch=None, fbunch=None, mode="calcium",
        override_intense_significance=False, pval_thr=0.05, 
        multicomp_correction=None, significance_update=False
    ):
        """
        Returns a dict with neuron ids as keys and their significantly correlated features as values
        Only neurons with "min_nspec" or more significantly correlated features will be returned

        Parameters
        ----------
        min_nspec : int
            Minimum number of significantly correlated features required
        cbunch : int, list or None
            Cell indices to analyze. By default (None), all neurons will be checked
        fbunch : str, list or None
            Feature names to check. By default (None), all features will be checked
        mode : str
            Data type: 'calcium' or 'spikes'
        override_intense_significance : bool, optional
            If True, recompute significance using pval_thr and multicomp_correction
            instead of using pre-computed INTENSE significance. Default is False.
        pval_thr : float, optional
            P-value threshold for significance testing. Default is 0.05.
            Only used if override_intense_significance=True.
        multicomp_correction : str or None, optional
            Multiple comparison correction method. Default is None (no correction).
            Options: None, 'bonferroni', 'holm', 'fdr_bh'
            Only used if override_intense_significance=True.
        significance_update : bool, optional
            If True, update the significance tables with new thresholds.
            If False (default), only use new thresholds for current query.
            Only used if override_intense_significance=True.

        Returns
        -------
        dict
            Dictionary with neuron IDs as keys and lists of significant features as values        """
        cell_ids = self._process_cbunch(cbunch)
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True, mode=mode)

        # Check relevance only for requested cells and features
        relevance = [
            self._check_stats_relevance(cell_id, feat_id, mode=mode)
            for cell_id in cell_ids
            for feat_id in feat_ids
        ]
        if not np.all(np.array(relevance)):
            raise ValueError("Stats relevance error")

        cell_feat_dict = {cell_id: [] for cell_id in cell_ids}
        
        if override_intense_significance:
            # Collect all p-values for multiple comparison correction
            all_pvals = []
            cell_feat_pvals = {}
            
            for cell_id in cell_ids:
                cell_feat_pvals[cell_id] = {}
                for feat_id in feat_ids:
                    pval = self.stats_tables[mode][feat_id][cell_id].get("pval", 1.0)
                    cell_feat_pvals[cell_id][feat_id] = pval
                    all_pvals.append(pval)
            
            # Calculate corrected threshold
            if multicomp_correction is None:
                corrected_threshold = pval_thr
            elif multicomp_correction == "bonferroni":
                corrected_threshold = get_multicomp_correction_thr(
                    pval_thr, mode="bonferroni", nhyp=len(all_pvals)
                )
            elif multicomp_correction in ["holm", "fdr_bh"]:
                corrected_threshold = get_multicomp_correction_thr(
                    pval_thr, mode=multicomp_correction, all_pvals=all_pvals
                )
            else:
                raise ValueError(
                    f"Unknown multicomp_correction method: {multicomp_correction}. "
                    "Options: None, 'bonferroni', 'holm', 'fdr_bh'"
                )
            
            # Determine significance based on new threshold
            for cell_id in cell_ids:
                for feat_id in feat_ids:
                    pval = cell_feat_pvals[cell_id][feat_id]
                    
                    # Check if significant according to new threshold
                    is_significant = pval < corrected_threshold
                    
                    if is_significant:
                        cell_feat_dict[cell_id].append(feat_id)
                    
                    # Update significance tables if requested
                    if significance_update:
                        self.significance_tables[mode][feat_id][cell_id]["stage2"] = is_significant
                        self.significance_tables[mode][feat_id][cell_id]["pval_thr"] = pval_thr
                        self.significance_tables[mode][feat_id][cell_id]["multicomp_correction"] = multicomp_correction
                        self.significance_tables[mode][feat_id][cell_id]["corrected_pval_thr"] = corrected_threshold
        else:
            # Use pre-computed INTENSE significance
            for cell_id in cell_ids:
                for feat_id in feat_ids:
                    if self.significance_tables[mode][feat_id][cell_id]["stage2"]:
                        cell_feat_dict[cell_id].append(feat_id)

        # filter out cells without enough specializations
        final_cell_feat_dict = {
            cell_id: cell_feat_dict[cell_id]
            for cell_id in cell_ids
            if len(cell_feat_dict[cell_id]) >= min_nspec
        }

        return final_cell_feat_dict

    def store_embedding(
        self, embedding, method_name, data_type="calcium", metadata=None
    ):
        """
        Store dimensionality reduction embedding in the experiment.
        
        This method stores a computed embedding in the experiment's internal
        embeddings dictionary. Previous embeddings with the same method_name
        and data_type will be overwritten without warning.

        Parameters
        ----------
        embedding : np.ndarray
            The embedding array, shape (n_timepoints, n_components). The number
            of timepoints must match self.n_frames or self.n_frames//ds if
            downsampling was used.
        method_name : str
            Name of the DR method (e.g., 'pca', 'umap', 'isomap'). This serves
            as the key for storing and retrieving the embedding.
        data_type : str, optional
            Type of data used ('calcium' or 'spikes'). Default is 'calcium'.
        metadata : dict, optional
            Additional metadata about the embedding. Common keys include:
            - 'ds': Downsampling factor used
            - 'n_components': Number of components
            - 'neuron_indices': Indices of neurons used
            - 'method_params': Parameters specific to the DR method
            
        Raises
        ------
        ValueError
            If data_type is not 'calcium' or 'spikes'.
        ValueError
            If embedding timepoints don't match expected frames. The expected
            number is self.n_frames//ds where ds is extracted from metadata
            (default 1).
            
        Notes
        -----
        The embedding is stored in self.embeddings[data_type][method_name] as a
        dictionary containing:
        - 'data': The embedding array
        - 'metadata': The provided metadata dict (or empty dict)
        - 'timestamp': Current time when stored (np.datetime64)
        - 'shape': Shape tuple of the embedding array
        
        Previous embeddings with the same method_name are silently overwritten.
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.experiment.exp_base import Experiment
        >>> 
        >>> # Create a simple experiment
        >>> calcium_data = np.random.randn(10, 1000)
        >>> exp = Experiment('test', calcium_data, None, {}, 
        ...                  {'fps': 30.0}, {}, verbose=False)
        >>> 
        >>> # Store a PCA embedding
        >>> embedding = np.random.randn(1000, 3)  # 1000 timepoints, 3 components
        >>> exp.store_embedding(embedding, 'pca', metadata={'n_components': 3})
        >>> 
        >>> # Verify storage
        >>> 'pca' in exp.embeddings['calcium']
        True
        >>> 
        >>> # Store a downsampled UMAP embedding
        >>> # If experiment has 1000 frames and ds=5, embedding should have 200 rows
        >>> downsampled_embedding = np.random.randn(200, 2)
        >>> exp.store_embedding(downsampled_embedding, 'umap', 
        ...                    metadata={'ds': 5, 'n_neighbors': 30})
        
        See Also
        --------
        ~driada.experiment.exp_base.get_embedding : Retrieve stored embeddings
        ~driada.experiment.exp_base.create_embedding : Create and store embeddings in one step        """
        if data_type not in ["calcium", "spikes"]:
            raise ValueError("data_type must be 'calcium' or 'spikes'")

        # Check if embedding matches expected timepoints (accounting for downsampling)
        ds = metadata.get("ds", 1) if metadata else 1
        expected_frames = self.n_frames // ds
        if embedding.shape[0] != expected_frames:
            raise ValueError(
                f"Embedding timepoints ({embedding.shape[0]}) must match expected frames "
                f"({expected_frames} = {self.n_frames} / ds={ds})"
            )

        self.embeddings[data_type][method_name] = {
            "data": embedding,
            "metadata": metadata or {},
            "timestamp": np.datetime64("now"),
            "shape": embedding.shape,
        }

    def create_embedding(
        self,
        method: str,
        n_components: int = 2,
        data_type: str = "calcium",
        neuron_selection: Optional[Union[str, List[int]]] = None,
        **dr_kwargs,
    ) -> np.ndarray:
        """
        Create dimensionality reduction embedding and store it.
        
        Notes
        -----
        This method modifies the experiment's state by storing the computed
        embedding. Previous embeddings with the same method name will be overwritten.
        The method uses MultiTimeSeries internally for data handling and applies
        the dimensionality reduction through the MVData interface.

        Parameters
        ----------
        method : str
            DR method name ('pca', 'umap', 'isomap', etc.).
        n_components : int, optional
            Number of embedding dimensions. Default is 2.
        data_type : str, optional
            Type of data to use ('calcium' or 'spikes'). Default is 'calcium'.
        neuron_selection : str, list or None, optional
            How to select neurons:
            - None or 'all': Use all neurons
            - 'significant': Use only significantly selective neurons
            - List of integers: Use specific neuron indices
        **dr_kwargs
            Additional arguments for the DR method (e.g., n_neighbors, min_dist).

        Returns
        -------
        embedding : np.ndarray
            The embedding array, shape (n_timepoints, n_components).
            
        Raises
        ------
        ValueError
            If n_components is not positive, data_type is invalid, downsampling
            factor 'ds' is not an integer, neuron indices are out of bounds,
            significant neurons requested without selectivity analysis, or
            embedding method drops timepoints.
        AttributeError
            If spike data requested but not available.
            
        Examples
        --------
        >>> import numpy as np
        >>> from driada.experiment.exp_base import Experiment
        >>> from driada.information.info_base import TimeSeries
        >>> 
        >>> # Create experiment with some data
        >>> calcium_data = np.random.randn(25, 1000)
        >>> speed = TimeSeries(np.random.rand(1000))
        >>> exp = Experiment('test', calcium_data, None, {},
        ...                  {'fps': 30.0}, {'speed': speed}, verbose=False)
        >>> 
        >>> # Create PCA embedding using all neurons
        >>> embedding = exp.create_embedding('pca', n_components=10)
        Calculating PCA embedding...
        >>> embedding.shape
        (1000, 10)
        >>> 
        >>> # Create downsampled PCA with specific neurons  
        >>> embedding = exp.create_embedding('pca', n_components=2,
        ...                                neuron_selection=[0, 1, 2, 3, 4], ds=10)
        Calculating PCA embedding...
        
        See Also
        --------
        ~driada.experiment.exp_base.store_embedding : Store computed embeddings
        ~driada.experiment.exp_base.get_embedding : Retrieve stored embeddings
        ~driada.experiment.exp_base.get_significant_neurons : Get neurons with significant selectivity        """
        from ..information.info_base import MultiTimeSeries
        from ..utils.data import check_positive
        
        # Validate inputs
        check_positive(n_components=n_components)
        if data_type not in ['calcium', 'spikes']:
            raise ValueError("data_type must be 'calcium' or 'spikes'")
            
        # Select neurons
        if neuron_selection is None or neuron_selection == "all":
            neuron_indices = np.arange(self.n_cells)
        elif neuron_selection == "significant":
            has_selectivity = (
                hasattr(self, "stats_tables") and 
                self.stats_tables is not None and
                data_type in self.stats_tables and
                len(self.stats_tables[data_type]) > 0
            )
            if not has_selectivity:
                raise ValueError(
                    "Cannot select significant neurons without selectivity analysis"
                )
            sig_neurons = self.get_significant_neurons()
            neuron_indices = np.array(list(sig_neurons.keys()))
            if len(neuron_indices) == 0:
                logging.warning("No significant neurons found, using all neurons")
                neuron_indices = np.arange(self.n_cells)
        else:
            neuron_indices = np.array(neuron_selection)
            # Validate neuron indices are within bounds
            if len(neuron_indices) > 0:
                if np.any(neuron_indices < 0) or np.any(neuron_indices >= self.n_cells):
                    raise ValueError(
                        f"Neuron indices must be in range [0, {self.n_cells-1}]"
                    )

        # Get neural data - calcium and spikes are already MultiTimeSeries
        if data_type == "calcium":
            multi_ts = self.calcium
        else:
            if not hasattr(self, "spikes") or self.spikes is None:
                raise AttributeError("Experiment has no spike data")
            multi_ts = self.spikes
            
        # Create subset MultiTimeSeries with selected neurons
        if len(neuron_indices) != self.n_cells:
            subset_data = multi_ts.data[neuron_indices, :]
            multi_ts = MultiTimeSeries(
                subset_data, 
                discrete=multi_ts.discrete,
                allow_zero_columns=(data_type == "spikes")
            )

        # Apply downsampling if requested
        ds = dr_kwargs.pop("ds", 1)  # Remove 'ds' from dr_kwargs
        if ds > 1:
            check_positive(ds=ds)
            if not isinstance(ds, int):
                raise ValueError("Downsampling factor 'ds' must be an integer")
            # Create downsampled MultiTimeSeries
            downsampled_data = multi_ts.data[:, ::ds]
            multi_ts = MultiTimeSeries(
                downsampled_data,
                discrete=multi_ts.discrete,
                allow_zero_columns=(data_type == "spikes")
            )
            logging.info(
                f"Downsampling data by factor {ds}: {multi_ts.data.shape[1]} timepoints"
            )

        # Prepare parameters for dimensionality reduction
        params = {"dim": n_components}
        params.update(dr_kwargs)  # Add all additional parameters

        # Get embedding using MultiTimeSeries/MVData method
        embedding_obj = multi_ts.get_embedding(method=method, **params)
        embedding = embedding_obj.coords.T  # Transpose to (n_timepoints, n_components)

        # Check if embedding has all timepoints (accounting for downsampling)
        expected_frames = self.n_frames // ds
        if embedding.shape[0] < expected_frames:
            n_missing = expected_frames - embedding.shape[0]
            raise ValueError(
                f"{method} embedding dropped {n_missing} timepoints due to graph disconnection. "
                f"This is not supported for INTENSE analysis. Try increasing n_neighbors or using a different method."
            )

        # Store metadata
        metadata = {
            "method": method,
            "n_components": n_components,
            "neuron_selection": neuron_selection,
            "neuron_indices": neuron_indices.tolist(),
            "n_neurons": len(neuron_indices),
            "dr_params": dr_kwargs,
            "data_type": data_type,
            "ds": ds,  # Store downsampling factor
        }

        # Store in experiment
        self.store_embedding(embedding, method, data_type, metadata)

        logging.info(
            f"Created {method} embedding with {n_components} components "
            f"using {len(neuron_indices)} neurons"
        )

        return embedding

    def get_embedding(self, method_name, data_type="calcium"):
        """
        Retrieve stored embedding.
        
        This method retrieves a previously stored dimensionality reduction
        embedding from the experiment's embeddings dictionary. The returned
        dictionary contains the embedding data along with metadata and timestamp.

        Parameters
        ----------
        method_name : str
            Name of the DR method to retrieve (e.g., 'pca', 'umap').
        data_type : str, optional
            Type of data used ('calcium' or 'spikes'). Default is 'calcium'.

        Returns
        -------
        dict
            Dictionary containing:
            - 'data': The embedding array (n_timepoints, n_components)
            - 'metadata': Dict with embedding parameters and settings
            - 'timestamp': np.datetime64 when the embedding was stored
            - 'shape': Tuple with shape of the embedding array
            
        Raises
        ------
        ValueError
            If data_type is not 'calcium' or 'spikes'.
        KeyError
            If no embedding found for the specified method and data type.
            
        Notes
        -----
        To see available embeddings, check exp.embeddings[data_type].keys().
        The returned dictionary is a reference to the stored data, so
        modifications will affect the stored embedding.
            
        Examples
        --------
        >>> import numpy as np
        >>> from driada.experiment.exp_base import Experiment
        >>> 
        >>> # Create experiment and store an embedding
        >>> calcium_data = np.random.randn(10, 1000)
        >>> exp = Experiment('test', calcium_data, None, {},
        ...                  {'fps': 30.0}, {}, verbose=False)
        >>> 
        >>> # Store an embedding first
        >>> embedding = np.random.randn(1000, 3)
        >>> exp.store_embedding(embedding, 'pca', metadata={'n_components': 3})
        >>> 
        >>> # Retrieve the stored PCA embedding
        >>> embedding_dict = exp.get_embedding('pca')
        >>> embedding_data = embedding_dict['data']
        >>> print(f"Embedding shape: {embedding_dict['shape']}")
        Embedding shape: (1000, 3)
        >>> 
        >>> # Check available embeddings before retrieval
        >>> available = list(exp.embeddings['calcium'].keys())
        >>> print(f"Available embeddings: {available}")
        Available embeddings: ['pca']
        
        See Also
        --------
        ~driada.experiment.exp_base.store_embedding : Store embeddings in the experiment
        ~driada.experiment.exp_base.create_embedding : Create and store embeddings in one step        """
        if data_type not in ["calcium", "spikes"]:
            raise ValueError("data_type must be 'calcium' or 'spikes'")

        if method_name not in self.embeddings[data_type]:
            raise KeyError(
                f"No embedding found for method '{method_name}' with data_type '{data_type}'"
            )

        return self.embeddings[data_type][method_name]

    def compute_rdm(
        self,
        items,
        data_type="calcium",
        metric="correlation",
        average_method="mean",
        use_cache=True,
    ):
        """
        Compute RDM with caching support.

        Parameters
        ----------
        items : str
            Name of dynamic feature to use as condition labels
        data_type : str, default 'calcium'
            Type of data to use ('calcium' or 'spikes')
        metric : str, default 'correlation'
            Distance metric for RDM computation
        average_method : str, default 'mean'
            How to average within conditions ('mean' or 'median')
        use_cache : bool, default True
            Whether to use cached results

        Returns
        -------
        rdm : np.ndarray
            Representational dissimilarity matrix
        labels : np.ndarray
            The unique labels/conditions        """
        # Generate cache key
        cache_key = (items, data_type, metric, average_method)

        # Check cache
        if use_cache and cache_key in self._rdm_cache:
            return self._rdm_cache[cache_key]

        # Import here to avoid circular dependency
        from ..rsa.integration import compute_experiment_rdm

        # Compute RDM
        result = compute_experiment_rdm(
            self,
            items,
            data_type=data_type,
            metric=metric,
            average_method=average_method,
        )

        # Cache result
        if use_cache:
            self._rdm_cache[cache_key] = result

        return result

    def clear_rdm_cache(self):
        """Clear the representational dissimilarity matrix (RDM) cache.
        
        Removes all cached RDM computations to free memory or force
        recalculation with updated data. This is necessary after modifying
        the underlying neural data or when memory usage is a concern.
        
        Notes
        -----
        The RDM cache stores previously computed dissimilarity matrices to
        avoid expensive recomputation. Clear the cache when:
        - Neural data has been modified or reprocessed
        - Embeddings have been updated
        - Memory usage needs to be reduced
        - You want to force fresh computation with different parameters
        
        After clearing, subsequent calls to compute_rdm() will recalculate
        the RDM from scratch, which may be computationally expensive for
        large datasets.
        
        See Also
        --------
        ~driada.experiment.exp_base.compute_rdm : Method that uses and populates the RDM cache
        
        Examples
        --------
        >>> import numpy as np
        >>> from driada.experiment.exp_base import Experiment
        >>> from driada.information.info_base import TimeSeries
        >>> 
        >>> # Create experiment with a categorical feature
        >>> calcium_data = np.random.randn(10, 1000)
        >>> conditions = TimeSeries(np.repeat([0, 1, 2], [333, 333, 334]), discrete=True)
        >>> exp = Experiment('test', calcium_data, None, {},
        ...                  {'fps': 30.0}, {'conditions': conditions}, verbose=False)
        >>> 
        >>> # Compute RDM (will be cached)
        >>> rdm, labels = exp.compute_rdm('conditions')
        >>> 
        >>> # Update embedding and clear cache
        >>> new_embedding = np.random.randn(1000, 3)
        >>> exp.store_embedding(new_embedding, 'pca')
        >>> exp.clear_rdm_cache()
        >>> 
        >>> # Verify cache is empty
        >>> len(exp._rdm_cache)
        0
        """
        self._rdm_cache = {}

