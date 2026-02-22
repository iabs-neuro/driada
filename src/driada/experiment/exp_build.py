import copy
import os
import os.path
import warnings
import numpy as np
import pickle

from .exp_base import Experiment
from ..information.info_base import TimeSeries, MultiTimeSeries, aggregate_multiple_ts
from ..information.time_series_types import analyze_time_series_type
from ..utils.naming import construct_session_name
from ..utils.output import show_output
from .neuron import DEFAULT_FPS, DEFAULT_T_OFF, DEFAULT_T_RISE
from ..gdrive.download import download_gdrive_data, initialize_iabs_router

# Reserved keys that should not become behavioral features (all lowercase)
# These are neural data or metadata, not behavioral variables
NEURAL_DATA_ALIASES = {"calcium", "activations", "neural_data", "activity", "rates"}
RESERVED_NEURAL_KEYS = NEURAL_DATA_ALIASES | {"spikes", "sp", "asp", "reconstructions"}
RESERVED_METADATA_KEYS = {"_metadata", "_sync_info"}


def _format_feature_subtype(type_info):
    """Format subtype information for verbose logging.

    Parameters
    ----------
    type_info : TimeSeriesType or None
        Type information from a TimeSeries object.

    Returns
    -------
    str
        Formatted subtype string, e.g., "linear", "circular (360°)", "binary".
        Returns empty string if subtype is None.
    """
    if type_info is None or type_info.subtype is None:
        return ""

    subtype_str = type_info.subtype

    if type_info.is_circular and type_info.circular_period is not None:
        period = type_info.circular_period
        if abs(period - 360) < 1:
            subtype_str = "circular (360)"
        elif abs(period - 2 * 3.14159265) < 0.1:
            subtype_str = "circular (2pi)"
        else:
            subtype_str = f"circular ({period:.1f})"

    return subtype_str


def load_exp_from_aligned_data(
    data_source,
    exp_params,
    data,
    force_continuous=[],
    feature_types=None,
    bad_frames=[],
    static_features=None,
    verbose=True,
    reconstruct_spikes=None,
    aggregate_features=None,
    n_jobs=-1,
    enable_parallelization=True,
    create_circular_2d=True,
):
    """Create an Experiment object from aligned neural and behavioral data.

    Constructs an Experiment instance from pre-aligned calcium imaging data
    and behavioral variables, automatically determining feature types and
    filtering out constant or invalid features.

    Parameters
    ----------
    data_source : str
        Identifier for the data source (e.g., 'IABS', 'custom').
        Used with exp_params to construct the experiment name.
    exp_params : dict
        Experiment parameters dictionary. For IABS data source, requires:

        - 'track': experimental paradigm (e.g., 'linear_track')
        - 'animal_id': subject identifier
        - 'session': session identifier

        For other sources, can contain any metadata for experiment naming.
    data : dict
        Dictionary containing aligned data with keys:

        - 'calcium' or 'Calcium': 2D array of calcium signals (neurons x time)
        - 'spikes' or 'Spikes': 2D array of spike data (optional)
        - Other keys: behavioral variables as 1D or 2D arrays.
          1D arrays (time,) are treated as single time series;
          2D arrays (components, time) are treated as MultiTimeSeries.
    force_continuous : list, optional
        **Deprecated.** Use ``feature_types`` instead. List of feature names
        to force as continuous. Converted to ``feature_types={f: 'continuous'}``
        internally if ``feature_types`` is not provided.
    feature_types : dict[str, str], optional
        Map of feature names to type strings, overriding auto-detection.
        See ``TimeSeries._create_type_from_string`` for valid strings.
        Also acts as a circular whitelist: unlisted auto-circular features
        are overridden to linear.
    bad_frames : list, optional
        List of frame indices to mark as bad/invalid. These frames will be
        masked in the resulting Experiment object. Useful for removing
        motion artifacts or recording gaps.
    static_features : dict, optional
        Static experimental parameters. Common keys:

        - 't_rise_sec': calcium rise time (default: 0.25)
        - 't_off_sec': calcium decay time (default: 2.0)
        - 'fps': frame rate in Hz (default: 20.0)
        - Any other experiment-specific constants
    verbose : bool, default=True
        Whether to print progress and feature information.
    reconstruct_spikes : str, bool, or None, default=None
        **DEPRECATED**: This parameter is deprecated. Load the experiment first,
        then call exp.reconstruct_all_neurons() separately for better control.

        If provided (for backward compatibility):
        - 'wavelet': wavelet-based detection (old batch method)
        - False/None: no reconstruction (recommended)

        New workflow (recommended):
        >>> exp = load_exp_from_aligned_data(data_source, exp_params, data)
        >>> exp.reconstruct_all_neurons(method='wavelet', n_iter=3)

    aggregate_features : dict, optional
        Dictionary mapping tuples of feature keys to combined names.
        Allows pre-specifying which features should be combined into
        MultiTimeSeries before Experiment building. This is useful for
        deterministic data hash generation.

        Format: {(key1, key2, ...): "combined_name", ...}

        Example:
        >>> aggregate_features = {
        ...     ("x", "y"): "position",  # Combine x, y into 2D MultiTimeSeries
        ...     ("speed", "direction"): "velocity",
        ... }

        The component features remain available as individual features
        in addition to the combined MultiTimeSeries.
    n_jobs : int, default=-1
        Number of parallel jobs for neuron construction and other parallel
        operations. Use -1 for all available cores, 1 to disable parallelization.
    enable_parallelization : bool, default=True
        Enable parallel processing for neuron construction and hash computation.
        Set to False to use sequential processing (useful for debugging).
    create_circular_2d : bool, default=True
        If True, automatically create `_2d` versions of circular features
        (detected via type_info.is_circular) as (cos, sin) MultiTimeSeries.
        Original features are preserved. E.g., 'headdirection' -> also creates
        'headdirection_2d'. This improves MI estimation accuracy for circular
        variables like head direction.

    Returns
    -------
    Experiment
        Initialized Experiment object with processed data.

    Raises
    ------
    TypeError
        If data or exp_params are not dictionaries.
    ValueError
        If data is empty or calcium data is missing.

    Side Effects
    ------------
    - Prints feature information if verbose=True
    - Creates deep copy of input data

    Notes
    -----
    - Features with ≤1 unique non-NaN values are filtered as "garbage"
    - Feature types (discrete/continuous) are automatically determined by checking
      if values appear to be categorical (few unique values) or continuous
    - Case-insensitive key matching for 'calcium' and 'spikes'
    - Creates a deep copy of input data to avoid modifying the original
    - The experiment name is constructed using construct_session_name()
    - Bad frames create a boolean mask; indices beyond data length are ignored
    - Scalar values (0D arrays) are ignored with a warning - use static_features instead
    - Non-numeric features (strings, objects) are ignored with a warning
    - 2D arrays are automatically converted to MultiTimeSeries objects

    Examples
    --------
    >>> # Basic usage with minimal data
    >>> np.random.seed(42)  # For reproducibility
    >>> data = {
    ...     'calcium': np.random.rand(50, 1000),  # 50 neurons, 1000 frames
    ...     'position': np.linspace(0, 100, 1000),  # Linear track position
    ...     'speed': np.random.rand(1000) * 10,  # Random speeds
    ...     'trial_type': np.repeat([0, 1, 0, 1], 250)  # Discrete variable
    ... }
    >>> exp_params = {
    ...     'track': 'linear_track',
    ...     'animal_id': 'mouse01',
    ...     'session': 'day1'
    ... }
    >>> exp = load_exp_from_aligned_data('IABS', exp_params, data, verbose=False)
    >>> exp.signature
    'Exp linear_track_mouse01_day1'
    >>> sorted(exp.dynamic_features.keys())
    ['position', 'speed', 'trial_type']

    >>> # Force discrete variable to be continuous
    >>> exp2 = load_exp_from_aligned_data(
    ...     'IABS', exp_params, data,
    ...     force_continuous=['trial_type'],
    ...     bad_frames=[10, 11, 12],  # Mark frames as bad
    ...     static_features={'fps': 30.0},  # Override default fps
    ...     verbose=False
    ... )
    >>> exp2.static_features['fps']
    30.0
    >>> exp2.dynamic_features['trial_type'].discrete  # Should be False due to force_continuous
    False
    """

    # Deprecation warning for reconstruct_spikes parameter
    if reconstruct_spikes is not None and reconstruct_spikes is not False:
        warnings.warn(
            "The 'reconstruct_spikes' parameter is deprecated. "
            "Load the experiment first, then call exp.reconstruct_all_neurons() separately. "
            "Example: exp = load_exp_from_aligned_data(...); exp.reconstruct_all_neurons(method='wavelet')",
            DeprecationWarning,
            stacklevel=2,
        )

    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dictionary, got {type(data).__name__}")
    if not data:
        raise ValueError("data dictionary cannot be empty")
    if not isinstance(exp_params, dict):
        raise TypeError(f"exp_params must be a dictionary, got {type(exp_params).__name__}")

    expname = construct_session_name(data_source, exp_params)
    adata = copy.deepcopy(data)
    key_mapping = {key.lower(): key for key in adata.keys()}

    if verbose:
        print(f"Building experiment {expname}...")

    neural_key = None
    for alias in NEURAL_DATA_ALIASES:
        if alias in key_mapping:
            neural_key = alias
            break
    if neural_key is not None:
        calcium = adata.pop(key_mapping[neural_key])
    else:
        raise ValueError(
            f"No neural data found. Use one of these keys: {sorted(NEURAL_DATA_ALIASES)}"
        )

    spikes = None
    if "spikes" in key_mapping:
        spikes = adata.pop(key_mapping["spikes"])

    # Extract asp (optional)
    asp = None
    if "asp" in key_mapping:
        asp = adata.pop(key_mapping["asp"])

    # Extract reconstructions (optional)
    reconstructions = None
    if "reconstructions" in key_mapping:
        reconstructions = adata.pop(key_mapping["reconstructions"])

    # Extract metadata (merge _sync_info into it if present)
    metadata = None
    if "_metadata" in adata:
        metadata = adata.pop("_metadata")
        if hasattr(metadata, 'item'):
            metadata = metadata.item()

    if "_sync_info" in adata:
        sync_info = adata.pop("_sync_info")
        if hasattr(sync_info, 'item'):
            sync_info = sync_info.item()
        if metadata is None:
            metadata = {}
        metadata['sync_info'] = sync_info

    # Process dynamic features, handling multidimensional arrays
    filt_dyn_features = {}

    # Process feature aggregations first (before individual feature processing)
    # Note: component features are NOT consumed - they remain available as individual features
    if aggregate_features:
        for component_keys, combined_name in aggregate_features.items():
            # Validate all component keys exist
            missing = [k for k in component_keys if k not in adata]
            if missing:
                if verbose:
                    warnings.warn(f"Skipping aggregation '{combined_name}': missing keys {missing}")
                continue

            # Read component arrays (don't pop - keep them for individual processing)
            ts_list = []
            for i, key in enumerate(component_keys):
                arr = np.asarray(adata[key])
                if arr.ndim != 1:
                    raise ValueError(f"Aggregation component '{key}' must be 1D, got {arr.ndim}D")
                ts = TimeSeries(arr, discrete=False, name=f"{combined_name}_{i}")
                ts_list.append(ts)

            # Create MultiTimeSeries from components (adds noise to break degeneracy)
            filt_dyn_features[combined_name] = aggregate_multiple_ts(*ts_list, name=combined_name)

    dyn_features = adata.copy()

    def is_garbage(vals):
        """Check if values are constant or all NaN.

        Parameters
        ----------
        vals : array-like
            Values to check for validity.

        Returns
        -------
        bool
            True if values are all NaN, constant, or empty.

        Notes
        -----
        Used to filter out uninformative features from dynamic data."""
        # Convert to numpy array for consistent handling
        arr = np.asarray(vals)

        # Check if empty
        if arr.size == 0:
            return True

        # Check if all NaN or all same value (ignoring NaN)
        nan_mask = np.isnan(arr)
        return np.all(nan_mask) or (len(np.unique(arr[~nan_mask])) <= 1)

    # Process remaining dynamic features
    # Deprecation bridge: convert force_continuous to feature_types
    if force_continuous and not feature_types:
        warnings.warn(
            "force_continuous is deprecated. Use feature_types={'name': 'linear'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        feature_types = {f: 'continuous' for f in force_continuous}
    feature_types = feature_types or {}

    for f, vals in dyn_features.items():
        # Skip reserved keys (case-insensitive for neural keys)
        if f.lower() in RESERVED_NEURAL_KEYS or f in RESERVED_METADATA_KEYS:
            continue

        # Convert to numpy array to check dimensions
        vals_array = np.asarray(vals)

        # Skip scalar values with warning
        if vals_array.ndim == 0:
            if verbose:
                print(
                    f"Warning: Ignoring scalar value '{f}' found in NPZ file. "
                    f"Scalar values should be provided via static_features parameter."
                )
            continue

        # Skip non-numeric features with warning
        if vals_array.dtype.kind in ["U", "S", "O"]:  # Unicode, bytes, or object
            if verbose:
                print(
                    f"Warning: Ignoring non-numeric feature '{f}' with dtype {vals_array.dtype}. "
                    f"Only numeric features are supported."
                )
            continue

        if is_garbage(vals):
            continue

        # Handle based on dimensionality
        if vals_array.ndim == 1:
            # 1D -> TimeSeries
            if f in feature_types:
                filt_dyn_features[f] = TimeSeries(vals_array, ts_type=feature_types[f], name=f)
            else:
                # Let TimeSeries auto-detect the type
                filt_dyn_features[f] = TimeSeries(vals_array, name=f)

        elif vals_array.ndim == 2:
            # 2D -> MultiTimeSeries (each row is a component)
            # This matches Experiment.__init__ behavior
            ts_list = [
                TimeSeries(vals_array[i, :], discrete=False, name=f"{f}_{i}") for i in range(vals_array.shape[0])
            ]
            filt_dyn_features[f] = MultiTimeSeries(ts_list, name=f)

        else:
            # Skip features with unsupported dimensions
            if verbose:
                print(f"Warning: Skipping feature '{f}' with unsupported {vals_array.ndim}D shape")

    if verbose:
        print("behaviour variables:")
        print()
        for f, ts in filt_dyn_features.items():
            dtype = "discrete" if ts.discrete else "continuous"

            if isinstance(ts, MultiTimeSeries):
                type_info = ts.ts_list[0].type_info if ts.ts_list else None
                subtype_str = _format_feature_subtype(type_info)
                dim_str = f"multi-dimensional ({ts.n_dim}D)"
                if subtype_str:
                    print(f"'{f}' {dtype} {dim_str} {subtype_str}")
                else:
                    print(f"'{f}' {dtype} {dim_str}")
            else:
                subtype_str = _format_feature_subtype(ts.type_info)
                if subtype_str:
                    print(f"'{f}' {dtype} {subtype_str}")
                else:
                    print(f"'{f}' {dtype}")

    # check for constant features
    constfeats = set(dyn_features.keys()) - set(filt_dyn_features.keys())

    if len(constfeats) != 0 and verbose:
        print(f"features {constfeats} dropped as constant or empty")

    auto_continuous = [fn for fn, ts in filt_dyn_features.items() if not ts.discrete]
    if verbose:
        print(f"features {auto_continuous} automatically determined as continuous")
        print()

    # Compare forced types with auto-detection and enforce circular whitelist
    if feature_types:
        circular_whitelist = {f for f, t in feature_types.items()
                              if t in ('circular', 'phase', 'angle')}

        for f, ts in list(filt_dyn_features.items()):
            if not isinstance(ts, TimeSeries) or ts.discrete:
                continue

            if f in feature_types:
                # Warn when forced type disagrees with auto-detection
                auto_type = analyze_time_series_type(ts.data, name=f)
                auto_sub = auto_type.subtype or auto_type.primary_type
                forced_sub = ts.type_info.subtype or ts.type_info.primary_type
                if auto_sub != forced_sub:
                    warnings.warn(
                        f"Feature '{f}' type overridden: auto-detected "
                        f"'{auto_sub}' (conf={auto_type.confidence:.2f}) "
                        f"-> forced '{forced_sub}'",
                        UserWarning,
                        stacklevel=2,
                    )
            elif ts.type_info and ts.type_info.is_circular:
                # Auto-detected circular but not whitelisted — override to linear
                warnings.warn(
                    f"Feature '{f}' auto-detected as circular but not in "
                    f"feature_types. Overriding to linear.",
                    UserWarning,
                    stacklevel=2,
                )
                filt_dyn_features[f] = TimeSeries(ts.data, ts_type='linear', name=f)

    signature = f"Exp {expname}"

    # set default static experiment features if not provided
    default_static_features = {
        "t_rise_sec": DEFAULT_T_RISE,
        "t_off_sec": DEFAULT_T_OFF,
        "fps": DEFAULT_FPS,
    }

    if static_features is None:
        static_features = dict()
    for sf in default_static_features.keys():
        if sf not in static_features:
            static_features.update({sf: default_static_features[sf]})

    # Auto-set fps from metadata if not already specified
    if metadata is not None and 'fps' in metadata:
        if 'fps' not in static_features or static_features['fps'] == DEFAULT_FPS:
            static_features['fps'] = metadata['fps']

    exp = Experiment(
        signature,
        calcium,
        spikes,
        exp_params,
        static_features,
        filt_dyn_features,
        reconstruct_spikes=reconstruct_spikes,
        # bad_frames_mask: True = bad frame to remove, False = good frame to keep
        bad_frames_mask=np.array([i in bad_frames for i in range(calcium.shape[1])]),
        verbose=verbose,
        n_jobs=n_jobs,
        enable_parallelization=enable_parallelization,
        asp=asp,
        reconstructions=reconstructions,
        metadata=metadata,
        create_circular_2d=create_circular_2d,
    )

    return exp


def load_experiment(
    data_source,
    exp_params,
    force_rebuild=False,
    force_reload=False,
    via_pydrive=True,
    gauth=None,
    root="DRIADA data",
    exp_path=None,
    data_path=None,
    force_continuous=[],
    feature_types=None,
    bad_frames=[],
    static_features=None,
    reconstruct_spikes="wavelet",
    save_to_pickle=False,
    verbose=True,
    router_source=None,
):
    """Load or create an Experiment object with automatic caching and cloud support.

    This function provides a high-level interface for loading experiments with
    smart caching, automatic cloud data download (for IABS data), and pickle
    serialization. It first checks for cached experiments, then loads from
    local data files, and finally downloads from cloud storage if needed.

    Parameters
    ----------
    data_source : str
        Data source identifier. 'IABS' enables automatic cloud download.
        Other sources (e.g., 'MyLab') require data_path parameter pointing
        to a local NPZ file.
    exp_params : dict
        Experiment parameters dictionary. See load_exp_from_aligned_data
        for required fields based on data_source.
    force_rebuild : bool, default=False
        If True, rebuild experiment from data files even if pickle cache exists.
        The existing pickle is ignored completely.
    force_reload : bool, default=False
        If True, re-download data from cloud even if local files exist.
        Also bypasses pickle cache (similar to force_rebuild).
    via_pydrive : bool, default=True
        Use PyDrive for Google Drive access. If False, uses alternative method.
    gauth : GoogleAuth object, optional
        Pre-authenticated GoogleAuth object for Drive access.
        If None, will create new authentication.
    root : str, default='DRIADA data'
        Root directory for storing experiments and data.
    exp_path : str, optional
        Custom path for experiment pickle file. If None, uses standard
        naming: {root}/{expname}/Exp {expname}.pickle
    data_path : str, optional
        Path to NPZ data file. Required for non-IABS data sources.
        For IABS, if None, uses standard naming:
        {root}/{expname}/Aligned data/{expname} syn data.npz
    force_continuous : list, optional
        **Deprecated.** See load_exp_from_aligned_data.
    feature_types : dict[str, str], optional
        Feature type overrides. See load_exp_from_aligned_data.
    bad_frames : list, optional
        Frame indices to mark as bad. See load_exp_from_aligned_data.
    static_features : dict, optional
        Static experimental parameters. See load_exp_from_aligned_data.
    reconstruct_spikes : str or bool, default='wavelet'
        Spike reconstruction method. See load_exp_from_aligned_data.
    save_to_pickle : bool, default=False
        Whether to save the experiment to pickle after creation.
    verbose : bool, default=True
        Print progress messages.
    router_source : str, pandas.DataFrame, or None, optional
        Source of the router data for IABS experiments:
        - None: Downloads from URL in config.py (default behavior)
        - str: Direct Google Sheets export URL
        - pandas.DataFrame: Pre-loaded router DataFrame
        Only used when data_source='IABS' and downloading from cloud.

    Returns
    -------
    tuple
        exp : Experiment
            The loaded or created Experiment object.
        load_log : list or None
            Cloud download log if data was downloaded, None otherwise.
            Always None for local loads or pickle loads.

    Raises
    ------
    ValueError
        If root exists but is not a directory.
        If data_source is not 'IABS' and no data_path provided.
    FileNotFoundError
        If data file not found and cannot be downloaded.

    Side Effects
    ------------
    - Creates root directory if it doesn't exist
    - Creates experiment subdirectory structure
    - Downloads data from cloud for IABS source (if needed)
    - Saves pickle file if save_to_pickle=True and building from data
    - Prints progress messages if verbose=True

    Notes
    -----
    Loading priority:
    1. If pickle exists and not force_rebuild/reload: load from pickle
    2. If local data exists and not force_reload: load from data file
    3. If IABS source: attempt cloud download
    4. Otherwise: raise error

    For IABS data, expects cloud structure with 'Aligned data' containing
    npz files with calcium and behavioral data.

    The function returns a tuple (exp, load_log) to maintain backward
    compatibility, even though load_log is often None.

    Examples
    --------
    >>> # Load IABS data with custom router URL
    >>> url = "https://docs.google.com/spreadsheets/d/.../export?format=xlsx"
    >>> exp, _ = load_experiment(  # doctest: +SKIP
    ...     'IABS',
    ...     {'track': 'linear', 'animal_id': 'CA1_01', 'session': '1'},
    ...     router_source=url,
    ...     verbose=False
    ... )

    >>> # Load external lab data from NPZ file
    >>> import tempfile
    >>> import numpy as np
    >>>
    >>> # Create test data file
    >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
    ...     temp_data = f.name
    >>> test_data = {
    ...     'calcium': np.random.rand(30, 500),
    ...     'position': np.random.rand(500) * 100
    ... }
    >>> np.savez(temp_data, **test_data)
    >>>
    >>> # Load from local file
    >>> exp, _ = load_experiment(
    ...     'MyLab',
    ...     {'name': 'test_exp'},
    ...     data_path=temp_data,
    ...     verbose=False
    ... )
    >>> exp.signature
    'Exp test_exp'
    >>> exp.n_cells
    30
    >>>
    >>> # Force rebuild even if pickle exists
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     exp2, _ = load_experiment(
    ...         'MyLab',
    ...         {'name': 'rebuild_test'},
    ...         data_path=temp_data,
    ...         root=tmpdir,
    ...         force_rebuild=True,
    ...         save_to_pickle=True,
    ...         verbose=False
    ...     )
    >>> exp2.n_cells
    30
    >>>
    >>> # Cleanup
    >>> import os
    >>> os.unlink(temp_data)"""

    if os.path.exists(root) and not os.path.isdir(root):
        raise ValueError("Root must be a folder!")
    os.makedirs(root, exist_ok=True)

    if exp_path is None:
        expname = construct_session_name(data_source, exp_params)
        exp_path = os.path.join(root, expname, f"Exp {expname}.pickle")

    if os.path.exists(exp_path) and not force_rebuild and not force_reload:
        Exp = load_exp_from_pickle(exp_path, verbose=verbose)
        return Exp, None

    else:
        if data_source == "IABS":
            if data_path is None:
                data_path = os.path.join(root, expname, "Aligned data", f"{expname} syn data.npz")
                if verbose:
                    print(f"Path to data: {data_path}")

            data_exists = os.path.exists(data_path)
            if verbose:
                if data_exists:
                    print("Aligned data for experiment construction found successfully")
                else:
                    print("Failed to locate aligned data for experiment construction")

            if force_reload or not data_exists:
                if verbose:
                    print("Loading data from cloud storage...")
                data_router, data_pieces = initialize_iabs_router(
                    root=root, router_source=router_source
                )
                success, load_log = download_gdrive_data(
                    data_router,
                    expname,
                    data_pieces=["Aligned data"],
                    via_pydrive=via_pydrive,
                    tdir=root,
                    gauth=gauth,
                )

                if not success:
                    print("===========   BEGINNING OF LOADING LOG   ============")
                    show_output(load_log)
                    print("===========   END OF LOADING LOG   ============")
                    raise FileNotFoundError(f"Cannot download {expname}, see loading log above")

            else:
                load_log = None

            aligned_data = dict(np.load(data_path))
            Exp = load_exp_from_aligned_data(
                data_source,
                exp_params,
                aligned_data,
                force_continuous=force_continuous,
                feature_types=feature_types,
                static_features=static_features,
                verbose=verbose,
                bad_frames=bad_frames,
                reconstruct_spikes=reconstruct_spikes,
            )

            if save_to_pickle:
                save_exp_to_pickle(Exp, exp_path, verbose=verbose)
            return Exp, load_log

        else:
            # Support for external (non-IABS) data sources loading from local files
            if data_path is None:
                raise ValueError(
                    f"For data source '{data_source}', you must provide the 'data_path' parameter "
                    "pointing to your NPZ data file."
                )

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            if verbose:
                print(f"Loading data from: {data_path}")

            # Load the NPZ file
            try:
                aligned_data = dict(np.load(data_path, allow_pickle=True))
            except Exception as e:
                raise ValueError(f"Failed to load NPZ file: {e}")

            # Check for required neural data key
            if not any(k in aligned_data for k in NEURAL_DATA_ALIASES):
                raise ValueError(
                    f"NPZ file must contain neural data under one of: {sorted(NEURAL_DATA_ALIASES)}"
                )

            # Create experiment using the existing function
            Exp = load_exp_from_aligned_data(
                data_source,
                exp_params,
                aligned_data,
                force_continuous=force_continuous,
                feature_types=feature_types,
                static_features=static_features,
                verbose=verbose,
                bad_frames=bad_frames,
                reconstruct_spikes=reconstruct_spikes,
            )

            # Save to pickle if requested
            if save_to_pickle:
                # Create experiment name and path if not provided
                if exp_path is None:
                    expname = construct_session_name(data_source, exp_params)
                    # Create a reasonable default path
                    exp_dir = os.path.join(root, data_source, expname)
                    os.makedirs(exp_dir, exist_ok=True)
                    exp_path = os.path.join(exp_dir, f"Exp {expname}.pickle")

                save_exp_to_pickle(Exp, exp_path, verbose=verbose)

            # No load_log for external data sources
            return Exp, None


def save_exp_to_pickle(exp, path, verbose=True):
    """Save an Experiment object to a pickle file.

    Parameters
    ----------
    exp : Experiment
        The Experiment object to save.
    path : str
        File path where the pickle will be saved.
    verbose : bool, default=True
        Whether to print save confirmation.

    Raises
    ------
    PermissionError
        If no write permission for the path.
    OSError
        If path is invalid or other OS-related errors.

    Examples
    --------
    >>> # Create a test experiment
    >>> import tempfile
    >>> import os
    >>> from driada.experiment import load_demo_experiment
    >>> exp = load_demo_experiment(verbose=False)
    >>>
    >>> # Save experiment to temporary file
    >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
    ...     temp_path = f.name
    >>> save_exp_to_pickle(exp, temp_path)  # doctest: +ELLIPSIS
    Experiment Exp demo saved to ...

    >>> # Save without verbose output
    >>> save_exp_to_pickle(exp, temp_path, verbose=False)
    >>>
    >>> # Cleanup
    >>> os.unlink(temp_path)

    Notes
    -----
    Uses Python's pickle module with default protocol.
    Creates parent directories if they don't exist."""
    # Create parent directories if they don't exist
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(exp, f)
        if verbose:
            print(f"Experiment {exp.signature} saved to {path}\n")


def load_exp_from_pickle(path, verbose=True):
    """Load an Experiment object from a pickle file.

    Parameters
    ----------
    path : str
        Path to the pickle file.
    verbose : bool, default=True
        Whether to print load confirmation.

    Returns
    -------
    Experiment
        The loaded Experiment object.

    Raises
    ------
    FileNotFoundError
        If the pickle file doesn't exist.
    PermissionError
        If no read permission for the file.
    OSError
        If path is invalid or other OS-related errors.

    Examples
    --------
    >>> # Create and save a test experiment first
    >>> import tempfile
    >>> from driada.experiment import load_demo_experiment
    >>> test_exp = load_demo_experiment(verbose=False)
    >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
    ...     temp_path = f.name
    >>> save_exp_to_pickle(test_exp, temp_path, verbose=False)
    >>>
    >>> # Load experiment from file
    >>> exp = load_exp_from_pickle(temp_path)  # doctest: +ELLIPSIS
    Experiment Exp demo loaded from ...

    >>> # Load without verbose output
    >>> exp = load_exp_from_pickle(temp_path, verbose=False)
    >>> exp.signature
    'Exp demo'
    >>>
    >>> # Cleanup
    >>> import os
    >>> os.unlink(temp_path)

    Notes
    -----
    Uses Python's pickle module for deserialization.
    Prints experiment signature upon successful load if verbose=True."""
    with open(path, "rb") as f:
        exp = pickle.load(
            f,
        )
        if verbose:
            print(f"Experiment {exp.signature} loaded from {path}\n")

    # Backward compatibility: Assign names to unnamed dynamic features
    from ..information.info_base import MultiTimeSeries
    if hasattr(exp, 'dynamic_features'):
        for feat_id, feat_obj in exp.dynamic_features.items():
            if hasattr(feat_obj, 'name') and (feat_obj.name is None or feat_obj.name == ''):
                feat_obj.name = str(feat_id)  # Use feature key as name

                # For MultiTimeSeries, also name components if they lack names
                if isinstance(feat_obj, MultiTimeSeries):
                    if hasattr(feat_obj, 'ts_list'):
                        for i, component in enumerate(feat_obj.ts_list):
                            if not hasattr(component, 'name') or component.name is None or component.name == '':
                                component.name = f"{feat_id}_{i}"

    # Backward compatibility: Assign names to neurons if missing
    if hasattr(exp, 'neurons'):
        for i, neuron in enumerate(exp.neurons):
            if hasattr(neuron, 'ca') and neuron.ca is not None:
                if not hasattr(neuron.ca, 'name') or neuron.ca.name is None or neuron.ca.name == '':
                    neuron.ca.name = f"neuron_{i}_ca"
            if hasattr(neuron, 'sp') and neuron.sp is not None:
                if not hasattr(neuron.sp, 'name') or neuron.sp.name is None or neuron.sp.name == '':
                    neuron.sp.name = f"neuron_{i}_sp"
            if hasattr(neuron, 'asp') and neuron.asp is not None:
                if not hasattr(neuron.asp, 'name') or neuron.asp.name is None or neuron.asp.name == '':
                    neuron.asp.name = f"neuron_{i}_asp"

    return exp


def load_demo_experiment(name="demo", verbose=False):
    """Load a demonstration experiment for documentation and testing.

    This is a convenience function for loading sample data in documentation
    examples and tests. It loads a synthetically generated calcium imaging dataset
    with behavioral data.

    Parameters
    ----------
    name : str, default='demo'
        Name identifier for the demo experiment. This becomes part of the
        experiment's signature. Common values:
        - 'demo': Basic demonstration
        - 'test': For unit tests
        - Any descriptive name for specific examples
    verbose : bool, default=False
        Whether to print loading messages.

    Returns
    -------
    Experiment
        A loaded Experiment object with:
        - 50 neurons
        - 10000 time points
        - Sample behavioral features (position, speed, etc.)
        - No spike reconstruction (for speed)

    Examples
    --------
    >>> from driada.experiment import load_demo_experiment
    >>>
    >>> # Basic usage
    >>> exp = load_demo_experiment()
    >>> print(f"Loaded {exp.n_cells} neurons, {exp.n_frames} frames")
    Loaded 50 neurons, 10000 frames
    >>>
    >>> # With custom name
    >>> exp = load_demo_experiment('pca_analysis')
    >>> print(exp.signature)
    Exp pca_analysis
    >>>
    >>> # Access data
    >>> calcium_data = exp.calcium.data  # (50, 10000) array
    >>> position = exp.position  # MultiTimeSeries with x,y coordinates

    Notes
    -----
    The demo data is located at 'examples/example_data/sample_recording.npz'
    relative to the DRIADA installation directory.

    See Also
    --------
    load_experiment : Full experiment loading with all options
    ~driada.experiment.synthetic.generators.generate_synthetic_exp : Generate synthetic data with custom properties
    """
    exp, _ = load_experiment(
        "MyLab",
        {"name": name},
        data_path="examples/example_data/sample_recording.npz",
        reconstruct_spikes=False,
        verbose=verbose,
        save_to_pickle=False,
    )
    return exp
