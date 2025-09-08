import copy
import os
import os.path
import numpy as np
import pickle

from .exp_base import Experiment
from ..information.info_base import TimeSeries, MultiTimeSeries
from ..utils.naming import construct_session_name
from ..utils.output import show_output
from .neuron import DEFAULT_FPS, DEFAULT_T_OFF, DEFAULT_T_RISE
from ..gdrive.download import download_gdrive_data, initialize_iabs_router


def load_exp_from_aligned_data(
    data_source,
    exp_params,
    data,
    force_continuous=[],
    bad_frames=[],
    static_features=None,
    verbose=True,
    reconstruct_spikes="wavelet",
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
        - Other keys: behavioral variables as 1D or 2D arrays
          * 1D arrays (time,): treated as single time series
          * 2D arrays (components, time): treated as MultiTimeSeries
    force_continuous : list, optional
        List of feature names to force as continuous variables.
        By default, features are automatically classified based on their values.
        Use this when discrete-looking data should be treated as continuous
        (e.g., binned position data).
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
    reconstruct_spikes : str or bool, default='wavelet'
        Method for spike reconstruction if spikes not provided.
        Options: 
        - 'wavelet': wavelet-based detection (recommended)
        - False: no reconstruction
        - Custom method name if implemented
        
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

    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dictionary, got {type(data).__name__}")
    if not data:
        raise ValueError("data dictionary cannot be empty")
    if not isinstance(exp_params, dict):
        raise TypeError(
            f"exp_params must be a dictionary, got {type(exp_params).__name__}"
        )

    expname = construct_session_name(data_source, exp_params)
    adata = copy.deepcopy(data)
    key_mapping = {key.lower(): key for key in adata.keys()}

    if verbose:
        print(f"Building experiment {expname}...")

    if "calcium" in key_mapping:
        calcium = adata.pop(key_mapping["calcium"])
    else:
        raise ValueError("No calcium data found!")

    spikes = None
    if "spikes" in key_mapping:
        spikes = adata.pop(key_mapping["spikes"])

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
        Used to filter out uninformative features from dynamic data.        """
        # Convert to numpy array for consistent handling
        arr = np.asarray(vals)
        
        # Check if empty
        if arr.size == 0:
            return True
            
        # Check if all NaN or all same value (ignoring NaN)
        nan_mask = np.isnan(arr)
        return np.all(nan_mask) or (len(np.unique(arr[~nan_mask])) <= 1)

    # Process dynamic features, handling multidimensional arrays
    filt_dyn_features = {}
    feat_is_continuous = {f: f in force_continuous for f in dyn_features.keys()} if force_continuous else {}
    
    for f, vals in dyn_features.items():
        # Convert to numpy array to check dimensions
        vals_array = np.asarray(vals)
        
        # Skip scalar values with warning
        if vals_array.ndim == 0:
            if verbose:
                print(f"Warning: Ignoring scalar value '{f}' found in NPZ file. "
                      f"Scalar values should be provided via static_features parameter.")
            continue
        
        # Skip non-numeric features with warning
        if vals_array.dtype.kind in ['U', 'S', 'O']:  # Unicode, bytes, or object
            if verbose:
                print(f"Warning: Ignoring non-numeric feature '{f}' with dtype {vals_array.dtype}. "
                      f"Only numeric features are supported.")
            continue
            
        if is_garbage(vals):
            continue
        
        # Handle based on dimensionality
        if vals_array.ndim == 1:
            # 1D -> TimeSeries
            if f in force_continuous:
                # User explicitly wants this to be continuous
                filt_dyn_features[f] = TimeSeries(vals_array, discrete=False)
            else:
                # Let TimeSeries auto-detect the type
                filt_dyn_features[f] = TimeSeries(vals_array)
            
        elif vals_array.ndim == 2:
            # 2D -> MultiTimeSeries (each row is a component)
            # This matches Experiment.__init__ behavior
            ts_list = [
                TimeSeries(vals_array[i, :], discrete=False)
                for i in range(vals_array.shape[0])
            ]
            filt_dyn_features[f] = MultiTimeSeries(ts_list)
            
        else:
            # Skip features with unsupported dimensions
            if verbose:
                print(f"Warning: Skipping feature '{f}' with unsupported {vals_array.ndim}D shape")

    if verbose:
        print("behaviour variables:")
        print()
        for f, ts in filt_dyn_features.items():
            if isinstance(ts, MultiTimeSeries):
                dtype = "discrete" if ts.discrete else "continuous"
                print(f"'{f}' {dtype} multi-dimensional ({ts.n_dim}D)")
            else:
                print(f"'{f}'", "discrete" if ts.discrete else "continuous")

    # check for constant features
    constfeats = set(dyn_features.keys()) - set(filt_dyn_features.keys())

    if len(constfeats) != 0 and verbose:
        print(f"features {constfeats} dropped as constant or empty")

    auto_continuous = [fn for fn, ts in filt_dyn_features.items() if not ts.discrete]
    if verbose:
        print(f"features {auto_continuous} automatically determined as continuous")
        print()

    if len(force_continuous) != 0:
        # Check if auto-determined continuous features match force_continuous
        force_continuous_in_data = set(force_continuous) & set(dyn_features.keys())
        if set(auto_continuous) != force_continuous_in_data:
            if verbose:
                print(
                    "Warning: auto determined continuous features do not coincide with force_continuous list! Automatic labelling will be overridden"
                )
            # Re-create time series with corrected discrete/continuous labels
            for fn in filt_dyn_features.keys():
                if fn in force_continuous:
                    # Force this feature to be continuous
                    if filt_dyn_features[fn].discrete:
                        filt_dyn_features[fn] = TimeSeries(
                            filt_dyn_features[fn].data, discrete=False
                        )
                        if verbose:
                            print(f"Feature '{fn}' forced to be continuous")
                else:
                    # Not in force_continuous - let auto-detection stand
                    # This preserves multi-valued discrete features
                    pass

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
        Feature names to force as continuous. See load_exp_from_aligned_data.
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
    >>> os.unlink(temp_data)    """

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
                data_path = os.path.join(
                    root, expname, "Aligned data", f"{expname} syn data.npz"
                )
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
                data_router, data_pieces = initialize_iabs_router(root=root, router_source=router_source)
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
                    raise FileNotFoundError(
                        f"Cannot download {expname}, see loading log above"
                    )

            else:
                load_log = None

            aligned_data = dict(np.load(data_path))
            Exp = load_exp_from_aligned_data(
                data_source,
                exp_params,
                aligned_data,
                force_continuous=force_continuous,
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
            
            # Check for required 'calcium' key
            if 'calcium' not in aligned_data:
                raise ValueError("NPZ file must contain 'calcium' key with neural data")
            
            # Create experiment using the existing function
            Exp = load_exp_from_aligned_data(
                data_source,
                exp_params,
                aligned_data,
                force_continuous=force_continuous,
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
    Creates parent directories if they don't exist.    """
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
    Prints experiment signature upon successful load if verbose=True.    """
    with open(path, "rb") as f:
        exp = pickle.load(
            f,
        )
        if verbose:
            print(f"Experiment {exp.signature} loaded from {path}\n")
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
    ~driada.experiment.exp_build.load_experiment : Full experiment loading with all options
    ~driada.experiment.synthetic.experiment_generators.generate_synthetic_exp : Generate synthetic data with custom properties
    """
    exp, _ = load_experiment(
        'MyLab',
        {'name': name},
        data_path='examples/example_data/sample_recording.npz',
        reconstruct_spikes=False,
        verbose=verbose,
        save_to_pickle=False
    )
    return exp
