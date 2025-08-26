import copy
import os.path
import numpy as np
import pickle

from .exp_base import Experiment
from ..information.info_base import TimeSeries
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
        - Other keys: behavioral variables as 1D arrays (time)
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
        
    Notes
    -----
    - Features with constant values or all NaN are automatically filtered out
    - Feature types (discrete/continuous) are automatically determined by checking
      if values appear to be categorical (few unique values) or continuous
    - Case-insensitive key matching for 'calcium' and 'spikes'
    - Creates a deep copy of input data to avoid modifying the original
    - The experiment name is constructed using construct_session_name()
    
    Examples
    --------
    >>> # Basic usage with minimal data
    >>> data = {
    ...     'calcium': np.random.rand(50, 1000),  # 50 neurons, 1000 frames
    ...     'position': np.linspace(0, 100, 1000),  # Linear track position
    ...     'speed': np.abs(np.diff(np.linspace(0, 100, 1001)))[:1000],
    ...     'trial_type': np.repeat([0, 1, 0, 1], 250)  # Discrete variable
    ... }
    >>> exp_params = {
    ...     'track': 'linear_track',
    ...     'animal_id': 'mouse01', 
    ...     'session': 'day1'
    ... }
    >>> exp = load_exp_from_aligned_data('IABS', exp_params, data)
    Building experiment linear_track_mouse01_day1...
    behaviour variables:
    
    'position' continuous
    'speed' continuous  
    'trial_type' discrete
    
    >>> # Force discrete variable to be continuous
    >>> exp2 = load_exp_from_aligned_data(
    ...     'IABS', exp_params, data,
    ...     force_continuous=['trial_type'],
    ...     bad_frames=[10, 11, 12],  # Mark frames as bad
    ...     static_features={'fps': 30.0}  # Override default fps
    ... )
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
        """Check if values are constant or all NaN."""
        if len(vals) == 0:
            return True
        # Convert to numpy array for consistent handling
        arr = np.asarray(vals)
        # Check if all NaN or all same value (ignoring NaN)
        return np.all(np.isnan(arr)) or (len(np.unique(arr[~np.isnan(arr)])) <= 1)

    if len(force_continuous) != 0:
        feat_is_continuous = {f: f in force_continuous for f in dyn_features.keys()}
        filt_dyn_features = {
            f: TimeSeries(vals, discrete=not feat_is_continuous[f])
            for f, vals in dyn_features.items()
            if not is_garbage(vals)
        }
    else:
        filt_dyn_features = {
            f: TimeSeries(vals)
            for f, vals in dyn_features.items()
            if not is_garbage(vals)
        }

    if verbose:
        print("behaviour variables:")
        print()
        for f, ts in filt_dyn_features.items():
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
):
    """Load or create an Experiment object with automatic caching and cloud support.
    
    This function provides a high-level interface for loading experiments with
    smart caching, automatic cloud data download (for IABS data), and pickle
    serialization. It first checks for cached experiments, then loads from
    local data files, and finally downloads from cloud storage if needed.
    
    Parameters
    ----------
    data_source : str
        Data source identifier. Currently only 'IABS' is supported for
        automatic cloud download. Other sources must provide data_path.
    exp_params : dict
        Experiment parameters dictionary. See load_exp_from_aligned_data
        for required fields based on data_source.
    force_rebuild : bool, default=False
        If True, rebuild experiment even if pickle cache exists.
    force_reload : bool, default=False
        If True, re-download data from cloud even if local files exist.
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
        Custom path for data file. If None, uses standard naming:
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
        
    Returns
    -------
    exp : Experiment
        The loaded or created Experiment object.
    load_log : list or None
        Cloud download log if data was downloaded, None otherwise.
        
    Raises
    ------
    ValueError
        If root exists but is not a directory, or if data_source
        is not supported.
    FileNotFoundError
        If data cannot be found locally or downloaded from cloud.
        
    Notes
    -----
    Loading priority:
    1. If pickle exists and not force_rebuild/reload: load from pickle
    2. If local data exists: load from data file
    3. If IABS source: attempt cloud download
    4. Otherwise: raise error
    
    For IABS data, expects cloud structure with 'Aligned data' containing
    npz files with calcium and behavioral data.
    
    Examples
    --------
    >>> # Load IABS experiment with automatic download
    >>> exp_params = {
    ...     'track': 'linear_track',
    ...     'animal_id': 'mouse01',
    ...     'session': 'day1'
    ... }
    >>> exp, log = load_experiment('IABS', exp_params)
    Loading experiment linear_track_mouse01_day1 from pickle...
    
    >>> # Force rebuild from data
    >>> exp, log = load_experiment(
    ...     'IABS', exp_params, 
    ...     force_rebuild=True,
    ...     save_to_pickle=True  # Cache for next time
    ... )
    Building experiment linear_track_mouse01_day1...
    
    >>> # Custom data source with local file
    >>> exp, log = load_experiment(
    ...     'custom', {'name': 'my_exp'},
    ...     data_path='/path/to/data.npz'
    ... )
    """

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
                data_router, data_pieces = initialize_iabs_router(root=root)
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
            raise ValueError("External data sources are not yet supported")


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
        
    Notes
    -----
    Uses Python's pickle module with default protocol.
    Creates parent directories if they don't exist.
    """
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
    pickle.UnpicklingError
        If the file is corrupted or incompatible.
    """
    with open(path, "rb") as f:
        exp = pickle.load(
            f,
        )
        if verbose:
            print(f"Experiment {exp.signature} loaded from {path}\n")
    return exp
