def construct_session_name(data_source, exp_params, allow_unknown=True):
    """Construct standardized session name from experimental parameters.
    
    Creates a consistent naming convention for experimental sessions based on
    the data source and experimental parameters. Supports IABS (Institute for
    Advanced Brain Studies) and generic lab data sources.
    
    Parameters
    ----------
    data_source : str
        The data source identifier (e.g., 'IABS', 'MyLab', 'NeuroLab').
    exp_params : dict
        Dictionary containing experimental parameters.
        
        For IABS, must include:
        - 'track': The experimental track/paradigm name
        - 'animal_id': The animal identifier
        - 'session': The session identifier
        
        For other sources, can include:
        - 'name': Explicit name (takes priority if present)
        - 'experiment': Experiment type or name
        - 'subject' or 'animal_id': Subject identifier
        - 'session': Session identifier
        - 'date': Recording date
        
    allow_unknown : bool, optional
        Whether to allow unknown track names (IABS only). If False, raises 
        ValueError for unrecognized tracks. Default is True.
        
    Returns
    -------
    str
        Standardized session name following the pattern appropriate for
        the data source and parameters provided.
        
    Raises
    ------
    ValueError
        If allow_unknown is False and an unknown track is encountered (IABS only).
        
    Examples
    --------
    >>> # IABS standard track
    >>> params = {'track': 'STFP', 'animal_id': 'M123', 'session': '1'}
    >>> construct_session_name('IABS', params)
    'STFP_M123_1'
    
    >>> # IABS old track with special naming
    >>> params = {'track': 'HT', 'animal_id': 'A5', 'session': '3'}
    >>> construct_session_name('IABS', params)
    'A5_HT3'
    
    >>> # Generic lab with explicit name
    >>> construct_session_name('MyLab', {'name': 'pilot_study_001'})
    'pilot_study_001'
    
    >>> # Generic lab with standard parameters
    >>> params = {'experiment': 'maze', 'subject': 'rat42', 'session': 'day3'}
    >>> construct_session_name('NeuroLab', params)
    'maze_rat42_day3'
    
    >>> # Generic lab with minimal parameters
    >>> construct_session_name('Lab1', {'subject': 'mouse5'})
    'mouse5'
    
    >>> # Generic lab with no standard parameters (uses timestamp)
    >>> result = construct_session_name('GenericLab', {})
    >>> result.startswith('GenericLab_') and len(result) == 26  # LabName_YYYYMMDD_HHMMSS
    True
    
    Notes
    -----
    For IABS data source:
    - Old tracks (HT, RT, FS): Use legacy naming patterns
    - Standard tracks (FcOY, STFP, AP, NOF, Trace, CC): Use {track}_{animal}_{session}
    - Unknown tracks: Use standard pattern if allow_unknown=True
    
    For other data sources:
    - If 'name' parameter exists, it's used directly
    - Otherwise combines available standard parameters in order:
      experiment, subject/animal_id, session, date
    - If no standard parameters exist, uses data_source name + timestamp    """
    if data_source == "IABS":
        track = exp_params["track"]
        animal_id, session = exp_params["animal_id"], exp_params["session"]

        # Old tracks with non-standard naming patterns
        old_track_patterns = {
            "HT": f"{animal_id}_HT{session}",
            "RT": f"RT_{animal_id}_{session}D",
            "FS": f"FS{animal_id}_{session}D",
        }

        # Check if it's an old track with special pattern
        if track in old_track_patterns:
            return old_track_patterns[track]

        # For newer tracks, use standard pattern: {track}_{animal_id}_{session}
        standard_tracks = {"FcOY", "STFP", "AP", "NOF", "Trace", "CC"}

        if track in standard_tracks:
            return f"{track}_{animal_id}_{session}"

        # Unknown track handling
        if not allow_unknown:
            raise ValueError(f"Unknown track: {track}!")
        else:
            return f"{track}_{animal_id}_{session}"

    else:
        # Generic naming for non-IABS data sources
        # First check if user provided explicit name
        if 'name' in exp_params:
            return exp_params['name']
        
        # Build name from common parameter patterns
        name_parts = []
        
        # Try common parameter names in order of preference
        for key in ['experiment', 'subject', 'animal_id', 'session', 'date']:
            if key in exp_params:
                name_parts.append(str(exp_params[key]))
        
        if name_parts:
            # Join with underscore, limiting length for practicality
            return '_'.join(name_parts[:4])  # Max 4 parts to avoid overly long names
        else:
            # Fallback: use data source with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"{data_source}_{timestamp}"
