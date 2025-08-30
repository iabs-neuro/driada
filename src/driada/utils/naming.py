def construct_session_name(data_source, exp_params, allow_unknown=True):
    """Construct standardized session name from experimental parameters.
    
    Creates a consistent naming convention for experimental sessions based on
    the data source and experimental parameters. Currently supports IABS
    (Institute for Advanced Brain Studies) data source.
    
    Parameters
    ----------
    data_source : str
        The data source identifier. Currently only "IABS" is supported.
    exp_params : dict
        Dictionary containing experimental parameters. For IABS, must include:
        - 'track': The experimental track/paradigm name
        - 'animal_id': The animal identifier
        - 'session': The session identifier
    allow_unknown : bool, optional
        Whether to allow unknown track names. If False, raises ValueError
        for unrecognized tracks. Default is True.
        
    Returns
    -------
    str
        Standardized session name following the pattern appropriate for
        the data source and track.
        
    Raises
    ------
    ValueError
        If allow_unknown is False and an unknown track is encountered.
    NotImplementedError
        If a data source other than "IABS" is specified.
        
    Examples
    --------
    >>> params = {'track': 'STFP', 'animal_id': 'M123', 'session': '1'}
    >>> construct_session_name('IABS', params)
    'STFP_M123_1'
    
    >>> # Old track with special naming
    >>> params = {'track': 'HT', 'animal_id': 'A5', 'session': '3'}
    >>> construct_session_name('IABS', params)
    'A5_HT3'
    
    Notes
    -----
    Different tracks have different naming conventions:
    - Old tracks (HT, RT, FS): Use legacy naming patterns
    - Standard tracks (FcOY, STFP, AP, NOF, Trace, CC): Use {track}_{animal}_{session}
    - Unknown tracks: Use standard pattern if allow_unknown=True
    
    DOC_VERIFIED
    """
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
        raise NotImplementedError("Other data sources are not yet supported")
