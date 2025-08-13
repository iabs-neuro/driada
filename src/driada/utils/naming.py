def construct_session_name(data_source, exp_params, allow_unknown=True):
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
