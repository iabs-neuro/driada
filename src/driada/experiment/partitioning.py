"""
Experiment partitioning utilities for DRIADA.

This module provides functionality to split Experiment objects into multiple
temporal segments for analysis of time-dependent properties.
"""

from typing import Optional, List, Tuple, Dict
import numpy as np

from .exp_base import Experiment
from ..information.info_base import TimeSeries, MultiTimeSeries


def split_experiment_session(
    experiment: Experiment,
    n_parts: Optional[int] = None,
    part_duration: Optional[int] = None,
    overlap: int = 0,
    suffix_format: str = 'part_{:03d}',
    return_full_experiments: bool = True,
    preserve_ground_truth: bool = True,
    verbose: bool = True
) -> List:
    """
    Split an Experiment session into multiple temporal parts.

    Creates multiple Experiment objects from a single parent experiment by
    dividing the time series data into segments. Useful for analyzing how
    metrics (e.g., MI, p-values) scale with recording duration.

    Parameters
    ----------
    experiment : Experiment
        The parent Experiment object to split.
    n_parts : int, optional
        Number of parts to split into. Must be specified if part_duration
        is not provided.
    part_duration : int, optional
        Duration of each part in frames. If specified, n_parts is calculated
        automatically.
    overlap : int, optional
        Number of overlapping frames between consecutive parts.
        Default is 0 (no overlap).
    suffix_format : str, optional
        Format string for naming parts. Default is 'part_{:03d}'.
    return_full_experiments : bool, optional
        If True, returns list of full Experiment objects.
        If False, returns dictionaries with sliced data only.
        Default is True.
    preserve_ground_truth : bool, optional
        If True, copies ground_truth metadata to parts (for synthetic data).
        Default is True.
    verbose : bool, optional
        Print progress messages. Default is True.

    Returns
    -------
    list
        List of Experiment objects (if return_full_experiments=True) or
        dictionaries with sliced data (if False).

    Raises
    ------
    ValueError
        If parameters are invalid or data is inconsistent.
    TypeError
        If experiment is not an Experiment object.

    Examples
    --------
    >>> # Split an experiment into 3 equal parts
    >>> parts = split_experiment_session(exp, n_parts=3)
    >>> len(parts)
    3

    >>> # Split into parts of 12000 frames each (10 min at 20 fps)
    >>> parts = split_experiment_session(exp, part_duration=12000)

    >>> # Split with overlap for sliding window analysis
    >>> parts = split_experiment_session(exp, n_parts=10, overlap=2400)
    """
    # Input validation
    if not isinstance(experiment, Experiment):
        raise TypeError(f"experiment must be an Experiment object, got {type(experiment)}")

    if n_parts is None and part_duration is None:
        raise ValueError("Either n_parts or part_duration must be specified")

    if n_parts is not None and n_parts <= 0:
        raise ValueError("n_parts must be positive")

    if part_duration is not None and part_duration <= 0:
        raise ValueError("part_duration must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if verbose:
        print(f"Splitting experiment '{experiment.signature}' with {experiment.n_frames} frames...")

    # Determine part boundaries
    n_frames = experiment.n_frames

    if part_duration is not None:
        # Calculate number of parts based on duration
        if overlap >= part_duration:
            raise ValueError("overlap must be less than part_duration")

        # Number of parts considering overlap
        step = part_duration - overlap
        n_parts = max(1, (n_frames - overlap) // step)
        if (n_frames - overlap) % step > 0:
            n_parts += 1

    # Create part boundaries
    boundaries = _calculate_split_boundaries(
        n_frames, n_parts, part_duration, overlap
    )

    # Process each part
    parts = []

    for i, (start, end) in enumerate(boundaries):
        if start >= end:  # Empty part
            continue

        if verbose:
            duration_sec = (end - start) / experiment.fps
            print(f"  Part {i}: frames {start}-{end} ({duration_sec:.1f}s)")

        # Extract data for this part
        part_data = _extract_experiment_part(experiment, start, end)

        if return_full_experiments:
            # Create new Experiment object
            part_exp = _create_part_experiment(
                experiment, part_data, i, suffix_format, verbose
            )

            # Preserve ground truth if available (for synthetic data)
            if preserve_ground_truth and hasattr(experiment, 'ground_truth'):
                part_exp.ground_truth = experiment.ground_truth.copy()

            parts.append(part_exp)
        else:
            # Return dictionary with data
            parts.append({
                'part_id': i,
                'start_frame': start,
                'end_frame': end,
                'duration_frames': end - start,
                'data': part_data,
                'original_signature': experiment.signature
            })

    if verbose:
        print(f"Successfully split into {len(parts)} parts")

    return parts


def _calculate_split_boundaries(
    n_frames: int,
    n_parts: int,
    part_duration: Optional[int],
    overlap: int
) -> List[Tuple[int, int]]:
    """
    Calculate frame boundaries for splitting.

    Parameters
    ----------
    n_frames : int
        Total number of frames.
    n_parts : int
        Number of parts to create.
    part_duration : int or None
        Duration of each part in frames.
    overlap : int
        Overlap between consecutive parts.

    Returns
    -------
    list of tuples
        List of (start_frame, end_frame) tuples.
    """
    boundaries = []

    if part_duration is not None:
        # Fixed duration parts with overlap
        step = part_duration - overlap
        start = 0

        while start < n_frames:
            end = min(start + part_duration, n_frames)
            boundaries.append((start, end))
            start += step

    else:
        # Equal-sized parts
        if overlap > 0:
            # With overlap, calculate effective part duration
            total_effective = n_frames + overlap * (n_parts - 1)
            part_duration = total_effective // n_parts
            step = part_duration - overlap
        else:
            # Without overlap, simple division
            part_length = n_frames // n_parts
            remainder = n_frames % n_parts
            step = part_length

        start = 0
        for i in range(n_parts):
            if overlap > 0:
                end = min(start + part_duration, n_frames)
            else:
                # Distribute remainder across first parts
                length = part_length + (1 if i < remainder else 0)
                end = min(start + length, n_frames)

            if end > start:
                boundaries.append((start, end))

            if overlap > 0:
                start += step
            else:
                start = end

            if start >= n_frames:
                break

    return boundaries


def _extract_experiment_part(
    experiment: Experiment,
    start_frame: int,
    end_frame: int
) -> Dict:
    """
    Extract data for a specific part of an experiment.

    Parameters
    ----------
    experiment : Experiment
        The parent experiment.
    start_frame : int
        Start frame index (inclusive).
    end_frame : int
        End frame index (exclusive).

    Returns
    -------
    dict
        Dictionary containing sliced data.
    """
    # Slice calcium data (MultiTimeSeries)
    calcium_data = experiment.calcium.data  # 2D array (n_neurons, n_frames)
    calcium_part = calcium_data[:, start_frame:end_frame]

    # Slice spike data (if available)
    spikes_part = None
    if hasattr(experiment, 'spikes') and experiment.spikes is not None:
        if hasattr(experiment.spikes, 'data'):
            spikes_part = experiment.spikes.data[:, start_frame:end_frame]

    # Slice dynamic features
    dynamic_features_part = {}
    for feat_name, feat_obj in experiment.dynamic_features.items():
        if hasattr(feat_obj, 'ts_list'):
            # MultiTimeSeries - slice each component
            sub_components = []
            for ts in feat_obj.ts_list:
                sub_data = ts.data[start_frame:end_frame]
                sub_components.append(
                    TimeSeries(data=sub_data, discrete=ts.discrete, name=ts.name)
                )
            dynamic_features_part[feat_name] = MultiTimeSeries(
                sub_components,
                name=feat_obj.name,
                allow_zero_columns=getattr(feat_obj, 'allow_zero_columns', True)
            )
        elif hasattr(feat_obj, 'data'):
            # TimeSeries - slice directly
            sub_data = feat_obj.data[start_frame:end_frame]
            dynamic_features_part[feat_name] = TimeSeries(
                data=sub_data, discrete=feat_obj.discrete, name=feat_obj.name
            )
        elif isinstance(feat_obj, np.ndarray):
            # Raw numpy array (for compatibility)
            if feat_obj.ndim == 1:
                dynamic_features_part[feat_name] = feat_obj[start_frame:end_frame]
            else:
                dynamic_features_part[feat_name] = feat_obj[:, start_frame:end_frame]
        else:
            # Unknown type - try indexing
            try:
                dynamic_features_part[feat_name] = feat_obj[start_frame:end_frame]
            except Exception as e:
                raise TypeError(
                    f"Cannot slice feature '{feat_name}' of type {type(feat_obj)}: {e}"
                )

    # Static features (copy all)
    static_features_part = experiment.static_features.copy()

    # Experiment identifiers
    exp_identificators_part = experiment.exp_identificators.copy()

    return {
        'calcium': calcium_part,
        'spikes': spikes_part,
        'dynamic_features': dynamic_features_part,
        'static_features': static_features_part,
        'exp_identificators': exp_identificators_part,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'original_signature': experiment.signature
    }


def _create_part_experiment(
    parent_experiment: Experiment,
    part_data: Dict,
    part_index: int,
    suffix_format: str,
    verbose: bool
) -> Experiment:
    """
    Create a new Experiment object for a part.

    Parameters
    ----------
    parent_experiment : Experiment
        The parent experiment.
    part_data : dict
        Data extracted for the part.
    part_index : int
        Index of the part.
    suffix_format : str
        Format string for the signature suffix.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    Experiment
        New Experiment object for the part.
    """
    # Create new signature
    new_signature = f"{parent_experiment.signature}_{suffix_format.format(part_index)}"

    # Update identifiers
    part_data['exp_identificators'].update({
        'original_experiment': parent_experiment.signature,
        'part_index': part_index,
        'start_frame': part_data['start_frame'],
        'end_frame': part_data['end_frame'],
        'duration_frames': part_data['end_frame'] - part_data['start_frame'],
    })

    # Create new experiment
    part_exp = Experiment(
        signature=new_signature,
        calcium=part_data['calcium'],
        spikes=part_data['spikes'],
        exp_identificators=part_data['exp_identificators'],
        static_features=part_data['static_features'],
        dynamic_features=part_data['dynamic_features'],
        reconstruct_spikes=False,  # Disable spike reconstruction for speed
        verbose=False,  # Suppress verbose output for parts
    )

    # Copy spike reconstruction metadata if present
    if hasattr(parent_experiment, 'spike_reconstruction_method'):
        part_exp.spike_reconstruction_method = parent_experiment.spike_reconstruction_method

    if hasattr(parent_experiment, '_reconstruction_metadata'):
        part_exp._reconstruction_metadata = parent_experiment._reconstruction_metadata.copy()

    return part_exp


def get_duration_in_minutes(experiment: Experiment) -> float:
    """
    Get experiment duration in minutes.

    Parameters
    ----------
    experiment : Experiment
        Experiment object.

    Returns
    -------
    float
        Duration in minutes.
    """
    return experiment.n_frames / (experiment.fps * 60)


def split_by_duration_minutes(
    experiment: Experiment,
    target_durations_min: List[int],
    method: str = 'first_n',
    **kwargs
) -> Dict[int, Experiment]:
    """
    Split experiment into parts with specific durations in minutes.

    Convenience function for creating subsampled experiments at different
    durations for scaling analysis.

    Parameters
    ----------
    experiment : Experiment
        Full experiment to split.
    target_durations_min : list of int
        Target durations in minutes (e.g., [10, 20, 30, 60]).
    method : str, optional
        Method for creating parts:
        - 'first_n': Take first N minutes (default)
        - 'random': Random contiguous segment (not yet implemented)
    **kwargs
        Additional arguments passed to split_experiment_session.

    Returns
    -------
    dict
        Dictionary mapping duration (minutes) to Experiment object.

    Examples
    --------
    >>> # Create 10-min and 30-min subsampled experiments
    >>> parts = split_by_duration_minutes(exp_60min, [10, 30])
    >>> parts[10].n_frames  # 12000 frames at 20 fps
    12000
    """
    fps = experiment.fps
    full_duration_min = get_duration_in_minutes(experiment)

    # Validate durations
    for duration in target_durations_min:
        if duration > full_duration_min:
            raise ValueError(
                f"Requested duration {duration} min exceeds full experiment "
                f"duration {full_duration_min:.1f} min"
            )

    if method != 'first_n':
        raise NotImplementedError(f"Method '{method}' not yet implemented")

    # Create parts
    parts_dict = {}

    for duration_min in sorted(target_durations_min):
        n_frames = int(duration_min * 60 * fps)

        # Use split_experiment_session with part_duration
        parts = split_experiment_session(
            experiment,
            n_parts=1,
            part_duration=n_frames,
            overlap=0,
            suffix_format=f'{duration_min}min',
            **kwargs
        )

        if parts:
            parts_dict[duration_min] = parts[0]

    return parts_dict
