#!/usr/bin/env python
"""
Load synchronized experiment .npz files and parse metadata to Experiment objects.

This script reads multiple .npz files from DRIADA data directories, parses the
IABS naming convention to extract experiment metadata (track, animal_id, session),
and creates properly configured Experiment objects.

IABS Naming Pattern
-------------------
The IABS (Institute for Advanced Brain Studies) naming convention follows:

    {track}_{animal_id}_{session}[_suffix].npz

Where:
- track: Experiment paradigm (NOF, LNOF, STFP, HT, RT, FS, etc.)
- animal_id: Subject identifier (letter prefix + number, e.g., H01, J05, M123)
- session: Recording session (typically number + D for day, e.g., 1D, 2D, 4D)
- suffix: Optional suffix like "aligned", "syn data", etc.

Examples:
- NOF_H01_1D syn data.npz -> track=NOF, animal_id=H01, session=1D
- LNOF_J01_4D_aligned.npz -> track=LNOF, animal_id=J01, session=4D

Metadata Sources
----------------
1. Filename parsing (always available):
   - track, animal_id, session

2. _metadata dict (new format files only):
   - session_name: Full session identifier
   - fps: Frame rate
   - export_timestamp: When data was exported
   - feedback_applied: Whether feedback was applied during CNMF
   - cnmf_params: CNMF extraction parameters
   - autoinspection_config: Auto-inspection settings
   - autoinspection_stats: Quality metrics (n_total, n_good, n_bad, etc.)
   - metrics_df: Per-component quality metrics

3. _sync_info dict (new format files only):
   - synchronizer_version: Version of synchronizer used
   - sync_timestamp: When synchronization was performed
   - experiment_name: Name from synchronizer
   - source_format: Data source format
   - alignment_mode: How alignment was performed
   - n_features: Number of behavioral features
   - validation_passed: Whether validation checks passed
   - n_timepoints: Number of aligned time points

Usage
-----
    python tools/load_synchronized_experiments.py

Or as a module:
    from tools.load_synchronized_experiments import (
        load_all_experiments,
        parse_iabs_filename,
        get_available_metadata
    )

    experiments = load_all_experiments()
    for exp in experiments:
        print(exp.signature, exp.animal_id, exp.session)
"""

import re
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ParsedFilename:
    """Parsed components from an IABS-style filename."""
    track: str
    animal_id: str
    session: str
    suffix: Optional[str] = None

    @property
    def exp_params(self) -> dict:
        """Return exp_params dict for use with load_exp_from_aligned_data."""
        return {
            'track': self.track,
            'animal_id': self.animal_id,
            'session': self.session,
        }


def parse_iabs_filename(filename: str) -> Optional[ParsedFilename]:
    """
    Parse an IABS-style filename to extract experiment metadata.

    Parameters
    ----------
    filename : str
        Filename (with or without path) to parse.

    Returns
    -------
    ParsedFilename or None
        Parsed components if successful, None if parsing failed.

    Examples
    --------
    >>> parse_iabs_filename('NOF_H01_1D syn data.npz')
    ParsedFilename(track='NOF', animal_id='H01', session='1D', suffix='syn data')

    >>> parse_iabs_filename('LNOF_J01_4D_aligned.npz')
    ParsedFilename(track='LNOF', animal_id='J01', session='4D', suffix='aligned')
    """
    # Get just the filename without path
    name = Path(filename).stem

    # Pattern: {track}_{animal_id}_{session}[_suffix or suffix]
    # track: letters (NOF, LNOF, STFP, HT, etc.)
    # animal_id: letter(s) + digits (H01, J05, M123, A5)
    # session: digits + optional letter (1D, 2D, 1, 2, etc.)

    # Try standard pattern: TRACK_ANIMAL_SESSION[_SUFFIX]
    pattern = r'^([A-Z]+)_([A-Z]\d+)_(\d+[A-Z]?)(?:[ _](.+))?$'
    match = re.match(pattern, name, re.IGNORECASE)

    if match:
        return ParsedFilename(
            track=match.group(1).upper(),
            animal_id=match.group(2).upper(),
            session=match.group(3).upper(),
            suffix=match.group(4) if match.group(4) else None
        )

    # Try alternative patterns for old naming conventions
    # Pattern for HT tracks: {animal_id}_HT{session}
    pattern_ht = r'^([A-Z]\d+)_HT(\d+)$'
    match = re.match(pattern_ht, name, re.IGNORECASE)
    if match:
        return ParsedFilename(
            track='HT',
            animal_id=match.group(1).upper(),
            session=match.group(2),
            suffix=None
        )

    # Pattern for RT tracks: RT_{animal_id}_{session}D
    pattern_rt = r'^RT_([A-Z]\d+)_(\d+)D$'
    match = re.match(pattern_rt, name, re.IGNORECASE)
    if match:
        return ParsedFilename(
            track='RT',
            animal_id=match.group(1).upper(),
            session=f'{match.group(2)}D',
            suffix=None
        )

    # Pattern for FS tracks: FS{animal_id}_{session}D
    pattern_fs = r'^FS([A-Z]\d+)_(\d+)D$'
    match = re.match(pattern_fs, name, re.IGNORECASE)
    if match:
        return ParsedFilename(
            track='FS',
            animal_id=match.group(1).upper(),
            session=f'{match.group(2)}D',
            suffix=None
        )

    return None


def get_npz_metadata(npz_path: Path) -> dict:
    """
    Extract metadata from an .npz file.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file.

    Returns
    -------
    dict
        Dictionary containing:
        - 'has_new_format': bool - whether file has _metadata/_sync_info
        - 'metadata': dict or None - contents of _metadata if present
        - 'sync_info': dict or None - contents of _sync_info if present
        - 'calcium_key': str - key used for calcium data ('Calcium' or 'calcium')
        - 'has_spikes': bool - whether spike data is present
        - 'has_reconstructions': bool - whether reconstruction data is present
        - 'n_neurons': int - number of neurons
        - 'n_frames': int - number of time frames
        - 'feature_names': list - names of behavioral features
        - 'fps': float or None - frame rate if available
    """
    data = np.load(npz_path, allow_pickle=True)

    result = {
        'has_new_format': '_metadata' in data.keys(),
        'metadata': None,
        'sync_info': None,
        'calcium_key': None,
        'has_spikes': False,
        'has_reconstructions': False,
        'n_neurons': 0,
        'n_frames': 0,
        'feature_names': [],
        'fps': None,
    }

    # Extract metadata dicts
    if '_metadata' in data.keys():
        result['metadata'] = data['_metadata'].item()
        if 'fps' in result['metadata']:
            result['fps'] = result['metadata']['fps']

    if '_sync_info' in data.keys():
        result['sync_info'] = data['_sync_info'].item()

    # Find calcium data
    for key in ['Calcium', 'calcium']:
        if key in data.keys():
            result['calcium_key'] = key
            calcium = data[key]
            result['n_neurons'] = calcium.shape[0]
            result['n_frames'] = calcium.shape[1]
            break

    # Check for spikes and reconstructions
    result['has_spikes'] = any(k.lower() == 'spikes' for k in data.keys())
    result['has_reconstructions'] = 'Reconstructions' in data.keys()

    # Get feature names (exclude special keys)
    special_keys = {
        'calcium', 'Calcium', 'spikes', 'Spikes',
        'Reconstructions', '_metadata', '_sync_info'
    }
    result['feature_names'] = [k for k in data.keys() if k not in special_keys]

    data.close()
    return result


def load_experiment_from_npz(
    npz_path: Path,
    parsed: Optional[ParsedFilename] = None,
    verbose: bool = True,
    reconstruct_spikes: bool = False,
) -> 'Experiment':
    """
    Load an Experiment object from an .npz file.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file.
    parsed : ParsedFilename, optional
        Pre-parsed filename metadata. If None, will parse from filename.
    verbose : bool
        Whether to print loading information.
    reconstruct_spikes : bool
        Whether to reconstruct spikes after loading.

    Returns
    -------
    Experiment
        Loaded experiment object.
    """
    from driada.experiment.exp_build import load_exp_from_aligned_data

    # Parse filename if not provided
    if parsed is None:
        parsed = parse_iabs_filename(npz_path.name)
        if parsed is None:
            raise ValueError(f"Could not parse filename: {npz_path.name}")

    # Load data
    data = dict(np.load(npz_path, allow_pickle=True))

    # Extract metadata to get fps
    metadata = get_npz_metadata(npz_path)

    # Build static features
    static_features = {}
    if metadata['fps'] is not None:
        static_features['fps'] = float(metadata['fps'])

    # Store original metadata in exp_params for later access
    exp_params = parsed.exp_params.copy()
    if metadata['metadata'] is not None:
        exp_params['_original_metadata'] = metadata['metadata']
    if metadata['sync_info'] is not None:
        exp_params['_sync_info'] = metadata['sync_info']

    # Remove metadata keys from data dict (they're not behavioral features)
    data.pop('_metadata', None)
    data.pop('_sync_info', None)

    # Create experiment
    exp = load_exp_from_aligned_data(
        data_source='IABS',
        exp_params=exp_params,
        data=data,
        static_features=static_features,
        verbose=verbose,
    )

    # Optionally reconstruct spikes
    if reconstruct_spikes:
        if verbose:
            print("Reconstructing spikes...")
        exp.reconstruct_all_neurons(method='wavelet', verbose=verbose)

    return exp


def find_npz_files(root_path: Optional[Path] = None) -> list[Path]:
    """
    Find all synchronized experiment .npz files.

    Parameters
    ----------
    root_path : Path, optional
        Root path to search. Defaults to driada project directory.

    Returns
    -------
    list[Path]
        List of paths to .npz files.
    """
    if root_path is None:
        root_path = Path(__file__).parent.parent

    # Search in known data directories
    data_dirs = [
        root_path / 'DRIADA data',
        root_path / 'science' / 'NOF data',
    ]

    npz_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            npz_files.extend(sorted(data_dir.glob('*.npz')))

    return npz_files


def load_all_experiments(
    npz_files: Optional[list[Path]] = None,
    verbose: bool = True,
    reconstruct_spikes: bool = False,
) -> list:
    """
    Load all experiments from .npz files.

    Parameters
    ----------
    npz_files : list[Path], optional
        List of .npz files to load. If None, finds all available files.
    verbose : bool
        Whether to print loading information.
    reconstruct_spikes : bool
        Whether to reconstruct spikes after loading.

    Returns
    -------
    list[Experiment]
        List of loaded Experiment objects.
    """
    if npz_files is None:
        npz_files = find_npz_files()

    experiments = []
    for npz_path in npz_files:
        try:
            parsed = parse_iabs_filename(npz_path.name)
            if parsed is None:
                if verbose:
                    print(f"Skipping {npz_path.name}: could not parse filename")
                continue

            if verbose:
                print(f"\nLoading {npz_path.name}...")
                print(f"  Track: {parsed.track}, Animal: {parsed.animal_id}, Session: {parsed.session}")

            exp = load_experiment_from_npz(
                npz_path,
                parsed=parsed,
                verbose=verbose,
                reconstruct_spikes=reconstruct_spikes,
            )
            experiments.append(exp)

        except Exception as e:
            print(f"Error loading {npz_path.name}: {e}")

    return experiments


def get_available_metadata(npz_files: Optional[list[Path]] = None) -> dict:
    """
    Analyze available metadata across all .npz files.

    Parameters
    ----------
    npz_files : list[Path], optional
        List of .npz files to analyze. If None, finds all available files.

    Returns
    -------
    dict
        Summary of available metadata including:
        - 'files': List of analyzed files with their metadata
        - 'unique_tracks': Set of unique track names
        - 'unique_animals': Set of unique animal IDs
        - 'unique_sessions': Set of unique session IDs
        - 'new_format_count': Number of files with new metadata format
        - 'old_format_count': Number of files without metadata
        - 'all_features': Set of all behavioral feature names
        - 'fps_values': Set of all FPS values found
    """
    if npz_files is None:
        npz_files = find_npz_files()

    result = {
        'files': [],
        'unique_tracks': set(),
        'unique_animals': set(),
        'unique_sessions': set(),
        'new_format_count': 0,
        'old_format_count': 0,
        'all_features': set(),
        'fps_values': set(),
    }

    for npz_path in npz_files:
        parsed = parse_iabs_filename(npz_path.name)
        metadata = get_npz_metadata(npz_path)

        file_info = {
            'path': npz_path,
            'filename': npz_path.name,
            'parsed': parsed,
            'metadata': metadata,
        }
        result['files'].append(file_info)

        if parsed:
            result['unique_tracks'].add(parsed.track)
            result['unique_animals'].add(parsed.animal_id)
            result['unique_sessions'].add(parsed.session)

        if metadata['has_new_format']:
            result['new_format_count'] += 1
        else:
            result['old_format_count'] += 1

        result['all_features'].update(metadata['feature_names'])

        if metadata['fps'] is not None:
            result['fps_values'].add(metadata['fps'])

    return result


def print_metadata_report(summary: Optional[dict] = None):
    """
    Print a formatted report of available metadata.

    Parameters
    ----------
    summary : dict, optional
        Summary from get_available_metadata(). If None, generates it.
    """
    if summary is None:
        summary = get_available_metadata()

    print("=" * 70)
    print("SYNCHRONIZED EXPERIMENTS METADATA REPORT")
    print("=" * 70)

    print(f"\nTotal files found: {len(summary['files'])}")
    print(f"  - New format (with _metadata): {summary['new_format_count']}")
    print(f"  - Old format (no metadata): {summary['old_format_count']}")

    print(f"\nUnique tracks: {sorted(summary['unique_tracks'])}")
    print(f"Unique animals: {sorted(summary['unique_animals'])}")
    print(f"Unique sessions: {sorted(summary['unique_sessions'])}")

    print(f"\nFPS values found: {sorted(summary['fps_values'])}")
    print(f"\nAll behavioral features: {sorted(summary['all_features'])}")

    print("\n" + "-" * 70)
    print("FILE DETAILS")
    print("-" * 70)

    for file_info in summary['files']:
        parsed = file_info['parsed']
        metadata = file_info['metadata']

        print(f"\n{file_info['filename']}")
        if parsed:
            print(f"  Track: {parsed.track}, Animal: {parsed.animal_id}, Session: {parsed.session}")
        else:
            print("  WARNING: Could not parse filename")

        print(f"  Format: {'New' if metadata['has_new_format'] else 'Old (no metadata)'}")
        print(f"  Neurons: {metadata['n_neurons']}, Frames: {metadata['n_frames']}")
        print(f"  FPS: {metadata['fps'] if metadata['fps'] else 'Not specified'}")
        print(f"  Features: {metadata['feature_names']}")


def print_improvement_suggestions():
    """Print suggestions for Experiment class metadata improvements."""

    print("\n" + "=" * 70)
    print("METADATA & EXPERIMENT CLASS IMPROVEMENT SUGGESTIONS")
    print("=" * 70)

    suggestions = """
## Current Metadata Available

### From Filename Parsing (IABS convention):
- track: Experiment paradigm (NOF, LNOF, STFP, HT, RT, FS, etc.)
- animal_id: Subject identifier (H01, J05, M123, etc.)
- session: Recording session (1D, 2D, 4D, etc.)

### From _metadata dict (new format only):
- session_name: Full session identifier
- fps: Frame rate (critical for timing)
- export_timestamp: Data export timestamp
- feedback_applied: CNMF feedback flag
- cnmf_params: Full CNMF extraction parameters (29 keys)
- autoinspection_stats: Quality metrics (n_total, n_good, n_bad, etc.)
- metrics_df: Per-component quality metrics (area, circularity, etc.)

### From _sync_info dict (new format only):
- synchronizer_version: Version tracking
- sync_timestamp: Synchronization timestamp
- experiment_name: Name from synchronizer
- source_format: Data source format
- alignment_mode: Alignment method used
- n_features: Number of behavioral features
- validation_passed: Validation status
- n_timepoints: Aligned timepoints

## Suggested Improvements to Experiment Class

### 1. Dedicated Metadata Attributes
Currently, metadata is stored in exp_identificators dict. Consider adding:
```python
class Experiment:
    # New dedicated attributes
    track: str              # Experiment paradigm
    animal_id: str          # Subject identifier
    session: str            # Session identifier (already exists)
    export_metadata: dict   # Original export metadata
    sync_metadata: dict     # Synchronization metadata
    quality_metrics: dict   # Per-neuron quality scores
```

### 2. Standardize FPS Handling
FPS is sometimes in _metadata['fps'] and sometimes needs to be provided.
```python
# Suggestion: Auto-extract from metadata if available
if '_metadata' in data and 'fps' in data['_metadata']:
    static_features['fps'] = data['_metadata']['fps']
```

### 3. Store Component Quality Metrics
The metrics_df contains valuable per-neuron quality info:
```python
# Store quality metrics per neuron
for neuron in exp.neurons:
    neuron.quality_metrics = {
        'area': ...,
        'circularity': ...,
        'max_edge': ...,
        'snr': ...,
    }
```

### 4. Add Provenance Tracking
```python
class Experiment:
    provenance: dict = {
        'source_file': str,          # Original file path
        'load_timestamp': datetime,  # When loaded
        'cnmf_version': str,         # CNMF version used
        'sync_version': str,         # Synchronizer version
    }
```

### 5. Session Hierarchy Support
For longitudinal studies with multiple sessions per animal:
```python
class ExperimentCollection:
    '''Container for related experiments (same animal, different days).'''
    animal_id: str
    sessions: dict[str, Experiment]  # session_id -> Experiment

    def get_session(self, session_id: str) -> Experiment
    def get_all_sessions(self) -> list[Experiment]
    def get_common_neurons(self) -> list[int]  # ROI matching
```

### 6. Backward Compatibility Layer
```python
def load_experiment_auto(path: Path) -> Experiment:
    '''Automatically detect format and load appropriately.'''
    metadata = get_npz_metadata(path)

    if metadata['has_new_format']:
        return load_new_format(path)
    else:
        return load_legacy_format(path)
```

## Data Format Standardization Needed

### Old format (NOF_H01_1D syn data.npz):
- No _metadata or _sync_info
- lowercase 'calcium'
- FPS not specified (defaults to 20.0)
- No quality metrics

### New format (LNOF_J01_4D_aligned.npz):
- Has _metadata and _sync_info
- Uppercase 'Calcium' and 'Reconstructions'
- FPS specified in metadata
- Full quality metrics available

### Recommendation:
1. Migrate old format files to include basic metadata
2. Standardize key names (prefer lowercase for consistency)
3. Always include FPS in metadata (don't rely on defaults)
4. Document expected keys in a schema
"""
    print(suggestions)


if __name__ == '__main__':
    # Run metadata analysis and print report
    print("Analyzing synchronized experiments...\n")

    summary = get_available_metadata()
    print_metadata_report(summary)
    print_improvement_suggestions()

    # Optionally load a sample experiment
    print("\n" + "=" * 70)
    print("SAMPLE EXPERIMENT LOADING")
    print("=" * 70)

    npz_files = find_npz_files()
    if npz_files:
        print(f"\nLoading first file as sample: {npz_files[0].name}")
        try:
            exp = load_experiment_from_npz(npz_files[0], verbose=True)
            print(f"\nLoaded experiment: {exp.signature}")
            print(f"  Neurons: {exp.n_cells}")
            print(f"  Frames: {exp.n_frames}")
            print(f"  FPS: {exp.fps}")
            print(f"  Features: {list(exp.dynamic_features.keys())}")

            # Show exp_identificators
            print(f"  exp_identificators: {exp.exp_identificators}")
        except Exception as e:
            print(f"Error loading sample: {e}")
