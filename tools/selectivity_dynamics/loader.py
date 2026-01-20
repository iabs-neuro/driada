"""Experiment loading utilities for INTENSE analysis.

Contains:
- load_experiment_from_npz: Load experiment from NPZ file
- build_feature_list: Build feature list excluding specified features
- get_skip_delays: Get list of features to skip delay optimization
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from driada.experiment.exp_build import load_exp_from_aligned_data
from driada.information.info_base import MultiTimeSeries
from driada.utils.naming import parse_iabs_filename

# Import get_npz_metadata from tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from load_synchronized_experiments import get_npz_metadata


def build_feature_list(exp, skip_features):
    """Build feature list excluding specified features."""
    feat_bunch = [
        feat_name for feat_name in exp.dynamic_features.keys()
        if feat_name not in skip_features
    ]
    return feat_bunch


def get_skip_delays(exp):
    """Get list of features to skip delay optimization (MultiTimeSeries)."""
    return [
        feat_name for feat_name, feat_data in exp.dynamic_features.items()
        if isinstance(feat_data, MultiTimeSeries)
    ]


def load_experiment_from_npz(npz_path, agg_features=None, verbose=True):
    """Load experiment from NPZ file.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file
    agg_features : dict, optional
        Feature aggregation mapping {(key1, key2): 'combined_name'}
        These features are combined into MultiTimeSeries during construction.
    verbose : bool
        Whether to print loading info

    Returns
    -------
    Experiment
        Loaded experiment object
    """
    npz_path = Path(npz_path)

    # Parse filename to get exp_params
    parsed = parse_iabs_filename(npz_path.name)
    if parsed is None:
        raise ValueError(f"Could not parse filename: {npz_path.name}")

    # Load data
    data = dict(np.load(npz_path, allow_pickle=True))

    # Get metadata for fps
    metadata = get_npz_metadata(npz_path)

    # Build static features
    static_features = {}
    if metadata['fps'] is not None:
        static_features['fps'] = float(metadata['fps'])

    # Build exp_params from parsed dict
    exp_params = {k: parsed[k] for k in ('track', 'animal_id', 'session')}

    # Remove metadata keys from data dict
    data.pop('_metadata', None)
    data.pop('_sync_info', None)

    # Create experiment with aggregate_features parameter
    # This creates MultiTimeSeries during construction, ensuring proper data hashes
    exp = load_exp_from_aligned_data(
        data_source='IABS',
        exp_params=exp_params,
        data=data,
        static_features=static_features,
        aggregate_features=agg_features,
        verbose=verbose,
    )

    return exp
