"""Loader of the earlier export of the NOF sessions ('<mouse>_<day> syn data.npz').

Only the example caches of Figures 1-2 are built from it. Its calcium matrix has
one extra leading row: row k here is row k-1 of the synchronized files that
INTENSE and the deposited dataset use.
"""

import numpy as np
from pathlib import Path


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import intense_paths as P  # noqa: E402

NOF_DATA_DIR = P.NOF_LEGACY
STATIC_FEATURES = {
    'fps': 20,
    't_rise_sec': 0.05,
    't_off_sec': 1.5,
}


def load_nof_experiment(animal_id='H01', day='3D', reconstruct_spikes=False, verbose=False):
    """Load NOF experiment data.

    Parameters
    ----------
    animal_id : str, default 'H01'
        Animal identifier (e.g., 'H01', 'H02', etc.)
    day : str, default '3D'
        Experimental day ('1D', '2D', '3D', or '4D')
    reconstruct_spikes : str or bool, default False
        Spike reconstruction method ('wavelet', 'threshold', or False)
    verbose : bool, default False
        Print progress messages

    Returns
    -------
    Exp : Experiment
        Loaded experiment object with neurons and calcium imaging data
    """
    from driada.experiment import load_exp_from_aligned_data

    # Construct filepath
    filename = f"NOF_{animal_id}_{day} syn data.npz"
    filepath = NOF_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"NOF data file not found: {filepath}")

    # Load raw data
    data = np.load(filepath)
    data_dict = {key: data[key] for key in data.keys()}

    # Experiment parameters
    exp_params = {
        'track': 'NOF',
        'animal_id': animal_id,
        'session': day
    }

    # Force position variables to be continuous
    force_continuous = ['x', 'y', 'speed', 'bodydirection', 'headdirection']

    # Build experiment
    exp = load_exp_from_aligned_data(
        data_source='NOF',
        exp_params=exp_params,
        data=data_dict,
        force_continuous=force_continuous,
        static_features=STATIC_FEATURES,
        verbose=verbose,
        reconstruct_spikes=reconstruct_spikes,
    )

    return exp


