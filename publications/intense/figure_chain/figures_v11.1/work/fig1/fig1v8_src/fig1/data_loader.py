"""Data loader for INTENSE figures."""

import numpy as np
from pathlib import Path


# Configuration: the panels drawn from a session read the earlier export ([data] nof_legacy) when it is
# there, otherwise deposited_session(); the example caches lie in figure_chain/saved_inputs
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # figure_chain/
import _paths_fig as _F  # noqa: E402

NOF_DATA_DIR = _F.P.NOF_LEGACY
CACHE_DIR = _F.SAVED
STATIC_FEATURES = {
    'fps': 20,
    't_rise_sec': 0.05,
    't_off_sec': 1.5,
}


def deposited_session(animal_id, day):
    """The session as the panels read it when the earlier export is not there.

    The calcium of the earlier export is the deposited calcium with one extra leading row, so with a copy
    of the first row in its place the row numbers of the panel scripts stay valid. Of its behavior the panels
    drawn from a session use one variable, saved in figure_chain/saved_inputs (README.md).
    """
    with np.load(_F.V.get("nof_v1") / f"NOF_{animal_id}_{day}_aligned.npz", allow_pickle=True) as d:
        data = {k: d[k] for k in d.files if not k.startswith("_")}
    calcium = data.pop("Calcium")
    data["calcium"] = np.vstack([calcium[:1], calcium])
    data["object4"] = np.load(_F.SAVED / f"NOF_{animal_id}_{day}_object4_earlier_export.npz")["object4"]
    return data


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

    # Load raw data
    if filepath.exists():
        data = np.load(filepath)
        data_dict = {key: data[key] for key in data.keys()}
    else:
        data_dict = deposited_session(animal_id, day)

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


# =============================================================================
# Cached data loaders for fast figure generation
# =============================================================================

def load_fig1a_cache():
    """Load cached data for Panel A (calcium traces).

    Returns
    -------
    dict with keys:
        - 'neuron_36_ca', 'neuron_37_ca', 'neuron_38_ca': calcium data arrays
        - 'neuron_indices': [36, 37, 38]
        - 'fps': 20

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist. fig1/cache_examples.py rebuilds it from the earlier export.
    """
    cache_path = CACHE_DIR / 'fig1a_data.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "It is in figure_chain/saved_inputs; fig1/cache_examples.py rebuilds it from the earlier export"
        )
    return dict(np.load(cache_path))


def load_fig1b_cache():
    """Load cached data for Panel B (behavioral traces).

    Returns
    -------
    dict with keys:
        - 'headdirection', 'object4', 'x', 'y': data arrays
        - 'fps': 20

    Raises
    ------
    FileNotFoundError
        If cache doesn't exist. fig1/cache_examples.py rebuilds it from the earlier export.
    """
    cache_path = CACHE_DIR / 'fig1b_data.npz'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "It is in figure_chain/saved_inputs; fig1/cache_examples.py rebuilds it from the earlier export"
        )
    return dict(np.load(cache_path))


def cache_exists():
    """Check if all cache files exist.

    Returns
    -------
    bool
        True if all cache files exist, False otherwise.
    """
    return all([
        (CACHE_DIR / 'fig1a_data.npz').exists(),
        (CACHE_DIR / 'fig1b_data.npz').exists(),
    ])
