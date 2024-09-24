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


def load_exp_from_aligned_data(data_source,
                               exp_params,
                               data,
                               force_continuous=[],
                               bad_frames=[],
                               static_features=None,
                               verbose=True,
                               reconstruct_spikes='wavelet'):

    expname = construct_session_name(data_source, exp_params)
    adata = copy.deepcopy(data)
    key_mapping = {key.lower(): key for key in adata.keys()}

    if verbose:
        print(f'Building experiment {expname}...')

    if 'calcium' in key_mapping:
        calcium = adata.pop(key_mapping['calcium'])
    else:
        raise ValueError('No calcium data found!')

    spikes = None
    if 'spikes' in key_mapping:
        spikes = adata.pop(key_mapping['spikes'])

    dyn_features = adata.copy()

    def is_garbage(vals):
        return len(set(vals)) == 1 or np.sum(np.isnan(vals)).astype(int) == len(vals)

    if len(force_continuous) != 0:
        feat_is_continuous = {f: f in force_continuous for f in dyn_features.keys()}
        filt_dyn_features = {f: TimeSeries(vals, discrete=not feat_is_continuous[f]) for f, vals in dyn_features.items() if not is_garbage(vals)}
    else:
        filt_dyn_features = {f: TimeSeries(vals) for f, vals in dyn_features.items() if not is_garbage(vals)}

    if verbose:
        print('behaviour variables:')
        print()
        for f, ts in filt_dyn_features.items():
            print(f"'{f}'", 'discrete' if ts.discrete else 'continuous')

    # check for constant features
    constfeats = set(dyn_features.keys() - set(filt_dyn_features.keys()))

    if len(constfeats) != 0 and verbose:
        print(f'features {constfeats} dropped as constant or empty')

    auto_continuous = [fn for fn, ts in filt_dyn_features.items() if not ts.discrete]
    if verbose:
        print(f'features {auto_continuous} automatically determined as continuous')
        print()

    if len(force_continuous) != 0:
        if set(auto_continuous) != (set(force_continuous) & set(dyn_features.keys())):
            print('Warning: auto determined continuous features do not coincide with force_continuous list! Automatic labelling will be overridden')
            for fn, ts in filt_dyn_features.items():
                if len(set(ts.data)) > 2 and not(fn in force_continuous):
                    filt_dyn_features[fn] = TimeSeries(ts.data.astype(bool).astype(int), discrete=True)
                    if verbose:
                        print(f'feature {fn} converted to integer')

    signature = f'Exp {expname}'

    # set default static experiment features if not provided
    default_static_features = {'t_rise_sec': DEFAULT_T_RISE,
                               't_off_sec': DEFAULT_T_OFF,
                               'fps': DEFAULT_FPS}

    if static_features is None:
        static_features = dict()
    for sf in default_static_features.keys():
        if sf not in static_features:
            static_features.update({sf: default_static_features[sf]})

    exp = Experiment(signature,
                     calcium,
                     spikes,
                     exp_params,
                     static_features,
                     filt_dyn_features,
                     reconstruct_spikes=reconstruct_spikes,
                     bad_frames_mask=np.array([True if _ in bad_frames else False for _ in range(calcium.shape[1])])
                     )


    return exp


# TODO: add support for g-cloud auth from desktop (for pydrive) https://cloud.google.com/docs/authentication/application-default-credentials
def load_experiment(data_source,
                    exp_params,
                    force_rebuild=False,
                    force_reload=False,
                    via_pydrive=True,
                    gauth=None,
                    root='DRIADA data',
                    force_continuous=[],
                    bad_frames=[],
                    static_features=None,
                    reconstruct_spikes='wavelet',
                    verbose=True):

    os.makedirs(root, exist_ok=True)
    if not os.path.isdir(root):
        raise ValueError('Root must be a folder!')

    expname = construct_session_name(data_source, exp_params)
    exppath = os.path.join(root, expname, f'Exp {expname}.pickle')
    if os.path.exists(exppath) and not force_rebuild and not force_reload:
        Exp = load_exp_from_pickle(exppath, verbose=verbose)
        return Exp, None

    else:
        if data_source == 'IABS':
            syn_data_name = os.path.join(root,
                                         expname,
                                         'Aligned data',
                                         f'{expname} syn data.npz')

            data_exists = os.path.exists(syn_data_name)
            if force_reload or not data_exists:
                data_router, data_pieces = initialize_iabs_router(root=root)
                success, load_log = download_gdrive_data(data_router,
                                                         expname,
                                                         data_pieces=['Aligned data'],
                                                         via_pydrive=via_pydrive,
                                                         tdir=root,
                                                         gauth=gauth)

                if not success:
                    print('===========   BEGINNING OF LOADING LOG:   ============')
                    show_output(load_log)
                    print('===========   END OF LOADING LOG:   ============')
                    raise FileNotFoundError(f'Cannot download {expname}, see loading log above')

            else:
                load_log = None

            aligned_data = dict(np.load(syn_data_name))
            Exp = load_exp_from_aligned_data(data_source,
                                             exp_params,
                                             aligned_data,
                                             force_continuous=force_continuous,
                                             static_features=static_features,
                                             verbose=verbose,
                                             bad_frames=bad_frames,
                                             reconstruct_spikes=reconstruct_spikes)

            save_exp_to_pickle(Exp, exppath, verbose=verbose)
            return Exp, load_log

        else:
            raise ValueError('External data sources not supported yet')


def save_exp_to_pickle(exp, path, verbose=True):
    with open(path, "wb") as f:
        pickle.dump(exp, f)
        if verbose:
            print(f'Experiment {exp.signature} saved to {path}\n')


def load_exp_from_pickle(path, verbose=True):
    with open(path, "rb") as f:
        exp = pickle.load(f,)
        if verbose:
            print(f'Experiment {exp.signature} loaded from {path}\n')
    return exp
