import copy
from .exp_base import Experiment
from ..signal.sig_base import TimeSeries
from ..utils.naming import construct_session_name_iabs
from .neuron import DEFAULT_FPS, DEFAULT_T_OFF, DEFAULT_T_RISE


def load_exp_from_aligned_data(exp_params, data, force_continuous=[], static_features=dict()):
    expname = construct_session_name_iabs(exp_params)
    adata = copy.deepcopy(data)
    key_mapping = {key.lower(): key for key in adata.keys()}

    if 'calcium' in key_mapping:
        calcium = adata.pop(key_mapping['calcium'])
    else:
        raise ValueError('No calcium data found!')

    spikes = None
    if 'spikes' in key_mapping:
        spikes = adata.pop(key_mapping['spikes'])

    dyn_features = adata.copy()
    if len(force_continuous) != 0:
        feat_is_continuous = {f: f in force_continuous for f in dyn_features.keys()}
        filt_dyn_features = {f: TimeSeries(vals, discrete=not feat_is_continuous[f]) for f, vals in dyn_features.items() if len(set(vals))!=1}
    else:
        filt_dyn_features = {f: TimeSeries(vals) for f, vals in dyn_features.items() if len(set(vals)) != 1}

    print('behaviour variables:')
    print()
    for f, ts in filt_dyn_features.items():
        print(f"'{f}'", 'discrete' if ts.discrete else 'continuous')

    # check for constant features
    constfeats = set(dyn_features.keys() - set(filt_dyn_features.keys()))
    if len(constfeats) != 0:
        print(f'features {constfeats} dropped as constant')

    auto_continuous = [fn for fn, ts in filt_dyn_features.items() if not ts.discrete]
    print(f'features {auto_continuous} automatically determined as continuous')
    print()

    if len(force_continuous) != 0:
        if set(auto_continuous) != (set(force_continuous) & set(dyn_features.keys())):
            print('Warning: auto determined continuous features do not coincide with force_continuous list! Automatic labelling will be overridden')
            for fn, ts in filt_dyn_features.items():
                if len(set(ts.data)) > 2 and not(fn in force_continuous):
                    filt_dyn_features[fn] = TimeSeries(ts.data.astype(bool).astype(int), discrete=True)
                    print(f'feature {fn} converted to integer')

    signature = f'Exp {expname}'
    Exp = Experiment(signature, calcium, spikes, exp_params, static_features, filt_dyn_features)

    return Exp