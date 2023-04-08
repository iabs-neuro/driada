def construct_session_name(exp_params):
    track = exp_params['track']

    if track == 'HT':
        animal_id, session = exp_params['animal_id'], exp_params['session']
        name = f'{animal_id}_HT{session}'
    elif track == 'RT':
        animal_id, session = exp_params['animal_id'], exp_params['session']
        name = f'{animal_id}_{session}D'
    elif track == 'FS':
        animal_id, session = exp_params['animal_id'], exp_params['session']
        name = f'FS{animal_id}_{session}D'
    elif track == 'FcOY':
        animal_id, session = exp_params['animal_id'], exp_params['session']
        name = f'FcOY_{animal_id}_{session}'

    return name