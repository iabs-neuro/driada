def construct_session_name(data_source, exp_params):
    if data_source == 'IABS':
        track = exp_params['track']
        animal_id, session = exp_params['animal_id'], exp_params['session']

        if track == 'HT':
            name = f'{animal_id}_HT{session}'
        elif track == 'RT':
            name = f'RT_{animal_id}_{session}D'
        elif track == 'FS':
            name = f'FS{animal_id}_{session}D'
        elif track == 'FcOY':
            name = f'FcOY_{animal_id}_{session}'
        elif track == 'STFP':
            name = f'STFP{animal_id}_{session}'
        elif track == 'AP':
            name = f'AP_{animal_id}_{session}'
        elif track == 'NOF':
            name = f'NOF_{animal_id}_{session}'
        else:
            raise ValueError(f'Unknown track: {track}!')

        return name

    else:
        raise NotImplementedError('Other data sources are not yet supported')