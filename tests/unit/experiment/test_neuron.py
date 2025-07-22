import numpy as np
from driada.experiment.neuron import *
from driada.utils.neural import generate_pseudo_calcium_multisignal
from driada.experiment.wavelet_event_detection import (WVT_EVENT_DETECTION_PARAMS,
                                      extract_wvt_events, events_to_ts_array, ridges_to_containers)



def test_init():
    pseudo_ca = np.random.random(size=10000)
    pseudo_sp = np.zeros(10000)
    neuron = Neuron(0, pseudo_ca, pseudo_sp)


def test_wavelet_spike_inference():
    wvt_kwargs = WVT_EVENT_DETECTION_PARAMS.copy()
    wvt_kwargs['fps'] = 20

    n = 100 # number of signals
    duration = 600  # seconds
    sampling_rate = 20  # 20 Hz
    event_rate = 0.2  # 0.2 event per second on average
    amplitude_range = (0.5, 2)  # Amplitude range from 0.5 to 2
    decay_time = 2  # Decay time constant of 2 seconds
    noise_std = 0.1  # Standard deviation of noise

    pseudo_calcium = generate_pseudo_calcium_multisignal(n,
                                                         duration,
                                                         sampling_rate,
                                                         event_rate,
                                                         amplitude_range,
                                                         decay_time,
                                                         noise_std)

    st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(pseudo_calcium, wvt_kwargs)
    all_wvt_ridges = [ridges_to_containers(ridges) for ridges in all_ridges]
    spikes = events_to_ts_array(pseudo_calcium.shape[1], st_ev_inds, end_ev_inds, wvt_kwargs['fps'])

    print(spikes.shape)
