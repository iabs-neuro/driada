#!/usr/bin/env python3
"""Check where kinetics values come from."""

import numpy as np
from pathlib import Path
from driada.experiment.neuron import Neuron

# Load neuron
data_path = Path("temp/wavelet_analysis/real calcium.npz")
data = np.load(data_path, allow_pickle=True)
ca_trace = data['arr_0'][192]
fps = 20.0

# Create neuron with defaults
neuron = Neuron("test", ca_trace, None, fps=fps)

print("Neuron kinetics after creation:")
print(f"  default_t_rise (frames): {neuron.default_t_rise}")
print(f"  default_t_off (frames): {neuron.default_t_off}")
print(f"  default_t_rise (seconds): {neuron.default_t_rise/fps:.2f}s")
print(f"  default_t_off (seconds): {neuron.default_t_off/fps:.2f}s")
print(f"  t_rise: {neuron.t_rise}")
print(f"  t_off: {neuron.t_off}")

# After reconstruction
neuron.reconstruct_spikes(method='wavelet', create_event_regions=True, verbose=False)

print("\nAfter reconstruction:")
print(f"  t_rise: {neuron.t_rise}")
print(f"  t_off: {neuron.t_off}")
if neuron.t_rise:
    print(f"  t_rise (seconds): {neuron.t_rise/fps:.2f}s")
if neuron.t_off:
    print(f"  t_off (seconds): {neuron.t_off/fps:.2f}s")
