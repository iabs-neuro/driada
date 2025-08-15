"""Test the updated wavelet implementation"""

import numpy as np
import os
import sys

# Test with numba disabled
os.environ['DRIADA_DISABLE_NUMBA'] = '1'

from src.driada.experiment.wavelet_event_detection import extract_wvt_events, WVT_EVENT_DETECTION_PARAMS
from src.driada.utils.jit import jit_info

print("Testing wavelet event detection with updated implementation")
print("=" * 60)
jit_info()
print("=" * 60)

# Create test data
np.random.seed(42)
n_traces = 5
trace_length = 1000
traces = []

for i in range(n_traces):
    # Create synthetic calcium trace with some events
    trace = np.random.randn(trace_length) * 0.1
    # Add some synthetic events
    for j in range(3):
        event_start = np.random.randint(100, 800)
        event_duration = np.random.randint(20, 100)
        event_amplitude = np.random.uniform(1, 3)
        trace[event_start:event_start + event_duration] += event_amplitude * np.exp(-np.linspace(0, 5, event_duration))
    traces.append(trace)

# Test extraction
print("\nExtracting events from", n_traces, "traces...")
st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(traces, WVT_EVENT_DETECTION_PARAMS)

print("\nResults:")
for i, (st, end) in enumerate(zip(st_ev_inds, end_ev_inds)):
    print(f"Trace {i}: Found {len(st)} events")
    
print("\nTotal ridges found:", sum(len(r) for r in all_ridges))

# Now test with numba enabled
print("\n" + "=" * 60)
print("Testing with numba enabled...")
os.environ['DRIADA_DISABLE_NUMBA'] = '0'

# Need to reload modules
import importlib
from src.driada.utils import jit as jit_module
from src.driada.experiment import wavelet_event_detection as wvt_module
from src.driada.experiment import wavelet_ridge as ridge_module

importlib.reload(jit_module)
importlib.reload(ridge_module)
importlib.reload(wvt_module)

from src.driada.utils.jit import jit_info
jit_info()

# Test again
from src.driada.experiment.wavelet_event_detection import extract_wvt_events
st_ev_inds2, end_ev_inds2, all_ridges2 = extract_wvt_events(traces, WVT_EVENT_DETECTION_PARAMS)

print("\nResults with numba:")
for i, (st, end) in enumerate(zip(st_ev_inds2, end_ev_inds2)):
    print(f"Trace {i}: Found {len(st)} events")

# Compare results
print("\nComparison:")
results_match = True
for i in range(n_traces):
    if len(st_ev_inds[i]) != len(st_ev_inds2[i]):
        print(f"Trace {i}: Mismatch in number of events!")
        results_match = False
    else:
        # Check if events are similar (small differences due to numerical precision)
        for j in range(len(st_ev_inds[i])):
            if abs(st_ev_inds[i][j] - st_ev_inds2[i][j]) > 1:
                print(f"Trace {i}, Event {j}: Start index mismatch")
                results_match = False

if results_match:
    print("✓ Results match between numba and non-numba versions!")
else:
    print("✗ Results differ between versions")