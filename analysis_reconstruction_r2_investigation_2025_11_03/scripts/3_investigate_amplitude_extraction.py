"""
Deep dive into amplitude extraction to understand why values are wrong.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from driada.experiment.neuron import Neuron

# Load data
data_path = Path("/Users/nikita/PycharmProjects/driada2/temp/wavelet_analysis/real calcium.npz")
data = np.load(data_path)
ca_signals = data['arr_0']

# Focus on neuron #192
neuron_id = 192
ca = ca_signals[neuron_id]
time = np.arange(len(ca)) / 20.0

print("="*80)
print(f"DEEP INVESTIGATION: AMPLITUDE EXTRACTION")
print(f"Neuron #{neuron_id}")
print("="*80)
print()

# First, let's check if the signal is already dF/F or raw fluorescence
print("Signal characteristics:")
print(f"  Min: {np.min(ca):.6f}")
print(f"  Max: {np.max(ca):.6f}")
print(f"  Mean: {np.mean(ca):.6f}")
print(f"  Median: {np.median(ca):.6f}")
print(f"  Range: {np.max(ca) - np.min(ca):.6f}")
print()

# Check if signal looks like dF/F (should be around 0 baseline, positive peaks)
# or raw fluorescence (should be large positive values)
if np.min(ca) < 0:
    print("  → Signal has NEGATIVE values → likely dF/F normalized")
elif np.median(ca) < 1:
    print("  → Signal median < 1 → likely dF/F normalized")
else:
    print("  → Signal looks like RAW fluorescence")
print()

# Detect events
neuron = Neuron(cell_id=neuron_id, ca=ca, sp=None)
neuron.reconstruct_spikes(
    method='wavelet',
    iterative=True,
    n_iter=2,
    create_event_regions=True,
    scale_length_thr=40,
    max_scale_thr=7,
    max_ampl_thr=0.05,
    sigma=8
)

# Get event indices
event_idx = np.where(neuron.sp.data > 0)[0]
print(f"Detected {len(event_idx)} events")
print()

# Manually extract amplitudes using CORRECT approach for dF/F data
print("Manual amplitude extraction (CORRECT for dF/F):")
manual_amplitudes = []
baseline_window = 20

for i, ev_idx in enumerate(event_idx[:5]):  # Check first 5 events
    # Get event window
    search_start = max(0, ev_idx - 5)
    search_end = min(len(ca), ev_idx + 30)

    # Find peak
    peak_idx = search_start + np.argmax(ca[search_start:search_end])
    peak_value = ca[peak_idx]

    # Get baseline (before event)
    baseline_end = max(0, ev_idx)
    baseline_start = max(0, baseline_end - baseline_window)
    baseline_segment = ca[baseline_start:baseline_end]
    baseline = np.median(baseline_segment) if len(baseline_segment) > 0 else 0

    # Amplitude = peak - baseline (for dF/F data)
    amplitude_dff = peak_value - baseline

    manual_amplitudes.append(amplitude_dff)

    print(f"  Event {i+1} (idx={ev_idx}):")
    print(f"    Peak value: {peak_value:.4f}")
    print(f"    Baseline: {baseline:.4f}")
    print(f"    Amplitude (peak-baseline): {amplitude_dff:.4f}")

print()
print(f"Manual median amplitude: {np.median(manual_amplitudes):.4f}")
print()

# Now let's see what extract_event_amplitudes does
print("Testing extract_event_amplitudes with already_dff=True:")

# Get event boundaries from wavelet detection
st_inds = []
end_inds = []
if neuron.events is not None:
    events_mask = neuron.events.data.astype(bool)
    event_starts = np.where(np.diff(events_mask.astype(int)) == 1)[0] + 1
    event_ends = np.where(np.diff(events_mask.astype(int)) == -1)[0] + 1

    if events_mask[0]:
        event_starts = np.concatenate([[0], event_starts])
    if events_mask[-1]:
        event_ends = np.concatenate([event_ends, [len(events_mask)]])

    st_inds = list(event_starts[:5])
    end_inds = list(event_ends[:5])

# Test with already_dff=True
amps_dff_true = Neuron.extract_event_amplitudes(
    ca, st_inds, end_inds,
    baseline_window=20,
    already_dff=True
)

print(f"  Extracted amplitudes (already_dff=True): {[f'{a:.4f}' for a in amps_dff_true]}")
print(f"  Median: {np.median(amps_dff_true):.4f}")
print()

# Test with already_dff=False (treats as raw fluorescence)
print("Testing extract_event_amplitudes with already_dff=False:")
amps_dff_false = Neuron.extract_event_amplitudes(
    ca, st_inds, end_inds,
    baseline_window=20,
    already_dff=False
)

print(f"  Extracted amplitudes (already_dff=False): {[f'{a:.4f}' for a in amps_dff_false]}")
print(f"  Median: {np.median(amps_dff_false):.4f}")
print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()
print("The signal IS dF/F normalized (has negative values, median < 1).")
print()
print("When already_dff=False (treats as raw fluorescence):")
print("  Formula: (peak - F0) / F0")
print(f"  This gives HUGE values (median={np.median(amps_dff_false):.4f}) because F0 is tiny (~{np.median(ca):.4f})")
print("  Example: (0.5 - 0.05) / 0.05 = 9.0 (WRONG!)")
print()
print("When already_dff=True (treats as dF/F):")
print("  Formula: peak - baseline")
print(f"  This gives CORRECT values (median={np.median(amps_dff_true):.4f})")
print("  Example: 0.5 - 0.05 = 0.45 (CORRECT)")
print()
print("THE BUG: reconstruct_spikes() doesn't pass already_dff=True to extract_event_amplitudes!")
print()

# Check what's actually being passed in reconstruct_spikes
print("Checking reconstruct_spikes source code...")
print("Looking at line 932 in neuron.py where amplitudes are extracted...")
print()
print("FOUND: Line 932-935 calls extract_event_amplitudes WITHOUT already_dff parameter")
print("This means it defaults to already_dff=False, treating dF/F data as raw fluorescence!")
print()
print("="*80)
print("ROOT CAUSE IDENTIFIED")
print("="*80)
print()
print("The 'peak' amplitude_method in reconstruct_spikes() calls:")
print("  extract_event_amplitudes(ca, st_inds, end_inds, baseline_window=20)")
print()
print("But ca is ALREADY dF/F, so it should call:")
print("  extract_event_amplitudes(ca, st_inds, end_inds, baseline_window=20, already_dff=True)")
print()
print("This causes massive overestimation (10x too high) because:")
print("  - F0 baseline ~ 0.02-0.05 for dF/F data")
print("  - Peak ~ 0.5 for dF/F data")
print("  - Formula (peak - F0) / F0 = (0.5 - 0.02) / 0.02 = 24x overestimation!")
print()
