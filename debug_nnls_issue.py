#!/usr/bin/env python3
"""
Debug NNLS event masking issue.

The reconstructions look terrible - investigate what's happening.
"""

import numpy as np
from pathlib import Path
from driada.experiment.neuron import Neuron
import matplotlib.pyplot as plt

neuron_id = 192

# Load neuron
data_path = Path("temp/wavelet_analysis/real calcium.npz")
data = np.load(data_path, allow_pickle=True)
ca_trace = data['arr_0'][neuron_id]
fps = 20.0

neuron = Neuron(f"neuron_{neuron_id}", ca_trace, None, fps=fps)

# Reconstruct WITH event mask
print("="*80)
print(f"Neuron #{neuron_id} - WITH event mask")
print("="*80)
neuron.reconstruct_spikes(
    method='wavelet',
    amplitude_method='deconvolution',
    iterative=True,
    n_iter=2,
    create_event_regions=True,
    verbose=False
)

ca_signal = neuron.ca.data
event_mask = neuron.events.data > 0
reconstructed_with_mask = neuron.reconstructed.data
asp_with_mask = neuron.asp.data

print(f"\nSignal stats:")
print(f"  Total frames: {len(ca_signal)}")
print(f"  Event frames: {np.sum(event_mask)} ({100*np.sum(event_mask)/len(ca_signal):.1f}%)")
print(f"  Baseline frames: {len(ca_signal) - np.sum(event_mask)}")

print(f"\nKinetics:")
t_rise = neuron.t_rise if neuron.t_rise else neuron.default_t_rise
t_off = neuron.t_off if neuron.t_off else neuron.default_t_off
print(f"  t_rise: {t_rise:.2f}s ({t_rise*fps:.0f} frames)")
print(f"  t_off: {t_off:.2f}s ({t_off*fps:.0f} frames)")

print(f"\nEvent detection:")
event_starts = np.where(np.diff(event_mask.astype(int)) == 1)[0] + 1
event_ends = np.where(np.diff(event_mask.astype(int)) == -1)[0] + 1
if event_mask[0]:
    event_starts = np.concatenate([[0], event_starts])
if event_mask[-1]:
    event_ends = np.concatenate([event_ends, [len(event_mask)]])

n_events = min(len(event_starts), len(event_ends))
event_widths = event_ends[:n_events] - event_starts[:n_events]
print(f"  Number of events: {n_events}")
print(f"  Event width: {np.median(event_widths)/fps:.2f}s median ({np.median(event_widths):.0f} frames)")
print(f"  Event width range: {np.min(event_widths)/fps:.2f}s - {np.max(event_widths)/fps:.2f}s")

print(f"\nReconstruction quality:")
r2 = neuron.get_reconstruction_r2()
print(f"  R²: {r2:.4f}")

print(f"\nAmplitudes (asp):")
spike_indices = np.where(asp_with_mask > 0)[0]
amplitudes_with_mask = asp_with_mask[spike_indices]
print(f"  Number of spikes: {len(spike_indices)}")
print(f"  Amplitude range: {np.min(amplitudes_with_mask):.3f} - {np.max(amplitudes_with_mask):.3f}")
print(f"  Amplitude median: {np.median(amplitudes_with_mask):.3f}")

# Create neuron WITHOUT event mask
print("\n" + "="*80)
print(f"Neuron #{neuron_id} - WITHOUT event mask (legacy)")
print("="*80)

neuron2 = Neuron(f"neuron_{neuron_id}", ca_trace, None, fps=fps)
neuron2.reconstruct_spikes(
    method='wavelet',
    amplitude_method='deconvolution',
    iterative=True,
    n_iter=2,
    create_event_regions=False,  # No mask
    verbose=False
)

reconstructed_without_mask = neuron2.reconstructed.data
asp_without_mask = neuron2.asp.data

print(f"\nReconstruction quality:")
r2_no_mask = neuron2.get_reconstruction_r2()
print(f"  R²: {r2_no_mask:.4f}")

print(f"\nAmplitudes (asp):")
spike_indices2 = np.where(asp_without_mask > 0)[0]
amplitudes_without_mask = asp_without_mask[spike_indices2]
print(f"  Number of spikes: {len(spike_indices2)}")
print(f"  Amplitude range: {np.min(amplitudes_without_mask):.3f} - {np.max(amplitudes_without_mask):.3f}")
print(f"  Amplitude median: {np.median(amplitudes_without_mask):.3f}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nR² change: {r2_no_mask:.4f} → {r2:.4f} (Δ={r2-r2_no_mask:.4f})")
print(f"Amplitude median: {np.median(amplitudes_without_mask):.3f} → {np.median(amplitudes_with_mask):.3f}")
print(f"Amplitude ratio (with/without mask): {np.median(amplitudes_with_mask)/np.median(amplitudes_without_mask):.3f}")

# Visualize one event
print("\n" + "="*80)
print("EVENT ANALYSIS")
print("="*80)

# Pick first event
event_idx = 0
event_start = event_starts[event_idx]
event_end = event_ends[event_idx]

# Extend window to see full calcium transient
kernel_length = int(5 * t_off * fps)  # 5 decay time constants
window_start = max(0, event_start - 20)
window_end = min(len(ca_signal), event_end + kernel_length)

print(f"\nEvent #{event_idx}:")
print(f"  Event region: frames {event_start}-{event_end} ({(event_end-event_start)/fps:.2f}s)")
print(f"  Analysis window: frames {window_start}-{window_end} ({(window_end-window_start)/fps:.2f}s)")
print(f"  Kernel decay: ~{kernel_length/fps:.1f}s ({kernel_length} frames)")

time = np.arange(window_start, window_end) / fps
ca_win = ca_signal[window_start:window_end]
recon_with_win = reconstructed_with_mask[window_start:window_end]
recon_without_win = reconstructed_without_mask[window_start:window_end]
event_win = event_mask[window_start:window_end]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time, ca_win, 'k-', linewidth=2, label='Observed', alpha=0.7)
ax.plot(time, recon_without_win, 'b-', linewidth=1.5, label='Recon (no mask, R²={:.3f})'.format(r2_no_mask), alpha=0.7)
ax.plot(time, recon_with_win, 'r-', linewidth=1.5, label='Recon (with mask, R²={:.3f})'.format(r2), alpha=0.7)

# Shade event region
event_time_start = event_start / fps
event_time_end = event_end / fps
ax.axvspan(event_time_start, event_time_end, alpha=0.2, color='green', label='Event region')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('ΔF/F', fontsize=12)
ax.set_title(f'Neuron #{neuron_id} - Event #{event_idx} Analysis', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_nnls_single_event.png', dpi=150)
print(f"\nSaved: debug_nnls_single_event.png")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print(f"\nProblem identified:")
print(f"  1. Calcium kernel decay: {t_off:.1f}s = {t_off*fps:.0f} frames")
print(f"  2. Event region width: {np.median(event_widths)/fps:.2f}s = {np.median(event_widths):.0f} frames")
print(f"  3. Ratio: Kernel is {t_off*fps/np.median(event_widths):.1f}x longer than event region!")
print(f"\n→ Event mask only covers {100*np.median(event_widths)/(t_off*fps):.1f}% of the calcium transient")
print(f"→ NNLS fits only the peak, ignoring the long exponential decay")
print(f"→ Amplitudes are optimized for wrong objective function")
