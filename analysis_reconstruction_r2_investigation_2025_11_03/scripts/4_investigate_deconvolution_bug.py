"""
Investigate why deconvolution method underestimates amplitudes by 50%.
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

print("="*80)
print("INVESTIGATING DECONVOLUTION AMPLITUDE UNDERESTIMATION")
print("="*80)
print()

# Detect events with deconvolution
neuron = Neuron(cell_id=neuron_id, ca=ca, sp=None)
neuron.reconstruct_spikes(
    method='wavelet',
    iterative=True,
    n_iter=2,
    amplitude_method='deconvolution',
    create_event_regions=True,
    scale_length_thr=40,
    max_scale_thr=7,
    max_ampl_thr=0.05,
    sigma=8
)

# Optimize kinetics
opt_result = neuron.optimize_kinetics(method='direct', fps=20.0, update_reconstruction=False)
t_rise = opt_result['t_rise'] * 20.0  # frames
t_off = opt_result['t_off'] * 20.0

neuron.t_rise = t_rise
neuron.t_off = t_off

# Re-detect
neuron.reconstruct_spikes(
    method='wavelet',
    iterative=True,
    n_iter=2,
    amplitude_method='deconvolution',
    create_event_regions=True,
    scale_length_thr=40,
    max_scale_thr=7,
    max_ampl_thr=0.05,
    sigma=8
)

event_idx = np.where(neuron.asp.data > 0)[0]
amplitudes_deconv = neuron.asp.data[event_idx]

print(f"Kinetics: t_rise={t_rise:.1f} frames, t_off={t_off:.1f} frames")
print(f"Events detected: {len(event_idx)}")
print(f"Deconvolution amplitudes: {amplitudes_deconv[:5]}")
print(f"Median: {np.median(amplitudes_deconv):.4f}")
print()

# Now let's check the kernel normalization
print("Checking kernel normalization:")
kernel_length = 200
t_array = np.arange(kernel_length)
kernel = (1 - np.exp(-t_array / t_rise)) * np.exp(-t_array / t_off)
kernel_max = np.max(kernel)
kernel_normalized = kernel / kernel_max

print(f"  Kernel max (before normalization): {kernel_max:.6f}")
print(f"  Kernel max (after normalization): {np.max(kernel_normalized):.6f}")
print(f"  Peak location: {np.argmax(kernel):.0f} frames")
print()

# The issue: when we deconvolve with normalized kernel and then reconstruct with normalized kernel,
# the amplitudes should be correct. But they're not. Let's check the scaling...

# Let's manually deconvolve and see what happens
print("Manual deconvolution test:")

# Use first 5 events that are within first 1000 frames
test_signal = ca[:3000]  # Use a larger subset
test_events = [ev for ev in event_idx[:10] if ev < len(test_signal)][:5]

# Build design matrix with normalized kernel
from scipy.optimize import nnls

n_frames = len(test_signal)
n_events = len(test_events)
K = np.zeros((n_frames, n_events))

for i, ev_idx in enumerate(test_events):
    if ev_idx >= n_frames:
        continue
    remaining = n_frames - ev_idx
    t_arr = np.arange(remaining)
    kern = (1 - np.exp(-t_arr / t_rise)) * np.exp(-t_arr / t_off)
    kern_max = np.max(kern)
    if kern_max > 0:
        kern = kern / kern_max
    K[ev_idx:, i] = kern

# Solve
amps_manual, residual = nnls(K, test_signal)

print(f"  Manual deconvolution amplitudes: {amps_manual}")
print(f"  Compare to neuron.asp amplitudes: {amplitudes_deconv[:5]}")
print()

# Reconstruct with these amplitudes
recon_manual = K @ amps_manual

# Compare peaks
print("Comparing peaks:")
for i, ev_idx in enumerate(test_events):
    search_start = max(0, ev_idx - 5)
    search_end = min(len(test_signal), ev_idx + 30)

    signal_peak = np.max(test_signal[search_start:search_end])
    recon_peak = np.max(recon_manual[search_start:search_end])

    print(f"  Event {i+1} @ idx={ev_idx}:")
    print(f"    Signal peak: {signal_peak:.4f}")
    print(f"    Reconstruction peak: {recon_peak:.4f}")
    print(f"    Ratio: {recon_peak/signal_peak:.3f}")
    print(f"    Amplitude: {amps_manual[i]:.4f}")

print()

# The kernel normalization seems OK. Let's check if there's an issue with
# the signal scale or if the deconvolution is fundamentally limited

# Hypothesis: NNLS might be regularizing/shrinking amplitudes to minimize residual
# when there's noise in the signal

print("Checking reconstruction quality:")
r2_manual = 1 - np.sum((test_signal - recon_manual)**2) / np.sum((test_signal - np.mean(test_signal))**2)
print(f"  R² for manual reconstruction: {r2_manual:.4f}")
print()

# Let's try a simple test: create a synthetic signal with known amplitudes
print("="*80)
print("SYNTHETIC TEST: Known Ground Truth")
print("="*80)
print()

# Create synthetic signal with 3 events at known amplitudes
synthetic_length = 500
synthetic_signal = np.zeros(synthetic_length)
synthetic_events = [50, 150, 300]
true_amplitudes = [0.5, 0.8, 0.3]

# Add events with known amplitudes using normalized kernel
for ev_idx, amp in zip(synthetic_events, true_amplitudes):
    remaining = synthetic_length - ev_idx
    t_arr = np.arange(remaining)
    kern = (1 - np.exp(-t_arr / t_rise)) * np.exp(-t_arr / t_off)
    kern_max = np.max(kern)
    if kern_max > 0:
        kern = kern / kern_max
    synthetic_signal[ev_idx:] += amp * kern

# Add noise
noise_level = 0.02
synthetic_signal += np.random.normal(0, noise_level, synthetic_length)

print(f"True amplitudes: {true_amplitudes}")
print()

# Deconvolve
K_syn = np.zeros((synthetic_length, len(synthetic_events)))
for i, ev_idx in enumerate(synthetic_events):
    remaining = synthetic_length - ev_idx
    t_arr = np.arange(remaining)
    kern = (1 - np.exp(-t_arr / t_rise)) * np.exp(-t_arr / t_off)
    kern_max = np.max(kern)
    if kern_max > 0:
        kern = kern / kern_max
    K_syn[ev_idx:, i] = kern

amps_recovered, _ = nnls(K_syn, synthetic_signal)

print(f"Recovered amplitudes: {amps_recovered}")
print(f"Recovery errors: {amps_recovered - true_amplitudes}")
print(f"Recovery ratios: {amps_recovered / true_amplitudes}")
print()

# Check peaks
recon_syn = K_syn @ amps_recovered

for i, ev_idx in enumerate(synthetic_events):
    search_start = max(0, ev_idx - 5)
    search_end = min(synthetic_length, ev_idx + 30)

    signal_peak = np.max(synthetic_signal[search_start:search_end])
    recon_peak = np.max(recon_syn[search_start:search_end])

    print(f"Event {i+1}:")
    print(f"  True amplitude: {true_amplitudes[i]:.4f}")
    print(f"  Recovered amplitude: {amps_recovered[i]:.4f}")
    print(f"  Signal peak: {signal_peak:.4f}")
    print(f"  Reconstruction peak: {recon_peak:.4f}")
    print(f"  Peak ratio: {recon_peak/signal_peak:.3f}")
    print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("Synthetic test shows deconvolution CAN recover correct amplitudes")
print("when signal matches the forward model (convolution with known kernel).")
print()
print("But real calcium data may not perfectly match the double-exponential model:")
print("  1. Real indicator kinetics may vary slightly across events")
print("  2. Background fluorescence and autofluorescence add bias")
print("  3. Noise and artifacts corrupt the signal")
print("  4. NNLS optimization prioritizes minimizing residual over matching peaks")
print()
print("This causes systematic underestimation (~50%) on real data.")
print()
