"""
Basic Neuron Usage Example
===========================

This script demonstrates the core functionality of the Neuron class:
1. Generating synthetic calcium signals
2. Creating a Neuron object
3. Reconstructing spikes with wavelet method
4. Optimizing calcium kinetics
5. Computing quality metrics (wavelet SNR, reconstruction R²)

Run this script to see a complete workflow on synthetic data.
"""

import numpy as np
from driada.experiment.neuron import Neuron
from driada.utils.neural import generate_pseudo_calcium_signal

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("NEURON BASIC USAGE EXAMPLE")
print("=" * 70)

# =============================================================================
# Step 1: Generate Synthetic Calcium Signal
# =============================================================================
print("\n1. Generating synthetic calcium signal...")

signal = generate_pseudo_calcium_signal(
    duration=100.0,              # Signal duration in seconds
    sampling_rate=20.0,          # Sampling rate (Hz)
    event_rate=0.15,             # Average event rate (Hz)
    amplitude_range=(1.0, 3.0),  # Event amplitude range (dF/F0)
    decay_time=2.0,              # Calcium decay time constant (seconds)
    rise_time=0.25,              # Calcium rise time constant (seconds)
    noise_std=0.05,              # Additive Gaussian noise level
    kernel='double_exponential'  # Realistic calcium kernel
)

print(f"   ✓ Generated signal: {len(signal)} frames ({len(signal)/20:.1f} seconds)")

# =============================================================================
# Step 2: Create Neuron Object
# =============================================================================
print("\n2. Creating Neuron object...")

neuron = Neuron(
    cell_id='example_neuron',
    ca=signal,              # Calcium signal
    sp=None,                # No ground-truth spikes (will be reconstructed)
    fps=20.0                # Sampling rate
)

print(f"   ✓ Neuron created: {neuron.cell_id}")
print(f"   ✓ Signal length: {neuron.n_frames} frames")
print(f"   ✓ Sampling rate: {neuron.fps} Hz")

# =============================================================================
# Step 3: Reconstruct Spikes with Wavelet Method
# =============================================================================
print("\n3. Reconstructing spikes using wavelet method...")

spikes = neuron.reconstruct_spikes(
    method='wavelet',
    create_event_regions=True  # Create event regions for quality metrics
)

n_events = int(np.sum(neuron.asp.data > 0))
print(f"   ✓ Detected {n_events} calcium events")
print(f"   ✓ Spike train stored in neuron.sp")
print(f"   ✓ Amplitude spikes stored in neuron.asp")

# =============================================================================
# Step 4: Optimize Calcium Kinetics
# =============================================================================
print("\n4. Optimizing calcium kinetics...")

kinetics = neuron.get_kinetics(
    method='direct',           # Direct measurement from detected events
    use_cached=False          # Force recomputation
)

print(f"   ✓ Optimized rise time (t_rise): {kinetics['t_rise']:.3f} seconds")
print(f"   ✓ Optimized decay time (t_off): {kinetics['t_off']:.3f} seconds")
print(f"   ✓ Events used: {kinetics['n_events_detected']}")

# =============================================================================
# Step 5: Calculate Wavelet SNR
# =============================================================================
print("\n5. Computing wavelet SNR...")

wavelet_snr = neuron.get_wavelet_snr()

print(f"   ✓ Wavelet SNR: {wavelet_snr:.2f}")
print(f"     (Ratio of event amplitude to baseline noise)")

# =============================================================================
# Step 6: Calculate Reconstruction Quality Metrics
# =============================================================================
print("\n6. Computing reconstruction quality metrics...")

# R² (coefficient of determination)
r2 = neuron.get_reconstruction_r2()
print(f"   ✓ Reconstruction R²: {r2:.4f}")
print(f"     (1.0 = perfect, >0.7 = good quality)")

# Event-only R² (focuses on event regions)
r2_events = neuron.get_reconstruction_r2(event_only=True)
print(f"   ✓ Event-only R²: {r2_events:.4f}")
print(f"     (Quality in event regions only)")

# Normalized RMSE
nrmse = neuron.get_nrmse()
print(f"   ✓ Normalized RMSE: {nrmse:.4f}")
print(f"     (Lower is better)")

# Normalized MAE
nmae = neuron.get_nmae()
print(f"   ✓ Normalized MAE: {nmae:.4f}")
print(f"     (Lower is better)")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Neuron ID:           {neuron.cell_id}")
print(f"Signal duration:     {neuron.n_frames / neuron.fps:.1f} seconds")
print(f"Events detected:     {n_events}")
print(f"Optimized t_rise:    {kinetics['t_rise']:.3f} s")
print(f"Optimized t_off:     {kinetics['t_off']:.3f} s")
print(f"Wavelet SNR:         {wavelet_snr:.2f}")
print(f"Reconstruction R²:   {r2:.4f}")
print(f"Event-only R²:       {r2_events:.4f}")
print("=" * 70)

print("\n✓ Example completed successfully!")
