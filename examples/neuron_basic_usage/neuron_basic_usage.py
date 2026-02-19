"""
Basic Neuron Usage Example
===========================

This script demonstrates the core functionality of the Neuron class:
1. Generating synthetic calcium signals
2. Creating a Neuron object
3. Reconstructing spikes with wavelet method
4. Optimizing calcium kinetics
5. Computing quality metrics (wavelet SNR, reconstruction R2)

Run this script to see a complete workflow on synthetic data.
"""

import numpy as np
from driada.experiment.neuron import Neuron
from driada.experiment.synthetic import generate_pseudo_calcium_signal

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
    duration=200.0,              # Signal duration in seconds
    sampling_rate=20.0,          # Sampling rate (Hz)
    event_rate=0.15,             # Average event rate (Hz)
    amplitude_range=(1.0, 2.0),  # Event amplitude range (dF/F0)
    decay_time=1.5,              # Calcium decay time constant (seconds)
    rise_time=0.15,              # Calcium rise time constant (seconds)
    noise_std=0.05,              # Additive Gaussian noise level
    kernel='double_exponential'  # Realistic calcium kernel
)

print(f"   [OK] Generated signal: {len(signal)} frames ({len(signal)/20:.1f} seconds)")

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

print(f"   [OK] Neuron created: {neuron.cell_id}")
print(f"   [OK] Signal length: {neuron.n_frames} frames")
print(f"   [OK] Sampling rate: {neuron.fps} Hz")

# =============================================================================
# Step 3: Reconstruct Spikes with Wavelet Method
# =============================================================================
print("\n3. Reconstructing spikes using wavelet method...")

spikes = neuron.reconstruct_spikes(
    method='wavelet',
    create_event_regions=True  # Create event regions for quality metrics
)

n_events = int(np.sum(neuron.asp.data > 0))
print(f"   [OK] Detected {n_events} calcium events")
print(f"   [OK] Spike train stored in neuron.sp")
print(f"   [OK] Amplitude spikes stored in neuron.asp")

# =============================================================================
# Step 4: Optimize Calcium Kinetics
# =============================================================================
print("\n4. Optimizing calcium kinetics...")

kinetics = neuron.get_kinetics(
    method='direct',           # Direct measurement from detected events
    use_cached=False          # Force recomputation
)

print(f"   [OK] Optimized rise time (t_rise): {kinetics['t_rise']:.3f} seconds")
print(f"   [OK] Optimized decay time (t_off): {kinetics['t_off']:.3f} seconds")
print(f"   [OK] Events used: {kinetics['n_events_detected']}")

# =============================================================================
# Step 5: Calculate Wavelet SNR
# =============================================================================
print("\n5. Computing wavelet SNR...")

wavelet_snr = neuron.get_wavelet_snr()

print(f"   [OK] Wavelet SNR: {wavelet_snr:.2f}")
print(f"       (Ratio of event amplitude to baseline noise)")

# =============================================================================
# Step 6: Calculate Reconstruction Quality Metrics
# =============================================================================
print("\n6. Computing reconstruction quality metrics...")

# R2 (coefficient of determination)
r2 = neuron.get_reconstruction_r2()
print(f"   [OK] Reconstruction R2: {r2:.4f}")
print(f"       (1.0 = perfect, >0.7 = good quality)")

# Event-only R2 (focuses on event regions)
r2_events = neuron.get_reconstruction_r2(event_only=True)
print(f"   [OK] Event-only R2: {r2_events:.4f}")
print(f"       (Quality in event regions only)")

# Normalized RMSE
nrmse = neuron.get_nrmse()
print(f"   [OK] Normalized RMSE: {nrmse:.4f}")
print(f"       (Lower is better)")

# Normalized MAE
nmae = neuron.get_nmae()
print(f"   [OK] Normalized MAE: {nmae:.4f}")
print(f"       (Lower is better)")

# =============================================================================
# Step 7: Surrogate Generation Methods
# =============================================================================
print("\n7. Surrogate generation methods...")
print("   Three calcium surrogate types and one spike surrogate type.")

# --- Calcium surrogates ---

# 7a. Roll-based: circular shift preserving all autocorrelations
shuffled_roll = neuron.get_shuffled_calcium(method='roll_based', seed=42)
print(f"\n   [Roll-based] Circular shift surrogate:")
print(f"       Mean: {np.mean(shuffled_roll):.4f}  (original: {np.mean(neuron.ca.data):.4f})")
print(f"       Std:  {np.std(shuffled_roll):.4f}  (original: {np.std(neuron.ca.data):.4f})")
print(f"       Preserves: autocorrelation structure, amplitude distribution")

# 7b. Waveform-based: shuffle detected spike times, reconstruct calcium
shuffled_wf = neuron.get_shuffled_calcium(method='waveform_based', seed=42)
print(f"\n   [Waveform-based] Spike-shuffle + reconstruct surrogate:")
print(f"       Mean: {np.mean(shuffled_wf):.4f}  (original: {np.mean(neuron.ca.data):.4f})")
print(f"       Std:  {np.std(shuffled_wf):.4f}  (original: {np.std(neuron.ca.data):.4f})")
print(f"       Preserves: individual waveform shapes, event count")

# 7c. Chunks-based: divide signal into chunks and reorder
shuffled_chunks = neuron.get_shuffled_calcium(method='chunks_based', seed=42)
print(f"\n   [Chunks-based] Chunk reordering surrogate:")
print(f"       Mean: {np.mean(shuffled_chunks):.4f}  (original: {np.mean(neuron.ca.data):.4f})")
print(f"       Std:  {np.std(shuffled_chunks):.4f}  (original: {np.std(neuron.ca.data):.4f})")
print(f"       Preserves: local structure within chunks")

# --- Spike surrogates ---

# 7d. ISI-based: shuffle inter-spike intervals, preserving ISI distribution
shuffled_sp = neuron.get_shuffled_spikes(method='isi_based', seed=42)
original_spike_count = int(np.sum(neuron.sp.data > 0))
shuffled_spike_count = int(np.sum(shuffled_sp > 0))
print(f"\n   [ISI-based] Spike train surrogate:")
print(f"       Spike count: {shuffled_spike_count}  (original: {original_spike_count})")
print(f"       Preserves: inter-spike interval distribution")

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
print(f"Reconstruction R2:   {r2:.4f}")
print(f"Event-only R2:       {r2_events:.4f}")
print("=" * 70)

print("\n[OK] Example completed successfully!")
