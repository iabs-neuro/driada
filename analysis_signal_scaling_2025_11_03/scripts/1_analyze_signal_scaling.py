#!/usr/bin/env python3
"""
Analyze signal-level scaling bias in reconstruction.

This script investigates whether poor R² is due to systematic amplitude
scaling bias that can be corrected with a global scaling factor.

CRITICAL: This analyzes SIGNAL-LEVEL reconstruction (calcium traces),
NOT spike amplitudes!
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project source to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root / "src"))

from driada.experiment.neuron import Neuron

def load_real_neuron(neuron_id=192):
    """Load real calcium imaging data."""
    data_path = Path("/Users/nikita/PycharmProjects/driada2/temp") / "wavelet_analysis" / "real calcium.npz"
    data = np.load(data_path, allow_pickle=True)
    ca_traces = data['arr_0']  # Shape: (663 neurons, 11836 timepoints)
    
    print(f"Loaded data shape: {ca_traces.shape}")
    print(f"Loading neuron #{neuron_id}")
    
    ca_trace = ca_traces[neuron_id]
    fps = 20.0  # From metadata
    
    # Create Neuron object (sp=None for real data without ground truth)
    neuron = Neuron(
        cell_id=f"real_neuron_{neuron_id}",
        ca=ca_trace,
        sp=None,
        fps=fps
    )
    
    return neuron

def reconstruct_and_get_signals(neuron):
    """
    Run reconstruction and extract both observed and reconstructed signals.
    
    Returns:
        observed: Original calcium signal
        reconstructed: Reconstructed calcium signal
        asp: Amplitude spike point process (spike train)
    """
    # Run reconstruction (wavelet + deconvolution + iterative, n_iter=2)
    result = neuron.reconstruct_spikes(
        method='wavelet',
        deconv_method='deconvolution',
        iterative=True,
        n_iter=2,
        create_event_regions=True,
        verbose=True
    )
    
    # Get observed signal
    observed = neuron.ca.data.copy()
    
    # Get reconstructed signal (this is the convolved spike train)
    reconstructed = neuron.reconstructed.data.copy()
    
    # Get spike train
    asp = neuron.asp.data.copy()
    
    # Count spikes
    n_spikes = np.sum(asp > 0)
    spike_indices = np.where(asp > 0)[0]
    amplitudes = asp[spike_indices]
    
    print(f"  Spikes detected: {n_spikes}")
    print(f"  Mean amplitude: {np.mean(amplitudes):.4f}")
    
    return observed, reconstructed, asp, amplitudes

def calculate_three_r2_metrics(observed, reconstructed):
    """
    Calculate three R² metrics on signals.
    
    Returns:
        raw_r2: Standard R² (current metric)
        scaled_r2: R² after optimal scaling
        corr_r2: Correlation coefficient squared
        alpha: Optimal scaling factor
    """
    # Raw R²
    ss_res = np.sum((observed - reconstructed)**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    raw_r2 = 1 - ss_res / ss_tot
    
    # Optimal scaling factor α
    # Minimize: sum((observed - α × reconstructed)²)
    # Solution: α = sum(observed × reconstructed) / sum(reconstructed²)
    alpha = np.sum(observed * reconstructed) / np.sum(reconstructed**2)
    
    # Scaled R²
    scaled_reconstructed = alpha * reconstructed
    ss_res_scaled = np.sum((observed - scaled_reconstructed)**2)
    scaled_r2 = 1 - ss_res_scaled / ss_tot
    
    # Correlation²
    corr, _ = pearsonr(observed, reconstructed)
    corr_r2 = corr**2
    
    return raw_r2, scaled_r2, corr_r2, alpha

def analyze_peak_ratios(observed, reconstructed, events_mask):
    """
    Analyze peak ratios in event regions.
    
    Returns:
        peak_ratios: Array of observed/reconstructed peak ratios
        median_ratio: Median ratio
        std_ratio: Standard deviation of ratios
    """
    if events_mask is None:
        print("WARNING: No events mask available, skipping peak analysis")
        return None, None, None
    
    # Find event regions (contiguous blocks of True values)
    events_bool = events_mask.astype(bool)
    
    # Find starts and ends of events
    diff = np.diff(events_bool.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if events_bool[0]:
        starts = np.concatenate([[0], starts])
    if events_bool[-1]:
        ends = np.concatenate([ends, [len(events_bool)]])
    
    peak_ratios = []
    for start, end in zip(starts, ends):
        obs_peak = np.max(observed[start:end])
        rec_peak = np.max(reconstructed[start:end])
        
        if rec_peak > 1e-10:  # Avoid division by zero
            ratio = obs_peak / rec_peak
            peak_ratios.append(ratio)
    
    if len(peak_ratios) == 0:
        print("WARNING: No valid peak ratios found")
        return None, None, None
    
    peak_ratios = np.array(peak_ratios)
    median_ratio = np.median(peak_ratios)
    std_ratio = np.std(peak_ratios)
    
    return peak_ratios, median_ratio, std_ratio

def analyze_single_neuron(neuron_id):
    """Analyze signal scaling for a single neuron."""
    print(f"\n{'='*70}")
    print(f"ANALYZING NEURON #{neuron_id}")
    print(f"{'='*70}")
    
    # Load data
    neuron = load_real_neuron(neuron_id)
    print(f"Neuron: {neuron.cell_id}")
    print(f"Duration: {neuron.n_frames / neuron.fps:.1f}s")
    print(f"FPS: {neuron.fps}")
    
    # Reconstruct
    print("\nRunning reconstruction...")
    observed, reconstructed, asp, amplitudes = reconstruct_and_get_signals(neuron)
    
    # Get events mask
    events_mask = neuron.events.data if hasattr(neuron, 'events') else None
    
    # Calculate three R² metrics
    raw_r2, scaled_r2, corr_r2, alpha = calculate_three_r2_metrics(observed, reconstructed)
    
    print(f"\n{'='*50}")
    print(f"R² METRICS (SIGNAL-LEVEL)")
    print(f"{'='*50}")
    print(f"Raw R²:         {raw_r2:.4f}")
    print(f"Scaled R²:      {scaled_r2:.4f}  (improvement: {scaled_r2 - raw_r2:+.4f})")
    print(f"Correlation²:   {corr_r2:.4f}")
    print(f"Optimal α:      {alpha:.4f}")
    print(f"{'='*50}")
    
    # Analyze peak ratios
    peak_ratios, median_ratio, std_ratio = analyze_peak_ratios(observed, reconstructed, events_mask)
    
    if peak_ratios is not None:
        print(f"\n{'='*50}")
        print(f"PEAK RATIO ANALYSIS")
        print(f"{'='*50}")
        print(f"Number of events: {len(peak_ratios)}")
        print(f"Median ratio:     {median_ratio:.4f}")
        print(f"Std ratio:        {std_ratio:.4f}")
        print(f"CV (std/median):  {std_ratio/median_ratio:.4f}")
        print(f"Min ratio:        {np.min(peak_ratios):.4f}")
        print(f"Max ratio:        {np.max(peak_ratios):.4f}")
        print(f"{'='*50}")
    
    # Calculate improvement metrics
    improvement_pct = ((scaled_r2 - raw_r2) / raw_r2) * 100 if raw_r2 > 0 else 0
    
    results = {
        'neuron_id': neuron_id,
        'raw_r2': raw_r2,
        'scaled_r2': scaled_r2,
        'corr_r2': corr_r2,
        'alpha': alpha,
        'improvement': scaled_r2 - raw_r2,
        'improvement_pct': improvement_pct,
        'n_spikes': len(amplitudes),
        'n_events': len(peak_ratios) if peak_ratios is not None else 0,
        'median_peak_ratio': median_ratio,
        'std_peak_ratio': std_ratio,
        'cv_peak_ratio': std_ratio / median_ratio if median_ratio else None,
        'mean_amplitude': np.mean(amplitudes)
    }
    
    return results, observed, reconstructed, peak_ratios, asp

def main():
    """Main analysis function."""
    # Analyze primary neuron
    results, observed, reconstructed, peak_ratios, asp = analyze_single_neuron(192)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics
    df = pd.DataFrame([results])
    df.to_csv(output_dir / "neuron_192_metrics.csv", index=False)
    print(f"\nSaved metrics to: {output_dir / 'neuron_192_metrics.csv'}")
    
    # Save signals
    np.savez(
        output_dir / "neuron_192_signals.npz",
        observed=observed,
        reconstructed=reconstructed,
        asp=asp,
        peak_ratios=peak_ratios if peak_ratios is not None else np.array([]),
        alpha=results['alpha']
    )
    print(f"Saved signals to: {output_dir / 'neuron_192_signals.npz'}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")
    
    if results['improvement'] > 0.05:
        print(f"✓ SIGNIFICANT IMPROVEMENT: Scaled R² improves by {results['improvement']:.4f}")
        print(f"  ({results['improvement_pct']:.1f}% relative improvement)")
        print(f"  This suggests SYSTEMATIC scaling bias exists.")
        print(f"  Optimal scaling factor α = {results['alpha']:.4f}")
        
        if abs(results['alpha'] - 2.0) < 0.5:
            print(f"\n  ✓ α ≈ 2.0 confirms the inverse of 0.49 ratio (as expected)")
        elif abs(results['alpha'] - 1.0) < 0.1:
            print(f"\n  ✓ α ≈ 1.0 suggests reconstruction already scaled correctly")
        
        if results['cv_peak_ratio'] is not None:
            cv = results['cv_peak_ratio']
            if cv < 0.3:
                print(f"\n  ✓ Peak ratio CV = {cv:.2f} < 0.3 → CONSISTENT scaling bias")
                print(f"  RECOMMENDATION: Apply global scaling correction α = {results['alpha']:.2f}")
            else:
                print(f"\n  ✗ Peak ratio CV = {cv:.2f} > 0.3 → VARIABLE errors")
                print(f"  RECOMMENDATION: Investigate per-event scaling or other error sources")
    else:
        print(f"✗ MINIMAL IMPROVEMENT: Scaled R² improves by only {results['improvement']:.4f}")
        print(f"  ({results['improvement_pct']:.1f}% relative improvement)")
        print(f"  Errors are too variable for global scaling fix.")
        print(f"  RECOMMENDATION: Investigate other error sources (timing, kinetics, etc.)")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
