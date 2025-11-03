#!/usr/bin/env python3
"""
Analyze signal-level scaling bias across multiple neurons.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project source to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root / "src"))

from driada.experiment.neuron import Neuron
from scipy.stats import pearsonr

def load_and_analyze_neuron(neuron_id):
    """Load and analyze a single neuron."""
    print(f"\n{'='*60}")
    print(f"NEURON #{neuron_id}")
    print(f"{'='*60}")
    
    # Load data
    data_path = Path("/Users/nikita/PycharmProjects/driada2/temp") / "wavelet_analysis" / "real calcium.npz"
    data = np.load(data_path, allow_pickle=True)
    ca_traces = data['arr_0']
    ca_trace = ca_traces[neuron_id]
    fps = 20.0
    
    # Create neuron
    neuron = Neuron(
        cell_id=f"real_neuron_{neuron_id}",
        ca=ca_trace,
        sp=None,
        fps=fps
    )
    
    # Reconstruct
    try:
        neuron.reconstruct_spikes(
            method='wavelet',
            deconv_method='deconvolution',
            iterative=True,
            n_iter=2,
            create_event_regions=True,
            verbose=False
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return None
    
    # Get signals
    observed = neuron.ca.data.copy()
    reconstructed = neuron.reconstructed.data.copy()
    asp = neuron.asp.data.copy()
    
    n_spikes = np.sum(asp > 0)
    amplitudes = asp[asp > 0]
    
    # Calculate R² metrics
    ss_res = np.sum((observed - reconstructed)**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    raw_r2 = 1 - ss_res / ss_tot
    
    alpha = np.sum(observed * reconstructed) / np.sum(reconstructed**2)
    scaled_reconstructed = alpha * reconstructed
    ss_res_scaled = np.sum((observed - scaled_reconstructed)**2)
    scaled_r2 = 1 - ss_res_scaled / ss_tot
    
    corr, _ = pearsonr(observed, reconstructed)
    corr_r2 = corr**2
    
    # Peak ratio analysis
    events_mask = neuron.events.data if hasattr(neuron, 'events') else None
    peak_ratios = None
    median_ratio = None
    std_ratio = None
    
    if events_mask is not None:
        events_bool = events_mask.astype(bool)
        diff = np.diff(events_bool.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if events_bool[0]:
            starts = np.concatenate([[0], starts])
        if events_bool[-1]:
            ends = np.concatenate([ends, [len(events_bool)]])
        
        peak_ratios = []
        for start, end in zip(starts, ends):
            obs_peak = np.max(observed[start:end])
            rec_peak = np.max(reconstructed[start:end])
            if rec_peak > 1e-10:
                peak_ratios.append(obs_peak / rec_peak)
        
        if len(peak_ratios) > 0:
            peak_ratios = np.array(peak_ratios)
            median_ratio = np.median(peak_ratios)
            std_ratio = np.std(peak_ratios)
    
    print(f"Raw R²:         {raw_r2:.4f}")
    print(f"Scaled R²:      {scaled_r2:.4f} (+{scaled_r2-raw_r2:.4f})")
    print(f"Corr²:          {corr_r2:.4f}")
    print(f"Alpha:          {alpha:.4f}")
    print(f"Spikes:         {n_spikes}")
    print(f"Events:         {len(peak_ratios) if peak_ratios is not None else 0}")
    if median_ratio is not None:
        print(f"Median ratio:   {median_ratio:.4f}")
        print(f"CV ratio:       {std_ratio/median_ratio:.4f}")
    
    improvement_pct = ((scaled_r2 - raw_r2) / raw_r2) * 100 if raw_r2 > 0 else 0
    
    return {
        'neuron_id': neuron_id,
        'raw_r2': raw_r2,
        'scaled_r2': scaled_r2,
        'corr_r2': corr_r2,
        'alpha': alpha,
        'improvement': scaled_r2 - raw_r2,
        'improvement_pct': improvement_pct,
        'n_spikes': n_spikes,
        'n_events': len(peak_ratios) if peak_ratios is not None else 0,
        'median_peak_ratio': median_ratio,
        'std_peak_ratio': std_ratio,
        'cv_peak_ratio': std_ratio / median_ratio if median_ratio else None,
        'mean_amplitude': np.mean(amplitudes) if len(amplitudes) > 0 else 0
    }

def main():
    """Main multi-neuron analysis."""
    # Analyze multiple neurons
    neuron_ids = [192, 33, 146, 288]
    
    results = []
    for nid in neuron_ids:
        result = load_and_analyze_neuron(nid)
        if result is not None:
            results.append(result)
    
    # Save combined results
    df = pd.DataFrame(results)
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "multi_neuron_metrics.csv", index=False)
    print(f"\n{'='*60}")
    print(f"Saved results to: {output_dir / 'multi_neuron_metrics.csv'}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Number of neurons: {len(df)}")
    print(f"\nRaw R²:")
    print(f"  Mean:   {df['raw_r2'].mean():.4f} ± {df['raw_r2'].std():.4f}")
    print(f"  Range:  {df['raw_r2'].min():.4f} - {df['raw_r2'].max():.4f}")
    print(f"\nScaled R²:")
    print(f"  Mean:   {df['scaled_r2'].mean():.4f} ± {df['scaled_r2'].std():.4f}")
    print(f"  Range:  {df['scaled_r2'].min():.4f} - {df['scaled_r2'].max():.4f}")
    print(f"\nImprovement:")
    print(f"  Mean:   {df['improvement'].mean():.4f} ± {df['improvement'].std():.4f}")
    print(f"  Range:  {df['improvement'].min():.4f} - {df['improvement'].max():.4f}")
    print(f"\nOptimal α:")
    print(f"  Mean:   {df['alpha'].mean():.4f} ± {df['alpha'].std():.4f}")
    print(f"  Range:  {df['alpha'].min():.4f} - {df['alpha'].max():.4f}")
    print(f"\nPeak ratio CV:")
    valid_cv = df[df['cv_peak_ratio'].notna()]['cv_peak_ratio']
    if len(valid_cv) > 0:
        print(f"  Mean:   {valid_cv.mean():.4f} ± {valid_cv.std():.4f}")
        print(f"  Range:  {valid_cv.min():.4f} - {valid_cv.max():.4f}")
    print(f"{'='*60}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("CROSS-NEURON INTERPRETATION")
    print(f"{'='*60}")
    
    mean_improvement = df['improvement'].mean()
    std_alpha = df['alpha'].std()
    mean_cv = valid_cv.mean() if len(valid_cv) > 0 else None
    
    if mean_improvement > 0.05:
        print(f"✓ Mean improvement: {mean_improvement:.4f} (SIGNIFICANT)")
        print(f"  Scaling helps across neurons")
    else:
        print(f"✗ Mean improvement: {mean_improvement:.4f} (MINIMAL)")
        print(f"  Scaling doesn't help much")
    
    if std_alpha < 0.5:
        print(f"\n✓ α std: {std_alpha:.4f} < 0.5 (CONSISTENT)")
        print(f"  Same scaling factor works across neurons")
        print(f"  RECOMMENDATION: Use α = {df['alpha'].mean():.2f} globally")
    else:
        print(f"\n✗ α std: {std_alpha:.4f} > 0.5 (VARIABLE)")
        print(f"  Optimal scaling varies per neuron")
        print(f"  RECOMMENDATION: Per-neuron calibration needed")
    
    if mean_cv is not None:
        if mean_cv < 0.5:
            print(f"\n✓ Mean CV: {mean_cv:.4f} < 0.5 (CONSISTENT)")
            print(f"  Peak ratios are relatively consistent")
        else:
            print(f"\n✗ Mean CV: {mean_cv:.4f} > 0.5 (VARIABLE)")
            print(f"  Peak ratios vary widely within neurons")
            print(f"  This suggests timing/kinetics issues, not just scaling")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
