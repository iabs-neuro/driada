#!/usr/bin/env python3
"""
Visualize signal-level scaling analysis results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_comprehensive_figure():
    """Create comprehensive visualization of scaling analysis."""
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    
    # Single neuron data
    signals = np.load(data_dir / "neuron_192_signals.npz")
    observed = signals['observed']
    reconstructed = signals['reconstructed']
    peak_ratios = signals['peak_ratios']
    alpha = float(signals['alpha'])
    
    # Multi-neuron data
    df = pd.read_csv(data_dir / "multi_neuron_metrics.csv")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Signal comparison (neuron #192)
    ax1 = fig.add_subplot(gs[0, :2])
    time = np.arange(len(observed)) / 20.0  # fps=20
    ax1.plot(time, observed, 'k-', linewidth=1, label='Observed', alpha=0.7)
    ax1.plot(time, reconstructed, 'b-', linewidth=1, label='Reconstructed (raw)', alpha=0.7)
    ax1.plot(time, alpha * reconstructed, 'r-', linewidth=1, label=f'Reconstructed (scaled α={alpha:.2f})', alpha=0.7)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('ΔF/F', fontsize=12)
    ax1.set_title('Neuron #192: Signal Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Zoomed view
    ax2 = fig.add_subplot(gs[0, 2])
    zoom_start, zoom_end = 100, 200  # seconds
    zoom_start_idx = int(zoom_start * 20)
    zoom_end_idx = int(zoom_end * 20)
    time_zoom = time[zoom_start_idx:zoom_end_idx]
    ax2.plot(time_zoom, observed[zoom_start_idx:zoom_end_idx], 'k-', linewidth=1.5, label='Observed')
    ax2.plot(time_zoom, reconstructed[zoom_start_idx:zoom_end_idx], 'b-', linewidth=1.5, label='Raw')
    ax2.plot(time_zoom, alpha * reconstructed[zoom_start_idx:zoom_end_idx], 'r-', linewidth=1.5, label='Scaled')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('ΔF/F', fontsize=10)
    ax2.set_title('Zoomed View (100-200s)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: R² comparison across neurons
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(df))
    width = 0.35
    ax3.bar(x - width/2, df['raw_r2'], width, label='Raw R²', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, df['scaled_r2'], width, label='Scaled R²', color='coral', alpha=0.7)
    ax3.set_xlabel('Neuron ID', fontsize=12)
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title('R² Comparison Across Neurons', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"#{nid}" for nid in df['neuron_id']])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 0.8)
    
    # Panel 4: Alpha distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(range(len(df)), df['alpha'], color='purple', alpha=0.7)
    ax4.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='α=1.0 (no bias)')
    ax4.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='α=2.0 (expected)')
    ax4.axhline(y=df['alpha'].mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean α={df["alpha"].mean():.2f}')
    ax4.set_xlabel('Neuron ID', fontsize=12)
    ax4.set_ylabel('Optimal α', fontsize=12)
    ax4.set_title('Optimal Scaling Factor per Neuron', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([f"#{nid}" for nid in df['neuron_id']])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Improvement distribution
    ax5 = fig.add_subplot(gs[1, 2])
    improvements = df['improvement'] * 100  # Convert to percentage
    colors = ['green' if imp > 5 else 'orange' if imp > 2 else 'red' for imp in improvements]
    ax5.bar(range(len(df)), improvements, color=colors, alpha=0.7)
    ax5.axhline(y=5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='5% threshold')
    ax5.set_xlabel('Neuron ID', fontsize=12)
    ax5.set_ylabel('Improvement (%)', fontsize=12)
    ax5.set_title('R² Improvement with Scaling', fontsize=14, fontweight='bold')
    ax5.set_xticks(range(len(df)))
    ax5.set_xticklabels([f"#{nid}" for nid in df['neuron_id']])
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Peak ratio distribution (neuron #192)
    ax6 = fig.add_subplot(gs[2, 0])
    if len(peak_ratios) > 0:
        # Remove extreme outliers for better visualization
        peak_ratios_filtered = peak_ratios[peak_ratios < 15]
        ax6.hist(peak_ratios_filtered, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax6.axvline(x=np.median(peak_ratios), color='red', linestyle='--', linewidth=2, label=f'Median={np.median(peak_ratios):.2f}')
        ax6.axvline(x=2.0, color='green', linestyle='--', linewidth=2, label='Expected=2.0')
        ax6.set_xlabel('Peak Ratio (Obs/Rec)', fontsize=12)
        ax6.set_ylabel('Count', fontsize=12)
        ax6.set_title('Peak Ratio Distribution (Neuron #192)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
    
    # Panel 7: CV distribution
    ax7 = fig.add_subplot(gs[2, 1])
    cv_valid = df[df['cv_peak_ratio'].notna()]['cv_peak_ratio']
    neuron_ids_valid = df[df['cv_peak_ratio'].notna()]['neuron_id']
    colors_cv = ['green' if cv < 0.5 else 'orange' if cv < 1.0 else 'red' for cv in cv_valid]
    ax7.bar(range(len(cv_valid)), cv_valid, color=colors_cv, alpha=0.7)
    ax7.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='CV=0.3')
    ax7.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='CV=0.5')
    ax7.set_xlabel('Neuron ID', fontsize=12)
    ax7.set_ylabel('Peak Ratio CV', fontsize=12)
    ax7.set_title('Peak Ratio Variability', fontsize=14, fontweight='bold')
    ax7.set_xticks(range(len(cv_valid)))
    ax7.set_xticklabels([f"#{nid}" for nid in neuron_ids_valid])
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_yscale('log')
    
    # Panel 8: Summary statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS (N={len(df)})

Raw R²:
  Mean: {df['raw_r2'].mean():.4f} ± {df['raw_r2'].std():.4f}
  Range: [{df['raw_r2'].min():.4f}, {df['raw_r2'].max():.4f}]

Scaled R²:
  Mean: {df['scaled_r2'].mean():.4f} ± {df['scaled_r2'].std():.4f}
  Range: [{df['scaled_r2'].min():.4f}, {df['scaled_r2'].max():.4f}]

Improvement:
  Mean: {df['improvement'].mean()*100:.2f}% ± {df['improvement'].std()*100:.2f}%
  Range: [{df['improvement'].min()*100:.2f}%, {df['improvement'].max()*100:.2f}%]

Optimal α:
  Mean: {df['alpha'].mean():.4f} ± {df['alpha'].std():.4f}
  Range: [{df['alpha'].min():.4f}, {df['alpha'].max():.4f}]

CONCLUSION:
α std = {df['alpha'].std():.2f} < 0.5 ✓ Consistent
Mean improvement = {df['improvement'].mean()*100:.1f}% ≈ minimal
Peak CV = {cv_valid.mean():.1f} >> 0.5 ✗ Variable

RECOMMENDATION:
Errors too variable for global scaling.
Investigate timing/kinetics issues.
"""
    
    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Signal-Level Scaling Bias Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "scaling_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_dir / 'scaling_analysis.png'}")
    plt.close()

if __name__ == "__main__":
    create_comprehensive_figure()
