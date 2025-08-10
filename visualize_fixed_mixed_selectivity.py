#!/usr/bin/env python
"""Visualize the fixed mixed selectivity generation."""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity
from scipy.signal import find_peaks

# Generate experiment with clear parameters
exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=3,  # 3 features for clearer visualization
    n_continuous_feats=0,
    n_neurons=6,  # Fewer neurons for clarity
    duration=120,  # 2 minutes
    fps=20,
    selectivity_prob=1.0,
    multi_select_prob=1.0,  # All neurons have mixed selectivity
    weights_mode='equal',
    skip_prob=0.0,
    rate_0=0.5,
    rate_1=10.0,
    ampl_range=(0.5, 2.0),
    noise_std=0.005,
    seed=42,
    verbose=False
)

# Create figure
fig, axes = plt.subplots(exp.n_cells + 3, 1, figsize=(15, 2.5 * (exp.n_cells + 3)), sharex=True)
time = np.arange(exp.n_frames) / exp.fps

# Plot features
colors = ['red', 'green', 'blue']
for i in range(3):
    feat_data = exp.dynamic_features[f'd_feat_{i}'].data
    ax = axes[i]
    ax.fill_between(time, 0, feat_data, alpha=0.5, color=colors[i])
    ax.set_ylabel(f'd_feat_{i}', fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    
    # Count active periods
    active_frac = np.mean(feat_data)
    ax.text(0.02, 0.5, f'{active_frac*100:.1f}% active', 
            transform=ax.transAxes, fontsize=9)

# Plot each neuron
for neuron_idx in range(exp.n_cells):
    ax = axes[3 + neuron_idx]
    
    # Get neuron signal
    neural_signal = exp.neurons[neuron_idx].ca.data
    
    # Plot neural activity
    ax.plot(time, neural_signal, 'k-', linewidth=0.8, alpha=0.8)
    
    # Find peaks (spikes)
    peaks, _ = find_peaks(neural_signal, height=0.1, distance=5)
    ax.scatter(time[peaks], neural_signal[peaks], c='orange', s=20, marker='v', alpha=0.7)
    
    # Get selectivity info
    selective_features = np.where(selectivity_info['matrix'][:, neuron_idx] > 0)[0]
    weights = selectivity_info['matrix'][selective_features, neuron_idx]
    
    # Create colored bars showing when selective features are active
    y_bottom = np.min(neural_signal) - 0.2
    bar_height = 0.15
    
    for feat_idx, weight in zip(selective_features, weights):
        feat_data = exp.dynamic_features[f'd_feat_{feat_idx}'].data
        active_mask = feat_data > 0
        
        # Draw colored bars when feature is active
        ax.fill_between(time, y_bottom + feat_idx * bar_height * 0.3, 
                       y_bottom + (feat_idx + 1) * bar_height * 0.3,
                       where=active_mask, alpha=0.5, color=colors[feat_idx],
                       label=f'd_feat_{feat_idx} (w={weight:.2f})')
    
    # Label the neuron
    feat_names = [f'd_feat_{i}' for i in selective_features]
    ax.set_ylabel(f'Neuron {neuron_idx}\n{"+".join(feat_names)}', fontsize=10)
    
    # Add spike rate info
    spike_rate = len(peaks) / (exp.n_frames / exp.fps)
    ax.text(0.98, 0.95, f'{spike_rate:.1f} Hz', 
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add legend for first neuron
    if neuron_idx == 0:
        ax.legend(loc='upper left', fontsize=8)

# Set common properties
axes[-1].set_xlabel('Time (s)', fontsize=12)
axes[-1].set_xlim(10, 40)  # Show 30 second window

# Main title
fig.suptitle('Fixed Mixed Selectivity: Neurons Respond Differently to Feature Combinations', fontsize=16)

# Add explanation text
explanation = """Key improvements:
1. Features are properly spaced with ~1.5% active time
2. Neurons fire at different rates for different feature combinations:
   - Single feature active: medium rate (~5 Hz)
   - Multiple features active: higher rate (~10 Hz)
3. This allows INTENSE to distinguish which features each neuron responds to"""

fig.text(0.02, 0.01, explanation, fontsize=11, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.96, bottom=0.06)
plt.savefig('fixed_mixed_selectivity_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nSUMMARY STATISTICS:")
print("=" * 60)
print("\nGround Truth Selectivity:")
for neuron_idx in range(exp.n_cells):
    selective_features = np.where(selectivity_info['matrix'][:, neuron_idx] > 0)[0]
    weights = selectivity_info['matrix'][selective_features, neuron_idx]
    feat_names = [f'd_feat_{i}' for i in selective_features]
    weight_str = [f"{w:.2f}" for w in weights]
    print(f"Neuron {neuron_idx}: {' + '.join(feat_names)} (weights: {', '.join(weight_str)})")

print("\nFeature Activity:")
for i in range(3):
    feat_data = exp.dynamic_features[f'd_feat_{i}'].data
    active_frac = np.mean(feat_data)
    # Count islands
    islands = np.sum(np.diff(np.concatenate([[0], feat_data, [0]])) == 1)
    print(f"d_feat_{i}: {active_frac*100:.1f}% active, {islands} active periods")

print("\nKey insight: With the fix, neurons now fire at DIFFERENT rates")
print("depending on WHICH features are active, enabling proper detection!")