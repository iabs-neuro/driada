#!/usr/bin/env python
"""Plot with higher spike rates to see more activity."""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity

# Try different rate parameters
fig, all_axes = plt.subplots(3, 1, figsize=(15, 12))

rate_configs = [
    {'rate_0': 0.1, 'rate_1': 2.0, 'title': 'Original rates (0.1 Hz baseline, 2.0 Hz peak)'},
    {'rate_0': 0.5, 'rate_1': 10.0, 'title': 'Higher rates (0.5 Hz baseline, 10.0 Hz peak)'},
    {'rate_0': 1.0, 'rate_1': 20.0, 'title': 'Very high rates (1.0 Hz baseline, 20.0 Hz peak)'}
]

for config_idx, config in enumerate(rate_configs):
    # Generate experiment with different rates
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=2,  # Just 2 for clearer visualization
        n_continuous_feats=0,
        n_neurons=5,
        duration=60,  # 1 minute
        fps=20,
        selectivity_prob=1.0,
        multi_select_prob=1.0,  # All have mixed selectivity
        weights_mode='equal',
        skip_prob=0.0,
        rate_0=config['rate_0'],
        rate_1=config['rate_1'],
        ampl_range=(0.5, 2.0),
        noise_std=0.005,
        seed=42,
        verbose=False
    )
    
    # Find a neuron with mixed selectivity to both features
    mixed_neuron = None
    for n in range(exp.n_cells):
        if np.sum(selectivity_info['matrix'][:, n] > 0) == 2:
            mixed_neuron = n
            break
    
    if mixed_neuron is None:
        continue
        
    time = np.arange(exp.n_frames) / exp.fps
    
    # Create subplot
    ax = all_axes[config_idx]
    
    # Plot features
    colors = ['red', 'green']
    for i in range(2):
        feat_data = exp.dynamic_features[f'd_feat_{i}'].data
        ax.fill_between(time, -0.5 - i*0.6, -0.5 - i*0.6 + feat_data*0.5, 
                       alpha=0.5, color=colors[i], label=f'd_feat_{i}')
    
    # Plot neural signal
    neural_signal = exp.neurons[mixed_neuron].ca.data
    ax.plot(time, neural_signal, 'b-', linewidth=0.8, label=f'Neuron {mixed_neuron} (Ca signal)')
    
    # Count spikes (peaks in calcium signal)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(neural_signal, height=0.1, distance=5)
    n_spikes = len(peaks)
    spike_rate = n_spikes / (exp.n_frames / exp.fps)
    
    ax.scatter(time[peaks], neural_signal[peaks], c='orange', s=30, marker='v', label=f'Detected spikes (n={n_spikes})')
    
    ax.set_ylabel('Signal')
    ax.set_title(f"{config['title']} - Spike rate: {spike_rate:.1f} Hz")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if config_idx == 2:
        ax.set_xlabel('Time (s)')

plt.suptitle('Effect of Spike Rates on Mixed Selectivity Signals', fontsize=16)
plt.tight_layout()
plt.savefig('spike_rate_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Also create a detailed view with the higher rate
print("\nGenerating detailed view with rate_1=10.0 Hz...")

exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=4,
    n_continuous_feats=0,
    n_neurons=10,
    duration=120,  # 2 minutes
    fps=20,
    selectivity_prob=1.0,
    multi_select_prob=0.8,
    weights_mode='equal',
    skip_prob=0.0,
    rate_0=0.5,
    rate_1=10.0,  # Higher peak rate
    ampl_range=(0.5, 2.0),
    noise_std=0.005,
    seed=42,
    verbose=False
)

# Find neurons with mixed selectivity
mixed_neurons = np.where(np.sum(selectivity_info['matrix'] > 0, axis=0) >= 2)[0]

# Create detailed plot
fig, axes = plt.subplots(6, 1, figsize=(15, 10), sharex=True)
time = np.arange(exp.n_frames) / exp.fps

# Plot features
for i in range(4):
    feat_data = exp.dynamic_features[f'd_feat_{i}'].data
    axes[i].plot(time, feat_data, 'k-', linewidth=0.5)
    axes[i].fill_between(time, 0, feat_data, alpha=0.3)
    axes[i].set_ylabel(f'd_feat_{i}')
    axes[i].set_ylim(-0.1, 1.1)

# Plot two mixed neurons
for idx in range(2):
    if idx < len(mixed_neurons):
        neuron_id = mixed_neurons[idx]
        ax = axes[4 + idx]
        
        neural_signal = exp.neurons[neuron_id].ca.data
        ax.plot(time, neural_signal, 'b-', linewidth=0.8)
        
        # Find and mark spikes
        peaks, _ = find_peaks(neural_signal, height=0.1, distance=5)
        ax.scatter(time[peaks], neural_signal[peaks], c='red', s=20, marker='v')
        
        # Get selectivity info
        selective_features = np.where(selectivity_info['matrix'][:, neuron_id] > 0)[0]
        feat_names = [f'd_feat_{i}' for i in selective_features]
        
        ax.set_ylabel(f'Neuron {neuron_id}')
        ax.text(0.02, 0.95, f"Selective to: {feat_names}\nSpikes: {len(peaks)}", 
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

axes[-1].set_xlabel('Time (s)')
axes[-1].set_xlim(20, 50)  # Show 30 second window

plt.suptitle('Mixed Selectivity with Higher Spike Rate (10 Hz peak)', fontsize=14)
plt.tight_layout()
plt.savefig('mixed_selectivity_higher_rate.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plots saved as 'spike_rate_comparison.png' and 'mixed_selectivity_higher_rate.png'")