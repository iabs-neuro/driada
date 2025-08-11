#!/usr/bin/env python
"""Visualize full signals with feature activity shown as colored vertical stripes."""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity
from driada.intense.pipelines import compute_cell_feat_significance
import os

os.makedirs('temp', exist_ok=True)

# Generate with test parameters
duration = 2000  # Full duration
exp, info = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=3,
    n_continuous_feats=0,
    n_neurons=20,
    duration=duration,
    fps=20,
    selectivity_prob=1.0,
    multi_select_prob=0.8,
    weights_mode='dominant',
    rate_0=0.5,
    rate_1=5.0,
    skip_prob=0.0,
    noise_std=0.1,
    seed=242,
    verbose=False
)

print(f"Generated experiment with {exp.n_frames} frames")
print(f"Duration: {exp.n_frames / exp.fps} seconds")

# Get selectivity info
selectivity_matrix = info['matrix']
feature_names = info['feature_names']
n_features_per_neuron = np.sum(selectivity_matrix > 0, axis=0)
two_feature_neurons = np.where(n_features_per_neuron == 2)[0]

# Run detection
test_neurons = two_feature_neurons[:4].tolist()
stats, sig, _, results = compute_cell_feat_significance(
    exp,
    cell_bunch=test_neurons,
    mode='two_stage',
    n_shuffles_stage1=50,
    n_shuffles_stage2=200,
    metric='mi',
    metric_distr_type='norm',
    pval_thr=0.1,
    multicomp_correction=None,
    with_disentanglement=False,
    find_optimal_delays=False,
    allow_mixed_dimensions=True,
    ds=5,
    enable_parallelization=False,
    verbose=False,
    seed=42
)

# Create full signal plot with vertical stripes
fig, axes = plt.subplots(len(test_neurons), 1, figsize=(20, 3*len(test_neurons)))
if len(test_neurons) == 1:
    axes = [axes]

# Define colors for features
feature_colors = {
    'd_feat_0': 'red',
    'd_feat_1': 'blue', 
    'd_feat_2': 'green'
}

# Time axis for full duration
time = np.arange(exp.n_frames) / exp.fps

for idx, neuron_id in enumerate(test_neurons):
    ax = axes[idx]
    
    # Get calcium signal
    calcium_signal = exp.neurons[neuron_id].ca.data
    
    # Plot calcium signal
    ax.plot(time, calcium_signal, 'k-', linewidth=0.5, alpha=0.8, zorder=5)
    
    # Add baseline line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    # Get features this neuron responds to
    selective_features = np.where(selectivity_matrix[:, neuron_id] > 0)[0]
    
    # Add vertical stripes for each feature
    for feat_idx in selective_features:
        feat_name = feature_names[feat_idx]
        feat_data = exp.dynamic_features[feat_name].data
        weight = selectivity_matrix[feat_idx, neuron_id]
        color = feature_colors[feat_name]
        
        # Find regions where feature is active
        active_regions = feat_data > 0
        
        # Create vertical stripes
        for start_idx in range(len(active_regions)):
            if active_regions[start_idx] and (start_idx == 0 or not active_regions[start_idx-1]):
                # Found start of active region
                end_idx = start_idx
                while end_idx < len(active_regions) and active_regions[end_idx]:
                    end_idx += 1
                
                # Add vertical stripe
                ax.axvspan(time[start_idx], time[end_idx-1], 
                          alpha=0.2 * weight,  # Alpha proportional to weight
                          color=color, 
                          zorder=1)
        
        # Add to legend
        ax.plot([], [], color=color, linewidth=4, alpha=0.5, 
                label=f'{feat_name} (w={weight:.2f})')
    
    # Add detection results
    neuron_sig = sig.get(neuron_id, {})
    detected = [feat for feat, sig_info in neuron_sig.items() 
                if isinstance(sig_info, dict) and sig_info.get('stage2', False)]
    
    # Set labels and title
    ax.set_ylabel(f'Neuron {neuron_id}\nCa²⁺')
    ax.set_title(f'Detected: {detected}', loc='right')
    
    # Set reasonable y-limits
    y_min = np.percentile(calcium_signal, 1)
    y_max = np.percentile(calcium_signal, 99.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    
    # Legend
    ax.legend(loc='upper left', fontsize=8)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # X-axis
    if idx == len(test_neurons) - 1:
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xticklabels([])

# Overall title
fig.suptitle(f'Full Signal Visualization (Duration: {duration}s, rate_0=0.5, rate_1=5.0, noise=0.1)', 
             fontsize=14)

plt.tight_layout()
plt.savefig('temp/debug_full_signal_stripes.png', dpi=100, bbox_inches='tight')
plt.close()

# Also create a zoomed version (first 200 seconds)
fig, axes = plt.subplots(len(test_neurons), 1, figsize=(16, 3*len(test_neurons)))
if len(test_neurons) == 1:
    axes = [axes]

zoom_frames = int(200 * exp.fps)  # 200 seconds
time_zoom = time[:zoom_frames]

for idx, neuron_id in enumerate(test_neurons):
    ax = axes[idx]
    
    calcium_signal = exp.neurons[neuron_id].ca.data[:zoom_frames]
    ax.plot(time_zoom, calcium_signal, 'k-', linewidth=1, alpha=0.8, zorder=5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    selective_features = np.where(selectivity_matrix[:, neuron_id] > 0)[0]
    
    for feat_idx in selective_features:
        feat_name = feature_names[feat_idx]
        feat_data = exp.dynamic_features[feat_name].data[:zoom_frames]
        weight = selectivity_matrix[feat_idx, neuron_id]
        color = feature_colors[feat_name]
        
        active_regions = feat_data > 0
        
        for start_idx in range(len(active_regions)):
            if active_regions[start_idx] and (start_idx == 0 or not active_regions[start_idx-1]):
                end_idx = start_idx
                while end_idx < len(active_regions) and active_regions[end_idx]:
                    end_idx += 1
                
                ax.axvspan(time_zoom[start_idx], time_zoom[end_idx-1], 
                          alpha=0.2 * weight,
                          color=color, 
                          zorder=1)
        
        ax.plot([], [], color=color, linewidth=4, alpha=0.5, 
                label=f'{feat_name} (w={weight:.2f})')
    
    neuron_sig = sig.get(neuron_id, {})
    detected = [feat for feat, sig_info in neuron_sig.items() 
                if isinstance(sig_info, dict) and sig_info.get('stage2', False)]
    
    ax.set_ylabel(f'Neuron {neuron_id}\nCa²⁺')
    ax.set_title(f'Detected: {detected}', loc='right')
    
    y_min = np.percentile(calcium_signal, 1)
    y_max = np.percentile(calcium_signal, 99.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, 200)
    
    if idx == len(test_neurons) - 1:
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xticklabels([])

fig.suptitle('Zoomed View: First 200 seconds', fontsize=14)
plt.tight_layout()
plt.savefig('temp/debug_signal_stripes_zoom.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPlots saved:")
print(f"- temp/debug_full_signal_stripes.png (full {duration}s duration)")
print(f"- temp/debug_signal_stripes_zoom.png (first 200s zoomed)")