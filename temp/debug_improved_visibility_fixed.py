#!/usr/bin/env python
"""Improved visualization with better feature visibility - fixed version."""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity
import os

os.makedirs('temp', exist_ok=True)

# Generate with rolled-back parameters
duration = 600
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
    noise_std=0.02,  # Lower noise
    seed=242,
    verbose=False
)

print(f"Generated experiment: {exp.n_frames} frames = {exp.n_frames/exp.fps} seconds")

# Get neurons with mixed selectivity
selectivity_matrix = info['matrix']
feature_names = info['feature_names']
n_features_per_neuron = np.sum(selectivity_matrix > 0, axis=0)
two_feature_neurons = np.where(n_features_per_neuron == 2)[0][:3]  # Take 3 neurons

# Create improved visualization
fig = plt.figure(figsize=(20, 12))

# Define feature colors
colors = {'d_feat_0': '#FF6B6B', 'd_feat_1': '#4ECDC4', 'd_feat_2': '#45B7D1'}

# Show first 60 seconds with detailed view
zoom_duration = 60
zoom_frames = int(zoom_duration * exp.fps)
time_zoom = np.arange(zoom_frames) / exp.fps

for idx, neuron_id in enumerate(two_feature_neurons):
    # Top subplot: Feature activity bars
    ax1 = plt.subplot(len(two_feature_neurons)*2, 1, idx*2 + 1)
    
    # Get features this neuron responds to
    selective_features = np.where(selectivity_matrix[:, neuron_id] > 0)[0]
    
    # Plot each feature as a separate row
    for i, feat_idx in enumerate(selective_features):
        feat_name = feature_names[feat_idx]
        feat_data = exp.dynamic_features[feat_name].data[:zoom_frames]
        weight = selectivity_matrix[feat_idx, neuron_id]
        
        # Create bar plot for feature activity
        y_pos = i * 0.4
        for t in range(len(feat_data)):
            if feat_data[t] > 0:
                ax1.bar(time_zoom[t], 0.3, width=1/exp.fps, bottom=y_pos, 
                       color=colors[feat_name], alpha=0.8, edgecolor='none')
        
        # Label
        ax1.text(-2, y_pos + 0.15, f'{feat_name}\n(w={weight:.2f})', 
                ha='right', va='center', fontsize=10, weight='bold', color=colors[feat_name])
    
    ax1.set_ylim(-0.1, len(selective_features) * 0.4)
    ax1.set_xlim(0, zoom_duration)
    ax1.set_ylabel(f'Neuron {neuron_id}\nFeatures', fontsize=11)
    ax1.set_xticklabels([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_yticks([])
    
    if idx == 0:
        ax1.set_title('Feature Activity (Binary)', fontsize=12, pad=10)
    
    # Bottom subplot: Calcium signal with shaded regions
    ax2 = plt.subplot(len(two_feature_neurons)*2, 1, idx*2 + 2, sharex=ax1)
    
    # Get calcium signal
    calcium_signal = exp.neurons[neuron_id].ca.data[:zoom_frames]
    
    # Plot calcium
    ax2.plot(time_zoom, calcium_signal, 'k-', linewidth=1.5, alpha=0.9, zorder=10)
    
    # Add baseline
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline', zorder=5)
    
    # Add shaded regions for each feature
    for feat_idx in selective_features:
        feat_name = feature_names[feat_idx]
        feat_data = exp.dynamic_features[feat_name].data[:zoom_frames]
        weight = selectivity_matrix[feat_idx, neuron_id]
        
        # Find continuous regions where feature is active
        active = feat_data > 0
        for start in range(len(active)):
            if active[start] and (start == 0 or not active[start-1]):
                end = start
                while end < len(active) and active[end]:
                    end += 1
                
                # Shade the region
                ax2.axvspan(time_zoom[start], time_zoom[end-1], 
                           alpha=0.3 * weight,  # Transparency based on weight
                           color=colors[feat_name], 
                           zorder=1)
    
    # Show which features this neuron responds to (ground truth)
    gt_features = [feature_names[i] for i in selective_features]
    gt_weights = selectivity_matrix[selective_features, neuron_id]
    gt_text = "Ground truth: " + ", ".join([f"{feat} (w={w:.2f})" for feat, w in zip(gt_features, gt_weights)])
    
    ax2.text(0.02, 0.95, gt_text, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Formatting
    ax2.set_ylabel('Ca²⁺ Signal', fontsize=11)
    ax2.set_ylim(np.min(calcium_signal) - 0.5, np.max(calcium_signal) + 0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    if idx == len(two_feature_neurons) - 1:
        ax2.set_xlabel('Time (s)', fontsize=11)
    else:
        ax2.set_xticklabels([])

plt.suptitle(f'Mixed Selectivity Visualization (First {zoom_duration}s of {duration}s)\n' + 
             'Parameters: rate_0=0.5, rate_1=5.0, noise=0.02, ds=5', fontsize=14)
plt.tight_layout()
plt.savefig('temp/debug_improved_visibility.png', dpi=150, bbox_inches='tight')
plt.close()

# Create a second plot showing feature statistics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Feature activity percentages
ax = axes[0, 0]
feature_active_pcts = []
for feat_name in feature_names:
    feat_data = exp.dynamic_features[feat_name].data
    active_pct = np.sum(feat_data > 0) / len(feat_data) * 100
    feature_active_pcts.append(active_pct)

bars = ax.bar(feature_names, feature_active_pcts, color=[colors[fn] for fn in feature_names])
ax.set_ylabel('Active Time (%)')
ax.set_title('Feature Activity')
for bar, pct in zip(bars, feature_active_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{pct:.1f}%', ha='center', va='bottom')

# Plot 2: Feature overlap matrix
ax = axes[0, 1]
overlap_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        feat_i = exp.dynamic_features[feature_names[i]].data
        feat_j = exp.dynamic_features[feature_names[j]].data
        if i == j:
            overlap_matrix[i, j] = np.sum(feat_i > 0) / len(feat_i) * 100
        else:
            overlap_matrix[i, j] = np.sum((feat_i > 0) & (feat_j > 0)) / len(feat_i) * 100

im = ax.imshow(overlap_matrix, cmap='Blues')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(feature_names)
ax.set_yticklabels(feature_names)
ax.set_title('Feature Overlap (%)')

# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{overlap_matrix[i, j]:.1f}%',
                      ha='center', va='center', color='black' if overlap_matrix[i, j] < 50 else 'white')

# Plot 3: Example calcium response to each feature
ax = axes[1, 0]

# Find a neuron that responds strongly to each feature
example_neurons = {}
for feat_idx, feat_name in enumerate(feature_names):
    # Find neuron with highest weight for this feature
    weights = selectivity_matrix[feat_idx, :]
    best_neuron = np.argmax(weights)
    if weights[best_neuron] > 0:
        example_neurons[feat_name] = (best_neuron, weights[best_neuron])

# Plot example responses
time_example = np.arange(600) / exp.fps  # 30 seconds
for i, (feat_name, (neuron_id, weight)) in enumerate(example_neurons.items()):
    feat_data = exp.dynamic_features[feat_name].data[:600]
    ca_data = exp.neurons[neuron_id].ca.data[:600]
    
    # Normalize calcium for display
    ca_normalized = (ca_data - np.min(ca_data)) / (np.max(ca_data) - np.min(ca_data)) * 0.8 + i
    feat_normalized = feat_data * 0.8 + i
    
    ax.plot(time_example, ca_normalized, color=colors[feat_name], linewidth=1.5, alpha=0.8)
    ax.fill_between(time_example, i, feat_normalized, color=colors[feat_name], alpha=0.2)
    ax.text(-1, i + 0.4, f'{feat_name}\nN{neuron_id}', ha='right', va='center', 
            fontsize=9, color=colors[feat_name])

ax.set_xlim(0, 30)
ax.set_xlabel('Time (s)')
ax.set_title('Example Calcium Responses')
ax.set_yticks([])
ax.spines['left'].set_visible(False)

# Plot 4: Summary statistics
ax = axes[1, 1]
ax.text(0.1, 0.9, 'Detection Challenge:', fontsize=12, weight='bold', transform=ax.transAxes)
summary_text = f"""
Feature Statistics:
• Active time: ~5% each
• Overlap: 0.0% (never co-occur)
• Avg duration: ~0.5 seconds

Signal Parameters:
• Baseline rate: 0.5 Hz
• Active rate: 5.0 Hz  
• Noise: 0.02
• Downsampling: 5x

Key Issue:
Sparse, non-overlapping features make
it difficult to detect mixed selectivity.
MI analysis often detects only the
dominant feature per neuron.
"""
ax.text(0.05, 0.05, summary_text, transform=ax.transAxes, 
        verticalalignment='bottom', fontsize=10, family='monospace')
ax.axis('off')

plt.tight_layout()
plt.savefig('temp/debug_feature_statistics.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlots saved:")
print("- temp/debug_improved_visibility.png")
print("- temp/debug_feature_statistics.png")