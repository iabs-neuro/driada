#!/usr/bin/env python
"""Debug mixed selectivity detection with lower baseline rate."""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity
from driada.intense.pipelines import compute_cell_feat_significance
import os

os.makedirs('temp', exist_ok=True)

# Generate with lower baseline rate
print("Generating synthetic data with lower baseline rate...")
exp, info = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=3,
    n_continuous_feats=0,
    n_neurons=20,
    duration=600,
    fps=20,
    selectivity_prob=1.0,
    multi_select_prob=0.8,
    weights_mode='equal',  # Changed from 'dominant' to 'equal' for better detection
    rate_0=0.05,  # Much lower baseline rate (was 0.5)
    rate_1=5.0,   # High active rate
    skip_prob=0.0,
    noise_std=0.02,
    seed=42,
    verbose=False
)

print(f"Generated experiment: {exp.n_frames} frames = {exp.n_frames/exp.fps} seconds")

# Get neurons with mixed selectivity
selectivity_matrix = info['matrix']
feature_names = info['feature_names']
n_features_per_neuron = np.sum(selectivity_matrix > 0, axis=0)
two_feature_neurons = np.where(n_features_per_neuron == 2)[0][:5]

print(f"\nFound {len(two_feature_neurons)} neurons with 2 features")
print(f"Test neurons: {two_feature_neurons}")

# Run analysis
print("\nRunning MI analysis...")
stats, sig, _, _, disent_results = compute_cell_feat_significance(
    exp,
    cell_bunch=two_feature_neurons.tolist(),
    mode='two_stage',
    n_shuffles_stage1=50,
    n_shuffles_stage2=200,
    metric='mi',
    metric_distr_type='norm',
    pval_thr=0.1,
    multicomp_correction=None,
    with_disentanglement=True,
    find_optimal_delays=False,
    allow_mixed_dimensions=True,
    ds=5,
    enable_parallelization=False,
    verbose=False,
    seed=42
)

# Analyze results
print("\n=== DETECTION RESULTS ===")
neurons_with_multi = 0
for neuron_id in two_feature_neurons:
    neuron_sig = sig.get(neuron_id, {})
    sig_features = [feat for feat, sig_info in neuron_sig.items() 
                   if isinstance(sig_info, dict) and sig_info.get('stage2', False)]
    
    ground_truth = np.where(selectivity_matrix[:, neuron_id] > 0)[0]
    gt_features = [feature_names[i] for i in ground_truth]
    
    print(f"\nNeuron {neuron_id}:")
    print(f"  Ground truth: {gt_features}")
    print(f"  Detected: {sig_features}")
    
    if len(sig_features) >= 2:
        neurons_with_multi += 1

print(f"\nTotal neurons with detected multi-selectivity: {neurons_with_multi}/{len(two_feature_neurons)}")

# Create visualization
fig = plt.figure(figsize=(20, 14))

# Show first 3 neurons
for idx in range(min(3, len(two_feature_neurons))):
    neuron_id = two_feature_neurons[idx]
    
    # Top subplot: Feature activity
    ax1 = plt.subplot(3, 2, idx*2 + 1)
    
    # Get features this neuron responds to
    selective_features = np.where(selectivity_matrix[:, neuron_id] > 0)[0]
    
    # Define colors
    colors = {'d_feat_0': '#FF6B6B', 'd_feat_1': '#4ECDC4', 'd_feat_2': '#45B7D1'}
    
    # Show first 60 seconds
    time_window = 60
    frames = int(time_window * exp.fps)
    time = np.arange(frames) / exp.fps
    
    # Plot features
    for i, feat_idx in enumerate(selective_features):
        feat_name = feature_names[feat_idx]
        feat_data = exp.dynamic_features[feat_name].data[:frames]
        weight = selectivity_matrix[feat_idx, neuron_id]
        
        # Plot feature as bars
        y_pos = i * 0.4
        for t in range(len(feat_data)):
            if feat_data[t] > 0:
                ax1.bar(time[t], 0.3, width=1/exp.fps, bottom=y_pos,
                       color=colors[feat_name], alpha=0.8, edgecolor='none')
        
        ax1.text(-2, y_pos + 0.15, f'{feat_name}\n(w={weight:.2f})',
                ha='right', va='center', fontsize=10, weight='bold', color=colors[feat_name])
    
    ax1.set_ylim(-0.1, len(selective_features) * 0.4)
    ax1.set_xlim(0, time_window)
    ax1.set_ylabel(f'Neuron {neuron_id}\nFeatures', fontsize=11)
    ax1.set_title(f'Feature Activity (First {time_window}s)', fontsize=12)
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    
    # Bottom subplot: Calcium signal
    ax2 = plt.subplot(3, 2, idx*2 + 2, sharex=ax1)
    
    # Get calcium signal
    calcium_signal = exp.neurons[neuron_id].ca.data[:frames]
    
    # Plot calcium
    ax2.plot(time, calcium_signal, 'k-', linewidth=1.5, alpha=0.9)
    
    # Add baseline reference
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero')
    
    # Add shaded regions for features
    for feat_idx in selective_features:
        feat_name = feature_names[feat_idx]
        feat_data = exp.dynamic_features[feat_name].data[:frames]
        weight = selectivity_matrix[feat_idx, neuron_id]
        
        # Shade active regions
        active = feat_data > 0
        for start in range(len(active)):
            if active[start] and (start == 0 or not active[start-1]):
                end = start
                while end < len(active) and active[end]:
                    end += 1
                ax2.axvspan(time[start], time[end-1],
                           alpha=0.3 * weight,
                           color=colors[feat_name],
                           zorder=1)
    
    # Show detection results
    neuron_sig = sig.get(neuron_id, {})
    detected = [feat for feat, sig_info in neuron_sig.items() 
                if isinstance(sig_info, dict) and sig_info.get('stage2', False)]
    
    gt_features = [feature_names[i] for i in selective_features]
    gt_weights = selectivity_matrix[selective_features, neuron_id]
    
    text = f"Ground truth: {', '.join([f'{f}({w:.2f})' for f, w in zip(gt_features, gt_weights)])}\n"
    text += f"Detected: {', '.join(detected) if detected else 'None'}"
    
    ax2.text(0.02, 0.95, text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax2.set_ylabel('Ca²⁺ Signal', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add calcium statistics
    ca_mean = np.mean(calcium_signal)
    ca_std = np.std(calcium_signal)
    ca_max = np.max(calcium_signal)
    ax2.text(0.98, 0.02, f'μ={ca_mean:.2f}, σ={ca_std:.2f}, max={ca_max:.2f}',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=9)

plt.suptitle(f'Mixed Selectivity Detection (rate_0={0.05}, rate_1={5.0}, noise={0.02})', fontsize=14)
plt.tight_layout()
plt.savefig('temp/debug_mixed_selectivity_detection.png', dpi=150, bbox_inches='tight')
plt.close()

# Create statistics plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Feature activity percentages
ax = axes[0, 0]
feature_active_pcts = []
for feat_name in feature_names:
    feat_data = exp.dynamic_features[feat_name].data
    active_pct = np.sum(feat_data > 0) / len(feat_data) * 100
    feature_active_pcts.append(active_pct)

colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(feature_names, feature_active_pcts, color=colors_list)
ax.set_ylabel('Active Time (%)')
ax.set_title('Feature Activity')
ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Target 5%')
for bar, pct in zip(bars, feature_active_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{pct:.1f}%', ha='center', va='bottom')
ax.legend()

# Plot 2: Signal-to-noise ratio for each neuron
ax = axes[0, 1]
snr_values = []
for neuron_id in range(exp.n_cells):
    signal = exp.neurons[neuron_id].ca.data
    baseline = np.percentile(signal, 10)  # 10th percentile as baseline
    peak_90 = np.percentile(signal, 90)   # 90th percentile as peak
    noise = np.std(signal[signal < np.median(signal)])  # Noise from lower half
    if noise > 0:
        snr = (peak_90 - baseline) / noise
        snr_values.append(snr)

ax.hist(snr_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
ax.set_xlabel('Signal-to-Noise Ratio')
ax.set_ylabel('Number of Neurons')
ax.set_title('SNR Distribution')
ax.axvline(x=np.mean(snr_values), color='red', linestyle='--', 
          label=f'Mean={np.mean(snr_values):.1f}')
ax.legend()

# Plot 3: MI values distribution
ax = axes[1, 0]
all_mi_values = []
for neuron_id in two_feature_neurons:
    neuron_stats = stats.get(neuron_id, {})
    for feat_name, feat_stats in neuron_stats.items():
        if isinstance(feat_stats, dict) and 'me' in feat_stats:
            all_mi_values.append(feat_stats['me'])

if all_mi_values:
    ax.hist(all_mi_values, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Count')
    ax.set_title('MI Values Distribution')
    ax.axvline(x=np.mean(all_mi_values), color='red', linestyle='--',
              label=f'Mean={np.mean(all_mi_values):.4f}')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No MI values computed', ha='center', va='center', transform=ax.transAxes)

# Plot 4: Detection summary
ax = axes[1, 1]
ax.text(0.1, 0.9, 'Detection Summary:', fontsize=12, weight='bold', transform=ax.transAxes)

summary_text = f"""
Parameters:
• Baseline rate: 0.05 Hz (lowered from 0.5)
• Active rate: 5.0 Hz
• Noise: 0.02
• Downsampling: 5x
• Duration: 600s

Feature Statistics:
• Active time: {np.mean(feature_active_pcts):.1f}% average
• Feature overlap: minimal

Detection Results:
• Neurons tested: {len(two_feature_neurons)}
• Multi-selectivity detected: {neurons_with_multi}
• Success rate: {neurons_with_multi/len(two_feature_neurons)*100:.0f}%

Signal Quality:
• Mean SNR: {np.mean(snr_values):.1f}
• Mean MI: {np.mean(all_mi_values) if all_mi_values else 0:.4f}
"""

ax.text(0.05, 0.05, summary_text, transform=ax.transAxes,
        verticalalignment='bottom', fontsize=10, family='monospace')
ax.axis('off')

plt.tight_layout()
plt.savefig('temp/debug_mixed_selectivity_statistics.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlots saved:")
print("- temp/debug_mixed_selectivity_detection.png")
print("- temp/debug_mixed_selectivity_statistics.png")

# If disentanglement was performed, show the count matrix
if disent_results and 'count_matrix' in disent_results:
    count_matrix = disent_results['count_matrix']
    if np.sum(count_matrix) > 0:
        plt.figure(figsize=(8, 6))
        plt.imshow(count_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Count')
        plt.xlabel('Feature 2')
        plt.ylabel('Feature 1')
        plt.title('Disentanglement Count Matrix')
        for i in range(count_matrix.shape[0]):
            for j in range(count_matrix.shape[1]):
                plt.text(j, i, str(int(count_matrix[i, j])),
                        ha='center', va='center',
                        color='white' if count_matrix[i, j] > np.max(count_matrix)/2 else 'black')
        plt.tight_layout()
        plt.savefig('temp/debug_disentanglement_matrix.png', dpi=150)
        plt.close()
        print("- temp/debug_disentanglement_matrix.png")