#!/usr/bin/env python
"""Final debug to understand the detection issue."""

import numpy as np
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity
from driada.intense.pipelines import compute_cell_feat_significance

# Generate experiment
exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=4,
    n_continuous_feats=0,
    n_neurons=20,
    duration=600,
    fps=20,
    selectivity_prob=1.0,
    multi_select_prob=0.8,
    weights_mode='equal',
    skip_prob=0.0,
    rate_0=0.5,
    rate_1=10.0,
    ampl_range=(0.5, 2.0),
    noise_std=0.005,
    seed=42,
    verbose=False
)

mixed_neurons = np.where(np.sum(selectivity_info['matrix'] > 0, axis=0) >= 2)[0]
cell_bunch = mixed_neurons[:5].tolist()  # Just test 5 neurons

print(f"Testing neurons: {cell_bunch}")

# Run with very permissive parameters
stats, significance, info, results = compute_cell_feat_significance(
    exp,
    cell_bunch=cell_bunch,
    feat_bunch=None,
    mode='stage1',
    n_shuffles_stage1=5,  # Very few shuffles
    metric='mi',
    metric_distr_type='norm',
    pval_thr=0.1,  # Very permissive
    multicomp_correction=None,
    find_optimal_delays=False,
    allow_mixed_dimensions=True,
    enable_parallelization=False,
    verbose=True,
    seed=42
)

# Debug: print what we got
print("\nSignificance results:")
for cell_id in cell_bunch[:2]:  # Just show first 2
    if cell_id in significance:
        print(f"\nNeuron {cell_id}:")
        for feat_name, sig_val in significance[cell_id].items():
            print(f"  {feat_name}: {sig_val}")

# Store results in experiment for disentanglement
exp.stats_tables['calcium'] = stats

# Create significance tables in the format expected by experiment
exp.significance_tables = {'calcium': {}}
for feat_name in exp.dynamic_features.keys():
    exp.significance_tables['calcium'][feat_name] = {}
    for cell_id in cell_bunch:
        exp.significance_tables['calcium'][feat_name][cell_id] = {
            'stage1': False,
            'stage2': False
        }
        # Mark as significant based on stage1 result
        if cell_id in significance and feat_name in significance[cell_id]:
            sig_info = significance[cell_id][feat_name]
            # Check if it's marked as significant in stage1
            if isinstance(sig_info, dict) and sig_info.get('stage1', False):
                exp.significance_tables['calcium'][feat_name][cell_id]['stage1'] = True
                exp.significance_tables['calcium'][feat_name][cell_id]['stage2'] = True

# Manually find neurons with multiple significant features
sig_neurons = []
for cell_id in cell_bunch:
    sig_count = 0
    sig_features = []
    for feat_name in exp.dynamic_features.keys():
        if exp.significance_tables['calcium'][feat_name][cell_id]['stage2']:
            sig_count += 1
            sig_features.append(feat_name)
    if sig_count >= 2:
        sig_neurons.append(cell_id)
        print(f"Neuron {cell_id}: significant for {sig_features}")

print(f"\nNeurons with >=2 significant features: {sig_neurons}")

# Print ground truth mixed selectivity
print("\n=== GROUND TRUTH MIXED SELECTIVITY ===")
for neuron_idx in cell_bunch:
    selective_features = np.where(selectivity_info['matrix'][:, neuron_idx] > 0)[0]
    if len(selective_features) >= 2:
        weights = selectivity_info['matrix'][selective_features, neuron_idx]
        feat_names = [f'd_feat_{i}' for i in selective_features]
        print(f"Neuron {neuron_idx}: {' + '.join(feat_names)} (weights: {weights})")

# Print detected selectivity
print("\n=== DETECTED SELECTIVITY ===")
for neuron_idx in sig_neurons:
    sig_features = []
    for feat_name in exp.dynamic_features.keys():
        if exp.significance_tables['calcium'][feat_name][neuron_idx]['stage2']:
            sig_features.append(feat_name)
    if len(sig_features) >= 2:
        print(f"Neuron {neuron_idx}: {' + '.join(sig_features)}")

# Check for mixed selectivity pairs manually
print("\n=== CHECKING FOR MIXED SELECTIVITY PAIRS ===")
mixed_pairs = []
for i, n1 in enumerate(sig_neurons):
    for j, n2 in enumerate(sig_neurons):
        if i >= j:  # Skip diagonal and lower triangle
            continue
        
        # Get significant features for each neuron
        feats1 = set()
        feats2 = set()
        for feat_name in exp.dynamic_features.keys():
            if exp.significance_tables['calcium'][feat_name][n1]['stage2']:
                feats1.add(feat_name)
            if exp.significance_tables['calcium'][feat_name][n2]['stage2']:
                feats2.add(feat_name)
        
        # Check for overlap
        overlap = feats1.intersection(feats2)
        if len(overlap) >= 2:
            mixed_pairs.append((n1, n2, overlap))
            print(f"Mixed pair: Neurons {n1} & {n2} share {overlap}")

print(f"\nTotal mixed selectivity pairs found: {len(mixed_pairs)}")