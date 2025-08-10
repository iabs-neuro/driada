#!/usr/bin/env python
"""Comprehensive test of disentanglement functionality including edge cases."""

import numpy as np
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity
from driada.intense.pipelines import compute_cell_feat_significance
from driada.intense.disentanglement import disentangle_pair, disentangle_all_selectivities
from driada.information.info_base import TimeSeries

# Test Case 1: Mixed neurons that CAN be disentangled (redundant features)
print("=" * 60)
print("CASE 1: Redundant features (one is a noisy version of another)")
print("=" * 60)

# Create simple synthetic data
n_frames = 2000
feature1 = np.random.binomial(1, 0.1, n_frames).astype(float)  # Binary feature
feature2 = feature1 + 0.1 * np.random.randn(n_frames)  # Noisy version
feature2[feature2 < 0] = 0

# Create neuron that responds to feature1 (and thus feature2)
neuron = np.zeros(n_frames)
spike_times = np.where(feature1 > 0)[0]
for t in spike_times:
    if t < n_frames - 50:
        neuron[t:t+50] += np.exp(-np.arange(50) / 10.0)  # Calcium decay

neuron += 0.05 * np.random.randn(n_frames)  # Add noise

# Test disentanglement
ts_neuron = TimeSeries(neuron, discrete=False)
ts_feat1 = TimeSeries(feature1, discrete=True)
ts_feat2 = TimeSeries(feature2, discrete=False)

result = disentangle_pair(ts_neuron, ts_feat1, ts_feat2, verbose=True)
print(f"Disentanglement result: {result}")
print(f"Expected: 0 (feature1 is primary) or close to 0")
print()

# Test Case 2: Mixed neurons that CANNOT be disentangled (true mixed selectivity)
print("=" * 60)
print("CASE 2: True mixed selectivity (XOR-like relationship)")
print("=" * 60)

# Create XOR features
feature1 = np.random.binomial(1, 0.3, n_frames).astype(float)
feature2 = np.random.binomial(1, 0.3, n_frames).astype(float)

# Neuron responds to XOR of features
neuron_xor = np.zeros(n_frames)
xor_active = (feature1 > 0) ^ (feature2 > 0)  # XOR
spike_times = np.where(xor_active)[0]
for t in spike_times:
    if t < n_frames - 50:
        neuron_xor[t:t+50] += np.exp(-np.arange(50) / 10.0)

neuron_xor += 0.05 * np.random.randn(n_frames)

ts_neuron_xor = TimeSeries(neuron_xor, discrete=False)
result_xor = disentangle_pair(ts_neuron_xor, ts_feat1, TimeSeries(feature2, discrete=True), verbose=True)
print(f"Disentanglement result: {result_xor}")
print(f"Expected: 0.5 (undistinguishable)")
print()

# Test Case 3: Zero MI case (one feature has no relationship)
print("=" * 60)
print("CASE 3: Zero MI (one feature unrelated to neuron)")
print("=" * 60)

# Create unrelated feature
feature_unrelated = np.random.binomial(1, 0.2, n_frames).astype(float)
ts_unrelated = TimeSeries(feature_unrelated, discrete=True)

result_zero = disentangle_pair(ts_neuron, ts_feat1, ts_unrelated, verbose=True)
print(f"Disentanglement result: {result_zero}")
print(f"Expected: 0 (feature1 is primary, unrelated has zero MI)")
print()

# Test Case 4: With experimental data structure
print("=" * 60)
print("CASE 4: Full experiment with disentanglement analysis")
print("=" * 60)

# Generate experiment with known mixed selectivity
exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=3,
    n_continuous_feats=0,
    n_neurons=10,
    duration=300,  # 5 minutes
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

# Find mixed selective neurons
mixed_neurons = np.where(np.sum(selectivity_info['matrix'] > 0, axis=0) >= 2)[0]
print(f"Found {len(mixed_neurons)} neurons with mixed selectivity")

# Run significance testing
stats, significance, info, results = compute_cell_feat_significance(
    exp,
    cell_bunch=mixed_neurons[:5].tolist(),
    feat_bunch=None,
    mode='stage1',
    n_shuffles_stage1=10,
    metric='mi',
    pval_thr=0.05,
    multicomp_correction=None,
    find_optimal_delays=False,
    allow_mixed_dimensions=True,
    enable_parallelization=False,
    verbose=False,
    seed=42
)

# Manually set up experiment structure for disentanglement
exp.stats_tables['calcium'] = stats
exp.significance_tables = {'calcium': {}}

for feat_name in exp.dynamic_features.keys():
    exp.significance_tables['calcium'][feat_name] = {}
    for cell_id in mixed_neurons[:5]:
        exp.significance_tables['calcium'][feat_name][cell_id] = {
            'stage1': False,
            'stage2': False
        }
        if cell_id in significance and feat_name in significance[cell_id]:
            sig_info = significance[cell_id][feat_name]
            if isinstance(sig_info, dict) and sig_info.get('stage1', False):
                exp.significance_tables['calcium'][feat_name][cell_id]['stage1'] = True
                exp.significance_tables['calcium'][feat_name][cell_id]['stage2'] = True

# Try disentanglement with a workaround
print("\nDirect disentanglement test on known mixed pairs...")

# Find a neuron with mixed selectivity
for neuron_idx in mixed_neurons[:5]:
    selective_features = np.where(selectivity_info['matrix'][:, neuron_idx] > 0)[0]
    if len(selective_features) >= 2:
        print(f"\nTesting neuron {neuron_idx} selective to features {selective_features}")
        
        # Get time series
        neur_ts = exp.neurons[neuron_idx].ca
        feat1_name = f'd_feat_{selective_features[0]}'
        feat2_name = f'd_feat_{selective_features[1]}'
        feat1_ts = exp.dynamic_features[feat1_name]
        feat2_ts = exp.dynamic_features[feat2_name]
        
        # Test disentanglement
        result = disentangle_pair(neur_ts, feat1_ts, feat2_ts, verbose=True)
        print(f"Result: {result}")
        
        # For equal weights, we expect 0.5 (undistinguishable)
        # or slight bias depending on noise
        break

print("\n" + "=" * 60)
print("SUMMARY OF EXPECTED BEHAVIORS:")
print("=" * 60)
print("1. Redundant features: Result ≈ 0 or 1 (one is primary)")
print("2. True mixed selectivity: Result = 0.5 (undistinguishable)")
print("3. Zero MI case: Result = 0 or 1 (non-zero MI feature is primary)")
print("4. Equal contribution: Result ≈ 0.5 (both contribute equally)")
print("\nThe test should handle ALL cases without crashing!")