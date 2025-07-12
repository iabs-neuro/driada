#!/usr/bin/env python3
"""Quick test of example functionality with minimal parameters."""

import sys
import os
sys.path.insert(0, 'src')

import driada

# Test 1: Basic usage pattern
print("Testing basic usage pattern...")
exp = driada.generate_synthetic_exp(n_dfeats=1, n_cfeats=1, nneurons=3, duration=30, seed=42)
stats, significance, info, results = driada.compute_cell_feat_significance(
    exp, mode='two_stage', n_shuffles_stage1=10, n_shuffles_stage2=50, verbose=False
)
sig = exp.get_significant_neurons()
print(f"✓ Basic test: {len(sig)} significant neurons")

# Test 2: Full pipeline with MultiTimeSeries
print("\nTesting full pipeline with MultiTimeSeries...")
exp2, selectivity_info = driada.experiment.generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=1, n_continuous_feats=2, n_neurons=3,
    n_multifeatures=1, duration=30, seed=42, fps=20, verbose=False
)

# Get skip_delays
skip_delays = []
for feat_name, feat_data in exp2.dynamic_features.items():
    if isinstance(feat_name, tuple):
        skip_delays.append(feat_name)
    elif hasattr(feat_data, '__class__') and feat_data.__class__.__name__ == 'MultiTimeSeries':
        skip_delays.append(feat_name)

stats2, significance2, info2, results2 = driada.compute_cell_feat_significance(
    exp2, mode='two_stage', n_shuffles_stage1=10, n_shuffles_stage2=50,
    allow_mixed_dimensions=True, skip_delays=skip_delays if skip_delays else None,
    verbose=False
)
sig2 = exp2.get_significant_neurons()
print(f"✓ MultiTimeSeries test: {len(sig2)} significant neurons, skip_delays={skip_delays}")

# Test 3: Visualization
print("\nTesting visualization...")
if sig2:
    cell_id = list(sig2.keys())[0]
    feat_name = sig2[cell_id][0]
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    driada.intense.plot_neuron_feature_pair(exp2, cell_id, feat_name, ax=ax)
    plt.close(fig)
    print("✓ Visualization test passed")

print("\nAll tests passed! Examples should work correctly.")