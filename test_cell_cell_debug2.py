#!/usr/bin/env python3
"""Debug script to understand when compute_cell_cell_significance fails"""

import numpy as np
from driada.experiment.synthetic import generate_synthetic_exp
from driada.intense.pipelines import compute_cell_cell_significance
from driada.information.info_base import TimeSeries

# Test Case 1: Normal case (should work)
print("Test Case 1: Normal synthetic experiment")
exp = generate_synthetic_exp(
    n_dfeats=2,
    n_cfeats=2,
    nneurons=5,
    duration=100,
    fps=10,
    seed=42,
    with_spikes=False
)

try:
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=[0, 1, 2],
        data_type='calcium',
        mode='stage1',
        n_shuffles_stage1=5,
        ds=5,
        verbose=False,
        enable_parallelization=False,
        seed=42
    )
    print("✓ Success! Shape:", sim_mat.shape)
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")

# Test Case 2: With identical neurons (shared TimeSeries object)
print("\nTest Case 2: Neurons sharing the same TimeSeries object")
exp2 = generate_synthetic_exp(
    n_dfeats=2,
    n_cfeats=2,
    nneurons=3,
    duration=100,
    fps=10,
    seed=42,
    with_spikes=False
)

# Make neuron 1 share the same TimeSeries object as neuron 0
exp2.neurons[1].ca = exp2.neurons[0].ca  # Same object reference

try:
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp2,
        cell_bunch=[0, 1, 2],
        data_type='calcium',
        mode='stage1',
        n_shuffles_stage1=5,
        ds=5,
        verbose=False,
        enable_parallelization=False,
        seed=42
    )
    print("✓ Success! Shape:", sim_mat.shape)
    print("  Note: This worked even with shared TimeSeries objects")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")

# Test Case 3: With very small experiment
print("\nTest Case 3: Very small experiment (1 neuron)")
exp3 = generate_synthetic_exp(
    n_dfeats=1,
    n_cfeats=1,
    nneurons=1,
    duration=100,
    fps=10,
    seed=42,
    with_spikes=False
)

try:
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp3,
        cell_bunch=[0],  # Only one neuron
        data_type='calcium',
        mode='stage1',
        n_shuffles_stage1=5,
        ds=5,
        verbose=False,
        enable_parallelization=False,
        seed=42
    )
    print("✓ Success! Shape:", sim_mat.shape)
    print("  Diagonal should be 0:", np.diag(sim_mat))
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")

# Test Case 4: Direct TimeSeries list (not from experiment)
print("\nTest Case 4: Direct TimeSeries list comparison")
from driada.intense.intense_base import compute_me_stats

# Create some time series
ts1 = TimeSeries(np.sin(np.linspace(0, 4*np.pi, 100)), discrete=False)
ts2 = TimeSeries(np.cos(np.linspace(0, 4*np.pi, 100)), discrete=False)
ts3 = TimeSeries(np.sin(np.linspace(0, 2*np.pi, 100)) + 0.1, discrete=False)

# Compare the same list with itself
ts_list = [ts1, ts2, ts3]

try:
    stats, significance, info = compute_me_stats(
        ts_list,
        ts_list,  # Same list!
        names1=['ts1', 'ts2', 'ts3'],
        names2=['ts1', 'ts2', 'ts3'],
        mode='stage1',
        metric='mi',
        n_shuffles_stage1=5,
        verbose=False,
        enable_parallelization=False,
        seed=42
    )
    print("✓ Success! Got stats for", len(stats), "time series")
    # Check diagonal
    print("  Diagonal comparisons:")
    for i, name in enumerate(['ts1', 'ts2', 'ts3']):
        if name in stats and name in stats[name]:
            print(f"    {name} vs {name}: {stats[name][name]}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")

# Test Case 5: With spike data
print("\nTest Case 5: Using spike data")
exp5 = generate_synthetic_exp(
    n_dfeats=2,
    n_cfeats=2,
    nneurons=3,
    duration=100,
    fps=10,
    seed=42,
    with_spikes=True  # Generate spike data
)

try:
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp5,
        cell_bunch=[0, 1, 2],
        data_type='spikes',  # Use spikes instead of calcium
        mode='stage1',
        n_shuffles_stage1=5,
        ds=5,
        verbose=False,
        enable_parallelization=False,
        seed=42
    )
    print("✓ Success! Shape:", sim_mat.shape)
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")