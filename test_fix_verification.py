#!/usr/bin/env python3
"""Verify the fix for compute_cell_cell_significance issue"""

import numpy as np
from driada.experiment.synthetic import generate_synthetic_exp
from driada.intense.pipelines import compute_cell_cell_significance
from driada.intense.intense_base import compute_me_stats
from driada.information.info_base import TimeSeries

# Create experiment with shared TimeSeries
exp = generate_synthetic_exp(
    n_dfeats=2,
    n_cfeats=2,
    nneurons=3,
    duration=100,
    fps=10,
    seed=42,
    with_spikes=False
)

# Make neuron 1 share the same TimeSeries object as neuron 0
exp.neurons[1].ca = exp.neurons[0].ca  # Same object reference

# Extract signals as compute_cell_cell_significance does
signals = [exp.neurons[i].ca for i in [0, 1, 2]]

print("Signal object IDs:")
for i, sig in enumerate(signals):
    print(f"  Signal {i}: {id(sig)}")

print("\nComparing signal references:")
print(f"signals[0] is signals[1]: {signals[0] is signals[1]}")
print(f"signals is signals: {signals is signals}")

# Create a precomputed mask that excludes comparisons of identical objects
n = len(signals)
mask = np.ones((n, n))

# Manually check for identical TimeSeries objects and mask them out
print("\nChecking for identical TimeSeries objects:")
for i in range(n):
    for j in range(n):
        if signals[i] is signals[j]:
            print(f"  signals[{i}] is signals[{j}]: True - masking out")
            mask[i, j] = 0

print(f"\nMask after manual identity check:\n{mask}")

# Try compute_me_stats with the corrected mask
try:
    stats, significance, info = compute_me_stats(
        signals,
        signals,
        names1=[0, 1, 2],
        names2=[0, 1, 2],
        mode='stage1',
        metric='mi',
        precomputed_mask_stage1=mask,
        precomputed_mask_stage2=mask,
        n_shuffles_stage1=5,
        verbose=False,
        enable_parallelization=False,
        seed=42
    )
    print("\n✓ Success with manual masking!")
    print("Stats keys:", list(stats.keys()))
except Exception as e:
    print(f"\n✗ Error even with manual masking: {e}")