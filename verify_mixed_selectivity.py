#!/usr/bin/env python
"""Verify mixed selectivity detection issue."""

import numpy as np
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity

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
    rate_0=0.1,
    rate_1=2.0,
    ampl_range=(0.5, 2.0),
    noise_std=0.005,
    seed=42,
    verbose=False
)

# Check which neurons are designed to have mixed selectivity
print("Ground truth mixed selectivity:")
for neuron_idx in range(20):
    selective_features = np.where(selectivity_info['matrix'][:, neuron_idx] > 0)[0]
    if len(selective_features) >= 2:
        feat_names = [selectivity_info['feature_names'][i] for i in selective_features]
        print(f"Neuron {neuron_idx}: selective to {feat_names}")

# Now check what get_significant_neurons returns (this would be populated after running INTENSE)
print("\n\nChecking get_significant_neurons method:")
# This will be empty because we haven't run significance testing yet
sig_neurons = exp.get_significant_neurons(min_nspec=2)
print(f"Neurons with mixed selectivity detected by INTENSE: {sig_neurons}")

# The problem: We need neurons to pass significance for MULTIPLE features
# With few shuffles and high noise, it's unlikely