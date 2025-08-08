#!/usr/bin/env python3
"""Debug script to understand compute_cell_cell_significance issue"""

import numpy as np
from driada.experiment.synthetic import generate_synthetic_exp
from driada.intense.pipelines import compute_cell_cell_significance

# Generate a simple experiment
exp = generate_synthetic_exp(
    n_dfeats=2,
    n_cfeats=2,
    nneurons=5,
    duration=100,
    fps=10,
    seed=42,
    with_spikes=False
)

# Try to run compute_cell_cell_significance
try:
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=[0, 1, 2],  # Just 3 neurons
        data_type='calcium',
        mode='stage1',
        n_shuffles_stage1=10,
        ds=5,
        verbose=True,
        enable_parallelization=False,
        seed=42
    )
    print("Success! Shape:", sim_mat.shape)
    print("Diagonal values:", np.diag(sim_mat))
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    
    # Try to understand the issue
    print("\nDebugging info:")
    print("Number of neurons:", exp.n_cells)
    print("Calcium data shapes:")
    for i in range(3):
        print(f"  Neuron {i}: {exp.neurons[i].ca.data.shape}")
    
    # Check if the TimeSeries objects are the same
    print("\nTimeSeries identity checks:")
    for i in range(3):
        for j in range(3):
            if i <= j:  # Only check upper triangle
                same_obj = exp.neurons[i].ca is exp.neurons[j].ca
                same_data = exp.neurons[i].ca.data is exp.neurons[j].ca.data
                print(f"  Neuron {i} vs {j}: same_obj={same_obj}, same_data={same_data}")