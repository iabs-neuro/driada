"""Debug the correlation test to understand why it's failing."""

import numpy as np
from tests.test_intense_pipelines import generate_synthetic_exp
from src.driada.intense.pipelines import compute_cell_cell_significance

# Recreate the test scenario
exp = generate_synthetic_exp(n_dfeats=2, n_cfeats=0, nneurons=5, seed=42, fps=20)

# Make some neurons correlated by copying signals
print("Original neuron data shapes:")
for i, neuron in enumerate(exp.neurons):
    print(f"  Neuron {i}: {neuron.ca.data.shape}")

# Create correlations
exp.neurons[1].ca.data = exp.neurons[0].ca.data + np.random.randn(len(exp.neurons[0].ca.data)) * 0.1
exp.neurons[3].ca.data = exp.neurons[2].ca.data + np.random.randn(len(exp.neurons[2].ca.data)) * 0.1

print("\nCorrelations created:")
print(f"Neuron 0 vs 1 correlation: {np.corrcoef(exp.neurons[0].ca.data, exp.neurons[1].ca.data)[0,1]:.3f}")
print(f"Neuron 2 vs 3 correlation: {np.corrcoef(exp.neurons[2].ca.data, exp.neurons[3].ca.data)[0,1]:.3f}")

# Compute cell-cell significance
sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
    exp,
    cell_bunch=None,  # All neurons
    data_type='calcium',
    mode='stage1',
    n_shuffles_stage1=50,
    n_shuffles_stage2=50,
    verbose=True,
    seed=42
)

print("\nMI Matrix:")
print(sim_mat)
print("\nSignificance Matrix:")
print(sig_mat)
print("\nP-value Matrix:")
print(pval_mat)

# Check the specific conditions
print("\nTest conditions:")
print(f"sim_mat[0, 1] = {sim_mat[0, 1]:.6f}")
print(f"sim_mat[0, 4] = {sim_mat[0, 4]:.6f}")
print(f"sim_mat[2, 3] = {sim_mat[2, 3]:.6f}")
print(f"sim_mat[2, 4] = {sim_mat[2, 4]:.6f}")

correlation_detected = (
    (sim_mat[0, 1] > 0 and sim_mat[0, 1] > sim_mat[0, 4]) or
    (sim_mat[2, 3] > 0 and sim_mat[2, 3] > sim_mat[2, 4])
)

print(f"\nCorrelation detected: {correlation_detected}")
print(f"Condition 1 (0,1 > 0,4): {sim_mat[0, 1] > 0 and sim_mat[0, 1] > sim_mat[0, 4]}")
print(f"Condition 2 (2,3 > 2,4): {sim_mat[2, 3] > 0 and sim_mat[2, 3] > sim_mat[2, 4]}")