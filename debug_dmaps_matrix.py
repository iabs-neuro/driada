"""Debug diffusion maps matrix construction"""
import numpy as np
from sklearn.datasets import make_swiss_roll
from driada.dim_reduction.data import MVData
from scipy.sparse import diags
import scipy.sparse as sp

# Generate swiss roll
n_samples = 100  # Smaller for easier debugging
data, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=42)
D = MVData(data.T)

# Create graph manually to inspect
metric_params = {"metric_name": "l2", "sigma": 1.0}
graph_params = {
    "g_method_name": "knn",
    "nn": 15,
    "weighted": 1,
    "max_deleted_nodes": 0.2,
    "dist_to_aff": "hk",
}

G = D.get_proximity_graph(metric_params, graph_params)
print(f"Graph adjacency shape: {G.adj.shape}")
print(f"Graph is weighted: {G.weighted}")

# Get affinity matrix W
W = G.adj.astype(float)
print(f"\nAffinity matrix W:")
print(f"Shape: {W.shape}, Non-zeros: {W.nnz}")
print(f"Values range: [{W.data.min():.4f}, {W.data.max():.4f}]")
print(f"Mean value: {W.data.mean():.4f}")

# Check symmetry
W_sym = (W + W.T) / 2
print(f"W symmetric? {np.allclose(W.toarray(), W.T.toarray())}")

# Compute degree matrix
D_vec = np.asarray(W.sum(axis=1)).flatten()
print(f"\nDegree vector D:")
print(f"Range: [{D_vec.min():.4f}, {D_vec.max():.4f}]")
print(f"Mean: {D_vec.mean():.4f}")

# Apply alpha normalization
alpha = 0.5
D_alpha = D_vec ** alpha
D_alpha_inv = 1.0 / (D_alpha + 1e-10)
W_alpha = diags(D_alpha_inv) @ W @ diags(D_alpha_inv)

# Check W_alpha
print(f"\nW_alpha (alpha={alpha}):")
if hasattr(W_alpha, 'data'):
    print(f"Values range: [{W_alpha.data.min():.4f}, {W_alpha.data.max():.4f}]")
    print(f"Mean value: {W_alpha.data.mean():.4f}")

# New degree for normalized kernel
D_alpha_norm = np.asarray(W_alpha.sum(axis=1)).flatten()
print(f"\nDegree after alpha normalization:")
print(f"Range: [{D_alpha_norm.min():.4f}, {D_alpha_norm.max():.4f}]")
print(f"Mean: {D_alpha_norm.mean():.4f}")

# Create Markov matrix
D_alpha_norm_inv = 1.0 / (D_alpha_norm + 1e-10)
P = diags(D_alpha_norm_inv) @ W_alpha

# Check P properties
print(f"\nMarkov matrix P:")
P_dense = P.toarray() if hasattr(P, 'toarray') else P
row_sums = P_dense.sum(axis=1)
print(f"Row sums range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
print(f"All row sums ≈ 1? {np.allclose(row_sums, 1.0)}")

# Compute eigenvalues
from scipy.sparse.linalg import eigsh
eigenvalues, eigenvectors = eigsh(P, k=5, which='LM', tol=1e-6)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\nTop 5 eigenvalues: {eigenvalues}")

# Check first eigenvector (should be constant)
v0 = eigenvectors[:, 0]
print(f"\nFirst eigenvector stats:")
print(f"Range: [{v0.min():.6f}, {v0.max():.6f}]")
print(f"Std: {v0.std():.6f}")
print(f"Is approximately constant? {v0.std() < 0.01}")

# Check spectral gap
if len(eigenvalues) > 1:
    gap = eigenvalues[0] - eigenvalues[1]
    print(f"\nSpectral gap (λ1 - λ2): {gap:.6f}")
    
# Try with different sigma values
print("\n=== Testing different sigma values ===")
for sigma in [0.1, 0.5, 1.0, 2.0, 5.0]:
    metric_params['sigma'] = sigma
    G = D.get_proximity_graph(metric_params, graph_params)
    W = G.adj.astype(float)
    
    # Quick eigenvalue check
    D_vec = np.asarray(W.sum(axis=1)).flatten()
    D_inv = 1.0 / (D_vec + 1e-10)
    P = diags(D_inv) @ W
    
    eigenvalues, _ = eigsh(P, k=3, which='LM', tol=1e-6)
    eigenvalues = np.sort(eigenvalues)[::-1]
    print(f"Sigma={sigma}: eigenvalues={eigenvalues[:3]}, gap={eigenvalues[0]-eigenvalues[1]:.4f}")