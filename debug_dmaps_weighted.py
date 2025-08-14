"""Debug diffusion maps with weighted graph"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from driada.dim_reduction.data import MVData

# Generate swiss roll
n_samples = 400
data, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=42)
D = MVData(data.T)

# Apply diffusion maps - this should now use weighted graph
dmaps_emb = D.get_embedding(method="dmaps", dim=2, dm_alpha=0.5, nn=15, metric="l2")

# Check if graph is weighted
print(f"Graph weighted: {dmaps_emb.graph.weighted}")
print(f"Graph adjacency type: {type(dmaps_emb.graph.adj)}")
print(f"Graph adjacency shape: {dmaps_emb.graph.adj.shape}")
print(f"Graph adjacency non-zeros: {dmaps_emb.graph.adj.nnz}")

# Check adjacency matrix values
if hasattr(dmaps_emb.graph.adj, 'data'):
    print(f"\nAdjacency matrix stats:")
    print(f"Min value: {dmaps_emb.graph.adj.data.min()}")
    print(f"Max value: {dmaps_emb.graph.adj.data.max()}")
    print(f"Mean value: {dmaps_emb.graph.adj.data.mean()}")
    print(f"Unique values (first 10): {np.unique(dmaps_emb.graph.adj.data)[:10]}")

# Check eigenvalues
eigenvalues = dmaps_emb.reducer_['eigenvalues']
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvalue ratios: {eigenvalues[:-1] / eigenvalues[1:]}")

# Check embedding
emb_points = dmaps_emb.coords.T
print(f"\nEmbedding shape: {emb_points.shape}")
print(f"Embedding std: {np.std(emb_points, axis=0)}")
print(f"Embedding range: [{emb_points.min()}, {emb_points.max()}]")

# Check raw eigenvectors before scaling
eigenvectors = dmaps_emb.reducer_['eigenvectors']
print(f"\nRaw eigenvector std (before scaling):")
for i in range(min(3, eigenvectors.shape[1])):
    print(f"  Eigenvector {i}: {np.std(eigenvectors[:, i])}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original data
ax1.scatter(data[:, 0], data[:, 2], c=color, cmap='viridis')
ax1.set_title('Original Swiss Roll')

# Embedding
ax2.scatter(emb_points[:, 0], emb_points[:, 1], c=color, cmap='viridis')
ax2.set_title('Diffusion Maps Embedding')

plt.savefig('dmaps_weighted_debug.png')
plt.close()

# Try different alpha values
print("\n=== Testing different alpha values ===")
for alpha in [0.0, 0.5, 1.0]:
    dmaps_emb = D.get_embedding(method="dmaps", dim=2, dm_alpha=alpha, nn=15, metric="l2")
    emb_points = dmaps_emb.coords.T
    std = np.std(emb_points, axis=0)
    print(f"Alpha={alpha}: std={std}, eigenvalues={dmaps_emb.reducer_['eigenvalues']}")