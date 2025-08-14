"""Debug diffusion maps eigenvalue scaling"""
import numpy as np
from sklearn.datasets import make_swiss_roll
from driada.dim_reduction.data import MVData
import matplotlib.pyplot as plt

# Generate swiss roll
n_samples = 400
data, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=42)
D = MVData(data.T)

# Try different diffusion times
print("=== Testing different diffusion times ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, t in enumerate([0.5, 1, 2, 5, 10, 20]):
    # Create embedding with specific diffusion time
    dmaps_emb = D.get_embedding(method="dmaps", dim=2, dm_alpha=0.5, nn=15, metric="l2")
    
    # Get eigenvalues and eigenvectors
    eigenvalues = dmaps_emb.reducer_['eigenvalues']
    eigenvectors = dmaps_emb.reducer_['eigenvectors']
    
    # Apply different time scaling manually
    eigenvalues_t = eigenvalues ** t
    coords_t = (eigenvectors * eigenvalues_t).T
    
    # Plot
    ax = axes[i]
    scatter = ax.scatter(coords_t[0], coords_t[1], c=color, cmap='viridis', s=20)
    ax.set_title(f't={t}, λ^t=[{eigenvalues_t[0]:.3f}, {eigenvalues_t[1]:.3f}]')
    ax.set_xlabel(f'std=[{np.std(coords_t[0]):.3f}, {np.std(coords_t[1]):.3f}]')
    
    print(f"t={t}: eigenvalues^t={eigenvalues_t}, std={np.std(coords_t, axis=1)}")

plt.tight_layout()
plt.savefig('dmaps_time_scaling.png')
plt.close()

# Now let's try NOT scaling by eigenvalues at all
print("\n=== Testing without eigenvalue scaling ===")
dmaps_emb = D.get_embedding(method="dmaps", dim=2, dm_alpha=0.5, nn=15, metric="l2")
eigenvectors = dmaps_emb.reducer_['eigenvectors']
eigenvalues = dmaps_emb.reducer_['eigenvalues']

# Use raw eigenvectors
coords_raw = eigenvectors.T

print(f"Raw eigenvectors std: {np.std(coords_raw, axis=1)}")
print(f"Eigenvalues: {eigenvalues}")

# Plot comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Original
ax1.scatter(data[:, 0], data[:, 2], c=color, cmap='viridis')
ax1.set_title('Original Swiss Roll')

# With eigenvalue scaling (current implementation)
emb_points = dmaps_emb.coords.T
ax2.scatter(emb_points[:, 0], emb_points[:, 1], c=color, cmap='viridis')
ax2.set_title(f'With λ^t scaling (std={np.std(emb_points, axis=0)})')

# Without scaling
ax3.scatter(coords_raw[0], coords_raw[1], c=color, cmap='viridis')
ax3.set_title(f'Without scaling (std={np.std(coords_raw, axis=1)})')

plt.tight_layout()
plt.savefig('dmaps_scaling_comparison.png')
plt.close()

# Check correlation with intrinsic parameter
from scipy.stats import spearmanr
corr_scaled = max(abs(spearmanr(color, emb_points[:, 0])[0]), 
                  abs(spearmanr(color, emb_points[:, 1])[0]))
corr_raw = max(abs(spearmanr(color, coords_raw[0])[0]), 
               abs(spearmanr(color, coords_raw[1])[0]))

print(f"\nCorrelation with intrinsic parameter:")
print(f"With scaling: {corr_scaled:.3f}")
print(f"Without scaling: {corr_raw:.3f}")

# Try different sigma values
print("\n=== Testing different sigma values ===")
for sigma in [0.5, 1.0, 2.0, 5.0]:
    metric_params = {"metric_name": "l2", "sigma": sigma}
    graph_params = {
        "g_method_name": "knn",
        "nn": 15,
        "weighted": 1,
        "max_deleted_nodes": 0.2,
        "dist_to_aff": "hk",
    }
    
    G = D.get_proximity_graph(metric_params, graph_params)
    D_temp = MVData(data.T)
    D_temp.graph = G
    
    from driada.dim_reduction.embedding import Embedding
    emb = Embedding(
        init_data=data.T,
        init_distmat=None,
        labels=None,
        params={"e_method_name": "dmaps", "e_method": None, "dim": 2, "dm_alpha": 0.5},
        g=G
    )
    emb.create_dmaps_embedding_()
    
    emb_points = emb.coords.T
    std = np.std(emb_points, axis=0)
    corr = max(abs(spearmanr(color, emb_points[:, 0])[0]), 
               abs(spearmanr(color, emb_points[:, 1])[0]))
    
    print(f"Sigma={sigma}: std={std}, corr={corr:.3f}, eigenvalues={emb.reducer_['eigenvalues']}")