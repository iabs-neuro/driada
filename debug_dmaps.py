"""Debug diffusion maps on swiss roll"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from driada.dim_reduction.data import MVData

# Generate swiss roll
n_samples = 400
data, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=42)
D = MVData(data.T)

# Apply diffusion maps
dmaps_emb = D.get_embedding(method="dmaps", dim=2, dm_alpha=1.0, nn=20, metric="l2")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original data colored by position
ax1.scatter(data[:, 0], data[:, 2], c=color, cmap='viridis')
ax1.set_title('Original Swiss Roll')

# Embedding colored by position
emb_points = dmaps_emb.coords.T
ax2.scatter(emb_points[:, 0], emb_points[:, 1], c=color, cmap='viridis')
ax2.set_title('Diffusion Maps Embedding')

plt.savefig('dmaps_swiss_roll.png')
plt.close()

# Check eigenvalues
eigenvalues = dmaps_emb.reducer_['eigenvalues']
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvalue ratios: {eigenvalues[:-1] / eigenvalues[1:]}")

# Check correlations
from scipy.stats import spearmanr
corr1 = spearmanr(color, emb_points[:, 0])[0]
corr2 = spearmanr(color, emb_points[:, 1])[0]
print(f"\nCorrelation with intrinsic parameter:")
print(f"Dimension 1: {corr1:.3f}")
print(f"Dimension 2: {corr2:.3f}")

# Also check if we're capturing local structure
from driada.dim_reduction import knn_preservation_rate
preservation = knn_preservation_rate(data, emb_points, k=15)
print(f"\nLocal structure preservation: {preservation:.3f}")