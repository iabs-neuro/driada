"""Test manifold metrics on circular data"""

import numpy as np
from driada.dim_reduction.manifold_metrics import (
    extract_angles_from_embedding,
    compute_embedding_alignment_metrics,
    compute_embedding_quality,
    circular_diff,
)
import matplotlib.pyplot as plt

# Generate simple circular data
n_points = 1000
true_angles = np.linspace(0, 4 * np.pi, n_points) % (2 * np.pi)

# Create perfect circular embedding
embedding = np.column_stack([np.cos(true_angles), np.sin(true_angles)])

# Add slight noise
embedding += 0.1 * np.random.randn(*embedding.shape)

# Extract angles
extracted_angles = extract_angles_from_embedding(embedding)

# Plot to visualize
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot embedding
ax = axes[0, 0]
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=true_angles, cmap="hsv", s=10)
ax.set_title("Embedding colored by true angles")
ax.set_aspect("equal")
plt.colorbar(scatter, ax=ax)

# Plot true vs extracted angles
ax = axes[0, 1]
ax.scatter(true_angles, extracted_angles, alpha=0.5, s=5)
ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "r--", label="y=x")
ax.plot([0, 2 * np.pi], [0, -2 * np.pi], "g--", label="y=-x")
ax.set_xlabel("True angles")
ax.set_ylabel("Extracted angles")
ax.set_title("True vs Extracted angles")
ax.legend()

# Plot velocities
true_vel = circular_diff(true_angles)
extracted_vel = circular_diff(extracted_angles)

ax = axes[1, 0]
ax.plot(true_vel, label="True velocity", alpha=0.7)
ax.plot(extracted_vel, label="Extracted velocity", alpha=0.7)
ax.set_title("Angular velocities")
ax.legend()

# Scatter plot of velocities
ax = axes[1, 1]
ax.scatter(true_vel, extracted_vel, alpha=0.5, s=5)
ax.set_xlabel("True velocity")
ax.set_ylabel("Extracted velocity")
ax.set_title(f"Velocity correlation: {np.corrcoef(true_vel, extracted_vel)[0,1]:.3f}")

plt.tight_layout()
plt.savefig("debug_metrics.png")

# Test metrics
print("Testing manifold metrics:")
print(f"True angles range: [{true_angles.min():.2f}, {true_angles.max():.2f}]")
print(
    f"Extracted angles range: [{extracted_angles.min():.2f}, {extracted_angles.max():.2f}]"
)

# Direct angle correlation (after unwrapping)
# Handle circular correlation properly
true_complex = np.exp(1j * true_angles)
extracted_complex = np.exp(1j * extracted_angles)
circular_corr = np.abs(np.mean(true_complex * np.conj(extracted_complex)))
print(f"\nDirect circular correlation: {circular_corr:.3f}")

# Alignment metrics (replaces temporal consistency)
alignment_metrics = compute_embedding_alignment_metrics(
    embedding, true_angles, "circular"
)
print(f"Velocity correlation: {alignment_metrics['velocity_correlation']:.3f}")
if "variance_ratio" in alignment_metrics:
    print(f"Variance ratio: {alignment_metrics['variance_ratio']:.3f}")

# Embedding quality
quality = compute_embedding_quality(embedding, true_angles, "circular")
print("\nEmbedding quality:")
print(f"  Train error: {quality['train_error']:.3f}")
print(f"  Test error: {quality['test_error']:.3f}")

# Check for sign flip
if (
    np.corrcoef(true_vel, -extracted_vel)[0, 1]
    > np.corrcoef(true_vel, extracted_vel)[0, 1]
):
    print("\nWARNING: Velocities are anti-correlated - possible sign flip!")
    print(
        f"Correlation with flipped sign: {np.corrcoef(true_vel, -extracted_vel)[0,1]:.3f}"
    )
