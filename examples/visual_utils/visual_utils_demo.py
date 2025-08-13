"""
Demonstration of DRIADA Visual Utilities
========================================

This example shows how to use the new visual utilities module for
creating publication-ready figures with consistent styling.
"""

import numpy as np
import matplotlib.pyplot as plt
from driada.utils.visual import (
    plot_embedding_comparison,
    plot_trajectories,
    plot_component_interpretation,
    plot_neuron_selectivity_summary,
    DEFAULT_DPI,
)

# Generate sample data
np.random.seed(42)
n_samples = 500

# Create embeddings for different methods
angles = np.linspace(0, 4 * np.pi, n_samples)
noise = 0.1

embeddings = {
    "PCA": np.column_stack(
        [
            np.cos(angles) + noise * np.random.randn(n_samples),
            np.sin(angles) + noise * np.random.randn(n_samples),
        ]
    ),
    "UMAP": np.column_stack(
        [
            1.2 * np.cos(angles) + noise * np.random.randn(n_samples),
            1.2 * np.sin(angles) + noise * np.random.randn(n_samples),
        ]
    ),
    "LE": np.column_stack(
        [
            0.8 * np.cos(angles) + noise * np.random.randn(n_samples),
            0.8 * np.sin(angles) + noise * np.random.randn(n_samples),
        ]
    ),
}

# Create features for coloring
features = {
    "angle": angles % (2 * np.pi) - np.pi,  # Wrap to [-π, π]
    "speed": np.abs(np.sin(angles * 2)) + 0.2 * np.random.randn(n_samples),
}

print("Demonstrating DRIADA Visual Utilities")
print("=====================================")

# 1. Embedding comparison with trajectories
print("\n1. Creating embedding comparison figure...")
fig1 = plot_embedding_comparison(
    embeddings=embeddings,
    features=features,
    with_trajectory=True,
    compute_metrics=True,
    figsize=(18, 15),
    save_path="visual_demo_embeddings.png",
    dpi=DEFAULT_DPI,
)
print("   Saved: visual_demo_embeddings.png")

# 2. Trajectory-only figure
print("\n2. Creating trajectory figure...")
fig2 = plot_trajectories(
    embeddings=embeddings,
    trajectory_kwargs={"arrow_spacing": 25, "linewidth": 1.0, "alpha": 0.5},
    figsize=(15, 5),
    save_path="visual_demo_trajectories.png",
)
print("   Saved: visual_demo_trajectories.png")

# 3. Component interpretation
print("\n3. Creating component interpretation figure...")
# Create sample MI values
mi_matrices = {
    "PCA": np.array(
        [
            [0.45, 0.12, 0.08, 0.05],  # Feature 1 vs components
            [0.15, 0.38, 0.22, 0.10],  # Feature 2 vs components
            [0.08, 0.15, 0.25, 0.35],  # Feature 3 vs components
        ]
    ),
    "UMAP": np.array(
        [[0.52, 0.18, 0.05, 0.03], [0.20, 0.42, 0.15, 0.08], [0.05, 0.12, 0.38, 0.28]]
    ),
    "LE": np.array(
        [[0.38, 0.22, 0.12, 0.08], [0.18, 0.35, 0.25, 0.15], [0.10, 0.18, 0.30, 0.32]]
    ),
}

# Add metadata for PCA
metadata = {"PCA": {"explained_variance_ratio": np.array([0.45, 0.25, 0.15, 0.10])}}

fig3 = plot_component_interpretation(
    mi_matrices=mi_matrices,
    feature_names=["Angular Position", "Speed", "Acceleration"],
    metadata=metadata,
    n_components=4,
    figsize=(18, 6),
    save_path="visual_demo_component_mi.png",
)
print("   Saved: visual_demo_component_mi.png")

# 4. Neuron selectivity summary
print("\n4. Creating neuron selectivity summary...")
selectivity_counts = {
    "Spatial\n(2D)": 85,
    "Spatial\n(X only)": 45,
    "Spatial\n(Y only)": 38,
    "Speed": 52,
    "Mixed\nSelectivity": 65,
    "Non-selective": 115,
}

fig4 = plot_neuron_selectivity_summary(
    selectivity_counts=selectivity_counts,
    total_neurons=400,
    figsize=(10, 6),
    save_path="visual_demo_selectivity.png",
)
print("   Saved: visual_demo_selectivity.png")

# Show all figures
plt.show()

print("\n✓ Visual utilities demonstration complete!")
print(f"  All figures saved at {DEFAULT_DPI} DPI")
print("\nKey features demonstrated:")
print("  - Consistent styling across all plots")
print("  - Configurable DPI for publication quality")
print("  - Automatic trajectory visualization")
print("  - Density contours and percentile markers")
print("  - Component interpretation with MI values")
print("  - Neuron selectivity summaries with percentages")
