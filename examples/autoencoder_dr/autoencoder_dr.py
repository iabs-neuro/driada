"""
Autoencoder dimensionality reduction example
=============================================

Demonstrates DRIADA's neural-network-based dimensionality reduction
on a circular manifold (head direction cells):

1. Standard autoencoder (AE) with continue_learning
2. Beta-VAE with KL divergence
3. PCA baseline
4. Side-by-side comparison
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    import torch  # noqa: F401
except ImportError:
    print("This example requires PyTorch. Install with: pip install torch")
    sys.exit(1)

import matplotlib.pyplot as plt

from driada.dim_reduction import MVData
from driada.experiment.synthetic import generate_circular_manifold_data


def main():
    print("=" * 60)
    print("DRIADA autoencoder DR example")
    print("=" * 60)

    output_dir = os.path.dirname(__file__)

    # ------------------------------------------------------------------
    # 1. Generate synthetic data (head direction cells on circular manifold)
    # ------------------------------------------------------------------
    print("\n[1] Generating synthetic head direction cell data")
    print("-" * 40)
    calcium, head_direction, preferred_dirs, rates = generate_circular_manifold_data(
        n_neurons=200,
        kappa=4.0,
        duration=1200,
        seed=42,
        verbose=True,
    )
    print(f"  Calcium shape: {calcium.shape}")
    print(f"  Head direction shape: {head_direction.shape}")

    mvdata = MVData(calcium, verbose=False)
    color = head_direction  # angle for coloring

    # ------------------------------------------------------------------
    # 2. Standard autoencoder with continue_learning
    # ------------------------------------------------------------------
    print("\n[2] Standard autoencoder")
    print("-" * 40)

    # Train for 5 epochs (not fully converged)
    emb_ae = mvdata.get_embedding(
        method="flexible_ae",
        dim=2,
        architecture="ae",
        inter_dim=64,
        epochs=5,
        lr=1e-3,
        feature_dropout=0.1,
        loss_components=[{"name": "reconstruction", "weight": 1.0, "loss_type": "mse"}],
        verbose=False,
    )
    print(f"  After 5 epochs   - loss: {emb_ae.nn_loss:.4f}")

    # Continue training for 25 more epochs
    emb_ae.continue_learning(25, lr=1e-3, verbose=False)
    print(f"  After 25 more    - loss: {emb_ae.nn_loss:.4f}")

    # Fine-tune with lower learning rate
    emb_ae.continue_learning(20, lr=1e-4, verbose=False)
    print(f"  After 20 fine-tune - loss: {emb_ae.nn_loss:.4f}")

    # ------------------------------------------------------------------
    # 3. Beta-VAE
    # ------------------------------------------------------------------
    print("\n[3] Beta-VAE (beta=4.0)")
    print("-" * 40)
    emb_vae = mvdata.get_embedding(
        method="flexible_ae",
        dim=2,
        architecture="vae",
        inter_dim=64,
        epochs=150,
        lr=1e-3,
        feature_dropout=0.1,
        loss_components=[
            {"name": "reconstruction", "weight": 1.0, "loss_type": "mse"},
            {"name": "beta_vae", "weight": 1.0, "beta": 4.0},
        ],
        verbose=False,
    )
    print(f"  Embedding shape: {emb_vae.coords.shape}")
    print(f"  Test loss (reconstruction + KL): {emb_vae.nn_loss:.4f}")

    # ------------------------------------------------------------------
    # 4. PCA baseline
    # ------------------------------------------------------------------
    print("\n[4] PCA baseline")
    print("-" * 40)
    emb_pca = mvdata.get_embedding(method="pca", dim=2)
    print(f"  Embedding shape: {emb_pca.coords.shape}")

    # ------------------------------------------------------------------
    # 5. Side-by-side visualization
    # ------------------------------------------------------------------
    print("\n[5] Creating comparison plot")
    print("-" * 40)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    embeddings = [
        (emb_pca, "PCA"),
        (emb_ae, "AE (5 + 25 + 20 epochs)"),
        (emb_vae, "Beta-VAE"),
    ]

    for ax, (emb, title) in zip(axes, embeddings):
        coords = emb.coords  # (dim, n_samples)
        sc = ax.scatter(
            coords[0], coords[1], c=color, cmap="hsv", s=2, alpha=0.5, vmin=0, vmax=2 * np.pi
        )
        ax.set_title(title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    fig.colorbar(sc, ax=axes[-1], label="Head direction (rad)")
    plt.suptitle("Circular manifold recovery (colored by head direction)")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "autoencoder_dr_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    print("\n" + "=" * 60)
    print("Autoencoder DR example complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
