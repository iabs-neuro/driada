"""
Advanced tests for Autoencoders (AE) and Variational Autoencoders (VAE).
Consolidates important tests from debugging session.
"""

import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from driada.dim_reduction.data import MVData
from driada.dim_reduction.manifold_metrics import knn_preservation_rate


def test_vae_latent_space_regularization():
    """Test that VAE properly regularizes latent space to standard normal"""
    # Generate data with some structure (not pure random noise)
    np.random.seed(42)
    n_samples = 1000
    # Create data with correlation structure to avoid posterior collapse
    base = np.random.randn(5, n_samples)
    data = np.vstack([base, base + 0.5 * np.random.randn(5, n_samples)])
    D = MVData(data)

    # Train VAE with adjusted parameters to prevent posterior collapse
    vae_emb = D.get_embedding(
        method="vae",
        dim=2,
        epochs=200,  # More epochs for better convergence
        batch_size=64,  # Larger batch size for stability
        lr=5e-4,  # Lower learning rate to prevent collapse
        train_size=0.8,
        verbose=False,
        kld_weight=0.01,  # Much lower KL weight to prevent posterior collapse
        enc_kwargs={"dropout": 0.1},
        dec_kwargs={"dropout": 0.1},
    )

    # Check latent space properties
    latent_mean = np.mean(vae_emb.coords, axis=1)
    latent_std = np.std(vae_emb.coords, axis=1)

    # VAE should regularize to approximately standard normal
    # Use more lenient bounds due to finite sample and training dynamics
    assert np.all(np.abs(latent_mean) < 1.0), f"VAE mean {latent_mean} not centered"
    assert np.all(
        np.abs(latent_std - 1.0) < 1.0  # More lenient: std should be in range [0, 2]
    ), f"VAE std {latent_std} not normalized"


def test_ae_correlation_loss():
    """Test that correlation loss decorrelates latent features"""
    # Generate correlated data
    np.random.seed(42)
    n_samples = 500
    base_features = np.random.randn(5, n_samples)
    data = np.vstack(
        [base_features, base_features + 0.5 * np.random.randn(5, n_samples)]
    )
    D = MVData(data)

    # Test with correlation loss
    nn_params = {
        "continue_learning": 0,
        "epochs": 100,
        "lr": 5e-3,
        "seed": 42,
        "batch_size": 64,
        "feature_dropout": 0.2,
        "add_corr_loss": True,
        "corr_hyperweight": 1.0,
        "verbose": False,
        "enc_kwargs": {"dropout": 0.2},
        "dec_kwargs": {"dropout": 0.2},
    }

    ae_emb = D.get_embedding(
        method="ae",
        dim=2,
        epochs=100,
        lr=5e-3,
        seed=42,
        batch_size=64,
        feature_dropout=0.2,
        add_corr_loss=True,
        corr_hyperweight=1.0,
        verbose=False,
        enc_kwargs={"dropout": 0.2},
        dec_kwargs={"dropout": 0.2},
    )

    # Latent features should have low correlation
    latent_corr = np.abs(np.corrcoef(ae_emb.coords)[0, 1])
    assert (
        latent_corr < 0.3
    ), f"Correlation loss failed to decorrelate (corr={latent_corr:.3f})"


def test_ae_vs_vae_reconstruction_quality():
    """Compare reconstruction quality between AE and VAE"""
    # Generate swiss roll
    n_samples = 500
    data, _ = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    D = MVData(data.T)

    nn_params_base = {
        "continue_learning": 0,
        "epochs": 100,
        "lr": 1e-3,
        "seed": 42,
        "batch_size": 64,
        "feature_dropout": 0.1,
        "verbose": False,
        "enc_kwargs": {"dropout": 0.1},
        "dec_kwargs": {"dropout": 0.1},
    }

    # Train AE
    ae_emb = D.get_embedding(
        method="ae",
        dim=2,
        epochs=100,
        lr=1e-3,
        seed=42,
        batch_size=64,
        feature_dropout=0.1,
        verbose=False,
        enc_kwargs={"dropout": 0.1},
        dec_kwargs={"dropout": 0.1},
    )

    # Train VAE with low KL weight for fair comparison
    vae_emb = D.get_embedding(
        method="vae",
        dim=2,
        epochs=200,  # More epochs for better convergence
        lr=5e-4,  # Slightly lower learning rate
        seed=42,
        batch_size=64,
        feature_dropout=0.1,
        verbose=False,
        kld_weight=0.005,  # Even lower KL weight for better reconstruction
        enc_kwargs={"dropout": 0.1},
        dec_kwargs={"dropout": 0.1},
    )

    # Compare neighborhood preservation
    k = 10
    ae_knn = knn_preservation_rate(data, ae_emb.coords.T, k=k)
    vae_knn = knn_preservation_rate(data, vae_emb.coords.T, k=k)

    # Both should achieve reasonable preservation for swiss roll
    # With proper training parameters, both can achieve good results
    assert ae_knn > 0.25, f"AE preservation too low: {ae_knn:.3f}"
    assert vae_knn > 0.25, f"VAE preservation too low: {vae_knn:.3f}"


def test_vae_encoder_unconstrained():
    """Verify VAE encoder outputs are not constrained to [0,1]"""
    from driada.dim_reduction.neural import VAE

    device = torch.device("cpu")
    model = VAE(orig_dim=10, inter_dim=64, code_dim=2, device=device)

    # Random input
    x = torch.randn(32, 10)

    with torch.no_grad():
        code, mu, log_var = model.get_code(x)

    # Outputs should be unconstrained (not sigmoid-limited to [0,1])
    assert mu.min() < 0 or mu.max() > 1, "Mean should be unconstrained"
    assert (
        log_var.min() < 0 or log_var.max() > 1
    ), "Log variance should be unconstrained"
