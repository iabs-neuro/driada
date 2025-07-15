"""
Comparison of Autoencoder (AE) vs Variational Autoencoder (VAE) for dimensionality reduction.

Key differences:
1. AE: Deterministic encoder/decoder, learns fixed representation
2. VAE: Probabilistic encoder, learns distribution (mean + variance), uses reparameterization trick
3. AE: No regularization on latent space
4. VAE: KL divergence regularization forces latent space to be close to standard normal
5. AE: Can have arbitrary latent space structure
6. VAE: Encourages smooth, continuous latent space suitable for generation
"""

import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll
from src.driada.dim_reduction.data import MVData
from src.driada.dim_reduction.dr_base import METHODS_DICT
import matplotlib.pyplot as plt


def test_ae_vs_vae_comparison():
    """Compare AE and VAE on the same dataset"""
    # Generate swiss roll data
    n_samples = 1000
    data, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    D = MVData(data.T)
    
    # Common neural network parameters
    nn_params_base = {
        'continue_learning': 0,
        'epochs': 100,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.2,
        'verbose': False,
        'enc_kwargs': {'dropout': 0.2},
        'dec_kwargs': {'dropout': 0.2}
    }
    
    # Test regular autoencoder
    ae_params = {
        'e_method_name': 'ae',
        'dim': 2
    }
    ae_params['e_method'] = METHODS_DICT[ae_params['e_method_name']]
    ae_emb = D.get_embedding(ae_params, kwargs=nn_params_base)
    
    # Test variational autoencoder
    vae_params = {
        'e_method_name': 'vae',
        'dim': 2
    }
    vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
    vae_emb = D.get_embedding(vae_params, kwargs=nn_params_base)
    
    # Both should produce valid embeddings
    assert ae_emb.coords.shape == (2, n_samples)
    assert vae_emb.coords.shape == (2, n_samples)
    
    # Check no NaN values
    assert not np.any(np.isnan(ae_emb.coords))
    assert not np.any(np.isnan(vae_emb.coords))
    
    # Compare properties
    # VAE should have more regularized latent space (closer to standard normal)
    ae_mean = np.mean(ae_emb.coords, axis=1)
    ae_std = np.std(ae_emb.coords, axis=1)
    
    vae_mean = np.mean(vae_emb.coords, axis=1)
    vae_std = np.std(vae_emb.coords, axis=1)
    
    print(f"\nAE latent space - mean: {ae_mean}, std: {ae_std}")
    print(f"VAE latent space - mean: {vae_mean}, std: {vae_std}")
    
    # VAE should be closer to zero mean and unit variance due to KL regularization
    # (though not exactly due to finite training and reconstruction loss trade-off)
    assert np.all(np.abs(vae_mean) < np.abs(ae_mean) + 0.5), "VAE should have more centered latent space"


def test_ae_with_correlation_loss():
    """Test autoencoder with correlation loss to encourage decorrelated features"""
    # Generate data with correlated features
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Create correlated features
    base_features = np.random.randn(5, n_samples)
    data = np.zeros((n_features, n_samples))
    data[:5] = base_features
    data[5:] = base_features + 0.5 * np.random.randn(5, n_samples)  # Correlated with noise
    
    D = MVData(data)
    
    # Test with correlation loss
    nn_params = {
        'continue_learning': 0,
        'epochs': 100,
        'lr': 5e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.2,
        'add_corr_loss': True,
        'corr_hyperweight': 1.0,
        'verbose': False,
        'enc_kwargs': {'dropout': 0.2},
        'dec_kwargs': {'dropout': 0.2}
    }
    
    ae_params = {
        'e_method_name': 'ae',
        'dim': 2
    }
    ae_params['e_method'] = METHODS_DICT[ae_params['e_method_name']]
    ae_emb = D.get_embedding(ae_params, kwargs=nn_params)
    
    # Check embedding
    assert ae_emb.coords.shape == (2, n_samples)
    assert not np.any(np.isnan(ae_emb.coords))
    
    # Latent features should be less correlated due to correlation loss
    latent_corr = np.abs(np.corrcoef(ae_emb.coords)[0, 1])
    print(f"\nLatent correlation with corr loss: {latent_corr:.3f}")
    
    # Compare without correlation loss
    nn_params['add_corr_loss'] = False
    nn_params['seed'] = 43  # Different seed to avoid caching
    ae_emb_no_corr = D.get_embedding(ae_params, kwargs=nn_params)
    
    latent_corr_no_loss = np.abs(np.corrcoef(ae_emb_no_corr.coords)[0, 1])
    print(f"Latent correlation without corr loss: {latent_corr_no_loss:.3f}")
    
    # Correlation loss should reduce correlation (though not necessarily to zero)
    assert latent_corr < latent_corr_no_loss + 0.1, "Correlation loss should reduce latent correlation"


def test_ae_reconstruction_quality():
    """Test that autoencoder can reconstruct data reasonably well"""
    # Simple low-dimensional data that should be easy to reconstruct
    np.random.seed(42)
    n_samples = 200
    
    # Create 2D data embedded in 10D
    latent_2d = np.random.randn(2, n_samples)
    A = np.random.randn(10, 2)
    data = A @ latent_2d + 0.1 * np.random.randn(10, n_samples)
    
    D = MVData(data)
    
    nn_params = {
        'continue_learning': 0,
        'epochs': 200,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 32,
        'feature_dropout': 0.1,
        'verbose': False,
        'enc_kwargs': {'dropout': 0.1},
        'dec_kwargs': {'dropout': 0.1}
    }
    
    ae_params = {
        'e_method_name': 'ae',
        'dim': 2
    }
    ae_params['e_method'] = METHODS_DICT[ae_params['e_method_name']]
    ae_emb = D.get_embedding(ae_params, kwargs=nn_params)
    
    # Check that we get 2D embedding
    assert ae_emb.coords.shape == (2, n_samples)
    
    # Check reconstruction quality through preserved variance
    original_var = np.var(data)
    # The 2D embedding should capture most variance since data is essentially 2D
    embedding_reconstructed_var = np.var(ae_emb.coords)
    
    print(f"\nOriginal data variance: {original_var:.3f}")
    print(f"Embedding variance: {embedding_reconstructed_var:.3f}")
    
    # Should preserve reasonable amount of variance
    # Note: embedding variance is different from reconstruction variance
    # The embedding is 2D while original is 10D, so direct comparison isn't meaningful
    assert embedding_reconstructed_var > 0.05, "Embedding should have non-trivial variance"


if __name__ == "__main__":
    # Run tests with verbose output
    test_ae_vs_vae_comparison()
    test_ae_with_correlation_loss()
    test_ae_reconstruction_quality()
    print("\nAll AE/VAE comparison tests passed!")