"""
Advanced tests for Autoencoders (AE) and Variational Autoencoders (VAE).
Consolidates important tests from debugging session.
"""

import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from src.driada.dim_reduction.data import MVData
from src.driada.dim_reduction.dr_base import METHODS_DICT
from src.driada.dim_reduction.manifold_metrics import knn_preservation_rate


def test_vae_latent_space_regularization():
    """Test that VAE properly regularizes latent space to standard normal"""
    # Generate random data
    np.random.seed(42)
    n_samples = 1000
    data = np.random.randn(10, n_samples)
    D = MVData(data)
    
    # Train VAE with moderate KL weight
    nn_params = {
        'continue_learning': 0,
        'epochs': 100,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.1,
        'verbose': False,
        'kld_weight': 0.1,
        'enc_kwargs': {'dropout': 0.1},
        'dec_kwargs': {'dropout': 0.1}
    }
    
    vae_params = {'e_method_name': 'vae', 'dim': 2}
    vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
    vae_emb = D.get_embedding(vae_params, kwargs=nn_params)
    
    # Check latent space properties
    latent_mean = np.mean(vae_emb.coords, axis=1)
    latent_std = np.std(vae_emb.coords, axis=1)
    
    # VAE should regularize to approximately standard normal
    assert np.all(np.abs(latent_mean) < 0.5), f"VAE mean {latent_mean} not centered"
    assert np.all(np.abs(latent_std - 1.0) < 0.5), f"VAE std {latent_std} not normalized"


def test_ae_correlation_loss():
    """Test that correlation loss decorrelates latent features"""
    # Generate correlated data
    np.random.seed(42)
    n_samples = 500
    base_features = np.random.randn(5, n_samples)
    data = np.vstack([base_features, base_features + 0.5 * np.random.randn(5, n_samples)])
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
    
    ae_params = {'e_method_name': 'ae', 'dim': 2}
    ae_params['e_method'] = METHODS_DICT[ae_params['e_method_name']]
    ae_emb = D.get_embedding(ae_params, kwargs=nn_params)
    
    # Latent features should have low correlation
    latent_corr = np.abs(np.corrcoef(ae_emb.coords)[0, 1])
    assert latent_corr < 0.3, f"Correlation loss failed to decorrelate (corr={latent_corr:.3f})"


def test_ae_vs_vae_reconstruction_quality():
    """Compare reconstruction quality between AE and VAE"""
    # Generate swiss roll
    n_samples = 500
    data, _ = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    D = MVData(data.T)
    
    nn_params_base = {
        'continue_learning': 0,
        'epochs': 100,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.1,
        'verbose': False,
        'enc_kwargs': {'dropout': 0.1},
        'dec_kwargs': {'dropout': 0.1}
    }
    
    # Train AE
    ae_params = {'e_method_name': 'ae', 'dim': 2}
    ae_params['e_method'] = METHODS_DICT[ae_params['e_method_name']]
    ae_emb = D.get_embedding(ae_params, kwargs=nn_params_base.copy())
    
    # Train VAE with low KL weight for fair comparison
    vae_params = {'e_method_name': 'vae', 'dim': 2}
    vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
    vae_nn_params = nn_params_base.copy()
    vae_nn_params['kld_weight'] = 0.01
    vae_emb = D.get_embedding(vae_params, kwargs=vae_nn_params)
    
    # Compare neighborhood preservation
    k = 10
    ae_knn = knn_preservation_rate(data, ae_emb.coords.T, k=k)
    vae_knn = knn_preservation_rate(data, vae_emb.coords.T, k=k)
    
    # Both should achieve reasonable preservation
    assert ae_knn > 0.2, f"AE preservation too low: {ae_knn:.3f}"
    assert vae_knn > 0.1, f"VAE preservation too low: {vae_knn:.3f}"
    
    # AE typically preserves structure better due to no regularization
    # But difference shouldn't be extreme with low KL weight
    assert ae_knn > vae_knn - 0.2, "VAE performing unexpectedly poorly"


def test_vae_encoder_unconstrained():
    """Verify VAE encoder outputs are not constrained to [0,1]"""
    from src.driada.dim_reduction.neural import VAE
    
    device = torch.device("cpu")
    model = VAE(orig_dim=10, inter_dim=64, code_dim=2, device=device)
    
    # Random input
    x = torch.randn(32, 10)
    
    with torch.no_grad():
        code, mu, log_var = model.get_code(x)
    
    # Outputs should be unconstrained (not sigmoid-limited to [0,1])
    assert mu.min() < 0 or mu.max() > 1, "Mean should be unconstrained"
    assert log_var.min() < 0 or log_var.max() > 1, "Log variance should be unconstrained"