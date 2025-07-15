"""
Test to verify VAE is working correctly after fixing the sigmoid activation bug.
"""

import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from src.driada.dim_reduction.data import MVData
from src.driada.dim_reduction.dr_base import METHODS_DICT
from src.driada.dim_reduction.manifold_metrics import (
    knn_preservation_rate, manifold_preservation_score
)


def test_vae_latent_space_properties():
    """Test that VAE learns proper latent distribution"""
    # Generate simple data
    np.random.seed(42)
    n_samples = 1000
    data = np.random.randn(10, n_samples)
    D = MVData(data)
    
    # Train VAE with moderate KL weight
    nn_params = {
        'continue_learning': 0,
        'epochs': 200,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.1,
        'verbose': True,
        'kld_weight': 0.1,
        'log_every': 50,
        'enc_kwargs': {'dropout': 0.1},
        'dec_kwargs': {'dropout': 0.1}
    }
    
    vae_params = {
        'e_method_name': 'vae',
        'dim': 2
    }
    vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
    
    print("\n" + "="*60)
    print("TESTING FIXED VAE:")
    print("="*60)
    
    vae_emb = D.get_embedding(vae_params, kwargs=nn_params)
    
    # Check latent space properties
    latent_mean = np.mean(vae_emb.coords, axis=1)
    latent_std = np.std(vae_emb.coords, axis=1)
    
    print(f"\nLatent space statistics:")
    print(f"  Mean: {latent_mean}")
    print(f"  Std:  {latent_std}")
    
    # With proper VAE, mean should be close to 0 and std close to 1
    assert np.all(np.abs(latent_mean) < 0.5), f"VAE mean {latent_mean} not centered"
    assert np.all(np.abs(latent_std - 1.0) < 0.5), f"VAE std {latent_std} not normalized"
    
    print("✓ VAE latent space is properly regularized")


def test_vae_reconstruction_quality():
    """Test VAE reconstruction on structured data"""
    # Create swiss roll
    n_samples = 500
    data, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    D = MVData(data.T)
    
    # Train VAE with low KL weight for better reconstruction
    nn_params = {
        'continue_learning': 0,
        'epochs': 200,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.1,
        'verbose': False,
        'kld_weight': 0.01,  # Low weight prioritizes reconstruction
        'enc_kwargs': {'dropout': 0.1},
        'dec_kwargs': {'dropout': 0.1}
    }
    
    vae_params = {
        'e_method_name': 'vae',
        'dim': 2
    }
    vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
    vae_emb = D.get_embedding(vae_params, kwargs=nn_params)
    
    # Compare with AE
    ae_params = {
        'e_method_name': 'ae',
        'dim': 2
    }
    ae_params['e_method'] = METHODS_DICT[ae_params['e_method_name']]
    ae_nn_params = nn_params.copy()
    ae_nn_params.pop('kld_weight', None)  # Remove VAE-specific parameter
    ae_emb = D.get_embedding(ae_params, kwargs=ae_nn_params)
    
    # Evaluate preservation
    k = 10
    vae_knn = knn_preservation_rate(data, vae_emb.coords.T, k=k)
    ae_knn = knn_preservation_rate(data, ae_emb.coords.T, k=k)
    
    print(f"\nNeighborhood preservation (k={k}):")
    print(f"  AE:  {ae_knn:.3f}")
    print(f"  VAE: {vae_knn:.3f}")
    
    # VAE should now achieve reasonable preservation (not as good as AE but not terrible)
    assert vae_knn > 0.1, f"VAE preservation {vae_knn:.3f} is too low"
    
    # Get full metrics
    vae_metrics = manifold_preservation_score(data, vae_emb.coords.T, k_neighbors=k)
    ae_metrics = manifold_preservation_score(data, ae_emb.coords.T, k_neighbors=k)
    
    print(f"\nFull metrics comparison:")
    print(f"  Metric          AE      VAE")
    print(f"  ---------      ----    ----")
    for metric in ['knn_preservation', 'trustworthiness', 'continuity', 'overall_score']:
        print(f"  {metric:<14} {ae_metrics[metric]:0.3f}   {vae_metrics[metric]:0.3f}")
    
    # VAE should have positive trustworthiness and continuity now
    assert vae_metrics['trustworthiness'] > 0, "VAE trustworthiness should be positive"
    assert vae_metrics['continuity'] > 0, "VAE continuity should be positive"
    
    print("\n✓ VAE now shows reasonable reconstruction quality")


def test_vae_on_simple_manifold():
    """Test VAE on simple circular data"""
    # Generate circular data
    np.random.seed(42)
    n_points = 200
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # 2D circle embedded in 5D
    circle_2d = np.array([np.cos(theta), np.sin(theta)])
    A = np.random.randn(5, 2)
    data = A @ circle_2d + 0.05 * np.random.randn(5, n_points)
    
    D = MVData(data)
    
    # Train VAE
    nn_params = {
        'continue_learning': 0,
        'epochs': 300,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 32,
        'feature_dropout': 0.05,
        'verbose': False,
        'kld_weight': 0.01,
        'enc_kwargs': {'dropout': 0.05},
        'dec_kwargs': {'dropout': 0.05}
    }
    
    vae_params = {
        'e_method_name': 'vae',
        'dim': 2
    }
    vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
    vae_emb = D.get_embedding(vae_params, kwargs=nn_params)
    
    # Check if circular structure is preserved
    from src.driada.dim_reduction.manifold_metrics import circular_structure_preservation
    
    circular_metrics = circular_structure_preservation(vae_emb.coords.T, true_angles=theta)
    
    print(f"\nVAE circular structure preservation:")
    print(f"  Distance CV: {circular_metrics['distance_cv']:.3f}")
    print(f"  Consecutive preservation: {circular_metrics['consecutive_preservation']:.3f}")
    if 'circular_correlation' in circular_metrics:
        print(f"  Circular correlation: {circular_metrics['circular_correlation']:.3f}")
    
    # Should preserve some structure
    assert circular_metrics['consecutive_preservation'] > 0.3, "VAE should preserve some circular structure"
    
    print("\n✓ VAE preserves basic manifold structure")


def test_vae_encoder_outputs():
    """Directly test that VAE encoder outputs are unconstrained"""
    # Create simple VAE
    from src.driada.dim_reduction.neural import VAE
    
    device = torch.device("cpu")
    model = VAE(orig_dim=10, inter_dim=64, code_dim=2, device=device)
    
    # Generate random input
    x = torch.randn(32, 10)  # batch_size=32, input_dim=10
    
    # Get encoder output
    with torch.no_grad():
        code, mu, log_var = model.get_code(x)
    
    print(f"\nVAE encoder output ranges:")
    print(f"  μ range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")
    print(f"  log σ² range: [{log_var.min().item():.3f}, {log_var.max().item():.3f}]")
    
    # Outputs should NOT be constrained to [0,1]
    assert mu.min() < 0 or mu.max() > 1, "Mean should be unconstrained"
    assert log_var.min() < 0 or log_var.max() > 1, "Log variance should be unconstrained"
    
    print("\n✓ VAE encoder outputs are properly unconstrained")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])