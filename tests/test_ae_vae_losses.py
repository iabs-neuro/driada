"""
Detailed comparison of AE vs VAE losses and reconstruction quality.
"""

import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from src.driada.dim_reduction.data import MVData
from src.driada.dim_reduction.dr_base import METHODS_DICT


def test_ae_vae_loss_comparison():
    """Compare reconstruction losses between AE and VAE"""
    # Generate swiss roll data
    n_samples = 1000
    data, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=42)
    D = MVData(data.T)
    
    # Common neural network parameters - more epochs for better convergence
    nn_params_base = {
        'continue_learning': 0,
        'epochs': 300,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.2,
        'verbose': True,  # Enable to see losses
        'enc_kwargs': {'dropout': 0.2},
        'dec_kwargs': {'dropout': 0.2},
        'train_size': 0.8,
        'log_every': 50
    }
    
    print("\n" + "="*60)
    print("AUTOENCODER (AE) TRAINING:")
    print("="*60)
    
    # Train regular autoencoder
    ae_params = {
        'e_method_name': 'ae',
        'dim': 2
    }
    ae_params['e_method'] = METHODS_DICT[ae_params['e_method_name']]
    ae_emb = D.get_embedding(ae_params, kwargs=nn_params_base.copy())
    
    # Get AE final loss
    ae_final_loss = ae_emb.nn_loss if hasattr(ae_emb, 'nn_loss') else None
    
    print("\n" + "="*60)
    print("VARIATIONAL AUTOENCODER (VAE) TRAINING:")
    print("="*60)
    
    # Train variational autoencoder
    vae_params = {
        'e_method_name': 'vae',
        'dim': 2
    }
    vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
    vae_nn_params = nn_params_base.copy()
    vae_nn_params['kld_weight'] = 0.1  # Control KL divergence weight
    vae_emb = D.get_embedding(vae_params, kwargs=vae_nn_params)
    
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY COMPARISON:")
    print("="*60)
    
    # Compare reconstruction quality by computing distances in latent space
    # and checking if they preserve distances in original space
    from src.driada.dim_reduction.manifold_metrics import (
        knn_preservation_rate, trustworthiness, continuity, stress
    )
    
    # Evaluate both methods
    k = 15
    ae_knn = knn_preservation_rate(data, ae_emb.coords.T, k=k)
    vae_knn = knn_preservation_rate(data, vae_emb.coords.T, k=k)
    
    ae_trust = trustworthiness(data, ae_emb.coords.T, k=k)
    vae_trust = trustworthiness(data, vae_emb.coords.T, k=k)
    
    ae_cont = continuity(data, ae_emb.coords.T, k=k)
    vae_cont = continuity(data, vae_emb.coords.T, k=k)
    
    ae_stress = stress(data, ae_emb.coords.T, normalized=True)
    vae_stress = stress(data, vae_emb.coords.T, normalized=True)
    
    print(f"\nNeighborhood preservation (k={k}):")
    print(f"  AE:  {ae_knn:.3f}")
    print(f"  VAE: {vae_knn:.3f}")
    print(f"  {'AE' if ae_knn > vae_knn else 'VAE'} preserves neighborhoods better")
    
    print(f"\nTrustworthiness:")
    print(f"  AE:  {ae_trust:.3f}")
    print(f"  VAE: {vae_trust:.3f}")
    print(f"  {'AE' if ae_trust > vae_trust else 'VAE'} is more trustworthy")
    
    print(f"\nContinuity:")
    print(f"  AE:  {ae_cont:.3f}")
    print(f"  VAE: {vae_cont:.3f}")
    print(f"  {'AE' if ae_cont > vae_cont else 'VAE'} has better continuity")
    
    print(f"\nStress (lower is better):")
    print(f"  AE:  {ae_stress:.3f}")
    print(f"  VAE: {vae_stress:.3f}")
    print(f"  {'AE' if ae_stress < vae_stress else 'VAE'} has lower stress")
    
    # Test passes if both produce valid embeddings
    assert ae_emb.coords.shape == (2, n_samples)
    assert vae_emb.coords.shape == (2, n_samples)
    
    # Return metrics for further analysis
    return {
        'ae': {'knn': ae_knn, 'trust': ae_trust, 'cont': ae_cont, 'stress': ae_stress},
        'vae': {'knn': vae_knn, 'trust': vae_trust, 'cont': vae_cont, 'stress': vae_stress}
    }


def test_vae_kl_weight_effect():
    """Test how KL divergence weight affects VAE performance"""
    # Use smaller dataset for faster testing
    n_samples = 500
    data = np.random.randn(10, n_samples)  # 10D random data
    D = MVData(data)
    
    kl_weights = [0.0, 0.1, 1.0, 10.0]
    results = []
    
    print("\n" + "="*60)
    print("VAE KL DIVERGENCE WEIGHT EFFECT:")
    print("="*60)
    
    for kl_weight in kl_weights:
        print(f"\nTesting KL weight = {kl_weight}")
        
        nn_params = {
            'continue_learning': 0,
            'epochs': 100,
            'lr': 1e-3,
            'seed': 42,
            'batch_size': 64,
            'feature_dropout': 0.2,
            'verbose': False,
            'kld_weight': kl_weight,
            'enc_kwargs': {'dropout': 0.2},
            'dec_kwargs': {'dropout': 0.2}
        }
        
        vae_params = {
            'e_method_name': 'vae',
            'dim': 2
        }
        vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
        vae_emb = D.get_embedding(vae_params, kwargs=nn_params)
        
        # Measure latent space properties
        latent_mean = np.mean(vae_emb.coords, axis=1)
        latent_std = np.std(vae_emb.coords, axis=1)
        
        results.append({
            'kl_weight': kl_weight,
            'mean': latent_mean,
            'std': latent_std,
            'mean_norm': np.linalg.norm(latent_mean),
            'std_diff_from_1': np.mean(np.abs(latent_std - 1.0))
        })
        
        print(f"  Latent mean norm: {results[-1]['mean_norm']:.3f}")
        print(f"  Latent std diff from 1: {results[-1]['std_diff_from_1']:.3f}")
    
    # Higher KL weight should push latent space closer to standard normal
    # (mean closer to 0, std closer to 1)
    assert results[0]['mean_norm'] > results[-1]['mean_norm'], \
        "Higher KL weight should center the latent space"
    assert results[0]['std_diff_from_1'] > results[-1]['std_diff_from_1'], \
        "Higher KL weight should normalize the latent variance"
    
    print("\n✓ KL weight correctly regularizes the latent space")


def test_ae_vae_on_simple_data():
    """Compare AE and VAE on simple, low-dimensional data where reconstruction should be easier"""
    # Create simple 2D data embedded in 10D
    np.random.seed(42)
    n_samples = 500
    
    # Generate 2D latent factors
    theta = np.linspace(0, 4*np.pi, n_samples)
    latent_2d = np.array([np.cos(theta), np.sin(theta)])
    
    # Embed in 10D with random linear transformation
    A = np.random.randn(10, 2)
    data = A @ latent_2d + 0.05 * np.random.randn(10, n_samples)
    
    D = MVData(data)
    
    nn_params = {
        'continue_learning': 0,
        'epochs': 200,
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.1,
        'verbose': False,
        'enc_kwargs': {'dropout': 0.1},
        'dec_kwargs': {'dropout': 0.1}
    }
    
    print("\n" + "="*60)
    print("SIMPLE DATA RECONSTRUCTION TEST:")
    print("="*60)
    
    # Train AE
    ae_params = {'e_method_name': 'ae', 'dim': 2}
    ae_params['e_method'] = METHODS_DICT[ae_params['e_method_name']]
    ae_emb = D.get_embedding(ae_params, kwargs=nn_params.copy())
    
    # Train VAE
    vae_params = {'e_method_name': 'vae', 'dim': 2}
    vae_params['e_method'] = METHODS_DICT[vae_params['e_method_name']]
    vae_nn_params = nn_params.copy()
    vae_nn_params['kld_weight'] = 0.01  # Low KL weight for better reconstruction
    vae_emb = D.get_embedding(vae_params, kwargs=vae_nn_params)
    
    # Check if embeddings preserve the circular structure
    # Original data lies on a circle in 2D latent space
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(data.T).T
    
    # Compute correlation with true latent factors
    # (up to rotation/reflection)
    def best_correlation(embedding, true_latent):
        """Find best correlation after optimal alignment"""
        corrs = []
        for i in range(2):
            for j in range(2):
                corr1 = np.abs(np.corrcoef(embedding[i], true_latent[j])[0,1])
                corr2 = np.abs(np.corrcoef(embedding[i], -true_latent[j])[0,1])
                corrs.extend([corr1, corr2])
        return max(corrs)
    
    ae_corr = best_correlation(ae_emb.coords, latent_2d)
    vae_corr = best_correlation(vae_emb.coords, latent_2d)
    pca_corr = best_correlation(pca_embedding, latent_2d)
    
    print(f"\nCorrelation with true latent factors:")
    print(f"  PCA: {pca_corr:.3f} (linear baseline)")
    print(f"  AE:  {ae_corr:.3f}")
    print(f"  VAE: {vae_corr:.3f}")
    
    # Both should recover structure reasonably well
    assert ae_corr > 0.7, "AE should recover latent structure"
    assert vae_corr > 0.7, "VAE should recover latent structure"
    
    print("\n✓ Both AE and VAE successfully recover latent structure")


if __name__ == "__main__":
    # Run comparison
    import pytest
    pytest.main([__file__, "-v", "-s"])