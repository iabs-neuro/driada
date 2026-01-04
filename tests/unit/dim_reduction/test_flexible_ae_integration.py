"""Integration tests for flexible_ae embedding method."""

import pytest
import numpy as np
from sklearn.datasets import make_swiss_roll

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from driada.dim_reduction.data import MVData
from driada.dim_reduction.manifold_metrics import knn_preservation_rate


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestFlexibleAEIntegration:
    """Integration tests for flexible_ae method."""
    
    def test_basic_ae(self):
        """Test basic autoencoder without additional losses."""
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(10, 100)
        mvdata = MVData(data)
        
        # Get embedding
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=3,
            architecture="ae",
            epochs=20,
            batch_size=32,
            verbose=False
        )
        
        assert emb.coords is not None
        assert emb.coords.shape == (3, 100)
    
    def test_ae_with_correlation_loss(self):
        """Test autoencoder with correlation loss."""
        # Generate correlated data
        np.random.seed(42)
        base = np.random.randn(5, 100)
        data = np.vstack([base, base + 0.5 * np.random.randn(5, 100)])
        mvdata = MVData(data)
        
        # Get embedding with correlation loss
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=3,
            architecture="ae",
            epochs=50,
            batch_size=32,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "correlation", "weight": 0.5}
            ],
            verbose=False
        )
        
        # Check that latent features are less correlated
        latent_corr = np.abs(np.corrcoef(emb.coords))
        off_diagonal = latent_corr[np.triu_indices_from(latent_corr, k=1)]
        assert np.mean(off_diagonal) < 0.5
    
    def test_ae_with_orthogonality_loss(self):
        """Test autoencoder with orthogonality loss."""
        # Generate data
        np.random.seed(42)
        data = np.random.randn(10, 100)
        external_data = np.random.randn(5, 100)
        mvdata = MVData(data)
        
        # Get embedding with orthogonality loss
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=3,
            architecture="ae",
            epochs=30,
            batch_size=32,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "orthogonality", "weight": 0.3, "external_data": external_data}
            ],
            verbose=False
        )
        
        assert emb.coords is not None
        assert emb.coords.shape == (3, 100)
    
    def test_basic_vae(self):
        """Test basic VAE."""
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(10, 100)
        mvdata = MVData(data)
        
        # Get VAE embedding
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=3,
            architecture="vae",
            epochs=30,
            batch_size=32,
            verbose=False
        )
        
        assert emb.coords is not None
        assert emb.coords.shape == (3, 100)
        
        # VAE should produce more regularized latent space
        latent_std = np.std(emb.coords, axis=1)
        assert np.all(latent_std < 3.0)  # Reasonably bounded
    
    def test_beta_vae(self):
        """Test β-VAE with custom beta."""
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(10, 100)
        mvdata = MVData(data)
        
        # Get β-VAE embedding
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=3,
            architecture="vae",
            epochs=30,
            batch_size=32,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "beta_vae", "weight": 1.0, "beta": 4.0}
            ],
            verbose=False
        )
        
        assert emb.coords is not None
        assert emb.coords.shape == (3, 100)
    
    def test_vae_with_sparsity(self):
        """Test VAE with sparsity constraint."""
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(10, 100)
        mvdata = MVData(data)
        
        # Get VAE with sparsity
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=5,
            architecture="vae",
            epochs=30,
            batch_size=32,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "beta_vae", "weight": 1.0, "beta": 1.0},
                {"name": "sparse", "weight": 0.1, "sparsity_target": 0.1}
            ],
            verbose=False
        )
        
        assert emb.coords is not None
        assert emb.coords.shape == (5, 100)
    
    def test_swiss_roll_preservation(self):
        """Test neighborhood preservation on Swiss roll."""
        # Generate Swiss roll
        n_samples = 500
        data, _ = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
        mvdata = MVData(data.T)
        
        # Get embedding with correlation loss for better structure
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=2,
            architecture="ae",
            epochs=100,
            lr=1e-3,
            batch_size=64,
            inter_dim=64,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "correlation", "weight": 0.1}
            ],
            verbose=False
        )
        
        # Check neighborhood preservation
        k = 10
        knn_rate = knn_preservation_rate(data, emb.coords.T, k=k)
        assert knn_rate > 0.2  # Reasonable preservation for nonlinear method
    
    def test_parameter_passing(self):
        """Test that all parameters are properly passed."""
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(10, 50)
        mvdata = MVData(data)
        
        # Test various parameters
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=2,
            architecture="ae",
            epochs=10,
            lr=5e-3,
            batch_size=16,
            seed=123,
            feature_dropout=0.3,
            train_size=0.7,
            inter_dim=32,
            enc_kwargs={"dropout": 0.1},
            dec_kwargs={"dropout": 0.2},
            log_every=5,
            verbose=False
        )
        
        assert emb.coords is not None
        assert emb.coords.shape == (2, 50)
        
        # Check that model was created with correct parameters
        assert emb.nnmodel.hidden_dim == 32
        assert emb.nnmodel.latent_dim == 2
        assert emb.nnmodel.encoder.dropout.p == 0.1
        assert emb.nnmodel.decoder.dropout.p == 0.2
    
    def test_multiple_losses_combination(self):
        """Test combining multiple loss components."""
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(10, 100)
        external = np.random.randn(3, 100)
        mvdata = MVData(data)
        
        # Combine multiple losses
        emb = mvdata.get_embedding(
            method="flexible_ae",
            dim=3,
            architecture="ae",
            epochs=50,
            batch_size=32,
            loss_components=[
                {"name": "reconstruction", "weight": 1.0},
                {"name": "correlation", "weight": 0.1},
                {"name": "orthogonality", "weight": 0.05, "external_data": external},
                {"name": "sparse", "weight": 0.01, "sparsity_target": 0.2}
            ],
            verbose=False
        )
        
        assert emb.coords is not None
        assert emb.coords.shape == (3, 100)
        
        # Should have applied all constraints
        latent_corr = np.abs(np.corrcoef(emb.coords))
        off_diagonal = latent_corr[np.triu_indices_from(latent_corr, k=1)]
        assert np.mean(off_diagonal) < 0.6  # Some decorrelation