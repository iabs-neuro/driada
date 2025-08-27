"""Tests for flexible autoencoder architectures."""

import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from driada.dim_reduction.flexible_ae import ModularAutoencoder, FlexibleVAE
from driada.dim_reduction.losses import LossRegistry


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestModularAutoencoder:
    """Test modular autoencoder architecture."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        ae = ModularAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        assert ae.input_dim == 20
        assert ae.latent_dim == 5
        assert ae.hidden_dim == 10
        assert len(ae.losses) == 1  # Default reconstruction loss
        assert ae.losses[0].__class__.__name__ == "ReconstructionLoss"
    
    def test_initialization_with_losses(self):
        """Test initialization with custom loss components."""
        loss_components = [
            {"name": "reconstruction", "weight": 1.0},
            {"name": "correlation", "weight": 0.1},
            {"name": "sparse", "weight": 0.05, "sparsity_target": 0.1}
        ]
        
        ae = ModularAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10,
            loss_components=loss_components
        )
        
        assert len(ae.losses) == 3
        assert ae.losses[0].__class__.__name__ == "ReconstructionLoss"
        assert ae.losses[1].__class__.__name__ == "CorrelationLoss"
        assert ae.losses[2].__class__.__name__ == "SparsityLoss"
        assert ae.losses[1].weight == 0.1
        assert ae.losses[2].weight == 0.05
    
    def test_forward_pass(self):
        """Test forward pass through autoencoder."""
        ae = ModularAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        x = torch.randn(32, 20)
        recon = ae(x)
        
        assert recon.shape == (32, 20)
    
    def test_encode_decode(self):
        """Test encode and decode separately."""
        ae = ModularAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        x = torch.randn(32, 20)
        
        # Encode
        z = ae.encode(x)
        assert z.shape == (32, 5)
        
        # Decode
        recon = ae.decode(z)
        assert recon.shape == (32, 20)
    
    def test_compute_loss(self):
        """Test loss computation."""
        loss_components = [
            {"name": "reconstruction", "weight": 1.0},
            {"name": "correlation", "weight": 0.1}
        ]
        
        ae = ModularAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10,
            loss_components=loss_components
        )
        
        inputs = torch.randn(32, 20)
        total_loss, loss_dict = ae.compute_loss(inputs)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == ()
        assert "total_loss" in loss_dict
        assert "ReconstructionLoss_0" in loss_dict
        assert "CorrelationLoss_1" in loss_dict
        assert "ReconstructionLoss_0_weighted" in loss_dict
        assert "CorrelationLoss_1_weighted" in loss_dict
    
    
    def test_get_latent_representation(self):
        """Test getting latent representation."""
        ae = ModularAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        x = torch.randn(32, 20)
        latent = ae.get_latent_representation(x)
        
        assert isinstance(latent, np.ndarray)
        assert latent.shape == (5, 32)  # Transposed for DRIADA
    
    def test_device_handling(self):
        """Test device handling."""
        device = torch.device("cpu")
        ae = ModularAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10,
            device=device
        )
        
        assert ae.device == device
        assert next(ae.parameters()).device == device
    
    def test_encoder_decoder_config(self):
        """Test encoder/decoder configuration."""
        enc_config = {"dropout": 0.2}
        dec_config = {"dropout": 0.3}
        
        ae = ModularAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10,
            encoder_config=enc_config,
            decoder_config=dec_config
        )
        
        # Check dropout was set correctly
        assert ae.encoder.dropout.p == 0.2
        assert ae.decoder.dropout.p == 0.3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestFlexibleVAE:
    """Test flexible VAE architecture."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        vae = FlexibleVAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        assert vae.input_dim == 20
        assert vae.latent_dim == 5
        assert vae.hidden_dim == 10
        assert len(vae.losses) == 2  # Reconstruction + Beta-VAE
        assert vae.losses[0].__class__.__name__ == "ReconstructionLoss"
        assert vae.losses[1].__class__.__name__ == "BetaVAELoss"
    
    def test_initialization_with_losses(self):
        """Test initialization with custom loss components."""
        loss_components = [
            {"name": "reconstruction", "weight": 1.0},
            {"name": "beta_vae", "weight": 1.0, "beta": 4.0},
            {"name": "correlation", "weight": 0.1}
        ]
        
        vae = FlexibleVAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10,
            loss_components=loss_components
        )
        
        assert len(vae.losses) == 3
        assert vae.losses[1].__class__.__name__ == "BetaVAELoss"
        assert vae.losses[2].__class__.__name__ == "CorrelationLoss"
    
    def test_encode(self):
        """Test VAE encoding."""
        vae = FlexibleVAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        x = torch.randn(32, 20)
        z, mu, log_var = vae.encode(x)
        
        assert z.shape == (32, 5)
        assert mu.shape == (32, 5)
        assert log_var.shape == (32, 5)
    
    def test_reparameterize(self):
        """Test reparameterization trick."""
        vae = FlexibleVAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        mu = torch.zeros(32, 5)
        log_var = torch.zeros(32, 5)
        
        z = vae.reparameterize(mu, log_var)
        assert z.shape == (32, 5)
        
        # With zero mean and log_var, should be close to standard normal
        assert torch.abs(z.mean()) < 0.3  # Relaxed threshold for randomness
        assert torch.abs(z.std() - 1.0) < 0.3
    
    def test_forward_pass(self):
        """Test forward pass through VAE."""
        vae = FlexibleVAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        x = torch.randn(32, 20)
        recon, mu, log_var = vae(x)
        
        assert recon.shape == (32, 20)
        assert mu.shape == (32, 5)
        assert log_var.shape == (32, 5)
    
    def test_compute_loss(self):
        """Test VAE loss computation."""
        loss_components = [
            {"name": "reconstruction", "weight": 1.0},
            {"name": "beta_vae", "weight": 1.0, "beta": 4.0}
        ]
        
        vae = FlexibleVAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10,
            loss_components=loss_components
        )
        
        inputs = torch.randn(32, 20)
        total_loss, loss_dict = vae.compute_loss(inputs)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == ()
        assert "total_loss" in loss_dict
        assert "ReconstructionLoss_0" in loss_dict
        assert "BetaVAELoss_1" in loss_dict
    
    def test_get_latent_representation(self):
        """Test getting latent representation from VAE."""
        vae = FlexibleVAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10
        )
        
        x = torch.randn(32, 20)
        latent = vae.get_latent_representation(x)
        
        assert isinstance(latent, np.ndarray)
        assert latent.shape == (5, 32)  # Transposed for DRIADA
    
    def test_tc_vae_configuration(self):
        """Test TC-VAE configuration."""
        loss_components = [
            {"name": "reconstruction", "weight": 1.0},
            {"name": "tc_vae", "weight": 1.0, "alpha": 1.0, "beta": 5.0, "gamma": 1.0}
        ]
        
        vae = FlexibleVAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=10,
            loss_components=loss_components
        )
        
        assert len(vae.losses) == 2
        assert vae.losses[1].__class__.__name__ == "TCVAELoss"
        
        # Test that it can compute loss
        inputs = torch.randn(32, 20)
        total_loss, loss_dict = vae.compute_loss(inputs)
        assert "TCVAELoss_1" in loss_dict