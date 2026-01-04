"""Tests for the modular loss system in flexible autoencoders."""

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

from driada.dim_reduction.losses import (
    AELoss, LossRegistry, ReconstructionLoss, CorrelationLoss,
    OrthogonalityLoss, BetaVAELoss, TCVAELoss, SparsityLoss,
    ContractiveLoss, MMDLoss, FactorVAELoss
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLossRegistry:
    """Test the loss registry system."""
    
    def test_registry_initialization(self):
        """Test that registry initializes with default losses."""
        registry = LossRegistry()
        
        # Check default losses are registered
        assert "reconstruction" in registry.losses
        assert "correlation" in registry.losses
        assert "orthogonality" in registry.losses
        assert "beta_vae" in registry.losses
        assert "tc_vae" in registry.losses
        assert "sparse" in registry.losses
        assert "contractive" in registry.losses
        assert "mmd" in registry.losses
    
    def test_create_loss(self):
        """Test creating losses from registry."""
        registry = LossRegistry()
        
        # Create reconstruction loss
        loss = registry.create("reconstruction", weight=2.0)
        assert isinstance(loss, ReconstructionLoss)
        assert loss.weight == 2.0
        
        # Create correlation loss
        loss = registry.create("correlation", weight=0.5)
        assert isinstance(loss, CorrelationLoss)
        assert loss.weight == 0.5
    
    def test_register_custom_loss(self):
        """Test registering a custom loss."""
        registry = LossRegistry()
        
        class CustomLoss(AELoss):
            def compute(self, code, recon, inputs, **kwargs):
                return torch.tensor(1.0)
        
        registry.register("custom", CustomLoss)
        assert "custom" in registry.losses
        
        loss = registry.create("custom", weight=3.0)
        assert isinstance(loss, CustomLoss)
        assert loss.weight == 3.0
    
    def test_invalid_loss_name(self):
        """Test error on invalid loss name."""
        registry = LossRegistry()
        
        with pytest.raises(ValueError, match="Unknown loss"):
            registry.create("nonexistent")
    
    def test_invalid_loss_class(self):
        """Test error on registering invalid loss class."""
        registry = LossRegistry()
        
        class NotALoss:
            pass
        
        with pytest.raises(ValueError, match="must inherit from AELoss"):
            registry.register("invalid", NotALoss)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestReconstructionLoss:
    """Test reconstruction loss."""
    
    def test_mse_loss(self):
        """Test MSE reconstruction loss."""
        loss = ReconstructionLoss(loss_type="mse", weight=1.0)
        
        inputs = torch.randn(10, 20)
        recon = inputs + 0.1 * torch.randn(10, 20)
        code = torch.randn(10, 5)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val.shape == ()  # Scalar
        assert loss_val > 0
    
    def test_bce_loss(self):
        """Test BCE reconstruction loss."""
        loss = ReconstructionLoss(loss_type="bce", weight=1.0)
        
        inputs = torch.sigmoid(torch.randn(10, 20))
        recon = torch.sigmoid(torch.randn(10, 20))
        code = torch.randn(10, 5)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val.shape == ()
        assert loss_val > 0
    
    def test_invalid_loss_type(self):
        """Test error on invalid loss type."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            ReconstructionLoss(loss_type="invalid")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCorrelationLoss:
    """Test correlation loss."""
    
    def test_uncorrelated_features(self):
        """Test loss on uncorrelated features."""
        loss = CorrelationLoss(weight=1.0)
        
        # Create uncorrelated features
        torch.manual_seed(42)
        code = torch.randn(100, 5)
        recon = torch.randn(100, 20)
        inputs = torch.randn(100, 20)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val.shape == ()
        assert loss_val < 0.2  # Should be low for uncorrelated
    
    def test_correlated_features(self):
        """Test loss on correlated features."""
        loss = CorrelationLoss(weight=1.0)
        
        # Create highly correlated features
        base = torch.randn(100, 1)
        code = torch.cat([base, base * 0.9, base * 0.8], dim=1)
        recon = torch.randn(100, 20)
        inputs = torch.randn(100, 20)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val.shape == ()
        assert loss_val > 0.5  # Should be high for correlated
    
    def test_single_feature(self):
        """Test with single latent feature."""
        loss = CorrelationLoss(weight=1.0)
        
        code = torch.randn(100, 1)
        recon = torch.randn(100, 20)
        inputs = torch.randn(100, 20)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val == 0  # No correlation for single feature


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOrthogonalityLoss:
    """Test orthogonality loss."""
    
    def test_no_external_data(self):
        """Test loss without external data."""
        loss = OrthogonalityLoss(external_data=None, weight=1.0)
        
        code = torch.randn(10, 5)
        recon = torch.randn(10, 20)
        inputs = torch.randn(10, 20)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val == 0
    
    def test_with_external_data(self):
        """Test loss with external data."""
        external_data = np.random.randn(3, 100)
        loss = OrthogonalityLoss(external_data=external_data, weight=1.0)
        
        code = torch.randn(10, 5)
        recon = torch.randn(10, 20)
        inputs = torch.randn(10, 20)
        indices = torch.arange(10)
        
        loss_val = loss.compute(code, recon, inputs, indices=indices)
        assert loss_val.shape == ()
        assert loss_val >= 0
    
    def test_correlated_with_external(self):
        """Test loss when code correlates with external data."""
        # Create external data
        external_data = np.random.randn(2, 100)
        loss = OrthogonalityLoss(external_data=external_data, weight=1.0)
        
        # Create code that correlates with external data
        indices = torch.arange(10)
        ext_batch = torch.tensor(external_data[:, :10].T, dtype=torch.float32)
        code = torch.cat([ext_batch, ext_batch * 0.9], dim=1)
        
        recon = torch.randn(10, 20)
        inputs = torch.randn(10, 20)
        
        loss_val = loss.compute(code, recon, inputs, indices=indices)
        assert loss_val > 0.5  # Should be high when correlated


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBetaVAELoss:
    """Test β-VAE loss."""
    
    def test_standard_kld(self):
        """Test standard KL divergence computation."""
        loss = BetaVAELoss(beta=1.0, weight=1.0)
        
        batch_size = 10
        latent_dim = 5
        
        mu = torch.randn(batch_size, latent_dim) * 0.1
        log_var = torch.randn(batch_size, latent_dim) * 0.1 - 1
        code = torch.randn(batch_size, latent_dim)
        recon = torch.randn(batch_size, 20)
        inputs = torch.randn(batch_size, 20)
        
        loss_val = loss.compute(code, recon, inputs, mu=mu, log_var=log_var)
        assert loss_val.shape == ()
        assert loss_val > 0
    
    def test_beta_scaling(self):
        """Test that beta parameter scales the loss."""
        batch_size = 10
        latent_dim = 5
        
        mu = torch.randn(batch_size, latent_dim)
        log_var = torch.randn(batch_size, latent_dim)
        code = torch.randn(batch_size, latent_dim)
        recon = torch.randn(batch_size, 20)
        inputs = torch.randn(batch_size, 20)
        
        # Test different beta values
        loss1 = BetaVAELoss(beta=1.0, weight=1.0)
        loss4 = BetaVAELoss(beta=4.0, weight=1.0)
        
        val1 = loss1.compute(code, recon, inputs, mu=mu, log_var=log_var)
        val4 = loss4.compute(code, recon, inputs, mu=mu, log_var=log_var)
        
        assert torch.allclose(val4, val1 * 4.0, rtol=1e-5)
    
    def test_missing_parameters(self):
        """Test error when mu/log_var not provided."""
        loss = BetaVAELoss(beta=1.0, weight=1.0)
        
        code = torch.randn(10, 5)
        recon = torch.randn(10, 20)
        inputs = torch.randn(10, 20)
        
        with pytest.raises(ValueError, match="requires mu and log_var"):
            loss.compute(code, recon, inputs)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSparsityLoss:
    """Test sparsity loss."""
    
    def test_sparse_code(self):
        """Test loss on sparse code."""
        loss = SparsityLoss(sparsity_target=0.05, weight=1.0)
        
        # Create sparse code (mostly zeros)
        code = torch.zeros(100, 10)
        code[torch.rand(100, 10) < 0.05] = 1.0
        
        recon = torch.randn(100, 20)
        inputs = torch.randn(100, 20)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val.shape == ()
        assert loss_val < 0.2  # Should be relatively low when matching target
    
    def test_dense_code(self):
        """Test loss on dense code."""
        loss = SparsityLoss(sparsity_target=0.05, weight=1.0)
        
        # Create dense code (mostly ones)
        code = torch.ones(100, 10)
        
        recon = torch.randn(100, 20)
        inputs = torch.randn(100, 20)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val.shape == ()
        assert loss_val > 1.0  # Should be high when far from target


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMMDLoss:
    """Test MMD loss."""
    
    def test_standard_normal_code(self):
        """Test loss when code matches standard normal."""
        loss = MMDLoss(mmd_weight=1.0, weight=1.0)
        
        # Code from standard normal
        torch.manual_seed(42)
        code = torch.randn(100, 5)
        recon = torch.randn(100, 20)
        inputs = torch.randn(100, 20)
        
        loss_val = loss.compute(code, recon, inputs)
        assert loss_val.shape == ()
        assert loss_val < 0.1  # Should be low when matching prior
    
    def test_shifted_code(self):
        """Test loss when code is shifted from standard normal."""
        loss = MMDLoss(mmd_weight=1.0, weight=1.0)
        
        # Get baseline loss for standard normal
        torch.manual_seed(42)
        normal_code = torch.randn(100, 5)
        normal_loss = loss.compute(normal_code, torch.randn(100, 20), torch.randn(100, 20))
        
        # Code shifted from standard normal
        shifted_code = torch.randn(100, 5) + 2.0  # Shifted mean
        recon = torch.randn(100, 20)
        inputs = torch.randn(100, 20)
        
        shifted_loss = loss.compute(shifted_code, recon, inputs)
        assert shifted_loss.shape == ()
        # Shifted code should have higher loss than normal code
        assert shifted_loss > normal_loss
        assert shifted_loss > 0.1  # Should be noticeably higher than baseline