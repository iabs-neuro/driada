"""Modular loss system for flexible autoencoders.

This module provides a flexible, extensible system for composing different loss
functions in autoencoder training. It supports standard reconstruction losses
as well as advanced disentanglement objectives like β-VAE, TC-VAE, and Factor-VAE.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any, Tuple
import logging

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


class AELoss(ABC):
    """Base class for all autoencoder loss components.
    
    Each loss component computes a specific objective (e.g., reconstruction,
    disentanglement, sparsity) and has an associated weight for balancing
    multiple objectives.
    """
    
    def __init__(self, weight: float = 1.0, **kwargs):
        """Initialize loss component.
        
        Parameters
        ----------
        weight : float, default=1.0
            Weight for this loss component when combining multiple losses.
        **kwargs
            Additional parameters specific to each loss type.
        """
        self._weight = weight
        self.kwargs = kwargs
    
    @abstractmethod
    def compute(
        self, 
        code: torch.Tensor, 
        recon: torch.Tensor, 
        inputs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute the loss value.
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation, shape (batch_size, code_dim).
        recon : torch.Tensor
            Reconstructed outputs, shape (batch_size, input_dim).
        inputs : torch.Tensor
            Original inputs, shape (batch_size, input_dim).
        **kwargs
            Additional tensors/parameters needed by specific losses
            (e.g., mu and log_var for VAE losses).
            
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        pass
    
    @property
    def weight(self) -> float:
        """Loss weight for balancing multiple objectives."""
        return self._weight
    
    @weight.setter
    def weight(self, value: float):
        """Set loss weight."""
        self._weight = value


class LossRegistry:
    """Registry for dynamically managing loss components."""
    
    def __init__(self):
        """Initialize the registry with default loss types."""
        self.losses: Dict[str, Type[AELoss]] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default loss implementations."""
        # Register all default losses
        defaults = {
            "reconstruction": ReconstructionLoss,
            "correlation": CorrelationLoss,
            "orthogonality": OrthogonalityLoss,
            "beta_vae": BetaVAELoss,
            "tc_vae": TCVAELoss,
            "factor_vae": FactorVAELoss,
            "sparse": SparsityLoss,
            "contractive": ContractiveLoss,
            "wasserstein": WassersteinLoss,
        }
        
        for name, loss_class in defaults.items():
            self.register(name, loss_class)
    
    def register(self, name: str, loss_class: Type[AELoss]):
        """Register a new loss type.
        
        Parameters
        ----------
        name : str
            Name identifier for the loss.
        loss_class : Type[AELoss]
            Loss class (must inherit from AELoss).
        """
        if not issubclass(loss_class, AELoss):
            raise ValueError(f"Loss class must inherit from AELoss, got {loss_class}")
        
        self.losses[name] = loss_class
        self._logger.debug(f"Registered loss '{name}' -> {loss_class.__name__}")
    
    def create(self, name: str, **kwargs) -> AELoss:
        """Create a loss instance by name.
        
        Parameters
        ----------
        name : str
            Registered name of the loss.
        **kwargs
            Parameters to pass to the loss constructor.
            
        Returns
        -------
        AELoss
            Instantiated loss component.
        """
        if name not in self.losses:
            raise ValueError(
                f"Unknown loss '{name}'. Available: {list(self.losses.keys())}"
            )
        
        return self.losses[name](**kwargs)


# Standard Losses

class ReconstructionLoss(AELoss):
    """Standard reconstruction loss (MSE or BCE)."""
    
    def __init__(self, loss_type: str = "mse", weight: float = 1.0):
        """Initialize reconstruction loss.
        
        Parameters
        ----------
        loss_type : str, default="mse"
            Type of reconstruction loss ("mse" or "bce").
        weight : float, default=1.0
            Loss weight.
        """
        super().__init__(weight=weight)
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss(reduction='mean')
        elif loss_type == "bce":
            self.criterion = nn.BCELoss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute reconstruction loss."""
        return self.criterion(recon, inputs)


class CorrelationLoss(AELoss):
    """Correlation loss to encourage decorrelated latent features."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize correlation loss.
        
        Parameters
        ----------
        weight : float, default=1.0
            Loss weight.
        """
        super().__init__(weight=weight)
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute average pairwise correlation amplitude in latent code."""
        # Handle single feature case
        if code.shape[1] == 1:
            return torch.tensor(0.0, device=code.device)
        
        # Transpose to have features as rows for corrcoef
        code_t = code.T
        
        # Compute correlation matrix
        corr = torch.corrcoef(code_t)
        
        # Average absolute pairwise correlation (excluding diagonal)
        n_features = corr.shape[0]
        off_diagonal_sum = torch.sum(torch.abs(corr)) - n_features
        n_pairs = n_features * (n_features - 1)
        
        avg_correlation = off_diagonal_sum / n_pairs if n_pairs > 0 else 0
        
        return avg_correlation


class OrthogonalityLoss(AELoss):
    """Orthogonality loss to minimize correlation with external data (MI proxy)."""
    
    def __init__(self, external_data: Optional[np.ndarray] = None, weight: float = 1.0):
        """Initialize orthogonality loss.
        
        Parameters
        ----------
        external_data : np.ndarray, optional
            External data to minimize correlation with, shape (n_features, n_samples).
        weight : float, default=1.0
            Loss weight.
        """
        super().__init__(weight=weight)
        self.external_data = external_data
        self._external_tensor = None
    
    def compute(self, code, recon, inputs, indices=None, **kwargs):
        """Compute correlation between latent code and external data."""
        if self.external_data is None:
            return torch.tensor(0.0, device=code.device)
        
        # Get batch indices
        if indices is None:
            batch_size = code.shape[0]
            indices = torch.arange(batch_size)
        
        # Convert external data to tensor if needed
        if self._external_tensor is None:
            self._external_tensor = torch.tensor(
                self.external_data, dtype=torch.float32
            )
        
        # Move to correct device
        if self._external_tensor.device != code.device:
            self._external_tensor = self._external_tensor.to(code.device)
        
        # Get relevant external data for this batch
        ext_batch = self._external_tensor[:, indices].T
        
        # Compute correlation between code and external data
        n_code = code.shape[1]
        n_ext = ext_batch.shape[1]
        
        # Concatenate for correlation computation
        combined = torch.cat([code, ext_batch], dim=1)
        corr = torch.corrcoef(combined.T)
        
        # Extract cross-correlation block
        cross_corr = corr[:n_code, n_code:]
        
        # Average absolute correlation
        avg_correlation = torch.mean(torch.abs(cross_corr))
        
        return avg_correlation


# VAE-based Disentanglement Losses

class BetaVAELoss(AELoss):
    """β-VAE loss for disentanglement via increased KL penalty."""
    
    def __init__(self, beta: float = 4.0, weight: float = 1.0):
        """Initialize β-VAE loss.
        
        Parameters
        ----------
        beta : float, default=4.0
            Beta parameter controlling KL penalty strength.
        weight : float, default=1.0
            Loss weight.
        """
        super().__init__(weight=weight)
        self.beta = beta
    
    def compute(self, code, recon, inputs, mu=None, log_var=None, **kwargs):
        """Compute β-weighted KL divergence loss."""
        if mu is None or log_var is None:
            raise ValueError("β-VAE loss requires mu and log_var")
        
        # KL divergence from N(0,1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / mu.shape[0]  # Average over batch
        
        return self.beta * kl_loss


class TCVAELoss(AELoss):
    """TC-VAE loss with decomposed ELBO for better disentanglement."""
    
    def __init__(
        self, 
        alpha: float = 1.0,
        beta: float = 1.0, 
        gamma: float = 1.0,
        weight: float = 1.0
    ):
        """Initialize TC-VAE loss.
        
        Parameters
        ----------
        alpha : float, default=1.0
            Weight for mutual information term.
        beta : float, default=1.0
            Weight for total correlation term.
        gamma : float, default=1.0
            Weight for dimension-wise KL term.
        weight : float, default=1.0
            Overall loss weight.
        """
        super().__init__(weight=weight)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def compute(self, code, recon, inputs, mu=None, log_var=None, **kwargs):
        """Compute TC-VAE loss with ELBO decomposition."""
        if mu is None or log_var is None:
            raise ValueError("TC-VAE loss requires mu and log_var")
        
        batch_size = mu.shape[0]
        latent_dim = mu.shape[1]
        
        # Standard KL divergence
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        # Log probability of code under the prior
        log_pz = self._log_density_gaussian(code, torch.zeros_like(code), torch.zeros_like(code))
        
        # Log probability of code under the posterior
        log_qz_given_x = self._log_density_gaussian(code, mu, log_var)
        
        # Log probability of code under the marginal posterior
        # Approximate with minibatch-weighted sampling
        log_qz = self._log_importance_weight_matrix(code, mu, log_var).logsumexp(dim=1) - np.log(batch_size)
        
        # Decomposed terms (sum over latent dimensions)
        mi = log_qz_given_x.sum(dim=1) - log_qz  # Mutual Information
        tc = log_qz - log_pz.sum(dim=1)  # Total Correlation
        dwkl = kl - mi - tc  # Dimension-wise KL
        
        # Weighted combination
        loss = self.alpha * mi.mean() + self.beta * tc.mean() + self.gamma * dwkl.mean()
        
        return loss
    
    def _log_density_gaussian(self, x, mu, log_var):
        """Compute log probability under a Gaussian."""
        normalization = -0.5 * (np.log(2 * np.pi) + log_var)
        inv_var = torch.exp(-log_var)
        log_density = normalization - 0.5 * ((x - mu).pow(2) * inv_var)
        return log_density
    
    def _log_importance_weight_matrix(self, batch, mu, log_var):
        """Compute importance weights for minibatch."""
        batch_size, latent_dim = batch.shape
        
        # Expand for broadcasting
        batch = batch.unsqueeze(1)  # (batch, 1, latent)
        mu = mu.unsqueeze(0)  # (1, batch, latent)
        log_var = log_var.unsqueeze(0)  # (1, batch, latent)
        
        # Compute log densities
        log_density = self._log_density_gaussian(batch, mu, log_var)
        
        # Sum over latent dimensions
        log_importance = log_density.sum(dim=2)
        
        return log_importance


class FactorVAELoss(AELoss):
    """Factor-VAE loss using a discriminator for disentanglement."""
    
    def __init__(
        self, 
        gamma: float = 10.0,
        discriminator_dims: Optional[list] = None,
        weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """Initialize Factor-VAE loss.
        
        Parameters
        ----------
        gamma : float, default=10.0
            Weight for the total correlation penalty.
        discriminator_dims : list, optional
            Hidden dimensions for discriminator network.
        weight : float, default=1.0
            Loss weight.
        device : torch.device, optional
            Device for discriminator.
        """
        super().__init__(weight=weight)
        self.gamma = gamma
        self.discriminator_dims = discriminator_dims or [256, 256]
        self.device = device or torch.device("cpu")
        self.discriminator = None
        self._latent_dim = None
    
    def _build_discriminator(self, latent_dim: int):
        """Build discriminator network."""
        layers = []
        in_dim = latent_dim
        
        for hidden_dim in self.discriminator_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 2))  # Binary classification
        
        self.discriminator = nn.Sequential(*layers).to(self.device)
        self._latent_dim = latent_dim
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute Factor-VAE discriminator-based loss."""
        batch_size = code.shape[0]
        latent_dim = code.shape[1]
        
        # Build discriminator if needed
        if self.discriminator is None:
            self._build_discriminator(latent_dim)
        
        # Create permuted code (break factorial structure)
        indices = torch.randperm(batch_size).to(code.device)
        permuted_code = code[indices]
        
        # Discriminator predictions
        real_pred = self.discriminator(code)
        fake_pred = self.discriminator(permuted_code.detach())
        
        # Discriminator loss (standard GAN loss)
        disc_loss = F.cross_entropy(real_pred, torch.ones(batch_size, dtype=torch.long).to(code.device))
        disc_loss += F.cross_entropy(fake_pred, torch.zeros(batch_size, dtype=torch.long).to(code.device))
        
        # Total correlation loss (fool discriminator)
        tc_loss = F.cross_entropy(real_pred, torch.zeros(batch_size, dtype=torch.long).to(code.device))
        
        return self.gamma * tc_loss


# Regularization Losses

class SparsityLoss(AELoss):
    """Sparsity loss to encourage sparse latent representations."""
    
    def __init__(self, sparsity_target: float = 0.05, weight: float = 1.0):
        """Initialize sparsity loss.
        
        Parameters
        ----------
        sparsity_target : float, default=0.05
            Target average activation level.
        weight : float, default=1.0
            Loss weight.
        """
        super().__init__(weight=weight)
        self.sparsity_target = sparsity_target
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute KL divergence between actual and target sparsity."""
        # Average activation per latent dimension
        avg_activation = torch.mean(code, dim=0)
        
        # KL divergence from target sparsity
        kl = self.sparsity_target * torch.log(self.sparsity_target / (avg_activation + 1e-8))
        kl += (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - avg_activation + 1e-8))
        
        return torch.sum(kl)


class ContractiveLoss(AELoss):
    """Contractive loss for robust representations via Jacobian penalty."""
    
    def __init__(self, lambda_c: float = 0.01, weight: float = 1.0):
        """Initialize contractive loss.
        
        Parameters
        ----------
        lambda_c : float, default=0.01
            Contraction strength parameter.
        weight : float, default=1.0
            Loss weight.
        """
        super().__init__(weight=weight)
        self.lambda_c = lambda_c
    
    def compute(self, code, recon, inputs, encoder=None, **kwargs):
        """Compute Frobenius norm of encoder Jacobian."""
        if encoder is None:
            raise ValueError("Contractive loss requires encoder module")
        
        # Enable gradient computation for inputs
        inputs = inputs.requires_grad_(True)
        
        # Forward through encoder
        code = encoder(inputs)
        
        # Compute Jacobian via backpropagation
        batch_size = inputs.shape[0]
        jacobian_norm = 0
        
        for i in range(code.shape[1]):
            # Gradient of i-th latent w.r.t. inputs
            grad = torch.autograd.grad(
                code[:, i].sum(), inputs, 
                retain_graph=True, create_graph=True
            )[0]
            
            # Frobenius norm squared
            jacobian_norm += torch.sum(grad ** 2) / batch_size
        
        return self.lambda_c * jacobian_norm


class WassersteinLoss(AELoss):
    """Wasserstein loss for better latent space interpolation."""
    
    def __init__(self, mmd_weight: float = 1.0, weight: float = 1.0):
        """Initialize Wasserstein loss.
        
        Parameters
        ----------
        mmd_weight : float, default=1.0
            Weight for Maximum Mean Discrepancy term.
        weight : float, default=1.0
            Loss weight.
        """
        super().__init__(weight=weight)
        self.mmd_weight = mmd_weight
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute MMD between latent distribution and prior."""
        batch_size = code.shape[0]
        latent_dim = code.shape[1]
        
        # Sample from prior
        prior_sample = torch.randn_like(code)
        
        # Compute MMD with RBF kernel
        mmd = self._compute_mmd(code, prior_sample)
        
        return self.mmd_weight * mmd
    
    def _compute_mmd(self, x, y, kernel='rbf', bandwidth=1.0):
        """Compute Maximum Mean Discrepancy between two distributions."""
        xx = self._kernel(x, x, kernel, bandwidth)
        yy = self._kernel(y, y, kernel, bandwidth)
        xy = self._kernel(x, y, kernel, bandwidth)
        
        return torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
    
    def _kernel(self, x, y, kernel='rbf', bandwidth=1.0):
        """Compute kernel matrix."""
        if kernel == 'rbf':
            # RBF kernel
            x_size = x.shape[0]
            y_size = y.shape[0]
            dim = x.shape[1]
            
            x = x.unsqueeze(1)  # (x_size, 1, dim)
            y = y.unsqueeze(0)  # (1, y_size, dim)
            
            distances = torch.sum((x - y) ** 2, dim=2)
            return torch.exp(-distances / (2 * bandwidth ** 2))
        else:
            raise ValueError(f"Unknown kernel: {kernel}")