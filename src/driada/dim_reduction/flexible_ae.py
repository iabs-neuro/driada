"""Flexible autoencoder architectures with modular loss composition.

This module provides flexible autoencoder and VAE implementations that support
dynamic loss composition for various objectives including reconstruction,
disentanglement, and regularization.
"""

from typing import List, Dict, Optional, Tuple, Any
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

from .neural import Encoder, Decoder, VAEEncoder
from .losses import AELoss, LossRegistry, ReconstructionLoss


class ModularAutoencoder(nn.Module):
    """Flexible autoencoder with modular loss composition.
    
    This autoencoder supports dynamic addition of loss components for various
    training objectives. It maintains backward compatibility while enabling
    advanced techniques like disentanglement and regularization.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input data.
    latent_dim : int
        Dimension of latent representation.
    hidden_dim : int, default=100
        Dimension of hidden layers.
    encoder_config : dict, optional
        Configuration for encoder (e.g., dropout rate).
    decoder_config : dict, optional
        Configuration for decoder.
    loss_components : list of dict, optional
        List of loss component configurations. Each dict should have:
        - "name": str, name of the loss type
        - "weight": float, weight for this loss
        - Additional parameters specific to each loss type
    device : torch.device, optional
        Device to run the model on.
    logger : logging.Logger, optional
        Logger instance for tracking training progress.
        
    Examples
    --------
    >>> # Standard autoencoder with correlation loss
    >>> ae = ModularAutoencoder(
    ...     input_dim=100, latent_dim=10,
    ...     loss_components=[
    ...         {"name": "reconstruction", "weight": 1.0},
    ...         {"name": "correlation", "weight": 0.1}
    ...     ]
    ... )
    
    >>> # Autoencoder with sparsity and orthogonality constraints
    >>> ae = ModularAutoencoder(
    ...     input_dim=100, latent_dim=10,
    ...     loss_components=[
    ...         {"name": "reconstruction", "weight": 1.0},
    ...         {"name": "sparse", "weight": 0.1, "sparsity_target": 0.05},
    ...         {"name": "orthogonality", "weight": 0.05, "external_data": data}
    ...     ]
    ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 100,
        encoder_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        loss_components: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Setup logger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Build encoder/decoder
        enc_config = encoder_config or {}
        dec_config = decoder_config or {}
        
        self.encoder = Encoder(
            orig_dim=input_dim,
            inter_dim=hidden_dim,
            code_dim=latent_dim,
            kwargs=enc_config,
            device=self.device
        )
        
        self.decoder = Decoder(
            code_dim=latent_dim,
            inter_dim=hidden_dim,
            orig_dim=input_dim,
            kwargs=dec_config,
            device=self.device
        )
        
        # Initialize loss system
        self.loss_registry = LossRegistry()
        self.losses = []
        
        # Add default reconstruction loss if no components specified
        if not loss_components:
            loss_components = [{"name": "reconstruction", "weight": 1.0}]
        
        # Initialize loss components
        for loss_config in loss_components:
            self.add_loss(**loss_config)
        
        # Move to device
        self.to(self.device)
        
        if self.logger:
            self.logger.info(
                f"Initialized {self.__class__.__name__} with {len(self.losses)} loss components on {self.device}"
            )
    
    def add_loss(self, name: str, weight: float = 1.0, **kwargs):
        """Add a loss component to the model.
        
        Parameters
        ----------
        name : str
            Name of the loss type (must be registered).
        weight : float, default=1.0
            Weight for this loss component.
        **kwargs
            Additional parameters for the loss.
        """
        loss = self.loss_registry.create(name, weight=weight, **kwargs)
        self.losses.append(loss)
        
        if self.logger:
            self.logger.debug(f"Added loss component: {name} (weight={weight})")
    
    def remove_loss(self, index: int):
        """Remove a loss component by index.
        
        Parameters
        ----------
        index : int
            Index of the loss component to remove.
        """
        if 0 <= index < len(self.losses):
            removed = self.losses.pop(index)
            if self.logger:
                self.logger.debug(f"Removed loss component at index {index}")
        else:
            raise IndexError(f"Loss index {index} out of range")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, input_dim).
            
        Returns
        -------
        torch.Tensor
            Reconstructed data, shape (batch_size, input_dim).
        """
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, input_dim).
            
        Returns
        -------
        torch.Tensor
            Latent code, shape (batch_size, latent_dim).
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent code, shape (batch_size, latent_dim).
            
        Returns
        -------
        torch.Tensor
            Reconstructed data, shape (batch_size, input_dim).
        """
        return self.decoder(z)
    
    def compute_loss(
        self, 
        inputs: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        **extra_args
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss from all components.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input data, shape (batch_size, input_dim).
        indices : torch.Tensor, optional
            Batch indices for data-dependent losses.
        **extra_args
            Additional arguments for specific losses.
            
        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all loss components.
        loss_dict : dict
            Individual loss values for logging.
        """
        # Forward pass
        code = self.encode(inputs)
        recon = self.decode(code)
        
        # Compute individual losses
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        
        for i, loss_component in enumerate(self.losses):
            loss_value = loss_component.compute(
                code=code,
                recon=recon,
                inputs=inputs,
                indices=indices,
                encoder=self.encoder,
                **extra_args
            )
            
            weighted_loss = loss_component.weight * loss_value
            total_loss = total_loss + weighted_loss
            
            # Store for logging
            loss_name = loss_component.__class__.__name__
            loss_dict[f"{loss_name}_{i}"] = loss_value.item()
            loss_dict[f"{loss_name}_{i}_weighted"] = weighted_loss.item()
        
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_latent_representation(self, x: torch.Tensor) -> np.ndarray:
        """Get latent representation for data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, input_dim).
            
        Returns
        -------
        np.ndarray
            Latent representation, shape (latent_dim, batch_size).
            Note: Transposed for DRIADA compatibility.
        """
        with torch.no_grad():
            code = self.encode(x)
            return code.detach().cpu().numpy().T


class FlexibleVAE(ModularAutoencoder):
    """Flexible Variational Autoencoder with modular loss composition.
    
    Extends ModularAutoencoder to support variational inference with
    flexible loss components for advanced disentanglement techniques.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input data.
    latent_dim : int
        Dimension of latent representation.
    hidden_dim : int, default=100
        Dimension of hidden layers.
    encoder_config : dict, optional
        Configuration for encoder.
    decoder_config : dict, optional
        Configuration for decoder.
    loss_components : list of dict, optional
        List of loss component configurations.
    device : torch.device, optional
        Device to run the model on.
    logger : logging.Logger, optional
        Logger instance.
        
    Examples
    --------
    >>> # β-VAE for disentanglement
    >>> vae = FlexibleVAE(
    ...     input_dim=100, latent_dim=10,
    ...     loss_components=[
    ...         {"name": "reconstruction", "weight": 1.0},
    ...         {"name": "beta_vae", "weight": 1.0, "beta": 4.0}
    ...     ]
    ... )
    
    >>> # TC-VAE with decomposed ELBO
    >>> vae = FlexibleVAE(
    ...     input_dim=100, latent_dim=10,
    ...     loss_components=[
    ...         {"name": "reconstruction", "weight": 1.0},
    ...         {"name": "tc_vae", "weight": 1.0, "alpha": 1.0, "beta": 5.0, "gamma": 1.0}
    ...     ]
    ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 100,
        encoder_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        loss_components: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize base class but skip encoder creation
        nn.Module.__init__(self)  # Skip ModularAutoencoder.__init__ temporarily
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Setup logger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Build VAE-specific encoder
        enc_config = encoder_config or {}
        dec_config = decoder_config or {}
        
        # VAE encoder outputs 2*latent_dim (mean and log_var)
        self.encoder = VAEEncoder(
            orig_dim=input_dim,
            inter_dim=hidden_dim,
            code_dim=2 * latent_dim,
            kwargs=enc_config,
            device=self.device
        )
        
        self.decoder = Decoder(
            code_dim=latent_dim,
            inter_dim=hidden_dim,
            orig_dim=input_dim,
            kwargs=dec_config,
            device=self.device
        )
        
        # Initialize loss system
        self.loss_registry = LossRegistry()
        self.losses = []
        
        # VAE requires at least reconstruction loss
        if not loss_components:
            loss_components = [
                {"name": "reconstruction", "weight": 1.0},
                {"name": "beta_vae", "weight": 1.0, "beta": 1.0}  # Standard VAE
            ]
        
        # Initialize loss components
        for loss_config in loss_components:
            self.add_loss(**loss_config)
        
        # Move to device
        self.to(self.device)
        
        if self.logger:
            self.logger.info(
                f"Initialized {self.__class__.__name__} with {len(self.losses)} loss components on {self.device}"
            )
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.
        
        Parameters
        ----------
        mu : torch.Tensor
            Mean of latent distribution, shape (batch_size, latent_dim).
        log_var : torch.Tensor
            Log variance of latent distribution, shape (batch_size, latent_dim).
            
        Returns
        -------
        torch.Tensor
            Sampled latent code, shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, input_dim).
            
        Returns
        -------
        z : torch.Tensor
            Sampled latent code, shape (batch_size, latent_dim).
        mu : torch.Tensor
            Mean of latent distribution, shape (batch_size, latent_dim).
        log_var : torch.Tensor
            Log variance of latent distribution, shape (batch_size, latent_dim).
        """
        # Get distribution parameters
        h = self.encoder(x)
        h = h.view(-1, 2, self.latent_dim)
        
        mu = h[:, 0, :]
        log_var = h[:, 1, :]
        
        # Sample latent code
        z = self.reparameterize(mu, log_var)
        
        return z, mu, log_var
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, input_dim).
            
        Returns
        -------
        recon : torch.Tensor
            Reconstructed data, shape (batch_size, input_dim).
        mu : torch.Tensor
            Mean of latent distribution.
        log_var : torch.Tensor
            Log variance of latent distribution.
        """
        z, mu, log_var = self.encode(x)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def compute_loss(
        self,
        inputs: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        **extra_args
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss from all components.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input data, shape (batch_size, input_dim).
        indices : torch.Tensor, optional
            Batch indices for data-dependent losses.
        **extra_args
            Additional arguments for specific losses.
            
        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all loss components.
        loss_dict : dict
            Individual loss values for logging.
        """
        # Forward pass
        recon, mu, log_var = self.forward(inputs)
        z, _, _ = self.encode(inputs)  # Get actual sampled code
        
        # Compute individual losses
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        
        for i, loss_component in enumerate(self.losses):
            loss_value = loss_component.compute(
                code=z,
                recon=recon,
                inputs=inputs,
                mu=mu,
                log_var=log_var,
                indices=indices,
                encoder=self.encoder,
                **extra_args
            )
            
            weighted_loss = loss_component.weight * loss_value
            total_loss = total_loss + weighted_loss
            
            # Store for logging
            loss_name = loss_component.__class__.__name__
            loss_dict[f"{loss_name}_{i}"] = loss_value.item()
            loss_dict[f"{loss_name}_{i}_weighted"] = weighted_loss.item()
        
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_latent_representation(self, x: torch.Tensor) -> np.ndarray:
        """Get latent representation for data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, input_dim).
            
        Returns
        -------
        np.ndarray
            Sampled latent representation, shape (latent_dim, batch_size).
            Note: Transposed for DRIADA compatibility.
        """
        with torch.no_grad():
            z, _, _ = self.encode(x)
            return z.detach().cpu().numpy().T