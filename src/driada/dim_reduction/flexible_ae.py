"""Flexible autoencoder architectures with modular loss composition.

This module provides flexible autoencoder and VAE implementations that support
dynamic loss composition for various objectives including reconstruction,
disentanglement, and regularization.
"""

from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
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
from ..utils.data import check_positive


class FlexibleAutoencoderBase(nn.Module, ABC):
    """Abstract base class for autoencoders with flexible loss composition.
    
    Provides common infrastructure for loss management, device handling,
    and logging. Subclasses must implement encode(), forward(), and
    compute_loss() methods with their specific signatures.
    
    Parameters
    ----------
    loss_components : list of dict, optional
        List of loss component configurations. Each dict should have:
        - "name": str, name of the loss type
        - "weight": float, weight for this loss
        - Additional parameters specific to each loss type
    device : torch.device, optional
        Device to run the model on. Defaults to CUDA if available, else CPU.
    logger : logging.Logger, optional
        Logger instance for tracking training progress.
        
    Attributes
    ----------
    loss_registry : LossRegistry
        Registry for creating loss components.
    losses : list
        List of loss components (dynamically modifiable).
    device : torch.device
        Device for computations.
    logger : logging.Logger
        Logger instance.
        
    Notes
    -----
    This base class handles:
    - Loss system initialization and management
    - Device selection and management
    - Logger setup
    - Common utilities for loss computation
    
    Subclasses define:
    - Encoder/decoder architecture
    - encode() method signature
    - forward() method signature
    - Specific loss computation logic    """
    
    def __init__(
        self,
        loss_components: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the base autoencoder infrastructure.
        
        Parameters
        ----------
        loss_components : list of dict, optional
            Loss component configurations. If None, subclasses should provide defaults.
        device : torch.device, optional
            Device for computations. If None, auto-selects CUDA if available.
        logger : logging.Logger, optional
            Logger instance. If None, creates default logger.        """
        super().__init__()
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Setup logger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize loss system
        self.loss_registry = LossRegistry()
        self.losses = []  # List of loss components
        
        # Initialize losses if provided
        if loss_components is not None:
            self._initialize_losses(loss_components)
    
    def _initialize_losses(self, loss_components: List[Dict]):
        """Initialize loss components from configuration.
        
        Parameters
        ----------
        loss_components : list of dict
            Loss configurations to initialize.
            
        Raises
        ------
        ValueError
            If loss component is malformed or loss name not registered.        """
        for i, loss_config in enumerate(loss_components):
            if not isinstance(loss_config, dict):
                raise ValueError(f"Loss component {i} must be a dict, got {type(loss_config)}")
            if "name" not in loss_config:
                raise ValueError(f"Loss component {i} missing required 'name' field")
            
            # Create loss component
            loss_params = loss_config.copy()
            name = loss_params.pop("name")
            weight = loss_params.pop("weight", 1.0)
            loss = self.loss_registry.create(name, weight=weight, **loss_params)
            self.losses.append(loss)
            
        if self.logger:
            self.logger.debug(f"Initialized {len(self.losses)} loss components")
    
    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Encode input to latent representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data.
            
        Returns
        -------
        Latent representation (type depends on subclass).        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Forward pass through the autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data.
            
        Returns
        -------
        Output (type depends on subclass).        """
        pass
    
    @abstractmethod
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
            Input data.
        indices : torch.Tensor, optional
            Batch indices for data-dependent losses.
        **extra_args
            Additional arguments for specific losses.
            
        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all loss components.
        loss_dict : dict
            Individual loss values for logging.        """
        pass
    
    def _aggregate_losses(
        self, 
        loss_inputs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Common loss aggregation logic.
        
        Parameters
        ----------
        loss_inputs : dict
            Dictionary of inputs to pass to loss components.
            Should include keys like 'code', 'recon', 'inputs', etc.
            
        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all loss components.
        loss_dict : dict
            Individual loss values for logging.
            
        Notes
        -----
        Subclasses can use this for common loss aggregation logic,
        passing appropriate inputs based on their architecture.        """
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        
        for i, loss_component in enumerate(self.losses):
            loss_value = loss_component.compute(**loss_inputs)
            
            weighted_loss = loss_component.weight * loss_value
            total_loss = total_loss + weighted_loss
            
            # Store for logging
            loss_name = loss_component.__class__.__name__
            loss_dict[f"{loss_name}_{i}"] = loss_value.item()
            loss_dict[f"{loss_name}_{i}_weighted"] = weighted_loss.item()
        
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict


class ModularAutoencoder(FlexibleAutoencoderBase):
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
        If not provided, defaults to [{"name": "reconstruction", "weight": 1.0}].
    device : torch.device, optional
        Device to run the model on. Defaults to CUDA if available, else CPU.
    logger : logging.Logger, optional
        Logger instance for tracking training progress.
        
    Attributes
    ----------
    encoder : Encoder
        Neural network encoder module.
    decoder : Decoder
        Neural network decoder module.
    loss_registry : LossRegistry
        Registry for creating loss components.
    losses : nn.ModuleList
        List of loss components (dynamically modifiable).
    input_dim : int
        Input data dimension.
    latent_dim : int
        Latent representation dimension.
    hidden_dim : int
        Hidden layer dimension.
    device : torch.device
        Device for computations.
    logger : logging.Logger
        Logger instance.
        
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
    >>> import numpy as np
    >>> data = np.random.randn(100, 1000)  # 100 features, 1000 samples
    >>> ae = ModularAutoencoder(
    ...     input_dim=100, latent_dim=10,
    ...     loss_components=[
    ...         {"name": "reconstruction", "weight": 1.0},
    ...         {"name": "sparse", "weight": 0.1, "sparsity_target": 0.05},
    ...         {"name": "orthogonality", "weight": 0.05, "external_data": data}
    ...     ]
    ... )
    
    Notes
    -----
    - Loss components are stored in an nn.ModuleList for proper PyTorch integration
    - Default reconstruction loss is automatically added if no components specified
    - The get_latent_representation method transposes output for DRIADA compatibility
    - Loss components can be dynamically added/removed during training
    
    See Also
    --------
    ~driada.dim_reduction.flexible_ae.FlexibleVAE : Variational version with probabilistic encoding.
    ~driada.dim_reduction.losses.LossRegistry : Available loss components.    """
    
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
        """Initialize the modular autoencoder.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input data. Must be positive.
        latent_dim : int
            Dimension of latent representation. Must be positive.
        hidden_dim : int, default=100
            Dimension of hidden layers. Must be positive.
        encoder_config : dict, optional
            Configuration for encoder (e.g., {"dropout": 0.2}).
        decoder_config : dict, optional
            Configuration for decoder.
        loss_components : list of dict, optional
            Loss component configurations. If None, uses default reconstruction loss.
        device : torch.device, optional
            Device for computations. If None, auto-selects CUDA if available.
        logger : logging.Logger, optional
            Logger instance. If None, creates default logger.
            
        Raises
        ------
        ValueError
            If any dimension is not positive.
            If loss component name is not registered.
            If loss component dict is malformed.        """
        # Add default reconstruction loss if no components specified
        if not loss_components:
            loss_components = [{"name": "reconstruction", "weight": 1.0}]
            
        # Initialize base class with loss system
        super().__init__(loss_components=loss_components, device=device, logger=logger)
        
        # Validate dimensions
        check_positive(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
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
        
        if self.logger:
            self.logger.info(
                f"Initialized {self.__class__.__name__} with {len(self.losses)} loss components on {self.device}"
            )
    
    
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
            
        Notes
        -----
        Applies encoder then decoder. Behavior affected by training/eval mode
        (dropout). No input validation performed.        """
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
            
        Notes
        -----
        Direct passthrough to encoder network. Applies Linear->ReLU->Dropout->Linear.
        Behavior affected by training/eval mode.        """
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
            Output constrained to [0, 1] by sigmoid activation.
            
        Notes
        -----
        Direct passthrough to decoder network. Final sigmoid activation
        constrains output to [0, 1] range.        """
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
            Additional arguments passed to all loss components.
            
        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all loss components. Scalar tensor on model device.
        loss_dict : dict
            Individual loss values with keys "{ClassName}_{index}" and
            "{ClassName}_{index}_weighted". Includes "total_loss".
            
        Notes
        -----
        Performs encode->decode forward pass. Uses base class helper for
        loss aggregation. All extra_args passed to every loss component.        """
        # Forward pass
        code = self.encode(inputs)
        recon = self.decode(code)
        
        # Prepare loss inputs
        loss_inputs = {
            'code': code,
            'recon': recon,
            'inputs': inputs,
            'indices': indices,
            'encoder': self.encoder,
            **extra_args
        }
        
        # Use base class aggregation
        return self._aggregate_losses(loss_inputs)
    
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
            
        Notes
        -----
        Runs in no_grad mode. Returns detached numpy array on CPU.        """
        with torch.no_grad():
            code = self.encode(x)
            return code.detach().cpu().numpy().T


class FlexibleVAE(FlexibleAutoencoderBase):
    """Flexible Variational Autoencoder with modular loss composition.
    
    Supports variational inference with flexible loss components for
    advanced disentanglement techniques. Uses VAEEncoder that outputs
    mean and log variance parameters.
    
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
        List of loss component configurations. Defaults to
        [{"name": "reconstruction", "weight": 1.0},
         {"name": "beta_vae", "weight": 1.0, "beta": 1.0}].
    device : torch.device, optional
        Device to run the model on.
    logger : logging.Logger, optional
        Logger instance.
        
    Notes
    -----
    - Uses VAEEncoder with output dimension 2*latent_dim for mean and log_var
    - Supports deterministic mode via use_mean parameter in get_latent_representation
    - forward() returns (recon, mu, log_var) unlike parent's single tensor
    - Default includes standard VAE losses if none specified
        
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
    ... )    """
    
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
        """Initialize FlexibleVAE.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input data. Must be positive.
        latent_dim : int  
            Dimension of latent representation. Must be positive.
        hidden_dim : int, default=100
            Dimension of hidden layers. Must be positive.
        encoder_config : dict, optional
            Configuration for VAE encoder.
        decoder_config : dict, optional
            Configuration for decoder.
        loss_components : list of dict, optional
            Loss configurations. If None, uses reconstruction + beta_vae.
        device : torch.device, optional
            Device for computations. Auto-selects CUDA if available.
        logger : logging.Logger, optional
            Logger instance.
            
        Notes
        -----
        Uses VAEEncoder with 2*latent_dim output for mean and log variance.
        Default loss configuration includes both reconstruction and KL divergence.        """
        # VAE requires at least reconstruction loss
        if not loss_components:
            loss_components = [
                {"name": "reconstruction", "weight": 1.0},
                {"name": "beta_vae", "weight": 1.0, "beta": 1.0}  # Standard VAE
            ]
        
        # Initialize base class with loss system
        super().__init__(loss_components=loss_components, device=device, logger=logger)
        
        # Validate dimensions
        check_positive(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
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
            
        Notes
        -----
        Always samples stochastically. No numerical stability checks for
        extreme log_var values. Use get_latent_representation(use_mean=True)
        for deterministic behavior.        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
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
            Output constrained to [0, 1] by sigmoid activation.
            
        Notes
        -----
        Direct passthrough to decoder network. Final sigmoid activation
        constrains output to [0, 1] range.        """
        return self.decoder(z)
    
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
            
        Notes
        -----
        Encoder must output exactly 2*latent_dim features. First half is
        interpreted as mean, second half as log variance. Always samples
        via reparameterization.        """
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
            Mean of latent distribution, shape (batch_size, latent_dim).
        log_var : torch.Tensor
            Log variance of latent distribution, shape (batch_size, latent_dim).
            
        Notes
        -----
        Returns tuple unlike parent's single tensor. Always uses sampled z
        for reconstruction, not the mean.        """
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
            
        Notes
        -----
        Performs single forward pass. Passes mu and log_var to all loss
        components. Uses base class aggregation helper.        """
        # Forward pass - single encoding
        z, mu, log_var = self.encode(inputs)
        recon = self.decode(z)
        
        # Prepare loss inputs - includes VAE-specific parameters
        loss_inputs = {
            'code': z,
            'recon': recon,
            'inputs': inputs,
            'mu': mu,
            'log_var': log_var,
            'indices': indices,
            'encoder': self.encoder,
            **extra_args
        }
        
        # Use base class aggregation
        return self._aggregate_losses(loss_inputs)
    
    def get_latent_representation(self, x: torch.Tensor, use_mean: bool = True) -> np.ndarray:
        """Get latent representation for data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, input_dim).
        use_mean : bool, default=True
            If True, return mean of latent distribution (deterministic).
            If False, return sampled latent code (stochastic).
            
        Returns
        -------
        np.ndarray
            Latent representation, shape (latent_dim, batch_size).
            Transposed for DRIADA compatibility.
            
        Notes
        -----
        Default behavior is deterministic (use_mean=True) for reproducible
        embeddings. Set use_mean=False to capture uncertainty via sampling.        """
        with torch.no_grad():
            z, mu, log_var = self.encode(x)
            if use_mean:
                return mu.detach().cpu().numpy().T
            else:
                return z.detach().cpu().numpy().T