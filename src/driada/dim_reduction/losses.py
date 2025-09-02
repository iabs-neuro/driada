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
    multiple objectives.    """
    
    def __init__(self, weight: float = 1.0, **kwargs):
        """Initialize loss component.
        
        Parameters
        ----------
        weight : float, default=1.0
            Weight for this loss component when combining multiple losses.
        **kwargs
            Additional parameters specific to each loss type.        """
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
            Scalar loss value.        """
        pass
    
    @property
    def weight(self) -> float:
        """
        Get the loss weight for balancing multiple objectives.
        
        The weight determines the relative importance of this loss component
        when combined with other losses in a multi-objective optimization.
        Higher weights increase the influence of this loss on the total loss.
        
        Returns
        -------
        float
            Current weight value for this loss component.        """
        return self._weight
    
    @weight.setter
    def weight(self, value: float):
        """
        Set the loss weight for multi-objective optimization.
        
        Parameters
        ----------
        value : float
            New weight value. Should be non-negative. A weight of 0 effectively
            disables this loss component.
            
        Notes
        -----
        Changing the weight during training can be used for curriculum learning
        or adaptive loss balancing strategies.
        
        Raises
        ------
        ValueError
            If value is negative.        """
        if value < 0:
            raise ValueError(f"Weight must be non-negative, got {value}")
        self._weight = value


class LossRegistry:
    """Registry for dynamically managing loss components.
    
    Provides a centralized system for registering and creating loss functions
    for autoencoders. Supports dynamic registration of custom losses and 
    maintains a catalog of available loss types.
    
    The registry pattern allows for easy extension with new loss types without
    modifying existing code. All registered losses must inherit from AELoss.
    
    Attributes
    ----------
    losses : Dict[str, Type[AELoss]]
        Mapping from loss names to their class types.
        
    Examples
    --------
    >>> registry = LossRegistry()
    >>> # Create a standard reconstruction loss
    >>> recon_loss = registry.create('reconstruction', loss_type='mse')
    >>> 
    >>> # Register a custom loss
    >>> class MyCustomLoss(AELoss):
    ...     def compute(self, code, recon, inputs, **kwargs):
    ...         return torch.tensor(0.0)
    >>> registry.register('custom', MyCustomLoss)
    >>> custom_loss = registry.create('custom', weight=2.0)    """
    
    def __init__(self):
        """
        Initialize the registry with default loss types.
        
        Creates an empty loss registry and populates it with standard loss
        functions commonly used in autoencoders:
        - 'reconstruction': Standard reconstruction loss (MSE/BCE)
        - 'correlation': Decorrelation loss for latent features
        - 'kl': KL divergence for variational autoencoders
        - 'activity': L1/L2 activity regularization
        - 'jacobian': Jacobian regularization for contractive autoencoders
        
        The registry can be extended with custom losses after initialization.        """
        self.losses: Dict[str, Type[AELoss]] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default loss implementations.
        
        Registers the standard set of loss functions available out-of-the-box:
        - reconstruction: Standard MSE/BCE reconstruction loss
        - correlation: Decorrelation loss for latent features  
        - orthogonality: Minimize correlation with external data
        - beta_vae: β-VAE loss for disentanglement
        - tc_vae: Total Correlation VAE loss
        - factor_vae: Factor-VAE with discriminator
        - sparse: Sparsity-inducing loss
        - contractive: Contractive autoencoder loss
        - mmd: Maximum Mean Discrepancy loss        """
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
            "mmd": MMDLoss,
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
            
        Raises
        ------
        ValueError
            If loss_class does not inherit from AELoss.        """
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
            
        Raises
        ------
        ValueError
            If the loss name is not registered.        """
        if name not in self.losses:
            raise ValueError(
                f"Unknown loss '{name}'. Available: {list(self.losses.keys())}"
            )
        
        return self.losses[name](**kwargs)


# Standard Losses

class ReconstructionLoss(AELoss):
    """
    Standard reconstruction loss for autoencoders (MSE or BCE).
    
    The reconstruction loss measures how well the autoencoder can reconstruct
    the input data from its latent representation. This is the primary loss
    component for training autoencoders.
    
    Supports two loss types:
    - MSE (Mean Squared Error): For continuous data reconstruction
    - BCE (Binary Cross-Entropy): For binary data or data in [0,1] range
    
    The loss encourages the decoder to accurately reconstruct inputs from
    the encoded representations, ensuring information preservation.
    
    Parameters
    ----------
    loss_type : {'mse', 'bce'}, default='mse'
        Type of reconstruction loss to use.
    weight : float, default=1.0
        Weight for this loss component in multi-objective optimization.
        
    Examples
    --------
    >>> # For continuous data
    >>> loss = ReconstructionLoss(loss_type='mse', weight=1.0)
    >>> 
    >>> # For binary or probabilistic data
    >>> loss = ReconstructionLoss(loss_type='bce', weight=2.0)    """
    
    def __init__(self, loss_type: str = "mse", weight: float = 1.0):
        """Initialize reconstruction loss.
        
        Parameters
        ----------
        loss_type : str, default="mse"
            Type of reconstruction loss ("mse" or "bce").
        weight : float, default=1.0
            Loss weight.
            
        Raises
        ------
        ValueError
            If loss_type is not 'mse' or 'bce'.        """
        super().__init__(weight=weight)
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss(reduction='mean')
        elif loss_type == "bce":
            self.criterion = nn.BCELoss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute reconstruction loss between input and reconstruction.
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation (unused for reconstruction loss).
        recon : torch.Tensor
            Reconstructed data, shape (batch_size, n_features).
        inputs : torch.Tensor
            Original input data, shape (batch_size, n_features).
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            Scalar loss value. MSE for continuous data or BCE for binary data,
            depending on loss_type specified in __init__.        """
        return self.criterion(recon, inputs)


class CorrelationLoss(AELoss):
    """
    Correlation loss to encourage decorrelated latent features.
    
    This loss minimizes correlations between different dimensions of the latent
    code, encouraging the autoencoder to learn a disentangled representation
    where each latent dimension captures independent factors of variation.
    
    The loss is computed as the sum of squared off-diagonal elements of the
    correlation matrix of latent codes. A fully decorrelated representation
    would have a diagonal correlation matrix (identity matrix after normalization).
    
    This regularization is particularly useful for:
    - Learning interpretable latent representations
    - Preventing redundancy in latent dimensions
    - Improving generalization by reducing overfitting
    
    Mathematical formulation:
        L_corr = (1/P) * sum_{i≠j} |corr(z_i, z_j)|
    where z_i is the i-th dimension of the latent code across the batch,
    and P is the number of off-diagonal pairs.
    
    Parameters
    ----------
    weight : float, default=1.0
        Weight for this loss component. Higher values enforce stronger
        decorrelation but may harm reconstruction quality.
        
    Notes
    -----
    The correlation is computed across the batch dimension, so larger batch
    sizes provide more accurate correlation estimates. Requires batch_size >= 2.    """
    
    def __init__(self, weight: float = 1.0):
        """Initialize correlation loss.
        
        Parameters
        ----------
        weight : float, default=1.0
            Loss weight.        """
        super().__init__(weight=weight)
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute average pairwise correlation amplitude in latent code.
        
        Encourages decorrelated latent features by penalizing correlations
        between different dimensions of the latent representation.
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation, shape (batch_size, code_dim).
        recon : torch.Tensor
            Reconstructed data (unused).
        inputs : torch.Tensor
            Original input data (unused).
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            Average absolute correlation across all pairs of latent dimensions.
            Returns 0 if code_dim = 1 or batch_size < 2.
            
        Notes
        -----
        Requires batch_size >= 2 for correlation computation. Returns 0 for
        single-sample batches.        """
        # Handle single feature case
        if code.shape[1] == 1:
            return torch.tensor(0.0, device=code.device)
        
        # Handle single sample case (corrcoef requires at least 2 samples)
        if code.shape[0] < 2:
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
    """Orthogonality loss to minimize correlation with external data (MI proxy).
    
    FUTURE: Replace correlation-based approach with proper mutual information
    estimation (e.g., using GCMI or KSG estimators from information module).
    Current implementation uses correlation as a crude proxy for MI.    """
    
    def __init__(self, external_data: Optional[np.ndarray] = None, weight: float = 1.0):
        """Initialize orthogonality loss.
        
        Parameters
        ----------
        external_data : np.ndarray, optional
            External data to minimize correlation with, shape (n_features, n_samples).
        weight : float, default=1.0
            Loss weight.        """
        super().__init__(weight=weight)
        self.external_data = external_data
        self._external_tensor = None
    
    def compute(self, code, recon, inputs, indices=None, **kwargs):
        """Compute correlation between latent code and external data.
        
        Used as a proxy for mutual information minimization. Encourages the
        latent representation to be orthogonal (uncorrelated) with provided
        external variables.
        
        FUTURE: Implement proper MI computation using driada.information module
        estimators instead of correlation-based approximation.
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation, shape (batch_size, code_dim).
        recon : torch.Tensor
            Reconstructed data (unused).
        inputs : torch.Tensor
            Original input data (unused).
        indices : torch.Tensor, optional
            Batch indices to select corresponding external data columns.
            If None, assumes first batch_size columns of external_data.
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            Average absolute correlation between latent code and external data.
            Returns 0 if no external data provided.
            
        Notes
        -----
        External data should have shape (n_features, n_samples) where n_samples
        should be >= batch_size if indices is None.        """
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
    """β-VAE loss for disentanglement via increased KL penalty.
    
    Implements the β-VAE objective which modifies the standard VAE loss by
    scaling the KL divergence term with a factor β > 1. This encourages the
    model to learn disentangled representations where each latent dimension
    captures at most one factor of variation.
    
    The full β-VAE loss (when combined with reconstruction) is:
    L = Reconstruction + β * KL(q(z|x)||p(z))
    
    Notes
    -----
    - β = 1 recovers the standard VAE
    - β > 1 encourages disentanglement
    - Too high β can hurt reconstruction quality
    - Typical values: β ∈ [4, 10] for disentanglement tasks
    
    References
    ----------
    Higgins, I., et al. (2017). β-VAE: Learning Basic Visual Concepts with
    a Constrained Variational Framework. ICLR 2017.    """
    
    def __init__(self, beta: float = 4.0, weight: float = 1.0):
        """Initialize β-VAE loss.
        
        Parameters
        ----------
        beta : float, default=4.0
            Beta parameter controlling KL penalty strength.
            Must be positive for valid disentanglement.
        weight : float, default=1.0
            Loss weight.
            
        Raises
        ------
        ValueError
            If beta <= 0.        """
        super().__init__(weight=weight)
        if beta <= 0:
            raise ValueError(f"Beta must be positive for disentanglement, got {beta}")
        self.beta = beta
    
    def compute(self, code, recon, inputs, mu=None, log_var=None, **kwargs):
        """Compute β-weighted KL divergence loss.
        
        Calculates the KL divergence between the learned posterior q(z|x) and
        the standard normal prior p(z) = N(0,I), scaled by the β parameter.
        Higher β values encourage greater disentanglement at the cost of
        reconstruction quality.
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation (unused, included for interface consistency).
        recon : torch.Tensor
            Reconstructed outputs (unused).
        inputs : torch.Tensor
            Original inputs (unused).
        mu : torch.Tensor
            Mean of the approximate posterior, shape (batch_size, latent_dim).
        log_var : torch.Tensor
            Log variance of the approximate posterior, shape (batch_size, latent_dim).
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            β-weighted KL divergence loss, averaged over the batch.
            
        Raises
        ------
        ValueError
            If mu or log_var are not provided.
            
        Notes
        -----
        The KL divergence for a Gaussian posterior is:
        KL(q(z|x)||p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        where the sum is over latent dimensions.
        
        References
        ----------
        Higgins, I., et al. (2017). β-VAE: Learning Basic Visual Concepts with
        a Constrained Variational Framework. ICLR 2017.        """
        if mu is None or log_var is None:
            raise ValueError("β-VAE loss requires mu and log_var")
        
        # KL divergence from N(0,1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / mu.shape[0]  # Average over batch
        
        return self.beta * kl_loss


class TCVAELoss(AELoss):
    """TC-VAE loss with decomposed ELBO for better disentanglement.
    
    Total Correlation VAE (TC-VAE) decomposes the KL divergence term of the
    ELBO into three meaningful components:
    1. Mutual Information between data and latent: I(x;z)
    2. Total Correlation (TC) measuring dependence between latents: TC(z)
    3. Dimension-wise KL divergence: Σ KL(q(z_i)||p(z_i))
    
    By separately weighting these terms, TC-VAE can specifically target the
    total correlation term to encourage factorial (independent) latent codes
    while maintaining good reconstruction.
    
    Notes
    -----
    The decomposition allows fine-grained control over different aspects:
    - α controls information preserved about data
    - β controls disentanglement (via total correlation)
    - γ controls deviation from the prior
    
    References
    ----------
    Chen, T. Q., et al. (2018). Isolating Sources of Disentanglement in
    Variational Autoencoders. NeurIPS 2018.    """
    
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
            
        Notes
        -----
        The decomposition is:
        KL[q(z|x)||p(z)] = I(x;z) + TC(z) + Σ KL[q(z_i)||p(z_i)]
        where I(x;z) is mutual information, TC(z) is total correlation.        """
        super().__init__(weight=weight)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def compute(self, code, recon, inputs, mu=None, log_var=None, **kwargs):
        """Compute TC-VAE loss with ELBO decomposition.
        
        Estimates the three components of the decomposed ELBO using importance
        sampling with the aggregate posterior as the proposal distribution.
        
        Parameters
        ----------
        code : torch.Tensor
            Sampled latent representation z ~ q(z|x), shape (batch_size, latent_dim).
            Used for computing log probabilities in the ELBO decomposition.
        recon : torch.Tensor
            Reconstructed outputs (unused).
        inputs : torch.Tensor
            Original inputs (unused).
        mu : torch.Tensor
            Mean of approximate posterior, shape (batch_size, latent_dim).
        log_var : torch.Tensor
            Log variance of approximate posterior, shape (batch_size, latent_dim).
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            Weighted sum of MI, TC, and dimension-wise KL components.
            
        Raises
        ------
        ValueError
            If mu or log_var are not provided.        """
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
        # Note: MI = I(x;z) = log q(z|x) - log q(z) = KL[q(z|x)||q(z)]
        mi = log_qz_given_x.sum(dim=1) - log_qz  # Mutual Information
        tc = log_qz - log_pz.sum(dim=1)  # Total Correlation
        dwkl = kl - mi - tc  # Dimension-wise KL
        
        # Weighted combination
        loss = self.alpha * mi.mean() + self.beta * tc.mean() + self.gamma * dwkl.mean()
        
        return loss
    
    def _log_density_gaussian(self, x, mu, log_var):
        """Compute log probability under a Gaussian.
        
        Calculates log p(x|μ,σ²) for a diagonal Gaussian distribution.
        
        Parameters
        ----------
        x : torch.Tensor
            Points to evaluate, shape (..., latent_dim).
        mu : torch.Tensor
            Mean parameters, shape (..., latent_dim).
        log_var : torch.Tensor
            Log variance parameters, shape (..., latent_dim).
            
        Returns
        -------
        torch.Tensor
            Log probabilities, shape (..., latent_dim).
            
        Notes
        -----
        Uses log variance for numerical stability. The log probability is:
        log p(x|μ,σ²) = -0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]        """
        normalization = -0.5 * (np.log(2 * np.pi) + log_var)
        inv_var = torch.exp(-log_var)
        log_density = normalization - 0.5 * ((x - mu).pow(2) * inv_var)
        return log_density
    
    def _log_importance_weight_matrix(self, batch, mu, log_var):
        """Compute importance weights for minibatch.
        
        Calculates log importance weights for estimating the marginal posterior
        q(z) from the conditional posteriors q(z|x) in a minibatch.
        
        Parameters
        ----------
        batch : torch.Tensor
            Latent samples, shape (batch_size, latent_dim).
        mu : torch.Tensor
            Posterior means for all samples, shape (batch_size, latent_dim).
        log_var : torch.Tensor
            Posterior log variances, shape (batch_size, latent_dim).
            
        Returns
        -------
        torch.Tensor
            Log importance weight matrix, shape (batch_size, batch_size).
            Element [i,j] contains log q(z_i|x_j).
            
        Notes
        -----
        Used for estimating log q(z) ≈ log(1/N Σ_j q(z|x_j)) via importance
        sampling, which is needed for the TC-VAE decomposition.        """
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
    """Factor-VAE loss using a discriminator for disentanglement.
    
    Factor-VAE encourages disentanglement by using an adversarial discriminator
    to estimate and minimize the total correlation (TC) in the latent space.
    The discriminator is trained to distinguish between samples from q(z)
    (the aggregate posterior) and the product of marginals Π q(z_i).
    
    This approach avoids the sampling difficulties of TC-VAE while still
    targeting the total correlation for disentanglement.
    
    Notes
    -----
    The discriminator provides a density ratio estimate that can be used to
    approximate the total correlation. The VAE is trained to minimize this
    estimate (fool the discriminator) while the discriminator is trained to
    maximize it (correctly classify real vs permuted codes).
    
    IMPORTANT: The discriminator requires separate optimization in the training
    loop. This loss only returns the TC penalty for the VAE.
    
    FUTURE: Implement a complete FactorVAE training system that handles both
    VAE and discriminator optimization, possibly as a separate Trainer class.
    
    References
    ----------
    Kim, H., & Mnih, A. (2018). Disentangling by Factorising. ICML 2018.    """
    
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
            Device for discriminator.        """
        super().__init__(weight=weight)
        self.gamma = gamma
        self.discriminator_dims = discriminator_dims or [256, 256]
        self.device = device or torch.device("cpu")
        self.discriminator = None
        self._latent_dim = None
    
    def _build_discriminator(self, latent_dim: int):
        """Build discriminator network.
        
        Creates a multi-layer perceptron to distinguish between samples from
        the aggregate posterior q(z) and the product of marginals.
        
        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
            
        Notes
        -----
        The discriminator architecture:
        - Input: latent code (latent_dim)
        - Hidden layers: specified by discriminator_dims
        - Output: 2 classes (real vs permuted)
        - Activation: LeakyReLU with dropout for regularization
        
        The discriminator requires separate optimization in the training loop.        """
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
        
        self.discriminator = nn.Sequential(*layers)
        self.discriminator.device = None  # Will be set when needed
        self._latent_dim = latent_dim
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute Factor-VAE discriminator-based loss.
        
        Trains a discriminator to estimate the total correlation and uses it
        to penalize the encoder. The discriminator distinguishes between:
        - Real: samples from q(z) (the aggregate posterior)
        - Fake: samples where dimensions are independently permuted
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation, shape (batch_size, latent_dim).
        recon : torch.Tensor
            Reconstructed outputs (unused).
        inputs : torch.Tensor
            Original inputs (unused).
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            Total correlation penalty estimated by the discriminator.
            
        Notes
        -----
        The permutation breaks dependencies between latent dimensions,
        approximating samples from the product of marginals Π q(z_i).
        The discriminator's confidence in classifying real samples
        indicates the strength of the total correlation.        """
        batch_size = code.shape[0]
        latent_dim = code.shape[1]
        
        # Build discriminator if needed
        if self.discriminator is None:
            self._build_discriminator(latent_dim)
        
        # Move discriminator to correct device if needed
        if self.discriminator.device != code.device:
            self.discriminator = self.discriminator.to(code.device)
        
        # Create permuted code (break factorial structure)
        indices = torch.randperm(batch_size).to(code.device)
        permuted_code = code[indices]
        
        # Discriminator predictions
        real_pred = self.discriminator(code)
        fake_pred = self.discriminator(permuted_code.detach())
        
        # Discriminator loss (standard GAN loss) - for reference, not returned
        # disc_loss = F.cross_entropy(real_pred, torch.ones(batch_size, dtype=torch.long).to(code.device))
        # disc_loss += F.cross_entropy(fake_pred, torch.zeros(batch_size, dtype=torch.long).to(code.device))
        
        # Total correlation loss (fool discriminator)
        tc_loss = F.cross_entropy(real_pred, torch.zeros(batch_size, dtype=torch.long).to(code.device))
        
        return self.gamma * tc_loss


# Regularization Losses

class SparsityLoss(AELoss):
    """Sparsity loss to encourage sparse latent representations.
    
    Implements a sparsity constraint on the latent activations by penalizing
    the KL divergence between the average activation and a target sparsity level.
    This encourages the model to use only a subset of latent dimensions for
    each input, leading to more interpretable representations.
    
    The loss is based on the KL divergence between:
    - ρ̂: average activation of each latent unit (across the batch)
    - ρ: target sparsity level (e.g., 0.05 for 5% average activation)
    
    References
    ----------
    Ng, A. (2011). Sparse autoencoder. CS294A Lecture notes, Stanford University.    """
    
    def __init__(self, sparsity_target: float = 0.05, weight: float = 1.0):
        """Initialize sparsity loss.
        
        Parameters
        ----------
        sparsity_target : float, default=0.05
            Target average activation level.
        weight : float, default=1.0
            Loss weight.        """
        super().__init__(weight=weight)
        self.sparsity_target = sparsity_target
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute KL divergence between actual and target sparsity.
        
        Calculates the sparsity penalty based on the average activation of each
        latent unit compared to the target sparsity level.
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation after activation (e.g., sigmoid),
            shape (batch_size, latent_dim). Values should be in [0, 1].
        recon : torch.Tensor
            Reconstructed outputs (unused).
        inputs : torch.Tensor
            Original inputs (unused).
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            KL divergence between actual and target sparsity, summed over
            all latent dimensions.
            
        Notes
        -----
        For each latent unit j:
        KL(ρ||ρ̂_j) = ρ log(ρ/ρ̂_j) + (1-ρ) log((1-ρ)/(1-ρ̂_j))
        where ρ̂_j is the average activation of unit j over the batch.
        
        Uses clamping for numerical stability to avoid log(0).        """
        # Average activation per latent dimension
        avg_activation = torch.mean(code, dim=0)
        
        # Clamp for numerical stability
        eps = 1e-6
        avg_activation = torch.clamp(avg_activation, eps, 1 - eps)
        rho = torch.clamp(torch.tensor(self.sparsity_target), eps, 1 - eps)
        
        # KL divergence from target sparsity
        kl = rho * torch.log(rho / avg_activation)
        kl += (1 - rho) * torch.log((1 - rho) / (1 - avg_activation))
        
        return torch.sum(kl)


class ContractiveLoss(AELoss):
    """Contractive loss for robust representations via Jacobian penalty.
    
    Contractive autoencoders learn robust representations by penalizing the
    Frobenius norm of the encoder's Jacobian. This encourages the encoder
    to be locally constant, making the learned features insensitive to small
    input perturbations.
    
    The contractive penalty ||∂h/∂x||²_F encourages the encoder mapping to
    contract the input space, except along the data manifold directions.
    
    Notes
    -----
    Computing the full Jacobian can be expensive for high-dimensional inputs.
    This implementation uses automatic differentiation to compute the penalty.
    
    WARNING: Computational cost scales linearly with latent dimension.
    Consider using stochastic approximations for latent_dim > 100.
    
    References
    ----------
    Rifai, S., et al. (2011). Contractive Auto-Encoders: Explicit Invariance
    During Feature Extraction. ICML 2011.    """
    
    def __init__(self, lambda_c: float = 0.01, weight: float = 1.0):
        """Initialize contractive loss.
        
        Parameters
        ----------
        lambda_c : float, default=0.01
            Contraction strength parameter.
        weight : float, default=1.0
            Loss weight.        """
        super().__init__(weight=weight)
        self.lambda_c = lambda_c
    
    def compute(self, code, recon, inputs, encoder=None, **kwargs):
        """Compute Frobenius norm of encoder Jacobian.
        
        Calculates the contractive penalty as the squared Frobenius norm of
        the Jacobian matrix ∂h/∂x, where h is the encoder output.
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation h(x), shape (batch_size, latent_dim).
        recon : torch.Tensor
            Reconstructed outputs (unused).
        inputs : torch.Tensor
            Original inputs x, shape (batch_size, input_dim).
        encoder : callable, optional
            Encoder function/module. Required for this loss.
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            Frobenius norm of Jacobian, averaged over the batch.
            
        Raises
        ------
        ValueError
            If encoder is not provided.
            
        Notes
        -----
        WARNING: Current implementation is O(latent_dim) in computation time.
        For high-dimensional latent spaces, consider using stochastic approximations
        or alternative regularization methods.
        
        For efficiency, we compute Tr(J^T J) = Σ_i ||∂h/∂x_i||² instead of
        forming the full Jacobian matrix.        """
        if encoder is None:
            raise ValueError("Contractive loss requires encoder module")
        
        # Enable gradient computation for inputs
        inputs = inputs.requires_grad_(True)
        
        # Forward through encoder to get fresh activations with gradients
        # Note: We ignore the passed 'code' and recompute to ensure gradient flow
        h = encoder(inputs)
        
        # Compute Jacobian via backpropagation
        batch_size = inputs.shape[0]
        jacobian_norm = 0
        
        # WARNING: This loop is expensive for large latent_dim
        for i in range(h.shape[1]):
            # Gradient of i-th latent w.r.t. inputs
            grad = torch.autograd.grad(
                h[:, i].sum(), inputs, 
                retain_graph=True, create_graph=True
            )[0]
            
            # Frobenius norm squared
            jacobian_norm += torch.sum(grad ** 2) / batch_size
        
        return self.lambda_c * jacobian_norm


class MMDLoss(AELoss):
    """Maximum Mean Discrepancy (MMD) loss for latent distribution matching.
    
    MMD measures the distance between two distributions by comparing their
    embeddings in a reproducing kernel Hilbert space (RKHS). It's used as an
    alternative to KL divergence for matching distributions without requiring
    explicit density calculations.
    
    MMD is used to match the latent distribution q(z) to a prior p(z) by
    minimizing the distance between their expected feature maps:
    MMD²(p,q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    where k is a kernel function (typically RBF/Gaussian).
    
    Notes
    -----
    - MMD = 0 if and only if the distributions are identical
    - Choice of kernel bandwidth affects sensitivity to different scales
    - MMD with Gaussian kernel can detect all distributional differences
    - More computationally efficient than adversarial approaches
    - Particularly useful for implicit generative models
    
    For neural data:
    - Useful when you want latent codes to follow a specific distribution
    - Can encourage more interpretable latent spaces for neural dynamics
    - Works well with non-Gaussian priors (e.g., mixture models)
    
    WARNING: Uses fixed bandwidth (1.0). For better results, consider adaptive
    bandwidth selection based on median heuristic or cross-validation.
    
    References
    ----------
    Gretton, A., et al. (2012). A kernel two-sample test.
    Journal of Machine Learning Research, 13(1), 723-773.
    
    Tolstikhin, I., et al. (2018). Wasserstein Auto-Encoders. ICLR 2018.
    (Note: WAE paper uses MMD as an alternative to Wasserstein distance)    """
    
    def __init__(self, mmd_weight: float = 1.0, weight: float = 1.0):
        """Initialize MMD loss.
        
        Parameters
        ----------
        mmd_weight : float, default=1.0
            Weight for Maximum Mean Discrepancy term.
        weight : float, default=1.0
            Loss weight.        """
        super().__init__(weight=weight)
        self.mmd_weight = mmd_weight
    
    def compute(self, code, recon, inputs, **kwargs):
        """Compute MMD between latent distribution and prior.
        
        Estimates the Maximum Mean Discrepancy between the encoded latent
        distribution and a standard normal prior using an RBF kernel.
        
        Parameters
        ----------
        code : torch.Tensor
            Latent representation, shape (batch_size, latent_dim).
        recon : torch.Tensor
            Reconstructed outputs (unused).
        inputs : torch.Tensor
            Original inputs (unused).
        **kwargs
            Additional arguments (unused).
            
        Returns
        -------
        torch.Tensor
            MMD loss value, scaled by mmd_weight.
            
        Notes
        -----
        Uses empirical estimates:
        - q(z): empirical distribution from encoder outputs
        - p(z): samples from N(0,I) prior
        The kernel bandwidth is fixed at 1.0 by default.        """
        batch_size = code.shape[0]
        latent_dim = code.shape[1]
        
        # Sample from prior
        prior_sample = torch.randn_like(code)
        
        # Compute MMD with RBF kernel
        mmd = self._compute_mmd(code, prior_sample)
        
        return self.mmd_weight * mmd
    
    def _compute_mmd(self, x, y, kernel='rbf', bandwidth=1.0):
        """Compute Maximum Mean Discrepancy between two distributions.
        
        Unbiased estimator of MMD² using U-statistics.
        
        Parameters
        ----------
        x : torch.Tensor
            Samples from first distribution, shape (n_samples, dim).
        y : torch.Tensor
            Samples from second distribution, shape (n_samples, dim).
        kernel : str, default='rbf'
            Kernel type (only 'rbf' supported).
        bandwidth : float, default=1.0
            RBF kernel bandwidth parameter.
            
        Returns
        -------
        torch.Tensor
            MMD² estimate.
            
        Notes
        -----
        Uses the unbiased U-statistic estimator that excludes diagonal terms:
        MMD² = (1/(n(n-1)))Σᵢ≠ⱼ k(xᵢ,xⱼ) + (1/(m(m-1)))Σᵢ≠ⱼ k(yᵢ,yⱼ) - (2/(nm))Σᵢⱼ k(xᵢ,yⱼ)
        
        This gives an unbiased estimate of the population MMD.        """
        n = x.shape[0]
        m = y.shape[0]
        
        # Compute kernel matrices
        xx = self._kernel(x, x, kernel, bandwidth)
        yy = self._kernel(y, y, kernel, bandwidth)
        xy = self._kernel(x, y, kernel, bandwidth)
        
        # For xx and yy, exclude diagonal and normalize properly
        # Clone to avoid in-place modification issues
        xx_no_diag = xx.clone()
        yy_no_diag = yy.clone()
        xx_no_diag.fill_diagonal_(0)
        yy_no_diag.fill_diagonal_(0)
        
        # Unbiased estimator
        if n > 1:
            xx_term = torch.sum(xx_no_diag) / (n * (n - 1))
        else:
            xx_term = torch.tensor(0.0, device=x.device)
            
        if m > 1:
            yy_term = torch.sum(yy_no_diag) / (m * (m - 1))
        else:
            yy_term = torch.tensor(0.0, device=y.device)
            
        xy_term = torch.sum(xy) / (n * m)
        
        return xx_term + yy_term - 2 * xy_term
    
    def _kernel(self, x, y, kernel='rbf', bandwidth=1.0):
        """Compute kernel matrix.
        
        Calculates the Gram matrix K[i,j] = k(x[i], y[j]) for the specified kernel.
        
        Parameters
        ----------
        x : torch.Tensor
            First set of points, shape (n, dim).
        y : torch.Tensor
            Second set of points, shape (m, dim).
        kernel : str, default='rbf'
            Kernel type. Currently only 'rbf' (Gaussian) is supported.
        bandwidth : float, default=1.0
            RBF kernel bandwidth σ. Larger values create smoother kernels.
            
        Returns
        -------
        torch.Tensor
            Kernel matrix, shape (n, m).
            
        Raises
        ------
        ValueError
            If kernel type is not 'rbf'.
            
        Notes
        -----
        RBF kernel: k(x,y) = exp(-||x-y||² / (2σ²))
        Bandwidth selection is important: too small may overfit, too large may
        underfit. Common heuristics use median pairwise distance.        """
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