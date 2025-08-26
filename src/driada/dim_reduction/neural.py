import sys

# Fix torch reimport issue during coverage testing
if "torch" in sys.modules:
    torch = sys.modules["torch"]
    nn = sys.modules["torch.nn"]
    F = sys.modules["torch.nn.functional"]
    optim = sys.modules["torch.optim"]
    Dataset = sys.modules["torch.utils.data"].Dataset
    DataLoader = sys.modules["torch.utils.data"].DataLoader
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset


class Encoder(nn.Module):
    """Neural network encoder for dimensionality reduction.
    
    Implements a two-layer neural network that encodes high-dimensional data
    into a lower-dimensional latent representation. Used as the encoder
    component in autoencoders.
    
    Parameters
    ----------
    orig_dim : int
        Original input dimension (number of features).
    inter_dim : int
        Intermediate hidden layer dimension.
    code_dim : int
        Output dimension of the encoded representation (latent space).
    kwargs : dict
        Additional parameters:
        - dropout : float, optional
            Dropout rate (0 to 1). Default is no dropout.
    device : torch.device, optional
        Device to run the model on. Defaults to CUDA if available, else CPU.
        
    Attributes
    ----------
    encoder_hidden_layer : nn.Linear
        First linear transformation layer.
    encoder_output_layer : nn.Linear
        Second linear transformation to latent space.
    dropout : nn.Dropout
        Dropout layer for regularization.
    """

    def __init__(self, orig_dim, inter_dim, code_dim, kwargs, device=None):
        super().__init__()
        dropout = kwargs.get("dropout", None)

        self.encoder_hidden_layer = nn.Linear(
            in_features=orig_dim, out_features=inter_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=inter_dim, out_features=code_dim
        )

        if dropout is not None:
            if 0 <= dropout < 1:
                self.dropout = nn.Dropout(p=dropout)
            else:
                raise ValueError("Dropout rate should be in the range 0<=dropout<1")
        else:
            self.dropout = nn.Dropout(0.0)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

    def forward(self, features):
        """Forward pass through the encoder.
        
        Parameters
        ----------
        features : torch.Tensor
            Input tensor of shape (batch_size, orig_dim).
            
        Returns
        -------
        torch.Tensor
            Encoded representation of shape (batch_size, code_dim).
            Values are bounded between 0 and 1 due to sigmoid activation.
        """
        activation = self.encoder_hidden_layer(features)
        activation = (
            self.dropout(torch.ones(activation.shape).to(self._device)) * activation
        )
        activation = F.leaky_relu(activation)
        # activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        # code = F.leaky_relu(code)

        return code


class VAEEncoder(nn.Module):
    """Variational encoder that outputs parameters for latent Gaussian distribution.
    
    Unlike standard autoencoders, VAE encoders output parameters (mean and log variance)
    for a Gaussian distribution in the latent space, enabling probabilistic sampling
    and regularization via KL divergence.
    
    Parameters
    ----------
    orig_dim : int
        Original input dimension (number of features).
    inter_dim : int
        Intermediate hidden layer dimension.
    code_dim : int
        Latent space dimension. The encoder outputs 2*code_dim values:
        first code_dim for means, next code_dim for log variances.
    kwargs : dict
        Additional parameters:
        - dropout : float, optional
            Dropout rate (0 to 1). Default is no dropout.
    device : torch.device, optional
        Device to run the model on. Defaults to CUDA if available, else CPU.
        
    Attributes
    ----------
    encoder_hidden_layer : nn.Linear
        First linear transformation layer.
    encoder_output_layer : nn.Linear
        Second linear transformation to latent parameters.
    dropout : nn.Dropout
        Dropout layer for regularization.
        
    Notes
    -----
    The output layer does not use sigmoid activation (unlike standard AE)
    because it needs to output unconstrained means and log variances for
    the Gaussian distribution.
    """

    def __init__(self, orig_dim, inter_dim, code_dim, kwargs, device=None):
        super().__init__()
        dropout = kwargs.get("dropout", None)

        self.encoder_hidden_layer = nn.Linear(
            in_features=orig_dim, out_features=inter_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=inter_dim, out_features=code_dim
        )

        if dropout is not None:
            if 0 <= dropout < 1:
                self.dropout = nn.Dropout(p=dropout)
            else:
                raise ValueError("Dropout rate should be in the range 0<=dropout<1")
        else:
            self.dropout = nn.Dropout(0.0)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

    def forward(self, features):
        """Forward pass through the VAE encoder.
        
        Parameters
        ----------
        features : torch.Tensor
            Input tensor of shape (batch_size, orig_dim).
            
        Returns
        -------
        torch.Tensor
            Encoded representation of shape (batch_size, 2*code_dim).
            First half contains means, second half contains log variances
            for the latent Gaussian distribution.
        """
        activation = self.encoder_hidden_layer(features)
        activation = (
            self.dropout(torch.ones(activation.shape).to(self._device)) * activation
        )
        activation = F.leaky_relu(activation)
        # No sigmoid activation for VAE! The output represents mean and log variance
        code = self.encoder_output_layer(activation)
        return code


class Decoder(nn.Module):
    """Neural network decoder for dimensionality reduction.
    
    Implements a two-layer neural network that decodes low-dimensional latent
    representations back to the original high-dimensional space. Used as the
    decoder component in autoencoders.
    
    Parameters
    ----------
    code_dim : int
        Input dimension of the latent representation.
    inter_dim : int
        Intermediate hidden layer dimension.
    orig_dim : int
        Output dimension (same as original data dimension).
    kwargs : dict
        Additional parameters:
        - dropout : float, optional
            Dropout rate (0 to 1). Default is no dropout.
    device : torch.device, optional
        Device to run the model on. Defaults to CUDA if available, else CPU.
        
    Attributes
    ----------
    decoder_hidden_layer : nn.Linear
        First linear transformation from latent space.
    decoder_output_layer : nn.Linear
        Second linear transformation to original space.
    dropout : nn.Dropout
        Dropout layer for regularization.
    """

    def __init__(self, code_dim, inter_dim, orig_dim, kwargs, device=None):
        super().__init__()
        dropout = kwargs.get("dropout", None)

        self.decoder_hidden_layer = nn.Linear(
            in_features=code_dim, out_features=inter_dim
        )
        self.decoder_output_layer = nn.Linear(
            in_features=inter_dim, out_features=orig_dim
        )

        if dropout is not None:
            if 0 <= dropout < 1:
                self.dropout = nn.Dropout(p=dropout)
            else:
                raise ValueError("Dropout rate should be in the range 0<=dropout<1")
        else:
            self.dropout = nn.Dropout(0.0)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

    def forward(self, features):
        """Forward pass through the decoder.
        
        Parameters
        ----------
        features : torch.Tensor
            Latent representation tensor of shape (batch_size, code_dim).
            
        Returns
        -------
        torch.Tensor
            Reconstructed data of shape (batch_size, orig_dim).
        """
        activation = self.decoder_hidden_layer(features)
        activation = (
            self.dropout(torch.ones(activation.shape).to(self._device)) * activation
        )
        # activation = torch.relu(activation)
        activation = F.leaky_relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = activation
        # reconstructed = torch.sigmoid(activation)
        return reconstructed


class AE(nn.Module):
    """Standard Autoencoder for non-linear dimensionality reduction.
    
    Combines an encoder and decoder to learn a compressed representation
    of high-dimensional data through reconstruction. The model is trained
    to minimize reconstruction error.
    
    Parameters
    ----------
    orig_dim : int
        Original input dimension (number of features).
    inter_dim : int
        Intermediate hidden layer dimension for both encoder and decoder.
    code_dim : int
        Dimension of the latent representation (bottleneck).
    enc_kwargs : dict
        Additional parameters for the encoder (e.g., dropout rate).
    dec_kwargs : dict
        Additional parameters for the decoder (e.g., dropout rate).
    device : torch.device
        Device to run the model on (CPU or CUDA).
        
    Attributes
    ----------
    encoder : Encoder
        The encoder network.
    decoder : Decoder
        The decoder network.
    orig_dim : int
        Original data dimension.
    inter_dim : int
        Hidden layer dimension.
    code_dim : int
        Latent space dimension.
        
    Examples
    --------
    >>> ae = AE(orig_dim=100, inter_dim=50, code_dim=10, 
    ...         enc_kwargs={'dropout': 0.2}, dec_kwargs={'dropout': 0.2},
    ...         device=torch.device('cuda'))
    >>> reconstructed = ae(data)
    >>> latent = ae.get_code_embedding(data)
    """

    def __init__(self, orig_dim, inter_dim, code_dim, enc_kwargs, dec_kwargs, device):
        super(AE, self).__init__()

        self.encoder = Encoder(
            orig_dim=orig_dim,
            inter_dim=inter_dim,
            code_dim=code_dim,
            kwargs=enc_kwargs,
            device=device,
        )
        self.decoder = Decoder(
            orig_dim=orig_dim,
            inter_dim=inter_dim,
            code_dim=code_dim,
            kwargs=dec_kwargs,
            device=device,
        )
        self.orig_dim = orig_dim
        self.inter_dim = inter_dim
        self.code_dim = code_dim
        self._device = device

    def forward(self, features):
        """Forward pass through the autoencoder.
        
        Parameters
        ----------
        features : torch.Tensor
            Input data of shape (batch_size, orig_dim).
            
        Returns
        -------
        torch.Tensor
            Reconstructed data of shape (batch_size, orig_dim).
        """
        code = self.encoder.forward(features)
        reconstructed = self.decoder.forward(code)
        return reconstructed

    def get_code_embedding(self, input_):
        """Extract latent representation from input data.
        
        Parameters
        ----------
        input_ : torch.Tensor
            Input data of shape (batch_size, orig_dim).
            
        Returns
        -------
        numpy.ndarray
            Latent representation of shape (code_dim, batch_size).
            Note: Output is transposed for compatibility with DRIADA conventions.
        """
        encoder = self.encoder
        embedding = encoder.forward(input_)
        return embedding.detach().cpu().numpy().T


class VAE(nn.Module):
    """Variational Autoencoder for probabilistic dimensionality reduction.
    
    Implements a VAE that learns a probabilistic mapping to a latent space.
    Unlike standard autoencoders, VAEs learn a distribution over the latent
    space, enabling generation of new samples and providing uncertainty estimates.
    
    The encoder outputs parameters of a Gaussian distribution (mean and log variance),
    and the latent code is sampled from this distribution using the reparameterization trick.
    
    Parameters
    ----------
    orig_dim : int
        Original input dimension (number of features).
    inter_dim : int
        Intermediate hidden layer dimension for both encoder and decoder.
    code_dim : int
        Dimension of the latent representation (bottleneck).
    enc_kwargs : dict, optional
        Additional parameters for the encoder (e.g., dropout rate).
    dec_kwargs : dict, optional
        Additional parameters for the decoder (e.g., dropout rate).
    device : torch.device, optional
        Device to run the model on. Defaults to CUDA if available, else CPU.
        
    Attributes
    ----------
    encoder : VAEEncoder
        The encoder network that outputs mean and log variance.
    decoder : Decoder
        The decoder network.
    orig_dim : int
        Original data dimension.
    inter_dim : int
        Hidden layer dimension.
    code_dim : int
        Latent space dimension.
        
    Notes
    -----
    The VAE loss consists of two terms:
    1. Reconstruction loss (e.g., MSE or BCE)
    2. KL divergence between the learned distribution and a standard Gaussian
    
    Examples
    --------
    >>> vae = VAE(orig_dim=100, inter_dim=50, code_dim=10,
    ...           enc_kwargs={'dropout': 0.2}, dec_kwargs={'dropout': 0.2})
    >>> reconstructed, mean, log_var = vae(data)
    >>> # Generate new samples
    >>> z = torch.randn(batch_size, code_dim)
    >>> generated = vae.decoder(z)
    """

    def __init__(
        self,
        orig_dim,
        inter_dim,
        code_dim,
        enc_kwargs=None,
        dec_kwargs=None,
        device=None,
    ):
        super(VAE, self).__init__()

        # Use VAEEncoder instead of regular Encoder
        self.encoder = VAEEncoder(
            orig_dim=orig_dim,
            inter_dim=inter_dim,
            code_dim=2 * code_dim,
            kwargs=enc_kwargs or {},
            device=device,
        )
        self.decoder = Decoder(
            orig_dim=orig_dim,
            inter_dim=inter_dim,
            code_dim=code_dim,
            kwargs=dec_kwargs or {},
            device=device,
        )
        self.orig_dim = orig_dim
        self.inter_dim = inter_dim
        self.code_dim = code_dim

    def reparameterization(self, mu, log_var):
        """Reparameterization trick for VAE.
        
        Samples from the latent distribution N(mu, sigma^2) in a way that allows
        backpropagation through the sampling operation.
        
        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian distribution, shape (batch_size, code_dim).
        log_var : torch.Tensor
            Log variance of the latent Gaussian distribution, shape (batch_size, code_dim).
            
        Returns
        -------
        torch.Tensor
            Sampled latent vector, shape (batch_size, code_dim).
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def get_code(self, features):
        """Extract latent code from input features.
        
        Parameters
        ----------
        features : torch.Tensor
            Input data of shape (batch_size, orig_dim).
            
        Returns
        -------
        tuple of torch.Tensor
            - code : Sampled latent representation, shape (batch_size, code_dim)
            - mu : Mean of latent distribution, shape (batch_size, code_dim)
            - log_var : Log variance of latent distribution, shape (batch_size, code_dim)
        """
        x = self.encoder.forward(features)

        # print('x shape:', x.shape)
        x = x.view(-1, 2, self.code_dim)

        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance

        # print('mu shape:', mu.shape)
        # get the latent vector through reparameterization
        code = self.reparameterization(mu, log_var)
        # print('code shape:', mu.shape)

        return code, mu, log_var

    def forward(self, features):
        """Forward pass through the VAE.
        
        Parameters
        ----------
        features : torch.Tensor
            Input data of shape (batch_size, orig_dim).
            
        Returns
        -------
        tuple of torch.Tensor
            - reconstructed : Reconstructed data, shape (batch_size, orig_dim)
            - mu : Mean of latent distribution, shape (batch_size, code_dim)
            - log_var : Log variance of latent distribution, shape (batch_size, code_dim)
            
        Notes
        -----
        The mu and log_var are needed to compute the KL divergence loss:
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        """
        # encoding
        code, mu, log_var = self.get_code(features)

        # decoding
        reconstructed = self.decoder.forward(code)
        return reconstructed, mu, log_var

    def get_code_embedding(self, input_):
        """Extract latent representation from input data.
        
        Parameters
        ----------
        input_ : torch.Tensor
            Input data of shape (batch_size, orig_dim).
            
        Returns
        -------
        numpy.ndarray
            Sampled latent representation of shape (code_dim, batch_size).
            Note: Output is transposed for compatibility with DRIADA conventions.
            This returns the sampled code, not the mean.
        """
        # encoder = self.encoder
        embedding, mu, log_var = self.get_code(input_)
        return embedding.detach().cpu().numpy().T


class NeuroDataset(Dataset):
    """PyTorch Dataset wrapper for neural activity data.
    
    Wraps neural data matrices for use with PyTorch DataLoader, enabling
    efficient batching and sampling during neural network training.
    
    Parameters
    ----------
    data : ndarray
        Input data matrix of shape (n_features, n_samples). Will be transposed
        internally to (n_samples, n_features) for PyTorch compatibility.
    transform : callable, optional
        Optional transform to be applied on each sample.
        
    Attributes
    ----------
    data : ndarray
        Transposed data matrix of shape (n_samples, n_features).
    transform : callable or None
        Transform function to apply to samples.
    """

    def __init__(self, data, transform=None):
        self.data = data.T
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        # Return sample and index (standard pattern)
        return sample, idx
