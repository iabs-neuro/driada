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
        
    Raises
    ------
    ValueError
        If dropout rate is not in the range [0, 1).
        
    Examples
    --------
    >>> import torch
    >>> encoder = Encoder(orig_dim=100, inter_dim=50, code_dim=10, 
    ...                   kwargs={'dropout': 0.2})
    >>> data = torch.randn(32, 100)  # batch of 32 samples
    >>> latent = encoder(data)
    >>> print(latent.shape)
    torch.Size([32, 10])
    
    Notes
    -----
    The encoder uses LeakyReLU activation for the hidden layer. The output
    layer has no activation function, producing unbounded latent codes to
    maximize the representational capacity of the latent space.
    
    See Also
    --------
    ~driada.dim_reduction.neural.Decoder : The corresponding decoder network.
    ~driada.dim_reduction.neural.VAEEncoder : Variational encoder for probabilistic latent representations.    """

    def __init__(self, orig_dim, inter_dim, code_dim, kwargs, device=None):
        """Initialize the encoder network.
        
        Sets up the two-layer neural network architecture with optional
        dropout regularization and moves the model to the specified device.
        
        Parameters
        ----------
        orig_dim : int
            Original input dimension (number of features).
        inter_dim : int
            Intermediate hidden layer dimension.
        code_dim : int
            Output dimension of the encoded representation.
        kwargs : dict
            Additional parameters, supports 'dropout' key.
        device : torch.device, optional
            Target device for computations.
            
        Raises
        ------
        ValueError
            If dropout rate is not in the range [0, 1).        """
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
            
        # Move model to the specified device
        self.to(self._device)

    def forward(self, features):
        """Forward pass through the encoder.
        
        Applies the two-layer neural network transformation to encode
        input features into a lower-dimensional latent representation.
        
        Parameters
        ----------
        features : torch.Tensor
            Input tensor of shape (batch_size, orig_dim).
            
        Returns
        -------
        torch.Tensor
            Encoded representation of shape (batch_size, code_dim).
            Values are unbounded to maximize representational capacity.
            
        Notes
        -----
        The forward pass applies the following transformations:
        1. Linear transformation to hidden dimension
        2. Dropout regularization (if enabled)
        3. LeakyReLU activation
        4. Linear transformation to latent dimension
        
        The output is unbounded to allow full representational capacity
        in the latent space.        """
        activation = self.encoder_hidden_layer(features)
        activation = self.dropout(activation)
        activation = F.leaky_relu(activation)
        code = self.encoder_output_layer(activation)
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
        Latent space dimension. Note: this should be 2*latent_dim if you want
        latent_dim means and latent_dim log variances. The encoder outputs
        code_dim values total.
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
        Second linear transformation to latent parameters (outputs code_dim values).
    dropout : nn.Dropout
        Dropout layer for regularization.
        
    Raises
    ------
    ValueError
        If dropout rate is not in the range [0, 1).
        
    Examples
    --------
    >>> import torch
    >>> # For 10-dim latent space, need code_dim=20 (10 means + 10 log variances)
    >>> vae_encoder = VAEEncoder(orig_dim=100, inter_dim=50, code_dim=20, 
    ...                          kwargs={'dropout': 0.2})
    >>> data = torch.randn(32, 100)  # batch of 32 samples
    >>> params = vae_encoder(data)
    >>> print(params.shape)
    torch.Size([32, 20])
        
    Notes
    -----
    The output layer does not use sigmoid activation (unlike standard AE)
    because it needs to output unconstrained means and log variances for
    the Gaussian distribution.
    
    See Also
    --------
    ~driada.dim_reduction.neural.VAE : Complete variational autoencoder that uses this encoder.
    ~driada.dim_reduction.neural.Encoder : Standard encoder with bounded outputs.    """

    def __init__(self, orig_dim, inter_dim, code_dim, kwargs, device=None):
        """Initialize the variational encoder network.
        
        Sets up the two-layer neural network architecture that outputs
        parameters for a Gaussian distribution in latent space.
        
        Parameters
        ----------
        orig_dim : int
            Original input dimension (number of features).
        inter_dim : int
            Intermediate hidden layer dimension.
        code_dim : int
            Total output dimension (should be 2*latent_dim for mean and log variance).
        kwargs : dict
            Additional parameters, supports 'dropout' key.
        device : torch.device, optional
            Target device for computations.
            
        Raises
        ------
        ValueError
            If dropout rate is not in the range [0, 1).        """
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
            
        # Move model to the specified device
        self.to(self._device)

    def forward(self, features):
        """Forward pass through the VAE encoder.
        
        Applies the two-layer neural network transformation to encode
        input features into parameters for a Gaussian distribution.
        
        Parameters
        ----------
        features : torch.Tensor
            Input tensor of shape (batch_size, orig_dim).
            
        Returns
        -------
        torch.Tensor
            Encoded representation of shape (batch_size, code_dim).
            Contains concatenated parameters for the latent Gaussian:
            typically first half for means, second half for log variances.
            
        Notes
        -----
        Unlike standard encoders, VAE encoders output unconstrained values
        (no sigmoid activation) since they represent distribution parameters.
        The output should be reshaped to extract means and log variances.        """
        activation = self.encoder_hidden_layer(features)
        activation = self.dropout(activation)
        activation = F.leaky_relu(activation)
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
        
    Raises
    ------
    ValueError
        If dropout rate is not in the range [0, 1).
        
    Examples
    --------
    >>> import torch
    >>> decoder = Decoder(code_dim=10, inter_dim=50, orig_dim=100,
    ...                   kwargs={'dropout': 0.2})
    >>> latent = torch.randn(32, 10)  # batch of 32 latent codes
    >>> reconstructed = decoder(latent)
    >>> print(reconstructed.shape)
    torch.Size([32, 100])
    
    Notes
    -----
    The decoder uses LeakyReLU activation for the hidden layer and
    no activation function on the output layer, allowing it to output
    unbounded values for reconstruction.
    
    See Also
    --------
    ~driada.dim_reduction.neural.Encoder : The corresponding encoder network.
    ~driada.dim_reduction.neural.AE : Complete autoencoder using this decoder.    """

    def __init__(self, code_dim, inter_dim, orig_dim, kwargs, device=None):
        """Initialize the decoder network.
        
        Sets up the two-layer neural network architecture with optional
        dropout regularization and moves the model to the specified device.
        
        Parameters
        ----------
        code_dim : int
            Input dimension of the latent representation.
        inter_dim : int
            Intermediate hidden layer dimension.
        orig_dim : int
            Output dimension (same as original data).
        kwargs : dict
            Additional parameters, supports 'dropout' key.
        device : torch.device, optional
            Target device for computations.
            
        Raises
        ------
        ValueError
            If dropout rate is not in the range [0, 1).        """
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
            
        # Move model to the specified device
        self.to(self._device)

    def forward(self, features):
        """Forward pass through the decoder.
        
        Applies the two-layer neural network transformation to decode
        latent representations back to the original data space.
        
        Parameters
        ----------
        features : torch.Tensor
            Latent representation tensor of shape (batch_size, code_dim).
            
        Returns
        -------
        torch.Tensor
            Reconstructed data of shape (batch_size, orig_dim).
            
        Notes
        -----
        The forward pass applies the following transformations:
        1. Linear transformation to hidden dimension
        2. Dropout regularization (if enabled)
        3. LeakyReLU activation
        4. Linear transformation to original dimension
        5. No output activation (unbounded reconstruction)        """
        activation = self.decoder_hidden_layer(features)
        activation = self.dropout(activation)
        activation = F.leaky_relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = activation
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
    >>> import torch
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> ae = AE(orig_dim=100, inter_dim=50, code_dim=10, 
    ...         enc_kwargs={'dropout': 0.2}, dec_kwargs={'dropout': 0.2},
    ...         device=device)
    >>> data = torch.randn(32, 100).to(device)
    >>> reconstructed = ae(data)
    >>> print(reconstructed.shape)
    torch.Size([32, 100])
    >>> latent = ae.get_code_embedding(data)
    >>> print(latent.shape)  # Note: transposed output
    (10, 32)
    
    Notes
    -----
    The encoder produces unbounded latent codes, while the decoder
    outputs unbounded reconstructions. This design is suitable for
    general-purpose dimensionality reduction of unbounded data.
    
    See Also
    --------
    ~driada.dim_reduction.neural.VAE : Variational autoencoder for probabilistic encoding.
    ~driada.dim_reduction.neural.Encoder : The encoder component.
    ~driada.dim_reduction.neural.Decoder : The decoder component.    """

    def __init__(self, orig_dim, inter_dim, code_dim, enc_kwargs, dec_kwargs, device):
        """Initialize the autoencoder.
        
        Creates encoder and decoder networks with the specified architecture.
        
        Parameters
        ----------
        orig_dim : int
            Original input dimension.
        inter_dim : int
            Hidden layer dimension for both networks.
        code_dim : int
            Latent representation dimension.
        enc_kwargs : dict
            Encoder parameters (e.g., {'dropout': 0.2}).
        dec_kwargs : dict
            Decoder parameters (e.g., {'dropout': 0.2}).
        device : torch.device
            Device for computations.        """
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
        
        Encodes input data to latent representation and then decodes it
        back to reconstruct the original data.
        
        Parameters
        ----------
        features : torch.Tensor
            Input data of shape (batch_size, orig_dim).
            
        Returns
        -------
        torch.Tensor
            Reconstructed data of shape (batch_size, orig_dim).
            
        Notes
        -----
        The forward pass performs: input → encoder → latent → decoder → reconstruction.
        Both latent codes and reconstructions are unbounded.        """
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
            
        Notes
        -----
        This method only runs the encoder portion and returns the latent
        codes as a numpy array. The transpose operation converts from
        PyTorch's (batch, features) to DRIADA's (features, samples) format.        """
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
        
    Examples
    --------
    >>> import torch
    >>> vae = VAE(orig_dim=100, inter_dim=50, code_dim=10,
    ...           enc_kwargs={'dropout': 0.2}, dec_kwargs={'dropout': 0.2})
    >>> data = torch.randn(32, 100)
    >>> reconstructed, mean, log_var = vae(data)
    >>> # Compute VAE loss
    >>> recon_loss = torch.nn.functional.mse_loss(reconstructed, data)
    >>> kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    >>> vae_loss = recon_loss + kl_loss
    >>> # Generate new samples
    >>> z = torch.randn(32, 10)
    >>> generated = vae.decoder(z)
    
    Notes
    -----
    The VAE loss consists of two terms:
    1. Reconstruction loss (e.g., MSE or BCE)
    2. KL divergence between the learned distribution and a standard Gaussian
    
    The encoder internally outputs 2*code_dim features which are split into
    mean and log variance parameters for the latent Gaussian distribution.
    
    See Also
    --------
    ~driada.dim_reduction.neural.AE : Standard deterministic autoencoder.
    ~driada.dim_reduction.neural.VAEEncoder : The probabilistic encoder component.    """

    def __init__(
        self,
        orig_dim,
        inter_dim,
        code_dim,
        enc_kwargs=None,
        dec_kwargs=None,
        device=None,
    ):
        """Initialize the Variational Autoencoder.
        
        Creates a VAE with encoder outputting distribution parameters
        (mean and log variance) and decoder for reconstruction. The encoder
        output dimension is doubled to accommodate both parameters.
        
        Parameters
        ----------
        orig_dim : int
            Original input dimension (number of features).
        inter_dim : int
            Hidden layer dimension for both encoder and decoder.
        code_dim : int
            Dimension of the latent representation. The encoder will output
            2 * code_dim features (code_dim for mean, code_dim for log variance).
        enc_kwargs : dict, optional
            Additional encoder parameters (e.g., {'dropout': 0.2}).
            Defaults to empty dict if None.
        dec_kwargs : dict, optional
            Additional decoder parameters (e.g., {'dropout': 0.2}).
            Defaults to empty dict if None.
        device : torch.device, optional
            Device for computations. If None, encoder/decoder handle device selection.
            
        Notes
        -----
        The encoder output dimension is set to 2 * code_dim to enable the
        VAE to learn both mean and log variance parameters for the latent
        Gaussian distribution. These parameters are later split in the
        get_code method.        """
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
        backpropagation through the sampling operation by expressing the sample
        as a deterministic function of the parameters and a separate noise variable.
        
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
            
        Examples
        --------
        >>> import torch
        >>> # Create a VAE instance to access the method
        >>> vae = VAE(orig_dim=100, inter_dim=50, code_dim=20,
        ...           enc_kwargs={}, dec_kwargs={}, device=torch.device('cpu'))
        >>> mu = torch.zeros(32, 10)
        >>> log_var = torch.ones(32, 10) * -2  # Small variance
        >>> z = vae.reparameterization(mu, log_var)
        >>> print(z.shape)
        torch.Size([32, 10])
        
        Notes
        -----
        The reparameterization trick transforms sampling from N(mu, sigma^2) into:
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        
        This allows gradients to flow through mu and log_var during backpropagation
        while maintaining the stochasticity through the random epsilon.        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # sample from N(0, I)
        sample = mu + (eps * std)  # reparameterized sample
        return sample

    def get_code(self, features):
        """Extract latent code from input features using VAE encoding.
        
        Encodes input features through the VAE encoder which outputs concatenated
        mean and log variance parameters. These are reshaped and separated, then
        used to sample from the latent distribution via reparameterization.
        
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
            
        Examples
        --------
        >>> import torch
        >>> # Create a VAE with latent dimension 10 (code_dim=20 for mean+logvar)
        >>> vae = VAE(orig_dim=100, inter_dim=50, code_dim=20,
        ...           enc_kwargs={}, dec_kwargs={}, 
        ...           device=torch.device('cpu'))
        >>> features = torch.randn(32, 100)
        >>> code, mu, log_var = vae.get_code(features)
        >>> print(code.shape, mu.shape, log_var.shape)
        torch.Size([32, 20]) torch.Size([32, 20]) torch.Size([32, 20])
        
        Notes
        -----
        The encoder outputs a tensor of shape (batch_size, 2 * code_dim) which
        is reshaped to (batch_size, 2, code_dim) where:
        - [:, 0, :] contains the mean parameters
        - [:, 1, :] contains the log variance parameters        """
        x = self.encoder.forward(features)
        
        # Reshape to separate mean and log variance
        x = x.view(-1, 2, self.code_dim)
        
        # Extract distribution parameters
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as log variance
        
        # Sample latent code via reparameterization
        code = self.reparameterization(mu, log_var)
        
        return code, mu, log_var

    def forward(self, features):
        """Forward pass through the VAE.
        
        Performs a complete forward pass: encoding input to latent distribution
        parameters, sampling from the distribution, and decoding back to
        reconstruction space. Returns both reconstruction and distribution
        parameters needed for VAE loss computation.
        
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
            
        Examples
        --------
        >>> import torch
        >>> import torch.nn.functional as F
        >>> # Create a simple VAE instance for demonstration
        >>> vae = VAE(orig_dim=100, inter_dim=50, code_dim=20,
        ...           enc_kwargs={}, dec_kwargs={}, 
        ...           device=torch.device('cpu'))
        >>> data = torch.randn(32, 100)
        >>> recon, mu, log_var = vae(data)
        >>> # Compute VAE loss
        >>> recon_loss = F.mse_loss(recon, data)
        >>> kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        >>> vae_loss = recon_loss + kl_loss
        >>> print(f"Reconstruction shape: {recon.shape}")
        Reconstruction shape: torch.Size([32, 100])
        >>> print(f"Latent mean shape: {mu.shape}")
        Latent mean shape: torch.Size([32, 20])
        >>> print(f"Latent log variance shape: {log_var.shape}")
        Latent log variance shape: torch.Size([32, 20])
            
        Notes
        -----
        The mu and log_var are needed to compute the KL divergence loss:
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        
        The total VAE loss is: L = reconstruction_loss + beta * KL_loss
        where beta is a hyperparameter controlling the regularization strength.
        
        See Also
        --------
        ~driada.dim_reduction.neural.get_code : For encoding only without reconstruction.
        ~driada.dim_reduction.neural.reparameterization : The sampling mechanism.        """
        # encoding
        code, mu, log_var = self.get_code(features)

        # decoding
        reconstructed = self.decoder.forward(code)
        return reconstructed, mu, log_var

    def get_code_embedding(self, input_, use_mean=True):
        """Extract latent representation from input data.
        
        Returns either the mean of the latent distribution (deterministic) or
        a sample from it (stochastic), transposed to match DRIADA conventions.
        
        Parameters
        ----------
        input_ : torch.Tensor
            Input data of shape (batch_size, orig_dim).
        use_mean : bool, default=True
            If True, returns the mean of the latent distribution (deterministic).
            If False, returns a sample from the distribution (stochastic).
            
        Returns
        -------
        numpy.ndarray
            Latent representation of shape (code_dim, batch_size).
            
        Examples
        --------
        >>> import torch
        >>> import numpy as np
        >>> from driada.dim_reduction.neural import VAE
        >>> # Create VAE
        >>> vae = VAE(orig_dim=100, inter_dim=50, code_dim=20,
        ...           enc_kwargs={}, dec_kwargs={}, 
        ...           device=torch.device('cpu'))
        >>> data = torch.randn(32, 100)
        >>> # Get deterministic embedding (mean)
        >>> embedding = vae.get_code_embedding(data, use_mean=True)
        >>> embedding2 = vae.get_code_embedding(data, use_mean=True)
        >>> print(np.allclose(embedding, embedding2))  # True - deterministic
        True
        >>> # Get stochastic embedding (sampled)
        >>> _ = torch.manual_seed(42)  # For reproducibility in doctest
        >>> embedding3 = vae.get_code_embedding(data, use_mean=False)
        >>> _ = torch.manual_seed(43)  # Different seed
        >>> embedding4 = vae.get_code_embedding(data, use_mean=False)
        >>> print(np.allclose(embedding3, embedding4))  # False - different samples
        False
        >>> print(embedding.shape)  # Note: transposed output
        (20, 32)
            
        Notes
        -----
        - Output is transposed: (batch, features) → (features, samples)
        - use_mean=True is recommended for visualization, downstream tasks,
          and when you need consistent embeddings
        - use_mean=False captures the uncertainty in the latent representation
        
        See Also
        --------
        ~driada.dim_reduction.neural.get_code : Returns code, mean, and log variance as tensors.
        ~driada.dim_reduction.neural.AE.get_code_embedding : Always deterministic (standard autoencoder).        """
        code, mu, log_var = self.get_code(input_)
        if use_mean:
            return mu.detach().cpu().numpy().T
        else:
            return code.detach().cpu().numpy().T


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
        
    Examples
    --------
    >>> import numpy as np
    >>> from torch.utils.data import DataLoader
    >>> # Create dataset with 100 neurons, 1000 time points
    >>> data = np.random.randn(100, 1000)
    >>> dataset = NeuroDataset(data)
    >>> # Create DataLoader for batching
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch_data, batch_idx in loader:
    ...     print(batch_data.shape)  # (32, 100) - batch_size x n_features
    ...     break
    torch.Size([32, 100])
    
    Notes
    -----
    The dataset returns tuples of (sample, index) where the index can be
    used for tracking which samples were selected during training.    """

    def __init__(self, data, transform=None):
        """Initialize the neural dataset.
        
        Transposes the input data from (n_features, n_samples) to 
        (n_samples, n_features) for PyTorch compatibility.
        
        Parameters
        ----------
        data : ndarray
            Input data matrix of shape (n_features, n_samples).
        transform : callable, optional
            Optional transform function to apply to each sample.        """
        self.data = data.T
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset.
        
        Returns
        -------
        int
            Number of samples (n_samples).        """
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a sample and its index from the dataset.
        
        Parameters
        ----------
        idx : int or torch.Tensor
            Index of the sample to retrieve. If tensor, will be converted
            to Python list/int for numpy indexing.
            
        Returns
        -------
        tuple
            - sample : ndarray
                Data sample of shape (n_features,), optionally transformed.
            - idx : int
                The index of the retrieved sample.
                
        Notes
        -----
        Returns both the sample and its index to allow tracking of which
        samples were used during training. This can be useful for debugging
        or sample weighting schemes.        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, idx
