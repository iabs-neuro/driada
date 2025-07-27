import sys

# Fix torch reimport issue during coverage testing
if 'torch' in sys.modules:
    torch = sys.modules['torch']
    nn = sys.modules['torch.nn']
    F = sys.modules['torch.nn.functional']
    optim = sys.modules['torch.optim']
    Dataset = sys.modules['torch.utils.data'].Dataset
    DataLoader = sys.modules['torch.utils.data'].DataLoader
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader


class Encoder(nn.Module):

    def __init__(self, orig_dim, inter_dim, code_dim, kwargs, device=None):
        super().__init__()
        dropout = kwargs.get('dropout', None)

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
                raise ValueError('Dropout rate should be in the range 0<=dropout<1')
        else:
            self.dropout = nn.Dropout(0.0)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = self.dropout(torch.ones(activation.shape).to(self._device)) * activation
        activation = F.leaky_relu(activation)
        # activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        # code = F.leaky_relu(code)

        return code


class VAEEncoder(nn.Module):
    """Special encoder for VAE that doesn't use sigmoid activation"""
    
    def __init__(self, orig_dim, inter_dim, code_dim, kwargs, device=None):
        super().__init__()
        dropout = kwargs.get('dropout', None)

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
                raise ValueError('Dropout rate should be in the range 0<=dropout<1')
        else:
            self.dropout = nn.Dropout(0.0)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = self.dropout(torch.ones(activation.shape).to(self._device)) * activation
        activation = F.leaky_relu(activation)
        # No sigmoid activation for VAE! The output represents mean and log variance
        code = self.encoder_output_layer(activation)
        return code


class Decoder(nn.Module):

    def __init__(self, code_dim, inter_dim, orig_dim, kwargs, device=None):
        super().__init__()
        dropout = kwargs.get('dropout', None)

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
                raise ValueError('Dropout rate should be in the range 0<=dropout<1')
        else:
            self.dropout = nn.Dropout(0.0)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

    def forward(self, features):
        activation = self.decoder_hidden_layer(features)
        activation = self.dropout(torch.ones(activation.shape).to(self._device)) * activation
        # activation = torch.relu(activation)
        activation = F.leaky_relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = activation
        # reconstructed = torch.sigmoid(activation)
        return reconstructed


class AE(nn.Module):

    def __init__(self, orig_dim, inter_dim, code_dim, enc_kwargs, dec_kwargs, device):
        super(AE, self).__init__()

        self.encoder = Encoder(orig_dim=orig_dim, inter_dim=inter_dim, code_dim=code_dim, kwargs=enc_kwargs, device=device)
        self.decoder = Decoder(orig_dim=orig_dim, inter_dim=inter_dim, code_dim=code_dim, kwargs=dec_kwargs, device=device)
        self.orig_dim = orig_dim
        self.inter_dim = inter_dim
        self.code_dim = code_dim
        self._device = device

    def forward(self, features):
        code = self.encoder.forward(features)
        reconstructed = self.decoder.forward(code)
        return reconstructed

    def get_code_embedding(self, input_):
        encoder = self.encoder
        embedding = encoder.forward(input_)
        return embedding.detach().cpu().numpy().T


class VAE(nn.Module):

    def __init__(self, orig_dim, inter_dim, code_dim, enc_kwargs=None, dec_kwargs=None, device=None):
        super(VAE, self).__init__()

        # Use VAEEncoder instead of regular Encoder
        self.encoder = VAEEncoder(orig_dim=orig_dim, inter_dim=inter_dim, code_dim=2 * code_dim, kwargs=enc_kwargs or {}, device=device)
        self.decoder = Decoder(orig_dim=orig_dim, inter_dim=inter_dim, code_dim=code_dim, kwargs=dec_kwargs or {}, device=device)
        self.orig_dim = orig_dim
        self.inter_dim = inter_dim
        self.code_dim = code_dim

    def reparameterization(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def get_code(self, features):
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
        # encoding
        code, mu, log_var = self.get_code(features)

        # decoding
        reconstructed = self.decoder.forward(code)
        return reconstructed, mu, log_var

    def get_code_embedding(self, input_):
        #encoder = self.encoder
        embedding, mu, log_var = self.get_code(input_)
        return embedding.detach().cpu().numpy().T


class NeuroDataset(Dataset):
    """Neural activity dataset."""

    def __init__(self, data, transform=None):

        self.data = data.T
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'vector': self.data[idx].reshape(-1, 1), 'target': 0}

        if self.transform:
            sample = self.transform(sample)

        return self.data[idx], -42, idx
        # return sample