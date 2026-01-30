import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder
    Combines encoder and decoder with reparameterization trick
    """

    def __init__(self, config):
        super(VAE, self).__init__()

        self.latent_dim = config.latent_dim
        self.beta = config.beta
        self.reconstruction_loss = config.reconstruction_loss

        self.encoder = Encoder(
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            img_channels=config.img_channels
        )

        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            img_channels=config.img_channels
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Encode
        mu, log_var = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, log_var

    def sample(self, num_samples, device):
        """
        Generate samples by sampling from prior p(z) = N(0, I)
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
        return samples
