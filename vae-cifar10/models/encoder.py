import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network: x -> mu, log_var
    Uses convolutional layers to encode 32x32 images to latent space
    """

    def __init__(self, latent_dim, hidden_dims, img_channels=3):
        super(Encoder, self).__init__()

        modules = []
        in_channels = img_channels

        # Build convolutional layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim

        self.conv_layers = nn.Sequential(*modules)

        # Calculate flattened size after convolutions
        # For CIFAR-10 (32x32) with 4 stride-2 convs: 32 -> 16 -> 8 -> 4 -> 2
        self.flatten_size = hidden_dims[-1] * 2 * 2

        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # Flatten

        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)

        return mu, log_var