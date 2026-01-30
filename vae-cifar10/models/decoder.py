import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder network: z -> x_reconstructed
    Uses transposed convolutions to decode latent vectors to images
    """

    def __init__(self, latent_dim, hidden_dims, img_channels=3):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        hidden_dims = hidden_dims[::-1]  # Reverse for decoder

        # Initial projection from latent to spatial
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 2 * 2)
        self.initial_size = 2

        # Build transpose convolutional layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )

        self.conv_layers = nn.Sequential(*modules)

        # Final layer to get RGB image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], img_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1] for BCE loss
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), -1, self.initial_size, self.initial_size)
        h = self.conv_layers(h)
        x_recon = self.final_layer(h)
        return x_recon
