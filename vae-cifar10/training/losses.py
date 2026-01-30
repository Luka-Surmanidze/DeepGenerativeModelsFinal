import torch
import torch.nn.functional as F


def vae_loss(x, x_recon, mu, log_var, beta=1.0, reconstruction_loss='bce'):
    """
    VAE loss = Reconstruction loss + Beta * KL divergence

    Args:
        x: Original images
        x_recon: Reconstructed images
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        beta: Weight for KL term (beta=1 for standard VAE)
        reconstruction_loss: 'bce' or 'mse'

    Returns:
        total_loss, recon_loss, kl_loss (all averaged over batch)
    """
    batch_size = x.size(0)

    # Reconstruction loss
    if reconstruction_loss == 'bce':
        # Binary cross-entropy (standard for images in [0,1])
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    elif reconstruction_loss == 'mse':
        # Mean squared error
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    else:
        raise ValueError(f"Unknown loss: {reconstruction_loss}")

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss (averaged over batch)
    total_loss = (recon_loss + beta * kl_loss) / batch_size

    return total_loss, recon_loss / batch_size, kl_loss / batch_size