import torch
from tqdm import tqdm
from training.losses import vae_loss


def train_epoch(model, train_loader, optimizer, epoch, config, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, log_var = model(data)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(
            data, recon_batch, mu, log_var,
            beta=config.beta,
            reconstruction_loss=config.reconstruction_loss
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item()
        })

    # Average losses
    num_batches = len(train_loader)
    avg_loss = train_loss / num_batches
    avg_recon = train_recon / num_batches
    avg_kl = train_kl / num_batches

    return avg_loss, avg_recon, avg_kl


def validate(model, test_loader, epoch, config, device):
    """Validate on test set"""
    model.eval()
    val_loss = 0
    val_recon = 0
    val_kl = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kl_loss = vae_loss(
                data, recon_batch, mu, log_var,
                beta=config.beta,
                reconstruction_loss=config.reconstruction_loss
            )

            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()

    num_batches = len(test_loader)
    avg_loss = val_loss / num_batches
    avg_recon = val_recon / num_batches
    avg_kl = val_kl / num_batches

    return avg_loss, avg_recon, avg_kl
