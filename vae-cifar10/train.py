import torch
import torch.optim as optim
import wandb
import os

from config import Config
from models import VAE
from utils import get_cifar10_dataloaders, visualize_reconstructions, visualize_samples
from training import train_epoch, validate


def save_checkpoint(model, optimizer, epoch, config, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    # Configuration
    config = Config()

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs('./samples', exist_ok=True)
    os.makedirs('./reconstructions', exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config.batch_size,
        data_dir=config.data_dir
    )

    # Model
    model = VAE(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config=vars(config)
    )

    print("Starting training...")
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, epoch, config, device
        )

        # Validate
        val_loss, val_recon, val_kl = validate(
            model, test_loader, epoch, config, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/recon_loss': train_recon,
            'train/kl_loss': train_kl,
            'val/loss': val_loss,
            'val/recon_loss': val_recon,
            'val/kl_loss': val_kl,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})')
        print(f'Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})')

        # Visualize reconstructions and samples
        if epoch % 5 == 0 or epoch == 1:
            visualize_reconstructions(model, test_loader, epoch, device=device)
            visualize_samples(model, epoch, device=device)

        # Save checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, config, f'vae_epoch_{epoch:03d}.pt')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, config, 'vae_best.pt')
            print(f"New best model! Val loss: {val_loss:.4f}")

    # Save final model
    save_checkpoint(model, optimizer, config.epochs, config, 'vae_final.pt')
    print("Training completed!")

    wandb.finish()


if __name__ == '__main__':
    main()
