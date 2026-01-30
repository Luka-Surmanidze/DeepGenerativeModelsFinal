import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import wandb
import os


def visualize_reconstructions(model, test_loader, epoch, save_dir='./reconstructions', num_images=8, device='cuda'):
    """Visualize original and reconstructed images"""
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Get a batch of test images
        data, _ = next(iter(test_loader))
        data = data[:num_images].to(device)

        # Reconstruct
        recon, _, _ = model(data)

        # Concatenate original and reconstruction
        comparison = torch.cat([data, recon])

        # Save
        filepath = os.path.join(save_dir, f'recon_epoch_{epoch:03d}.png')
        save_image(comparison.cpu(), filepath, nrow=num_images, normalize=False)

        # Log to wandb
        wandb.log({
            "reconstructions": wandb.Image(filepath),
            "epoch": epoch
        })


def visualize_samples(model, epoch, save_dir='./samples', num_images=64, device='cuda'):
    """Generate and visualize samples from prior"""
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        samples = model.sample(num_images, device)

        # Save
        filepath = os.path.join(save_dir, f'samples_epoch_{epoch:03d}.png')
        save_image(samples.cpu(), filepath, nrow=8, normalize=False)

        # Log to wandb
        wandb.log({
            "samples": wandb.Image(filepath),
            "epoch": epoch
        })


def plot_training_curves(history_file='training_history.png'):
    """Plot loss curves from wandb history"""
    history = wandb.run.history()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(history['epoch'], history['train/loss'], label='Train')
    axes[0].plot(history['epoch'], history['val/loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Reconstruction loss
    axes[1].plot(history['epoch'], history['train/recon_loss'], label='Train')
    axes[1].plot(history['epoch'], history['val/recon_loss'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)

    # KL loss
    axes[2].plot(history['epoch'], history['train/kl_loss'], label='Train')
    axes[2].plot(history['epoch'], history['val/kl_loss'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Loss')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(history_file, dpi=150)
    wandb.log({"training_curves": wandb.Image(history_file)})
    plt.show()
