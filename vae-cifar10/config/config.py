class Config:
    """Configuration class for VAE training"""

    # Data
    dataset = 'CIFAR10'
    data_dir = './data'
    img_size = 32
    img_channels = 3

    # Model Architecture
    latent_dim = 128  # Size of latent space z
    hidden_dims = [32, 64, 128, 256]  # Encoder/Decoder hidden dimensions

    # Training
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3
    beta = 1.0  # KL weight (beta=1 for standard VAE)

    # Loss type: 'mse' or 'bce'
    reconstruction_loss = 'bce'  # BCE is standard for images in [0,1]

    # Checkpointing
    checkpoint_dir = './checkpoints'
    save_every = 10  # Save checkpoint every N epochs

    # Evaluation
    num_samples = 10000  # Number of samples to generate for FID
    fid_batch_size = 50

    # Wandb
    project_name = 'vae-cifar10-baseline'
    experiment_name = 'baseline_vae_latent128'

    def __repr__(self):
        return f"Config(latent_dim={self.latent_dim}, beta={self.beta}, loss={self.reconstruction_loss})"

