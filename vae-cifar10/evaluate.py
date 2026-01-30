import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

from config import Config
from models import VAE
from utils import get_cifar10_dataloaders, FIDCalculator, plot_training_curves
import wandb


def generate_samples_for_fid(model, num_samples, batch_size, save_dir, device):
    """Generate samples and save them as images for FID calculation"""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    num_generated = 0

    print(f"Generating {num_samples} samples...")

    with torch.no_grad():
        while num_generated < num_samples:
            batch_size_actual = min(batch_size, num_samples - num_generated)

            # Generate samples
            samples = model.sample(batch_size_actual, device)

            # Save each image
            for i in range(batch_size_actual):
                img_path = os.path.join(save_dir, f'sample_{num_generated:05d}.png')
                save_image(samples[i], img_path, normalize=False)
                num_generated += 1

            if num_generated % 1000 == 0:
                print(f"Generated {num_generated}/{num_samples} samples")

    print(f"All samples saved to {save_dir}")


def save_real_images_for_fid(test_loader, num_samples, save_dir):
    """Save real CIFAR-10 test images for FID calculation"""
    os.makedirs(save_dir, exist_ok=True)

    num_saved = 0

    print(f"Saving {num_samples} real images...")

    for data, _ in test_loader:
        for i in range(data.size(0)):
            if num_saved >= num_samples:
                break

            img_path = os.path.join(save_dir, f'real_{num_saved:05d}.png')
            save_image(data[i], img_path, normalize=False)
            num_saved += 1

        if num_saved >= num_samples:
            break

    print(f"All real images saved to {save_dir}")


def load_images_from_folder(folder_path, max_images=10000):
    """Load images from a folder"""
    images = []

    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])[:max_images]

    for img_file in tqdm(image_files, desc=f"Loading images from {folder_path}"):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('RGB')

        # Convert to tensor [0, 1]
        transform = transforms.ToTensor()
        img_tensor = transform(img)
        images.append(img_tensor)

    return images


def main():
    # Configuration
    config = Config()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = VAE(config).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, 'vae_best.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config.checkpoint_dir, 'vae_final.pt')

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    _, test_loader = get_cifar10_dataloaders(
        batch_size=config.batch_size,
        data_dir=config.data_dir
    )

    # Generate samples
    generated_dir = './fid_generated'
    real_dir = './fid_real'

    generate_samples_for_fid(
        model,
        config.num_samples,
        config.fid_batch_size,
        generated_dir,
        device
    )

    save_real_images_for_fid(
        test_loader,
        config.num_samples,
        real_dir
    )

    # Calculate FID
    print("\nCalculating FID score from scratch...")
    fid_calc = FIDCalculator(device)

    # Load images
    print("\nLoading images...")
    real_images = load_images_from_folder(real_dir, config.num_samples)
    generated_images = load_images_from_folder(generated_dir, config.num_samples)

    print(f"Loaded {len(real_images)} real images")
    print(f"Loaded {len(generated_images)} generated images")

    # Calculate statistics
    print("\nCalculating statistics for real images...")
    mu_real, sigma_real = fid_calc.calculate_activation_statistics(real_images, batch_size=50)

    print("\nCalculating statistics for generated images...")
    mu_gen, sigma_gen = fid_calc.calculate_activation_statistics(generated_images, batch_size=50)

    # Calculate FID
    print("\nCalculating Fréchet distance...")
    fid_value = fid_calc.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    print(f"\n{'=' * 50}")
    print(f"FID Score: {fid_value:.2f}")
    print(f"{'=' * 50}")

    # Generate final sample grid
    print("\nGenerating final sample grid...")
    final_samples = model.sample(64, device)
    save_image(final_samples.cpu(), 'final_samples_eval.png', nrow=8, normalize=False)

    print(f"\nEvaluation complete!")
    print(f"Final samples saved to: final_samples_eval.png")


if __name__ == '__main__':
    main()
