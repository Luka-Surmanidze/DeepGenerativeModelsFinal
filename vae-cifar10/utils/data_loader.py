import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloaders(batch_size=128, data_dir='./data', num_workers=2):
    """
    Load CIFAR-10 dataset and create dataloaders.
    Images are normalized to [0, 1] range for BCE loss.

    Args:
        batch_size: Batch size for training
        data_dir: Directory to store/load CIFAR-10
        num_workers: Number of workers for data loading

    Returns:
        train_loader, test_loader
    """
    # Transform: Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and changes to CHW format
    ])

    # Load datasets (labels not used - unconditional generation)
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, test_loader