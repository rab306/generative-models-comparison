# src/utils.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from config import DATA_DIR, BATCH_SIZE_VAE, BATCH_SIZE_DDPM, NUM_WORKERS

def get_cifar10_loaders(batch_size, normalize_to_minus_one=False):
    """
    Returns train and test DataLoaders for CIFAR-10.
    """
    if normalize_to_minus_one:
        # DDPM standard normalization [-1, 1]
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # VAE standard normalization [0, 1]
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, 
        train=False, 
        download=True, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader

def save_image_grid(images, filename, nrow=8):
    """
    Saves a grid of images.
    images: Tensor of shape (N, C, H, W) in range [0, 1]
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # save_image with normalize=True expects data in [0, 1] or [-1, 1]
    # Since our VAE output is sigmoid (0, 1), we don't need to adjust.
    save_image(images, filename, nrow=nrow, normalize=True, pad_value=1.0)