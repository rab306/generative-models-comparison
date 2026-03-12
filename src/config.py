# src/config.py

import torch
import os
import argparse
import json

class Config:
    # DATA CONFIGURATION
    DATASET_NAME = "CIFAR10"
    NUM_CHANNELS = 3
    IMAGE_SIZE = 32
    DATA_DIR = "./data"
    
    # TRAINING COMMON
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0
    
    # Batch sizes
    BATCH_SIZE_VAE = 64
    BATCH_SIZE_DDPM = 64
    
    # Training epochs
    EPOCHS_VAE = 50
    EPOCHS_DDPM = 50
    
    # VAE SPECIFIC
    VAE_LATENT_DIM = 64  
    VAE_LEARNING_RATE = 1e-3
    VAE_BETA = 1.0  
    
    # DDPM SPECIFIC
    DDPM_TIMESTEPS = 1000  # ✅ Standard DDPM from original paper
    DDPM_LEARNING_RATE = 2e-4
    DDPM_IMAGE_CHANNELS = 3
    DDPM_CHANNELS = 64
    DDPM_CHANNEL_MULTS = (1, 2, 2, 2)
    DDPM_BETA_START = 1e-4      # Kept for reference (not used with cosine schedule)
    DDPM_BETA_END = 0.02         # Kept for reference (not used with cosine schedule)
    DDPM_BETA_SCHEDULE = 'cosine'  # ✅ Cosine annealing schedule
    
    # EVALUATION
    NUM_GENERATED_SAMPLES = 5000
    NUM_FID_REAL_SAMPLES = 10000
    NUM_VISUALIZE_SAMPLES = 36
    SAMPLE_INTERVAL = 5
    SAVE_INTERVAL = 5
    
    RANDOM_SEED = 42
    
    # PATHS 
    BASE_DIR = None
    CHECKPOINT_DIR = None
    LOG_DIR = None
    RESULTS_DIR = None
    SAMPLE_DIR = None
    
    def __init__(self, args=None):
        """Initialize paths and apply command-line args"""
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.CHECKPOINT_DIR = os.path.join(self.BASE_DIR, "checkpoints")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        self.SAMPLE_DIR = os.path.join(self.RESULTS_DIR, "samples")
        
        for dir_path in [self.CHECKPOINT_DIR, self.LOG_DIR, self.RESULTS_DIR, self.SAMPLE_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        if args:
            self._apply_args(args)
    
    def _apply_args(self, args):
        """Override config with command-line arguments"""
        arg_to_attr = {
            'vae_beta': 'VAE_BETA',
            'vae_latent_dim': 'VAE_LATENT_DIM',
            'vae_learning_rate': 'VAE_LEARNING_RATE',
            'batch_size_vae': 'BATCH_SIZE_VAE',
            'epochs_vae': 'EPOCHS_VAE',
            'batch_size_ddpm': 'BATCH_SIZE_DDPM',
            'epochs_ddpm': 'EPOCHS_DDPM',
            'ddpm_timesteps': 'DDPM_TIMESTEPS',
            'sample_interval': 'SAMPLE_INTERVAL',
            'num_visualize_samples': 'NUM_VISUALIZE_SAMPLES',
            'seed': 'RANDOM_SEED',
        }
        
        for arg_name, attr_name in arg_to_attr.items():
            value = getattr(args, arg_name, None)
            if value is not None and hasattr(self, attr_name):
                setattr(self, attr_name, value)
                print(f"⚙ Overriding {attr_name}: {value}")
    
    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4, default=str)
        print(f"   📄 Config saved to: {filepath}")


def get_argparser():
    """Return argument parser for command-line overrides"""
    parser = argparse.ArgumentParser(description='Override configuration parameters')
    
    parser.add_argument('--vae_beta', type=float, help='VAE beta (KL weight)')
    parser.add_argument('--vae_latent_dim', type=int, help='VAE latent dimension')
    parser.add_argument('--vae_learning_rate', type=float, help='VAE learning rate')
    parser.add_argument('--batch_size_vae', type=int, help='VAE batch size')
    parser.add_argument('--epochs_vae', type=int, help='Number of VAE training epochs')
    parser.add_argument('--batch_size_ddpm', type=int, help='DDPM batch size')
    parser.add_argument('--epochs_ddpm', type=int, help='Number of DDPM training epochs')
    parser.add_argument('--ddpm_timesteps', type=int, help='DDPM number of timesteps')
    parser.add_argument('--sample_interval', type=int, help='Epochs between sample generation')
    parser.add_argument('--num_visualize_samples', type=int, help='Number of samples to visualize')
    parser.add_argument('--test', action='store_true', help='Run in test mode (1 epoch)')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    return parser


def get_config(args=None):
    """Create and return configuration object"""
    config = Config(args)
    
    if args and hasattr(args, 'test') and args.test:
        print("🧪 Running in TEST MODE - overriding epochs to 1")
        config.EPOCHS_VAE = 1
        config.EPOCHS_DDPM = 1
        config.NUM_VISUALIZE_SAMPLES = 4
        config.SAMPLE_INTERVAL = 1
    
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    return config