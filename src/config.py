"""
Configuration file for VAE and DDPM training on CIFAR10.
Supports command-line overrides for flexible experimentation.
"""

import torch
import os
import argparse
import json

# Default Configuration
class Config:
    # DATA CONFIGURATION
    DATASET_NAME = "CIFAR10"
    NUM_CHANNELS = 3
    IMAGE_SIZE = 32
    DATA_DIR = "./data"
    
    # TRAINING COMMON
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4 if DEVICE == "cpu" else 0
    
    # Batch sizes
    BATCH_SIZE_VAE = 64
    BATCH_SIZE_DDPM = 64
    
    # Training epochs
    EPOCHS_VAE = 50
    EPOCHS_DDPM = 50
    
    # VAE SPECIFIC
    VAE_LATENT_DIM = 32
    VAE_LEARNING_RATE = 1e-3
    VAE_BETA = 0.001
    
    # DDPM SPECIFIC
    DDPM_TIMESTEPS = 400
    DDPM_LEARNING_RATE = 2e-4
    DDPM_IMAGE_CHANNELS = 3
    DDPM_CHANNELS = 64
    DDPM_CHANNEL_MULTS = (1, 2, 2, 2)
    DDPM_EMA_DECAY = 0.9999
    DDPM_BETA_START = 1e-4
    DDPM_BETA_END = 0.02
    DDPM_BETA_SCHEDULE = 'linear'
    
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
        # Set base directory
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set paths
        self.CHECKPOINT_DIR = os.path.join(self.BASE_DIR, "checkpoints")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        self.SAMPLE_DIR = os.path.join(self.RESULTS_DIR, "samples")
        
        # Create directories
        for dir_path in [self.CHECKPOINT_DIR, self.LOG_DIR, self.RESULTS_DIR, self.SAMPLE_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Apply command-line overrides
        if args:
            self._apply_args(args)
    
    def _apply_args(self, args):
        """Override config with command-line arguments"""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
                print(f"⚙ Overriding {key}: {value}")
    
    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4, default=str)


# Command Line Parser
def get_argparser():
    """Return argument parser for command-line overrides"""
    parser = argparse.ArgumentParser(description='Override configuration parameters')
    
    # Training parameters
    parser.add_argument('--epochs_vae', type=int, help='Number of VAE training epochs')
    parser.add_argument('--batch_size_vae', type=int, help='VAE batch size')
    parser.add_argument('--vae_latent_dim', type=int, help='VAE latent dimension')
    parser.add_argument('--vae_learning_rate', type=float, help='VAE learning rate')
    parser.add_argument('--vae_beta', type=float, help='VAE beta (KL weight)')
    
    # DDPM parameters
    parser.add_argument('--epochs_ddpm', type=int, help='Number of DDPM training epochs')
    parser.add_argument('--batch_size_ddpm', type=int, help='DDPM batch size')
    parser.add_argument('--ddpm_timesteps', type=int, help='DDPM number of timesteps')
    
    # Evaluation parameters
    parser.add_argument('--sample_interval', type=int, help='Epochs between sample generation')
    parser.add_argument('--num_visualize_samples', type=int, help='Number of samples to visualize')
    
    # Quick test mode
    parser.add_argument('--test', action='store_true', help='Run in test mode (1 epoch)')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    return parser


# Function to get config
def get_config(args=None):
    """Create and return configuration object"""
    config = Config(args)
    
    # Handle test mode
    if args and hasattr(args, 'test') and args.test:
        print("🧪 Running in TEST MODE - overriding epochs to 1")
        config.EPOCHS_VAE = 1
        config.EPOCHS_DDPM = 1
        config.NUM_VISUALIZE_SAMPLES = 4
        config.SAMPLE_INTERVAL = 1
    
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    return config


# ============================================================================
# For backward compatibility with existing scripts
# ============================================================================
# Create a default config instance for direct imports
_default_config = get_config()
DATASET_NAME = _default_config.DATASET_NAME
NUM_CHANNELS = _default_config.NUM_CHANNELS
IMAGE_SIZE = _default_config.IMAGE_SIZE
DATA_DIR = _default_config.DATA_DIR
DEVICE = _default_config.DEVICE
NUM_WORKERS = _default_config.NUM_WORKERS
BATCH_SIZE_VAE = _default_config.BATCH_SIZE_VAE
BATCH_SIZE_DDPM = _default_config.BATCH_SIZE_DDPM
EPOCHS_VAE = _default_config.EPOCHS_VAE
EPOCHS_DDPM = _default_config.EPOCHS_DDPM
CHECKPOINT_DIR = _default_config.CHECKPOINT_DIR
LOG_DIR = _default_config.LOG_DIR
RESULTS_DIR = _default_config.RESULTS_DIR
SAMPLE_DIR = _default_config.SAMPLE_DIR
VAE_LATENT_DIM = _default_config.VAE_LATENT_DIM
VAE_LEARNING_RATE = _default_config.VAE_LEARNING_RATE
VAE_BETA = _default_config.VAE_BETA
DDPM_TIMESTEPS = _default_config.DDPM_TIMESTEPS
DDPM_LEARNING_RATE = _default_config.DDPM_LEARNING_RATE
DDPM_IMAGE_CHANNELS = _default_config.DDPM_IMAGE_CHANNELS
DDPM_CHANNELS = _default_config.DDPM_CHANNELS
DDPM_CHANNEL_MULTS = _default_config.DDPM_CHANNEL_MULTS
DDPM_EMA_DECAY = _default_config.DDPM_EMA_DECAY
DDPM_BETA_START = _default_config.DDPM_BETA_START
DDPM_BETA_END = _default_config.DDPM_BETA_END
DDPM_BETA_SCHEDULE = _default_config.DDPM_BETA_SCHEDULE
NUM_GENERATED_SAMPLES = _default_config.NUM_GENERATED_SAMPLES
NUM_FID_REAL_SAMPLES = _default_config.NUM_FID_REAL_SAMPLES
NUM_VISUALIZE_SAMPLES = _default_config.NUM_VISUALIZE_SAMPLES
SAMPLE_INTERVAL = _default_config.SAMPLE_INTERVAL
SAVE_INTERVAL = _default_config.SAVE_INTERVAL
RANDOM_SEED = _default_config.RANDOM_SEED
BASE_DIR = _default_config.BASE_DIR