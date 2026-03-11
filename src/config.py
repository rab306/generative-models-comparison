"""
Configuration file for VAE and DDPM training on CIFAR10.
All hyperparameters are centralized here for easy experimentation.
"""

import torch
import os

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

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAMPLE_DIR = os.path.join(RESULTS_DIR, "samples")

for dir_path in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, SAMPLE_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# VAE SPECIFIC CONFIGURATION
VAE_LATENT_DIM = 32
VAE_LEARNING_RATE = 1e-3
VAE_BETA = 0.001

# DDPM SPECIFIC CONFIGURATION
DDPM_TIMESTEPS = 400      # Reduced from standard 1000 for faster training
DDPM_LEARNING_RATE = 2e-4
DDPM_IMAGE_CHANNELS = 3

# UNet architecture
DDPM_CHANNELS = 64 
DDPM_CHANNEL_MULTS = (1, 2, 2, 2)

DDPM_EMA_DECAY = 0.9999

# Noise schedule parameters
DDPM_BETA_START = 1e-4
DDPM_BETA_END = 0.02
DDPM_BETA_SCHEDULE = 'linear'


# EVALUATION CONFIGURATION
NUM_GENERATED_SAMPLES = 5000
NUM_FID_REAL_SAMPLES = 10000
NUM_VISUALIZE_SAMPLES = 36
SAMPLE_INTERVAL = 5
SAVE_INTERVAL = 5


RANDOM_SEED = 42
