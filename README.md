# Generative Models from Scratch: VAE vs DDPM

A comprehensive implementation and comparative analysis of **Variational Autoencoders (VAE)** and **Denoising Diffusion Probabilistic Models (DDPM)** trained from scratch on CIFAR-10.

## Overview

This project implements two fundamental generative modeling approaches from first principles:

- **VAE**: Encoder-decoder with latent regularization (tested with MSE and BCE losses)
- **DDPM**: UNet-based iterative denoising (tested with 400 and 1000 timesteps)

All models are trained on CIFAR-10 and evaluated using FID and Inception Score metrics, with detailed quantitative and qualitative analysis.

## Key Results

| Model | FID Score | Inception Score | Training Time | Notes |
|-------|-----------|-----------------|---------------|-------|
| VAE (MSE) | 206.19 | 2.84 ± 0.36 | 30 min | Better latent space structure |
| VAE (BCE) | 243.01 | 2.30 ± 0.16 | 30 min | Severe KL collapse |
| DDPM (400 steps) | 94.73 | 4.13 ± 0.34 | 123.6 min | Fast generation (1.7 min/500 imgs) |
| DDPM (1000 steps) | **75.16** | **4.37 ± 0.46** | 125.9 min | Best quality (4.6 min/500 imgs) |

**Key Finding**: DDPM overwhelmingly outperforms VAE (56–61% better FID), confirming why diffusion models have displaced autoencoders in modern generative AI.

## Repository Structure

```
generative-models-comparison/
├── src/
│   ├── models.py              # VAE and DDPM architectures
│   ├── config.py              # Configuration class and argument parsing
│   ├── utils.py               # Data loading, visualization utilities
│   ├── train_vae.py           # VAE training script
│   ├── train_ddpm.py          # DDPM training script
│   ├── evaluate.py            # FID/IS evaluation for both models
│   └── visualize_losses.py    # Loss curve visualization
│
├── logs/                      # Model checkpoints (.pth, .csv, .json)
│   ├── vae_run_mse/
│   ├── vae_run_bce/
│   ├── ddpm_run_400/
│   └── ddpm_run_1000/
├── results/                   # Generated samples and plots
│   ├── vae_run_mse/
│   ├── vae_run_bce/
│   ├── ddpm_run_400/
│   └── ddpm_run_1000/
├── notebooks/
│   └── generative-models-comparison.ipynb  # Kaggle notebook with full pipeline
├── VAE_vs_DDPM_Comparative_Analysis.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Local Setup

```bash
git clone https://github.com/rab306/generative-models-comparison.git
cd generative-models-comparison

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Kaggle Notebook

The project is also set up as a Kaggle notebook for GPU-accelerated training:

```bash
!git clone https://github.com/rab306/generative-models-comparison.git
%cd ./generative-models-comparison
!pip install -r requirements.txt
```

## Usage

### Training

All training scripts automatically save:
- Model checkpoints
- Generated samples
- Loss histories as CSV/JSON for analysis

#### VAE Training

**MSE Loss (Recommended)**:
```bash
python src/train_vae.py --vae_latent_dim 64 --vae_beta 1.0 --epochs_vae 100
```
```

#### DDPM Training

**400 Timesteps (Fast Generation)**:
```bash
python src/train_ddpm.py --epochs_ddpm 100 --ddpm_timesteps 400
```

**1000 Timesteps (High Quality)**:
```bash
python src/train_ddpm.py --epochs_ddpm 100 --ddpm_timesteps 1000
```

### Evaluation

Compute FID and Inception Score metrics:

```bash
# Evaluate VAE
python src/evaluate.py --model vae

# Evaluate DDPM
python src/evaluate.py --model ddpm

# Evaluate all trained models
python src/evaluate.py --model both
```

### Visualization

Generate loss curve plots:

```bash
# Visualize all models
python src/visualize_losses.py --model both

# Individual models
python src/visualize_losses.py --model vae
python src/visualize_losses.py --model ddpm
```

## Key Implementation Details

### VAE Architecture

- **Encoder**: 3 Conv2d layers (64→128→256 channels) + 2 linear layers to latent space
- **Decoder**: Symmetric with ConvTranspose2d, final Sigmoid activation for [0,1] range
- **Loss**: MSE (Gaussian likelihood) or BCE (Bernoulli likelihood) + KL divergence
- **Parameters**: 2,108,163 trainable parameters
- **Training**: Adam (lr=1e-3), 100 epochs, ~30 minutes on T4 GPU

### DDPM Architecture

- **UNet Backbone**: 
  - 4 DownBlocks with channel multipliers [1, 2, 2, 2]
  - Middle block with attention
  - 4 UpBlocks with skip connections
  - Attention at resolution 4×4 and 2×2 only
  
- **Time Embedding**: Sinusoidal positional encoding (256-dim) injected via ResNet blocks
- **Normalization**: GroupNorm(32) throughout (superior to BatchNorm for small batch sizes)
- **Activation**: SiLU (better than ReLU for continuous denoising processes)
- **Parameters**: 5,589,635 trainable parameters
- **Noise Schedule**: Cosine annealing (superior to linear)
- **Training**: Adam (lr=2e-4), Cosine annealing, 100 epochs, ~125 minutes on T4 GPU


### Timestep Analysis

Experiments comparing 400 vs 1000 timesteps reveal:

| Aspect | Finding |
|--------|---------|
| **Training Loss** | Identical (0.057) — timesteps irrelevant to training |
| **Validation Loss** | Nearly identical (0.056 vs 0.062) |
| **Generation Speed** | 2.7× faster with 400 steps (1.7 min vs 4.6 min for 500 images) |
| **Sample Quality** | 20% FID improvement with 1000 steps (94.73 → 75.16) |

**Conclusion**: Timestep count affects ONLY sampling, not training. Choice depends on quality/speed trade-off.

## Hardware Requirements

- **GPU**: NVIDIA T4 (16GB VRAM minimum recommended)
- **RAM**: 16GB system RAM
- **Storage**: ~5GB (CIFAR-10 + models + results)
- **Training Time**: 
  - VAE: 30 minutes (100 epochs)
  - DDPM: 125 minutes (100 epochs)


## Kaggle Notebook

For quick experimentation on Kaggle (with free GPU):

1. Open notebook: `notebooks/generative-models-comparison.ipynb`
2. Follow cells in order (clone repo → install → train → evaluate)
3. Results save to Kaggle working directory automatically

## Key Insights

1. **DDPM >> VAE**: Diffusion models achieve 56–61% better FID than autoencoders, explaining the industry shift to diffusion-based generative AI.

2. **Loss Function Matters (VAE)**: MSE produces better generalization (206 FID) than BCE (243 FID) despite BCE's pixel-level sharpness, demonstrating the importance of latent space regularization.

3. **Timestep Trade-Off (DDPM)**: 400 timesteps are sufficient for many applications (94.73 FID, 2.7× faster), while 1000 timesteps maximize quality (75.16 FID) for offline synthesis.

4. **Training is Timestep-Agnostic**: DDPM converges identically with 400 or 1000 timesteps, confirming timesteps affect ONLY sampling, not learning.

5. **GPU Parallelism**: Batch generation is efficient—16 images take 3.6 seconds total (not 16 × per-image time) due to GPU parallelization.