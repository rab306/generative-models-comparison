"""
Simple Quantitative Evaluation Script for VAE
Computes FID and Inception Score only.
Run: python scripts/evaluate_simple.py
"""

import torch
import glob
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("⚠️  torchmetrics not installed. Run: pip install torchmetrics")
    exit(1)

# Direct imports from config
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models import VAE
from utils import get_cifar10_loaders

# Initialize config
config = get_config()


def find_latest_vae_run():
    """Find latest VAE training run"""
    runs = glob.glob('results/vae_run_*')
    if not runs:
        raise Exception("No VAE runs found in results/")
    return max(runs, key=os.path.getctime)


def load_best_vae_model(run_dir, config):
    """Load best VAE model from checkpoint"""
    model = VAE(latent_dim=config.VAE_LATENT_DIM).to(config.DEVICE)
    
    # Try to load best_model.pth first
    best_path = os.path.join(run_dir, 'best_model.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"  ✅ Loaded VAE from epoch {epoch}")
    else:
        # Fallback: load latest checkpoint
        checkpoints = glob.glob(os.path.join(run_dir, 'checkpoints', '*.pth'))
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(run_dir, '*.pth'))
        if not checkpoints:
            raise Exception(f"No checkpoints found in {run_dir}")
        
        latest = max(checkpoints, key=os.path.getctime)
        checkpoint = torch.load(latest, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"  ✅ Loaded VAE checkpoint from epoch {epoch}")
    
    model.eval()
    return model


@torch.no_grad()
def generate_vae_samples(model, config, n_samples=2000, batch_size=64):
    """Generate samples from VAE decoder"""
    samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"  Generating {n_samples} VAE samples...")
    for _ in tqdm(range(n_batches), desc="  ", leave=False):
        z = torch.randn(batch_size, config.VAE_LATENT_DIM).to(config.DEVICE)
        batch_samples = model.decode(z)
        samples.append(batch_samples.cpu())
    
    samples = torch.cat(samples)[:n_samples]
    return samples


def get_real_images(config, n_samples=5000):
    """Load real CIFAR-10 test images"""
    print(f"  Loading {n_samples} real CIFAR-10 images...")
    _, test_loader = get_cifar10_loaders(
        batch_size=128, 
        normalize_to_minus_one=False,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR
    )
    
    real_images = []
    for images, _ in test_loader:
        real_images.append(images)
        if len(torch.cat(real_images)) >= n_samples:
            break
    
    real_images = torch.cat(real_images)[:n_samples]
    return real_images


def prepare_for_metrics(images):
    """Convert images to format expected by torchmetrics (0-255 uint8)"""
    images = (images * 255).clamp(0, 255).byte()
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return images


def compute_fid(real_images, generated_images):
    """Compute Fréchet Inception Distance"""
    print("  Computing FID...")
    real = prepare_for_metrics(real_images)
    gen = prepare_for_metrics(generated_images)
    
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real, real=True)
    fid.update(gen, real=False)
    fid_score = fid.compute().item()
    
    return fid_score


def compute_is(generated_images):
    """Compute Inception Score"""
    print("  Computing Inception Score...")
    gen = prepare_for_metrics(generated_images)
    
    inception = InceptionScore(normalize=True)
    inception.update(gen)
    is_mean, is_std = inception.compute()
    
    return is_mean.item(), is_std.item()


def main():
    print("\n" + "="*60)
    print("🔬 VAE QUANTITATIVE EVALUATION")
    print("="*60)
    
    # Find and load model
    print("\n📂 Finding latest VAE run...")
    vae_run_dir = find_latest_vae_run()
    print(f"  Run: {vae_run_dir}")
    
    print("\n🔄 Loading VAE model...")
    vae_model = load_best_vae_model(vae_run_dir, config)
    
    # Load real images
    print("\n🖼️  Loading real CIFAR-10 test images...")
    real_images = get_real_images(config, n_samples=5000)
    print(f"  ✅ Loaded {len(real_images)} real images")
    
    # Generate samples
    print("\n🎨 Generating VAE samples...")
    vae_samples = generate_vae_samples(vae_model, config, n_samples=2000)
    print(f"  ✅ Generated {len(vae_samples)} samples")
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    fid_score = compute_fid(real_images, vae_samples)
    is_mean, is_std = compute_is(vae_samples)
    
    # Print results
    print("\n" + "="*60)
    print("📈 RESULTS")
    print("="*60)
    print(f"FID Score:       {fid_score:.2f} (lower is better)")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f} (higher is better)")
    print("="*60)
    
    # Save to CSV
    results_df = pd.DataFrame({
        'Model': ['VAE'],
        'FID': [fid_score],
        'IS_Mean': [is_mean],
        'IS_Std': [is_std]
    })
    
    csv_path = os.path.join(vae_run_dir, 'metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to: {csv_path}")
    
    print("\n✨ Evaluation complete!\n")


if __name__ == "__main__":
    main()