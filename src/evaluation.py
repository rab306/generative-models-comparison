"""
Universal Quantitative Evaluation Script for VAE and DDPM
Computes FID and Inception Score for either model.

Usage:
    python evaluate.py --model vae
    python evaluate.py --model ddpm
    python evaluate.py --model both
"""

import torch
import glob
import os
import argparse
import time
from tqdm import tqdm
import pandas as pd

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("⚠️  torchmetrics not installed. Run: pip install torchmetrics")
    exit(1)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models import VAE, UNet, DDPM
from utils import get_cifar10_loaders

config = get_config()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_latest_run(model_type):
    """Find latest run directory for given model type."""
    possible_paths = [
        f'results/{model_type}_run*',
        f'/kaggle/working/generative-models-comparison/results/{model_type}_run*',
        os.path.expanduser(f'~/results/{model_type}_run*'),
    ]
    
    for pattern in possible_paths:
        runs = glob.glob(pattern)
        if runs:
            print(f"✅ Found {model_type.upper()} runs in: {pattern}")
            return max(runs, key=os.path.getctime)
    
    print(f"❌ No {model_type.upper()} training runs found in any of:")
    for p in possible_paths:
        print(f"   - {p}")
    return None



def get_real_images(n_samples=2000, normalize_to_minus_one=False):
    """Load real CIFAR-10 test images"""
    print(f"  Loading {n_samples} real CIFAR-10 images...")
    _, test_loader = get_cifar10_loaders(
        batch_size=16, 
        normalize_to_minus_one=normalize_to_minus_one,
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
    # Ensure images are in [0, 1] range
    images = (images + 1) / 2 if images.min() < 0 else images  # Convert [-1, 1] to [0, 1]
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


# ============================================================================
# VAE EVALUATION
# ============================================================================

def load_best_vae_model(run_dir):
    """Load best VAE model from checkpoint"""
    model = VAE(latent_dim=config.VAE_LATENT_DIM).to(config.DEVICE)
    
    best_path = os.path.join(run_dir, 'best_model.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"  ✅ Loaded VAE from epoch {epoch}")
    else:
        checkpoints = glob.glob(os.path.join(run_dir, 'checkpoints', '*.pth'))
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(run_dir, '*.pth'))
        if not checkpoints:
            raise Exception(f"No checkpoints found in {run_dir}")
        
        latest = max(checkpoints, key=os.path.getctime)
        checkpoint = torch.load(latest, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"  ✅ Loaded VAE from epoch {epoch}")
    
    model.eval()
    return model


@torch.no_grad()
def generate_vae_samples(model, n_samples=500, batch_size=16):
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


def evaluate_vae(run_dir):
    """Evaluate VAE model"""
    print("\n" + "="*70)
    print("🔬 VAE QUANTITATIVE EVALUATION")
    print("="*70)
    
    print("\n🔄 Loading VAE model...")
    vae_model = load_best_vae_model(run_dir)
    
    print("\n🖼️  Loading real CIFAR-10 images...")
    real_images = get_real_images(n_samples=2000, normalize_to_minus_one=False)
    print(f"  ✅ Loaded {len(real_images)} real images")
    
    print("\n🎨 Generating VAE samples...")
    vae_samples = generate_vae_samples(vae_model, n_samples=500)
    print(f"  ✅ Generated {len(vae_samples)} samples")
    
    print("\n📊 Computing metrics...")
    fid_score = compute_fid(real_images, vae_samples)
    is_mean, is_std = compute_is(vae_samples)
    
    print("\n" + "="*70)
    print("📈 VAE RESULTS")
    print("="*70)
    print(f"FID Score:       {fid_score:.2f} (lower is better)")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f} (higher is better)")
    print("="*70)
    
    # Save to CSV
    results_df = pd.DataFrame({
        'Model': ['VAE'],
        'FID': [fid_score],
        'IS_Mean': [is_mean],
        'IS_Std': [is_std]
    })
    
    csv_path = os.path.join(run_dir, 'evaluation_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to: {csv_path}")
    
    return {
        'model': 'VAE',
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std
    }


# ============================================================================
# DDPM EVALUATION
# ============================================================================

def load_best_ddpm_model(run_dir):
    """Load best DDPM model from checkpoint"""
    unet = UNet(
        in_channels=config.DDPM_IMAGE_CHANNELS,
        out_channels=config.DDPM_IMAGE_CHANNELS,
        time_dim=256,
        base_channels=config.DDPM_CHANNELS,
        channel_mults=config.DDPM_CHANNEL_MULTS
    ).to(config.DEVICE)
    
    model = DDPM(
        unet=unet,
        timesteps=config.DDPM_TIMESTEPS,
        schedule=config.DDPM_BETA_SCHEDULE
    ).to(config.DEVICE)
    
    best_path = os.path.join(run_dir, 'best_model.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"  ✅ Loaded DDPM from epoch {epoch}")
    else:
        checkpoints = glob.glob(os.path.join(run_dir, 'checkpoints', '*.pth'))
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(run_dir, '*.pth'))
        if not checkpoints:
            raise Exception(f"No checkpoints found in {run_dir}")
        
        latest = max(checkpoints, key=os.path.getctime)
        checkpoint = torch.load(latest, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"  ✅ Loaded DDPM from epoch {epoch}")
    
    model.eval()
    return model


@torch.no_grad()
def sample_ddpm(model, n_samples=500, batch_size=16):
    """Generate samples from DDPM with CORRECT posterior variance and clipping"""
    samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"  Generating {n_samples} DDPM samples (this may take a while)...")
    
    for batch_idx in tqdm(range(n_batches), desc="  ", leave=False):
        batch_size_actual = min(batch_size, n_samples - batch_idx * batch_size)
        
        # Start from pure noise
        x = torch.randn(batch_size_actual, 3, 32, 32, device=config.DEVICE)
        
        # Reverse diffusion process
        for t in reversed(range(model.timesteps)):
            t_tensor = torch.full((batch_size_actual,), t, device=config.DEVICE, dtype=torch.long)
            
            # Predict noise
            noise_pred = model.denoise(x, t_tensor)
            
            # Get alpha values
            alpha = model.alphas[t]
            alpha_bar = model.alpha_bars[t]
            beta = model.betas[t]
            
            def bc(v): return v.view(1, 1, 1, 1)
            
            # ✅ Reconstruct x_0 with clipping (the fix!)
            x0_pred = (x - bc(torch.sqrt(1 - alpha_bar)) * noise_pred) / bc(torch.sqrt(alpha_bar))
            x0_pred = x0_pred.clamp(-1, 1)  # CRITICAL!
            
            if t > 0:
                alpha_bar_prev = model.alpha_bars[t - 1]
                
                # Posterior mean (DDPM paper eq. 7)
                coeff1 = bc(torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar))
                coeff2 = bc(torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
                x_mean = coeff1 * x0_pred + coeff2 * x
                
                # Posterior variance
                beta_tilde = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta
                x = x_mean + torch.sqrt(bc(beta_tilde)) * torch.randn_like(x)
            else:
                x = x0_pred
        
        samples.append(x.cpu())
    
    samples = torch.cat(samples)[:n_samples]
    return samples


def evaluate_ddpm(run_dir):
    """Evaluate DDPM model"""
    print("\n" + "="*70)
    print("🔬 DDPM QUANTITATIVE EVALUATION")
    print("="*70)
    
    print("\n🔄 Loading DDPM model...")
    ddpm_model = load_best_ddpm_model(run_dir)
    
    print("\n🖼️  Loading real CIFAR-10 images...")
    real_images = get_real_images(n_samples=2000, normalize_to_minus_one=True)
    print(f"  ✅ Loaded {len(real_images)} real images")
    
    print("\n🎨 Generating DDPM samples...")
    print("   (This takes ~30 minutes, please be patient...)")
    start_time = time.time()
    ddpm_samples = sample_ddpm(ddpm_model, n_samples=500)
    elapsed = time.time() - start_time
    print(f"  ✅ Generated {len(ddpm_samples)} samples in {elapsed/60:.1f} minutes")
    
    print("\n📊 Computing metrics...")
    fid_score = compute_fid(real_images, ddpm_samples)
    is_mean, is_std = compute_is(ddpm_samples)
    
    print("\n" + "="*70)
    print("📈 DDPM RESULTS")
    print("="*70)
    print(f"FID Score:       {fid_score:.2f} (lower is better)")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f} (higher is better)")
    print(f"Generation Time: {elapsed/60:.1f} minutes for 500 images")
    print("="*70)
    
    # Save to CSV
    results_df = pd.DataFrame({
        'Model': ['DDPM'],
        'FID': [fid_score],
        'IS_Mean': [is_mean],
        'IS_Std': [is_std],
        'Generation_Time_Minutes': [elapsed/60]
    })
    
    csv_path = os.path.join(run_dir, 'evaluation_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to: {csv_path}")
    
    return {
        'model': 'DDPM',
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std,
        'generation_time': elapsed/60
    }


# ============================================================================
# COMPARISON
# ============================================================================

def compare_models(vae_results, ddpm_results):
    """Compare VAE and DDPM results"""
    print("\n" + "="*70)
    print("📊 VAE vs DDPM COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<20} {'VAE':<20} {'DDPM':<20} {'Better':<10}")
    print("-"*70)
    
    # FID (lower is better)
    fid_winner = "VAE" if vae_results['fid'] < ddpm_results['fid'] else "DDPM"
    print(f"{'FID':<20} {vae_results['fid']:<20.2f} {ddpm_results['fid']:<20.2f} {fid_winner:<10}")
    
    # IS (higher is better)
    is_winner = "DDPM" if ddpm_results['is_mean'] > vae_results['is_mean'] else "VAE"
    print(f"{'Inception Score':<20} {vae_results['is_mean']:<20.4f} {ddpm_results['is_mean']:<20.4f} {is_winner:<10}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VAE and/or DDPM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --model vae
  python evaluate.py --model ddpm
  python evaluate.py --model both
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['vae', 'ddpm', 'both'],
        default='both',
        help='Which model to evaluate (default: both)'
    )
    
    args = parser.parse_args()
    
    results = {}
    
    try:
        if args.model in ['vae', 'both']:
            print("\n🔍 Evaluating VAE...")
            vae_run = find_latest_run('vae')
            print(f"   Run: {vae_run}")
            vae_results = evaluate_vae(vae_run)
            results['vae'] = vae_results
        
        if args.model in ['ddpm', 'both']:
            print("\n🔍 Evaluating DDPM...")
            ddpm_run = find_latest_run('ddpm')
            print(f"   Run: {ddpm_run}")
            ddpm_results = evaluate_ddpm(ddpm_run)
            results['ddpm'] = ddpm_results
        
        if args.model == 'both' and len(results) == 2:
            compare_models(results['vae'], results['ddpm'])
        
        print("\n✨ Evaluation complete!\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        exit(1)


if __name__ == "__main__":
    main()