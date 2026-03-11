#!/usr/bin/env python3
"""
VAE Training Script for CIFAR-10

This script trains a Variational Autoencoder on CIFAR-10 with support for:
- Command-line argument overrides
- Checkpointing and best model saving
- Periodic sample generation for visual monitoring
- Validation loss tracking

Usage:
    python src/train_vae.py                    # Default config
    python src/train_vae.py --test             # Test mode (1 epoch)
    python src/train_vae.py --vae_beta 0.5     # Override KL weight
    python src/train_vae.py --vae_latent_dim 128 --epochs_vae 100
"""

import torch
import torch.optim as optim
import os
from datetime import datetime
from tqdm import tqdm

from config import get_config, get_argparser
from utils import get_cifar10_loaders, save_image_grid, set_seed


def train_vae(config):
    """
    Train VAE with given configuration.
    
    Args:
        config: Config object with hyperparameters
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting VAE Training on {config.DEVICE}")
    print(f"{'='*60}")
    print(f"   Batch Size:      {config.BATCH_SIZE_VAE}")
    print(f"   Latent Dim:      {config.VAE_LATENT_DIM}")
    print(f"   KL Beta:         {config.VAE_BETA}")
    print(f"   Epochs:          {config.EPOCHS_VAE}")
    print(f"   Learning Rate:   {config.VAE_LEARNING_RATE}")
    print(f"   Num Workers:     {config.NUM_WORKERS}")
    print(f"{'='*60}\n")
    
    # Import models inside function to avoid circular imports
    from models import VAE, vae_loss_function
    
    # 1. Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.RESULTS_DIR, f"vae_run_{timestamp}")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save config to run directory for reproducibility
    config.save(os.path.join(run_dir, "config.json"))
    
    # 2. Data Loaders
    # normalize_to_minus_one=False keeps data in [0, 1] for VAE (BCE/MSE + Sigmoid)
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config.BATCH_SIZE_VAE, 
        normalize_to_minus_one=False,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR
    )
    print(f"   Training samples:   {len(train_loader.dataset):,}")
    print(f"   Validation samples: {len(test_loader.dataset):,}\n")
    
    # 3. Model, Optimizer
    model = VAE(latent_dim=config.VAE_LATENT_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.VAE_LEARNING_RATE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters:   {total_params:,}")
    print(f"   Trainable Params:   {trainable_params:,}\n")
    
    # 4. Fixed Noise for Consistent Sampling
    # Using the same noise vector every epoch allows direct visual comparison
    fixed_noise = torch.randn(config.NUM_VISUALIZE_SAMPLES, config.VAE_LATENT_DIM).to(config.DEVICE)
    
    # 5. Training Tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 6. Training Loop
    for epoch in range(1, config.EPOCHS_VAE + 1):
        # ========== TRAINING ==========
        model.train()
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            x_recon, mu, logvar = model(data)
            
            # Compute Loss (returns total, recon, kl components)
            total_loss, recon_loss, kl_loss = vae_loss_function(
                x_recon, data, mu, logvar, beta=config.VAE_BETA
            )
            
            # Backward Pass
            total_loss.backward()
            
            # Gradient Clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate Metrics
            total_loss_epoch += total_loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()
            num_batches += 1
            
            # Update progress bar with per-sample losses
            progress_bar.set_postfix({
                'loss': f"{total_loss.item()/len(data):.3f}",
                'recon': f"{recon_loss.item()/len(data):.3f}",
                'kl': f"{kl_loss.item()/len(data):.3f}"
            })
        
        # Calculate average losses (per sample, averaged over batches)
        avg_total = total_loss_epoch / len(train_loader)
        avg_recon = recon_loss_epoch / len(train_loader)
        avg_kl = kl_loss_epoch / len(train_loader)
        
        # ========== VALIDATION ==========
        model.eval()
        val_total = 0
        val_recon = 0
        val_kl = 0
        
        with torch.no_grad():
            for test_data, _ in test_loader:
                test_data = test_data.to(config.DEVICE)
                x_recon, mu, logvar = model(test_data)
                v_loss, v_recon, v_kl = vae_loss_function(
                    x_recon, test_data, mu, logvar, beta=config.VAE_BETA
                )
                val_total += v_loss.item()
                val_recon += v_recon.item()
                val_kl += v_kl.item()
        
        val_loss_avg = val_total / len(test_loader.dataset)
        val_recon_avg = val_recon / len(test_loader.dataset)
        val_kl_avg = val_kl / len(test_loader.dataset)
        
        # Track losses for plotting
        train_losses.append(avg_total)
        val_losses.append(val_loss_avg)
        
        # ========== LOGGING ==========
        kl_ratio = avg_kl / avg_recon if avg_recon > 0 else 0
        print(f"\nEpoch {epoch:03d} | "
              f"Train: {avg_total:.3f} (Recon: {avg_recon:.3f}, KL: {avg_kl:.3f}, Ratio: {kl_ratio:.3f}) | "
              f"Val: {val_loss_avg:.3f}")
        
        # ========== GENERATE SAMPLES ==========
        if epoch % config.SAMPLE_INTERVAL == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                samples = model.decode(fixed_noise)
                sample_filename = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
                save_image_grid(samples, sample_filename, nrow=int(config.NUM_VISUALIZE_SAMPLES**0.5))
                print(f"   📸 Samples saved to: {sample_filename}")
        
        # ========== SAVE CHECKPOINT ==========
        if epoch % config.SAVE_INTERVAL == 0 or epoch == config.EPOCHS_VAE:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_total,
                'val_loss': val_loss_avg,
                'config': config.to_dict(),
            }, checkpoint_path)
            
            # Save Best Model (based on validation loss)
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_path = os.path.join(run_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss_avg,
                }, best_path)
                print(f"   ⭐ New best model! (Val Loss: {val_loss_avg:.3f})")
        
        print("")  # Empty line for readability
    
    # ========== TRAINING COMPLETE ==========
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"   Results saved to:  {run_dir}")
    print(f"   Best Val Loss:     {best_val_loss:.3f}")
    print(f"   Final Train Loss:  {train_losses[-1]:.3f}")
    print(f"   Final Val Loss:    {val_losses[-1]:.3f}")
    print(f"{'='*60}\n")
    
    # Save loss history for plotting
    import json
    with open(os.path.join(run_dir, "loss_history.json"), 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, f, indent=2)
    
    return model, run_dir, train_losses, val_losses


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args()
    
    # Get configuration with overrides
    config = get_config(args)
    
    # Print configuration summary
    print(f"\n📋 Configuration Summary:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Random Seed: {config.RANDOM_SEED}")
    print(f"   Results Dir: {config.RESULTS_DIR}")
    print("")
    
    # Run training
    model, run_dir, train_losses, val_losses = train_vae(config)
    
    print(f"🎉 VAE Training Finished Successfully!")
    print(f"   Next step: Run 'python src/train_ddpm.py' for Diffusion Model")