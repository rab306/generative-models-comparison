# src/train_vae.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
from datetime import datetime
from tqdm import tqdm

from config import (
    DEVICE, EPOCHS_VAE, VAE_LEARNING_RATE, VAE_BETA, 
    SAVE_DIR, BATCH_SIZE_VAE, VAE_LATENT_DIM, CHECKPOINT_DIR
)
from models import VAE, vae_loss_function
from utils import get_cifar10_loaders, save_image_grid

def validate(model, test_loader):
    """Run validation loop"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(DEVICE)
            x_recon, mu, logvar = model(data)
            total_loss, _, _ = vae_loss_function(x_recon, data, mu, logvar, beta=VAE_BETA)
            val_loss += total_loss.item()
    return val_loss / len(test_loader)

def train_vae():
    print(f"🚀 Starting VAE Training on {DEVICE}")
    print(f"   Batch Size: {BATCH_SIZE_VAE}, Latent Dim: {VAE_LATENT_DIM}, Beta: {VAE_BETA}")
    print(f"   Epochs: {EPOCHS_VAE}, Learning Rate: {VAE_LEARNING_RATE}")
    
    # 1. Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"vae_run_{timestamp}")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # 2. Data Loaders
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE_VAE, 
        normalize_to_minus_one=False
    )
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(test_loader.dataset)}")
    
    # 3. Model, Optimizer, Scheduler
    model = VAE(latent_dim=VAE_LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=VAE_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 4. Fixed Noise for Consistent Sampling
    fixed_noise = torch.randn(36, VAE_LATENT_DIM).to(DEVICE)  # 6x6 grid
    
    # 5. Training Tracking
    best_loss = float('inf')
    
    # 6. Training Loop
    for epoch in range(1, EPOCHS_VAE + 1):
        model.train()
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            x_recon, mu, logvar = model(data)
            
            # Compute Loss
            total_loss, recon_loss, kl_loss = vae_loss_function(
                x_recon, data, mu, logvar, beta=VAE_BETA
            )
            
            # Backward Pass with Gradient Clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate Metrics
            total_loss_epoch += total_loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.2f}",
                'recon': f"{recon_loss.item():.2f}",
                'kl': f"{kl_loss.item():.2f}"
            })
        
        # Calculate average losses (per batch)
        avg_total = total_loss_epoch / len(train_loader)
        avg_recon = recon_loss_epoch / len(train_loader)
        avg_kl = kl_loss_epoch / len(train_loader)
        
        # Validation
        val_loss = validate(model, test_loader)
        
        # Learning Rate Scheduling
        scheduler.step(val_loss)
        
        # Logging
        print(f"\nEpoch {epoch:03d} | "
              f"Train Total: {avg_total:.2f} (Recon: {avg_recon:.2f}, KL: {avg_kl:.2f}) | "
              f"Val: {val_loss:.2f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Generate and Save Samples
        if epoch % 5 == 0 or epoch == 1:  # Save every 5 epochs
            model.eval()
            with torch.no_grad():
                samples = model.decode(fixed_noise)
                sample_filename = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
                save_image_grid(samples, sample_filename, nrow=6)
        
        # Save Checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_total,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"   ✓ New best model! (Val Loss: {val_loss:.2f})")
        
        print("")
    
    print(f"\n✅ Training Complete!")
    print(f"   Results saved to: {run_dir}")
    print(f"   Best validation loss: {best_loss:.2f}")
    
    return model, run_dir

if __name__ == "__main__":
    train_vae()