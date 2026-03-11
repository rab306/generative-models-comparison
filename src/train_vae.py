# src/train_vae.py

import torch
import torch.optim as optim
import os
import sys
from datetime import datetime
from tqdm import tqdm

# Import config with argument support
from config import get_config, get_argparser

def train_vae(config):
    """Train VAE with given configuration"""
    print(f"🚀 Starting VAE Training on {config.DEVICE}")
    print(f"   Batch Size: {config.BATCH_SIZE_VAE}, Latent Dim: {config.VAE_LATENT_DIM}, Beta: {config.VAE_BETA}")
    print(f"   Epochs: {config.EPOCHS_VAE}, Learning Rate: {config.VAE_LEARNING_RATE}")
    
    # Import here to avoid circular imports
    from models import VAE, vae_loss_function
    from utils import get_cifar10_loaders, save_image_grid
    
    # 1. Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.RESULTS_DIR, f"vae_run_{timestamp}")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save config to run directory
    config.save(os.path.join(run_dir, "config.json"))
    
    # 2. Data Loaders
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config.BATCH_SIZE_VAE, 
        normalize_to_minus_one=False
    )
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(test_loader.dataset)}")
    
    # 3. Model, Optimizer
    model = VAE(latent_dim=config.VAE_LATENT_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.VAE_LEARNING_RATE)
    
    # 4. Fixed Noise for Consistent Sampling
    fixed_noise = torch.randn(config.NUM_VISUALIZE_SAMPLES, config.VAE_LATENT_DIM).to(config.DEVICE)
    
    # 5. Training Tracking
    best_loss = float('inf')
    
    # 6. Training Loop
    for epoch in range(1, config.EPOCHS_VAE + 1):
        model.train()
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            x_recon, mu, logvar = model(data)
            
            # Compute Loss
            total_loss, recon_loss, kl_loss = vae_loss_function(
                x_recon, data, mu, logvar, beta=config.VAE_BETA
            )
            
            # Backward Pass
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
        
        # Calculate average losses
        avg_total = total_loss_epoch / len(train_loader)
        avg_recon = recon_loss_epoch / len(train_loader)
        avg_kl = kl_loss_epoch / len(train_loader)
        
        # Simple validation (first batch of test set)
        model.eval()
        with torch.no_grad():
            test_batch, _ = next(iter(test_loader))
            test_batch = test_batch.to(config.DEVICE)
            x_recon, mu, logvar = model(test_batch)
            val_loss, _, _ = vae_loss_function(x_recon, test_batch, mu, logvar, beta=config.VAE_BETA)
            val_loss = val_loss.item() / len(test_batch)  # Per-sample loss
        
        # Logging
        print(f"\nEpoch {epoch:03d} | "
              f"Train Total: {avg_total:.2f} (Recon: {avg_recon:.2f}, KL: {avg_kl:.2f}) | "
              f"Val: {val_loss:.2f}")
        
        # Generate and Save Samples
        if epoch % config.SAMPLE_INTERVAL == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                samples = model.decode(fixed_noise)
                sample_filename = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
                save_image_grid(samples, sample_filename, nrow=int(config.NUM_VISUALIZE_SAMPLES**0.5))
        
        # Save Checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total,
                'val_loss': val_loss,
                'config': config.to_dict(),
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
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args()
    
    # Get configuration with overrides
    config = get_config(args)
    
    # Run training
    train_vae(config)