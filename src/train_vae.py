# src/train_vae.py

import torch
import torch.optim as optim
import os
from datetime import datetime
from tqdm import tqdm

# ✅ Use relative imports within the src package
from .config import get_config, get_argparser
from .utils import get_cifar10_loaders, save_image_grid, set_seed


def train_vae(config):
    """Train VAE with given configuration."""
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
    
    # ✅ Import models with relative import
    from .models import VAE, vae_loss_function
    
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
    fixed_noise = torch.randn(config.NUM_VISUALIZE_SAMPLES, config.VAE_LATENT_DIM).to(config.DEVICE)
    
    # 5. Training Tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 6. Training Loop
    for epoch in range(1, config.EPOCHS_VAE + 1):
        model.train()
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(config.DEVICE)
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(data)
            
            total_loss, recon_loss, kl_loss = vae_loss_function(
                x_recon, data, mu, logvar, beta=config.VAE_BETA
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss_epoch += total_loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()
            
            progress_bar.set_postfix({
                'loss': f"{total_loss.item()/len(data):.3f}",
                'recon': f"{recon_loss.item()/len(data):.3f}",
                'kl': f"{kl_loss.item()/len(data):.3f}"
            })
        
       # Calculate average losses (per sample, averaged over batches)
        avg_total = total_loss_epoch / len(train_loader)  
        avg_recon = recon_loss_epoch / len(train_loader)
        avg_kl = kl_loss_epoch / len(train_loader)
        
        # Validation
        model.eval()
        val_total = 0
        with torch.no_grad():
            for test_data, _ in test_loader:
                test_data = test_data.to(config.DEVICE)
                x_recon, mu, logvar = model(test_data)
                v_loss, _, _ = vae_loss_function(x_recon, test_data, mu, logvar, beta=config.VAE_BETA)
                val_total += v_loss.item()
        
        val_loss_avg = val_total / len(test_loader.dataset)
        train_losses.append(avg_total)
        val_losses.append(val_loss_avg)
        
        # Logging
        kl_ratio = avg_kl / avg_recon if avg_recon > 0 else 0
        print(f"\nEpoch {epoch:03d} | "
              f"Train: {avg_total:.3f} (Recon: {avg_recon:.3f}, KL: {avg_kl:.3f}, Ratio: {kl_ratio:.3f}) | "
              f"Val: {val_loss_avg:.3f}")
        
        # Generate Samples
        if epoch % config.SAMPLE_INTERVAL == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                samples = model.decode(fixed_noise)
                sample_filename = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
                save_image_grid(samples, sample_filename, nrow=int(config.NUM_VISUALIZE_SAMPLES**0.5))
                print(f"   📸 Samples saved to: {sample_filename}")
        
        # Save Checkpoint
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
            
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_path = os.path.join(run_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss_avg,
                }, best_path)
                print(f"   ⭐ New best model! (Val Loss: {val_loss_avg:.3f})")
        
        print("")
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"   Results saved to:  {run_dir}")
    print(f"   Best Val Loss:     {best_val_loss:.3f}")
    print(f"{'='*60}\n")
    
    # Save loss history
    import json
    with open(os.path.join(run_dir, "loss_history.json"), 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f, indent=2)
    
    return model, run_dir, train_losses, val_losses


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    config = get_config(args)
    
    print(f"\n📋 Configuration Summary:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Random Seed: {config.RANDOM_SEED}")
    print(f"   Results Dir: {config.RESULTS_DIR}")
    print("")
    
    model, run_dir, train_losses, val_losses = train_vae(config)
    
    print(f"🎉 VAE Training Finished Successfully!")