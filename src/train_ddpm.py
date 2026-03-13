# src/train_ddpm.py

import torch
import torch.optim as optim
import os
import time
import csv
import json
import math
from datetime import datetime
from tqdm import tqdm

from config import get_config, get_argparser
from utils import get_cifar10_loaders, save_image_grid, set_seed
from models import UNet, DDPM


@torch.no_grad()
def sample_ddpm(model, batch_size=1, device='cuda'):
    """
    DDPM sampling: reverse diffusion process.
    Iteratively denoise from pure noise to image.
    """
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    
    for t in reversed(range(model.timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = model.denoise(x, t_tensor)
        
        alpha_dict = model.get_alpha_values(t)
        alpha = alpha_dict['alpha']
        alpha_bar = alpha_dict['alpha_bar']
        beta = alpha_dict['beta']
        
        coeff = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        x = (1 / torch.sqrt(alpha)) * (x - coeff * noise_pred)
        
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta) * noise
    
    return x


@torch.no_grad()
def sample_ddpm_with_timing(model, batch_size=1, device='cuda'):
    """
    DDPM sampling with detailed timing information.
    """
    timing_info = {
        'total_time': 0,
        'forward_passes': [],
        'denoise_steps': [],
        'timesteps': model.timesteps,
        'avg_forward_pass': 0,
        'avg_denoise_step': 0,
    }
    
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    total_start = time.time()
    
    for step_idx, t in enumerate(reversed(range(model.timesteps))):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        step_start = time.time()
        noise_pred = model.denoise(x, t_tensor)
        forward_time = time.time() - step_start
        timing_info['forward_passes'].append(forward_time)
        
        alpha_dict = model.get_alpha_values(t)
        alpha = alpha_dict['alpha']
        alpha_bar = alpha_dict['alpha_bar']
        beta = alpha_dict['beta']
        
        denoise_start = time.time()
        coeff = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        x = (1 / torch.sqrt(alpha)) * (x - coeff * noise_pred)
        
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta) * noise
        
        denoise_time = time.time() - denoise_start
        timing_info['denoise_steps'].append(denoise_time)
        
        if (step_idx + 1) % 100 == 0:
            avg_forward = sum(timing_info['forward_passes'][-100:]) / 100
            print(f"      Step {step_idx + 1}/{model.timesteps} - Forward: {avg_forward*1000:.2f}ms")
    
    total_time = time.time() - total_start
    timing_info['total_time'] = total_time
    timing_info['avg_forward_pass'] = sum(timing_info['forward_passes']) / len(timing_info['forward_passes'])
    timing_info['avg_denoise_step'] = sum(timing_info['denoise_steps']) / len(timing_info['denoise_steps'])
    
    return x, timing_info


def train_ddpm(config):
    """Train DDPM with given configuration."""
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*70}")
    print(f"🚀 Starting DDPM Training on {config.DEVICE}")
    print(f"{'='*70}")
    print(f"   Batch Size:      {config.BATCH_SIZE_DDPM}")
    print(f"   Timesteps:       {config.DDPM_TIMESTEPS}")
    print(f"   Schedule:        {config.DDPM_BETA_SCHEDULE}")
    print(f"   Epochs:          {config.EPOCHS_DDPM}")
    print(f"   Learning Rate:   {config.DDPM_LEARNING_RATE}")
    print(f"   Num Workers:     {config.NUM_WORKERS}")
    print(f"{'='*70}\n")
    
    # Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.RESULTS_DIR, f"ddpm_run_{timestamp}")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    config.save(os.path.join(run_dir, "config.json"))
    
    # Data Loaders
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config.BATCH_SIZE_DDPM, 
        normalize_to_minus_one=True,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR
    )
    print(f"   Training samples:   {len(train_loader.dataset):,}")
    print(f"   Test samples:       {len(test_loader.dataset):,}\n")
    
    # Create UNet and DDPM
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
    
    optimizer = optim.AdamW(model.parameters(), lr=config.DDPM_LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.EPOCHS_DDPM,
        eta_min=1e-6
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters:   {total_params:,}")
    print(f"   Trainable Params:   {trainable_params:,}\n")
    
    # Training Tracking
    best_train_loss = float('inf')
    train_losses = []
    val_losses = []
    epoch_times = []
    val_times = []
    
    # Speed estimation
    speed_info = estimate_sampling_speed(model, config)
    print(f"\n⏱️  Speed Estimates:")
    print(f"   Forward pass:      {speed_info['time_per_forward']*1000:.2f} ms")
    print(f"   Full sampling:     {speed_info['time_per_sample']:.2f} sec per image")
    
    time_500_images = speed_info['time_per_sample'] * 500
    print(f"   Estimated time for 500 images: {time_500_images/60:.1f} minutes\n")
    
    # Training Loop
    for epoch in range(1, config.EPOCHS_DDPM + 1):
        epoch_start = time.time()
        
        model.train()
        loss_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(config.DEVICE)
            optimizer.zero_grad()
            
            loss = model.compute_loss(data)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_epoch += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average training loss
        avg_train_loss = loss_epoch / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ✅ VALIDATION: Compute loss on entire test set
        model.eval()
        val_loss = 0
        val_start = time.time()
        with torch.no_grad():
            for test_data, _ in test_loader:
                test_data = test_data.to(config.DEVICE)
                val_loss += model.compute_loss(test_data).item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        val_time = time.time() - val_start
        val_times.append(val_time)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Logging (with validation)
        print(f"Epoch {epoch:03d} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"Val Time: {val_time:.1f}s | "
              f"LR: {current_lr:.2e} | "
              f"Total: {epoch_time:.1f}s")
        
        # Generate Sample with timing at specific epochs
        sample_epochs = [20, 40, 60, 80, 100]
        if config.EPOCHS_DDPM < 20:
            sample_epochs = [config.EPOCHS_DDPM]
        
        if epoch in sample_epochs:
            model.eval()
            with torch.no_grad():
                print(f"\n   🖼️  Generating 1 sample with detailed timing...")
                samples, timing_info = sample_ddpm_with_timing(
                    model, 
                    batch_size=1,
                    device=config.DEVICE
                )
                
                print(f"\n   ⏱️  Timing Results for Epoch {epoch}:")
                print(f"      Total sampling time:     {timing_info['total_time']:.2f}s")
                print(f"      Avg forward pass:        {timing_info['avg_forward_pass']*1000:.2f}ms")
                print(f"      Avg denoise step:        {timing_info['avg_denoise_step']*1000:.2f}ms")
                print(f"      Forward passes (1000):   {timing_info['avg_forward_pass'] * 1000:.2f}s")
                
                est_500_images = timing_info['total_time'] * 500
                print(f"      Est. time for 500 images: {est_500_images/60:.1f} minutes")
                print(f"      Est. time for 500 images: {est_500_images/3600:.2f} hours\n")
                
                sample_filename = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
                save_image_grid(samples, sample_filename, nrow=1)
                print(f"   📸 Sample saved: {sample_filename}\n")
        
        # Save checkpoint
        if epoch % config.SAVE_INTERVAL == 0 or epoch == config.EPOCHS_DDPM:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_train_loss:
            best_train_loss = avg_val_loss
            best_path = os.path.join(run_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_path)
            print(f"   ⭐ New best model! (Train Loss: {avg_val_loss:.4f})\n")
    
    # Training complete
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}")
    print(f"   Results saved to:  {run_dir}")
    print(f"   Best Train Loss:   {best_train_loss:.4f}")
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_val_time = sum(val_times) / len(val_times)
    total_train_time = sum(epoch_times)
    
    print(f"   Avg time per epoch: {avg_epoch_time:.1f}s")
    print(f"   Avg validation time: {avg_val_time:.1f}s ({(avg_val_time/avg_epoch_time)*100:.1f}% of epoch)")
    print(f"   Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f} minutes)")
    print(f"{'='*70}\n")
    
    # Save loss history as CSV
    csv_file = os.path.join(run_dir, "loss_history.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'epoch_time_seconds', 'val_time_seconds'])
        for epoch_idx in range(len(train_losses)):
            writer.writerow([
                epoch_idx + 1, 
                train_losses[epoch_idx], 
                val_losses[epoch_idx],
                epoch_times[epoch_idx],
                val_times[epoch_idx]
            ])
    
    # Save loss history as JSON
    json_file = os.path.join(run_dir, "loss_history.json")
    with open(json_file, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'val_times': val_times,
            'total_training_time': sum(epoch_times),
            'avg_epoch_time': sum(epoch_times) / len(epoch_times),
            'avg_val_time': sum(val_times) / len(val_times),
            'val_time_percentage': (sum(val_times) / sum(epoch_times)) * 100,
        }, f, indent=2)
    
    return model, run_dir, train_losses, val_losses, epoch_times, val_times


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    config = get_config(args)
    
    print(f"\n📋 Configuration Summary:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Random Seed: {config.RANDOM_SEED}")
    print(f"   Results Dir: {config.RESULTS_DIR}")
    print("")
    
    model, run_dir, train_losses, val_losses, epoch_times, val_times = train_ddpm(config)
    
    print(f"🎉 DDPM Training Finished Successfully!")