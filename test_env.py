# test_vae.py

import torch
import torch.nn.functional as F
from src.models import VAE, vae_loss_function
from src.config import DEVICE, VAE_LATENT_DIM

def test_vae_architecture():
    print(f"Running on Device: {DEVICE}")
    
    # 1. Initialize Model
    model = VAE(latent_dim=VAE_LATENT_DIM).to(DEVICE)
    print(f"✅ Model Initialized. Latent Dim: {VAE_LATENT_DIM}")
    
    # 2. Create Dummy Input with PROPER normalization [0, 1]
    # Using torch.rand() instead of torch.randn() to match CIFAR-10 data range
    dummy_input = torch.rand(4, 3, 32, 32).to(DEVICE)
    print(f"✅ Input Created. Shape: {dummy_input.shape}, Range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    
    # 3. Forward Pass
    x_recon, mu, logvar = model(dummy_input)
    print(f"✅ Forward Pass Successful. Output Shape: {x_recon.shape}")
    print(f"✅ Output Range: [{x_recon.min():.3f}, {x_recon.max():.3f}] (should be [0, 1] due to sigmoid)")
    
    # 4. Loss with Components
    total_loss, recon_loss, kl_loss = vae_loss_function(
        x_recon, dummy_input, mu, logvar, beta=1.0
    )
    
    # 5. Print Detailed Breakdown
    batch_size = dummy_input.size(0)
    num_pixels = 3 * 32 * 32
    print(f"\n--- Loss Breakdown (Batch={batch_size}) ---")
    print(f"Reconstruction Loss (sum): {recon_loss.item():.2f}")
    print(f"Reconstruction Loss (per image): {recon_loss.item()/batch_size:.2f}")
    print(f"Reconstruction Loss (per pixel): {recon_loss.item()/(batch_size * num_pixels):.4f}")
    print(f"KL Divergence (sum): {kl_loss.item():.2f}")
    print(f"KL Divergence (per image): {kl_loss.item()/batch_size:.2f}")
    print(f"Total Loss (sum): {total_loss.item():.2f}")
    print(f"Total Loss (per image): {total_loss.item()/batch_size:.2f}")
    
    # KL/Recon Ratio
    if recon_loss.item() > 0:
        ratio = kl_loss.item() / recon_loss.item()
        print(f"KL/Recon Ratio: {ratio:.3f}")
    else:
        print("KL/Recon Ratio: N/A (recon_loss = 0)")
    
    # 6. Backprop
    total_loss.backward()
    print(f"\n✅ Backpropagation Successful. Gradients computed.")
    
    # 7. Beta Tuning Guidance
    print(f"\n--- Beta Tuning Guidance ---")
    if recon_loss.item() > 0:
        ratio = kl_loss.item() / recon_loss.item()
        if ratio < 0.1:
            print("⚠️  KL is very small. Consider increasing beta (e.g., 2.0-5.0) or use KL annealing")
        elif ratio > 5.0:
            print("⚠️  KL dominates. Consider decreasing beta (e.g., 0.1-0.5)")
        else:
            print("✅ Loss balance looks reasonable for beta=1.0")
    
    print(f"\n🎉 All VAE Tests Passed! Ready for Training.")
    
    return recon_loss.item(), kl_loss.item()

if __name__ == "__main__":
    recon, kl = test_vae_architecture()