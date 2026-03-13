# test_ddpm.py
import torch
from src.models import UNet, DDPM
from src.config import get_config

def test_ddpm():
    print("🔬 Testing DDPM implementation...")
    
    # Get config
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create UNet (small version for testing)
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_dim=128,  # Smaller for testing
        base_channels=32,  # Smaller for testing
        channel_mults=(1, 2, 2)
    ).to(device)
    
    # Create DDPM
    ddpm = DDPM(
        unet=unet,
        timesteps=config.DDPM_TIMESTEPS,
        schedule='cosine'
    ).to(device)
    
    print(f"✅ Models created successfully")
    print(f"   UNet params: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"   DDPM timesteps: {ddpm.timesteps}")
    print(f"   Betas shape: {ddpm.betas.shape}")
    print(f"   Betas range: [{ddpm.betas.min():.4f}, {ddpm.betas.max():.4f}]")
    
    # Test forward diffusion
    print("\n🔄 Testing forward diffusion...")
    batch_size = 4
    x_0 = torch.randn(batch_size, 3, 32, 32).to(device)
    t = torch.randint(0, ddpm.timesteps, (batch_size,), device=device)
    
    x_t, noise = ddpm.forward_diffusion(x_0, t)
    print(f"   x_0 shape: {x_0.shape}")
    print(f"   x_t shape: {x_t.shape}")
    print(f"   t: {t.cpu().tolist()}")
    
    # Test denoise
    print("\n🔄 Testing denoise...")
    noise_pred = ddpm.denoise(x_t, t)
    print(f"   noise_pred shape: {noise_pred.shape}")
    
    # Test loss computation
    print("\n📉 Testing loss computation...")
    loss = ddpm.compute_loss(x_0)
    print(f"   Loss: {loss.item():.6f}")
    
    # Test alpha values getter
    print("\n📊 Testing get_alpha_values...")
    alpha_dict = ddpm.get_alpha_values(t[0])
    for k, v in alpha_dict.items():
        print(f"   {k}: {v.item():.6f}")
    
    print("\n✅ All tests passed! DDPM implementation is working.")

if __name__ == "__main__":
    test_ddpm()





def estimate_sampling_speed(model, config):
    """Estimate time required for sampling a single image."""
    print("⏱️  Estimating sampling speed (single forward pass)...")
    
    model.eval()
    device = config.DEVICE
    
    dummy_x = torch.randn(1, 3, 32, 32, device=device)
    dummy_t = torch.randint(0, config.DDPM_TIMESTEPS, (1,), device=device)
    
    for _ in range(10):
        _ = model.denoise(dummy_x, dummy_t)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(50):
        _ = model.denoise(dummy_x, dummy_t)
    torch.cuda.synchronize() if device == 'cuda' else None
    
    time_per_forward = (time.time() - start) / 50
    time_per_sample_full = time_per_forward * config.DDPM_TIMESTEPS
    
    return {
        'time_per_forward': time_per_forward,
        'time_per_sample': time_per_sample_full,
    }