# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import get_config

# Get default config values
_config = get_config()
VAE_LATENT_DIM = _config.VAE_LATENT_DIM
DEVICE = _config.DEVICE

class VAE(nn.Module):
    def __init__(self, latent_dim=VAE_LATENT_DIM):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Bottleneck
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
    
    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = h.view(-1, 256, 4, 4)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        return torch.sigmoid(self.dec_conv3(h))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss_function(x_recon, x, mu, logvar, beta):
    """
    Computes VAE loss with separate components for monitoring.
    Returns: total_loss, recon_loss, kl_loss
    """
    batch_size = x.size(0)
    
    # Reconstruction Loss (MSE) - normalized by batch
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
    
    # KL Divergence - normalized by batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embeddings as proposed in Transformer.
    Takes timestep t and converts to embedding vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        # t shape: (batch_size,)
        half_dim = self.dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -embeddings)
        
        # Shape: (batch_size, half_dim)
        embeddings = t[:, None].float() * embeddings[None, :]
        
        # Concatenate sin and cos
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings  # Shape: (batch_size, dim)


class ResNetBlock(nn.Module):
    """
    ResNet block with GroupNorm, Conv, Time embedding projection, SiLU.
    """
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        # First norm + conv
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        # Second norm + conv
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection if dimensions change
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb):
        # x shape: (batch, in_channels, h, w)
        # t_emb shape: (batch, time_dim)
        
        # First block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        t = self.time_mlp(F.silu(t_emb))
        t = t[:, :, None, None]  # Shape: (batch, out_channels, 1, 1)
        h = h + t
        
        # Second block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block with GroupNorm.
    """
    def __init__(self, channels, num_groups=32):
        super().__init__()
        # Ensure num_groups doesn't exceed channels
        num_groups = min(num_groups, channels)
        self.norm = nn.GroupNorm(num_groups, channels)
        
        # Q, K, V projections
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        
        # Output projection
        self.proj = nn.Linear(channels, channels)
        
    def forward(self, x):
        # x shape: (batch, channels, h, w)
        batch, channels, h, w = x.shape
        
        # Normalize
        h_norm = self.norm(x)
        
        # Reshape to (batch, h*w, channels)
        h_flat = h_norm.view(batch, channels, -1).permute(0, 2, 1)
        
        # Compute Q, K, V
        Q = self.q(h_flat)
        K = self.k(h_flat)
        V = self.v(h_flat)
        
        # Attention scores with proper scaling
        attn_weights = torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(channels)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention
        attn_out = torch.bmm(attn_weights, V)
        
        # Project and reshape back
        attn_out = self.proj(attn_out)
        attn_out = attn_out.permute(0, 2, 1).view(batch, channels, h, w)
        
        # Skip connection
        return x + attn_out


class DownBlock(nn.Module):
    """
    Downsampling block: ResNet + Attention + Downsample
    Saves skip connection BEFORE downsampling
    """
    def __init__(self, in_channels, out_channels, time_dim, has_attention=False):
        super().__init__()
        
        self.resnet = ResNetBlock(in_channels, out_channels, time_dim)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()
        
        # Downsampling (stride=2 convolution)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x, t_emb):
        h = self.resnet(x, t_emb)
        h = self.attention(h)
        
        # CRITICAL: Save skip BEFORE downsampling
        skip = h
        
        # Then downsample
        h = self.downsample(h)
        
        return h, skip


class UpBlock(nn.Module):
    """
    Upsampling block: Upsample + Concatenate Skip + ResNet + Attention
    """
    def __init__(self, in_channels, out_channels, time_dim, has_attention=False):
        super().__init__()
        
        # Upsampling (transposed convolution)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        
        # ResNet takes concatenated input (upsample output + skip)
        self.resnet = ResNetBlock(out_channels * 2, out_channels, time_dim)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()
        
    def forward(self, x, t_emb, skip):
        # x shape: (batch, in_channels, h, w)
        # skip shape: (batch, out_channels, 2h, 2w) from corresponding down block
        
        h = self.upsample(x)
        
        # Concatenate with skip connection
        h = torch.cat([h, skip], dim=1)
        
        h = self.resnet(h, t_emb)
        h = self.attention(h)
        return h


class UNet(nn.Module):
    """
    Complete UNet for DDPM following the book structure:
    - Conv_in
    - Time embedding
    - Down blocks (with optional attention)
    - Middle block
    - Up blocks (with optional attention)
    - Conv_out
    """
    def __init__(self, 
                 in_channels=3,
                 out_channels=3,
                 time_dim=256,
                 base_channels=64,
                 channel_mults=(1, 2, 2, 2)):
        super().__init__()
        
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Calculate channels at each level
        channels = [base_channels * mult for mult in channel_mults]
        
        # Down blocks
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            # Add attention at deeper levels (last 2 down blocks)
            has_attention = i >= len(channels) - 2
            self.downs.append(DownBlock(in_ch, out_ch, time_dim, has_attention))
            in_ch = out_ch
        
        # Middle block (with attention)
        self.middle_resnet1 = ResNetBlock(in_ch, in_ch, time_dim)
        self.middle_attention = AttentionBlock(in_ch)
        self.middle_resnet2 = ResNetBlock(in_ch, in_ch, time_dim)
        
        # Up blocks
        self.ups = nn.ModuleList()
        for i, out_ch in enumerate(reversed(channels)):
            # Add attention at deeper levels (first 2 up blocks get attention)
            has_attention = i < 2
            self.ups.append(UpBlock(in_ch, out_ch, time_dim, has_attention))
            in_ch = out_ch
        
        # Output convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x, t):
        # x shape: (batch, in_channels, h, w)
        # t shape: (batch,) - timesteps
        
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Store skip connections for upsampling
        skips = []
        
        # Down blocks - CORRECTED: get both h and skip
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)
        
        # Middle block
        h = self.middle_resnet1(h, t_emb)
        h = self.middle_attention(h)
        h = self.middle_resnet2(h, t_emb)
        
        # Up blocks (reverse order of skips) - CORRECTED: use skip from list
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, t_emb, skip)
        
        # Output
        h = self.conv_out(h)
        return h
    

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model with cosine noise schedule.
    
    This class handles:
    - Noise schedule (cosine annealing)
    - Forward diffusion (add noise)
    - Noise prediction (UNet forward pass)
    - Training loss computation
    """
    def __init__(self, 
                 unet,
                 timesteps=1000,
                 schedule='cosine',
                 beta_start=1e-4,
                 beta_end=0.02):
        super().__init__()
        
        self.unet = unet
        self.timesteps = timesteps
        
        # Create noise schedule
        if schedule == 'cosine':
            # Cosine annealing schedule from Improved DDPM paper
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float32) / timesteps
            alpha_bar = torch.cos(((steps + s) / (1 + s)) * (torch.pi / 2)) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # Normalize to alpha_bar[0] = 1
            betas = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 0, 0.999)
        else:
            # Linear schedule (fallback)
            betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # Register as buffers for proper device handling
        self.register_buffer('betas', betas)
        
        # Pre-calculate useful quantities for efficiency
        alphas = 1 - betas
        self.register_buffer('alphas', alphas)
        
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('alpha_bars', alpha_bars)
        
        # For forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1 - alpha_bars))
        
        # For reverse diffusion: useful pre-calculated values
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        
    def forward_diffusion(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Adds noise to clean image according to noise schedule.
        
        Args:
            x_0: Clean image (batch, channels, h, w) in range [-1, 1]
            t: Timesteps (batch,) - indices from 0 to timesteps-1
            noise: Optional pre-generated noise (batch, channels, h, w)
                   If None, will generate random noise
        
        Returns:
            x_t: Noisy image (batch, channels, h, w)
            noise: The noise that was added (for training loss)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get alpha values for each timestep in batch
        # sqrt_alpha_bars[t] shape: (batch,) -> reshape to (batch, 1, 1, 1) for broadcasting
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        
        # Diffusion equation: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def denoise(self, x_t, t):
        """
        Predict noise from noisy image using UNet.
        
        Args:
            x_t: Noisy image (batch, channels, h, w)
            t: Timesteps (batch,)
        
        Returns:
            noise_pred: Predicted noise (batch, channels, h, w)
        """
        return self.unet(x_t, t)
    
    def compute_loss(self, x_0):
        """
        Training loss: MSE between predicted noise and actual noise.
        This is the main loss function for DDPM training.
        
        Args:
            x_0: Clean image batch (batch, channels, h, w)
        
        Returns:
            loss: Scalar MSE loss
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps uniformly for each image in batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Generate random noise
        noise = torch.randn_like(x_0)
        
        # Add noise to create x_t
        x_t, _ = self.forward_diffusion(x_0, t, noise)
        
        # Predict noise using UNet
        noise_pred = self.denoise(x_t, t)
        
        # MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def get_alpha_values(self, t):
        """
        Get pre-calculated alpha values for a given timestep.
        Useful for sampling in training script.
        
        Args:
            t: Timestep (scalar or tensor)
        
        Returns:
            Dict with alpha, alpha_bar, beta, sqrt_alpha, sqrt_one_minus_alpha_bar
        """
        return {
            'alpha': self.alphas[t],
            'alpha_bar': self.alpha_bars[t],
            'beta': self.betas[t],
            'sqrt_alpha': self.sqrt_alphas[t],
            'sqrt_one_minus_alpha_bar': self.sqrt_one_minus_alpha_bars[t],
        }
    