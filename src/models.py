# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Use relative import within the src package
from .config import VAE_LATENT_DIM, DEVICE

class VAE(nn.Module):
    def __init__(self, latent_dim=VAE_LATENT_DIM):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)   # 64 x 16 x 16
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 128 x 8 x 8
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)# 256 x 4 x 4
        
        # Bottleneck
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # --- Decoder ---
        self.dec_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 128 x 8 x 8
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 64 x 16 x 16
        self.dec_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)    # 3 x 32 x 32

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, 256, 4, 4)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        return torch.sigmoid(self.dec_conv3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss_function(x_recon, x, mu, logvar, beta=1.0):
    """
    Computes VAE loss with separate components for monitoring.
    Returns: total_loss, recon_loss, kl_loss
    """
    batch_size = x.size(0)
    
    # Reconstruction Loss (MSE) - normalized by batch
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
    
    # KL Divergence - normalized by batch
    # Formula: -0.5 * sum(1 + log(var) - mu^2 - var)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss