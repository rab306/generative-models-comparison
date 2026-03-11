# scripts/visualize_loss.py
"""
Simple script to visualize VAE training losses.
Shows plots in console and saves them.
Run after training: python scripts/visualize_loss.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find latest run
runs = glob.glob('results/vae_run_*')
if not runs:
    print("No training runs found in results/")
    exit()

latest_run = max(runs, key=os.path.getctime)
loss_file = os.path.join(latest_run, 'loss_history.csv')

if not os.path.exists(loss_file):
    print(f"No loss history found at {loss_file}")
    exit()

# Load data
df = pd.read_csv(loss_file)
print(f"\n📊 Loaded loss history from: {loss_file}")
print(f"   Epochs: {len(df)}")
print(f"   Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")
print(f"   Final Val Loss: {df['val_loss'].iloc[-1]:.4f}")

# Create figure for DISPLAY
plt.figure(figsize=(14, 6))

# Plot 1: Train vs Val Loss
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], 'b-', label='Train', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], 'r-', label='Validation', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss Components
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['recon_loss'], 'g-', label='Reconstruction', linewidth=2)
plt.plot(df['epoch'], df['kl_loss'], 'orange', label='KL Divergence', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Components')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# SAVE the plot
output_file = os.path.join(latest_run, 'loss_curves.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_file}")

plt.show()

# Also print final ratio
kl_ratio = df['kl_loss'].iloc[-1] / df['recon_loss'].iloc[-1]
print(f"   Final KL/Recon Ratio: {kl_ratio:.4f}")