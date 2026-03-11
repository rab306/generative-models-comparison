# scripts/visualize_loss.py
"""
Simple script to visualize VAE training losses.
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
loss_file = os.path.join(latest_run, 'loss_history.json')

if not os.path.exists(loss_file):
    print(f"No loss history found at {loss_file}")
    exit()

# Load data
df = pd.read_csv(loss_file)

# Create figure
plt.figure(figsize=(12, 5))

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

# Save
output_file = os.path.join(latest_run, 'loss_curves.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✅ Loss curves saved to: {output_file}")

plt.show()