"""
Universal loss visualization script for VAE and DDPM training.
Shows plots in console and saves them.

Usage:
    python visualize_losses.py --model vae
    python visualize_losses.py --model ddpm
    python visualize_losses.py --model both
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse


def find_latest_run(model_type):
    """Find latest run directory for given model type."""
    pattern = f'results/{model_type}_run_*'
    runs = glob.glob(pattern)
    if not runs:
        print(f"❌ No {model_type.upper()} training runs found in results/")
        return None
    return max(runs, key=os.path.getctime)


def visualize_vae(run_dir):
    """Visualize VAE training losses."""
    loss_file = os.path.join(run_dir, 'loss_history.csv')
    
    if not os.path.exists(loss_file):
        print(f"❌ No loss history found at {loss_file}")
        return
    
    # Load data
    df = pd.read_csv(loss_file)
    print(f"\n📊 VAE Loss History")
    print(f"   Epochs: {len(df)}")
    print(f"   Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"   Final Val Loss: {df['val_loss'].iloc[-1]:.4f}")
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('VAE Training Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Train vs Val Loss
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Components (if available)
    ax2 = plt.subplot(1, 3, 2)
    if 'recon_loss' in df.columns and 'kl_loss' in df.columns:
        ax2.plot(df['epoch'], df['recon_loss'], 'g-', label='Reconstruction', linewidth=2, marker='o', markersize=3)
        ax2.plot(df['epoch'], df['kl_loss'], 'orange', label='KL Divergence', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Print KL ratio
        kl_ratio = df['kl_loss'].iloc[-1] / df['recon_loss'].iloc[-1]
        print(f"   Final KL/Recon Ratio: {kl_ratio:.4f}")
    else:
        ax2.text(0.5, 0.5, 'Loss components\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    
    # Plot 3: Loss Difference (Val - Train)
    ax3 = plt.subplot(1, 3, 3)
    loss_gap = df['val_loss'] - df['train_loss']
    ax3.fill_between(df['epoch'], 0, loss_gap, alpha=0.3, label='Val-Train Gap')
    ax3.plot(df['epoch'], loss_gap, 'purple', linewidth=2, marker='o', markersize=3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.set_title('Validation-Training Gap (Overfitting)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(run_dir, 'loss_curves.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ VAE plot saved to: {output_file}\n")


def visualize_ddpm(run_dir):
    """Visualize DDPM training losses."""
    loss_file = os.path.join(run_dir, 'loss_history.csv')
    
    if not os.path.exists(loss_file):
        print(f"❌ No loss history found at {loss_file}")
        return
    
    # Load data
    df = pd.read_csv(loss_file)
    print(f"\n📊 DDPM Loss History")
    print(f"   Epochs: {len(df)}")
    print(f"   Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"   Final Val Loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"   Avg Epoch Time: {df['epoch_time_seconds'].mean():.1f}s")
    print(f"   Avg Val Time: {df['val_time_seconds'].mean():.1f}s ({(df['val_time_seconds'].mean()/df['epoch_time_seconds'].mean())*100:.1f}% of epoch)")
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('DDPM Training Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Train vs Val Loss
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Difference (Overfitting indicator)
    ax2 = plt.subplot(1, 3, 2)
    loss_gap = df['val_loss'] - df['train_loss']
    ax2.fill_between(df['epoch'], 0, loss_gap, alpha=0.3, label='Val-Train Gap')
    ax2.plot(df['epoch'], loss_gap, 'purple', linewidth=2, marker='o', markersize=3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Difference')
    ax2.set_title('Validation-Training Gap (Overfitting)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epoch Time
    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(df['epoch'], df['epoch_time_seconds'], alpha=0.7, label='Total Epoch Time', color='skyblue')
    ax3.plot(df['epoch'], df['val_time_seconds'], 'r-', linewidth=2, marker='o', markersize=3, label='Validation Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time per Epoch')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(run_dir, 'loss_curves.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ DDPM plot saved to: {output_file}\n")


def visualize_both(vae_run, ddpm_run):
    """Visualize both VAE and DDPM for comparison."""
    vae_loss_file = os.path.join(vae_run, 'loss_history.csv')
    ddpm_loss_file = os.path.join(ddpm_run, 'loss_history.csv')
    
    if not os.path.exists(vae_loss_file) or not os.path.exists(ddpm_loss_file):
        print("❌ Could not find loss history for both models")
        return
    
    vae_df = pd.read_csv(vae_loss_file)
    ddpm_df = pd.read_csv(ddpm_loss_file)
    
    print(f"\n📊 VAE vs DDPM Comparison")
    print(f"   VAE Final Val Loss: {vae_df['val_loss'].iloc[-1]:.4f}")
    print(f"   DDPM Final Val Loss: {ddpm_df['val_loss'].iloc[-1]:.4f}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('VAE vs DDPM Training Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Final losses comparison
    ax1 = plt.subplot(1, 3, 1)
    models = ['VAE', 'DDPM']
    train_losses = [vae_df['train_loss'].iloc[-1], ddpm_df['train_loss'].iloc[-1]]
    val_losses = [vae_df['val_loss'].iloc[-1], ddpm_df['val_loss'].iloc[-1]]
    
    x = range(len(models))
    width = 0.35
    ax1.bar([i - width/2 for i in x], train_losses, width, label='Train', alpha=0.8)
    ax1.bar([i + width/2 for i in x], val_losses, width, label='Validation', alpha=0.8)
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Final Loss Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: VAE convergence
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(vae_df['epoch'], vae_df['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    ax2.plot(vae_df['epoch'], vae_df['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('VAE Training Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: DDPM convergence
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(ddpm_df['epoch'], ddpm_df['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    ax3.plot(ddpm_df['epoch'], ddpm_df['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('DDPM Training Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save in results directory
    output_file = os.path.join('results', 'vae_vs_ddpm_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training losses for VAE and/or DDPM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_losses.py --model vae
  python visualize_losses.py --model ddpm
  python visualize_losses.py --model both
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['vae', 'ddpm', 'both'],
        default='both',
        help='Which model to visualize (default: both)'
    )
    
    args = parser.parse_args()
    
    if args.model in ['vae', 'both']:
        vae_run = find_latest_run('vae')
        if vae_run:
            visualize_vae(vae_run)
    
    if args.model in ['ddpm', 'both']:
        ddpm_run = find_latest_run('ddpm')
        if ddpm_run:
            visualize_ddpm(ddpm_run)
    
    if args.model == 'both':
        vae_run = find_latest_run('vae')
        ddpm_run = find_latest_run('ddpm')
        if vae_run and ddpm_run:
            visualize_both(vae_run, ddpm_run)
    
    plt.show()


if __name__ == "__main__":
    main()