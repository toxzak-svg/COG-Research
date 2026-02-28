"""
Training Script for Image-Based VAE on Website Aesthetics

Uses the Beta-TC-VAE objective with aggressive parameters
to prevent posterior collapse.

Based on lessons learned from structural VAE training:
- Use aggressive β-TC (10→25) to prevent collapse
- Use free-bits regularization (1.0 nats)
- Train longer with patience
- Disable contrastive loss (ineffective with limited component diversity)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webpage_models.image_vae import ImageWebpageVAE
from webpage_models.vae_world_model import BetaTCVAELoss
from webpage_data_collection.aesthetics_loader import create_dataloaders


def train_epoch(
    model: ImageWebpageVAE,
    train_loader,
    loss_fn: BetaTCVAELoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    recon_loss = 0
    tc_loss = 0
    outcome_loss = 0
    
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, scores, metadata in pbar:
        images = images.to(device)
        scores = scores.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(images, return_latent=True)
        
        # Compute loss
        total_loss_tensor, loss_dict = loss_fn(
            model=model,
            x=images,
            x_recon=output['recon'],
            outcomes_pred=output['outcome'],
            outcomes_true=scores,
            mu=output['mu'],
            logvar=output['logvar'],
            z=output['z'],
        )
        
        # Backward pass
        total_loss_tensor.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += total_loss_tensor.item()
        recon_loss += loss_dict['reconstruction']
        tc_loss += loss_dict['total_correlation']
        outcome_loss += loss_dict['outcome_prediction']
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss_dict['total'],
            'recon': loss_dict['reconstruction'],
            'tc': loss_dict['total_correlation'],
        })
    
    return {
        'total': total_loss / n_batches,
        'reconstruction': recon_loss / n_batches,
        'total_correlation': tc_loss / n_batches,
        'outcome_prediction': outcome_loss / n_batches,
    }


@torch.no_grad()
def validate_epoch(
    model: ImageWebpageVAE,
    val_loader,
    loss_fn: BetaTCVAELoss,
    device: str,
) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    recon_loss = 0
    tc_loss = 0
    outcome_loss = 0
    
    n_batches = 0
    
    for images, scores, metadata in val_loader:
        images = images.to(device)
        scores = scores.to(device)
        
        # Forward pass
        output = model(images, return_latent=True)
        
        # Compute loss
        total_loss_tensor, loss_dict = loss_fn(
            model=model,
            x=images,
            x_recon=output['recon'],
            outcomes_pred=output['outcome'],
            outcomes_true=scores,
            mu=output['mu'],
            logvar=output['logvar'],
            z=output['z'],
        )
        
        # Accumulate metrics
        total_loss += total_loss_tensor.item()
        recon_loss += loss_dict['reconstruction']
        tc_loss += loss_dict['total_correlation']
        outcome_loss += loss_dict['outcome_prediction']
        n_batches += 1
    
    return {
        'total': total_loss / n_batches,
        'reconstruction': recon_loss / n_batches,
        'total_correlation': tc_loss / n_batches,
        'outcome_prediction': outcome_loss / n_batches,
    }


def train(args):
    """Main training loop"""
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        comparison_dataset_path=args.comparison_dataset,
        rating_dataset_path=args.rating_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating model with latent_dim={args.latent_dim}...")
    model = ImageWebpageVAE(
        input_channels=3,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        dropout=args.dropout,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function with β-TC-VAE
    print(f"\nConfiguring Beta-TC-VAE loss...")
    print(f"  beta_tc: {args.beta_tc} -> {args.beta_tc_max} (annealed over {args.beta_tc_anneal_epochs} epochs)")
    print(f"  free_bits: {args.free_bits} nats")
    print(f"  beta_contrast: {args.beta_contrast}")
    
    loss_fn = BetaTCVAELoss(
        beta_recon=1.0,
        beta_index=0.0,  # Not using index-code MI for images
        beta_tc=args.beta_tc,
        beta_dwkl=1.0,
        beta_outcome=1.0,
        beta_contrast=args.beta_contrast,
        beta_consistency=0.0,  # No consistency loss for images
        free_bits=args.free_bits,
        feature_masks=False,  # Explicitly disable feature masks for images
    )
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        
        # Update β_tc annealing
        current_beta_tc = min(
            args.beta_tc + (args.beta_tc_max - args.beta_tc) * epoch / args.beta_tc_anneal_epochs,
            args.beta_tc_max
        )
        loss_fn.beta_tc = current_beta_tc
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)
        
        # Update scheduler
        scheduler.step(val_metrics['total'])
        
        # Log
        print(f"\nEpoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_metrics['total']:.4f} | "
              f"Val Loss: {val_metrics['total']:.4f} | "
              f"beta_tc: {current_beta_tc:.3f}")
        print(f"  Train - Recon: {train_metrics['reconstruction']:.6f}, "
              f"TC: {train_metrics['total_correlation']:.6f}, "
              f"Outcome: {train_metrics['outcome_prediction']:.6f}")
        print(f"  Val   - Recon: {val_metrics['reconstruction']:.6f}, "
              f"TC: {val_metrics['total_correlation']:.6f}, "
              f"Outcome: {val_metrics['outcome_prediction']:.6f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'beta_tc': current_beta_tc,
        })
        
        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total'],
                'latent_dim': args.latent_dim,
                'image_size': args.image_size,
            }
            
            torch.save(checkpoint, output_dir / f'vae_best_epoch_{epoch}.pt')
            print(f"  [OK] Saved best checkpoint (val_loss: {val_metrics['total']:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"(no improvement for {args.early_stopping_patience} epochs)")
            break
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['total'],
        'latent_dim': args.latent_dim,
        'image_size': args.image_size,
    }
    torch.save(final_checkpoint, output_dir / 'vae_final.pt')
    print(f"[OK] Saved final checkpoint: {output_dir / 'vae_final.pt'}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history: {history_path}")
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {val_metrics['total']:.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] Training Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train Image-Based VAE for Website Aesthetics')
    
    # Data paths
    parser.add_argument(
        '--comparison-dataset',
        type=str,
        default='website-aesthetics-datasets/comparison-based-dataset',
        help='Path to comparison-based dataset'
    )
    parser.add_argument(
        '--rating-dataset',
        type=str,
        default='website-aesthetics-datasets/rating-based-dataset',
        help='Path to rating-based dataset'
    )
    
    # Model hyperparameters
    parser.add_argument('--latent-dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--image-size', type=int, default=256, help='Image size (square)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    
    # Loss hyperparameters (aggressive β-TC from Strategy 1a)
    parser.add_argument('--beta-tc', type=float, default=10.0, help='Initial beta for TC')
    parser.add_argument('--beta-tc-max', type=float, default=25.0, help='Max beta for TC')
    parser.add_argument('--beta-tc-anneal-epochs', type=int, default=60, help='Epochs to anneal beta_tc')
    parser.add_argument('--free-bits', type=float, default=1.0, help='Free bits per dimension')
    parser.add_argument('--beta-contrast', type=float, default=0.0, help='Beta for contrastive loss')
    
    # Training config
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='results/image_vae_aesthetics', help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
