"""
Training Script: Flask App World Model

Trains the VAE world-model on Flask app evolution data to predict
test outcomes from schema changes.

Usage:
    python train_app_world_model.py --data_dir flask_app_data/collected --epochs 100
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# TensorBoard - optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask_app_data.dataset.dataset_builder import create_flask_dataloaders
from flask_app_data.models.app_world_model import create_app_world_model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    kl_weight: float = 0.1,
) -> dict:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_outcome_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        # Move to device
        state_before = batch['state_before'].to(device)
        edit = batch['edit'].to(device)
        state_after = batch['state_after'].to(device)
        outcome = batch['outcome'].to(device)
        
        # Forward pass
        result = model(state_before, edit, state_after, outcome)
        
        # Backward pass
        optimizer.zero_grad()
        result['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += result['total_loss'].item()
        total_recon_loss += result['recon_loss'].item()
        total_kl_loss += result['kl_loss'].item()
        total_outcome_loss += result['outcome_loss'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
        'outcome_loss': total_outcome_loss / n_batches,
    }


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> dict:
    """Evaluate the model"""
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_outcome_loss = 0.0
    n_batches = 0
    
    # For outcome prediction accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            state_before = batch['state_before'].to(device)
            edit = batch['edit'].to(device)
            state_after = batch['state_after'].to(device)
            outcome = batch['outcome'].to(device)
            
            result = model(state_before, edit, state_after, outcome)
            
            total_loss += result['total_loss'].item()
            total_recon_loss += result['recon_loss'].item()
            total_kl_loss += result['kl_loss'].item()
            total_outcome_loss += result['outcome_loss'].item()
            n_batches += 1
            
            # Outcome accuracy
            pred_probs = result['outcome_prob'].squeeze()
            pred_labels = (pred_probs > 0.5).float()
            correct += (pred_labels == outcome).sum().item()
            total += outcome.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
        'outcome_loss': total_outcome_loss / n_batches,
        'outcome_accuracy': accuracy,
    }


def predict_outcomes(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> dict:
    """
    Run prediction on test set and return results
    
    Returns:
        Dictionary with predictions and ground truth
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            state_before = batch['state_before'].to(device)
            edit = batch['edit'].to(device)
            outcome = batch['outcome'].to(device)
            
            # Predict
            result = model.predict(state_before, edit)
            
            probs = result['test_pass_probability'].squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(float)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(outcome.cpu().numpy().tolist())
    
    # Compute metrics
    correct = sum([1 for p, t in zip(all_preds, all_targets) if p == t])
    accuracy = correct / len(all_preds) if all_preds else 0.0
    
    # True positives, false positives, etc.
    tp = sum([1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 1])
    fp = sum([1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 0])
    tn = sum([1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 0])
    fn = sum([1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 1])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
    }


def main(args):
    """Main training function"""
    # Setup
    device = args.device
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_flask_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        state_dim=args.state_dim,
        edit_dim=args.edit_dim,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_app_world_model(
        state_dim=args.state_dim,
        latent_dim=args.latent_dim,
        edit_dim=args.edit_dim,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )
    
    # Tensorboard (optional)
    if HAS_TENSORBOARD:
        writer = SummaryWriter(output_dir / 'logs')
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, args.kl_weight
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train loss: {train_metrics['loss']:.4f} | "
              f"recon: {train_metrics['recon_loss']:.4f} | "
              f"kl: {train_metrics['kl_loss']:.4f} | "
              f"outcome: {train_metrics['outcome_loss']:.4f}")
        print(f"  Val loss:   {val_metrics['loss']:.4f} | "
              f"recon: {val_metrics['recon_loss']:.4f} | "
              f"kl: {val_metrics['kl_loss']:.4f} | "
              f"outcome: {val_metrics['outcome_loss']:.4f} | "
              f"acc: {val_metrics['outcome_accuracy']:.2%}")
        
        # Tensorboard (optional)
        if HAS_TENSORBOARD:
            for key, value in {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}.items():
                writer.add_scalar(key, value, epoch)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['outcome_accuracy'],
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        print()
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = predict_outcomes(model, test_loader, device)
    
    print(f"Test Accuracy: {test_results['accuracy']:.2%}")
    print(f"Test Precision: {test_results['precision']:.2%}")
    print(f"Test Recall: {test_results['recall']:.2%}")
    print(f"Test F1: {test_results['f1']:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {test_results['confusion_matrix']['tp']}")
    print(f"  FP: {test_results['confusion_matrix']['fp']}")
    print(f"  TN: {test_results['confusion_matrix']['tn']}")
    print(f"  FN: {test_results['confusion_matrix']['fn']}")
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1': test_results['f1'],
            'confusion_matrix': test_results['confusion_matrix'],
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Target check
    target_accuracy = 0.75
    if test_results['accuracy'] >= target_accuracy:
        print(f"\n🎉 Target achieved! {test_results['accuracy']:.0%} >= {target_accuracy:.0%}")
    else:
        print(f"\n⚠️ Target not met: {test_results['accuracy']:.0%} < {target_accuracy:.0%}")
        print("   Need more training data or better model architecture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flask App World Model")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='flask_app_data/collected',
                        help='Directory with collected app data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Model
    parser.add_argument('--state_dim', type=int, default=170,
                        help='Input state dimension')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent state dimension')
    parser.add_argument('--edit_dim', type=int, default=32,
                        help='Edit vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help='KL divergence weight')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='flask_app_data/results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Auto-detect CUDA
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    main(args)

