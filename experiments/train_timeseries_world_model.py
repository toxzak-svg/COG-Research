"""
Training script for world-model (VAE) on real-world time series from Timeseries-PILE.
Adapted from imagination_first_learning/experiments/train_vae.py.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from imagination_first_learning.models.vae import VAE
from timeseries_pile_data.forecasting_loader import create_dataloaders
from experiments.timeseries_config import TimeseriesExperimentConfig, get_config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: VAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    kl_weight: float = 1.0,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    n_batches = 0
    
    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        # Concatenate input and target for VAE training
        # VAE learns to reconstruct entire sequences
        input_seq = input_seq.to(device)  # (batch, seq_len, features)
        
        batch_size, seq_len, n_features = input_seq.shape
        
        # Flatten sequence dimension for VAE
        x = input_seq.reshape(batch_size, -1)  # (batch, seq_len * features)
        
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, mu, logvar = model(x)
        
        # Compute losses
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / batch_size
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        loss = recon_loss + kl_weight * kl_loss
        
        # Backward pass
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
    }


def evaluate(
    model: VAE,
    dataloader: DataLoader,
    device: str,
    kl_weight: float = 1.0,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            
            batch_size, seq_len, n_features = input_seq.shape
            x = input_seq.reshape(batch_size, -1)
            
            x_recon, mu, logvar = model(x)
            
            recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / batch_size
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            
            loss = recon_loss + kl_weight * kl_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
    }


def main(config: TimeseriesExperimentConfig):
    """Main training function."""
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    print(f"Experiment: {config.experiment_name}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Model: VAE (world-model)")
    print(f"Sequence length: {config.seq_len}, Prediction length: {config.pred_len}")
    
    # Create dataloaders
    print("\nLoading data...")
    dataloaders = create_dataloaders(
        dataset_name=config.dataset_name,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        batch_size=config.batch_size,
        data_root=config.data_root,
        normalize=config.normalize,
        num_workers=config.num_workers,
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    metadata = dataloaders['metadata']
    n_features = metadata['features']
    
    # VAE input dimension is seq_len * n_features
    input_dim = config.seq_len * n_features
    
    print(f"Number of features: {n_features}")
    print(f"VAE input dimension: {input_dim}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = VAE(
        input_dim=input_dim,
        latent_dim=config.vae_latent_dim,
    ).to(config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    if config.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Create scheduler
    if config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
    elif config.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.epochs // 3,
            gamma=0.1,
        )
    else:
        scheduler = None
    
    # KL warmup schedule
    def get_kl_weight(epoch):
        if epoch < config.kl_warmup_epochs:
            return config.kl_weight * (epoch / config.kl_warmup_epochs)
        return config.kl_weight
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
    }
    
    for epoch in range(config.epochs):
        kl_weight = get_kl_weight(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, config.device,
            kl_weight=kl_weight, grad_clip=config.grad_clip
        )
        
        # Validate
        if epoch % config.eval_freq == 0:
            val_metrics = evaluate(model, val_loader, config.device, kl_weight=kl_weight)
            
            history['train_loss'].append(train_metrics['loss'])
            history['train_recon_loss'].append(train_metrics['recon_loss'])
            history['train_kl_loss'].append(train_metrics['kl_loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_recon_loss'].append(val_metrics['recon_loss'])
            history['val_kl_loss'].append(val_metrics['kl_loss'])
            
            print(f"Epoch {epoch+1}/{config.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.3f} "
                  f"(Recon: {train_metrics['recon_loss']:.3f}, KL: {train_metrics['kl_loss']:.3f}) | "
                  f"Val Loss: {val_metrics['loss']:.3f} "
                  f"(Recon: {val_metrics['recon_loss']:.3f}, KL: {val_metrics['kl_loss']:.3f})")
            
            # Check for improvement
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                if config.save_checkpoints:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'config': config.__dict__,
                    }, output_dir / 'best_model.pth')
            else:
                patience_counter += 1
                
                if patience_counter >= config.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    if config.save_checkpoints:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
        }, output_dir / 'final_model.pth')
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.3f}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'best_val_loss': best_val_loss,
        'final_epoch': epoch,
        'history': history,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train world-model (VAE) on Timeseries-PILE datasets')
    parser.add_argument('--preset', type=str, default='ETTh1_short',
                       help='Preset configuration name')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides preset)')
    parser.add_argument('--seq-len', type=int, default=None,
                       help='Input sequence length (overrides preset)')
    parser.add_argument('--pred-len', type=int, default=None,
                       help='Prediction length (overrides preset)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden dimension (overrides preset)')
    parser.add_argument('--latent-dim', type=int, default=None,
                       help='Latent dimension (overrides preset)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides preset)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides preset)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for training')
    
    args = parser.parse_args()
    
    # Get base config from preset
    config = get_config(args.preset)
    config.model_type = 'world_model'
    
    # Override with command-line arguments
    if args.dataset is not None:
        config.dataset_name = args.dataset
    if args.seq_len is not None:
        config.seq_len = args.seq_len
    if args.pred_len is not None:
        config.pred_len = args.pred_len
    if args.hidden_dim is not None:
        config.vae_hidden_dim = args.hidden_dim
    if args.latent_dim is not None:
        config.vae_latent_dim = args.latent_dim
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.seed is not None:
        config.seed = args.seed
    config.device = args.device
    
    # Run training
    results = main(config)
