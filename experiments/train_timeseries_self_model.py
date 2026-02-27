"""
Training script for self-model on real-world time series from Timeseries-PILE.
Adapted from minimal_self_model/experiments/train_self_model.py.
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

from minimal_self_model.models.self_model import SelfModel
from timeseries_pile_data.forecasting_loader import create_dataloaders
from experiments.timeseries_config import TimeseriesExperimentConfig, get_config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        input_seq = input_seq.to(device)  # (batch, seq_len, features)
        target_seq = target_seq.to(device)  # (batch, pred_len, features)
        
        optimizer.zero_grad()
        
        # Forward pass
        # Self-model takes sequence and predicts all next-step transitions
        batch_size, seq_len, n_features = input_seq.shape
        
        # Concatenate input and target for training on full sequence
        full_seq = torch.cat([input_seq, target_seq], dim=1)  # (batch, seq_len+pred_len, features)
        
        # Model predicts next-step transitions for the sequence
        # Input: full_seq[:, :-1], Target: full_seq[:, 1:]
        predictions = model(full_seq[:, :-1, :])  # (batch, seq_len+pred_len-1, features)
        
        # We only care about predictions on the target portion
        # The last pred_len predictions correspond to the forecast horizon
        pred_len = target_seq.shape[1]
        predictions = predictions[:, -pred_len:, :]  # (batch, pred_len, features)
        
        # Compute loss
        loss = nn.functional.mse_loss(predictions, target_seq)
        
        # Backward pass
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            batch_size, seq_len, n_features = input_seq.shape
            pred_len = target_seq.shape[1]
            
            # Concatenate input and target
            full_seq = torch.cat([input_seq, target_seq], dim=1)
            
            # Model predicts next-step transitions
            predictions = model(full_seq[:, :-1, :])
            
            # Extract predictions for the forecast horizon
            predictions = predictions[:, -pred_len:, :]
            loss = nn.functional.mse_loss(predictions, target_seq)
            
            total_loss += loss.item()
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
    }


def main(config: TimeseriesExperimentConfig):
    """Main training function."""
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
    print(f"Model: {config.model_type}")
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
    
    print(f"Number of features: {n_features}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = SelfModel(
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        output_dim=n_features,
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
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    for epoch in range(config.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, config.device, config.grad_clip
        )
        
        # Validate
        if epoch % config.eval_freq == 0:
            val_metrics = evaluate(model, val_loader, config.device)
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            
            print(f"Epoch {epoch+1}/{config.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.6f} | "
                  f"Val Loss: {val_metrics['loss']:.6f}")
            
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
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'best_val_loss': best_val_loss,
        'final_epoch': epoch,
        'history': history,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train self-model on Timeseries-PILE datasets')
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
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides preset)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides preset)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for training')
    
    args = parser.parse_args()
    
    # Get base config from preset
    config = get_config(args.preset)
    
    # Override with command-line arguments
    if args.dataset is not None:
        config.dataset_name = args.dataset
    if args.seq_len is not None:
        config.seq_len = args.seq_len
    if args.pred_len is not None:
        config.pred_len = args.pred_len
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.seed is not None:
        config.seed = args.seed
    config.device = args.device
    
    # Run training
    results = main(config)
