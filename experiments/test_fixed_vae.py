"""
Quick test to verify the fixed VAE works correctly on ETTh1 data.
Tests the improved TimeseriesVAE with proper activations and loss scaling.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from imagination_first_learning.models.vae_timeseries import TimeseriesVAE
from timeseries_pile_data.forecasting_loader import create_dataloaders

def test_vae_timeseries():
    """Test the improved VAE on a small batch of ETTh1 data."""
    print("="*80)
    print("Testing Improved TimeseriesVAE")
    print("="*80)
    
    # Load a small batch of data
    print("\nLoading ETTh1 data...")
    dataloaders = create_dataloaders(
        dataset_name='ETTh1',
        seq_len=96,
        pred_len=24,
        batch_size=32,
        data_root='data/Timeseries-PILE/forecasting/autoformer',
        normalize='standard',
        num_workers=0,
    )
    
    train_loader = dataloaders['train']
    metadata = dataloaders['metadata']
    n_features = metadata['features']
    input_dim = 96 * n_features
    
    print(f"Features: {n_features}")
    print(f"Input dim: {input_dim}")
    
    # Create improved model
    print("\nCreating TimeseriesVAE...")
    model = TimeseriesVAE(
        input_dim=input_dim,
        latent_dim=16,
        hidden_dims=[256, 128],
        use_layer_norm=False,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    input_seq, _ = next(iter(train_loader))
    batch_size, seq_len, n_features = input_seq.shape
    x = input_seq.reshape(batch_size, -1)
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Input mean: {x.mean():.3f}, std: {x.std():.3f}")
    
    with torch.no_grad():
        x_recon, mu, logvar = model(x)
    
    print(f"\nOutput shape: {x_recon.shape}")
    print(f"Output range: [{x_recon.min():.3f}, {x_recon.max():.3f}]")
    print(f"Output mean: {x_recon.mean():.3f}, std: {x_recon.std():.3f}")
    
    # Compute losses
    recon_loss = torch.nn.functional.mse_loss(x_recon, x, reduction='sum') / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    # Normalized loss
    recon_loss_norm = recon_loss / input_dim
    
    print(f"\nLosses:")
    print(f"  Reconstruction loss (raw): {recon_loss.item():.3f}")
    print(f"  Reconstruction loss (normalized): {recon_loss_norm.item():.6f}")
    print(f"  KL divergence: {kl_loss.item():.3f}")
    print(f"  Latent mu range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  Latent logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")
    
    # Check for issues
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    issues = []
    
    if torch.isnan(x_recon).any():
        issues.append("❌ NaN in reconstruction")
    else:
        print("✓ No NaN in reconstruction")
    
    if torch.isinf(kl_loss):
        issues.append("❌ Inf in KL loss")
    else:
        print("✓ KL loss is finite")
    
    if kl_loss.item() > 100:
        issues.append(f"⚠️  KL loss is high: {kl_loss.item():.1f}")
    else:
        print(f"✓ KL loss is reasonable: {kl_loss.item():.3f}")
    
    if abs(x_recon.mean().item()) > 2:
        issues.append(f"⚠️  Reconstruction mean far from 0: {x_recon.mean():.3f}")
    else:
        print(f"✓ Reconstruction mean is reasonable: {x_recon.mean():.3f}")
    
    # Compare with input distribution
    if abs(x_recon.std().item() - x.std().item()) > 1.0:
        issues.append(f"⚠️  Reconstruction std differs significantly: input={x.std():.3f}, output={x_recon.std():.3f}")
    else:
        print(f"✓ Reconstruction std is close to input: input={x.std():.3f}, output={x_recon.std():.3f}")
    
    if recon_loss_norm.item() > 1.0:
        issues.append(f"⚠️  Normalized recon loss is high: {recon_loss_norm.item():.3f}")
    else:
        print(f"✓ Normalized reconstruction loss is reasonable: {recon_loss_norm.item():.6f}")
    
    print("\n" + "="*80)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ ALL CHECKS PASSED!")
    print("="*80)
    
    return model, recon_loss_norm.item(), kl_loss.item()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    model, recon_loss, kl_loss = test_vae_timeseries()
    
    print(f"\nSummary:")
    print(f"  Final normalized reconstruction loss: {recon_loss:.6f}")
    print(f"  Final KL loss: {kl_loss:.3f}")
    print(f"\nThis model should achieve ~0.1-0.2 validation loss after training")
    print(f"(compared to ~0.09 for self-model)")
