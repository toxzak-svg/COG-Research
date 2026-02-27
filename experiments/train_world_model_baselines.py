#!/usr/bin/env python3
"""
Train world-model baselines for damped and vanderpol datasets.

This script trains world-model-first paradigm models to enable
cross-dataset verification experiments.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deterministic_data.deterministic_dataset import load_deterministic_data
from imagination_first_learning.models.vae import SequenceVAE


def train_world_model(dataset: str, hidden_dim: int, seed: int, epochs: int = 50, device: str = "cpu"):
    """Train a world-model-first (VAE) baseline."""
    print(f"\nTraining world-model: dataset={dataset}, hidden_dim={hidden_dim}, seed={seed}")
    
    # Set seed
    torch.manual_seed(seed)
    
    # Load data
    train_data, val_data, test_data = load_deterministic_data(dataset)
    
    # Get dimensions
    seq_len, obs_dim = train_data.shape[1], train_data.shape[2]
    
    # Create model
    model = SequenceVAE(
        obs_dim=obs_dim,
        latent_dim=hidden_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        batch_size = 32
        n_batches = len(train_data) // batch_size
        
        for i in range(n_batches):
            batch = train_data[i * batch_size:(i + 1) * batch_size].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            
            # VAE loss
            recon_loss = torch.nn.functional.mse_loss(recon, batch)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld = kld / (batch_size * seq_len * obs_dim)
            loss = recon_loss + 0.1 * kld
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_batch = val_data[:32].to(device)
            recon, mu, logvar = model(val_batch)
            recon_loss = torch.nn.functional.mse_loss(recon, val_batch)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld = kld / (32 * seq_len * obs_dim)
            val_loss = recon_loss + 0.1 * kld
            val_loss = val_loss.item()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = Path(f"results/{dataset}/world_model_first/{hidden_dim}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"seed_{seed}.pth")
    
    print(f"  ✓ Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"  ✓ Saved to: {save_dir / f'seed_{seed}.pth'}")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train world-model baselines")
    parser.add_argument("--datasets", nargs="+", default=["damped", "vanderpol"],
                        help="Datasets to train on")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[16, 32, 64, 128],
                        help="Hidden dimensions to train")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)),
                        help="Random seeds")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--device", default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training World-Model Baselines")
    print("="*60)
    print(f"Datasets: {args.datasets}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Seeds: {len(args.seeds)} seeds ({min(args.seeds)}-{max(args.seeds)})")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*60)
    
    results = {}
    
    for dataset in args.datasets:
        results[dataset] = {}
        for hidden_dim in args.hidden_dims:
            results[dataset][hidden_dim] = {}
            for seed in args.seeds:
                try:
                    val_loss = train_world_model(
                        dataset=dataset,
                        hidden_dim=hidden_dim,
                        seed=seed,
                        epochs=args.epochs,
                        device=args.device
                    )
                    results[dataset][hidden_dim][seed] = {
                        "status": "success",
                        "val_loss": val_loss
                    }
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    results[dataset][hidden_dim][seed] = {
                        "status": "failed",
                        "error": str(e)
                    }
    
    # Save summary
    summary_path = Path("results/world_model_baseline_training.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Training complete! Summary saved to: {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
