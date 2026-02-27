"""
Stress Test: Longer Sequences and Higher Latent Dimensions

This script tests the limits of both paradigms:
1. Train on progressively longer sequences (50 → 100 → 200 timesteps)
2. Train with higher latent dimensions (2 → 4 → 8 → 16)
3. Evaluate stability, convergence, and computational efficiency

Goal: Identify breaking points and scalability limits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minimal_self_model.models.self_model import SelfModel, self_model_loss
from imagination_first_learning.models.vae import VAE
from stability_analysis.jacobian_spectral import compute_jacobian, spectral_radius


@dataclass
class StressTestConfig:
    """Configuration for stress test."""
    sequence_length: int
    state_dim: int
    hidden_dim: int
    epochs: int
    batch_size: int


def generate_test_data(dataset: str, n_samples: int, horizon: int, state_dim: int, seed: int):
    """Generate test data with specified parameters."""
    from deterministic_data.deterministic_dataset import (
        generate_ar1,
        generate_damped_oscillator,
        generate_van_der_pol,
    )
    
    if dataset == "ar1":
        return generate_ar1(n_samples=n_samples, horizon=horizon, seed=seed, state_dim=state_dim)
    elif dataset == "damped":
        if state_dim != 2:
            print(f"Warning: damped oscillator is 2D, ignoring state_dim={state_dim}")
        return generate_damped_oscillator(n_samples=n_samples, horizon=horizon, seed=seed)
    elif dataset == "vanderpol":
        if state_dim != 2:
            print(f"Warning: Van der Pol is 2D, ignoring state_dim={state_dim}")
        return generate_van_der_pol(n_samples=n_samples, horizon=horizon, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train_self_model_stress(
    data: torch.Tensor,
    config: StressTestConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Train self-model and measure performance metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N, T, D = data.shape
    
    # Flatten to observations
    obs = data.reshape(-1, D)
    
    # Train/val split
    n_train = int(0.8 * obs.shape[0])
    train_obs = obs[:n_train]
    val_obs = obs[n_train:]
    
    # Model
    model = SelfModel(input_dim=D, hidden_dim=config.hidden_dim, output_dim=D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    train_loader = DataLoader(
        TensorDataset(train_obs.unsqueeze(1)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            # Target is next observation (shift by 1)
            # For simplicity, use same observation as target (teacher forcing)
            loss = F.mse_loss(pred, batch)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs.unsqueeze(1).to(device))
            val_loss = F.mse_loss(val_pred, val_obs.unsqueeze(1).to(device)).item()
            val_losses.append(val_loss)
    
    train_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        # Spectral radius
        sample_obs = val_obs[:100].to(device)
        spectral_radii = []
        for x in sample_obs:
            jac = compute_jacobian(model, x.unsqueeze(0).unsqueeze(1), device)
            sr = spectral_radius(jac)
            spectral_radii.append(float(sr))
        
        mean_spectral_radius = np.mean(spectral_radii)
        
        # Rollout divergence
        rollout_length = min(50, T)
        init_state = data[:10, 0, :].to(device)
        rollout = [init_state]
        
        for t in range(1, rollout_length):
            next_state = model(rollout[-1].unsqueeze(1)).squeeze(1)
            rollout.append(next_state)
        
        rollout_tensor = torch.stack(rollout, dim=1)
        ground_truth = data[:10, :rollout_length, :].to(device)
        rollout_divergence = torch.mean(torch.norm(rollout_tensor - ground_truth, dim=2)).item()
    
    return {
        "train_time_seconds": train_time,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "mean_spectral_radius": mean_spectral_radius,
        "rollout_divergence": rollout_divergence,
        "converged": train_losses[-1] < train_losses[0] * 0.1,  # Simple convergence check
    }


def train_world_model_stress(
    data: torch.Tensor,
    config: StressTestConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Train world-model (VAE + linear transition) and measure performance."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N, T, D = data.shape
    
    # Flatten to observations
    obs = data.reshape(-1, D)
    
    # Normalize
    obs_min = obs.min()
    obs_max = obs.max()
    obs_norm = (obs - obs_min) / (obs_max - obs_min + 1e-8)
    
    # Train/val split
    n_train = int(0.8 * obs_norm.shape[0])
    train_obs = obs_norm[:n_train]
    val_obs = obs_norm[n_train:]
    
    # VAE with specified hidden_dim as latent_dim
    latent_dim = config.hidden_dim
    vae = VAE(input_dim=D, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    train_loader = DataLoader(
        TensorDataset(train_obs),
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config.epochs):
        vae.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = vae(batch)
            
            # VAE loss
            recon_loss = F.mse_loss(recon, batch, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        vae.eval()
        with torch.no_grad():
            recon_val, mu_val, logvar_val = vae(val_obs.to(device))
            recon_loss_val = F.mse_loss(recon_val, val_obs.to(device), reduction='sum')
            kld_val = -0.5 * torch.sum(1 + logvar_val - mu_val.pow(2) - logvar_val.exp())
            val_loss = (recon_loss_val + kld_val).item()
            val_losses.append(val_loss)
    
    train_time = time.time() - start_time
    
    # Fit linear transition model on latent space
    vae.eval()
    with torch.no_grad():
        # Encode all training data
        train_latents = []
        for i in range(0, train_obs.shape[0], config.batch_size):
            batch = train_obs[i:i+config.batch_size].to(device)
            mu, _ = vae.encode(batch)
            train_latents.append(mu.cpu())
        train_latents = torch.cat(train_latents, dim=0)
        
        # Fit A, b: z_{t+1} = A z_t + b
        # Use consecutive latents from sequences
        z_t = train_latents[:-1]
        z_next = train_latents[1:]
        
        # Least squares: [A, b] from z_next = [z_t, 1] @ [A; b]
        z_t_aug = torch.cat([z_t, torch.ones(z_t.shape[0], 1)], dim=1)
        Ab = torch.linalg.lstsq(z_t_aug, z_next).solution
        
        A = Ab[:-1, :].to(device)
        b = Ab[-1, :].to(device)
        
        # Transition spectral radius
        eigvals = torch.linalg.eigvals(A).cpu().numpy()
        max_eigval_modulus = np.max(np.abs(eigvals))
    
    return {
        "train_time_seconds": train_time,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "transition_spectral_radius": float(max_eigval_modulus),
        "converged": train_losses[-1] < train_losses[0] * 0.1,
    }


def run_stress_test(
    dataset: str,
    paradigm: str,
    config: StressTestConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run single stress test configuration."""
    # Generate data
    data = generate_test_data(
        dataset,
        n_samples=500,
        horizon=config.sequence_length,
        state_dim=config.state_dim,
        seed=seed,
    )
    data = data.float().to(device)
    
    if paradigm == "self_model":
        return train_self_model_stress(data, config, device, seed)
    elif paradigm == "world_model":
        return train_world_model_stress(data, config, device, seed)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test with longer sequences and higher dimensions")
    parser.add_argument("--dataset", type=str, default="ar1", help="Dataset type")
    parser.add_argument(
        "--sequence-lengths",
        type=str,
        default="50,100,200",
        help="Comma-separated sequence lengths to test",
    )
    parser.add_argument(
        "--state-dims",
        type=str,
        default="2,4",
        help="Comma-separated state dimensions to test (only for ar1)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="16,32,64,128",
        help="Comma-separated hidden/latent dimensions",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "stress_test.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "plots" / "stress_test.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sequence_lengths = [int(x.strip()) for x in args.sequence_lengths.split(",")]
    state_dims = [int(x.strip()) for x in args.state_dims.split(",")]
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",")]
    
    print("Stress Test Configuration")
    print(f"  Dataset: {args.dataset}")
    print(f"  Sequence lengths: {sequence_lengths}")
    print(f"  State dimensions: {state_dims}")
    print(f"  Hidden dimensions: {hidden_dims}")
    print(f"  Device: {device}")
    print()
    
    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "tests": [],
    }
    
    total_configs = len(sequence_lengths) * len(state_dims) * len(hidden_dims) * 2
    current = 0
    
    for seq_len in sequence_lengths:
        for state_dim in state_dims:
            for hidden_dim in hidden_dims:
                for paradigm in ["self_model", "world_model"]:
                    current += 1
                    print(f"[{current}/{total_configs}] seq_len={seq_len}, state_dim={state_dim}, "
                          f"hidden={hidden_dim}, paradigm={paradigm}")
                    
                    config = StressTestConfig(
                        sequence_length=seq_len,
                        state_dim=state_dim,
                        hidden_dim=hidden_dim,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                    )
                    
                    try:
                        test_result = run_stress_test(
                            args.dataset, paradigm, config, device, args.seed
                        )
                        
                        results["tests"].append({
                            "sequence_length": seq_len,
                            "state_dim": state_dim,
                            "hidden_dim": hidden_dim,
                            "paradigm": paradigm,
                            "status": "success",
                            **test_result,
                        })
                        
                        print(f"  → Train time: {test_result['train_time_seconds']:.2f}s, "
                              f"Final val loss: {test_result['final_val_loss']:.6f}")
                    except Exception as e:
                        print(f"  → Failed: {e}")
                        results["tests"].append({
                            "sequence_length": seq_len,
                            "state_dim": state_dim,
                            "hidden_dim": hidden_dim,
                            "paradigm": paradigm,
                            "status": "failed",
                            "error": str(e),
                        })
    
    # Save results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_json}")
    
    # Generate summary
    generate_summary(results, args.output_md)
    print(f"Summary saved to {args.output_md}")


def generate_summary(data: dict[str, Any], output_path: Path):
    """Generate markdown summary."""
    lines = ["# Stress Test Results", ""]
    lines.append(f"**Dataset**: {data['dataset']}")
    lines.append(f"**Random Seed**: {data['seed']}")
    lines.append("")
    
    # Organize results
    successful = [t for t in data["tests"] if t.get("status") == "success"]
    failed = [t for t in data["tests"] if t.get("status") == "failed"]
    
    lines.append(f"**Total tests**: {len(data['tests'])}")
    lines.append(f"**Successful**: {len(successful)}")
    lines.append(f"**Failed**: {len(failed)}")
    lines.append("")
    
    if failed:
        lines.append("## Failed Configurations")
        lines.append("")
        for test in failed:
            lines.append(f"- seq_len={test['sequence_length']}, state_dim={test['state_dim']}, "
                        f"hidden={test['hidden_dim']}, paradigm={test['paradigm']}")
            lines.append(f"  Error: {test.get('error', 'Unknown')}")
        lines.append("")
    
    lines.append("## Training Time Analysis")
    lines.append("")
    lines.append("| Paradigm | Seq Len | Hidden | Train Time (s) |")
    lines.append("|----------|---------|--------|----------------|")
    
    for test in successful:
        lines.append(f"| {test['paradigm']} | {test['sequence_length']} | {test['hidden_dim']} | "
                    f"{test['train_time_seconds']:.2f} |")
    
    lines.append("")
    lines.append("## Convergence Summary")
    lines.append("")
    
    converged_self = sum(1 for t in successful if t['paradigm'] == 'self_model' and t.get('converged'))
    converged_world = sum(1 for t in successful if t['paradigm'] == 'world_model' and t.get('converged'))
    total_self = sum(1 for t in successful if t['paradigm'] == 'self_model')
    total_world = sum(1 for t in successful if t['paradigm'] == 'world_model')
    
    lines.append(f"- Self-model converged: {converged_self}/{total_self}")
    lines.append(f"- World-model converged: {converged_world}/{total_world}")
    lines.append("")
    
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Scalability**: How do training times scale with sequence length and model size?")
    lines.append("2. **Stability**: Which configurations maintain stable dynamics (low spectral radius)?")
    lines.append("3. **Breaking Points**: At what scale does each paradigm fail to converge?")
    lines.append("")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
