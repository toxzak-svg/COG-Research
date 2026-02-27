"""
Unified Multi-Seed Experiment Runner for Comparative Validation of Self-Model-First vs World-Model-First Paradigms

This module provides infrastructure for rigorous empirical comparison of:
- Self-Model-First: RNN trained on self-dynamics (predicting own next state)
- World-Model-First: VAE with latent transition model

Key functionality:
- Multi-seed experiments (10+ seeds) for statistical significance
- Parameter count sweeps (hidden_dims: [16, 32, 64, 128])
- Unified evaluation pipeline across both paradigms
- Results aggregation with mean/std statistics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from minimal_self_model.models.self_model import SelfModel, self_model_loss
from imagination_first_learning.models.vae import VAE
from deterministic_data.deterministic_dataset import (
    generate_ar1,
    generate_damped_oscillator,
    generate_van_der_pol,
)

# Stability analysis imports
from stability_analysis.jacobian_spectral import (
    compute_jacobian,
    spectral_radius,
    find_equilibrium,
    classify_stability,
)
from stability_analysis.perturbation_return import (
    inject_perturbation,
    rollout_from_perturbed,
    compute_return_rate,
    estimate_return_time,
)


# ==============================================================================
# Experiment Configuration
# ==============================================================================

DATASETS = ["ar1", "damped", "vanderpol"]
DEFAULT_HIDDEN_DIMS = [16, 32, 64, 128]
DEFAULT_N_SEEDS = 10
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3

# Paradigm names
SELF_MODEL_FIRST = "self_model_first"
WORLD_MODEL_FIRST = "world_model_first"


def get_data_path(dataset: str, split: str = "train") -> Path:
    """Get path to deterministic dataset file."""
    return PROJECT_ROOT / "data" / "deterministic" / f"{dataset}_{split}.pt"


def generate_dataset_if_needed(dataset: str, seed: int = 42):
    """Generate dataset if not already present."""
    train_path = get_data_path(dataset, "train")
    if train_path.exists():
        return
    
    print(f"Generating dataset: {dataset}")
    if dataset == "ar1":
        data = generate_ar1(n_samples=1400, horizon=50, seed=seed, state_dim=2)
    elif dataset == "damped":
        data = generate_damped_oscillator(n_samples=1400, horizon=50, seed=seed)
    elif dataset == "vanderpol":
        data = generate_van_der_pol(n_samples=1400, horizon=50, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Split into train/val/test
    train, val, test = split_data(data, 1000, 200, 200, seed)
    
    # Save
    os.makedirs(train_path.parent, exist_ok=True)
    torch.save(train, train_path)
    torch.save(val, get_data_path(dataset, "val"))
    torch.save(test, get_data_path(dataset, "test"))


def split_data(data: torch.Tensor, train_size: int, val_size: int, test_size: int, seed: int):
    """Split data into train/val/test."""
    torch.manual_seed(seed)
    indices = torch.randperm(len(data))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:train_size + val_size + test_size]
    return data[train_idx], data[val_idx], data[test_idx]


# ==============================================================================
# Training Interfaces
# ==============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    """Get torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def train_self_model(
    dataset: str,
    hidden_dim: int,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str = "auto",
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Train a self-model (RNN) on self-dynamics.
    
    Args:
        dataset: Dataset name (ar1, damped, vanderpol)
        hidden_dim: RNN hidden dimension
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        output_dir: Directory to save checkpoint
        
    Returns:
        Dictionary with training metadata
    """
    set_seed(seed)
    device = get_device(device)
    
    # Load data
    data_path = get_data_path(dataset, "train")
    generate_dataset_if_needed(dataset)
    sequences = torch.load(data_path)
    num_samples, seq_len, obs_dim = sequences.shape
    
    # Prepare inputs/targets
    inputs = sequences[:, :-1, :]
    targets = sequences[:, 1:, :]
    dataset_tensor = TensorDataset(inputs, targets)
    
    # Split for validation
    val_fraction = 0.1
    val_size = max(1, int(len(dataset_tensor) * val_fraction))
    train_size = len(dataset_tensor) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset_tensor, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SelfModel(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=obs_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train
    last_train_mse = None
    last_val_mse = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_mse_sum = 0.0
        train_elements = 0
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = self_model_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_mse_sum += F.mse_loss(pred, target, reduction="sum").item()
            train_elements += target.numel()
        last_train_mse = train_mse_sum / train_elements
        
        # Validation
        model.eval()
        val_mse_sum = 0.0
        val_elements = 0
        with torch.no_grad():
            for x, target in val_loader:
                x, target = x.to(device), target.to(device)
                pred = model(x)
                val_mse_sum += F.mse_loss(pred, target, reduction="sum").item()
                val_elements += target.numel()
        last_val_mse = val_mse_sum / val_elements
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_mse={last_train_mse:.6f}, val_mse={last_val_mse:.6f}")
    
    # Save checkpoint
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    checkpoint_path = output_dir / dataset / SELF_MODEL_FIRST / str(hidden_dim) / f"seed_{seed}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": int(obs_dim),
            "hidden_dim": int(hidden_dim),
            "output_dim": int(obs_dim),
            "paradigm": SELF_MODEL_FIRST,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "dataset": dataset,
            "last_train_mse": float(last_train_mse),
            "last_val_mse": float(last_val_mse),
        },
    }
    torch.save(checkpoint, checkpoint_path)
    
    return {
        "checkpoint_path": str(checkpoint_path),
        "train_mse": float(last_train_mse),
        "val_mse": float(last_val_mse),
    }


def train_world_model(
    dataset: str,
    hidden_dim: int,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str = "auto",
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Train a world-model (VAE with latent transition) on the dataset.
    
    The world-model consists of:
    1. VAE encoder/decoder for observation space
    2. Linear transition model in latent space
    
    Args:
        dataset: Dataset name (ar1, damped, vanderpol)
        hidden_dim: Latent dimension (matches hidden_dim for fair comparison)
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        output_dir: Directory to save checkpoint
        
    Returns:
        Dictionary with training metadata
    """
    set_seed(seed)
    device = get_device(device)
    
    # Load data
    generate_dataset_if_needed(dataset)
    data_path = get_data_path(dataset, "train")
    sequences = torch.load(data_path)
    num_samples, seq_len, obs_dim = sequences.shape
    
    # For VAE, we use flattened observations (each timestep as independent sample)
    # This is the standard VAE approach for world modeling
    obs_data = sequences.reshape(-1, obs_dim)  # (num_samples * seq_len, obs_dim)

    # Ensure data is in [0, 1] for VAE sigmoid decoder
    seq_min = sequences.min()
    seq_max = sequences.max()
    obs_data = (obs_data - seq_min) / (seq_max - seq_min + 1e-8)
    
    dataset_tensor = TensorDataset(obs_data)
    
    # Split for validation
    val_fraction = 0.1
    val_size = max(1, int(len(dataset_tensor) * val_fraction))
    train_size = len(dataset_tensor) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset_tensor, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create VAE model
    # Use hidden_dim as latent dimension for fair comparison
    latent_dim = hidden_dim
    model = VAE(input_dim=obs_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train VAE
    last_train_loss = None
    last_val_loss = None
    
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, log_var = model(x)
            
            # VAE loss
            recon_loss = F.mse_loss(recon_x, x, reduction="sum")
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_div
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_samples += x.size(0)
        
        last_train_loss = train_loss_sum / train_samples
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon_x, mu, log_var = model(x)
                recon_loss = F.mse_loss(recon_x, x, reduction="sum")
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_div
                val_loss_sum += loss.item()
                val_samples += x.size(0)
        
        last_val_loss = val_loss_sum / val_samples
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={last_train_loss:.6f}, val_loss={last_val_loss:.6f}")
    
    # Fit linear latent transition model with true within-sequence pairs.
    # Avoid flattened-adjacency leakage across sequence boundaries.
    model.eval()
    x_t = sequences[:, :-1, :].reshape(-1, obs_dim)
    x_t_plus_1 = sequences[:, 1:, :].reshape(-1, obs_dim)
    x_t = (x_t - seq_min) / (seq_max - seq_min + 1e-8)
    x_t_plus_1 = (x_t_plus_1 - seq_min) / (seq_max - seq_min + 1e-8)

    all_latents = []
    all_next_latents = []
    with torch.no_grad():
        for i in range(0, x_t.shape[0], batch_size):
            batch = x_t[i:i + batch_size].to(device)
            next_batch = x_t_plus_1[i:i + batch_size].to(device)
            mu, _ = model.encode(batch)
            next_mu, _ = model.encode(next_batch)
            all_latents.append(mu.cpu())
            all_next_latents.append(next_mu.cpu())
    
    if not all_latents:
        raise ValueError("No latent representations could be computed")
    
    latents = torch.cat(all_latents, dim=0).numpy()
    next_latents = torch.cat(all_next_latents, dim=0).numpy()
    
    # Check shapes match
    if latents.shape[0] != next_latents.shape[0]:
        raise ValueError(f"Latent shape mismatch: {latents.shape[0]} vs {next_latents.shape[0]}")
    
    # Fit linear transition: z_{t+1} = A @ z_t + b
    ones = np.ones((latents.shape[0], 1))
    design = np.concatenate([latents, ones], axis=1)
    
    # Solve least squares: design @ coeff = next_latents
    coeff, _, _, _ = np.linalg.lstsq(design, next_latents, rcond=None)
    
    A = coeff[:latent_dim, :].astype(np.float32)
    b = coeff[latent_dim, :].astype(np.float32)
    
    # Convert to lists for proper serialization
    A_list = A.tolist()
    b_list = b.tolist()
    
    # Compute latent transition MSE
    z_pred = latents @ A + b
    latent_mse = float(np.mean((z_pred - next_latents) ** 2))
    
    # Save checkpoint
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    checkpoint_path = output_dir / dataset / WORLD_MODEL_FIRST / str(hidden_dim) / f"seed_{seed}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": int(obs_dim),
            "latent_dim": int(latent_dim),
            "paradigm": WORLD_MODEL_FIRST,
            "A": A_list,
            "b": b_list,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "dataset": dataset,
            "last_train_loss": float(last_train_loss),
            "last_val_loss": float(last_val_loss),
            "latent_transition_mse": latent_mse,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    
    return {
        "checkpoint_path": str(checkpoint_path),
        "train_loss": float(last_train_loss),
        "val_loss": float(last_val_loss),
        "latent_transition_mse": latent_mse,
    }


# ==============================================================================
# Evaluation Pipeline
# ==============================================================================

def evaluate_self_model_one_step(checkpoint_path: str, dataset: str, device: str = "auto", checkpoint: dict = None) -> float:
    """Compute one-step MSE for self-model."""
    device = get_device(device)
    
    # Load data
    data_path = get_data_path(dataset, "test")
    sequences = torch.load(data_path)
    
    # Load model - use provided checkpoint or load from file
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    
    hidden_dim = config["hidden_dim"]
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    
    model = SelfModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Compute MSE
    inputs = sequences[:, :-1, :].reshape(-1, input_dim).to(device)
    targets = sequences[:, 1:, :].reshape(-1, output_dim).to(device)
    
    with torch.no_grad():
        # Handle sequence dimension
        inputs_seq = inputs.unsqueeze(1)  # (N, 1, input_dim)
        pred = model(inputs_seq).squeeze(1)  # (N, output_dim)
        mse = F.mse_loss(pred, targets).item()
    
    return mse


def evaluate_world_model_one_step(checkpoint_path: str, dataset: str, device: str = "auto", checkpoint: dict = None) -> float:
    """Compute one-step MSE for world-model (VAE + latent transition)."""
    device = get_device(device)
    
    # Load data
    data_path = get_data_path(dataset, "test")
    sequences = torch.load(data_path)
    obs_dim = sequences.shape[2]
    
    # Load model - use provided checkpoint or load from file
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    
    latent_dim = config["latent_dim"]
    input_dim = config["input_dim"]
    A = np.array(config["A"])
    b = np.array(config["b"])
    
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Normalize data to [0, 1]
    seq_min = sequences.min()
    seq_max = sequences.max()
    sequences_norm = (sequences - seq_min) / (seq_max - seq_min + 1e-8)
    
    # Compute MSE using latent transition + decode
    inputs = sequences_norm[:, :-1, :].reshape(-1, obs_dim).to(device)
    targets = sequences_norm[:, 1:, :].reshape(-1, obs_dim).to(device)
    
    with torch.no_grad():
        # Encode current observations
        mu, _ = model.encode(inputs)
        # Transition in latent space to predict next observations
        z_next = mu @ torch.tensor(A, dtype=torch.float32, device=device) + torch.tensor(b, dtype=torch.float32, device=device)
        # Decode predicted next observations
        pred = model.decode(z_next)
        # Compute next-step MSE
        mse = F.mse_loss(pred, targets).item()
    
    return mse


def evaluate_rollout_divergence(
    checkpoint_path: str,
    dataset: str,
    horizon: int = 50,
    device: str = "auto",
    paradigm: str = SELF_MODEL_FIRST,
    checkpoint: dict = None,
) -> float:
    """
    Compute 50-step rollout divergence (MSE between rollout and ground truth).
    
    For self-model: use RNN hidden state dynamics
    For world-model: use VAE latent transition + decode
    """
    device = get_device(device)
    
    # Load data
    data_path = get_data_path(dataset, "test")
    sequences = torch.load(data_path)
    obs_dim = sequences.shape[2]
    seq_len = sequences.shape[1]
    
    # Cap horizon to available sequence length minus 1 (to have ground truth targets)
    max_horizon = seq_len - 1
    actual_horizon = min(horizon, max_horizon)
    if actual_horizon < horizon:
        print(f"    Warning: requested horizon {horizon} exceeds sequence length {seq_len}, using {actual_horizon}")
    
    # Load model - use provided checkpoint or load from file
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    
    if paradigm == SELF_MODEL_FIRST:
        hidden_dim = config["hidden_dim"]
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        
        model = SelfModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Compute rollout divergence
        with torch.no_grad():
            # Use first half of sequences as starting points
            start_states = sequences[:100, 0, :].to(device)  # (100, obs_dim)
            targets = sequences[:100, actual_horizon, :].to(device)  # (100, obs_dim) at horizon
            
            # Rollout
            current = start_states
            for _ in range(actual_horizon):
                current = model(current.unsqueeze(1)).squeeze(1)
            
            mse = F.mse_loss(current, targets).item()
    
    else:  # WORLD_MODEL_FIRST
        latent_dim = config["latent_dim"]
        input_dim = config["input_dim"]
        A = np.array(config["A"])
        b = np.array(config["b"])
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
        b_tensor = torch.tensor(b, dtype=torch.float32, device=device)
        
        model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Normalize
        seq_min = sequences.min()
        seq_max = sequences.max()
        sequences_norm = (sequences - seq_min) / (seq_max - seq_min + 1e-8)
        
        with torch.no_grad():
            start_states = sequences_norm[:100, 0, :].to(device)
            targets = sequences_norm[:100, actual_horizon, :].to(device)
            
            # Encode start states
            mu, _ = model.encode(start_states)
            z = mu
            
            # Rollout in latent space
            for _ in range(actual_horizon):
                z = z @ A_tensor + b_tensor
            
            # Decode
            pred = model.decode(z)
            
            mse = F.mse_loss(pred, targets).item()
    
    return mse


def compute_spectral_radius_self_model(checkpoint_path: str, dataset: str, device: str = "auto", checkpoint: dict = None) -> float:
    """Compute spectral radius (largest singular value) of Jacobian at equilibrium."""
    device = get_device(device)
    
    # Load model - use provided checkpoint or load from file
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    
    hidden_dim = config["hidden_dim"]
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    
    model = SelfModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Find equilibrium and compute Jacobian
    eq_state, _ = find_equilibrium(model, n_iterations=100, device=device)
    jacobian = compute_jacobian(model, eq_state.unsqueeze(0))
    sr = spectral_radius(jacobian)
    
    return sr


def compute_perturbation_return_rate(
    checkpoint_path: str,
    dataset: str,
    n_trials: int = 50,
    perturbation_magnitude: float = 1.0,
    horizon: int = 50,
    device: str = "auto",
    paradigm: str = SELF_MODEL_FIRST,
    checkpoint: dict = None,
) -> float:
    """
    Compute perturbation return rate (how fast trajectories reconverge after perturbation).
    
    Returns the mean return rate across trials.
    """
    device = get_device(device)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    data_path = get_data_path(dataset, "test")
    sequences = torch.load(data_path)
    
    # Load model - use provided checkpoint or load from file
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    
    if paradigm == SELF_MODEL_FIRST:
        hidden_dim = config["hidden_dim"]
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        
        model = SelfModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Get initial states
        initial_states = sequences[:n_trials, 0, :].to(device)
        
        return_rates = []
        for i in range(n_trials):
            initial_state = initial_states[i:i+1]
            
            # Generate perturbation
            perturbation = torch.randn_like(initial_state) * perturbation_magnitude
            perturbed_state = initial_state + perturbation
            
            # Baseline rollout
            with torch.no_grad():
                baseline_traj = torch.zeros(horizon, output_dim, device=device)
                current = initial_state
                for t in range(horizon):
                    baseline_traj[t] = current
                    current = model(current.unsqueeze(1)).squeeze(1)
            
            # Perturbed rollout
            with torch.no_grad():
                perturbed_traj = torch.zeros(horizon, output_dim, device=device)
                current = perturbed_state
                for t in range(horizon):
                    perturbed_traj[t] = current
                    current = model(current.unsqueeze(1)).squeeze(1)
            
            # Compute return rate
            result = compute_return_rate(
                perturbed_traj.unsqueeze(0),
                baseline_traj.unsqueeze(0),
                metric="mse"
            )
            return_rates.append(result["return_rate"])
        
        return np.mean(return_rates)
    
    else:  # WORLD_MODEL_FIRST
        latent_dim = config["latent_dim"]
        input_dim = config["input_dim"]
        A = np.array(config["A"]).astype(np.float32)
        b = np.array(config["b"]).astype(np.float32)
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
        b_tensor = torch.tensor(b, dtype=torch.float32, device=device)
        
        model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Normalize
        seq_min = sequences.min()
        seq_max = sequences.max()
        sequences_norm = (sequences - seq_min) / (seq_max - seq_min + 1e-8)
        
        # Get initial states
        initial_states = sequences_norm[:n_trials, 0, :].to(device)
        
        return_rates = []
        for i in range(n_trials):
            initial_state = initial_states[i:i+1]
            
            # Encode to latent
            with torch.no_grad():
                mu, _ = model.encode(initial_state)
            
            # Generate perturbation in latent space
            perturbation = torch.randn_like(mu) * perturbation_magnitude
            perturbed_z = mu + perturbation
            
            # Baseline rollout in latent space
            baseline_z = mu
            baseline_traj = torch.zeros(horizon, input_dim, device=device)
            for t in range(horizon):
                baseline_traj[t] = model.decode(baseline_z)
                baseline_z = baseline_z @ A_tensor + b_tensor
            
            # Perturbed rollout
            perturbed_traj = torch.zeros(horizon, input_dim, device=device)
            z = perturbed_z
            for t in range(horizon):
                perturbed_traj[t] = model.decode(z)
                z = z @ A_tensor + b_tensor
            
            # Compute return rate in observation space
            result = compute_return_rate(
                perturbed_traj.unsqueeze(0),
                baseline_traj.unsqueeze(0),
                metric="mse"
            )
            return_rates.append(result["return_rate"])
        
        return np.mean(return_rates)


def evaluate_model(
    checkpoint_path: str,
    dataset: str,
    device: str = "auto",
    paradigm: str = SELF_MODEL_FIRST,
) -> Dict[str, float]:
    """
    Evaluate a trained model on all metrics.
    
    Returns:
        Dictionary with all metric values
    """
    print(f"  Evaluating {paradigm} model on {dataset}...")
    
    device = get_device(device)
    
    # Load checkpoint ONCE to avoid serialization issues
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # One-step MSE
    if paradigm == SELF_MODEL_FIRST:
        one_step_mse = evaluate_self_model_one_step(checkpoint_path, dataset, device, checkpoint=checkpoint)
    else:
        one_step_mse = evaluate_world_model_one_step(checkpoint_path, dataset, device, checkpoint=checkpoint)
    
    # 50-step rollout divergence
    rollout_div = evaluate_rollout_divergence(checkpoint_path, dataset, horizon=50, device=device, paradigm=paradigm, checkpoint=checkpoint)
    
    # Spectral radius
    if paradigm == SELF_MODEL_FIRST:
        try:
            spectral_radius = compute_spectral_radius_self_model(checkpoint_path, dataset, device, checkpoint=checkpoint)
        except Exception as e:
            print(f"    Warning: Could not compute spectral radius: {e}")
            spectral_radius = float('nan')
    else:
        # For VAE, compute spectral radius of latent transition matrix A
        A = np.array(checkpoint["config"]["A"])
        _, s, _ = np.linalg.svd(A)
        spectral_radius = float(s[0]) if len(s) > 0 else float('nan')
    
    # Perturbation return rate
    try:
        return_rate = compute_perturbation_return_rate(checkpoint_path, dataset, paradigm=paradigm, device=device, checkpoint=checkpoint)
    except Exception as e:
        print(f"    Warning: Could not compute return rate: {e}")
        return_rate = float('nan')
    
    return {
        "one_step_mse": one_step_mse,
        "rollout_divergence_50": rollout_div,
        "spectral_radius": spectral_radius,
        "perturbation_return_rate": return_rate,
    }


# ==============================================================================
# Results Aggregation
# ==============================================================================

def aggregate_results(results_dir: Path, dataset: str, paradigm: str, hidden_dim: int) -> Dict[str, Any]:
    """
    Aggregate results across seeds for a given configuration.
    
    Returns:
        Dictionary with mean and std for each metric
    """
    config_dir = results_dir / dataset / paradigm / str(hidden_dim)
    
    # Find all seed files
    seed_files = list(config_dir.glob("seed_*.pth"))
    if not seed_files:
        return None
    
    # Load metrics from each seed
    all_metrics = []
    for seed_file in seed_files:
        metrics_file = seed_file.with_suffix(".json")
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
    
    if not all_metrics:
        return None
    
    # Compute statistics
    metric_names = ["one_step_mse", "rollout_divergence_50", "spectral_radius", "perturbation_return_rate"]
    
    stats = {"n_seeds": len(all_metrics)}
    for metric in metric_names:
        values = [m.get(metric, float('nan')) for m in all_metrics]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            stats[metric] = {"mean": float('nan'), "std": float('nan'), "min": float('nan'), "max": float('nan')}
    
    return stats


def save_metrics(checkpoint_path: str, metrics: Dict[str, float]):
    """Save metrics to JSON file alongside checkpoint."""
    metrics_path = Path(checkpoint_path).with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


# ==============================================================================
# Main Experiment Runner
# ==============================================================================

def run_single_experiment(
    dataset: str,
    paradigm: str,
    hidden_dim: int,
    seed: int,
    output_dir: Path,
    device: str = "auto",
    epochs: int = DEFAULT_EPOCHS,
) -> Dict[str, Any]:
    """Run a single experiment (train + evaluate) for one seed."""
    
    print(f"\n{'='*60}")
    print(f"Running: dataset={dataset}, paradigm={paradigm}, hidden_dim={hidden_dim}, seed={seed}")
    print(f"{'='*60}")
    
    # Train
    if paradigm == SELF_MODEL_FIRST:
        train_result = train_self_model(
            dataset=dataset,
            hidden_dim=hidden_dim,
            seed=seed,
            epochs=epochs,
            output_dir=output_dir,
            device=device,
        )
    else:
        train_result = train_world_model(
            dataset=dataset,
            hidden_dim=hidden_dim,
            seed=seed,
            epochs=epochs,
            output_dir=output_dir,
            device=device,
        )
    
    checkpoint_path = train_result["checkpoint_path"]
    
    # Evaluate
    metrics = evaluate_model(checkpoint_path, dataset, device=device, paradigm=paradigm)
    
    # Add training metrics
    if paradigm == SELF_MODEL_FIRST:
        metrics["train_mse"] = train_result["train_mse"]
        metrics["val_mse"] = train_result["val_mse"]
    else:
        metrics["train_loss"] = train_result["train_loss"]
        metrics["val_loss"] = train_result["val_loss"]
        metrics["latent_transition_mse"] = train_result["latent_transition_mse"]
    
    # Save metrics
    save_metrics(checkpoint_path, metrics)
    
    print(f"  Results: one_step_mse={metrics['one_step_mse']:.6f}, "
          f"rollout_div={metrics['rollout_divergence_50']:.6f}, "
          f"spectral_radius={metrics['spectral_radius']:.6f}")
    
    return {
        "checkpoint_path": checkpoint_path,
        "metrics": metrics,
    }


def run_experiments(
    datasets: List[str],
    paradigms: List[str],
    hidden_dims: List[int],
    n_seeds: int = DEFAULT_N_SEEDS,
    seed_start: int = 0,
    output_dir: Path = None,
    device: str = "auto",
    epochs: int = DEFAULT_EPOCHS,
):
    """Run all experiments for the given configuration."""
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    
    print(f"\n{'#'*60}")
    print(f"# Multi-Seed Comparison Experiment")
    print(f"# Datasets: {datasets}")
    print(f"# Paradigms: {paradigms}")
    print(f"# Hidden dims: {hidden_dims}")
    print(f"# Seeds per config: {n_seeds}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")
    
    # Generate datasets if needed
    for dataset in datasets:
        generate_dataset_if_needed(dataset)
    
    # Run experiments
    results = []
    if n_seeds <= 0:
        raise ValueError("n_seeds must be >= 1")
    if seed_start < 0:
        raise ValueError("seed_start must be >= 0")
    seeds = list(range(seed_start, seed_start + n_seeds))
    
    for dataset in datasets:
        for paradigm in paradigms:
            for hidden_dim in hidden_dims:
                for seed in seeds:
                    result = run_single_experiment(
                        dataset=dataset,
                        paradigm=paradigm,
                        hidden_dim=hidden_dim,
                        seed=seed,
                        output_dir=output_dir,
                        device=device,
                        epochs=epochs,
                    )
                    results.append(result)
    
    # Aggregate results
    print(f"\n{'#'*60}")
    print(f"# Aggregating Results")
    print(f"{'#'*60}")
    
    aggregated = {}
    for dataset in datasets:
        aggregated[dataset] = {}
        for paradigm in paradigms:
            aggregated[dataset][paradigm] = {}
            for hidden_dim in hidden_dims:
                stats = aggregate_results(output_dir, dataset, paradigm, hidden_dim)
                if stats:
                    aggregated[dataset][paradigm][hidden_dim] = stats
    
    # Save aggregated results
    agg_path = output_dir / "aggregated_results.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated results to {agg_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for paradigm in paradigms:
            print(f"\n  Paradigm: {paradigm}")
            for hidden_dim in hidden_dims:
                stats = aggregated.get(dataset, {}).get(paradigm, {}).get(hidden_dim)
                if stats:
                    print(f"    hidden_dim={hidden_dim}:")
                    print(f"      one_step_mse: {stats['one_step_mse']['mean']:.6f} ± {stats['one_step_mse']['std']:.6f}")
                    print(f"      rollout_div: {stats['rollout_divergence_50']['mean']:.6f} ± {stats['rollout_divergence_50']['std']:.6f}")
                    print(f"      spectral_radius: {stats['spectral_radius']['mean']:.6f} ± {stats['spectral_radius']['std']:.6f}")
                    print(f"      return_rate: {stats['perturbation_return_rate']['mean']:.6f} ± {stats['perturbation_return_rate']['std']:.6f}")
    
    return aggregated


# ==============================================================================
# CLI Interface
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-seed comparison of self-model-first vs world-model-first paradigms"
    )
    
    parser.add_argument(
        "--paradigms",
        type=str,
        default="both",
        help="Which paradigms to run: self-model, world-model, or both (comma-separated or 'both')",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="ar1",
        help="Which datasets: ar1, damped, vanderpol (comma-separated)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="16,32,64,128",
        help="Parameter counts to sweep (comma-separated)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of seeds per configuration",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed index (default: 0). Seeds run in [seed_start, seed_start + n_seeds).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to train on",
    )
    
    args = parser.parse_args()
    
    # Parse paradigms
    if args.paradigms.lower() == "both":
        paradigms = [SELF_MODEL_FIRST, WORLD_MODEL_FIRST]
    else:
        paradigms = args.paradigms.split(",")
        paradigms = [p.strip() for p in paradigms]
        # Normalize names
        paradigms = [SELF_MODEL_FIRST if "self" in p.lower() else WORLD_MODEL_FIRST for p in paradigms]
    
    # Parse datasets
    if args.datasets.lower() == "all":
        datasets = DATASETS
    else:
        datasets = args.datasets.split(",")
        datasets = [d.strip() for d in datasets]
    
    # Parse hidden dims
    hidden_dims = [int(h.strip()) for h in args.hidden_dims.split(",")]
    
    # Parse output dir
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    return {
        "paradigms": paradigms,
        "datasets": datasets,
        "hidden_dims": hidden_dims,
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "output_dir": output_dir,
        "epochs": args.epochs,
        "device": args.device,
    }


def main():
    """Main entry point."""
    args = parse_args()
    
    run_experiments(
        datasets=args["datasets"],
        paradigms=args["paradigms"],
        hidden_dims=args["hidden_dims"],
        n_seeds=args["n_seeds"],
        seed_start=args["seed_start"],
        output_dir=args["output_dir"],
        device=args["device"],
        epochs=args["epochs"],
    )


if __name__ == "__main__":
    main()