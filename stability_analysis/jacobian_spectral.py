"""
Jacobian Spectral Analysis for Stability Measurement in Cognitive Architecture Research.

This module provides tools to analyze the stability of RNN self-models by computing
the Jacobian spectral radius at equilibrium points. This is critical for comparing
"self-model-first" vs "world-model-first" architectures.

The key metric is the largest singular value (spectral radius) of the Jacobian:
- σ_max < 1: contracting (stable)
- σ_max ≈ 1: near-critical (marginally stable)
- σ_max > 1: explosive (unstable)
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn_module
from torch.nn import functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minimal_self_model.models.self_model import SelfModel


def get_rnn_hidden_transition(model: nn.Module) -> callable:
    """
    Extract the RNN hidden state transition function from a SelfModel.
    
    The self-model predicts its own internal state transitions. This function
    extracts the transition function f(h) = h' where h is the hidden state.
    
    Args:
        model: A SelfModel instance
        
    Returns:
        A function that computes the next hidden state given current hidden state
        and optional input
    """
    # Get the RNN module
    rnn = model.rnn
    
    # For a standard RNN: h_{t+1} = tanh(W_ih @ x_t + W_hh @ h_t + b_h)
    # We extract the hidden-to-hidden weight matrix
    weight_hh = rnn.weight_hh_l0.data  # (hidden_dim, hidden_dim)
    bias_hh = rnn.bias_hh_l0.data  # (hidden_dim,)
    
    def transition_fn(hidden_state: torch.Tensor, input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the next hidden state given current hidden state.
        
        Args:
            hidden_state: Current hidden state, shape (batch, hidden_dim)
            input_seq: Optional input sequence, shape (batch, seq_len, input_dim)
                      If None, uses zero input
            
        Returns:
            Next hidden state, shape (batch, hidden_dim)
        """
        hidden_dim = weight_hh.shape[0]
        batch_size = hidden_state.shape[0]
        
        # If no input provided, use zeros
        if input_seq is None:
            input_tensor = torch.zeros(batch_size, 1, rnn.input_size, device=hidden_state.device)
        else:
            # Use the last timestep of input if sequence provided
            input_tensor = input_seq[:, -1:, :]  # (batch, 1, input_dim)
        
        # Compute RNN cell step manually to get hidden state transition
        # PyTorch RNN: h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
        weight_ih = rnn.weight_ih_l0.data  # (hidden_dim, input_dim)
        bias_ih = rnn.bias_ih_l0.data  # (hidden_dim,)
        
        # Linear transformation
        if input_tensor.shape[1] > 0:
            ih_term = F.linear(input_tensor.squeeze(1), weight_ih, bias_ih)
        else:
            ih_term = torch.zeros(batch_size, hidden_dim, device=hidden_state.device)
        
        hh_term = F.linear(hidden_state, weight_hh, bias_hh)
        
        # RNN activation
        next_hidden = torch.tanh(ih_term + hh_term)
        
        return next_hidden
    
    return transition_fn


def compute_jacobian(
    model: nn.Module, 
    x: torch.Tensor,
    input_seq: Optional[torch.Tensor] = None,
    method: str = "autograd"
) -> torch.Tensor:
    """
    Compute the Jacobian of the RNN hidden state transition function.
    
    Computes ∂f(x)/∂x where f is the RNN transition function mapping
    hidden state to next hidden state.
    
    Args:
        model: SelfModel instance
        x: Hidden state at which to compute Jacobian, shape (hidden_dim,) or (batch, hidden_dim)
        input_seq: Optional input sequence, shape (batch, seq_len, input_dim)
        method: "autograd" or "manual" for Jacobian computation
        
    Returns:
        Jacobian matrix of shape (hidden_dim, hidden_dim)
    """
    model.eval()
    
    # Ensure x is 2D (batch, hidden_dim)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    hidden_dim = x.shape[1]
    batch_size = x.shape[0]
    
    # Get the transition function
    transition_fn = get_rnn_hidden_transition(model)
    
    if method == "autograd":
        # Use torch.autograd.functional.jacobian
        def jacobian_fn(x_flat):
            """Wrapper for jacobian computation."""
            x_reshaped = x_flat.reshape(batch_size, hidden_dim)
            output = transition_fn(x_reshaped, input_seq)
            # Return only first sample's output for jacobian
            return output[0]  # (hidden_dim,)
        
        # Compute jacobian for first sample
        x_flat = x[0].detach().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(jacobian_fn, x_flat)
        
    else:
        # Manual computation using gradient
        x_grad = x[0].detach().requires_grad_(True)
        
        # Compute output
        output = transition_fn(x_grad, input_seq)
        
        # Compute Jacobian column by column
        jacobian = torch.zeros(hidden_dim, hidden_dim, device=x.device)
        
        for i in range(hidden_dim):
            # Compute gradient of output[i] with respect to input
            grad = torch.autograd.grad(
                outputs=output[0, i],
                inputs=x_grad,
                retain_graph=True,
                create_graph=False
            )[0]
            jacobian[i] = grad
    
    return jacobian


def spectral_radius(jacobian: torch.Tensor) -> float:
    """
    Compute the spectral radius (largest singular value) of a Jacobian matrix.
    
    The spectral radius is the key stability metric:
    - σ_max < 1: contracting dynamics
    - σ_max > 1: expanding dynamics
    - σ_max ≈ 1: near-critical (marginally stable)
    
    Args:
        jacobian: Jacobian matrix, shape (hidden_dim, hidden_dim)
        
    Returns:
        Largest singular value (spectral radius)
    """
    # Compute singular values using SVD
    # torch.svd returns U, S, V where S contains singular values
    _, singular_values, _ = torch.linalg.svd(jacobian)
    
    # Largest singular value is the spectral radius
    spectral_rad = singular_values[0].item()
    
    return spectral_rad


def find_equilibrium(
    model: nn.Module,
    init: Optional[torch.Tensor] = None,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    device: str = "cpu"
) -> Tuple[torch.Tensor, float]:
    """
    Find a fixed point (equilibrium) of the RNN transition function.
    
    Uses gradient descent to find x* where f(x*) ≈ x*, i.e., the residual
    ||f(x*) - x*|| is minimized.
    
    Args:
        model: SelfModel instance
        init: Initial hidden state, shape (hidden_dim,). If None, uses random init
        n_iterations: Maximum number of gradient descent iterations
        learning_rate: Learning rate for gradient descent
        tolerance: Convergence tolerance for residual norm
        device: Device to run computation on
        
    Returns:
        Tuple of (equilibrium state, final residual norm)
    """
    model.eval()
    
    hidden_dim = model.rnn.hidden_size
    
    # Initialize if not provided
    if init is None:
        init = torch.randn(hidden_dim, device=device) * 0.1
    
    # Make it learnable
    x = init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=learning_rate)
    
    transition_fn = get_rnn_hidden_transition(model)
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Compute next hidden state
        next_hidden = transition_fn(x.unsqueeze(0)).squeeze(0)
        
        # Compute residual (fixed point error)
        residual = next_hidden - x
        loss = torch.sum(residual ** 2)
        
        # Check convergence
        residual_norm = torch.norm(residual).item()
        
        if residual_norm < tolerance:
            break
        
        # Backpropagate
        loss.backward()
        optimizer.step()
    
    final_residual = torch.norm(transition_fn(x.unsqueeze(0)).squeeze(0) - x).item()
    
    return x.detach(), final_residual


def classify_stability(spectral_radius_val: float) -> str:
    """
    Classify stability based on spectral radius.
    
    Args:
        spectral_radius_val: The largest singular value of the Jacobian
        
    Returns:
        Stability classification string:
        - "contracting": σ_max < 0.95 (stable, trajectories converge)
        - "near-critical": 0.95 ≤ σ_max < 1.05 (marginally stable)
        - "explosive": σ_max ≥ 1.05 (unstable, trajectories diverge)
    """
    if spectral_radius_val < 0.95:
        return "contracting"
    elif spectral_radius_val < 1.05:
        return "near-critical"
    else:
        return "explosive"


def compute_stability_statistics(
    model: nn.Module,
    dataset: str = "ar1",
    n_samples: int = 100,
    device: str = "cpu",
    seed: int = 42
) -> dict:
    """
    Compute stability statistics over multiple samples from a dataset.
    
    This function:
    1. Loads samples from the specified dataset
    2. Finds equilibrium points
    3. Computes Jacobian at each equilibrium
    4. Computes spectral radius statistics
    
    Args:
        model: SelfModel instance
        dataset: Dataset name ("ar1", "damped", "vanderpol")
        n_samples: Number of samples to analyze
        device: Device to run computation on
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with stability statistics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model.eval()
    model = model.to(device)
    
    hidden_dim = model.rnn.hidden_size
    
    # Load dataset
    data_path = f"data/deterministic/{dataset}_test.pt"
    if not os.path.exists(data_path):
        # Try train set
        data_path = f"data/deterministic/{dataset}_train.pt"
    
    if os.path.exists(data_path):
        data = torch.load(data_path, map_location=device)
        # Take random samples
        if len(data) > n_samples:
            indices = torch.randperm(len(data))[:n_samples]
            samples = data[indices]
        else:
            samples = data
    else:
        # Generate random samples if data not found
        samples = torch.randn(n_samples, 10, hidden_dim, device=device) * 0.5
    
    spectral_radii = []
    equilibrium_states = []
    residuals = []
    
    print(f"Computing stability metrics for {n_samples} samples from {dataset}...")
    
    # Get random initializations for equilibrium search
    torch.manual_seed(seed)
    init_guesses = torch.randn(min(n_samples, len(samples)), hidden_dim, device=device) * 0.5
    
    for i in range(min(n_samples, len(samples))):
        # Use learned initializations for equilibrium search
        init_hidden = init_guesses[i]
        
        # Find equilibrium starting from this point
        eq_state, residual = find_equilibrium(
            model, 
            init=init_hidden,
            n_iterations=100,
            device=device
        )
        
        # Compute Jacobian at equilibrium
        jacobian = compute_jacobian(model, eq_state.unsqueeze(0))
        
        # Compute spectral radius
        sr = spectral_radius(jacobian)
        
        spectral_radii.append(sr)
        equilibrium_states.append(eq_state.cpu())
        residuals.append(residual)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{min(n_samples, len(samples))} samples")
    
    spectral_radii = np.array(spectral_radii)
    residuals = np.array(residuals)
    
    # Compute statistics
    mean_sr = float(np.mean(spectral_radii))
    std_sr = float(np.std(spectral_radii))
    min_sr = float(np.min(spectral_radii))
    max_sr = float(np.max(spectral_radii))
    
    # Classify stability
    stability_class = classify_stability(mean_sr)
    
    # Count classifications
    classifications = [classify_stability(sr) for sr in spectral_radii]
    class_counts = {
        "contracting": sum(1 for c in classifications if c == "contracting"),
        "near-critical": sum(1 for c in classifications if c == "near-critical"),
        "explosive": sum(1 for c in classifications if c == "explosive"),
    }
    
    return {
        "dataset": dataset,
        "n_samples": len(spectral_radii),
        "spectral_radius_mean": mean_sr,
        "spectral_radius_std": std_sr,
        "spectral_radius_min": min_sr,
        "spectral_radius_max": max_sr,
        "stability_classification": stability_class,
        "classification_counts": class_counts,
        "equilibrium_residual_mean": float(np.mean(residuals)),
        "equilibrium_residual_std": float(np.std(residuals)),
    }


def load_model(model_path: str, device: str = "cpu") -> SelfModel:
    """
    Load a trained SelfModel from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded SelfModel instance
    """
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            # Check if it's a full checkpoint with model_state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            # Direct state dict
            state_dict = checkpoint
        
        # Infer dimensions from state dict
        hidden_dim = state_dict['rnn.weight_hh_l0'].shape[0]
        input_dim = state_dict['rnn.weight_ih_l0'].shape[1]
        output_dim = state_dict['fc.weight'].shape[0]
        
        # Create model with correct dimensions
        model = SelfModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        
        # Load state dict
        try:
            model.load_state_dict(state_dict)
            print(f"Loaded model with: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")
            print("Using random weights with inferred dimensions")
    else:
        # Create a new model with default dimensions if checkpoint not found
        # This allows testing even without trained models
        print(f"Warning: Model not found at {model_path}, using default dimensions")
        hidden_dim = 64
        input_dim = 2
        output_dim = 2
        model = SelfModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    return model.to(device)


def main():
    """CLI interface for Jacobian spectral analysis."""
    parser = argparse.ArgumentParser(
        description="Jacobian spectral analysis for self-model stability measurement"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="self_model.pth",
        help="Path to trained self-model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ar1", "damped", "vanderpol"],
        default="ar1",
        help="Which dataset to use for analysis"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to compute Jacobian statistics over"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run computation on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Jacobian Spectral Analysis for Self-Model Stability")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("-" * 60)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device=args.device)
    print(f"Model loaded: input_dim={model.rnn.input_size}, "
          f"hidden_dim={model.rnn.hidden_size}, "
          f"output_dim={model.fc.out_features}")
    
    # Compute stability statistics
    results = compute_stability_statistics(
        model,
        dataset=args.dataset,
        n_samples=args.n_samples,
        device=args.device,
        seed=args.seed
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: {results['dataset']}")
    print(f"Samples analyzed: {results['n_samples']}")
    print(f"\nSpectral Radius Statistics:")
    print(f"  Mean: {results['spectral_radius_mean']:.6f}")
    print(f"  Std:  {results['spectral_radius_std']:.6f}")
    print(f"  Min:  {results['spectral_radius_min']:.6f}")
    print(f"  Max:  {results['spectral_radius_max']:.6f}")
    print(f"\nStability Classification: {results['stability_classification']}")
    print(f"Classification counts:")
    print(f"  Contracting: {results['classification_counts']['contracting']}")
    print(f"  Near-critical: {results['classification_counts']['near-critical']}")
    print(f"  Explosive: {results['classification_counts']['explosive']}")
    print(f"\nEquilibrium residual:")
    print(f"  Mean: {results['equilibrium_residual_mean']:.6f}")
    print(f"  Std:  {results['equilibrium_residual_std']:.6f}")
    print("=" * 60)
    
    # Save results if output path specified
    if args.output:
        import json
        # Convert numpy types to Python types for JSON serialization
        results_json = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in results.items()
        }
        with open(args.output, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()