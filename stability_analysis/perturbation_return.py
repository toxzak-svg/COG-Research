"""
Perturbation Return Rate Analysis for Cognitive Architecture Research.

This module measures how quickly a trained model "absorbs" perturbations - returns to
baseline behavior after an intervention. This is distinct from reconstruction quality
and measures dynamical stability under intervention.

Key functionality:
- Perturbation injection (gaussian, uniform, adversarial)
- Rollout from perturbed initial conditions
- Return rate computation (how fast trajectories reconverge)
- Perturbation return time estimation (exponential decay fit)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minimal_self_model.models.self_model import SelfModel
from stability_analysis.jacobian_spectral import load_model as load_self_model, get_rnn_hidden_transition


def inject_perturbation(
    state: torch.Tensor,
    magnitude: float,
    noise_type: str = "gaussian",
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Add perturbations to initial state conditions.
    
    Args:
        state: Initial state tensor, shape (batch, state_dim) or (state_dim,)
        magnitude: Scale of perturbation (std for gaussian, range for uniform)
        noise_type: Type of noise ("gaussian", "uniform", "adversarial")
        seed: Optional random seed for reproducibility
        
    Returns:
        Perturbed state tensor of same shape as input
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Ensure 2D tensor (batch, dim)
    original_shape = state.shape
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    batch_size = state.shape[0]
    state_dim = state.shape[1]
    
    if noise_type == "gaussian":
        # Gaussian noise with std = magnitude
        noise = torch.randn_like(state) * magnitude
    
    elif noise_type == "uniform":
        # Uniform noise in [-magnitude, magnitude]
        noise = (torch.rand_like(state) * 2 - 1) * magnitude
    
    elif noise_type == "adversarial":
        # Adversarial perturbation: worst-case direction (maximize divergence)
        # Compute gradient of ||state - target|| w.r.t. state perturbation
        # For simplicity, use direction of maximum variance based on state magnitude
        with torch.no_grad():
            # Direction that pushes state away from origin
            direction = torch.sign(state)  # Unit direction away from origin
            noise = direction * magnitude
    
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Choose from: gaussian, uniform, adversarial")
    
    perturbed_state = state + noise
    
    # Restore original shape
    if len(original_shape) == 1:
        perturbed_state = perturbed_state.squeeze(0)
    
    return perturbed_state


def rollout_from_perturbed(
    model: nn.Module,
    initial_state: torch.Tensor,
    perturbation: torch.Tensor,
    horizon: int,
    use_rnn_hidden: bool = True,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Run model forward from perturbed initial state.
    
    Args:
        model: SelfModel instance
        initial_state: Original (unperturbed) initial state, shape (batch, state_dim)
        perturbation: Perturbation vector added to initial state
        horizon: Number of rollout steps
        use_rnn_hidden: If True, use RNN hidden state dynamics; else use output prediction
        device: Device to run computation on
        
    Returns:
        Trajectory tensor of shape (batch, horizon, state_dim)
    """
    model.eval()
    model = model.to(device)
    
    # Ensure initial_state is 2D
    if initial_state.dim() == 1:
        initial_state = initial_state.unsqueeze(0)
    if perturbation.dim() == 1:
        perturbation = perturbation.unsqueeze(0)
    
    batch_size = initial_state.shape[0]
    state_dim = initial_state.shape[1]
    
    # Compute perturbed initial state
    perturbed_state = initial_state + perturbation
    
    # Initialize trajectory storage
    trajectory = torch.zeros(batch_size, horizon, state_dim, device=device)
    
    if use_rnn_hidden:
        # Use RNN hidden state transition dynamics
        transition_fn = get_rnn_hidden_transition(model)
        
        # Initialize hidden state with perturbed state
        # Pad to match hidden dimension if needed
        hidden_dim = model.rnn.hidden_size
        
        # Create initial hidden state from perturbed observation
        # Use perturbed_state as input to initialize hidden state
        # PyTorch RNN expects hidden of shape (num_layers, batch, hidden_dim)
        num_layers = 1
        hidden = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        
        # First, run RNN to get initial hidden state from perturbed observation
        if state_dim <= model.rnn.input_size:
            # Pad or use directly
            input_tensor = perturbed_state.unsqueeze(1)  # (batch, seq=1, input_dim)
            if input_tensor.shape[2] < model.rnn.input_size:
                # Pad with zeros
                padding = torch.zeros(batch_size, 1, model.rnn.input_size - state_dim, device=device)
                input_tensor = torch.cat([input_tensor, padding], dim=2)
            
            with torch.no_grad():
                _, hidden = model.rnn(input_tensor, hidden)
        else:
            # Use first state_dim of hidden from perturbed_state
            hidden[0, :, :state_dim] = perturbed_state
        
        # Rollout
        for t in range(horizon):
            # Store current state (use hidden state projection or raw hidden)
            if model.fc.out_features == state_dim:
                # Project hidden to state space
                # hidden is (num_layers, batch, hidden_dim), take last layer
                current_state = model.fc(hidden[-1])  # (batch, output_dim)
            else:
                # Use raw hidden state - take last layer
                current_state = hidden[-1, :, :state_dim]  # (batch, state_dim)
            
            trajectory[:, t, :] = current_state
            
            # Compute next hidden state
            # transition_fn expects 2D input (batch, hidden_dim), so extract last layer
            next_hidden_2d = transition_fn(hidden[-1])  # (batch, hidden_dim)
            # Re-wrap to 3D (num_layers, batch, hidden_dim)
            hidden = next_hidden_2d.unsqueeze(0)
    else:
        # Use model output prediction (simpler, uses output head)
        current_state = perturbed_state
        
        for t in range(horizon):
            trajectory[:, t, :] = current_state
            
            # Prepare input for next step (use current prediction as input)
            input_tensor = current_state.unsqueeze(1)  # (batch, seq=1, state_dim)
            
            # If model input dim differs, pad or truncate
            if model.rnn.input_size != state_dim:
                if model.rnn.input_size > state_dim:
                    padding = torch.zeros(batch_size, 1, model.rnn.input_size - state_dim, device=device)
                    input_tensor = torch.cat([input_tensor, padding], dim=2)
                else:
                    input_tensor = input_tensor[:, :, :model.rnn.input_size]
            
            with torch.no_grad():
                rnn_out, _ = model.rnn(input_tensor)
                # Get prediction
                prediction = model.fc(rnn_out).squeeze(1)
                current_state = prediction
    
    return trajectory


def compute_divergence(
    perturbed_trajectory: torch.Tensor,
    baseline_trajectory: torch.Tensor,
    metric: str = "mse"
) -> np.ndarray:
    """
    Compute divergence between perturbed and baseline trajectories at each timestep.
    
    Args:
        perturbed_trajectory: Shape (batch, horizon, state_dim)
        baseline_trajectory: Shape (batch, horizon, state_dim)
        metric: "mse" for mean squared error, "relative_divergence" for relative error
        
    Returns:
        Array of shape (horizon,) with divergence at each timestep
    """
    if perturbed_trajectory.shape != baseline_trajectory.shape:
        raise ValueError("Trajectories must have same shape")
    
    batch_size, horizon, state_dim = perturbed_trajectory.shape
    
    divergences = []
    
    for t in range(horizon):
        p_state = perturbed_trajectory[:, t, :]
        b_state = baseline_trajectory[:, t, :]
        
        if metric == "mse":
            # MSE between states
            diff = p_state - b_state
            div = torch.mean(diff ** 2, dim=1)  # Per-sample MSE
            divergences.append(torch.mean(div).item())
        
        elif metric == "relative_divergence":
            # Relative divergence: |p - b| / |b|
            diff = torch.abs(p_state - b_state)
            norm_diff = torch.norm(diff, dim=1)  # Per-sample L2 norm
            norm_baseline = torch.norm(b_state, dim=1) + 1e-8  # Avoid division by zero
            rel_div = norm_diff / norm_baseline
            divergences.append(torch.mean(rel_div).item())
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return np.array(divergences)


def compute_return_rate(
    perturbed_trajectory: torch.Tensor,
    baseline_trajectory: torch.Tensor,
    metric: str = "mse"
) -> Dict[str, float]:
    """
    Measure how quickly perturbed trajectory converges back to baseline.
    
    Args:
        perturbed_trajectory: Shape (batch, horizon, state_dim)
        baseline_trajectory: Shape (batch, horizon, state_dim)
        metric: "mse" or "relative_divergence"
        
    Returns:
        Dictionary with return rate metrics:
        - mean_divergence: Average divergence across horizon
        - final_divergence: Divergence at final timestep
        - initial_divergence: Divergence at first timestep
        - decay_ratio: initial_divergence / final_divergence
    """
    divergences = compute_divergence(perturbed_trajectory, baseline_trajectory, metric)
    
    initial_div = float(divergences[0])
    final_div = float(divergences[-1])
    mean_div = float(np.mean(divergences))
    
    # Compute decay ratio (how much the divergence has decreased)
    decay_ratio = initial_div / (final_div + 1e-8) if final_div > 1e-8 else float('inf')
    
    # Compute return rate as slope of log-divergence over time
    # If divergence follows exp(-lambda*t), log(div) = log(A) - lambda*t
    # Return rate = lambda (higher = faster return)
    valid_divs = divergences[divergences > 1e-8]
    if len(valid_divs) >= 2:
        log_divs = np.log(valid_divs)
        t_indices = np.arange(len(valid_divs))
        # Linear fit: log(div) = a - lambda * t
        # lambda (return rate) is the negative slope
        coeffs = np.polyfit(t_indices, log_divs, 1)
        return_rate = -coeffs[0]  # Negative of slope
    else:
        return_rate = 0.0
    
    return {
        "mean_divergence": mean_div,
        "final_divergence": final_div,
        "initial_divergence": initial_div,
        "decay_ratio": decay_ratio,
        "return_rate": return_rate,
        "divergences": divergences.tolist()
    }


def estimate_return_time(divergence_by_horizon: np.ndarray) -> Dict[str, float]:
    """
    Estimate perturbation return time by fitting exponential decay.
    
    Fits: divergence(t) ≈ A * exp(-λ * t)
    
    Return time = 1/λ (time constant), lower = faster stabilization.
    
    Args:
        divergence_by_horizon: Array of divergence values at each timestep
        
    Returns:
        Dictionary with:
        - return_time: Estimated return time constant (1/lambda)
        - amplitude: Initial amplitude A
        - lambda_val: Decay rate λ
        - r_squared: Goodness of fit (R²)
    """
    # Filter out zero/negative values for log
    valid_mask = divergence_by_horizon > 1e-8
    if np.sum(valid_mask) < 2:
        return {
            "return_time": float('inf'),
            "amplitude": 0.0,
            "lambda_val": 0.0,
            "r_squared": 0.0
        }
    
    t = np.arange(len(divergence_by_horizon))[valid_mask]
    divs = divergence_by_horizon[valid_mask]
    
    # Log-linear fit: log(div) = log(A) - λ*t
    log_divs = np.log(divs)
    coeffs = np.polyfit(t, log_divs, 1)
    
    lambda_val = -coeffs[0]  # Decay rate (positive)
    log_amplitude = coeffs[1]
    amplitude = np.exp(log_amplitude)
    
    # Return time = 1/λ
    return_time = 1.0 / lambda_val if lambda_val > 0 else float('inf')
    
    # Compute R² for goodness of fit
    predicted_log_divs = coeffs[0] * t + coeffs[1]
    ss_res = np.sum((log_divs - predicted_log_divs) ** 2)
    ss_tot = np.sum((log_divs - np.mean(log_divs)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "return_time": return_time,
        "amplitude": amplitude,
        "lambda_val": lambda_val,
        "r_squared": r_squared
    }


def run_perturbation_experiment(
    model: nn.Module,
    dataset: str,
    n_trials: int = 100,
    perturbation_magnitude: float = 1.0,
    horizon: int = 50,
    noise_type: str = "gaussian",
    metric: str = "mse",
    seed: int = 42,
    device: str = "cpu"
) -> Dict:
    """
    Run perturbation return rate experiment across multiple trials.
    
    Args:
        model: SelfModel instance
        dataset: Dataset name ("ar1", "damped", "vanderpol")
        n_trials: Number of perturbation trials
        perturbation_magnitude: Scale of perturbation
        horizon: Rollout length
        noise_type: Type of perturbation noise
        metric: Divergence metric
        seed: Random seed
        device: Device to run on
        
    Returns:
        Dictionary with aggregated statistics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model.eval()
    model = model.to(device)
    
    # Load dataset
    data_path = f"data/deterministic/{dataset}_test.pt"
    if not os.path.exists(data_path):
        data_path = f"data/deterministic/{dataset}_train.pt"
    
    if os.path.exists(data_path):
        data = torch.load(data_path, map_location=device)
    else:
        # Generate synthetic data if file not found
        print(f"Warning: Dataset not found at {data_path}, generating synthetic data")
        if dataset == "ar1":
            from deterministic_data.deterministic_dataset import generate_ar1
            data = generate_ar1(n_samples=200, horizon=horizon, seed=seed, state_dim=2)
        elif dataset == "damped":
            from deterministic_data.deterministic_dataset import generate_damped_oscillator
            data = generate_damped_oscillator(n_samples=200, horizon=horizon, seed=seed)
        else:
            from deterministic_data.deterministic_dataset import generate_van_der_pol
            data = generate_van_der_pol(n_samples=200, horizon=horizon, seed=seed)
    
    # Get initial states from dataset
    n_samples = min(n_trials, len(data))
    initial_states = data[:n_samples, 0, :]  # First timestep of each trajectory
    
    # Storage for results
    return_times = []
    return_rates = []
    final_divergences = []
    decay_ratios = []
    all_divergences = []
    
    print(f"Running {n_trials} perturbation trials on {dataset}...")
    
    for trial in range(n_trials):
        # Get initial state for this trial
        trial_idx = trial % n_samples
        initial_state = initial_states[trial_idx:trial_idx+1].to(device)
        
        # Generate perturbation
        trial_seed = seed + trial if seed is not None else None
        perturbation = inject_perturbation(
            initial_state,
            magnitude=perturbation_magnitude,
            noise_type=noise_type,
            seed=trial_seed
        ) - initial_state
        
        # Compute baseline rollout (no perturbation)
        with torch.no_grad():
            baseline_trajectory = rollout_from_perturbed(
                model, initial_state, 
                perturbation=torch.zeros_like(perturbation),
                horizon=horizon,
                device=device
            )
        
        # Compute perturbed rollout
        with torch.no_grad():
            perturbed_trajectory = rollout_from_perturbed(
                model, initial_state,
                perturbation=perturbation,
                horizon=horizon,
                device=device
            )
        
        # Compute return rate
        result = compute_return_rate(perturbed_trajectory, baseline_trajectory, metric)
        
        # Estimate return time from divergence curve
        div_array = np.array(result["divergences"])
        return_time_result = estimate_return_time(div_array)
        
        return_times.append(return_time_result["return_time"])
        return_rates.append(result["return_rate"])
        final_divergences.append(result["final_divergence"])
        decay_ratios.append(result["decay_ratio"])
        all_divergences.append(div_array)
        
        if (trial + 1) % 20 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials")
    
    # Aggregate statistics
    return_times = np.array(return_times)
    return_rates = np.array(return_rates)
    final_divergences = np.array(final_divergences)
    decay_ratios = np.array(decay_ratios)
    
    # Filter out infinities for statistics
    valid_return_times = return_times[np.isfinite(return_times)]
    
    stats = {
        "dataset": dataset,
        "n_trials": n_trials,
        "perturbation_magnitude": perturbation_magnitude,
        "noise_type": noise_type,
        "horizon": horizon,
        "metric": metric,
        
        # Return time statistics
        "return_time_mean": float(np.mean(valid_return_times)) if len(valid_return_times) > 0 else float('inf'),
        "return_time_std": float(np.std(valid_return_times)) if len(valid_return_times) > 0 else 0.0,
        "return_time_median": float(np.median(valid_return_times)) if len(valid_return_times) > 0 else float('inf'),
        "return_time_min": float(np.min(valid_return_times)) if len(valid_return_times) > 0 else float('inf'),
        "return_time_max": float(np.max(valid_return_times)) if len(valid_return_times) > 0 else float('inf'),
        
        # Return rate statistics
        "return_rate_mean": float(np.mean(return_rates)),
        "return_rate_std": float(np.std(return_rates)),
        
        # Divergence statistics
        "final_divergence_mean": float(np.mean(final_divergences)),
        "final_divergence_std": float(np.std(final_divergences)),
        "decay_ratio_mean": float(np.mean(decay_ratios)),
        "decay_ratio_std": float(np.std(decay_ratios)),
        
        # Metadata
        "valid_trials": len(valid_return_times),
    }
    
    return stats


def main():
    """CLI interface for perturbation return rate measurement."""
    parser = argparse.ArgumentParser(
        description="Measure perturbation return rate for self-model stability"
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
        help="Which dataset to use for perturbation experiments"
    )
    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=1.0,
        help="Scale of perturbation (default: 1.0)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Rollout length (default: 50)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of perturbation trials (default: 100)"
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        choices=["gaussian", "uniform", "adversarial"],
        default="gaussian",
        help="Type of perturbation noise"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["mse", "relative_divergence"],
        default="mse",
        help="Divergence metric"
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
    print("Perturbation Return Rate Analysis")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Perturbation magnitude: {args.perturbation_magnitude}")
    print(f"Noise type: {args.noise_type}")
    print(f"Horizon: {args.horizon}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Metric: {args.metric}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("-" * 60)
    
    # Load model
    print("Loading model...")
    model = load_self_model(args.model_path, device=args.device)
    print(f"Model loaded: input_dim={model.rnn.input_size}, "
          f"hidden_dim={model.rnn.hidden_size}, "
          f"output_dim={model.fc.out_features}")
    
    # Run experiment
    results = run_perturbation_experiment(
        model=model,
        dataset=args.dataset,
        n_trials=args.n_trials,
        perturbation_magnitude=args.perturbation_magnitude,
        horizon=args.horizon,
        noise_type=args.noise_type,
        metric=args.metric,
        seed=args.seed,
        device=args.device
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: {results['dataset']}")
    print(f"Trials: {results['n_trials']}")
    print(f"Valid trials: {results['valid_trials']}")
    print(f"\nPerturbation Return Time:")
    print(f"  Mean: {results['return_time_mean']:.4f}")
    print(f"  Std:  {results['return_time_std']:.4f}")
    print(f"  Median: {results['return_time_median']:.4f}")
    print(f"  Min:  {results['return_time_min']:.4f}")
    print(f"  Max:  {results['return_time_max']:.4f}")
    print(f"\nReturn Rate:")
    print(f"  Mean: {results['return_rate_mean']:.6f}")
    print(f"  Std:  {results['return_rate_std']:.6f}")
    print(f"\nDivergence at Horizon:")
    print(f"  Mean: {results['final_divergence_mean']:.6f}")
    print(f"  Std:  {results['final_divergence_std']:.6f}")
    print(f"\nDecay Ratio:")
    print(f"  Mean: {results['decay_ratio_mean']:.4f}")
    print(f"  Std:  {results['decay_ratio_std']:.4f}")
    print("=" * 60)
    
    # Save results if output path specified
    if args.output:
        # Convert numpy types to Python types for JSON serialization
        results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in results.items() if k != "divergences"}
        with open(args.output, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
