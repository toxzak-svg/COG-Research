"""Deterministic Dataset Generators for testing long-horizon rollout stability."""

import argparse
import os
from typing import Optional

import torch


def generate_ar1(
    n_samples: int,
    horizon: int,
    seed: int,
    state_dim: int = 2,
    rho: float = 0.9,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """
    Generate AR(1) mean-reverting trajectories.

    Formula: x_{t+1} = rho * x_t + noise

    Args:
        n_samples: Number of sample trajectories
        horizon: Number of timesteps per trajectory
        seed: Random seed for reproducibility
        state_dim: Dimension of state vector (1-4)
        rho: Autoregressive coefficient (0.9 for near-critical behavior)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Tensor of shape (n_samples, horizon, state_dim)
    """
    if not 1 <= state_dim <= 4:
        raise ValueError(f"state_dim must be between 1 and 4, got {state_dim}")

    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Initialize states
    x = torch.randn(n_samples, state_dim, generator=generator) * noise_std
    trajectory = torch.zeros(n_samples, horizon, state_dim)
    trajectory[:, 0, :] = x

    # Generate trajectory
    for t in range(1, horizon):
        noise = torch.randn(n_samples, state_dim, generator=generator) * noise_std
        x = rho * x + noise
        trajectory[:, t, :] = x

    return trajectory


def generate_damped_oscillator(
    n_samples: int,
    horizon: int,
    seed: int,
    zeta: float = 0.1,
    omega: float = 2.0,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Generate damped oscillator trajectories.

    Dynamics: ẍ + 2*zeta*omega*ẋ + omega^2*x = 0
    State: [x, ẋ] (2D)

    Args:
        n_samples: Number of sample trajectories
        horizon: Number of timesteps per trajectory
        seed: Random seed for reproducibility
        zeta: Damping ratio
        omega: Natural frequency
        noise_std: Standard deviation of process noise

    Returns:
        Tensor of shape (n_samples, horizon, 2)
    """
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    dt = 0.05  # Time step

    # Initialize states [x, v]
    x = torch.randn(n_samples, 2, generator=generator)
    x = x * 0.5  # Scale initial conditions

    trajectory = torch.zeros(n_samples, horizon, 2)
    trajectory[:, 0, :] = x

    # Damped oscillator: dx/dt = v, dv/dt = -2*zeta*omega*v - omega^2*x
    for t in range(1, horizon):
        v = x[:, 1]
        dx = v
        dv = -2 * zeta * omega * v - omega**2 * x[:, 0]

        # Euler step
        x_new = torch.zeros(n_samples, 2)
        x_new[:, 0] = x[:, 0] + dt * dx
        x_new[:, 1] = x[:, 1] + dt * dv

        # Add small noise for stochasticity
        noise = torch.randn(n_samples, 2, generator=generator) * noise_std
        x = x_new + noise
        trajectory[:, t, :] = x

    return trajectory


def generate_van_der_pol(
    n_samples: int,
    horizon: int,
    seed: int,
    mu: float = 1.0,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Generate Van der Pol oscillator trajectories (limit cycle).

    Dynamics: ẍ - mu*(1 - x²)*ẋ + x = 0
    State: [x, ẋ] (2D)

    Args:
        n_samples: Number of sample trajectories
        horizon: Number of timesteps per trajectory
        seed: Random seed for reproducibility
        mu: Damping parameter (non-zero gives limit cycle)
        noise_std: Standard deviation of process noise

    Returns:
        Tensor of shape (n_samples, horizon, 2)
    """
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    dt = 0.02  # Small time step for stability

    # Initialize states [x, v]
    # Sample from different radii to show convergence to limit cycle
    radii = torch.rand(n_samples, generator=generator) * 2 + 0.5
    angles = torch.rand(n_samples, generator=generator) * 2 * torch.pi

    x = torch.zeros(n_samples, 2)
    x[:, 0] = radii * torch.cos(angles)  # position
    x[:, 1] = radii * torch.sin(angles)  # velocity

    trajectory = torch.zeros(n_samples, horizon, 2)

    # Compute trajectory at the given horizon resolution
    step_interval = max(1, int(0.05 / dt))  # Resample to ~0.05 intervals
    actual_horizon = horizon * step_interval

    for t in range(actual_horizon):
        # Van der Pol: dx/dt = v, dv/dt = mu*(1-x²)*v - x
        dx = x[:, 1]
        dv = mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]

        # Euler step
        x = x + dt * torch.stack([dx, dv], dim=1)

        # Add small noise
        noise = torch.randn(n_samples, 2, generator=generator) * noise_std
        x = x + noise

        # Record at specified intervals
        if t % step_interval == 0:
            trajectory[:, t // step_interval, :] = x

    return trajectory


def split_data(
    data: torch.Tensor,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split data into train/val/test sets."""
    torch.manual_seed(seed)
    indices = torch.randperm(len(data))

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size : train_size + val_size + test_size]

    return data[train_idx], data[val_idx], data[test_idx]


def main():
    """CLI interface for generating deterministic datasets."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic datasets for testing rollout stability"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train-size", type=int, default=1000, help="Training set size"
    )
    parser.add_argument("--val-size", type=int, default=200, help="Validation set size")
    parser.add_argument("--test-size", type=int, default=200, help="Test set size")
    parser.add_argument(
        "--horizon", type=int, default=50, help="Trajectory horizon length"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/deterministic",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ar1", "damped", "vanderpol", "both"],
        default="both",
        help="Dataset type(s) to generate",
    )
    parser.add_argument(
        "--ar1-state-dim",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="State dimension for AR(1) dataset",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    total_size = args.train_size + args.val_size + args.test_size

    if args.dataset in ["ar1", "both"]:
        print(f"Generating AR(1) dataset with state_dim={args.ar1_state_dim}...")
        data = generate_ar1(
            n_samples=total_size,
            horizon=args.horizon,
            seed=args.seed,
            state_dim=args.ar1_state_dim,
            rho=0.9,
        )
        train, val, test = split_data(data, args.train_size, args.val_size, args.test_size, args.seed)

        torch.save(train, os.path.join(args.output_dir, "ar1_train.pt"))
        torch.save(val, os.path.join(args.output_dir, "ar1_val.pt"))
        torch.save(test, os.path.join(args.output_dir, "ar1_test.pt"))
        print(f"  Saved ar1_train.pt, ar1_val.pt, ar1_test.pt")

    if args.dataset in ["damped", "both"]:
        print(f"Generating damped oscillator dataset...")
        data = generate_damped_oscillator(
            n_samples=total_size,
            horizon=args.horizon,
            seed=args.seed,
        )
        train, val, test = split_data(data, args.train_size, args.val_size, args.test_size, args.seed)

        torch.save(train, os.path.join(args.output_dir, "damped_train.pt"))
        torch.save(val, os.path.join(args.output_dir, "damped_val.pt"))
        torch.save(test, os.path.join(args.output_dir, "damped_test.pt"))
        print(f"  Saved damped_train.pt, damped_val.pt, damped_test.pt")

    if args.dataset == "vanderpol":
        print(f"Generating Van der Pol oscillator dataset...")
        data = generate_van_der_pol(
            n_samples=total_size,
            horizon=args.horizon,
            seed=args.seed,
        )
        train, val, test = split_data(data, args.train_size, args.val_size, args.test_size, args.seed)

        torch.save(train, os.path.join(args.output_dir, "vanderpol_train.pt"))
        torch.save(val, os.path.join(args.output_dir, "vanderpol_val.pt"))
        torch.save(test, os.path.join(args.output_dir, "vanderpol_test.pt"))
        print(f"  Saved vanderpol_train.pt, vanderpol_val.pt, vanderpol_test.pt")

    print(f"Done! Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
