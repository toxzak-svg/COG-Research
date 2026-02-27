# Script to generate synthetic data for Minimal Self-Model First Architectures

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "minimal_self_model" / "data" / "synthetic_sequential_data.npy"


def generate_synthetic_data(
    num_samples=1000,
    sequence_length=20,
    input_dim=5,
    seed=42,
    process="ar1",
    noise_std=0.03,
    alpha=0.9,
):
    """Generate synthetic sequential data for training the self-model.

    Processes:
    - iid: independent U[0,1] noise at every timestep (hard / mostly unlearnable for next-step prediction)
    - ar1: mean-reverting AR(1)-style dynamics with additive Gaussian noise (learnable)
    """
    rng = np.random.default_rng(seed)

    if process == "iid":
        return rng.random((num_samples, sequence_length, input_dim), dtype=np.float32)

    if process == "ar1":
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")

        data = np.empty((num_samples, sequence_length, input_dim), dtype=np.float32)

        # Sequence-specific attractor states make trajectories structured but diverse.
        attractor = rng.random((num_samples, input_dim), dtype=np.float32)
        data[:, 0, :] = rng.random((num_samples, input_dim), dtype=np.float32)

        # Weak coupling mixes dimensions so the model cannot solve each dim independently by copy.
        coupling = rng.normal(0.0, 0.03, size=(input_dim, input_dim)).astype(np.float32)
        np.fill_diagonal(coupling, 0.0)

        for t in range(1, sequence_length):
            prev = data[:, t - 1, :]
            coupled = prev @ coupling
            noise = rng.normal(0.0, noise_std, size=(num_samples, input_dim)).astype(np.float32)
            next_state = alpha * prev + (1.0 - alpha) * attractor + coupled + noise
            data[:, t, :] = np.clip(next_state, 0.0, 1.0)

        return data

    raise ValueError(f"Unknown process '{process}'. Expected one of: iid, ar1")


def parse_args():
    parser = ArgumentParser(description="Generate synthetic sequential data for the self-model prototype.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of sequences to generate.")
    parser.add_argument("--sequence-length", type=int, default=20, help="Length of each sequence.")
    parser.add_argument("--input-dim", type=int, default=5, help="Feature dimension for each time step.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--process",
        type=str,
        default="ar1",
        choices=("iid", "ar1"),
        help="Sequence generation process (default: ar1 for a learnable dynamics task).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.03,
        help="Gaussian noise std for ar1 process (ignored for iid).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="AR(1) persistence coefficient in [0, 1] (ignored for iid).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination .npy path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = generate_synthetic_data(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        input_dim=args.input_dim,
        seed=args.seed,
        process=args.process,
        noise_std=args.noise_std,
        alpha=args.alpha,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, data)
    print(
        "Synthetic sequential data saved to "
        f"{args.output_path} "
        f"(process={args.process}, seed={args.seed}, shape={tuple(data.shape)})"
    )
    if args.process == "iid":
        print("Note: iid U[0,1] data has an irreducible next-step MSE floor near 1/12 ~= 0.08333.")
