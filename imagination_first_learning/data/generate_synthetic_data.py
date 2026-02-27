# Script to generate synthetic data for Imagination-First Learning

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "imagination_first_learning" / "data" / "synthetic_data.npy"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_data(
    num_samples=1000,
    input_dim=10,
    seed=42,
    process="latent_mixture",
    latent_dim=2,
    noise_std=0.05,
    num_clusters=4,
):
    """Generate synthetic data for training the VAE.

    Processes:
    - iid: independent U[0,1] noise per feature (unstructured; VAE likely collapses)
    - latent_mixture: low-dimensional clustered latent factors projected to observation space (learnable)
    """
    rng = np.random.default_rng(seed)

    if process == "iid":
        return rng.random((num_samples, input_dim), dtype=np.float32)

    if process == "latent_mixture":
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if num_clusters < 1:
            raise ValueError(f"num_clusters must be >= 1, got {num_clusters}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")

        cluster_centers = rng.normal(0.0, 1.25, size=(num_clusters, latent_dim)).astype(np.float32)
        cluster_ids = rng.integers(0, num_clusters, size=num_samples)
        z = cluster_centers[cluster_ids] + rng.normal(0.0, 0.25, size=(num_samples, latent_dim)).astype(np.float32)

        # Random nonlinear observation map; values are squashed to [0,1] for compatibility with Sigmoid decoder.
        W1 = rng.normal(0.0, 1.0, size=(latent_dim, max(16, input_dim))).astype(np.float32)
        b1 = rng.normal(0.0, 0.2, size=(max(16, input_dim),)).astype(np.float32)
        h = np.tanh(z @ W1 + b1)

        W2 = rng.normal(0.0, 0.8, size=(h.shape[1], input_dim)).astype(np.float32)
        b2 = rng.normal(0.0, 0.2, size=(input_dim,)).astype(np.float32)
        x = h @ W2 + b2
        x += rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        x = _sigmoid(x).astype(np.float32)
        return x

    raise ValueError(f"Unknown process '{process}'. Expected one of: iid, latent_mixture")


def parse_args():
    parser = ArgumentParser(description="Generate synthetic data for the VAE prototype.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--input-dim", type=int, default=10, help="Feature dimension for each sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--process",
        type=str,
        default="latent_mixture",
        choices=("iid", "latent_mixture"),
        help="Data generation process (default: latent_mixture for learnable structure).",
    )
    parser.add_argument("--latent-dim", type=int, default=2, help="Latent factor size for latent_mixture process.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Observation noise std for latent_mixture.")
    parser.add_argument("--num-clusters", type=int, default=4, help="Number of latent clusters for latent_mixture.")
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
        input_dim=args.input_dim,
        seed=args.seed,
        process=args.process,
        latent_dim=args.latent_dim,
        noise_std=args.noise_std,
        num_clusters=args.num_clusters,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, data)
    print(
        "Synthetic data saved to "
        f"{args.output_path} "
        f"(process={args.process}, seed={args.seed}, shape={tuple(data.shape)})"
    )
    if args.process == "iid":
        print("Note: iid U[0,1] data is unstructured and often causes VAE posterior collapse on this setup.")
