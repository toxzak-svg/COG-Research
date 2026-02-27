from argparse import ArgumentParser
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "minimal_self_model" / "data" / "synthetic_sequential_data.npy"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "minimal_self_model" / "data" / "synthetic_observations.npy"


def parse_args():
    parser = ArgumentParser(description="Flatten sequential data [N,T,D] into observation matrix [N*T,D].")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Source sequential .npy file.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output 2D observation .npy file.")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=("all", "inputs", "targets"),
        help="Which timesteps to include: all, x_t inputs ([:-1]), or x_(t+1) targets ([1:]).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    data = np.load(args.input_path)
    if data.ndim != 3:
        raise ValueError(f"Expected shape [N,T,D], got {data.shape}")
    if data.shape[1] < 2 and args.mode in {"inputs", "targets"}:
        raise ValueError("Need sequence length >= 2 for inputs/targets modes.")

    if args.mode == "all":
        obs = data.reshape(-1, data.shape[-1])
    elif args.mode == "inputs":
        obs = data[:, :-1, :].reshape(-1, data.shape[-1])
    elif args.mode == "targets":
        obs = data[:, 1:, :].reshape(-1, data.shape[-1])
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    obs = np.asarray(obs, dtype=np.float32)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, obs)
    print(
        f"Saved flattened observations to {args.output_path} "
        f"(mode={args.mode}, shape={tuple(obs.shape)}, from={args.input_path})"
    )


if __name__ == "__main__":
    main()
