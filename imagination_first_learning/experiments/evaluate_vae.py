# Evaluation script for the Variational Autoencoder (VAE) in Imagination-First Learning

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from models.vae import VAE
from vae_experiment_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    LEGACY_DATA_PATH,
    PACKAGE_DATA_PATH,
    get_eval_config_defaults,
)



def parse_args():
    bootstrap_parser = ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    parser = ArgumentParser(description="Evaluate a VAE checkpoint on reconstruction metrics.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=bootstrap_args.config_path,
        help=f"JSON config file for experiment defaults (default: {bootstrap_args.config_path})",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to VAE checkpoint.")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to .npy evaluation data.")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device.",
    )
    parser.set_defaults(**get_eval_config_defaults(bootstrap_args.config_path))
    return parser.parse_args()


def resolve_data_path(path_arg):
    if path_arg is not None:
        return path_arg
    if PACKAGE_DATA_PATH.exists():
        return PACKAGE_DATA_PATH
    return LEGACY_DATA_PATH


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_numpy_data(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.load(data_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array [num_samples, input_dim], got shape {data.shape}")
    return torch.tensor(data, dtype=torch.float32)


def load_checkpoint(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint_obj = torch.load(model_path, map_location=device)

    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
        config = checkpoint_obj.get("config", {})
        metadata = checkpoint_obj
    elif isinstance(checkpoint_obj, dict):
        # Backward compatibility: legacy file saved with torch.save(vae.state_dict(), ...)
        state_dict = checkpoint_obj
        config = {}
        metadata = {"model_state_dict": state_dict}
    else:
        raise TypeError(f"Unsupported checkpoint format in {model_path}")

    return metadata, state_dict, config


def infer_dimensions(state_dict, config, data_tensor):
    input_dim = config.get("input_dim")
    latent_dim = config.get("latent_dim")

    if input_dim is None:
        input_dim = int(data_tensor.shape[1])
    if latent_dim is None:
        if "mu.weight" in state_dict:
            latent_dim = int(state_dict["mu.weight"].shape[0])
        elif "decoder.0.weight" in state_dict:
            latent_dim = int(state_dict["decoder.0.weight"].shape[1])
        else:
            raise KeyError("Could not infer latent_dim from checkpoint; missing expected keys.")

    if int(input_dim) != int(data_tensor.shape[1]):
        raise ValueError(
            f"Data input_dim ({data_tensor.shape[1]}) does not match checkpoint input_dim ({input_dim})."
        )

    return int(input_dim), int(latent_dim)


def evaluate_vae(model_path, data_path, batch_size=256, device_arg="auto"):
    """Evaluate the VAE model on reconstruction metrics."""
    device = get_device(device_arg)
    data_tensor = load_numpy_data(data_path)
    metadata, state_dict, config = load_checkpoint(model_path, device=device)
    input_dim, latent_dim = infer_dimensions(state_dict, config, data_tensor)

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    vae.load_state_dict(state_dict)
    vae.eval()

    dataloader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
    bce_compatible = bool(torch.all((data_tensor >= 0) & (data_tensor <= 1)).item())

    totals = {"samples": 0, "mse_sum": 0.0, "recon_bce_sum": 0.0, "kl_sum": 0.0}

    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)
            reconstructed, mu, log_var = vae(x)

            batch_size_actual = x.size(0)
            totals["samples"] += batch_size_actual
            totals["mse_sum"] += F.mse_loss(reconstructed, x, reduction="sum").item()

            if bce_compatible:
                recon_bce = F.binary_cross_entropy(reconstructed, x, reduction="sum")
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                totals["recon_bce_sum"] += recon_bce.item()
                totals["kl_sum"] += kl_div.item()

    samples = totals["samples"]
    mse_per_sample = totals["mse_sum"] / samples

    print(f"Model: {model_path}")
    print(f"Data: {data_path} (samples={samples}, input_dim={input_dim})")
    print(f"Device: {device}, latent_dim={latent_dim}")
    print(f"MSE reconstruction error per sample: {mse_per_sample:.6f}")

    if bce_compatible:
        recon_per_sample = totals["recon_bce_sum"] / samples
        kl_per_sample = totals["kl_sum"] / samples
        total_per_sample = (totals["recon_bce_sum"] + totals["kl_sum"]) / samples
        print(f"BCE recon per sample: {recon_per_sample:.6f}")
        print(f"KL divergence per sample: {kl_per_sample:.6f}")
        print(f"VAE loss per sample (BCE + KL): {total_per_sample:.6f}")
    else:
        print("Skipped BCE/KL metrics because data values are outside [0, 1].")

    training_meta = metadata.get("training") if isinstance(metadata, dict) else None
    if training_meta:
        print(f"Checkpoint training epochs: {training_meta.get('epochs')}")


def main():
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    evaluate_vae(
        model_path=args.model_path,
        data_path=data_path,
        batch_size=args.batch_size,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
