# Training script for the Variational Autoencoder (VAE) in Imagination-First Learning

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from models.vae import VAE
from vae_experiment_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    LEGACY_DATA_PATH,
    PACKAGE_DATA_PATH,
    get_train_config_defaults,
)



def parse_args():
    bootstrap_parser = ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    parser = ArgumentParser(description="Train a simple VAE on synthetic data.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=bootstrap_args.config_path,
        help=f"JSON config file for experiment defaults (default: {bootstrap_args.config_path})",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Path to .npy training data.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_MODEL_PATH, help="Checkpoint output path.")
    parser.add_argument("--latent-dim", type=int, default=2, help="Latent dimension size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction [0, 1).")
    parser.add_argument(
        "--recon-loss",
        type=str,
        default="mse",
        choices=("mse", "bce"),
        help="Reconstruction loss for continuous data (default: mse).",
    )
    parser.add_argument("--beta", type=float, default=1.0, help="KL coefficient (beta-VAE style).")
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=20,
        help="Linearly ramp KL weight from 0 to beta over this many epochs (0 disables warmup).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.set_defaults(**get_train_config_defaults(bootstrap_args.config_path))
    return parser.parse_args()


def resolve_data_path(path_arg):
    if path_arg is not None:
        return path_arg
    if PACKAGE_DATA_PATH.exists():
        return PACKAGE_DATA_PATH
    return LEGACY_DATA_PATH


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_tensor_dataset(data_path):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            f"Generate it with imagination_first_learning/data/generate_synthetic_data.py"
        )

    data = np.load(data_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array [num_samples, input_dim], got shape {data.shape}")
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    data_tensor = torch.tensor(data, dtype=torch.float32)
    return data_tensor


def split_dataset(dataset, val_fraction, seed):
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in [0, 1), got {val_fraction}")

    dataset_size = len(dataset)
    if dataset_size < 2 or val_fraction == 0.0:
        return dataset, None

    val_size = max(1, int(dataset_size * val_fraction))
    train_size = dataset_size - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no training samples.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset


def compute_batch_losses_with_config(recon_x, x, mu, log_var, recon_loss_type="mse", kl_weight=1.0):
    if recon_loss_type == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    elif recon_loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    else:
        raise ValueError(f"Unsupported recon loss: {recon_loss_type}")

    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + (kl_weight * kl_div)
    return total_loss, recon_loss, kl_div


def run_epoch(model, dataloader, device, optimizer=None, recon_loss_type="mse", kl_weight=1.0):
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "samples": 0}

    for (x,) in dataloader:
        x = x.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            recon_x, mu, log_var = model(x)
            loss, recon_loss, kl_div = compute_batch_losses_with_config(
                recon_x,
                x,
                mu,
                log_var,
                recon_loss_type=recon_loss_type,
                kl_weight=kl_weight,
            )

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        totals["loss"] += loss.item()
        totals["recon"] += recon_loss.item()
        totals["kl"] += kl_div.item()
        totals["samples"] += batch_size

    if totals["samples"] == 0:
        raise RuntimeError("Empty dataloader; cannot compute metrics.")

    samples = totals["samples"]
    return {
        "loss_per_sample": totals["loss"] / samples,
        "recon_per_sample": totals["recon"] / samples,
        "kl_per_sample": totals["kl"] / samples,
        "kl_weight": float(kl_weight),
        "samples": samples,
    }


def get_kl_weight(epoch_index, beta, kl_warmup_epochs):
    if kl_warmup_epochs <= 0:
        return float(beta)
    progress = min(1.0, float(epoch_index + 1) / float(kl_warmup_epochs))
    return float(beta) * progress


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    data_path = resolve_data_path(args.data_path)

    data_tensor = load_tensor_dataset(data_path)
    if torch.any((data_tensor < 0) | (data_tensor > 1)):
        raise ValueError("VAE decoder output is sigmoid; expected input data scaled to [0, 1].")

    dataset = TensorDataset(data_tensor)
    train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    input_dim = data_tensor.shape[1]
    vae = VAE(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    print(f"Training VAE on {data_path} (samples={len(dataset)}, input_dim={input_dim})")
    print(
        f"Device: {device}, latent_dim={args.latent_dim}, epochs={args.epochs}, "
        f"recon_loss={args.recon_loss}, beta={args.beta}, kl_warmup_epochs={args.kl_warmup_epochs}"
    )

    last_train_metrics = None
    last_val_metrics = None
    for epoch in range(args.epochs):
        kl_weight = get_kl_weight(epoch, beta=args.beta, kl_warmup_epochs=args.kl_warmup_epochs)
        last_train_metrics = run_epoch(
            vae,
            train_loader,
            device=device,
            optimizer=optimizer,
            recon_loss_type=args.recon_loss,
            kl_weight=kl_weight,
        )
        if val_loader is not None:
            last_val_metrics = run_epoch(
                vae,
                val_loader,
                device=device,
                optimizer=None,
                recon_loss_type=args.recon_loss,
                kl_weight=kl_weight,
            )
            print(
                f"Epoch {epoch + 1:03d} | "
                f"kl_w={kl_weight:.3f} | "
                f"train loss={last_train_metrics['loss_per_sample']:.4f} "
                f"(recon={last_train_metrics['recon_per_sample']:.4f}, kl={last_train_metrics['kl_per_sample']:.4f}) | "
                f"val loss={last_val_metrics['loss_per_sample']:.4f} "
                f"(recon={last_val_metrics['recon_per_sample']:.4f}, kl={last_val_metrics['kl_per_sample']:.4f})"
            )
        else:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"kl_w={kl_weight:.3f} | "
                f"train loss={last_train_metrics['loss_per_sample']:.4f} "
                f"(recon={last_train_metrics['recon_per_sample']:.4f}, kl={last_train_metrics['kl_per_sample']:.4f})"
            )

    checkpoint = {
        "model_state_dict": vae.state_dict(),
        "config": {
            "input_dim": int(input_dim),
            "latent_dim": int(args.latent_dim),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "recon_loss": str(args.recon_loss),
            "beta": float(args.beta),
            "kl_warmup_epochs": int(args.kl_warmup_epochs),
            "seed": int(args.seed),
            "val_fraction": float(args.val_fraction),
            "data_path": str(data_path),
            "last_train_metrics": last_train_metrics,
            "last_val_metrics": last_val_metrics,
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_path)
    print(f"Saved checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
