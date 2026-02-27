# Training script for the Minimal Self-Model in Minimal Self-Model First Architectures

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PACKAGE_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from models.self_model import SelfModel, self_model_loss
from imagination_first_learning.models.vae import VAE
from self_model_experiment_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LATENT_TRANSITION_PATH,
    DEFAULT_SELF_MODEL_PATH,
    DEFAULT_VAE_MODEL_PATH,
    LEGACY_DATA_PATH,
    PACKAGE_DATA_PATH,
    get_train_config_defaults,
)


def parse_args():
    bootstrap_parser = ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    parser = ArgumentParser(description="Train the minimal self-model and fit a VAE latent transition baseline.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=bootstrap_args.config_path,
        help=f"JSON config file for experiment defaults (default: {bootstrap_args.config_path})",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Path to sequential .npy training data.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_SELF_MODEL_PATH, help="Self-model checkpoint path.")
    parser.add_argument("--vae-model-path", type=Path, default=DEFAULT_VAE_MODEL_PATH, help="Pretrained VAE checkpoint.")
    parser.add_argument(
        "--latent-transition-path",
        type=Path,
        default=DEFAULT_LATENT_TRANSITION_PATH,
        help="Output path for fitted VAE latent transition (.npz).",
    )
    parser.add_argument("--hidden-dim", type=int, default=16, help="RNN hidden size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction [0, 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training/evaluation device.",
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


def load_sequential_data(data_path):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            f"Generate it with minimal_self_model/data/generate_synthetic_data.py"
        )
    data = np.load(data_path)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array [num_samples, seq_len, input_dim], got shape {data.shape}")
    if data.shape[1] < 2:
        raise ValueError("Sequence length must be at least 2 to form next-step targets.")
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    return torch.tensor(data, dtype=torch.float32)


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


def run_epoch(model, dataloader, device, optimizer=None):
    is_training = optimizer is not None
    model.train(mode=is_training)

    totals = {"mse_sum": 0.0, "sequences": 0, "elements": 0}
    for x, target in dataloader:
        x = x.to(device)
        target = target.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predicted = model(x)
            loss = self_model_loss(predicted, target)

        if is_training:
            loss.backward()
            optimizer.step()

        totals["mse_sum"] += F.mse_loss(predicted, target, reduction="sum").item()
        totals["sequences"] += x.size(0)
        totals["elements"] += target.numel()

    if totals["sequences"] == 0:
        raise RuntimeError("Empty dataloader; cannot compute metrics.")

    return {
        "mse_per_element": totals["mse_sum"] / totals["elements"],
        "mse_per_sequence": totals["mse_sum"] / totals["sequences"],
        "sequences": totals["sequences"],
    }


def load_checkpoint(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    checkpoint_obj = torch.load(model_path, map_location=device)

    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
        config = checkpoint_obj.get("config", {})
        metadata = checkpoint_obj
    elif isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
        config = {}
        metadata = {"model_state_dict": state_dict}
    else:
        raise TypeError(f"Unsupported checkpoint format in {model_path}")
    return metadata, state_dict, config


def infer_self_model_dims(state_dict, config):
    input_dim = config.get("input_dim")
    hidden_dim = config.get("hidden_dim")
    output_dim = config.get("output_dim")

    if input_dim is None or hidden_dim is None:
        if "rnn.weight_ih_l0" not in state_dict:
            raise KeyError("Could not infer self-model dims from checkpoint.")
        rnn_w = state_dict["rnn.weight_ih_l0"]
        hidden_dim = int(rnn_w.shape[0]) if hidden_dim is None else int(hidden_dim)
        input_dim = int(rnn_w.shape[1]) if input_dim is None else int(input_dim)
    if output_dim is None:
        if "fc.weight" not in state_dict:
            raise KeyError("Could not infer self-model output_dim from checkpoint.")
        output_dim = int(state_dict["fc.weight"].shape[0])

    return int(input_dim), int(hidden_dim), int(output_dim)


def infer_vae_dims(state_dict, config):
    input_dim = config.get("input_dim")
    latent_dim = config.get("latent_dim")

    if input_dim is None:
        if "encoder.0.weight" not in state_dict:
            raise KeyError("Could not infer VAE input_dim from checkpoint.")
        input_dim = int(state_dict["encoder.0.weight"].shape[1])
    if latent_dim is None:
        if "mu.weight" not in state_dict:
            raise KeyError("Could not infer VAE latent_dim from checkpoint.")
        latent_dim = int(state_dict["mu.weight"].shape[0])

    return int(input_dim), int(latent_dim)


def pad_feature_dim(x, target_dim):
    current_dim = x.shape[-1]
    if current_dim > target_dim:
        raise ValueError(
            f"Data feature dim ({current_dim}) exceeds VAE input_dim ({target_dim}); "
            "cannot zero-pad to match."
        )
    if current_dim == target_dim:
        return x
    pad_width = target_dim - current_dim
    return F.pad(x, (0, pad_width), mode="constant", value=0.0)


def encode_vae_mu_in_batches(vae_model, x_flat, batch_size, device):
    mus = []
    dataset = TensorDataset(x_flat)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            mu, _ = vae_model.encode(x_batch)
            mus.append(mu.detach().cpu())
    return torch.cat(mus, dim=0)


def fit_linear_latent_transition(vae_model, sequences, vae_input_dim, batch_size, device):
    padded = pad_feature_dim(sequences, vae_input_dim)
    x_t = padded[:, :-1, :].reshape(-1, vae_input_dim)
    x_t_plus_1 = padded[:, 1:, :].reshape(-1, vae_input_dim)

    z_t = encode_vae_mu_in_batches(vae_model, x_t, batch_size=batch_size, device=device).numpy()
    z_t_plus_1 = encode_vae_mu_in_batches(vae_model, x_t_plus_1, batch_size=batch_size, device=device).numpy()

    ones = np.ones((z_t.shape[0], 1), dtype=z_t.dtype)
    design = np.concatenate([z_t, ones], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(design, z_t_plus_1, rcond=None)

    latent_dim = z_t.shape[1]
    A = coeff[:latent_dim, :].astype(np.float32)
    b = coeff[latent_dim, :].astype(np.float32)

    z_pred = z_t @ A + b
    latent_mse = float(np.mean((z_pred - z_t_plus_1) ** 2))
    return A, b, latent_mse


def evaluate_vae_baseline_one_step(vae_model, dataloader, A, b, obs_dim, vae_input_dim, device):
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    b_t = torch.tensor(b, dtype=torch.float32, device=device)

    totals = {"mse_obs_sum": 0.0, "mse_full_sum": 0.0, "elements_obs": 0, "elements_full": 0}
    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)

            x_pad = pad_feature_dim(x, vae_input_dim)
            target_pad = pad_feature_dim(target, vae_input_dim)

            x_flat = x_pad.reshape(-1, vae_input_dim)
            target_flat = target_pad.reshape(-1, vae_input_dim)

            mu_t, _ = vae_model.encode(x_flat)
            z_pred = mu_t @ A_t + b_t
            x_pred_flat = vae_model.decode(z_pred)
            x_pred = x_pred_flat.reshape_as(target_pad)

            totals["mse_full_sum"] += F.mse_loss(x_pred, target_pad, reduction="sum").item()
            totals["elements_full"] += target_pad.numel()

            x_pred_obs = x_pred[..., :obs_dim]
            totals["mse_obs_sum"] += F.mse_loss(x_pred_obs, target, reduction="sum").item()
            totals["elements_obs"] += target.numel()

    return {
        "mse_per_element_obs_dim": totals["mse_obs_sum"] / totals["elements_obs"],
        "mse_per_element_full_dim": totals["mse_full_sum"] / totals["elements_full"],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    data_path = resolve_data_path(args.data_path)

    sequences = load_sequential_data(data_path)
    num_samples, sequence_length, obs_dim = sequences.shape

    inputs = sequences[:, :-1, :]
    targets = sequences[:, 1:, :]
    dataset = TensorDataset(inputs, targets)
    train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.seed)
    if hasattr(train_dataset, "indices"):
        train_indices = [int(i) for i in train_dataset.indices]
    else:
        train_indices = list(range(len(dataset)))
    if val_dataset is not None and hasattr(val_dataset, "indices"):
        val_indices = [int(i) for i in val_dataset.indices]
    else:
        val_indices = None

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

    self_model = SelfModel(input_dim=obs_dim, hidden_dim=args.hidden_dim, output_dim=obs_dim).to(device)
    optimizer = torch.optim.Adam(self_model.parameters(), lr=args.lr)

    print(
        f"Training SelfModel on {data_path} "
        f"(samples={num_samples}, seq_len={sequence_length}, obs_dim={obs_dim})"
    )
    print(f"Device: {device}, hidden_dim={args.hidden_dim}, epochs={args.epochs}")

    last_train_metrics = None
    last_val_metrics = None
    for epoch in range(args.epochs):
        last_train_metrics = run_epoch(self_model, train_loader, device=device, optimizer=optimizer)
        if val_loader is not None:
            last_val_metrics = run_epoch(self_model, val_loader, device=device, optimizer=None)
            print(
                f"Epoch {epoch + 1:03d} | "
                f"train mse/elem={last_train_metrics['mse_per_element']:.6f} | "
                f"val mse/elem={last_val_metrics['mse_per_element']:.6f}"
            )
        else:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"train mse/elem={last_train_metrics['mse_per_element']:.6f}"
            )

    eval_loader = val_loader or train_loader
    eval_split_name = "val" if val_loader is not None else "train"
    self_eval_metrics = run_epoch(self_model, eval_loader, device=device, optimizer=None)
    print(
        f"Self-model one-step MSE on {eval_split_name} split: "
        f"{self_eval_metrics['mse_per_element']:.6f} per element"
    )

    vae_meta, vae_state_dict, vae_config = load_checkpoint(args.vae_model_path, device=device)
    vae_input_dim, vae_latent_dim = infer_vae_dims(vae_state_dict, vae_config)
    vae_model = VAE(input_dim=vae_input_dim, latent_dim=vae_latent_dim).to(device)
    vae_model.load_state_dict(vae_state_dict)
    vae_model.eval()

    if obs_dim > vae_input_dim:
        raise ValueError(
            f"Self-model data obs_dim ({obs_dim}) exceeds VAE input_dim ({vae_input_dim}). "
            "Use a VAE trained on matching or larger feature dimension."
        )
    if torch.any((sequences < 0) | (sequences > 1)):
        print("Warning: sequential data contains values outside [0, 1]; VAE baseline metrics may degrade.")
    vae_training_meta = vae_meta.get("training") if isinstance(vae_meta, dict) else None
    if vae_training_meta:
        vae_data_path = str(vae_training_meta.get("data_path", ""))
        if "minimal_self_model" not in vae_data_path:
            print(
                "Warning: VAE checkpoint appears to be trained on a non-sequential/other dataset "
                f"({vae_data_path}). This baseline is not a like-for-like comparison."
            )
        if int(vae_input_dim) != int(obs_dim):
            print(
                "Warning: VAE input_dim does not match self-model observation dim "
                f"({vae_input_dim} vs {obs_dim}); zero-padding will be used for the VAE baseline."
            )

    A, b, latent_fit_mse = fit_linear_latent_transition(
        vae_model=vae_model,
        sequences=sequences,
        vae_input_dim=vae_input_dim,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"Fitted VAE latent transition (latent_dim={vae_latent_dim}); latent MSE={latent_fit_mse:.6f}")

    vae_baseline_metrics = evaluate_vae_baseline_one_step(
        vae_model=vae_model,
        dataloader=eval_loader,
        A=A,
        b=b,
        obs_dim=int(obs_dim),
        vae_input_dim=int(vae_input_dim),
        device=device,
    )
    print(
        f"VAE latent-transition one-step MSE on {eval_split_name} split "
        f"(obs dims only): {vae_baseline_metrics['mse_per_element_obs_dim']:.6f}"
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    self_checkpoint = {
        "model_state_dict": self_model.state_dict(),
        "config": {
            "input_dim": int(obs_dim),
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(obs_dim),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "val_fraction": float(args.val_fraction),
            "data_path": str(data_path),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "last_train_metrics": last_train_metrics,
            "last_val_metrics": last_val_metrics,
            "eval_split": eval_split_name,
            "eval_metrics": self_eval_metrics,
        },
        "baseline": {
            "vae_model_path": str(args.vae_model_path),
            "latent_transition_path": str(args.latent_transition_path),
            "vae_input_dim": int(vae_input_dim),
            "vae_latent_dim": int(vae_latent_dim),
            "latent_fit_mse": float(latent_fit_mse),
            "one_step_metrics": vae_baseline_metrics,
        },
    }
    torch.save(self_checkpoint, args.output_path)
    print(f"Saved self-model checkpoint to {args.output_path}")

    args.latent_transition_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.latent_transition_path,
        A=A,
        b=b,
        obs_dim=np.int64(obs_dim),
        vae_input_dim=np.int64(vae_input_dim),
        vae_latent_dim=np.int64(vae_latent_dim),
        latent_fit_mse=np.float32(latent_fit_mse),
    )
    print(f"Saved VAE latent transition to {args.latent_transition_path}")

    if isinstance(vae_meta, dict) and vae_meta.get("training"):
        epochs = vae_meta["training"].get("epochs")
        print(f"Loaded VAE checkpoint metadata: training epochs={epochs}")


if __name__ == "__main__":
    main()
