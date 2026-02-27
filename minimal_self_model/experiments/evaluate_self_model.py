# Evaluation script for the Minimal Self-Model in Minimal Self-Model First Architectures

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PACKAGE_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from models.self_model import SelfModel
from imagination_first_learning.models.vae import VAE
from self_model_experiment_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LATENT_TRANSITION_PATH,
    DEFAULT_SELF_MODEL_PATH,
    DEFAULT_VAE_MODEL_PATH,
    LEGACY_DATA_PATH,
    PACKAGE_DATA_PATH,
    get_eval_config_defaults,
)


def parse_args():
    bootstrap_parser = ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    parser = ArgumentParser(description="Evaluate self-model and VAE latent-transition baselines.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=bootstrap_args.config_path,
        help=f"JSON config file for experiment defaults (default: {bootstrap_args.config_path})",
    )
    parser.add_argument("--self-model-path", type=Path, default=DEFAULT_SELF_MODEL_PATH, help="Self-model checkpoint.")
    parser.add_argument("--vae-model-path", type=Path, default=DEFAULT_VAE_MODEL_PATH, help="VAE checkpoint.")
    parser.add_argument(
        "--latent-transition-path",
        type=Path,
        default=DEFAULT_LATENT_TRANSITION_PATH,
        help="Path to fitted VAE latent transition (.npz).",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Sequential .npy evaluation data.")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("all", "train", "val"),
        help="Which dataset split to evaluate. 'train'/'val' reuse checkpoint split metadata when available.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Rollout horizons to evaluate (positive integers).",
    )
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


def load_sequential_data(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.load(data_path)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array [num_samples, seq_len, input_dim], got shape {data.shape}")
    if data.shape[1] < 2:
        raise ValueError("Sequence length must be at least 2 to evaluate next-step prediction.")
    return torch.tensor(data, dtype=torch.float32)


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
    return F.pad(x, (0, target_dim - current_dim), mode="constant", value=0.0)


def load_latent_transition(latent_transition_path, expected_latent_dim):
    if not latent_transition_path.exists():
        raise FileNotFoundError(f"Latent transition file not found: {latent_transition_path}")
    latent = np.load(latent_transition_path)
    if "A" not in latent or "b" not in latent:
        raise KeyError(f"Expected keys 'A' and 'b' in {latent_transition_path}")

    A = np.asarray(latent["A"], dtype=np.float32)
    b = np.asarray(latent["b"], dtype=np.float32)
    if A.ndim != 2 or b.ndim != 1:
        raise ValueError(f"Invalid latent transition shapes: A={A.shape}, b={b.shape}")
    if A.shape[0] != expected_latent_dim or A.shape[1] != expected_latent_dim or b.shape[0] != expected_latent_dim:
        raise ValueError(
            "Latent transition dimensions do not match VAE latent_dim: "
            f"A={A.shape}, b={b.shape}, latent_dim={expected_latent_dim}"
        )

    metadata = {}
    for key in ("obs_dim", "vae_input_dim", "vae_latent_dim", "latent_fit_mse"):
        if key in latent:
            value = latent[key]
            metadata[key] = value.item() if np.ndim(value) == 0 else value
    return A, b, metadata


def select_sequences_split(sequences, split_name, checkpoint_meta):
    if split_name == "all":
        return sequences, "all"

    if not isinstance(checkpoint_meta, dict) or "training" not in checkpoint_meta:
        raise ValueError(
            f"Requested --split {split_name}, but self-model checkpoint lacks training metadata."
        )

    training_meta = checkpoint_meta["training"] or {}
    indices_key = f"{split_name}_indices"
    saved_indices = training_meta.get(indices_key)
    if saved_indices is not None:
        idx = torch.tensor(saved_indices, dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError(f"Checkpoint stored empty {indices_key}.")
        if int(torch.max(idx).item()) >= sequences.shape[0]:
            raise ValueError(
                f"Checkpoint {indices_key} references index outside current dataset "
                f"(max={int(torch.max(idx).item())}, dataset_size={sequences.shape[0]})."
            )
        return sequences.index_select(0, idx), f"{split_name} (checkpoint indices)"

    val_fraction = training_meta.get("val_fraction")
    seed = training_meta.get("seed")
    if val_fraction is None or seed is None:
        raise ValueError(
            f"Requested --split {split_name}, but checkpoint is missing val_fraction/seed and {indices_key}."
        )
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError(f"Invalid val_fraction in checkpoint metadata: {val_fraction}")

    dataset_size = sequences.shape[0]
    if split_name == "val" and (dataset_size < 2 or float(val_fraction) == 0.0):
        raise ValueError("Checkpoint training config has no validation split to evaluate.")

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(dataset_size, generator=generator)

    if dataset_size < 2 or float(val_fraction) == 0.0:
        train_idx = perm
        val_idx = torch.empty(0, dtype=torch.long)
    else:
        val_size = max(1, int(dataset_size * float(val_fraction)))
        train_size = dataset_size - val_size
        if train_size <= 0:
            raise ValueError("Checkpoint validation split leaves no training samples.")
        train_idx = perm[:train_size]
        val_idx = perm[train_size:]

    idx = train_idx if split_name == "train" else val_idx
    if idx.numel() == 0:
        raise ValueError(f"Selected split '{split_name}' is empty.")
    return sequences.index_select(0, idx), f"{split_name} (reconstructed from checkpoint seed)"


def evaluate_self_model_one_step(model, sequences, batch_size, device):
    inputs = sequences[:, :-1, :]
    targets = sequences[:, 1:, :]
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)

    total_mse_sum = 0.0
    total_elements = 0
    with torch.no_grad():
        for x, target in loader:
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            total_mse_sum += F.mse_loss(pred, target, reduction="sum").item()
            total_elements += target.numel()

    return total_mse_sum / total_elements


def evaluate_self_model_rollouts(model, sequences, horizons, batch_size, device):
    seq_len = sequences.shape[1]
    results = {}
    with torch.no_grad():
        for horizon in horizons:
            if horizon <= 0:
                raise ValueError(f"Horizons must be positive integers, got {horizon}")
            if horizon >= seq_len:
                results[horizon] = None
                continue

            start_states = sequences[:, : seq_len - horizon, :].reshape(-1, sequences.shape[-1])
            targets = sequences[:, horizon:, :].reshape(-1, sequences.shape[-1])
            loader = DataLoader(TensorDataset(start_states, targets), batch_size=batch_size, shuffle=False)

            mse_sum = 0.0
            elements = 0
            for start_batch, target_batch in loader:
                current = start_batch.to(device)
                target_batch = target_batch.to(device)

                for _ in range(horizon):
                    current = model(current.unsqueeze(1)).squeeze(1)

                mse_sum += F.mse_loss(current, target_batch, reduction="sum").item()
                elements += target_batch.numel()

            results[horizon] = mse_sum / elements
    return results


def evaluate_vae_rollouts(vae_model, A, b, sequences, obs_dim, vae_input_dim, horizons, batch_size, device):
    seq_len = sequences.shape[1]
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    b_t = torch.tensor(b, dtype=torch.float32, device=device)
    results = {}

    with torch.no_grad():
        for horizon in horizons:
            if horizon <= 0:
                raise ValueError(f"Horizons must be positive integers, got {horizon}")
            if horizon >= seq_len:
                results[horizon] = None
                continue

            start_states = sequences[:, : seq_len - horizon, :].reshape(-1, obs_dim)
            targets = sequences[:, horizon:, :].reshape(-1, obs_dim)
            loader = DataLoader(TensorDataset(start_states, targets), batch_size=batch_size, shuffle=False)

            mse_sum = 0.0
            elements = 0
            for start_batch, target_batch in loader:
                start_batch = start_batch.to(device)
                target_batch = target_batch.to(device)
                start_pad = pad_feature_dim(start_batch, vae_input_dim)

                mu, _ = vae_model.encode(start_pad)
                z = mu
                for _ in range(horizon):
                    z = z @ A_t + b_t

                pred_pad = vae_model.decode(z)
                pred = pred_pad[:, :obs_dim]

                mse_sum += F.mse_loss(pred, target_batch, reduction="sum").item()
                elements += target_batch.numel()

            results[horizon] = mse_sum / elements

    return results


def format_rollout_results(name, results):
    print(f"{name} rollout MSE per element:")
    for horizon, mse in results.items():
        if mse is None:
            print(f"  Horizon {horizon}: skipped (horizon >= sequence length)")
        else:
            print(f"  Horizon {horizon}: {mse:.6f}")


def main():
    args = parse_args()
    device = get_device(args.device)
    data_path = resolve_data_path(args.data_path)
    horizons = [int(h) for h in args.horizons]

    sequences_all = load_sequential_data(data_path)
    _, seq_len_all, obs_dim_all = sequences_all.shape

    self_meta, self_state_dict, self_config = load_checkpoint(args.self_model_path, device=device)
    self_input_dim, self_hidden_dim, self_output_dim = infer_self_model_dims(self_state_dict, self_config)
    if self_input_dim != obs_dim_all or self_output_dim != obs_dim_all:
        raise ValueError(
            f"Self-model dims ({self_input_dim}->{self_output_dim}) do not match data obs_dim ({obs_dim_all})."
        )
    self_model = SelfModel(
        input_dim=self_input_dim,
        hidden_dim=self_hidden_dim,
        output_dim=self_output_dim,
    ).to(device)
    self_model.load_state_dict(self_state_dict)
    self_model.eval()

    vae_meta, vae_state_dict, vae_config = load_checkpoint(args.vae_model_path, device=device)
    vae_input_dim, vae_latent_dim = infer_vae_dims(vae_state_dict, vae_config)
    if obs_dim_all > vae_input_dim:
        raise ValueError(
            f"Data obs_dim ({obs_dim_all}) exceeds VAE input_dim ({vae_input_dim}); VAE baseline cannot be evaluated."
        )
    vae_model = VAE(input_dim=vae_input_dim, latent_dim=vae_latent_dim).to(device)
    vae_model.load_state_dict(vae_state_dict)
    vae_model.eval()

    A, b, latent_meta = load_latent_transition(args.latent_transition_path, expected_latent_dim=vae_latent_dim)
    if "obs_dim" in latent_meta and int(latent_meta["obs_dim"]) != int(obs_dim_all):
        print(
            f"Warning: latent transition metadata obs_dim={latent_meta['obs_dim']} "
            f"does not match evaluation data obs_dim={obs_dim_all}"
        )
    if "vae_input_dim" in latent_meta and int(latent_meta["vae_input_dim"]) != int(vae_input_dim):
        print(
            f"Warning: latent transition metadata vae_input_dim={latent_meta['vae_input_dim']} "
            f"does not match checkpoint vae_input_dim={vae_input_dim}"
        )

    sequences, split_source = select_sequences_split(sequences_all, args.split, self_meta)
    _, seq_len, obs_dim = sequences.shape

    print(f"Data: {data_path} (samples={sequences_all.shape[0]}, seq_len={seq_len_all}, obs_dim={obs_dim_all})")
    print(f"Evaluation split: {split_source} (samples={sequences.shape[0]})")
    print(f"Device: {device}")
    print(
        f"Self-model checkpoint: {args.self_model_path} "
        f"(input_dim={self_input_dim}, hidden_dim={self_hidden_dim}, output_dim={self_output_dim})"
    )
    print(f"VAE checkpoint: {args.vae_model_path} (input_dim={vae_input_dim}, latent_dim={vae_latent_dim})")
    vae_training_meta = vae_meta.get("training") if isinstance(vae_meta, dict) else None
    if vae_training_meta:
        vae_data_path = str(vae_training_meta.get("data_path", ""))
        if "minimal_self_model" not in vae_data_path:
            print(
                "Warning: VAE checkpoint appears to be trained on a different dataset "
                f"({vae_data_path}); baseline may not be comparable."
            )
        if int(vae_input_dim) != int(obs_dim):
            print(
                "Warning: VAE input_dim does not match self-model observation dim "
                f"({vae_input_dim} vs {obs_dim}); zero-padding is used for the VAE baseline."
            )
    if "latent_fit_mse" in latent_meta:
        print(f"Latent transition fit MSE (saved): {float(latent_meta['latent_fit_mse']):.6f}")

    self_one_step_mse = evaluate_self_model_one_step(
        self_model, sequences=sequences, batch_size=args.batch_size, device=device
    )
    print(f"Self-model one-step MSE per element: {self_one_step_mse:.6f}")

    self_rollouts = evaluate_self_model_rollouts(
        self_model,
        sequences=sequences,
        horizons=horizons,
        batch_size=args.batch_size,
        device=device,
    )
    format_rollout_results("Self-model", self_rollouts)

    vae_rollouts = evaluate_vae_rollouts(
        vae_model,
        A=A,
        b=b,
        sequences=sequences,
        obs_dim=int(obs_dim),
        vae_input_dim=int(vae_input_dim),
        horizons=horizons,
        batch_size=args.batch_size,
        device=device,
    )
    one_step_vae_mse = vae_rollouts.get(1)
    if one_step_vae_mse is not None:
        print(f"VAE latent-transition one-step MSE per element: {one_step_vae_mse:.6f}")
    format_rollout_results("VAE latent-transition", vae_rollouts)

    if isinstance(self_meta, dict) and self_meta.get("training"):
        print(f"Self-model checkpoint training epochs: {self_meta['training'].get('epochs')}")
    if isinstance(vae_meta, dict) and vae_meta.get("training"):
        print(f"VAE checkpoint training epochs: {vae_meta['training'].get('epochs')}")


if __name__ == "__main__":
    main()
