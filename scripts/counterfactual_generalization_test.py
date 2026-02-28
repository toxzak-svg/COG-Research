from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from imagination_first_learning.models.vae import VAE
from minimal_self_model.models.self_model import SelfModel


DEFAULT_VAE_PATH = Path("vae_model.pth")
DEFAULT_SELF_MODEL_PATH = Path("self_model.pth")


def parse_range(value: str) -> tuple[float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected range formatted as 'low,high', got: {value}")
    low, high = float(parts[0]), float(parts[1])
    if not low < high:
        raise ValueError(f"Range must satisfy low < high, got: ({low}, {high})")
    return low, high


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint_obj = torch.load(path, map_location=device)

    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
        config = checkpoint_obj.get("config", {}) or {}
        metadata = checkpoint_obj
    elif isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
        config = {}
        metadata = {"model_state_dict": state_dict}
    else:
        raise TypeError(f"Unsupported checkpoint format in {path}")

    return metadata, state_dict, config


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


def infer_self_model_dims(state_dict, config):
    input_dim = config.get("input_dim")
    hidden_dim = config.get("hidden_dim")
    output_dim = config.get("output_dim")

    if input_dim is None or hidden_dim is None:
        if "rnn.weight_ih_l0" not in state_dict:
            raise KeyError("Could not infer self-model input_dim/hidden_dim from checkpoint.")
        rnn_w = state_dict["rnn.weight_ih_l0"]
        hidden_dim = int(rnn_w.shape[0]) if hidden_dim is None else int(hidden_dim)
        input_dim = int(rnn_w.shape[1]) if input_dim is None else int(input_dim)
    if output_dim is None:
        if "fc.weight" not in state_dict:
            raise KeyError("Could not infer self-model output_dim from checkpoint.")
        output_dim = int(state_dict["fc.weight"].shape[0])

    return int(input_dim), int(hidden_dim), int(output_dim)


def load_vae_model(model_path: Path, device: torch.device):
    metadata, state_dict, config = load_checkpoint(model_path, device)
    input_dim, latent_dim = infer_vae_dims(state_dict, config)
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, {"input_dim": input_dim, "latent_dim": latent_dim}, metadata


def load_self_model(model_path: Path, device: torch.device):
    metadata, state_dict, config = load_checkpoint(model_path, device)
    input_dim, hidden_dim, output_dim = infer_self_model_dims(state_dict, config)
    model = SelfModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim}, metadata


def generate_uniform_data(num_samples, value_range, input_dim, sequence_length=None, rng=None):
    rng = rng or np.random.default_rng()
    low, high = value_range
    if sequence_length is None:
        shape = (int(num_samples), int(input_dim))
    else:
        shape = (int(num_samples), int(sequence_length), int(input_dim))
    return rng.uniform(low, high, size=shape).astype(np.float32)


def write_json(path: Path | None, payload: dict):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved JSON results to {path}")


def maybe_plot(x, series, xlabel, ylabel, title, save_path: Path | None, show_plot: bool):
    if save_path is None and not show_plot:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or omit --show-plot/--save-plot."
        ) from exc

    plt.figure(figsize=(8, 5))
    for values, label in series:
        plt.plot(x, values, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close()


def evaluate_vae_ood(model_path: Path, ood_data: np.ndarray, device_arg: str = "auto") -> float:
    if ood_data.ndim != 2:
        raise ValueError(f"Expected 2D VAE data [num_samples, input_dim], got {ood_data.shape}")
    device = get_device(device_arg)
    vae, dims, _ = load_vae_model(model_path, device)
    if int(ood_data.shape[1]) != int(dims["input_dim"]):
        raise ValueError(
            f"OOD VAE data input_dim ({ood_data.shape[1]}) does not match checkpoint input_dim ({dims['input_dim']})"
        )

    x = torch.tensor(ood_data, dtype=torch.float32, device=device)
    with torch.no_grad():
        # Use the encoder mean for a deterministic reconstruction metric.
        mu, _ = vae.encode(x)
        reconstructed = vae.decode(mu)
        mse = F.mse_loss(reconstructed, x).item()
    return float(mse)


def evaluate_self_model_ood(model_path: Path, ood_data: np.ndarray, device_arg: str = "auto") -> float:
    if ood_data.ndim != 3 or ood_data.shape[1] < 2:
        raise ValueError(f"Expected 3D sequential data [num_samples, seq_len>=2, input_dim], got {ood_data.shape}")
    device = get_device(device_arg)
    self_model, dims, _ = load_self_model(model_path, device)
    if int(ood_data.shape[2]) != int(dims["input_dim"]):
        raise ValueError(
            "OOD self-model data feature dim "
            f"({ood_data.shape[2]}) does not match checkpoint input_dim ({dims['input_dim']})"
        )
    if int(dims["output_dim"]) != int(dims["input_dim"]):
        raise ValueError(
            f"This script expects self-model output_dim == input_dim, got {dims['output_dim']} vs {dims['input_dim']}"
        )

    x = torch.tensor(ood_data, dtype=torch.float32, device=device)
    inputs, targets = x[:, :-1, :], x[:, 1:, :]
    with torch.no_grad():
        predictions = self_model(inputs)
        mse = F.mse_loss(predictions, targets).item()
    return float(mse)


def shift_severity_sweep(
    model_path_self: Path,
    model_path_vae: Path,
    train_range: tuple[float, float],
    sweep_ranges: list[tuple[float, float]],
    sequence_length_self: int,
    num_samples: int,
    seed: int,
    device_arg: str,
):
    rng = np.random.default_rng(seed)
    device = get_device(device_arg)
    _, self_dims, _ = load_self_model(model_path_self, device)
    _, vae_dims, _ = load_vae_model(model_path_vae, device)

    self_model_errors = []
    vae_errors = []
    shift_magnitudes = []

    for value_range in sweep_ranges:
        ood_data_self_model = generate_uniform_data(
            num_samples=num_samples,
            value_range=value_range,
            input_dim=self_dims["input_dim"],
            sequence_length=sequence_length_self,
            rng=rng,
        )
        ood_data_vae = generate_uniform_data(
            num_samples=num_samples,
            value_range=value_range,
            input_dim=vae_dims["input_dim"],
            sequence_length=None,
            rng=rng,
        )

        self_model_mse = evaluate_self_model_ood(model_path_self, ood_data_self_model, device_arg=device_arg)
        vae_mse = evaluate_vae_ood(model_path_vae, ood_data_vae, device_arg=device_arg)

        self_model_errors.append(float(self_model_mse))
        vae_errors.append(float(vae_mse))
        shift_magnitudes.append(float(value_range[1] - train_range[1]))

    self_model_audc = float(np.trapezoid(self_model_errors, shift_magnitudes))
    vae_audc = float(np.trapezoid(vae_errors, shift_magnitudes))

    return {
        "train_range": [float(train_range[0]), float(train_range[1])],
        "sweep_ranges": [[float(lo), float(hi)] for lo, hi in sweep_ranges],
        "shift_magnitudes": shift_magnitudes,
        "self_model_errors": self_model_errors,
        "vae_errors": vae_errors,
        "self_model_audc": self_model_audc,
        "vae_audc": vae_audc,
    }


def perturbation_return_test(
    model_path_self: Path,
    sequence_length: int,
    perturbation_magnitude: float,
    num_samples: int,
    value_range: tuple[float, float],
    seed: int,
    device_arg: str,
):
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    rng = np.random.default_rng(seed)
    device = get_device(device_arg)
    self_model, dims, _ = load_self_model(model_path_self, device)

    initial_conditions = generate_uniform_data(
        num_samples=num_samples,
        value_range=value_range,
        input_dim=dims["input_dim"],
        sequence_length=sequence_length,
        rng=rng,
    )
    perturbed_conditions = initial_conditions + rng.normal(
        0.0,
        perturbation_magnitude,
        size=initial_conditions.shape,
    ).astype(np.float32)

    initial_tensor = torch.tensor(initial_conditions, dtype=torch.float32, device=device)
    perturbed_tensor = torch.tensor(perturbed_conditions, dtype=torch.float32, device=device)

    divergence = []
    with torch.no_grad():
        for t in range(sequence_length):
            initial_output = self_model(initial_tensor[:, : t + 1, :])[:, -1, :]
            perturbed_output = self_model(perturbed_tensor[:, : t + 1, :])[:, -1, :]
            divergence_t = torch.mean((initial_output - perturbed_output) ** 2).item()
            divergence.append(float(divergence_t))

    short_horizon_growth = float(divergence[1] - divergence[0]) if len(divergence) > 1 else 0.0
    long_horizon_saturation = float(divergence[-1]) if divergence else 0.0

    return {
        "value_range": [float(value_range[0]), float(value_range[1])],
        "sequence_length": int(sequence_length),
        "perturbation_magnitude": float(perturbation_magnitude),
        "num_samples": int(num_samples),
        "divergence": divergence,
        "short_horizon_growth": short_horizon_growth,
        "long_horizon_saturation": long_horizon_saturation,
    }


def crude_lyapunov_estimate(
    model_path_self: Path,
    sequence_length: int,
    num_samples: int,
    value_range: tuple[float, float],
    seed: int,
    device_arg: str,
):
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")

    rng = np.random.default_rng(seed)
    device = get_device(device_arg)
    self_model, dims, _ = load_self_model(model_path_self, device)

    initial_conditions = generate_uniform_data(
        num_samples=num_samples,
        value_range=value_range,
        input_dim=dims["input_dim"],
        sequence_length=sequence_length,
        rng=rng,
    )
    initial_tensor = torch.tensor(initial_conditions, dtype=torch.float32, device=device)

    divergence = []
    targets = initial_tensor[:, 1:, :]
    with torch.no_grad():
        for t in range(sequence_length - 1):
            inputs = initial_tensor[:, : t + 1, :]
            predictions = self_model(inputs)[:, -1, :]
            divergence_t = torch.mean((predictions - targets[:, t, :]) ** 2).item()
            divergence.append(float(divergence_t))

    if all(d < 1e-3 for d in divergence):
        stability = "Contracting (Stable)"
    elif any(d > 1.0 for d in divergence):
        stability = "Explosive"
    else:
        stability = "Near-Critical"

    return {
        "value_range": [float(value_range[0]), float(value_range[1])],
        "sequence_length": int(sequence_length),
        "num_samples": int(num_samples),
        "divergence": divergence,
        "stability_regime": stability,
    }


def print_shift_results(results: dict):
    print("Shift severity sweep")
    print(f"  Train range: {tuple(results['train_range'])}")
    for shift, self_err, vae_err in zip(
        results["shift_magnitudes"],
        results["self_model_errors"],
        results["vae_errors"],
        strict=True,
    ):
        print(f"  Shift {shift:+.3f} | Self-model MSE={self_err:.6f} | VAE MSE={vae_err:.6f}")
    print(f"  Self-model AUDC: {results['self_model_audc']:.6f}")
    print(f"  VAE AUDC: {results['vae_audc']:.6f}")


def print_perturbation_results(results: dict):
    print("Perturbation return test")
    print(f"  Short-horizon growth: {results['short_horizon_growth']:.6f}")
    print(f"  Long-horizon saturation: {results['long_horizon_saturation']:.6f}")


def print_lyapunov_results(results: dict):
    print("Crude Lyapunov estimate")
    print(f"  Stability regime: {results['stability_regime']}")
    print(f"  Divergence points: {len(results['divergence'])}")


def build_parser():
    parser = ArgumentParser(description="Counterfactual / robustness probes for the VAE and self-model checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_plot = {
        "show_plot": dict(action="store_true", help="Display plot with matplotlib (optional dependency)."),
        "save_plot": dict(type=Path, default=None, help="Save plot image (requires matplotlib)."),
        "json_out": dict(type=Path, default=None, help="Optional JSON output path."),
    }

    shift = subparsers.add_parser("shift-sweep", help="Evaluate OOD error under increasing uniform-range shifts.")
    shift.add_argument("--self-model-path", type=Path, default=DEFAULT_SELF_MODEL_PATH)
    shift.add_argument("--vae-model-path", type=Path, default=DEFAULT_VAE_PATH)
    shift.add_argument("--train-range", type=parse_range, default=(0.5, 1.0))
    shift.add_argument(
        "--sweep-ranges",
        type=parse_range,
        nargs="+",
        default=[(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)],
        help="List of OOD ranges formatted as 'low,high'.",
    )
    shift.add_argument("--sequence-length", type=int, default=20, help="Sequence length for self-model OOD data.")
    shift.add_argument("--num-samples", type=int, default=1000)
    shift.add_argument("--seed", type=int, default=42)
    shift.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    shift.add_argument("--show-plot", **common_plot["show_plot"])
    shift.add_argument("--save-plot", **common_plot["save_plot"])
    shift.add_argument("--json-out", **common_plot["json_out"])

    perturb = subparsers.add_parser("perturbation-return", help="Measure divergence under perturbed sequence prefixes.")
    perturb.add_argument("--self-model-path", type=Path, default=DEFAULT_SELF_MODEL_PATH)
    perturb.add_argument("--value-range", type=parse_range, default=(0.5, 1.0))
    perturb.add_argument("--sequence-length", type=int, default=20)
    perturb.add_argument("--perturbation-magnitude", type=float, default=0.1)
    perturb.add_argument("--num-samples", type=int, default=100)
    perturb.add_argument("--seed", type=int, default=42)
    perturb.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    perturb.add_argument("--show-plot", **common_plot["show_plot"])
    perturb.add_argument("--save-plot", **common_plot["save_plot"])
    perturb.add_argument("--json-out", **common_plot["json_out"])

    lyap = subparsers.add_parser("lyapunov", help="Run a crude Lyapunov-style stability proxy on random sequences.")
    lyap.add_argument("--self-model-path", type=Path, default=DEFAULT_SELF_MODEL_PATH)
    lyap.add_argument("--value-range", type=parse_range, default=(0.5, 1.0))
    lyap.add_argument("--sequence-length", type=int, default=20)
    lyap.add_argument("--num-samples", type=int, default=100)
    lyap.add_argument("--seed", type=int, default=42)
    lyap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    lyap.add_argument("--show-plot", **common_plot["show_plot"])
    lyap.add_argument("--save-plot", **common_plot["save_plot"])
    lyap.add_argument("--json-out", **common_plot["json_out"])

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "shift-sweep":
        results = shift_severity_sweep(
            model_path_self=args.self_model_path,
            model_path_vae=args.vae_model_path,
            train_range=args.train_range,
            sweep_ranges=list(args.sweep_ranges),
            sequence_length_self=args.sequence_length,
            num_samples=args.num_samples,
            seed=args.seed,
            device_arg=args.device,
        )
        print_shift_results(results)
        write_json(args.json_out, results)
        maybe_plot(
            x=results["shift_magnitudes"],
            series=[
                (results["self_model_errors"], "Self-Model"),
                (results["vae_errors"], "VAE"),
            ],
            xlabel="Shift Magnitude (range_high - train_range_high)",
            ylabel="Mean Squared Error (MSE)",
            title="Degradation Curve: Error vs Shift Magnitude",
            save_path=args.save_plot,
            show_plot=args.show_plot,
        )
        return

    if args.command == "perturbation-return":
        results = perturbation_return_test(
            model_path_self=args.self_model_path,
            sequence_length=args.sequence_length,
            perturbation_magnitude=args.perturbation_magnitude,
            num_samples=args.num_samples,
            value_range=args.value_range,
            seed=args.seed,
            device_arg=args.device,
        )
        print_perturbation_results(results)
        write_json(args.json_out, results)
        maybe_plot(
            x=list(range(results["sequence_length"])),
            series=[(results["divergence"], "Divergence")],
            xlabel="Time Step",
            ylabel="Divergence (MSE)",
            title="Perturbation Return Test: Divergence Over Time",
            save_path=args.save_plot,
            show_plot=args.show_plot,
        )
        return

    if args.command == "lyapunov":
        results = crude_lyapunov_estimate(
            model_path_self=args.self_model_path,
            sequence_length=args.sequence_length,
            num_samples=args.num_samples,
            value_range=args.value_range,
            seed=args.seed,
            device_arg=args.device,
        )
        print_lyapunov_results(results)
        write_json(args.json_out, results)
        maybe_plot(
            x=list(range(len(results["divergence"]))),
            series=[(results["divergence"], "Prediction Error Proxy")],
            xlabel="Time Step",
            ylabel="Divergence (MSE)",
            title="Crude Lyapunov Estimate: Divergence Over Time",
            save_path=args.save_plot,
            show_plot=args.show_plot,
        )
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
