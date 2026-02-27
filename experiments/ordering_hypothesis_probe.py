from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imagination_first_learning.models.vae import VAE
from minimal_self_model.models.self_model import SelfModel


LOWER_IS_BETTER = {"one_step_mse", "rollout_divergence_50", "spectral_radius"}
HIGHER_IS_BETTER = {"perturbation_return_rate"}


@dataclass
class MetricSeries:
    values: list[float]
    seeds: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe ordering hypothesis (self-model-first vs world-model-first) with attribution stats and mechanisms."
    )
    parser.add_argument("--dataset", type=str, default="ar1", help="Dataset under results/<dataset> (default: ar1)")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results", help="Results root directory")
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="16,32,64,128",
        help="Comma-separated hidden dims to inspect (script intersects with available checkpoints)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="one_step_mse,rollout_divergence_50,spectral_radius,perturbation_return_rate",
        help="Comma-separated metrics expected in seed_*.json files",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for CI")
    parser.add_argument("--probe-points", type=int, default=128, help="Observation points per seed for local-gain probe")
    parser.add_argument("--probe-eps", type=float, default=1e-2, help="Finite-difference perturbation scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"], help="Torch device")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "ordering_hypothesis_probe.json",
        help="Path to write machine-readable report",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "plots" / "ordering_hypothesis_probe.md",
        help="Path to write markdown summary",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def parse_hidden_dims(raw: str) -> list[int]:
    return [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def parse_metrics(raw: str) -> list[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def extract_seed(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("seed_"):
        raise ValueError(f"Unexpected seed file name: {path.name}")
    return int(stem.split("_")[1])


def load_metric_series(
    base_dir: Path,
    dataset: str,
    paradigm: str,
    hidden_dim: int,
    metric: str,
) -> MetricSeries:
    dim_dir = base_dir / dataset / paradigm / str(hidden_dim)
    if not dim_dir.exists():
        return MetricSeries(values=[], seeds=[])

    pairs: list[tuple[int, float]] = []
    for metrics_path in sorted(dim_dir.glob("seed_*.json")):
        seed = extract_seed(metrics_path)
        with open(metrics_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if metric not in payload:
            continue
        value = payload[metric]
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        pairs.append((seed, value))

    pairs.sort(key=lambda item: item[0])
    return MetricSeries(values=[value for _, value in pairs], seeds=[seed for seed, _ in pairs])


def align_by_seed(lhs: MetricSeries, rhs: MetricSeries) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rhs_map = {seed: value for seed, value in zip(rhs.seeds, rhs.values)}
    aligned_seeds: list[int] = []
    left_vals: list[float] = []
    right_vals: list[float] = []
    for seed, value in zip(lhs.seeds, lhs.values):
        if seed not in rhs_map:
            continue
        aligned_seeds.append(seed)
        left_vals.append(value)
        right_vals.append(rhs_map[seed])
    return np.asarray(left_vals, dtype=np.float64), np.asarray(right_vals, dtype=np.float64), aligned_seeds


def bootstrap_mean_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.shape != y.shape:
        raise ValueError("Expected paired arrays with equal shape for bootstrap mean diff.")

    diffs = x - y
    observed = float(np.mean(diffs))
    samples = np.empty(n_samples, dtype=np.float64)
    n = diffs.shape[0]
    for idx in range(n_samples):
        sample_idx = rng.integers(0, n, size=n)
        samples[idx] = np.mean(diffs[sample_idx])
    low = float(np.percentile(samples, 2.5))
    high = float(np.percentile(samples, 97.5))
    return observed, low, high


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diffs = x - y
    if diffs.size < 2:
        return float("nan")
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return float("inf") if float(np.mean(diffs)) != 0.0 else 0.0
    return float(np.mean(diffs) / std)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = 0
    lesser = 0
    for xv in x:
        greater += int(np.sum(xv > y))
        lesser += int(np.sum(xv < y))
    total = x.size * y.size
    return float((greater - lesser) / total)


def load_self_model(checkpoint_path: Path, device: torch.device) -> SelfModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    config = payload["config"]
    model = SelfModel(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_world_model(checkpoint_path: Path, device: torch.device) -> tuple[VAE, torch.Tensor, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    config = payload["config"]
    model = VAE(input_dim=int(config["input_dim"]), latent_dim=int(config["latent_dim"])).to(device)
    model.load_state_dict(state)
    model.eval()
    A = torch.tensor(np.asarray(config["A"], dtype=np.float32), device=device)
    b = torch.tensor(np.asarray(config["b"], dtype=np.float32), device=device)
    return model, A, b


def get_test_observations(dataset: str) -> torch.Tensor:
    path = PROJECT_ROOT / "data" / "deterministic" / f"{dataset}_test.pt"
    if not path.exists():
        raise FileNotFoundError(f"Test split not found: {path}")
    seq = torch.load(path)
    if seq.ndim != 3:
        raise ValueError(f"Expected [N,T,D] tensor at {path}, got {tuple(seq.shape)}")
    return seq.reshape(-1, seq.shape[-1]).float()


def local_gain_self(model: SelfModel, obs: torch.Tensor, eps: float, probe_points: int, rng: np.random.Generator) -> float:
    n = obs.shape[0]
    take = min(probe_points, n)
    idx = rng.choice(n, size=take, replace=False)
    x = obs[idx]
    delta = torch.randn_like(x)
    delta = eps * delta / (torch.norm(delta, dim=1, keepdim=True) + 1e-12)

    with torch.no_grad():
        f_x = model(x.unsqueeze(1)).squeeze(1)
        f_xp = model((x + delta).unsqueeze(1)).squeeze(1)

    num = torch.norm(f_xp - f_x, dim=1)
    den = torch.norm(delta, dim=1) + 1e-12
    return float(torch.mean(num / den).item())


def local_gain_world(
    model: VAE,
    A: torch.Tensor,
    b: torch.Tensor,
    obs: torch.Tensor,
    eps: float,
    probe_points: int,
    rng: np.random.Generator,
) -> float:
    n = obs.shape[0]
    take = min(probe_points, n)
    idx = rng.choice(n, size=take, replace=False)
    x = obs[idx]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)

    delta = torch.randn_like(x_norm)
    delta = eps * delta / (torch.norm(delta, dim=1, keepdim=True) + 1e-12)

    with torch.no_grad():
        mu, _ = model.encode(x_norm)
        mu_p, _ = model.encode(x_norm + delta)
        z_next = mu @ A + b
        z_next_p = mu_p @ A + b
        y = model.decode(z_next)
        y_p = model.decode(z_next_p)

    num = torch.norm(y_p - y, dim=1)
    den = torch.norm(delta, dim=1) + 1e-12
    return float(torch.mean(num / den).item())


def world_latent_cov_condition(model: VAE, obs: torch.Tensor, batch_size: int = 512) -> float:
    x_norm = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_norm.shape[0], batch_size):
            chunk = x_norm[start : start + batch_size]
            mu, _ = model.encode(chunk)
            latents.append(mu.detach().cpu().numpy())
    z = np.concatenate(latents, axis=0)
    cov = np.cov(z, rowvar=False)
    if cov.ndim == 0:
        return 1.0
    eigvals = np.linalg.eigvalsh(cov + 1e-8 * np.eye(cov.shape[0]))
    eigvals = np.maximum(eigvals, 1e-12)
    return float(np.max(eigvals) / np.min(eigvals))


def checkpoint_path_for(results_dir: Path, dataset: str, paradigm: str, hidden_dim: int, seed: int) -> Path:
    return results_dir / dataset / paradigm / str(hidden_dim) / f"seed_{seed}.pth"


def mechanism_probe_for_dim(
    results_dir: Path,
    dataset: str,
    hidden_dim: int,
    seeds: list[int],
    device: torch.device,
    probe_points: int,
    probe_eps: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    obs = get_test_observations(dataset).to(device)

    self_gains: list[float] = []
    world_gains: list[float] = []
    world_cov_conditions: list[float] = []
    world_transition_norms: list[float] = []

    for seed in seeds:
        sm_path = checkpoint_path_for(results_dir, dataset, "self_model_first", hidden_dim, seed)
        wm_path = checkpoint_path_for(results_dir, dataset, "world_model_first", hidden_dim, seed)
        if not sm_path.exists() or not wm_path.exists():
            continue

        self_model = load_self_model(sm_path, device=device)
        world_model, A, b = load_world_model(wm_path, device=device)

        self_gain = local_gain_self(self_model, obs, eps=probe_eps, probe_points=probe_points, rng=rng)
        world_gain = local_gain_world(world_model, A, b, obs, eps=probe_eps, probe_points=probe_points, rng=rng)
        cov_cond = world_latent_cov_condition(world_model, obs)
        trans_norm = float(torch.linalg.norm(A, ord=2).item())

        self_gains.append(self_gain)
        world_gains.append(world_gain)
        world_cov_conditions.append(cov_cond)
        world_transition_norms.append(trans_norm)

    def stat(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "n_seed_pairs": len(self_gains),
        "self_local_gain": stat(self_gains),
        "world_local_gain": stat(world_gains),
        "world_latent_cov_condition": stat(world_cov_conditions),
        "world_transition_spectral_norm": stat(world_transition_norms),
        "gain_ratio_world_over_self": float(np.mean(world_gains) / (np.mean(self_gains) + 1e-12)) if self_gains else float("nan"),
    }


def summarize_claim_direction(metric: str, mean_diff_self_minus_world: float) -> str:
    if metric in LOWER_IS_BETTER:
        if mean_diff_self_minus_world < 0:
            return "self_model_first_better"
        if mean_diff_self_minus_world > 0:
            return "world_model_first_better"
        return "tie"
    if metric in HIGHER_IS_BETTER:
        if mean_diff_self_minus_world > 0:
            return "self_model_first_better"
        if mean_diff_self_minus_world < 0:
            return "world_model_first_better"
        return "tie"
    return "unknown"


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Ordering Hypothesis Probe ({report['dataset']})")
    lines.append("")
    lines.append("This report compares **Self-Model-First** vs **World-Model-First** across matched seeds and hidden dimensions.")
    lines.append("")
    lines.append("## Attribution Summary")
    lines.append("")

    for dim_key, dim_payload in report["by_hidden_dim"].items():
        lines.append(f"### Hidden Dim {dim_key}")
        lines.append("")
        lines.append(f"- Matched seed pairs: {dim_payload['n_seed_pairs']}")
        lines.append("")
        lines.append("| Metric | Self Mean | World Mean | Δ(Self-World) | 95% CI | Winner |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for metric_name, metric_payload in dim_payload["metrics"].items():
            ci_low = metric_payload["ci95"]["low"]
            ci_high = metric_payload["ci95"]["high"]
            lines.append(
                "| "
                f"{metric_name} | {metric_payload['self_mean']:.6g} | {metric_payload['world_mean']:.6g} | "
                f"{metric_payload['mean_diff_self_minus_world']:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | {metric_payload['winner']} |"
            )

        mech = dim_payload.get("mechanism", {})
        if mech:
            lines.append("")
            lines.append("Mechanism probes:")
            lines.append(
                f"- Local gain (mean): self={mech['self_local_gain']['mean']:.6g}, "
                f"world={mech['world_local_gain']['mean']:.6g}, "
                f"ratio(world/self)={mech['gain_ratio_world_over_self']:.6g}"
            )
            lines.append(
                f"- World latent covariance condition (mean): {mech['world_latent_cov_condition']['mean']:.6g}"
            )
            lines.append(
                f"- World transition spectral norm (mean): {mech['world_transition_spectral_norm']['mean']:.6g}"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Lower-is-better metrics: one_step_mse, rollout_divergence_50, spectral_radius")
    lines.append("- Higher-is-better metric: perturbation_return_rate")
    lines.append("- CI is computed on paired seed differences via bootstrap resampling")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    metrics = parse_metrics(args.metrics)

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "results_dir": str(args.results_dir),
        "seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "probe_points": int(args.probe_points),
        "probe_eps": float(args.probe_eps),
        "by_hidden_dim": {},
    }

    for hidden_dim in hidden_dims:
        dim_entry: dict[str, Any] = {"metrics": {}, "n_seed_pairs": 0}
        common_seeds: set[int] | None = None

        for metric in metrics:
            self_series = load_metric_series(args.results_dir, args.dataset, "self_model_first", hidden_dim, metric)
            world_series = load_metric_series(args.results_dir, args.dataset, "world_model_first", hidden_dim, metric)
            self_vals, world_vals, aligned = align_by_seed(self_series, world_series)

            if common_seeds is None:
                common_seeds = set(aligned)
            else:
                common_seeds &= set(aligned)

            if self_vals.size == 0 or world_vals.size == 0:
                continue

            mean_diff, ci_low, ci_high = bootstrap_mean_diff(
                self_vals,
                world_vals,
                n_samples=args.bootstrap_samples,
                rng=rng,
            )

            dim_entry["metrics"][metric] = {
                "n": int(self_vals.size),
                "self_mean": float(np.mean(self_vals)),
                "self_std": float(np.std(self_vals)),
                "world_mean": float(np.mean(world_vals)),
                "world_std": float(np.std(world_vals)),
                "mean_diff_self_minus_world": float(mean_diff),
                "ci95": {"low": float(ci_low), "high": float(ci_high)},
                "cohens_d_paired": float(cohens_d_paired(self_vals, world_vals)),
                "cliffs_delta": float(cliffs_delta(self_vals, world_vals)),
                "winner": summarize_claim_direction(metric, mean_diff),
                "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
            }

        usable_seeds = sorted(common_seeds) if common_seeds else []
        dim_entry["n_seed_pairs"] = len(usable_seeds)
        if usable_seeds:
            dim_entry["mechanism"] = mechanism_probe_for_dim(
                results_dir=args.results_dir,
                dataset=args.dataset,
                hidden_dim=hidden_dim,
                seeds=usable_seeds,
                device=device,
                probe_points=args.probe_points,
                probe_eps=args.probe_eps,
                rng=rng,
            )

        if dim_entry["metrics"]:
            report["by_hidden_dim"][str(hidden_dim)] = dim_entry

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.output_md)

    print(f"Saved JSON report to: {args.output_json}")
    print(f"Saved Markdown report to: {args.output_md}")


if __name__ == "__main__":
    main()
