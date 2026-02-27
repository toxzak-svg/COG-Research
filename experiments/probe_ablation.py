"""
Probe Ablation Study: Test Robustness of Ordering Hypothesis Probe

This script systematically tests the sensitivity of the ordering hypothesis probe to:
1. Bootstrap sample count (100, 500, 1000, 2000, 5000)
2. Probe points (16, 32, 64, 128, 256)
3. Finite-difference epsilon (1e-3, 1e-2, 1e-1)

Goal: Verify that conclusions are robust across reasonable parameter choices.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ordering_hypothesis_probe import (
    extract_seed,
    load_metric_series,
    load_self_model,
    load_world_model,
    get_test_observations,
    local_gain_self,
    local_gain_world,
    world_latent_cov_condition,
    checkpoint_path_for,
    parse_hidden_dims,
    parse_metrics,
    resolve_device,
)


@dataclass
class AblationConfig:
    """Configuration for a single ablation run."""
    bootstrap_samples: int
    probe_points: int
    probe_eps: float


def run_single_ablation(
    config: AblationConfig,
    dataset: str,
    results_dir: Path,
    hidden_dim: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """
    Run ordering hypothesis probe with specific ablation configuration.
    
    Returns summary statistics for this configuration.
    """
    from experiments.ordering_hypothesis_probe import (
        bootstrap_difference,
        cohens_d_paired,
        cliffs_delta,
        LOWER_IS_BETTER,
        HIGHER_IS_BETTER,
    )
    
    rng = np.random.default_rng(seed)
    
    # Load metrics for both paradigms
    metrics = ["one_step_mse", "rollout_divergence_50", "spectral_radius", "perturbation_return_rate"]
    
    results = {}
    for metric in metrics:
        self_series = load_metric_series(results_dir, dataset, "self_model_first", hidden_dim, metric)
        world_series = load_metric_series(results_dir, dataset, "world_model_first", hidden_dim, metric)
        
        # Ensure matched seeds
        common_seeds = sorted(set(self_series.seeds) & set(world_series.seeds))
        if not common_seeds:
            continue
        
        self_vals = [self_series.values[self_series.seeds.index(s)] for s in common_seeds]
        world_vals = [world_series.values[world_series.seeds.index(s)] for s in common_seeds]
        
        # Bootstrap difference
        diffs = [s - w for s, w in zip(self_vals, world_vals)]
        ci = bootstrap_difference(diffs, config.bootstrap_samples, rng)
        
        mean_diff = float(np.mean(diffs))
        ci_excludes_zero = (ci["low"] > 0) or (ci["high"] < 0)
        
        # Determine winner
        if ci_excludes_zero:
            if metric in LOWER_IS_BETTER:
                winner = "self_model_first_better" if mean_diff < 0 else "world_model_first_better"
            elif metric in HIGHER_IS_BETTER:
                winner = "self_model_first_better" if mean_diff > 0 else "world_model_first_better"
            else:
                winner = "unclear"
        else:
            winner = "no_significant_difference"
        
        results[metric] = {
            "n": len(common_seeds),
            "mean_diff": mean_diff,
            "ci95": ci,
            "ci_excludes_zero": ci_excludes_zero,
            "winner": winner,
        }
    
    # Mechanism probes
    obs = get_test_observations(dataset).to(device)
    
    # Sample a few seeds for mechanism probe
    sample_seeds = common_seeds[:3] if len(common_seeds) >= 3 else common_seeds
    
    self_gains = []
    world_gains = []
    
    for s in sample_seeds:
        # Self-model gain
        self_path = checkpoint_path_for(results_dir, dataset, "self_model_first", hidden_dim, s)
        if self_path.exists():
            self_model = load_self_model(self_path, device)
            gain = local_gain_self(self_model, obs, config.probe_eps, config.probe_points, rng)
            self_gains.append(gain)
        
        # World-model gain
        world_path = checkpoint_path_for(results_dir, dataset, "world_model_first", hidden_dim, s)
        if world_path.exists():
            vae, A, b = load_world_model(world_path, device)
            gain = local_gain_world(vae, A, b, obs, config.probe_eps, config.probe_points, rng)
            world_gains.append(gain)
    
    mechanism_summary = {
        "local_gain_self_mean": float(np.mean(self_gains)) if self_gains else None,
        "local_gain_world_mean": float(np.mean(world_gains)) if world_gains else None,
        "n_seeds_probed": len(sample_seeds),
    }
    
    return {
        "config": {
            "bootstrap_samples": config.bootstrap_samples,
            "probe_points": config.probe_points,
            "probe_eps": config.probe_eps,
        },
        "metrics": results,
        "mechanism": mechanism_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study for ordering hypothesis probe parameters"
    )
    parser.add_argument("--dataset", type=str, default="ar1", help="Dataset to analyze")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results", help="Results root")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension to test")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "probe_ablation.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "plots" / "probe_ablation.md",
        help="Output markdown summary",
    )
    return parser.parse_args()


def generate_ablation_configs() -> list[AblationConfig]:
    """Generate grid of ablation configurations to test."""
    bootstrap_options = [100, 500, 2000, 5000]
    probe_points_options = [16, 32, 128, 256]
    probe_eps_options = [1e-3, 1e-2, 1e-1]
    
    configs = []
    
    # Test bootstrap variations (fix other params)
    for bs in bootstrap_options:
        configs.append(AblationConfig(bs, 128, 1e-2))
    
    # Test probe points variations
    for pp in probe_points_options:
        configs.append(AblationConfig(2000, pp, 1e-2))
    
    # Test epsilon variations
    for eps in probe_eps_options:
        configs.append(AblationConfig(2000, 128, eps))
    
    # Add baseline configuration
    configs.insert(0, AblationConfig(2000, 128, 1e-2))
    
    return configs


def main():
    args = parse_args()
    device = resolve_device(args.device)
    
    print(f"Probe Ablation Study")
    print(f"  Dataset: {args.dataset}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Device: {device}")
    print()
    
    configs = generate_ablation_configs()
    print(f"Testing {len(configs)} configurations...")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Bootstrap={config.bootstrap_samples}, "
              f"Points={config.probe_points}, Eps={config.probe_eps:.0e}")
        
        result = run_single_ablation(
            config,
            args.dataset,
            args.results_dir,
            args.hidden_dim,
            device,
            args.seed,
        )
        results.append(result)
    
    # Save JSON results
    output = {
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "ablations": results,
    }
    
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {args.output_json}")
    
    # Generate markdown summary
    generate_summary(output, args.output_md)
    print(f"Summary saved to {args.output_md}")


def generate_summary(data: dict[str, Any], output_path: Path):
    """Generate markdown summary of ablation results."""
    lines = ["# Probe Ablation Study", ""]
    lines.append(f"Dataset: {data['dataset']}")
    lines.append(f"Hidden Dimension: {data['hidden_dim']}")
    lines.append(f"Random Seed: {data['seed']}")
    lines.append("")
    
    lines.append("## Configuration Robustness")
    lines.append("")
    lines.append("Testing sensitivity to probe design choices:")
    lines.append("")
    
    # Organize by variation type
    baseline = data["ablations"][0]
    bootstrap_variations = data["ablations"][1:5]
    point_variations = data["ablations"][5:9]
    eps_variations = data["ablations"][9:12]
    
    lines.append("### Baseline Configuration")
    lines.append("")
    lines.append(f"- Bootstrap samples: {baseline['config']['bootstrap_samples']}")
    lines.append(f"- Probe points: {baseline['config']['probe_points']}")
    lines.append(f"- Finite-diff epsilon: {baseline['config']['probe_eps']:.0e}")
    lines.append("")
    
    for metric in ["one_step_mse", "rollout_divergence_50", "spectral_radius", "perturbation_return_rate"]:
        if metric in baseline["metrics"]:
            winner = baseline["metrics"][metric]["winner"]
            ci = baseline["metrics"][metric]["ci95"]
            lines.append(f"- **{metric}**: {winner}")
            lines.append(f"  - CI: [{ci['low']:.6f}, {ci['high']:.6f}]")
    
    lines.append("")
    lines.append("### Bootstrap Sample Variations")
    lines.append("")
    lines.append("| Bootstrap | one_step_mse | rollout_div | spectral_radius | return_rate |")
    lines.append("|-----------|--------------|-------------|-----------------|-------------|")
    
    for abl in bootstrap_variations:
        bs = abl["config"]["bootstrap_samples"]
        row = [str(bs)]
        for metric in ["one_step_mse", "rollout_divergence_50", "spectral_radius", "perturbation_return_rate"]:
            if metric in abl["metrics"]:
                winner = abl["metrics"][metric]["winner"]
                symbol = "✓S" if "self" in winner else ("✓W" if "world" in winner else "—")
                row.append(symbol)
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    lines.append("### Probe Points Variations")
    lines.append("")
    lines.append("| Points | one_step_mse | rollout_div | spectral_radius | return_rate |")
    lines.append("|--------|--------------|-------------|-----------------|-------------|")
    
    for abl in point_variations:
        pts = abl["config"]["probe_points"]
        row = [str(pts)]
        for metric in ["one_step_mse", "rollout_divergence_50", "spectral_radius", "perturbation_return_rate"]:
            if metric in abl["metrics"]:
                winner = abl["metrics"][metric]["winner"]
                symbol = "✓S" if "self" in winner else ("✓W" if "world" in winner else "—")
                row.append(symbol)
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    lines.append("### Epsilon Variations")
    lines.append("")
    lines.append("| Epsilon | one_step_mse | rollout_div | spectral_radius | return_rate |")
    lines.append("|---------|--------------|-------------|-----------------|-------------|")
    
    for abl in eps_variations:
        eps = abl["config"]["probe_eps"]
        row = [f"{eps:.0e}"]
        for metric in ["one_step_mse", "rollout_divergence_50", "spectral_radius", "perturbation_return_rate"]:
            if metric in abl["metrics"]:
                winner = abl["metrics"][metric]["winner"]
                symbol = "✓S" if "self" in winner else ("✓W" if "world" in winner else "—")
                row.append(symbol)
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("**Key:** ✓S = Self-model-first wins, ✓W = World-model-first wins, — = No clear winner")
    lines.append("")
    lines.append("If conclusions are consistent across parameter variations, this indicates robustness.")
    lines.append("")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
