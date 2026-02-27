"""
Mechanistic Deep-Dive: Latent Structure, Error Modes, and Generalization

This script performs detailed mechanistic analysis of trained models:
1. Latent transition structure: eigenvalue spectra, attractors, state-space geometry
2. Error modes: where and why do models fail? OOD vs in-distribution
3. Generalization: test on longer sequences, different initial conditions
4. Information flow: gradient flow analysis, sensitivity maps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ordering_hypothesis_probe import (
    load_self_model,
    load_world_model,
    get_test_observations,
    checkpoint_path_for,
    resolve_device,
)
from stability_analysis.jacobian_spectral import compute_jacobian, spectral_radius


def analyze_latent_transition_structure(
    vae,
    A: torch.Tensor,
    b: torch.Tensor,
    obs: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """
    Analyze the structure of latent transitions in world-model.
    
    Returns:
    - Eigenvalue spectrum of A
    - Fixed points (if any)
    - State-space geometry (latent cloud structure)
    """
    # Eigenvalue spectrum
    eigvals = torch.linalg.eigvals(A).cpu().numpy()
    
    analysis = {
        "transition_matrix_eigenvalues": {
            "real": eigvals.real.tolist(),
            "imag": eigvals.imag.tolist(),
            "max_modulus": float(np.max(np.abs(eigvals))),
            "min_modulus": float(np.min(np.abs(eigvals))),
        },
    }
    
    # Fixed point analysis: z* = Az* + b => z* = (I - A)^{-1} b
    I = torch.eye(A.shape[0], device=device)
    try:
        z_star = torch.linalg.solve(I - A, b.unsqueeze(1)).squeeze()
        analysis["fixed_point"] = {
            "exists": True,
            "location": z_star.cpu().numpy().tolist(),
            "norm": float(torch.norm(z_star).item()),
        }
        
        # Stability of fixed point: eigenvalues of A
        stable = all(np.abs(eigvals) < 1.0)
        analysis["fixed_point"]["stable"] = stable
    except:
        analysis["fixed_point"] = {"exists": False}
    
    # Latent space geometry
    with torch.no_grad():
        x_norm = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
        mu, logvar = vae.encode(x_norm)
        z = mu  # Use mean encoding
    
    z_np = z.cpu().numpy()
    
    # PCA on latent space
    z_centered = z_np - z_np.mean(axis=0)
    cov = np.cov(z_centered, rowvar=False)
    eigvals_cov, eigvecs_cov = np.linalg.eigh(cov)
    eigvals_cov = eigvals_cov[::-1]  # Sort descending
    
    total_var = np.sum(eigvals_cov)
    explained_var_ratio = eigvals_cov / total_var if total_var > 0 else eigvals_cov
    
    analysis["latent_geometry"] = {
        "dimension": int(z.shape[1]),
        "n_samples": int(z.shape[0]),
        "variance_explained_by_pc": explained_var_ratio.tolist(),
        "cumulative_variance": np.cumsum(explained_var_ratio).tolist(),
        "effective_rank": float(np.exp(-np.sum(explained_var_ratio * np.log(explained_var_ratio + 1e-12)))),
    }
    
    return analysis


def analyze_self_model_structure(
    model,
    obs: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """
    Analyze the dynamical structure of self-model.
    
    Returns:
    - Jacobian spectrum at various points
    - Hidden state dynamics distribution
    """
    model.eval()
    
    # Sample points for Jacobian analysis
    n_samples = min(100, obs.shape[0])
    idx = torch.randperm(obs.shape[0])[:n_samples]
    x_samples = obs[idx]
    
    spectral_radii = []
    jacobian_norms = []
    
    for x in x_samples:
        x_batch = x.unsqueeze(0).unsqueeze(1)  # [1, 1, D]
        jac = compute_jacobian(model, x_batch, device)
        
        sr = spectral_radius(jac)
        spectral_radii.append(float(sr))
        jacobian_norms.append(float(torch.norm(jac).item()))
    
    analysis = {
        "jacobian_statistics": {
            "spectral_radius_mean": float(np.mean(spectral_radii)),
            "spectral_radius_std": float(np.std(spectral_radii)),
            "spectral_radius_max": float(np.max(spectral_radii)),
            "spectral_radius_min": float(np.min(spectral_radii)),
            "frobenius_norm_mean": float(np.mean(jacobian_norms)),
            "frobenius_norm_std": float(np.std(jacobian_norms)),
        },
        "n_points_analyzed": n_samples,
    }
    
    return analysis


def analyze_error_modes(
    model,
    data: torch.Tensor,
    device: torch.device,
    is_self_model: bool = True,
) -> dict[str, Any]:
    """
    Analyze where and why models make errors.
    
    For self-models: one-step prediction errors
    For world-models: reconstruction errors and latent consistency
    """
    model.eval()
    
    # data shape: [N, T, D]
    N, T, D = data.shape
    
    if is_self_model:
        # One-step prediction
        with torch.no_grad():
            # Flatten to observations
            obs = data.reshape(-1, D)  # [N*T, D]
            
            # Predict next step (skip last timestep)
            seq_data = data[:, :-1, :]  # [N, T-1, D]
            seq_flat = seq_data.reshape(-1, D)  # [N*(T-1), D]
            
            pred = model(seq_flat.unsqueeze(1)).squeeze(1)  # [N*(T-1), D]
            
            # Ground truth next step
            target = data[:, 1:, :].reshape(-1, D)  # [N*(T-1), D]
            
            # Compute per-sample errors
            errors = torch.norm(pred - target, dim=1).cpu().numpy()
    else:
        # World-model: reconstruction error
        with torch.no_grad():
            obs = data.reshape(-1, D)
            obs_norm = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
            
            recon, _, _ = model(obs_norm)
            errors = torch.norm(recon - obs_norm, dim=1).cpu().numpy()
    
    # Analyze error distribution
    analysis = {
        "error_statistics": {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "median": float(np.median(errors)),
            "max": float(np.max(errors)),
            "min": float(np.min(errors)),
            "q95": float(np.percentile(errors, 95)),
            "q99": float(np.percentile(errors, 99)),
        },
        "n_samples": len(errors),
    }
    
    # Identify high-error samples
    high_error_threshold = np.percentile(errors, 90)
    high_error_count = np.sum(errors > high_error_threshold)
    
    analysis["high_error_samples"] = {
        "threshold_p90": float(high_error_threshold),
        "count": int(high_error_count),
        "fraction": float(high_error_count / len(errors)),
    }
    
    return analysis


def test_generalization_longer_sequences(
    model,
    dataset: str,
    device: torch.device,
    is_self_model: bool = True,
    max_horizon: int = 100,
) -> dict[str, Any]:
    """
    Test how models generalize to longer sequences than training.
    """
    # Generate longer test sequence
    from deterministic_data.deterministic_dataset import (
        generate_ar1,
        generate_damped_oscillator,
        generate_van_der_pol,
    )
    
    if dataset == "ar1":
        data = generate_ar1(n_samples=50, horizon=max_horizon, seed=12345, state_dim=2)
    elif dataset == "damped":
        data = generate_damped_oscillator(n_samples=50, horizon=max_horizon, seed=12345)
    elif dataset == "vanderpol":
        data = generate_van_der_pol(n_samples=50, horizon=max_horizon, seed=12345)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    data = data.to(device).float()
    model.eval()
    
    # Rollout and measure divergence at different horizons
    horizons_to_test = [10, 20, 30, 50, max_horizon]
    
    results = {}
    
    for h in horizons_to_test:
        if h > data.shape[1]:
            continue
        
        test_data = data[:, :h, :]
        
        if is_self_model:
            # Autoregressive rollout
            with torch.no_grad():
                x = test_data[:, 0, :]  # Initial state
                rollout = [x]
                
                for t in range(1, h):
                    x = model(x.unsqueeze(1)).squeeze(1)
                    rollout.append(x)
                
                rollout = torch.stack(rollout, dim=1)  # [N, h, D]
                
                mse = F.mse_loss(rollout, test_data).item()
                divergence = torch.mean(torch.norm(rollout - test_data, dim=2)).item()
        else:
            # World-model doesn't naturally do long rollouts in observation space
            # We measure reconstruction quality instead
            with torch.no_grad():
                obs = test_data.reshape(-1, test_data.shape[-1])
                obs_norm = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
                
                recon, _, _ = model(obs_norm)
                mse = F.mse_loss(recon, obs_norm).item()
                divergence = torch.mean(torch.norm(recon - obs_norm, dim=1)).item()
        
        results[f"horizon_{h}"] = {
            "mse": float(mse),
            "divergence": float(divergence),
        }
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mechanistic deep-dive analysis")
    parser.add_argument("--dataset", type=str, default="ar1", help="Dataset to analyze")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dim to analyze")
    parser.add_argument("--seed", type=int, default=0, help="Model seed to analyze")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max-horizon", type=int, default=100, help="Max horizon for generalization test")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "mechanistic_deepdive.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "plots" / "mechanistic_deepdive.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    
    print("Mechanistic Deep-Dive Analysis")
    print(f"  Dataset: {args.dataset}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    print()
    
    # Load test data
    test_path = PROJECT_ROOT / "data" / "deterministic" / f"{args.dataset}_test.pt"
    test_data = torch.load(test_path).to(device).float()
    obs = get_test_observations(args.dataset).to(device)
    
    results = {
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "analyses": {},
    }
    
    # Analyze self-model
    print("Analyzing self-model...")
    self_path = checkpoint_path_for(
        args.results_dir, args.dataset, "self_model_first", args.hidden_dim, args.seed
    )
    
    if self_path.exists():
        self_model = load_self_model(self_path, device)
        
        results["analyses"]["self_model"] = {
            "dynamics": analyze_self_model_structure(self_model, obs, device),
            "error_modes": analyze_error_modes(self_model, test_data, device, is_self_model=True),
            "generalization": test_generalization_longer_sequences(
                self_model, args.dataset, device, is_self_model=True, max_horizon=args.max_horizon
            ),
        }
        print("  Self-model analysis complete.")
    else:
        print(f"  Self-model checkpoint not found: {self_path}")
    
    # Analyze world-model
    print("Analyzing world-model...")
    world_path = checkpoint_path_for(
        args.results_dir, args.dataset, "world_model_first", args.hidden_dim, args.seed
    )
    
    if world_path.exists():
        vae, A, b = load_world_model(world_path, device)
        
        results["analyses"]["world_model"] = {
            "latent_structure": analyze_latent_transition_structure(vae, A, b, obs, device),
            "error_modes": analyze_error_modes(vae, test_data, device, is_self_model=False),
            "generalization": test_generalization_longer_sequences(
                vae, args.dataset, device, is_self_model=False, max_horizon=args.max_horizon
            ),
        }
        print("  World-model analysis complete.")
    else:
        print(f"  World-model checkpoint not found: {world_path}")
    
    # Save results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_json}")
    
    # Generate summary
    generate_summary(results, args.output_md)
    print(f"Summary saved to {args.output_md}")


def generate_summary(data: dict[str, Any], output_path: Path):
    """Generate markdown summary."""
    lines = ["# Mechanistic Deep-Dive Analysis", ""]
    lines.append(f"**Dataset**: {data['dataset']}")
    lines.append(f"**Hidden Dimension**: {data['hidden_dim']}")
    lines.append(f"**Model Seed**: {data['seed']}")
    lines.append("")
    
    if "self_model" in data["analyses"]:
        lines.append("## Self-Model Analysis")
        lines.append("")
        
        sm = data["analyses"]["self_model"]
        
        if "dynamics" in sm:
            lines.append("### Jacobian Dynamics")
            jac = sm["dynamics"]["jacobian_statistics"]
            lines.append(f"- Spectral radius: {jac['spectral_radius_mean']:.4f} ± {jac['spectral_radius_std']:.4f}")
            lines.append(f"- Range: [{jac['spectral_radius_min']:.4f}, {jac['spectral_radius_max']:.4f}]")
            lines.append(f"- Frobenius norm: {jac['frobenius_norm_mean']:.4f} ± {jac['frobenius_norm_std']:.4f}")
            lines.append("")
        
        if "error_modes" in sm:
            lines.append("### Error Distribution")
            err = sm["error_modes"]["error_statistics"]
            lines.append(f"- Mean error: {err['mean']:.6f}")
            lines.append(f"- Median error: {err['median']:.6f}")
            lines.append(f"- 95th percentile: {err['q95']:.6f}")
            lines.append(f"- Max error: {err['max']:.6f}")
            
            high_err = sm["error_modes"]["high_error_samples"]
            lines.append(f"- High-error samples (top 10%): {high_err['count']} ({high_err['fraction']*100:.1f}%)")
            lines.append("")
        
        if "generalization" in sm:
            lines.append("### Generalization (Longer Horizons)")
            lines.append("")
            lines.append("| Horizon | MSE | Divergence |")
            lines.append("|---------|-----|------------|")
            for key in sorted(sm["generalization"].keys()):
                h = key.replace("horizon_", "")
                mse = sm["generalization"][key]["mse"]
                div = sm["generalization"][key]["divergence"]
                lines.append(f"| {h} | {mse:.6f} | {div:.6f} |")
            lines.append("")
    
    if "world_model" in data["analyses"]:
        lines.append("## World-Model Analysis")
        lines.append("")
        
        wm = data["analyses"]["world_model"]
        
        if "latent_structure" in wm:
            lines.append("### Latent Transition Structure")
            trans = wm["latent_structure"]["transition_matrix_eigenvalues"]
            lines.append(f"- Max eigenvalue modulus: {trans['max_modulus']:.4f}")
            lines.append(f"- Min eigenvalue modulus: {trans['min_modulus']:.4f}")
            
            if wm["latent_structure"]["fixed_point"]["exists"]:
                fp = wm["latent_structure"]["fixed_point"]
                lines.append(f"- Fixed point exists: {fp['stable'] and 'stable' or 'unstable'}")
                lines.append(f"- Fixed point norm: {fp['norm']:.4f}")
            else:
                lines.append("- No fixed point (A is singular)")
            lines.append("")
            
            lines.append("### Latent Space Geometry")
            geom = wm["latent_structure"]["latent_geometry"]
            lines.append(f"- Latent dimension: {geom['dimension']}")
            lines.append(f"- Effective rank: {geom['effective_rank']:.2f}")
            cum_var = geom['cumulative_variance']
            for i, cv in enumerate(cum_var[:3], 1):
                lines.append(f"- PC{i} cumulative variance: {cv:.4f}")
            lines.append("")
        
        if "error_modes" in wm:
            lines.append("### Reconstruction Error Distribution")
            err = wm["error_modes"]["error_statistics"]
            lines.append(f"- Mean error: {err['mean']:.6f}")
            lines.append(f"- Median error: {err['median']:.6f}")
            lines.append(f"- 95th percentile: {err['q95']:.6f}")
            lines.append("")
        
        if "generalization" in wm:
            lines.append("### Generalization (Longer Sequences)")
            lines.append("")
            lines.append("| Horizon | MSE | Divergence |")
            lines.append("|---------|-----|------------|")
            for key in sorted(wm["generalization"].keys()):
                h = key.replace("horizon_", "")
                mse = wm["generalization"][key]["mse"]
                div = wm["generalization"][key]["divergence"]
                lines.append(f"| {h} | {mse:.6f} | {div:.6f} |")
            lines.append("")
    
    lines.append("## Key Insights")
    lines.append("")
    lines.append("1. **Stability**: Compare spectral radii and eigenvalue moduli between paradigms")
    lines.append("2. **Error Modes**: Identify where failures occur (distributional outliers)")
    lines.append("3. **Generalization**: Track performance degradation with longer horizons")
    lines.append("4. **Latent Structure**: Examine effective dimensionality and geometry")
    lines.append("")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
