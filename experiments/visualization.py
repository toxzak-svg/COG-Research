"""
Visualization Module for Parameter Count vs Stability Analysis

This module generates publication-quality plots from multi-seed experiment results
to visualize the relationship between model parameter count and stability metrics.

Key plots:
- One-step MSE vs Parameters: shows which paradigm fits better
- 50-step Rollout Divergence vs Parameters: long-horizon stability  
- Perturbation Return Time vs Parameters: perturbation absorption speed
- Spectral Radius vs Parameters: dynamical stability (σ_max)
- Efficiency plot: stability-per-parameter (ratio metric)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paradigm names (matching multi_seed_comparison.py)
SELF_MODEL_FIRST = "self_model_first"
WORLD_MODEL_FIRST = "world_model_first"

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
PARADIGM_COLORS = {
    SELF_MODEL_FIRST: '#2E86AB',      # Blue
    WORLD_MODEL_FIRST: '#A23B72',     # Magenta
}
PARADIGM_MARKERS = {
    SELF_MODEL_FIRST: 'o',
    WORLD_MODEL_FIRST: 's',
}
PARADIGM_LABELS = {
    SELF_MODEL_FIRST: 'Self-Model-First',
    WORLD_MODEL_FIRST: 'World-Model-First',
}


# ==============================================================================
# Result Loading
# ==============================================================================

def load_results(results_dir: str, dataset: str, paradigm: str) -> Dict[int, Dict[str, Any]]:
    """
    Load aggregated metrics for a given dataset and paradigm.
    
    Args:
        results_dir: Path to results directory (containing aggregated_results.json)
        dataset: Dataset name (ar1, damped, vanderpol)
        paradigm: Paradigm name (self_model_first or world_model_first)
        
    Returns:
        Dictionary mapping hidden_dim -> metrics dict with mean/std
    """
    results_path = Path(results_dir) / "aggregated_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path) as f:
        all_results = json.load(f)
    
    if dataset not in all_results:
        raise ValueError(f"Dataset '{dataset}' not found in results")
    
    if paradigm not in all_results[dataset]:
        # Return empty dict instead of raising error
        return {}
    
    return all_results[dataset][paradigm]


def get_parameter_count(hidden_dim: int, paradigm: str, obs_dim: int = 2) -> int:
    """
    Calculate parameter count for a given configuration.
    
    For Self-Model-First (RNN): hidden_dim * (hidden_dim + obs_dim) + hidden_dim (W_hh + W_xh + b_h) + hidden_dim * obs_dim + obs_dim (W_hy + b_y)
    For World-Model-First (VAE + Latent): encoder + decoder + latent transition (A, b)
    """
    if paradigm == SELF_MODEL_FIRST:
        # RNN parameters
        # Hidden to hidden: hidden_dim x hidden_dim
        # Input to hidden: obs_dim x hidden_dim  
        # Hidden bias: hidden_dim
        # Hidden to output: hidden_dim x obs_dim
        # Output bias: obs_dim
        w_hh = hidden_dim * hidden_dim
        w_xh = obs_dim * hidden_dim
        b_h = hidden_dim
        w_hy = hidden_dim * obs_dim
        b_y = obs_dim
        return w_hh + w_xh + b_h + w_hy + b_y
    else:
        # VAE parameters (simplified estimation)
        # Encoder: obs_dim -> hidden_dim -> latent_dim
        w_enc1 = obs_dim * hidden_dim
        b_enc1 = hidden_dim
        w_enc2 = hidden_dim * hidden_dim
        b_enc2 = hidden_dim
        w_enc3 = hidden_dim * hidden_dim
        b_enc3 = hidden_dim
        
        # Decoder: latent_dim -> hidden_dim -> obs_dim
        w_dec1 = hidden_dim * hidden_dim
        b_dec1 = hidden_dim
        w_dec2 = hidden_dim * hidden_dim
        b_dec2 = hidden_dim
        w_dec3 = hidden_dim * obs_dim
        b_dec3 = obs_dim
        
        # Latent transition: A (hidden_dim x hidden_dim) + b (hidden_dim)
        w_trans = hidden_dim * hidden_dim
        b_trans = hidden_dim
        
        return (w_enc1 + b_enc1 + w_enc2 + b_enc2 + w_enc3 + b_enc3 +
                w_dec1 + b_dec1 + w_dec2 + b_dec2 + w_dec3 + b_dec3 +
                w_trans + b_trans)


def load_all_results(
    results_dir: str,
    dataset: str,
    hidden_dims: List[int] = [16, 32, 64, 128],
    obs_dim: int = 2,
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
    """
    Load results for both paradigms and prepare data for plotting.
    
    Returns:
        Tuple of (params, metrics_by_paradigm) where:
        - params: list of parameter counts
        - metrics_by_paradigm: dict of paradigm -> metric_name -> {means, stds}
    """
    results = {SELF_MODEL_FIRST: {}, WORLD_MODEL_FIRST: {}}
    
    for paradigm in [SELF_MODEL_FIRST, WORLD_MODEL_FIRST]:
        paradigm_results = load_results(results_dir, dataset, paradigm)
        
        # Skip if no results for this paradigm
        if not paradigm_results:
            continue
        
        for hidden_dim in hidden_dims:
            if str(hidden_dim) in paradigm_results:
                stats = paradigm_results[str(hidden_dim)]
                param_count = get_parameter_count(hidden_dim, paradigm, obs_dim)
                
                results[paradigm][hidden_dim] = {
                    'params': param_count,
                    'one_step_mse': stats.get('one_step_mse', {}),
                    'rollout_divergence_50': stats.get('rollout_divergence_50', {}),
                    'spectral_radius': stats.get('spectral_radius', {}),
                    'perturbation_return_rate': stats.get('perturbation_return_rate', {}),
                }
    
    return results


# ==============================================================================
# Plotting Functions
# ==============================================================================

def setup_figure(figsize: Tuple[float, float] = (10, 6), title: str = None):
    """Set up a publication-quality figure."""
    fig, ax = plt.subplots(figsize= figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    return fig, ax


def plot_single_metric(
    ax,
    results: Dict[str, Dict],
    metric: str,
    ylabel: str,
    hidden_dims: List[int],
    invert_y: bool = False,
    ylim: Tuple[float, float] = None,
):
    """
    Plot a single metric for both paradigms.
    
    Args:
        ax: matplotlib axis
        results: results dictionary from load_all_results
        metric: metric name key
        ylabel: y-axis label
        hidden_dims: list of hidden dimensions
        invert_y: whether to invert y-axis (for error metrics, lower is better)
        ylim: y-axis limits (optional)
    """
    for paradigm in [SELF_MODEL_FIRST, WORLD_MODEL_FIRST]:
        means = []
        stds = []
        params = []
        
        for hd in hidden_dims:
            if hd in results[paradigm]:
                data = results[paradigm][hd]
                metric_data = data.get(metric, {})
                mean_val = metric_data.get('mean')
                if mean_val is not None and not np.isnan(mean_val):
                    means.append(mean_val)
                    stds.append(metric_data.get('std', 0))
                    params.append(data['params'])
        
        if not means:
            continue
            
        params = np.array(params)
        means = np.array(means)
        stds = np.array(stds)
        
        # Sort by parameter count
        sort_idx = np.argsort(params)
        params = params[sort_idx]
        means = means[sort_idx]
        stds = stds[sort_idx]
        
        color = PARADIGM_COLORS[paradigm]
        marker = PARADIGM_MARKERS[paradigm]
        label = PARADIGM_LABELS[paradigm]
        
        ax.errorbar(
            params, means, yerr=stds,
            label=label,
            color=color,
            marker=marker,
            markersize=8,
            capsize=4,
            capthick=1.5,
            linewidth=2,
            elinewidth=1.5,
        )
    
    ax.set_xlabel('Parameter Count', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Use log scale for x-axis (parameter count)
    ax.set_xscale('log')
    
    if invert_y:
        ax.invert_yaxis()
    
    if ylim:
        ax.set_ylim(ylim)


def plot_mse_vs_params(
    results: Dict[str, Dict],
    hidden_dims: List[int],
    output_path: Path,
    dataset: str,
):
    """Plot one-step MSE vs parameter count."""
    fig, ax = setup_figure(figsize=(10, 6), title=f'One-Step MSE vs Parameters ({dataset})')
    plot_single_metric(
        ax, results, 'one_step_mse',
        'One-Step MSE (↓ better)',
        hidden_dims,
        invert_y=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_rollout_divergence(
    results: Dict[str, Dict],
    hidden_dims: List[int],
    output_path: Path,
    dataset: str,
):
    """Plot 50-step rollout divergence vs parameter count."""
    fig, ax = setup_figure(figsize=(10, 6), title=f'50-Step Rollout Divergence vs Parameters ({dataset})')
    plot_single_metric(
        ax, results, 'rollout_divergence_50',
        '50-Step Rollout Divergence (↓ better)',
        hidden_dims,
        invert_y=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_spectral_radius(
    results: Dict[str, Dict],
    hidden_dims: List[int],
    output_path: Path,
    dataset: str,
):
    """Plot spectral radius vs parameter count."""
    fig, ax = setup_figure(figsize=(10, 6), title=f'Spectral Radius (σ_max) vs Parameters ({dataset})')
    plot_single_metric(
        ax, results, 'spectral_radius',
        'Spectral Radius σ_max (↓ better)',
        hidden_dims,
        invert_y=True,
    )
    # Add reference line at σ=1 (stability boundary)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Stability boundary (σ=1)')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_perturbation_return_rate(
    results: Dict[str, Dict],
    hidden_dims: List[int],
    output_path: Path,
    dataset: str,
):
    """Plot perturbation return rate vs parameter count."""
    fig, ax = setup_figure(figsize=(10, 6), title=f'Perturbation Return Rate vs Parameters ({dataset})')
    plot_single_metric(
        ax, results, 'perturbation_return_rate',
        'Perturbation Return Rate (↑ better)',
        hidden_dims,
        invert_y=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_efficiency(
    results: Dict[str, Dict],
    hidden_dims: List[int],
    output_path: Path,
    dataset: str,
    metric: str = 'spectral_radius',
):
    """
    Plot efficiency metric (stability per parameter).
    
    Efficiency = 1 / (metric_value * parameter_count)
    Higher is better: more stability per parameter.
    """
    fig, ax = setup_figure(figsize=(10, 6), title=f'Efficiency: {metric} per Parameter ({dataset})')
    
    metric_titles = {
        'spectral_radius': 'Stability (1/σ_max)',
        'perturbation_return_rate': 'Return Rate',
        'rollout_divergence_50': '1/Divergence',
    }
    
    for paradigm in [SELF_MODEL_FIRST, WORLD_MODEL_FIRST]:
        efficiencies = []
        stds = []
        params = []
        
        for hd in hidden_dims:
            if hd in results[paradigm]:
                data = results[paradigm][hd]
                metric_data = data.get(metric, {})
                mean_val = metric_data.get('mean')
                
                if mean_val is not None and not np.isnan(mean_val) and mean_val > 0:
                    # Efficiency: higher is better
                    # For spectral_radius: lower is better, so 1/mean is efficiency
                    # For return_rate: higher is better, so mean is efficiency
                    if metric == 'spectral_radius':
                        efficiency = 1.0 / (mean_val * data['params'])
                        efficiency_std = (metric_data.get('std', 0) / mean_val**2) * (1.0 / data['params'])
                    elif metric == 'perturbation_return_rate':
                        efficiency = mean_val / data['params']
                        efficiency_std = (metric_data.get('std', 0) / data['params'])
                    else:  # rollout_divergence
                        efficiency = 1.0 / (mean_val * data['params'])
                        efficiency_std = (metric_data.get('std', 0) / mean_val**2) * (1.0 / data['params'])
                    
                    efficiencies.append(efficiency)
                    stds.append(efficiency_std if not np.isnan(efficiency_std) else 0)
                    params.append(data['params'])
        
        if not efficiencies:
            continue
            
        params = np.array(params)
        efficiencies = np.array(efficiencies)
        stds = np.array(stds)
        
        # Sort by parameter count
        sort_idx = np.argsort(params)
        params = params[sort_idx]
        efficiencies = efficiencies[sort_idx]
        stds = stds[sort_idx]
        
        color = PARADIGM_COLORS[paradigm]
        marker = PARADIGM_MARKERS[paradigm]
        label = PARADIGM_LABELS[paradigm]
        
        ax.errorbar(
            params, efficiencies, yerr=stds,
            label=label,
            color=color,
            marker=marker,
            markersize=8,
            capsize=4,
            capthick=1.5,
            linewidth=2,
            elinewidth=1.5,
        )
    
    ax.set_xlabel('Parameter Count', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Efficiency (↑ better)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scaling_law(
    results_dir: str,
    dataset: str,
    output_dir: str,
    hidden_dims: List[int] = [16, 32, 64, 128],
    obs_dim: int = 2,
) -> Dict[str, Path]:
    """
    Generate all scaling law plots.
    
    Args:
        results_dir: Path to results directory
        dataset: Dataset name
        output_dir: Output directory for plots
        hidden_dims: List of hidden dimensions
        obs_dim: Observation dimension
        
    Returns:
        Dictionary mapping plot name -> output path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_all_results(results_dir, dataset, hidden_dims, obs_dim)
    
    plots = {}
    
    # Generate each plot
    plot_mse_vs_params(
        results, hidden_dims,
        output_path / f'{dataset}_one_step_mse.png',
        dataset,
    )
    plots['mse'] = output_path / f'{dataset}_one_step_mse.png'
    
    plot_rollout_divergence(
        results, hidden_dims,
        output_path / f'{dataset}_rollout_divergence.png',
        dataset,
    )
    plots['rollout_divergence'] = output_path / f'{dataset}_rollout_divergence.png'
    
    plot_spectral_radius(
        results, hidden_dims,
        output_path / f'{dataset}_spectral_radius.png',
        dataset,
    )
    plots['spectral_radius'] = output_path / f'{dataset}_spectral_radius.png'
    
    plot_perturbation_return_rate(
        results, hidden_dims,
        output_path / f'{dataset}_perturbation_return.png',
        dataset,
    )
    plots['perturbation_return'] = output_path / f'{dataset}_perturbation_return.png'
    
    # Efficiency plots
    plot_efficiency(
        results, hidden_dims,
        output_path / f'{dataset}_efficiency_spectral_radius.png',
        dataset,
        metric='spectral_radius',
    )
    plots['efficiency_spectral'] = output_path / f'{dataset}_efficiency_spectral_radius.png'
    
    plot_efficiency(
        results, hidden_dims,
        output_path / f'{dataset}_efficiency_rollout.png',
        dataset,
        metric='rollout_divergence_50',
    )
    plots['efficiency_rollout'] = output_path / f'{dataset}_efficiency_rollout.png'
    
    return plots


def generate_summary_table(
    results_dir: str,
    dataset: str,
    hidden_dims: List[int] = [16, 32, 64, 128],
    obs_dim: int = 2,
) -> str:
    """
    Generate a markdown summary table of results.
    
    Returns:
        Markdown formatted table string
    """
    results = load_all_results(results_dir, dataset, hidden_dims, obs_dim)
    
    lines = []
    lines.append(f"# Results Summary: {dataset}")
    lines.append("")
    lines.append("## Parameter Counts")
    lines.append("")
    lines.append(f"- Hidden dimensions: {hidden_dims}")
    lines.append(f"- Self-Model-First (RNN): ~{get_parameter_count(hidden_dims[0], SELF_MODEL_FIRST, obs_dim)} - {get_parameter_count(hidden_dims[-1], SELF_MODEL_FIRST, obs_dim)} parameters")
    lines.append(f"- World-Model-First (VAE): ~{get_parameter_count(hidden_dims[0], WORLD_MODEL_FIRST, obs_dim)} - {get_parameter_count(hidden_dims[-1], WORLD_MODEL_FIRST, obs_dim)} parameters")
    lines.append("")
    lines.append("## Metrics by Configuration")
    lines.append("")
    lines.append("| Hidden Dim | Paradigm | Params | One-Step MSE | 50-Step Div | Spectral Radius | Return Rate |")
    lines.append("|------------|----------|--------|--------------|-------------|-----------------|-------------|")
    
    for hd in hidden_dims:
        for paradigm in [SELF_MODEL_FIRST, WORLD_MODEL_FIRST]:
            if hd in results[paradigm]:
                data = results[paradigm][hd]
                params = data['params']
                
                mse = data.get('one_step_mse', {}).get('mean', np.nan)
                mse_std = data.get('one_step_mse', {}).get('std', 0)
                
                div = data.get('rollout_divergence_50', {}).get('mean', np.nan)
                div_std = data.get('rollout_divergence_50', {}).get('std', 0)
                
                sr = data.get('spectral_radius', {}).get('mean', np.nan)
                sr_std = data.get('spectral_radius', {}).get('std', 0)
                
                rr = data.get('perturbation_return_rate', {}).get('mean', np.nan)
                rr_std = data.get('perturbation_return_rate', {}).get('std', 0)
                
                paradigm_label = "Self-Model" if paradigm == SELF_MODEL_FIRST else "World-Model"
                
                lines.append(
                    f"| {hd} | {paradigm_label} | {params} | "
                    f"{mse:.4f}±{mse_std:.4f} | {div:.4f}±{div_std:.4f} | "
                    f"{sr:.4f}±{sr_std:.4f} | {rr:.4f}±{rr_std:.4f} |"
                )
    
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    
    # Analyze efficiency
    efficiency_comparison = analyze_efficiency(results, hidden_dims)
    lines.append(efficiency_comparison)
    
    return "\n".join(lines)


def analyze_efficiency(results: Dict, hidden_dims: List[int]) -> str:
    """Analyze efficiency differences between paradigms."""
    lines = []
    
    # Compare spectral radius efficiency
    sm_efficiencies = []
    wm_efficiencies = []
    
    for hd in hidden_dims:
        if hd in results[SELF_MODEL_FIRST]:
            sm_data = results[SELF_MODEL_FIRST][hd]
            sr = sm_data.get('spectral_radius', {}).get('mean')
            if sr and not np.isnan(sr) and sr > 0:
                sm_efficiencies.append(1.0 / (sr * sm_data['params']))
        
        if hd in results[WORLD_MODEL_FIRST]:
            wm_data = results[WORLD_MODEL_FIRST][hd]
            sr = wm_data.get('spectral_radius', {}).get('mean')
            if sr and not np.isnan(sr) and sr > 0:
                wm_efficiencies.append(1.0 / (sr * wm_data['params']))
    
    if sm_efficiencies and wm_efficiencies:
        avg_sm = np.mean(sm_efficiencies)
        avg_wm = np.mean(wm_efficiencies)
        
        lines.append(f"- **Stability Efficiency (1/σ_max per param)**: ")
        lines.append(f"  - Self-Model-First: {avg_sm:.2e}")
        lines.append(f"  - World-Model-First: {avg_wm:.2e}")
        
        if avg_sm > avg_wm:
            ratio = avg_sm / avg_wm
            lines.append(f"  - Self-Model-First is {ratio:.1f}x more efficient")
        else:
            ratio = avg_wm / avg_sm
            lines.append(f"  - World-Model-First is {ratio:.1f}x more efficient")
    
    return "\n".join(lines)


def save_summary_markdown(
    results_dir: str,
    dataset: str,
    output_dir: str,
    hidden_dims: List[int] = [16, 32, 64, 128],
):
    """Save the summary markdown table to file."""
    summary = generate_summary_table(results_dir, dataset, hidden_dims)
    
    output_path = Path(output_dir) / f'{dataset}_summary.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Saved: {output_path}")
    return output_path


# ==============================================================================
# CLI Interface
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate parameter count vs stability plots from experiment results"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing aggregated_results.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ar1",
        choices=["ar1", "damped", "vanderpol"],
        help="Which dataset to plot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Where to save generated plots",
    )
    parser.add_argument(
        "--plot-types",
        type=str,
        default="all",
        help="Which plots to generate: all, mse, stability, efficiency",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="16,32,64,128",
        help="Comma-separated list of hidden dimensions",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=2,
        help="Observation dimension",
    )
    
    args = parser.parse_args()
    
    # Parse hidden dims
    hidden_dims = [int(h.strip()) for h in args.hidden_dims.split(",")]
    
    return {
        "results_dir": args.results_dir,
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "plot_types": args.plot_types,
        "hidden_dims": hidden_dims,
        "obs_dim": args.obs_dim,
    }


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"{'='*60}")
    print("Generating Parameter Count vs Stability Plots")
    print(f"{'='*60}")
    print(f"Results directory: {args['results_dir']}")
    print(f"Dataset: {args['dataset']}")
    print(f"Output directory: {args['output_dir']}")
    print(f"Plot types: {args['plot_types']}")
    print(f"Hidden dims: {args['hidden_dims']}")
    print(f"{'='*60}")
    
    # Check if results exist
    results_path = Path(args['results_dir']) / "aggregated_results.json"
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Please run multi_seed_comparison.py first to generate results.")
        return
    
    # Generate plots
    if args['plot_types'] == 'all' or args['plot_types'] == 'mse':
        print("\n--- Generating MSE plots ---")
        results = load_all_results(
            args['results_dir'], args['dataset'],
            args['hidden_dims'], args['obs_dim']
        )
        
        output_path = Path(args['output_dir'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_mse_vs_params(
            results, args['hidden_dims'],
            output_path / f'{args["dataset"]}_one_step_mse.png',
            args['dataset'],
        )
    
    if args['plot_types'] == 'all' or args['plot_types'] == 'stability':
        print("\n--- Generating stability plots ---")
        results = load_all_results(
            args['results_dir'], args['dataset'],
            args['hidden_dims'], args['obs_dim']
        )
        
        output_path = Path(args['output_dir'])
        
        plot_rollout_divergence(
            results, args['hidden_dims'],
            output_path / f'{args["dataset"]}_rollout_divergence.png',
            args['dataset'],
        )
        
        plot_spectral_radius(
            results, args['hidden_dims'],
            output_path / f'{args["dataset"]}_spectral_radius.png',
            args['dataset'],
        )
        
        plot_perturbation_return_rate(
            results, args['hidden_dims'],
            output_path / f'{args["dataset"]}_perturbation_return.png',
            args['dataset'],
        )
    
    if args['plot_types'] == 'all' or args['plot_types'] == 'efficiency':
        print("\n--- Generating efficiency plots ---")
        results = load_all_results(
            args['results_dir'], args['dataset'],
            args['hidden_dims'], args['obs_dim']
        )
        
        output_path = Path(args['output_dir'])
        
        plot_efficiency(
            results, args['hidden_dims'],
            output_path / f'{args["dataset"]}_efficiency_spectral_radius.png',
            args['dataset'],
            metric='spectral_radius',
        )
        
        plot_efficiency(
            results, args['hidden_dims'],
            output_path / f'{args["dataset"]}_efficiency_rollout.png',
            args['dataset'],
            metric='rollout_divergence_50',
        )
    
    # Generate summary table
    print("\n--- Generating summary table ---")
    save_summary_markdown(
        args['results_dir'],
        args['dataset'],
        args['output_dir'],
        args['hidden_dims'],
    )
    
    print(f"\n{'='*60}")
    print("Done! Plots saved to:", args['output_dir'])
    print(f"{'='*60}")


if __name__ == "__main__":
    main()