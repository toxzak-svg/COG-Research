"""
Visualization script for Timeseries-PILE benchmark results.
Creates publication-quality plots comparing self-model vs world-model.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_experiment_history(exp_dir: Path) -> Optional[Dict]:
    """Load training history from experiment directory."""
    history_path = exp_dir / 'history.json'
    if not history_path.exists():
        return None
    
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_training_curves(
    results_dir: Path,
    dataset: str,
    output_dir: Path,
):
    """Plot training curves for all seeds of both models."""
    
    # Find all experiment directories
    exp_dirs = list(results_dir.glob(f"{dataset}_*_h*"))
    
    self_model_histories = []
    world_model_histories = []
    
    for exp_dir in exp_dirs:
        history = load_experiment_history(exp_dir)
        if history is None:
            continue
        
        if 'self_model' in exp_dir.name:
            self_model_histories.append(history)
        elif 'world_model' in exp_dir.name:
            world_model_histories.append(history)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot self-model
    ax = axes[0]
    for i, history in enumerate(self_model_histories):
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        epochs = range(1, len(train_loss) + 1)
        
        ax.plot(epochs, train_loss, 'b-', alpha=0.3, linewidth=1)
        ax.plot(epochs, val_loss, 'r-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(f'Self-Model Training Curves ({len(self_model_histories)} seeds)', fontsize=14)
    ax.legend(['Train', 'Validation'], loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot world-model
    ax = axes[1]
    for i, history in enumerate(world_model_histories):
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        epochs = range(1, len(train_loss) + 1)
        
        ax.plot(epochs, train_loss, 'b-', alpha=0.3, linewidth=1)
        ax.plot(epochs, val_loss, 'r-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'World-Model Training Curves ({len(world_model_histories)} seeds)', fontsize=14)
    ax.legend(['Train', 'Validation'], loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{dataset}_training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {output_path}")
    plt.close()


def plot_comparison_bar(
    comparison_path: Path,
    output_dir: Path,
):
    """Plot bar chart comparing final validation losses."""
    
    with open(comparison_path, 'r') as f:
        comparison = json.load(f)
    
    dataset = comparison['dataset']
    sm_agg = comparison.get('self_model_aggregate')
    wm_agg = comparison.get('world_model_aggregate')
    
    if sm_agg is None or wm_agg is None:
        print("Incomplete comparison data, skipping bar plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['Self-Model', 'World-Model']
    means = [sm_agg['mean_val_loss'], wm_agg['mean_val_loss']]
    stds = [sm_agg['std_val_loss'], wm_agg['std_val_loss']]
    
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7,
                  color=['#2ecc71', '#3498db'], edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Validation Loss (MSE)', fontsize=13)
    ax.set_title(f'{dataset} Benchmark: Self-Model vs World-Model', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add winner annotation
    if sm_agg['mean_val_loss'] < wm_agg['mean_val_loss']:
        winner_text = "✓ Self-Model wins"
        improvement = ((wm_agg['mean_val_loss'] - sm_agg['mean_val_loss']) / wm_agg['mean_val_loss']) * 100
    else:
        winner_text = "✓ World-Model wins"
        improvement = ((sm_agg['mean_val_loss'] - wm_agg['mean_val_loss']) / sm_agg['mean_val_loss']) * 100
    
    ax.text(0.5, 0.95, f"{winner_text} ({improvement:.1f}% better)",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{dataset}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def plot_seed_variance(
    results_dir: Path,
    dataset: str,
    output_dir: Path,
):
    """Plot final validation loss for each seed to show variance."""
    
    # Find all experiment directories
    exp_dirs = list(results_dir.glob(f"{dataset}_*_h*"))
    
    self_model_seeds = {}
    world_model_seeds = {}
    
    for exp_dir in exp_dirs:
        history = load_experiment_history(exp_dir)
        if history is None:
            continue
        
        # Extract seed number from directory name
        parts = exp_dir.name.split('_')
        seed_part = [p for p in parts if p.startswith('seed')]
        if not seed_part:
            continue
        seed = int(seed_part[0].replace('seed', ''))
        
        val_losses = history.get('val_loss', [])
        if len(val_losses) == 0:
            continue
        
        final_val_loss = min(val_losses)
        
        if 'self_model' in exp_dir.name:
            self_model_seeds[seed] = final_val_loss
        elif 'world_model' in exp_dir.name:
            world_model_seeds[seed] = final_val_loss
    
    if len(self_model_seeds) == 0 or len(world_model_seeds) == 0:
        print("Insufficient seed data, skipping variance plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by seed
    sm_seeds = sorted(self_model_seeds.keys())
    wm_seeds = sorted(world_model_seeds.keys())
    
    sm_losses = [self_model_seeds[s] for s in sm_seeds]
    wm_losses = [world_model_seeds[s] for s in wm_seeds]
    
    # Plot lines
    ax.plot(sm_seeds, sm_losses, 'o-', linewidth=2, markersize=8,
            label='Self-Model', color='#2ecc71')
    ax.plot(wm_seeds, wm_losses, 's-', linewidth=2, markersize=8,
            label='World-Model', color='#3498db')
    
    # Add horizontal lines for means
    ax.axhline(np.mean(sm_losses), color='#2ecc71', linestyle='--',
               alpha=0.5, label=f'Self-Model mean: {np.mean(sm_losses):.4f}')
    ax.axhline(np.mean(wm_losses), color='#3498db', linestyle='--',
               alpha=0.5, label=f'World-Model mean: {np.mean(wm_losses):.4f}')
    
    ax.set_xlabel('Random Seed', fontsize=12)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title(f'{dataset}: Validation Loss Across Seeds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{dataset}_seed_variance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved seed variance plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Timeseries-PILE benchmark results'
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--results-dir', type=str, default='results/timeseries_pile',
                       help='Results directory')
    parser.add_argument('--output-dir', type=str, default='plots/timeseries',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TIMESERIES-PILE RESULTS VISUALIZATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(results_dir, args.dataset, output_dir)
    
    # Plot comparison bar chart
    comparison_path = results_dir / 'comparisons' / f'{args.dataset}_comparison.json'
    if comparison_path.exists():
        print("\nGenerating comparison bar chart...")
        plot_comparison_bar(comparison_path, output_dir)
    else:
        print(f"\nComparison file not found: {comparison_path}")
    
    # Plot seed variance
    print("\nGenerating seed variance plot...")
    plot_seed_variance(results_dir, args.dataset, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
