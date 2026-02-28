#!/usr/bin/env python3
"""
Cross-Dataset Analysis Script

Analyzes patterns across multiple datasets to identify universal principles
vs dataset-specific behaviors in self-model vs world-model performance.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: Path, dataset: str) -> Dict:
    """Load results for a specific dataset."""
    dataset_paths = [
        results_dir / dataset / "aggregated_results.json",
        results_dir / f"{dataset}_comparison.json",
        results_dir / "timeseries_pile" / f"{dataset}_comparison.json",
    ]
    
    for path in dataset_paths:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    
    return None


def extract_metrics(data: Dict) -> Dict[str, Dict]:
    """Extract key metrics from results data."""
    metrics = {
        'self_model': {},
        'world_model': {}
    }
    
    # Try different data structures
    if 'self_model_first' in data:
        for hidden_dim, stats in data.get('self_model_first', {}).items():
            metrics['self_model'][hidden_dim] = stats
    
    if 'world_model_first' in data:
        for hidden_dim, stats in data.get('world_model_first', {}).items():
            metrics['world_model'][hidden_dim] = stats
    
    return metrics


def compare_datasets(results_dir: Path, datasets: List[str], output_dir: Path):
    """Compare performance across multiple datasets."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_data = []
    
    for dataset in datasets:
        print(f"\nAnalyzing {dataset}...")
        data = load_results(results_dir, dataset)
        
        if data is None:
            print(f"  ⚠️ No results found for {dataset}")
            continue
        
        metrics = extract_metrics(data)
        
        # Extract comparison metrics
        for model_type in ['self_model', 'world_model']:
            if model_type not in metrics or not metrics[model_type]:
                continue
            
            for hidden_dim, stats in metrics[model_type].items():
                comparison_data.append({
                    'dataset': dataset,
                    'model': model_type.replace('_', '-'),
                    'hidden_dim': int(hidden_dim) if hidden_dim.isdigit() else hidden_dim,
                    'one_step_mse': stats.get('one_step_mse', {}).get('mean', None),
                    'one_step_mse_std': stats.get('one_step_mse', {}).get('std', None),
                    'spectral_radius': stats.get('spectral_radius', {}).get('mean', None),
                    'spectral_radius_std': stats.get('spectral_radius', {}).get('std', None),
                    'rollout_divergence': stats.get('rollout_divergence_50', {}).get('mean', None),
                    'rollout_divergence_std': stats.get('rollout_divergence_50', {}).get('std', None),
                })
    
    if not comparison_data:
        print("\n❌ No data to analyze. Run experiments first.")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Save raw data
    df.to_csv(output_dir / 'cross_dataset_comparison.csv', index=False)
    print(f"\n✅ Saved comparison data to {output_dir / 'cross_dataset_comparison.csv'}")
    
    # Generate visualizations
    create_visualizations(df, output_dir)
    
    # Generate insights
    generate_insights(df, output_dir)


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations."""
    
    sns.set_style("whitegrid")
    
    # 1. One-step MSE comparison across datasets
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MSE comparison
    if 'one_step_mse' in df.columns and df['one_step_mse'].notna().any():
        sns.barplot(data=df, x='dataset', y='one_step_mse', hue='model', ax=axes[0])
        axes[0].set_title('One-Step MSE (Lower is Better)')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
    
    # Spectral radius comparison
    if 'spectral_radius' in df.columns and df['spectral_radius'].notna().any():
        sns.barplot(data=df, x='dataset', y='spectral_radius', hue='model', ax=axes[1])
        axes[1].set_title('Spectral Radius (Lower is Better)')
        axes[1].set_ylabel('Spectral Radius')
        axes[1].set_yscale('log')
        axes[1].tick_params(axis='x', rotation=45)
    
    # Rollout divergence comparison
    if 'rollout_divergence' in df.columns and df['rollout_divergence'].notna().any():
        sns.barplot(data=df, x='dataset', y='rollout_divergence', hue='model', ax=axes[2])
        axes[2].set_title('Rollout Divergence (Lower is Better)')
        axes[2].set_ylabel('Divergence')
        axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_dataset_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_dir / 'cross_dataset_comparison.png'}")
    plt.close()


def generate_insights(df: pd.DataFrame, output_dir: Path):
    """Generate insights from cross-dataset comparison."""
    
    insights = ["# Cross-Dataset Analysis Insights\n\n"]
    
    # Count datasets analyzed
    n_datasets = df['dataset'].nunique()
    insights.append(f"**Datasets Analyzed:** {n_datasets}\n")
    insights.append(f"**Datasets:** {', '.join(df['dataset'].unique())}\n\n")
    
    # Compare self-model vs world-model winners
    insights.append("## Model Performance Summary\n\n")
    
    for metric in ['one_step_mse', 'spectral_radius', 'rollout_divergence']:
        if metric not in df.columns or df[metric].isna().all():
            continue
        
        insights.append(f"### {metric.replace('_', ' ').title()}\n\n")
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            
            self_model = dataset_df[dataset_df['model'] == 'self-model'][metric].values
            world_model = dataset_df[dataset_df['model'] == 'world-model'][metric].values
            
            if len(self_model) == 0 or len(world_model) == 0:
                continue
            
            self_val = self_model[0]
            world_val = world_model[0]
            
            if np.isnan(self_val) or np.isnan(world_val):
                continue
            
            winner = "self-model" if self_val < world_val else "world-model"
            ratio = world_val / self_val if self_val < world_val else self_val / world_val
            
            insights.append(f"- **{dataset}:** {winner} wins ({ratio:.2f}x better)\n")
        
        insights.append("\n")
    
    # Overall conclusions
    insights.append("## Universal Patterns\n\n")
    
    # Count wins per model per metric
    for metric in ['one_step_mse', 'spectral_radius']:
        if metric not in df.columns or df[metric].isna().all():
            continue
        
        self_wins = 0
        world_wins = 0
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            self_model = dataset_df[dataset_df['model'] == 'self-model'][metric].values
            world_model = dataset_df[dataset_df['model'] == 'world-model'][metric].values
            
            if len(self_model) > 0 and len(world_model) > 0:
                if self_model[0] < world_model[0]:
                    self_wins += 1
                else:
                    world_wins += 1
        
        insights.append(f"**{metric.replace('_', ' ').title()}:** Self-model wins {self_wins}/{self_wins+world_wins} datasets\n\n")
    
    # Save insights
    with open(output_dir / 'cross_dataset_insights.md', 'w') as f:
        f.writelines(insights)
    
    print(f"✅ Saved insights to {output_dir / 'cross_dataset_insights.md'}")
    
    # Print to console
    print("\n" + "="*60)
    print("".join(insights))
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Cross-dataset analysis')
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                      help='Results directory')
    parser.add_argument('--datasets', type=str, 
                      default='ar1,damped,vanderpol,ETTh1,ETTm1,Weather',
                      help='Comma-separated list of datasets')
    parser.add_argument('--output-dir', type=Path, default=Path('results/cross_dataset'),
                      help='Output directory')
    
    args = parser.parse_args()
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    print("="*60)
    print("Cross-Dataset Analysis")
    print("="*60)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    
    compare_datasets(args.results_dir, datasets, args.output_dir)


if __name__ == '__main__':
    main()
