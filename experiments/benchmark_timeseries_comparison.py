"""
Benchmark comparison script for self-model vs world-model on Timeseries-PILE datasets.
Runs both models with multiple seeds and generates comparison report.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
from pathlib import Path
import subprocess
from typing import List, Dict
import numpy as np


def run_experiment(
    model_type: str,
    preset: str,
    dataset: str,
    seed: int,
    device: str = 'cpu',
) -> Dict:
    """Run a single experiment."""
    
    if model_type == 'self_model':
        script = 'experiments/train_timeseries_self_model.py'
    elif model_type == 'world_model':
        script = 'experiments/train_timeseries_world_model.py'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    cmd = [
        sys.executable, script,
        '--preset', preset,
        '--dataset', dataset,
        '--seed', str(seed),
        '--device', device,
    ]
    
    print(f"\nRunning {model_type} on {dataset} with seed {seed}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return {'success': True, 'output': result.stdout}
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return {'success': False, 'error': str(e)}


def load_experiment_results(
    output_dir: str,
    experiment_name: str,
) -> Dict:
    """Load results from a completed experiment."""
    exp_path = Path(output_dir) / experiment_name
    
    if not exp_path.exists():
        return None
    
    # Load history
    history_path = exp_path / 'history.json'
    if not history_path.exists():
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Load config
    config_path = exp_path / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract best validation loss
    if 'val_loss' in history and len(history['val_loss']) > 0:
        best_val_loss = min(history['val_loss'])
    else:
        best_val_loss = None
    
    return {
        'config': config,
        'history': history,
        'best_val_loss': best_val_loss,
    }


def aggregate_results(
    results: List[Dict],
) -> Dict:
    """Aggregate results across multiple seeds."""
    val_losses = [r['best_val_loss'] for r in results if r['best_val_loss'] is not None]
    
    if len(val_losses) == 0:
        return None
    
    return {
        'mean_val_loss': np.mean(val_losses),
        'std_val_loss': np.std(val_losses),
        'min_val_loss': np.min(val_losses),
        'max_val_loss': np.max(val_losses),
        'n_runs': len(val_losses),
    }


def generate_report(
    comparison_results: Dict,
    output_path: str,
):
    """Generate a markdown report comparing the models."""
    
    report_lines = [
        "# Timeseries-PILE Benchmark Comparison Report",
        "",
        f"**Dataset:** {comparison_results['dataset']}",
        f"**Preset:** {comparison_results['preset']}",
        f"**Number of seeds:** {comparison_results['n_seeds']}",
        "",
        "## Results Summary",
        "",
        "### Self-Model (RNN-based)",
        "",
    ]
    
    if comparison_results['self_model_aggregate'] is not None:
        sm_agg = comparison_results['self_model_aggregate']
        report_lines.extend([
            f"- **Mean validation loss:** {sm_agg['mean_val_loss']:.6f} ± {sm_agg['std_val_loss']:.6f}",
            f"- **Best validation loss:** {sm_agg['min_val_loss']:.6f}",
            f"- **Worst validation loss:** {sm_agg['max_val_loss']:.6f}",
            f"- **Successful runs:** {sm_agg['n_runs']}/{comparison_results['n_seeds']}",
        ])
    else:
        report_lines.append("- No successful runs")
    
    report_lines.extend([
        "",
        "### World-Model (VAE-based)",
        "",
    ])
    
    if comparison_results['world_model_aggregate'] is not None:
        wm_agg = comparison_results['world_model_aggregate']
        report_lines.extend([
            f"- **Mean validation loss:** {wm_agg['mean_val_loss']:.6f} ± {wm_agg['std_val_loss']:.6f}",
            f"- **Best validation loss:** {wm_agg['min_val_loss']:.6f}",
            f"- **Worst validation loss:** {wm_agg['max_val_loss']:.6f}",
            f"- **Successful runs:** {wm_agg['n_runs']}/{comparison_results['n_seeds']}",
        ])
    else:
        report_lines.append("- No successful runs")
    
    # Comparison
    report_lines.extend([
        "",
        "## Comparison",
        "",
    ])
    
    if (comparison_results['self_model_aggregate'] is not None and 
        comparison_results['world_model_aggregate'] is not None):
        
        sm_mean = comparison_results['self_model_aggregate']['mean_val_loss']
        wm_mean = comparison_results['world_model_aggregate']['mean_val_loss']
        
        if sm_mean < wm_mean:
            winner = "Self-Model"
            improvement = ((wm_mean - sm_mean) / wm_mean) * 100
        else:
            winner = "World-Model"
            improvement = ((sm_mean - wm_mean) / sm_mean) * 100
        
        report_lines.extend([
            f"**Winner:** {winner}",
            f"**Improvement:** {improvement:.2f}%",
            "",
            f"**Effect size:** {abs(sm_mean - wm_mean):.6f}",
        ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmark comparison between self-model and world-model'
    )
    parser.add_argument('--preset', type=str, default='ETTh1_short',
                       help='Preset configuration name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='Random seeds to use')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for training')
    parser.add_argument('--output-dir', type=str, default='results/timeseries_pile',
                       help='Output directory')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only generate report from existing results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TIMESERIES-PILE BENCHMARK COMPARISON")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Preset: {args.preset}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Run experiments
    if not args.skip_training:
        for seed in args.seeds:
            # Run self-model
            run_experiment(
                model_type='self_model',
                preset=args.preset,
                dataset=args.dataset,
                seed=seed,
                device=args.device,
            )
            
            # Run world-model
            run_experiment(
                model_type='world_model',
                preset=args.preset,
                dataset=args.dataset,
                seed=seed,
                device=args.device,
            )
    
    # Load results
    print("\n" + "="*80)
    print("LOADING RESULTS")
    print("="*80)
    
    self_model_results = []
    world_model_results = []
    
    for seed in args.seeds:
        # Self-model
        sm_name = f"{args.dataset}_self_model_seed{seed}_h64"  # Adjust based on config
        sm_result = load_experiment_results(args.output_dir, sm_name)
        if sm_result is not None:
            self_model_results.append(sm_result)
            print(f"Loaded self-model seed {seed}: val_loss = {sm_result['best_val_loss']:.6f}")
        else:
            print(f"Self-model seed {seed}: NOT FOUND")
        
        # World-model
        wm_name = f"{args.dataset}_world_model_seed{seed}_h64"
        wm_result = load_experiment_results(args.output_dir, wm_name)
        if wm_result is not None:
            world_model_results.append(wm_result)
            print(f"Loaded world-model seed {seed}: val_loss = {wm_result['best_val_loss']:.6f}")
        else:
            print(f"World-model seed {seed}: NOT FOUND")
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    sm_aggregate = aggregate_results(self_model_results)
    wm_aggregate = aggregate_results(world_model_results)
    
    if sm_aggregate is not None:
        print(f"\nSelf-Model: {sm_aggregate['mean_val_loss']:.6f} ± {sm_aggregate['std_val_loss']:.6f}")
    if wm_aggregate is not None:
        print(f"World-Model: {wm_aggregate['mean_val_loss']:.6f} ± {wm_aggregate['std_val_loss']:.6f}")
    
    # Generate report
    comparison_results = {
        'dataset': args.dataset,
        'preset': args.preset,
        'n_seeds': len(args.seeds),
        'self_model_aggregate': sm_aggregate,
        'world_model_aggregate': wm_aggregate,
        'self_model_results': self_model_results,
        'world_model_results': world_model_results,
    }
    
    # Save comparison results
    output_dir = Path(args.output_dir) / 'comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f'{args.dataset}_comparison.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        comparison_json = {
            'dataset': comparison_results['dataset'],
            'preset': comparison_results['preset'],
            'n_seeds': comparison_results['n_seeds'],
            'self_model_aggregate': {
                k: float(v) if isinstance(v, (np.number, np.ndarray)) else v
                for k, v in sm_aggregate.items()
            } if sm_aggregate else None,
            'world_model_aggregate': {
                k: float(v) if isinstance(v, (np.number, np.ndarray)) else v
                for k, v in wm_aggregate.items()
            } if wm_aggregate else None,
        }
        json.dump(comparison_json, f, indent=2)
    
    # Generate markdown report
    report_path = output_dir / f'{args.dataset}_comparison.md'
    generate_report(comparison_results, report_path)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_path}")
    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
