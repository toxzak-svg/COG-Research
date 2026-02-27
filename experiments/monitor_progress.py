"""
Progress monitoring script for running experiments.
Displays status of both Timeseries-PILE and verification experiments.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import time


def get_experiment_status(exp_dir: Path) -> Dict:
    """Get status of a single experiment."""
    history_path = exp_dir / 'history.json'
    config_path = exp_dir / 'config.json'
    
    if not history_path.exists():
        return {
            'status': 'not_started',
            'epochs': 0,
            'total_epochs': 100,
            'latest_val_loss': None,
            'best_val_loss': None,
            'progress_pct': 0.0,
        }
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])
    
    current_epoch = len(train_losses)
    total_epochs = config.get('epochs', 100)
    
    if current_epoch >= total_epochs:
        status = 'completed'
    elif current_epoch > 0:
        status = 'running'
    else:
        status = 'initializing'
    
    latest_val_loss = val_losses[-1] if val_losses else None
    best_val_loss = min(val_losses) if val_losses else None
    
    return {
        'status': status,
        'epochs': current_epoch,
        'total_epochs': total_epochs,
        'latest_val_loss': latest_val_loss,
        'best_val_loss': best_val_loss,
        'progress_pct': (current_epoch / total_epochs * 100) if total_epochs > 0 else 0,
    }


def monitor_timeseries_benchmark(results_dir: Path, dataset: str, seeds: List[int]):
    """Monitor Timeseries-PILE benchmark progress."""
    print(f"\n{'='*80}")
    print(f"TIMESERIES-PILE BENCHMARK: {dataset}")
    print(f"{'='*80}")
    
    total_experiments = len(seeds) * 2  # self-model + world-model
    completed = 0
    running = 0
    
    for seed in seeds:
        for model_type in ['self_model', 'world_model']:
            # Try to find experiment directory (hidden_dim may vary)
            exp_pattern = f"{dataset}_{model_type}_seed{seed}_h*"
            exp_dirs = list(results_dir.glob(exp_pattern))
            
            if not exp_dirs:
                print(f"  [{model_type:12s}] Seed {seed}: NOT STARTED")
                continue
            
            exp_dir = exp_dirs[0]
            status = get_experiment_status(exp_dir)
            
            if status['status'] == 'completed':
                completed += 1
                emoji = "✓"
                color = "DONE"
            elif status['status'] == 'running':
                running += 1
                emoji = "▶"
                color = "RUN"
            else:
                emoji = "○"
                color = "INIT"
            
            progress_bar = "█" * int(status['progress_pct'] / 5) + "░" * (20 - int(status['progress_pct'] / 5))
            
            val_loss_str = f"val={status['latest_val_loss']:.6f}" if status['latest_val_loss'] else ""
            best_str = f"best={status['best_val_loss']:.6f}" if status['best_val_loss'] else ""
            
            print(f"  {emoji} [{model_type:12s}] Seed {seed}: "
                  f"[{progress_bar}] {status['epochs']}/{status['total_epochs']} epochs "
                  f"({status['progress_pct']:.1f}%) {val_loss_str} {best_str}")
    
    print(f"\n  Summary: {completed}/{total_experiments} completed, {running} running")
    return completed, total_experiments


def monitor_verification_experiments(results_dir: Path, system: str, paradigm: str, seed_range: tuple):
    """Monitor verification experiment progress."""
    print(f"\n{'='*80}")
    print(f"VERIFICATION: {system.upper()} - {paradigm}")
    print(f"{'='*80}")
    
    seed_start, seed_end = seed_range
    seeds = range(seed_start, seed_end + 1)
    
    total_experiments = len(seeds)
    completed = 0
    running = 0
    
    for seed in seeds:
        # Look for experiment checkpoints in results directory
        # Structure: results/{system}/{paradigm}/{hidden_dim}/seed_{seed}.pth
        exp_pattern = f"{system}/{paradigm}/*/seed_{seed}.pth"
        exp_paths = list(results_dir.glob(exp_pattern))
        
        if not exp_paths:
            print(f"  [Seed {seed:2d}] NOT STARTED")
            continue
        
        # Check if experiment completed by looking at checkpoint
        import torch
        checkpoint = torch.load(exp_paths[0], map_location='cpu')
        
        current_epoch = checkpoint.get('training', {}).get('epochs', 0)
        val_mse = checkpoint.get('training', {}).get('last_val_mse', None)
        
        if current_epoch >= 50:
            completed += 1
            emoji = "✓"
            status = "DONE"
        else:
            running += 1
            emoji = "▶"
            status = "RUN"
        
        val_str = f"val_mse={val_mse:.6f}" if val_mse else ""
        print(f"  {emoji} [Seed {seed:2d}] {status} - {current_epoch} epochs {val_str}")
    
    print(f"\n  Summary: {completed}/{total_experiments} completed, {running} running")
    return completed, total_experiments


def main():
    parser = argparse.ArgumentParser(description='Monitor experiment progress')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in seconds (0 for single check)')
    parser.add_argument('--timeseries-dataset', type=str, default='ETTh1',
                       help='Timeseries dataset to monitor')
    parser.add_argument('--timeseries-seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                       help='Seeds for timeseries benchmark')
    parser.add_argument('--verification-system', type=str, default='damped',
                       help='Verification system to monitor')
    parser.add_argument('--verification-paradigm', type=str, default='world_model_first',
                       help='Verification paradigm')
    parser.add_argument('--verification-seed-range', type=int, nargs=2, default=[10, 20],
                       help='Verification seed range [start, end]')
    
    args = parser.parse_args()
    
    timeseries_dir = Path('results/timeseries_pile')
    verification_dir = Path('results')
    
    if args.interval == 0:
        # Single check
        print(f"\n{'='*80}")
        print(f"EXPERIMENT PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        ts_completed, ts_total = monitor_timeseries_benchmark(
            timeseries_dir, args.timeseries_dataset, args.timeseries_seeds
        )
        
        ver_completed, ver_total = monitor_verification_experiments(
            verification_dir, args.verification_system,
            args.verification_paradigm, tuple(args.verification_seed_range)
        )
        
        print(f"\n{'='*80}")
        print(f"OVERALL PROGRESS")
        print(f"{'='*80}")
        print(f"  Timeseries: {ts_completed}/{ts_total} ({ts_completed/ts_total*100:.1f}%)")
        print(f"  Verification: {ver_completed}/{ver_total} ({ver_completed/ver_total*100:.1f}%)")
        total_completed = ts_completed + ver_completed
        total = ts_total + ver_total
        print(f"\n  Total: {total_completed}/{total} ({total_completed/total*100:.1f}%)")
        print(f"{'='*80}\n")
    else:
        # Continuous monitoring
        try:
            while True:
                # Clear screen (platform-independent)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"\n{'='*80}")
                print(f"EXPERIMENT PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  (Updating every {args.interval} seconds, Press Ctrl+C to exit)")
                print(f"{'='*80}")
                
                ts_completed, ts_total = monitor_timeseries_benchmark(
                    timeseries_dir, args.timeseries_dataset, args.timeseries_seeds
                )
                
                ver_completed, ver_total = monitor_verification_experiments(
                    verification_dir, args.verification_system,
                    args.verification_paradigm, tuple(args.verification_seed_range)
                )
                
                print(f"\n{'='*80}")
                print(f"OVERALL PROGRESS")
                print(f"{'='*80}")
                print(f"  Timeseries: {ts_completed}/{ts_total} ({ts_completed/ts_total*100:.1f}%)")
                print(f"  Verification: {ver_completed}/{ver_total} ({ver_completed/ver_total*100:.1f}%)")
                total_completed = ts_completed + ver_completed
                total = ts_total + ver_total
                print(f"\n  Total: {total_completed}/{total} ({total_completed/total*100:.1f}%)")
                print(f"{'='*80}\n")
                
                if total_completed >= total:
                    print("🎉 ALL EXPERIMENTS COMPLETE! 🎉\n")
                    break
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.\n")


if __name__ == '__main__':
    main()
