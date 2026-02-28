#!/usr/bin/env python3
"""
Master Benchmark Script
Runs all required benchmarks:
1. ETTm1 Full Benchmark (5 seeds)
2. Traffic Benchmark
3. Damped Oscillator Self-Model Verification
4. VanderPol Chaotic Dynamics Benchmark
5. Cross-Dataset Pattern Analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import time

# Configuration
SEEDS = [42, 43, 44, 45, 46]
DEVICE = 'cpu'

# Task configurations
TASKS = {
    'ettm1': {
        'name': 'ETTm1 Full Benchmark',
        'dataset': 'ETTm1',
        'preset': 'ETTm1_short',
        'epochs': 50,  # Reduced for faster completion
    },
    'traffic': {
        'name': 'Traffic Benchmark',
        'dataset': 'traffic',
        'preset': 'traffic_short',
        'epochs': 30,  # Fewer epochs due to high dimensionality
    },
    'damped': {
        'name': 'Damped Oscillator Self-Model Verification',
        'dataset': 'damped',
        'type': 'deterministic',
    },
    'vanderpol': {
        'name': 'VanderPol Chaotic Dynamics',
        'dataset': 'vanderpol',
        'type': 'deterministic',
    },
}


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and report results."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        if result.returncode == 0:
            print(f"✓ {description} - SUCCESS")
            if result.stdout:
                print(result.stdout[-1000:])  # Last 1000 chars
            return True
        else:
            print(f"✗ {description} - FAILED")
            print(f"Stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False


def run_ettm1_benchmark():
    """Run ETTm1 full benchmark with 5 seeds."""
    print("\n" + "="*80)
    print("TASK 1: ETTm1 Full Benchmark (5 seeds)")
    print("="*80)
    
    results = {'self_model': [], 'world_model': []}
    
    for seed in SEEDS:
        # Self-model
        cmd = [
            sys.executable, 'experiments/train_timeseries_self_model.py',
            '--preset', 'ETTm1_short',
            '--dataset', 'ETTm1',
            '--seed', str(seed),
            '--epochs', '50',
            '--device', DEVICE,
        ]
        success = run_command(cmd, f"ETTm1 Self-Model Seed {seed}")
        if success:
            results['self_model'].append(seed)
        
        # World-model
        cmd = [
            sys.executable, 'experiments/train_timeseries_world_model.py',
            '--preset', 'ETTm1_short',
            '--dataset', 'ETTm1',
            '--seed', str(seed),
            '--epochs', '50',
            '--device', DEVICE,
        ]
        success = run_command(cmd, f"ETTm1 World-Model Seed {seed}")
        if success:
            results['world_model'].append(seed)
    
    # Generate comparison report
    cmd = [
        sys.executable, 'experiments/benchmark_timeseries_comparison.py',
        '--preset', 'ETTm1_short',
        '--dataset', 'ETTm1',
        '--seeds', *[str(s) for s in SEEDS],
        '--skip-training',
        '--device', DEVICE,
    ]
    run_command(cmd, "ETTm1 Comparison Report")
    
    return results


def run_traffic_benchmark():
    """Run Traffic benchmark with 5 seeds."""
    print("\n" + "="*80)
    print("TASK 2: Traffic Benchmark")
    print("="*80)
    
    results = {'self_model': [], 'world_model': []}
    
    for seed in SEEDS:
        # Self-model
        cmd = [
            sys.executable, 'experiments/train_timeseries_self_model.py',
            '--preset', 'traffic_short',
            '--dataset', 'traffic',
            '--seed', str(seed),
            '--epochs', '30',
            '--device', DEVICE,
        ]
        success = run_command(cmd, f"Traffic Self-Model Seed {seed}")
        if success:
            results['self_model'].append(seed)
        
        # World-model - skip for now due to high dimensionality (862 features)
        # Would require too much memory
        print(f"Skipping Traffic World-Model (862 features too high)")
    
    # Generate comparison report
    cmd = [
        sys.executable, 'experiments/benchmark_timeseries_comparison.py',
        '--preset', 'traffic_short',
        '--dataset', 'traffic',
        '--seeds', *[str(s) for s in SEEDS],
        '--skip-training',
        '--device', DEVICE,
    ]
    run_command(cmd, "Traffic Comparison Report")
    
    return results


def run_damped_oscillator():
    """Run Damped Oscillator self-model verification."""
    print("\n" + "="*80)
    print("TASK 3: Damped Oscillator Self-Model Verification")
    print("="*80)
    
    # First generate the deterministic data
    cmd = [
        sys.executable, '-c',
        'from deterministic_data.deterministic_dataset import main; main()',
    ]
    # Generate damped data
    run_command([sys.executable, 'deterministic_data/deterministic_dataset.py', 
                '--dataset', 'damped'], "Generate Damped Oscillator Data")
    
    # Generate AR1 data for comparison
    run_command([sys.executable, 'deterministic_data/deterministic_dataset.py', 
                '--dataset', 'ar1'], "Generate AR1 Data")
    
    # Train self-model on damped oscillator
    results = []
    for seed in SEEDS[:3]:  # Use 3 seeds for deterministic
        cmd = [
            sys.executable, 'experiments/train_timeseries_self_model.py',
            '--preset', 'ETTh1_short',  # Use as template
            '--dataset', 'damped',
            '--seed', str(seed),
            '--epochs', '50',
            '--device', DEVICE,
        ]
        success = run_command(cmd, f"Damped Self-Model Seed {seed}")
        if success:
            results.append(seed)
    
    return {'self_model': results}


def run_vanderpol():
    """Run VanderPol chaotic dynamics benchmark."""
    print("\n" + "="*80)
    print("TASK 4: VanderPol Chaotic Dynamics Benchmark")
    print("="*80)
    
    # Generate VanderPol data
    run_command([sys.executable, 'deterministic_data/deterministic_dataset.py', 
                '--dataset', 'vanderpol'], "Generate VanderPol Data")
    
    results = []
    for seed in SEEDS[:3]:  # Use 3 seeds
        cmd = [
            sys.executable, 'experiments/train_timeseries_self_model.py',
            '--preset', 'ETTh1_short',
            '--dataset', 'vanderpol',
            '--seed', str(seed),
            '--epochs', '50',
            '--device', DEVICE,
        ]
        success = run_command(cmd, f"VanderPol Self-Model Seed {seed}")
        if success:
            results.append(seed)
    
    return {'self_model': results}


def run_cross_dataset_analysis():
    """Run cross-dataset pattern analysis."""
    print("\n" + "="*80)
    print("TASK 5: Cross-Dataset Pattern Analysis")
    print("="*80)
    
    cmd = [
        sys.executable, 'scripts/analyze_cross_dataset_patterns.py',
        '--datasets', 'ar1,damped,vanderpol,ETTh1,ETTm1',
        '--results-dir', 'results',
        '--output-dir', 'results/cross_dataset',
    ]
    run_command(cmd, "Cross-Dataset Analysis")
    
    return {'status': 'complete'}


def main():
    parser = argparse.ArgumentParser(description='Master Benchmark Runner')
    parser.add_argument('--tasks', nargs='+', 
                       choices=['ettm1', 'traffic', 'damped', 'vanderpol', 'cross_dataset', 'all'],
                       default=['all'],
                       help='Tasks to run')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    
    if 'all' in args.tasks:
        tasks_to_run = ['ettm1', 'traffic', 'damped', 'vanderpol', 'cross_dataset']
    else:
        tasks_to_run = args.tasks
    
    print("="*80)
    print("MASTER BENCHMARK RUNNER")
    print("="*80)
    print(f"Tasks: {', '.join(tasks_to_run)}")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    
    all_results = {}
    start_time = time.time()
    
    for task in tasks_to_run:
        task_start = time.time()
        
        if task == 'ettm1':
            all_results['ettm1'] = run_ettm1_benchmark()
        elif task == 'traffic':
            all_results['traffic'] = run_traffic_benchmark()
        elif task == 'damped':
            all_results['damped'] = run_damped_oscillator()
        elif task == 'vanderpol':
            all_results['vanderpol'] = run_vanderpol()
        elif task == 'cross_dataset':
            all_results['cross_dataset'] = run_cross_dataset_analysis()
        
        task_time = time.time() - task_start
        print(f"\n✓ Task '{task}' completed in {task_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    
    # Save summary
    summary = {
        'tasks_completed': tasks_to_run,
        'seeds': SEEDS,
        'device': DEVICE,
        'total_time_minutes': total_time / 60,
        'results': {k: str(v) for k, v in all_results.items()},
    }
    
    with open('results/benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL BENCHMARKS COMPLETE")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Summary saved to: results/benchmark_summary.json")


if __name__ == '__main__':
    main()

