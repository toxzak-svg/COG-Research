"""Regenerate markdown summary from probe ablation JSON."""
import json
from collections import Counter
from pathlib import Path

def determine_winner(ablation):
    """Determine overall winner from metrics."""
    metric_winners = []
    for metric_name, metric_data in ablation['metrics'].items():
        winner = metric_data.get('winner', 'no_winner')
        if winner == 'self_model_first_better':
            metric_winners.append('self')
        elif winner == 'world_model_first_better':
            metric_winners.append('world')
    
    counts = Counter(metric_winners)
    if counts['self'] > counts['world']:
        return 'self_model_first'
    elif counts['world'] > counts['self']:
        return 'world_model_first'
    else:
        return 'no_winner'

def identify_varied_param(config, baseline):
    """Identify which parameter differs from baseline."""
    if config['bootstrap_samples'] != baseline['bootstrap_samples']:
        return 'bootstrap_samples', config['bootstrap_samples']
    elif config['probe_points'] != baseline['probe_points']:
        return 'probe_points', config['probe_points']
    elif config['probe_eps'] != baseline['probe_eps']:
        return 'epsilon', config['probe_eps']
    else:
        return 'baseline', 'all'

def generate_summary(data, output_path):
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
    
    # Baseline config (first one)
    baseline = data['ablations'][0]['config']
    
    # Organize by variation type
    bootstrap_configs = []
    points_configs = []
    epsilon_configs = []
    
    for ablation in data['ablations']:
        param, value = identify_varied_param(ablation['config'], baseline)
        winner = determine_winner(ablation)
        ablation['winner'] = winner
        ablation['varied_param'] = param
        ablation['varied_value'] = value
        
        if param == 'bootstrap_samples' or param == 'baseline':
            bootstrap_configs.append(ablation)
        if param == 'probe_points' or param == 'baseline':
            points_configs.append(ablation)
        if param == 'epsilon' or param == 'baseline':
            epsilon_configs.append(ablation)
    
    # Sort by value
    bootstrap_configs.sort(key=lambda x: x['config']['bootstrap_samples'])
    points_configs.sort(key=lambda x: x['config']['probe_points'])
    epsilon_configs.sort(key=lambda x: x['config']['probe_eps'])
    
    # Bootstrap samples table
    lines.append("### Bootstrap Samples")
    lines.append("")
    lines.append("| Samples | One-Step MSE | Rollout Div | Spectral Radius | Perturbation Return | **Overall Winner** |")
    lines.append("|---------|--------------|-------------|-----------------|---------------------|--------------------|")
    for cfg in bootstrap_configs:
        samples = cfg['config']['bootstrap_samples']
        winners = []
        for metric in ['one_step_mse', 'rollout_divergence_50', 'spectral_radius', 'perturbation_return_rate']:
            w = cfg['metrics'][metric]['winner']
            if w == 'self_model_first_better':
                winners.append('✓S')
            elif w == 'world_model_first_better':
                winners.append('✓W')
            else:
                winners.append('—')
        overall = cfg['winner']
        overall_str = "**Self-first**" if overall == 'self_model_first' else "**World-first**" if overall == 'world_model_first' else "Tie"
        lines.append(f"| {samples} | {winners[0]} | {winners[1]} | {winners[2]} | {winners[3]} | {overall_str} |")
    lines.append("")
    
    # Probe points table
    lines.append("### Probe Points")
    lines.append("")
    lines.append("| Points | One-Step MSE | Rollout Div | Spectral Radius | Perturbation Return | **Overall Winner** |")
    lines.append("|--------|--------------|-------------|-----------------|---------------------|--------------------|")
    for cfg in points_configs:
        points = cfg['config']['probe_points']
        winners = []
        for metric in ['one_step_mse', 'rollout_divergence_50', 'spectral_radius', 'perturbation_return_rate']:
            w = cfg['metrics'][metric]['winner']
            if w == 'self_model_first_better':
                winners.append('✓S')
            elif w == 'world_model_first_better':
                winners.append('✓W')
            else:
                winners.append('—')
        overall = cfg['winner']
        overall_str = "**Self-first**" if overall == 'self_model_first' else "**World-first**" if overall == 'world_model_first' else "Tie"
        lines.append(f"| {points} | {winners[0]} | {winners[1]} | {winners[2]} | {winners[3]} | {overall_str} |")
    lines.append("")
    
    # Epsilon table
    lines.append("### Perturbation Scale (ε)")
    lines.append("")
    lines.append("| Epsilon | One-Step MSE | Rollout Div | Spectral Radius | Perturbation Return | **Overall Winner** |")
    lines.append("|---------|--------------|-------------|-----------------|---------------------|--------------------|")
    for cfg in epsilon_configs:
        eps = cfg['config']['probe_eps']
        winners = []
        for metric in ['one_step_mse', 'rollout_divergence_50', 'spectral_radius', 'perturbation_return_rate']:
            w = cfg['metrics'][metric]['winner']
            if w == 'self_model_first_better':
                winners.append('✓S')
            elif w == 'world_model_first_better':
                winners.append('✓W')
            else:
                winners.append('—')
        overall = cfg['winner']
        overall_str = "**Self-first**" if overall == 'self_model_first' else "**World-first**" if overall == 'world_model_first' else "Tie"
        lines.append(f"| {eps} | {winners[0]} | {winners[1]} | {winners[2]} | {winners[3]} | {overall_str} |")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    winners = [a['winner'] for a in data['ablations']]
    self_count = winners.count('self_model_first')
    world_count = winners.count('world_model_first')
    tie_count = winners.count('no_winner')
    
    lines.append(f"- Self-model-first wins: {self_count}/{len(winners)} configurations")
    lines.append(f"- World-model-first wins: {world_count}/{len(winners)} configurations")
    lines.append(f"- Ties: {tie_count}/{len(winners)} configurations")
    lines.append("")
    
    lines.append("## Interpretation")
    lines.append("")
    lines.append("**Key:** ✓S = Self-model-first better on this metric, ✓W = World-model-first better, — = No clear winner")
    lines.append("")
    if self_count == len(winners):
        lines.append("✅ **Strong robustness**: Self-model-first wins consistently across all parameter variations.")
    elif world_count == len(winners):
        lines.append("✅ **Strong robustness**: World-model-first wins consistently across all parameter variations.")
    elif self_count > len(winners) * 0.8 or world_count > len(winners) * 0.8:
        lines.append("✅ **Good robustness**: One paradigm wins in >80% of configurations.")
    else:
        lines.append("⚠️ **Limited robustness**: Winner changes based on probe configuration.")
    lines.append("")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding='utf-8')

if __name__ == "__main__":
    with open('results/probe_ablation.json') as f:
        data = json.load(f)
    generate_summary(data, Path('results/probe_ablation.md'))
    print("Summary saved to results/probe_ablation.md")
