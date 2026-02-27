"""Regenerate markdown summary from probe ablation JSON."""
import json
from pathlib import Path

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
    
    # Organize by variation type
    bootstrap_configs = [a for a in data['ablations'] if a['config']['name'] == 'bootstrap_samples']
    points_configs = [a for a in data['ablations'] if a['config']['name'] == 'probe_points']
    epsilon_configs = [a for a in data['ablations'] if a['config']['name'] == 'epsilon']
    
    # Bootstrap samples ablation
    if bootstrap_configs:
        lines.append("### Bootstrap Samples")
        lines.append("")
        lines.append("| Samples | Winner | Effect Size | Confidence |")
        lines.append("|---------|--------|-------------|------------|")
        for cfg in bootstrap_configs:
            winner = cfg['winner']
            winner_str = "Self-first" if winner == "self_model_first" else "World-first" if winner == "world_model_first" else "No winner"
            if winner != "no_winner":
                effect = cfg['statistics']['cohens_d']
                ci_low, ci_high = cfg['statistics']['bootstrap_ci']
                lines.append(f"| {cfg['config']['value']} | {winner_str} | {effect:.3f} | [{ci_low:.3f}, {ci_high:.3f}] |")
            else:
                lines.append(f"| {cfg['config']['value']} | {winner_str} | — | — |")
        lines.append("")
    
    # Probe points ablation
    if points_configs:
        lines.append("### Probe Points")
        lines.append("")
        lines.append("| Points | Winner | Effect Size | Confidence |")
        lines.append("|--------|--------|-------------|------------|")
        for cfg in points_configs:
            winner = cfg['winner']
            winner_str = "Self-first" if winner == "self_model_first" else "World-first" if winner == "world_model_first" else "No winner"
            if winner != "no_winner":
                effect = cfg['statistics']['cohens_d']
                ci_low, ci_high = cfg['statistics']['bootstrap_ci']
                lines.append(f"| {cfg['config']['value']} | {winner_str} | {effect:.3f} | [{ci_low:.3f}, {ci_high:.3f}] |")
            else:
                lines.append(f"| {cfg['config']['value']} | {winner_str} | — | — |")
        lines.append("")
    
    # Epsilon ablation
    if epsilon_configs:
        lines.append("### Perturbation Scale (ε)")
        lines.append("")
        lines.append("| Epsilon | Winner | Effect Size | Confidence |")
        lines.append("|---------|--------|-------------|------------|")
        for cfg in epsilon_configs:
            winner = cfg['winner']
            winner_str = "Self-first" if winner == "self_model_first" else "World-first" if winner == "world_model_first" else "No winner"
            if winner != "no_winner":
                effect = cfg['statistics']['cohens_d']
                ci_low, ci_high = cfg['statistics']['bootstrap_ci']
                lines.append(f"| {cfg['config']['epsilon']} | {winner_str} | {effect:.3f} | [{ci_low:.3f}, {ci_high:.3f}] |")
            else:
                lines.append(f"| {cfg['config']['epsilon']} | {winner_str} | — | — |")
        lines.append("")
    
    lines.append("## Winner Summary")
    lines.append("")
    
    # Count winners
    winners = [a['winner'] for a in data['ablations']]
    self_count = winners.count('self_model_first')
    world_count = winners.count('world_model_first')
    no_winner_count = winners.count('no_winner')
    
    lines.append(f"- Self-model-first: {self_count}/{len(winners)} configurations")
    lines.append(f"- World-model-first: {world_count}/{len(winners)} configurations")
    lines.append(f"- No clear winner: {no_winner_count}/{len(winners)} configurations")
    lines.append("")
    
    # Visual summary grid
    lines.append("### Configuration Grid")
    lines.append("")
    lines.append("✓S = Self-model-first wins | ✓W = World-model-first wins | — = No clear winner")
    lines.append("")
    lines.append("| Parameter | Configuration | Result |")
    lines.append("|-----------|---------------|--------|")
    for cfg in data['ablations']:
        param = cfg['config']['name']
        value = cfg['config']['value'] if 'value' in cfg['config'] else cfg['config']['epsilon']
        winner = cfg['winner']
        if winner == 'self_model_first':
            result_symbol = "✓S"
        elif winner == 'world_model_first':
            result_symbol = "✓W"
        else:
            result_symbol = "—"
        lines.append(f"| {param} | {value} | {result_symbol} |")
    lines.append("")
    
    lines.append("## Interpretation")
    lines.append("")
    lines.append("**Key:** ✓S = Self-model-first wins, ✓W = World-model-first wins, — = No clear winner")
    lines.append("")
    lines.append("If conclusions are consistent across parameter variations, this indicates robustness.")
    lines.append("")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding='utf-8')

if __name__ == "__main__":
    with open('results/probe_ablation.json') as f:
        data = json.load(f)
    generate_summary(data, Path('results/probe_ablation.md'))
    print("Summary saved to results/probe_ablation.md")
