"""
Training script for HierarchicalSelfModel on Timeseries-PILE datasets.

Trains and evaluates the 3-level hierarchical self-model:
  Level 0: Fast predictions (1-10 steps) — same as flat SelfModel
  Level 1: Medium-term predictions (10-50 steps) via meta-learner
  Level 2: Goal-conditioned predictions via goal encoder

Compares against flat SelfModel baseline at multiple horizons.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from minimal_self_model.models.self_model import (
    SelfModel,
    HierarchicalSelfModel,
    MultiTimescaleSelfModel,
)
from timeseries_pile_data.forecasting_loader import create_dataloaders
from experiments.timeseries_config import TimeseriesExperimentConfig, get_config


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_flat_self_model(
    model: SelfModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train flat SelfModel for one epoch (next-step prediction)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for input_seq, target_seq in dataloader:
        input_seq = input_seq.to(device)   # (B, seq_len, F)
        target_seq = target_seq.to(device)  # (B, pred_len, F)

        full_seq = torch.cat([input_seq, target_seq], dim=1)  # (B, T, F)
        pred_len = target_seq.shape[1]

        optimizer.zero_grad()
        predictions = model(full_seq[:, :-1, :])          # (B, T-1, F)
        predictions = predictions[:, -pred_len:, :]        # (B, pred_len, F)
        loss = nn.functional.mse_loss(predictions, target_seq)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


def train_hierarchical_level0(
    model: HierarchicalSelfModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train Level 0 of HierarchicalSelfModel (fast next-step prediction)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for input_seq, target_seq in dataloader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        full_seq = torch.cat([input_seq, target_seq], dim=1)
        pred_len = target_seq.shape[1]

        optimizer.zero_grad()
        predictions = model(full_seq[:, :-1, :], level=0)  # Level 0 forward
        predictions = predictions[:, -pred_len:, :]
        loss = nn.functional.mse_loss(predictions, target_seq)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


def train_hierarchical_level1(
    model: HierarchicalSelfModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train Level 1 of HierarchicalSelfModel (medium-term meta-learner)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for input_seq, target_seq in dataloader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()

        # Level 1 predicts a sequence of medium-term steps
        level1_preds = model(input_seq, level=1)  # (B, level1_iterations, F)

        # Align with target: use first min(pred_len, level1_iterations) steps
        pred_len = min(target_seq.shape[1], level1_preds.shape[1])
        loss = nn.functional.mse_loss(
            level1_preds[:, :pred_len, :],
            target_seq[:, :pred_len, :]
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


def train_multiscale(
    model: MultiTimescaleSelfModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train MultiTimescaleSelfModel — predicts at all horizons simultaneously."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for input_seq, target_seq in dataloader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()

        # Multi-scale model predicts from the last hidden state
        predictions = model(input_seq)  # dict: {horizon: (B, F)}

        # Compute loss for each horizon that fits within target_seq
        pred_len = target_seq.shape[1]
        horizon_losses = []
        for h, pred in predictions.items():
            if h <= pred_len:
                # Target at step h-1 (0-indexed)
                target_at_h = target_seq[:, h - 1, :]  # (B, F)
                horizon_losses.append(nn.functional.mse_loss(pred, target_at_h))

        if horizon_losses:
            loss = torch.stack(horizon_losses).mean()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_flat(
    model: SelfModel,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            full_seq = torch.cat([input_seq, target_seq], dim=1)
            pred_len = target_seq.shape[1]

            predictions = model(full_seq[:, :-1, :])
            predictions = predictions[:, -pred_len:, :]
            loss = nn.functional.mse_loss(predictions, target_seq)

            total_loss += loss.item()
            n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


def evaluate_hierarchical(
    model: HierarchicalSelfModel,
    dataloader: DataLoader,
    device: str,
    level: int = 0,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            if level == 0:
                full_seq = torch.cat([input_seq, target_seq], dim=1)
                pred_len = target_seq.shape[1]
                predictions = model(full_seq[:, :-1, :], level=0)
                predictions = predictions[:, -pred_len:, :]
                loss = nn.functional.mse_loss(predictions, target_seq)
            elif level == 1:
                level1_preds = model(input_seq, level=1)
                pred_len = min(target_seq.shape[1], level1_preds.shape[1])
                loss = nn.functional.mse_loss(
                    level1_preds[:, :pred_len, :],
                    target_seq[:, :pred_len, :]
                )
            elif level == 2:
                level2_preds = model(input_seq, level=2)
                pred_len = min(target_seq.shape[1], level2_preds.shape[1])
                loss = nn.functional.mse_loss(
                    level2_preds[:, :pred_len, :],
                    target_seq[:, :pred_len, :]
                )
            else:
                raise ValueError(f"Invalid level: {level}")

            total_loss += loss.item()
            n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


def evaluate_multiscale(
    model: MultiTimescaleSelfModel,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    per_horizon_losses: Dict[int, List[float]] = {}
    n_batches = 0

    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            predictions = model(input_seq)
            pred_len = target_seq.shape[1]

            horizon_losses = []
            for h, pred in predictions.items():
                if h <= pred_len:
                    target_at_h = target_seq[:, h - 1, :]
                    h_loss = nn.functional.mse_loss(pred, target_at_h).item()
                    horizon_losses.append(h_loss)
                    per_horizon_losses.setdefault(h, []).append(h_loss)

            if horizon_losses:
                total_loss += np.mean(horizon_losses)
                n_batches += 1

    result = {'loss': total_loss / max(n_batches, 1)}
    for h, losses in per_horizon_losses.items():
        result[f'loss_h{h}'] = float(np.mean(losses))
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training_loop(
    model: nn.Module,
    model_name: str,
    train_fn,
    eval_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TimeseriesExperimentConfig,
    output_dir: Path,
    train_kwargs: Dict = None,  # type: ignore[assignment]
    eval_kwargs: Dict = None,   # type: ignore[assignment]
) -> Dict:
    """Generic training loop with early stopping and checkpointing."""
    train_kwargs = train_kwargs if train_kwargs is not None else {}
    eval_kwargs = eval_kwargs if eval_kwargs is not None else {}

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(config.epochs):
        train_metrics = train_fn(
            model, train_loader, optimizer, config.device,
            grad_clip=config.grad_clip, **train_kwargs
        )

        if epoch % config.eval_freq == 0:
            val_metrics = eval_fn(model, val_loader, config.device, **eval_kwargs)

            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])

            print(f"  Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train: {train_metrics['loss']:.6f} | "
                  f"Val: {val_metrics['loss']:.6f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                if config.save_checkpoints:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'config': config.__dict__,
                    }, output_dir / f'{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        scheduler.step()

    # Save history
    with open(output_dir / f'{model_name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"  Best val loss: {best_val_loss:.6f}")
    return {'best_val_loss': best_val_loss, 'history': history}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: TimeseriesExperimentConfig):
    set_seed(config.seed)

    exp_name = config.experiment_name or (
        f"{config.dataset_name}_hierarchical_seed{config.seed}_h{config.hidden_dim}"
    )
    output_dir = Path(config.output_dir) / 'hierarchical' / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    print(f"\nHierarchical Self-Model Experiment")
    print(f"Dataset: {config.dataset_name} | Seed: {config.seed}")
    print(f"Output: {output_dir}")

    # Load data
    print("\nLoading data...")
    dataloaders = create_dataloaders(
        dataset_name=config.dataset_name,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        batch_size=config.batch_size,
        data_root=config.data_root,
        normalize=config.normalize,
        num_workers=config.num_workers,
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    n_features = dataloaders['metadata']['features']
    print(f"Features: {n_features} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    all_results = {}

    # ------------------------------------------------------------------
    # 1. Flat SelfModel (baseline)
    # ------------------------------------------------------------------
    flat_model = SelfModel(
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        output_dim=n_features,
    ).to(config.device)

    flat_results = run_training_loop(
        model=flat_model,
        model_name='flat_self_model',
        train_fn=train_flat_self_model,
        eval_fn=evaluate_flat,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
    )
    all_results['flat_self_model'] = flat_results

    # ------------------------------------------------------------------
    # 2. HierarchicalSelfModel — Level 0 (same as flat, different arch)
    # ------------------------------------------------------------------
    hier_model = HierarchicalSelfModel(
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        output_dim=n_features,
        n_levels=2,
        level0_horizon=10,
        level1_horizon=config.pred_len,
    ).to(config.device)

    hier_l0_results = run_training_loop(
        model=hier_model,
        model_name='hierarchical_level0',
        train_fn=train_hierarchical_level0,
        eval_fn=evaluate_hierarchical,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        eval_kwargs={'level': 0},
    )
    all_results['hierarchical_level0'] = hier_l0_results

    # ------------------------------------------------------------------
    # 3. HierarchicalSelfModel — Level 1 (meta-learner)
    # ------------------------------------------------------------------
    hier_l1_results = run_training_loop(
        model=hier_model,
        model_name='hierarchical_level1',
        train_fn=train_hierarchical_level1,
        eval_fn=evaluate_hierarchical,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        eval_kwargs={'level': 1},
    )
    all_results['hierarchical_level1'] = hier_l1_results

    # ------------------------------------------------------------------
    # 4. MultiTimescaleSelfModel
    # ------------------------------------------------------------------
    # Determine valid horizons given pred_len
    all_horizons = [1, 5, 10, 24, 48]
    valid_horizons = [h for h in all_horizons if h <= config.pred_len]
    if not valid_horizons:
        valid_horizons = [1]

    multiscale_model = MultiTimescaleSelfModel(
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        output_dim=n_features,
        horizons=valid_horizons,
    ).to(config.device)

    multiscale_results = run_training_loop(
        model=multiscale_model,
        model_name='multiscale_self_model',
        train_fn=train_multiscale,
        eval_fn=evaluate_multiscale,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
    )
    all_results['multiscale_self_model'] = multiscale_results

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("HIERARCHICAL SELF-MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Best Val Loss':>15} {'Params':>12}")
    print(f"{'-'*60}")

    model_param_counts = {
        'flat_self_model': sum(p.numel() for p in flat_model.parameters()),
        'hierarchical_level0': sum(p.numel() for p in hier_model.parameters()),
        'hierarchical_level1': sum(p.numel() for p in hier_model.parameters()),
        'multiscale_self_model': sum(p.numel() for p in multiscale_model.parameters()),
    }

    for name, result in all_results.items():
        params = model_param_counts.get(name, 0)
        print(f"  {name:<28} {result['best_val_loss']:>15.6f} {params:>12,}")

    # Determine winner
    best_model = min(all_results, key=lambda k: all_results[k]['best_val_loss'])
    flat_loss = all_results['flat_self_model']['best_val_loss']
    best_loss = all_results[best_model]['best_val_loss']
    improvement = ((flat_loss - best_loss) / flat_loss) * 100 if flat_loss > 0 else 0.0

    print(f"\n  Winner: {best_model}")
    print(f"  Improvement over flat baseline: {improvement:.2f}%")

    # Save summary
    summary = {
        'dataset': config.dataset_name,
        'seed': config.seed,
        'pred_len': config.pred_len,
        'results': {
            name: {
                'best_val_loss': float(r['best_val_loss']),
                'n_params': model_param_counts.get(name, 0),
            }
            for name, r in all_results.items()
        },
        'winner': best_model,
        'improvement_over_flat_pct': float(improvement),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    report_lines = [
        "# Hierarchical Self-Model Experiment Results",
        "",
        f"**Dataset:** {config.dataset_name}",
        f"**Seed:** {config.seed}",
        f"**Prediction horizon:** {config.pred_len} steps",
        "",
        "## Model Comparison",
        "",
        "| Model | Best Val Loss | Parameters |",
        "|-------|--------------|------------|",
    ]
    for name, result in all_results.items():
        params = model_param_counts.get(name, 0)
        report_lines.append(
            f"| {name} | {result['best_val_loss']:.6f} | {params:,} |"
        )
    report_lines.extend([
        "",
        f"**Winner:** `{best_model}`",
        f"**Improvement over flat baseline:** {improvement:.2f}%",
        "",
        "## Key Insights",
        "",
        "- Level 0 (fast predictions): Direct next-step prediction, same as flat SelfModel",
        "- Level 1 (meta-learner): Context-conditioned medium-term predictions",
        "- MultiTimescale: Simultaneous predictions at multiple horizons",
        "",
        "## Interpretation",
        "",
        "If hierarchical > flat: Temporal abstraction helps for this dataset/horizon.",
        "If flat >= hierarchical: Single-level dynamics are sufficient; hierarchy adds overhead.",
    ])

    with open(output_dir / 'report.md', 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Report:  {output_dir / 'report.md'}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and compare hierarchical self-model architectures'
    )
    parser.add_argument('--preset', type=str, default='ETTh1_short',
                        help='Preset configuration name')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (overrides preset)')
    parser.add_argument('--pred-len', type=int, default=None,
                        help='Prediction horizon (overrides preset)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Hidden dimension (overrides preset)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides preset)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides preset)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for training (cpu/cuda)')

    args = parser.parse_args()

    config = get_config(args.preset)
    config.model_type = 'hierarchical_self_model'

    if args.dataset is not None:
        config.dataset_name = args.dataset
    if args.pred_len is not None:
        config.pred_len = args.pred_len
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.seed is not None:
        config.seed = args.seed
    config.device = args.device

    config.experiment_name = (
        f"{config.dataset_name}_hierarchical_seed{config.seed}_h{config.hidden_dim}"
    )

    main(config)
