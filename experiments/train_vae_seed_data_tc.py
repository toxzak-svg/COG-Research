"""
Train VAE World-Model with Beta-TC-VAE + Contrastive Learning

Enhanced training with:
1. Beta-TC-VAE decomposition (prevents posterior collapse)
2. Contrastive loss (enforces semantic structure)
3. Semantic augmentation (10x data diversity)  
4. Free-bits regularization (prevents dimension collapse)

Usage:
    python experiments/train_vae_seed_data_tc.py --epochs 100 --device cpu
    python experiments/train_vae_seed_data_tc.py --augment-factor 10 --aggressive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webpage_data_collection import (
    AwwwardsLoader, WebpageState
)
from webpage_data_collection.semantic_augmentation import create_semantic_augmenter
from webpage_models import WebpageVAE, BetaTCVAELoss


class EnhancedWebpageDataset(torch.utils.data.Dataset):
    """
    Enhanced in-memory dataset with semantic augmentation
    """
    
    def __init__(self, states: List[WebpageState]):
        """
        Args:
            states: List of WebpageState objects (original + augmented)
        """
        self.states = states
        
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single training sample
        
        Returns:
            {
                'state_vector': (190,) tensor - Vectorized state
                'outcomes': (10,) tensor - Outcome metrics
                'num_components': int - Number of components (for contrastive loss)
            }
        """
        state = self.states[idx]
        
        # Vectorize state
        state_vector = self._vectorize_state(state)
        
        # Extract outcomes
        outcomes = self._extract_outcomes(state)
        
        # Extract num_components for contrastive loss
        num_components = len(state.components) if state.components else 0
        
        return {
            'state_vector': state_vector,
            'outcomes': outcomes,
            'num_components': num_components,
        }
    
    def _vectorize_state(self, state: WebpageState) -> torch.Tensor:
        """
        Convert WebpageState to 190D vector
        
        Enhanced version with more feature extraction.
        """
        vector = torch.zeros(190, dtype=torch.float32)
        
        # Visual/Design [0:80]
        if state.layout:
            vector[0] = min(len(state.layout), 10) / 10  # Depth
            # Layout algorithm distribution
            for i, layout in enumerate(state.layout[:5]):
                vector[1 + i] = 1.0 if layout.layout_type.value == "grid" else 0.0
                vector[6 + i] = 1.0 if layout.layout_type.value == "flex" else 0.0
        
        # Component features [10:30]
        if state.components:
            num_comp = len(state.components)
            vector[10] = min(num_comp, 20) / 20  # Count normalized
            
            # Component type distribution
            comp_types = [c.component_type.value for c in state.components]
            vector[11] = comp_types.count("hero") / max(num_comp, 1)
            vector[12] = comp_types.count("nav") / max(num_comp, 1)
            vector[13] = comp_types.count("footer") / max(num_comp, 1)
            vector[14] = comp_types.count("cta") / max(num_comp, 1)
            
            # Emphasis distribution (emphasis_level is int 0-5)
            emphases = [c.emphasis_level for c in state.components]
            vector[15] = sum(1 for e in emphases if e >= 4) / max(num_comp, 1)  # CRITICAL/HIGH
            vector[16] = sum(1 for e in emphases if e == 3) / max(num_comp, 1)  # MEDIUM
            vector[17] = sum(1 for e in emphases if e == 2) / max(num_comp, 1)  # LOW
            vector[18] = sum(1 for e in emphases if e <= 1) / max(num_comp, 1)  # MINIMAL
            
            # Average size levels
            avg_size = sum(c.size_level for c in state.components) / num_comp
            vector[19] = avg_size / 5  # Normalize to 0-1
            vector[20] = 0.5  # Placeholder
        
        # Content features [80:110]
        if state.components:
            # Text content lengths (if available)
            text_lengths = [len(c.content_text) if hasattr(c, 'content_text') and c.content_text else 0 
                           for c in state.components]
            avg_text_len = sum(text_lengths) / max(len(text_lengths), 1)
            vector[80] = min(avg_text_len / 1000, 1.0)  # Normalize to 1000 chars
        
        # Performance [110:130]
        if state.performance:
            vector[110] = state.performance.lighthouse_performance / 100 if state.performance.lighthouse_performance else 0.5
            vector[111] = state.performance.lighthouse_accessibility / 100 if state.performance.lighthouse_accessibility else 0.5
            vector[112] = state.performance.lighthouse_seo / 100 if state.performance.lighthouse_seo else 0.5
            vector[113] = min(state.performance.lcp_ms / 5000, 1.0) if state.performance.lcp_ms else 0.5
            vector[114] = min(state.performance.fid_ms / 1000, 1.0) if state.performance.fid_ms else 0.5
            vector[115] = min(state.performance.cls_score * 10, 1.0) if state.performance.cls_score else 0.5
        
        # Outcomes [130:140]
        if state.outcomes:
            vector[130] = state.outcomes.conversion_rate if state.outcomes.conversion_rate is not None else 0.02
            vector[131] = state.outcomes.bounce_rate if state.outcomes.bounce_rate is not None else 0.5
            vector[132] = min(state.outcomes.avg_time_on_page / 600, 1.0) if state.outcomes.avg_time_on_page is not None else 0.3
            vector[133] = state.outcomes.scroll_depth_avg if state.outcomes.scroll_depth_avg is not None else 0.6
            vector[134] = (state.outcomes.clicks_per_session / 10) if state.outcomes.clicks_per_session is not None else 0.3
        
        # Cognitive load [140:150]
        if state.cognitive_load:
            vector[140] = state.cognitive_load.complexity_score if state.cognitive_load.complexity_score is not None else 0.5
            vector[141] = state.cognitive_load.clutter_score if state.cognitive_load.clutter_score is not None else 0.5
            vector[142] = state.cognitive_load.visual_weight if state.cognitive_load.visual_weight is not None else 0.5
        
        # Context [150:170]
        # Device, viewport, etc. (simplified)
        vector[150] = 1.0 if state.device_type == "desktop" else 0.5 if state.device_type == "tablet" else 0.0
        vector[151] = 0.5  # Viewport normalized
        
        # Portfolio [170:190]
        if state.project_category:
            vector[170] = 1.0  # Has project category
            # Category encoding (simplified)
            cat = state.project_category
            vector[171] = 1.0 if cat == "saas" else 0.0
            vector[172] = 1.0 if cat == "ecommerce" else 0.0
            vector[173] = 1.0 if cat == "portfolio" else 0.0
        
        # Fill remaining with small noise
        mask = (vector == 0)
        vector[mask] = torch.rand(mask.sum()) * 0.01
        
        return vector
    
    def _extract_outcomes(self, state: WebpageState) -> torch.Tensor:
        """Extract outcome metrics as 10D tensor"""
        outcomes = torch.zeros(10, dtype=torch.float32)
        
        if state.outcomes:
            outcomes[0] = state.outcomes.conversion_rate if state.outcomes.conversion_rate is not None else 0.02
            outcomes[1] = state.outcomes.bounce_rate if state.outcomes.bounce_rate is not None else 0.5
            outcomes[2] = min(state.outcomes.avg_time_on_page / 600, 1.0) if state.outcomes.avg_time_on_page is not None else 0.3
            outcomes[3] = state.outcomes.scroll_depth_avg if state.outcomes.scroll_depth_avg is not None else 0.6
            outcomes[4] = (state.outcomes.clicks_per_session / 10) if state.outcomes.clicks_per_session is not None else 0.3
            outcomes[5] = (state.outcomes.pages_per_session / 5) if state.outcomes.pages_per_session is not None else 0.4
            outcomes[6] = state.outcomes.exit_rate if state.outcomes.exit_rate is not None else 0.5
            outcomes[7] = 0.3  # return visitor rate (not in schema)
            outcomes[8] = 1.0 - outcomes[1]  # inverse bounce rate
            outcomes[9] = outcomes[5]  # pages per session (duplicate)
        
        return outcomes


def load_and_augment_data(
    webalchemist_path: str,
    augment_factor: int = 10,
    aggressive: bool = False,
    seed: int = 42,
) -> Tuple[List[WebpageState], Dict]:
    """
    Load Awwwards seed data and apply semantic augmentation
    
    Returns:
        augmented_states: List of all states (original + augmented)
        stats: Dictionary of augmentation statistics
    """
    print(f"Loading Awwwards seed data from {webalchemist_path}...")
    
    # Load original states
    from webpage_data_collection.awwwards_loader import load_seed_data
    original_states = load_seed_data(webalchemist_path)
    
    print(f"[OK] Loaded {len(original_states)} original states")
    
    # Create augmenter
    augmenter = create_semantic_augmenter(
        augment_factor=augment_factor,
        aggressive=aggressive,
        seed=seed,
    )
    
    print(f"Applying semantic augmentation (factor={augment_factor})...")
    
    # Augment each state
    augmented_states = augmenter.augment_batch(original_states)
    
    print(f"[OK] Created {len(augmented_states)} total states")
    print(f"  ({len(original_states)} original + {len(augmented_states) - len(original_states)} augmented)")
    
    stats = {
        'num_original': len(original_states),
        'num_augmented': len(augmented_states) - len(original_states),
        'num_total': len(augmented_states),
        'augment_factor': augment_factor,
        'aggressive': aggressive,
    }
    
    return augmented_states, stats


def create_dataloaders(
    states: List[WebpageState],
    batch_size: int = 8,
    train_split: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Create dataset
    dataset = EnhancedWebpageDataset(states)
    
    # Split
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # For BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    print(f"[OK] Split dataset: {train_size} train, {val_size} val")
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: BetaTCVAELoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'index_code_mi': 0.0,
        'total_correlation': 0.0,
        'dimension_wise_kl': 0.0,
        'total_kl': 0.0,
        'outcome_prediction': 0.0,
        'contrastive': 0.0,
        'consistency': 0.0,
    }
    
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for batch in progress_bar:
        # Move to device
        x = batch['state_vector'].to(device)
        outcomes_true = batch['outcomes'].to(device)
        num_components = batch['num_components'].to(device)
        
        # Forward pass
        x_recon, outcomes_pred, mu, logvar = model(x, deterministic=False)
        
        # Reparameterize
        from webpage_models import reparameterize
        z = reparameterize(mu, logvar)
        
        # Compute β-TC-VAE loss
        total_loss, loss_dict = loss_fn(
            model=model,
            x=x,
            x_recon=x_recon,
            outcomes_pred=outcomes_pred,
            outcomes_true=outcomes_true,
            mu=mu,
            logvar=logvar,
            z=z,
            num_components=num_components,
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        for key, value in loss_dict.items():
            if key in total_losses:
                total_losses[key] += value
        
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'recon': f"{loss_dict['reconstruction']:.4f}",
            'TC': f"{loss_dict['total_correlation']:.4f}",
            'contrast': f"{loss_dict['contrastive']:.4f}",
        })
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: BetaTCVAELoss,
    device: str,
) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'index_code_mi': 0.0,
        'total_correlation': 0.0,
        'dimension_wise_kl': 0.0,
        'total_kl': 0.0,
        'outcome_prediction': 0.0,
        'contrastive': 0.0,
        'consistency': 0.0,
    }
    
    num_batches = 0
    
    for batch in val_loader:
        x = batch['state_vector'].to(device)
        outcomes_true = batch['outcomes'].to(device)
        num_components = batch['num_components'].to(device)
        
        # Forward pass
        x_recon, outcomes_pred, mu, logvar = model(x, deterministic=False)
        
        from webpage_models import reparameterize
        z = reparameterize(mu, logvar)
        
        # Compute loss
        _, loss_dict = loss_fn(
            model=model,
            x=x,
            x_recon=x_recon,
            outcomes_pred=outcomes_pred,
            outcomes_true=outcomes_true,
            mu=mu,
            logvar=logvar,
            z=z,
            num_components=num_components,
        )
        
        for key, value in loss_dict.items():
            if key in total_losses:
                total_losses[key] += value
        
        num_batches += 1
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: BetaTCVAELoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: str,
    epochs: int,
    checkpoint_dir: Path,
    early_stopping_patience: int = 20,
    beta_tc_max: float = 10.0,
    beta_tc_anneal_epochs: int = 40,
) -> Dict:
    """Main training loop"""
   
    print(f"Training VAE for {epochs} epochs...")
    print(f"beta_tc annealing: 0.0 -> {beta_tc_max} over {beta_tc_anneal_epochs} epochs")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print()
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'beta_tc_schedule': [],
        'learning_rates': [],
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1, epochs + 1):
        # Anneal β_tc
        beta_tc = min(beta_tc_max, epoch / beta_tc_anneal_epochs * beta_tc_max)
        loss_fn.update_beta_tc(beta_tc)
        
        # Train
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        
        # Validate
        val_losses = validate_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_losses'].append(train_losses)
        history['val_losses'].append(val_losses)
        history['beta_tc_schedule'].append(beta_tc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_losses['total']:.4f} | "
              f"Val Loss: {val_losses['total']:.4f} | "
              f"beta_tc: {beta_tc:.3f}")
        print(f"  Train - Recon: {train_losses['reconstruction']:.4f}, "
              f"TC: {train_losses['total_correlation']:.4f}, "
              f"Contrast: {train_losses['contrastive']:.4f}")
        print(f"  Val   - Recon: {val_losses['reconstruction']:.4f}, "
              f"TC: {val_losses['total_correlation']:.4f}, "
              f"Contrast: {val_losses['contrastive']:.4f}")
        
        # Check for improvement
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            epochs_without_improvement = 0
            
            # Save best checkpoint
            checkpoint_path = checkpoint_dir / f'vae_best_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'train_loss': train_losses['total'],
            }, checkpoint_path)
            
            print(f"  [OK] Saved best checkpoint (val_loss: {val_losses['total']:.4f})")
        else:
            epochs_without_improvement += 1
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"(no improvement for {early_stopping_patience} epochs)")
                break
        
        print()
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / 'vae_final.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_checkpoint_path)
    print(f"[OK] Saved final checkpoint: {final_checkpoint_path}")
    
    # Save history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history: {history_path}")
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {val_losses['total']:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train β-TC-VAE on Awwwards seed data')
    
    # Data
    parser.add_argument('--webalchemist-path', type=str,
                       default='c:/dev/projects/WebAlchemist',
                       help='Path to WebAlchemist data')
    parser.add_argument('--augment-factor', type=int, default=10,
                       help='Augmentation factor (# of augmented versions per original)')
    parser.add_argument('--aggressive', action='store_true',
                       help='Use aggressive augmentation probabilities')
    
    # Model
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                       default=[128, 96, 64],
                       help='Hidden layer dimensions')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Loss weights
    parser.add_argument('--beta-recon', type=float, default=1.0,
                       help='Reconstruction loss weight')
    parser.add_argument('--beta-index', type=float, default=1.0,
                       help='Index-code MI weight')
    parser.add_argument('--beta-tc', type=float, default=6.0,
                       help='Initial Total Correlation weight')
    parser.add_argument('--beta-tc-max', type=float, default=10.0,
                       help='Maximum Total Correlation weight (after annealing)')
    parser.add_argument('--beta-tc-anneal-epochs', type=int, default=40,
                       help='Epochs to anneal β_tc over')
    parser.add_argument('--beta-dwkl', type=float, default=1.0,
                       help='Dimension-wise KL weight')
    parser.add_argument('--beta-outcome', type=float, default=1.0,
                       help='Outcome prediction loss weight')
    parser.add_argument('--beta-contrast', type=float, default=0.5,
                       help='Contrastive loss weight')
    parser.add_argument('--beta-consistency', type=float, default=0.2,
                       help='Consistency loss weight')
    parser.add_argument('--free-bits', type=float, default=0.5,
                       help='Free bits per dimension')
    parser.add_argument('--target-kl', type=float, default=160.0,
                       help='Target KL for Lagrangian constraint')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str,
                       default='results/vae_tc_training',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("="*60)
    print("Beta-TC-VAE Training with Contrastive Learning")
    print("="*60)
    print(f"Configuration:")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Hidden dimensions: {args.hidden_dims}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Augmentation factor: {args.augment_factor}")
    print(f"  beta_tc: {args.beta_tc} -> {args.beta_tc_max} over {args.beta_tc_anneal_epochs} epochs")
    print(f"  beta_contrast: {args.beta_contrast}")
    print(f"  Free bits: {args.free_bits}")
    print()
    
    # Load and augment data
    states, aug_stats = load_and_augment_data(
        webalchemist_path=args.webalchemist_path,
        augment_factor=args.augment_factor,
        aggressive=args.aggressive,
        seed=args.seed,
    )
    print()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        states=states,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print()
    
    # Create model
    model = WebpageVAE(
        input_dim=190,
        latent_dim=args.latent_dim,
        hidden_dims_encoder=args.hidden_dims,
        hidden_dims_decoder=list(reversed(args.hidden_dims)),
        outcome_dim=10,
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Created VAE model ({total_params:,} parameters)")
    print()
    
    # Create β-TC-VAE loss
    loss_fn = BetaTCVAELoss(
        beta_recon=args.beta_recon,
        beta_index=args.beta_index,
        beta_tc=args.beta_tc,
        beta_dwkl=args.beta_dwkl,
        beta_outcome=args.beta_outcome,
        beta_contrast=args.beta_contrast,
        beta_consistency=args.beta_consistency,
        free_bits=args.free_bits,
        target_kl=args.target_kl,
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        epochs=args.epochs,
        checkpoint_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
        beta_tc_max=args.beta_tc_max,
        beta_tc_anneal_epochs=args.beta_tc_anneal_epochs,
    )
    
    print("\n" + "="*60)
    print("[OK] Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
