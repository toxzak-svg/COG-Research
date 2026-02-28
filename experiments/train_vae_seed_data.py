"""
Train VAE World-Model on Awwwards Seed Data

Phase 1 Training: Learn webpage state compression and outcome prediction
from 24 award-winning design patterns.

Usage:
    python experiments/train_vae_seed_data.py [--epochs 100] [--device cpu]
    python experiments/train_vae_seed_data.py --batch-size 4 --lr 0.001
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
from webpage_models import WebpageVAE, VAELoss, anneal_beta_kl


class InMemoryWebpageDataset(torch.utils.data.Dataset):
    """
    Simple in-memory dataset for webpage states
    Used for seed training before full data collection
    """
    
    def __init__(self, states: List[WebpageState]):
        """
        Args:
            states: List of WebpageState objects
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
                'state_raw': WebpageState object
            }
        """
        state = self.states[idx]
        
        # Vectorize state
        state_vector = self._vectorize_state(state)
        
        # Extract outcomes
        outcomes = self._extract_outcomes(state)
        
        return {
            'state_vector': state_vector,
            'outcomes': outcomes,
            'state_raw': state,
        }
    
    def _vectorize_state(self, state: WebpageState) -> torch.Tensor:
        """
        Convert WebpageState to 190D vector
        
        For now, simplified version that creates a valid tensor.
        TODO: Implement full vectorization from dataset_builder.py
        """
        # Create placeholder 190D vector
        # Values should be in [0, 1] range for sigmoid outputs
        vector = torch.zeros(190, dtype=torch.float32)
        
        # Fill with some derived features
        # Layout features [0:10]
        if state.layout:
            vector[0] = min(len(state.layout), 10) / 10  # Normalize depth
            vector[1] = 1.0 if any(l.layout_type.value == "grid" for l in state.layout) else 0.0
            vector[2] = 1.0 if any(l.layout_type.value == "flex" for l in state.layout) else 0.0
        
        # Component features [10:20]
        if state.components:
            vector[10] = min(len(state.components), 20) / 20  # Normalize count
            # Component type distribution
            comp_types = [c.component_type.value for c in state.components]
            hero_count = comp_types.count("hero")
            nav_count = comp_types.count("nav")
            vector[11] = min(hero_count, 5) / 5
            vector[12] = min(nav_count, 5) / 5
        
        # Outcomes [130:140]
        if state.outcomes:
            vector[130] = state.outcomes.conversion_rate
            vector[131] = state.outcomes.bounce_rate
            vector[132] = min(state.outcomes.avg_time_on_page / 600, 1.0)  # Normalize to 10 min
            vector[133] = state.outcomes.scroll_depth_avg
            vector[134] = state.outcomes.cta_click_rate
        
        # Performance [110:120]
        if state.performance:
            if state.performance.lighthouse_performance is not None:
                vector[110] = state.performance.lighthouse_performance / 100
            if state.performance.lighthouse_accessibility is not None:
                vector[111] = state.performance.lighthouse_accessibility / 100
        
        # Fill remaining with small random noise to prevent all zeros
        mask = (vector == 0)
        vector[mask] = torch.rand(mask.sum()) * 0.01
        
        return vector
    
    def _extract_outcomes(self, state: WebpageState) -> torch.Tensor:
        """
        Extract outcome metrics as 10D tensor
        
        Returns:
            [conversion_rate, bounce_rate, avg_time_on_page_normalized,
             scroll_depth, cta_click_rate, form_start_rate,
             form_completion_rate, return_visitor_rate, 
             exit_rate, pages_per_session_normalized]
        """
        outcomes = torch.zeros(10, dtype=torch.float32)
        
        if state.outcomes:
            outcomes[0] = state.outcomes.conversion_rate
            outcomes[1] = state.outcomes.bounce_rate
            outcomes[2] = min(state.outcomes.avg_time_on_page / 600, 1.0)  # Normalize to [0, 1]
            outcomes[3] = state.outcomes.scroll_depth_avg
            outcomes[4] = state.outcomes.cta_click_rate
            outcomes[5] = state.outcomes.form_start_rate if hasattr(state.outcomes, 'form_start_rate') else 0.0
            outcomes[6] = state.outcomes.form_completion_rate if hasattr(state.outcomes, 'form_completion_rate') else 0.0
            # Fill remaining with reasonable defaults
            outcomes[7] = 0.3  # return visitor rate
            outcomes[8] = 1.0 - outcomes[1]  # exit rate (inverse of bounce)
            outcomes[9] = 0.2  # pages per session (normalized)
        
        return outcomes

class InMemoryWebpageDataset(torch.utils.data.Dataset):
    """
    Simple in-memory dataset for webpage states
    Used for seed training before full data collection
    """
    
    def __init__(self, states: List[WebpageState]):
        """
        Args:
            states: List of WebpageState objects
        """
        self.states = states
        
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                'state_vector': (190,) tensor - Vectorized state
                'outcomes': (10,) tensor - Outcome metrics
        """
        state = self.states[idx]
        state_vector = self._vectorize_state(state)
        outcomes = self._extract_outcomes(state)
        
        return {
            'state_vector': state_vector,
            'outcomes': outcomes,
        }
    
    def _vectorize_state(self, state: WebpageState) -> torch.Tensor:
        """
        Convert WebpageState to 190D vector
        
        NOTE: This is a PLACEHOLDER implementation for seed training.
        It creates valid tensors with basic derived features from the state.
        For production training, implement full vectorization matching 
        dataset_builder.py's WebpageStateDataset logic.
        
        Feature groups:
        - Visual/Design [0:80]: Layout structure, emphasis, colors
        - Content [80:110]: Text, CTAs, media
        - Performance [110:130]: Lighthouse scores
        - Outcomes [130:140]: Conversion, engagement
        - Cognitive load [140:150]: Complexity metrics
        - Context [150:170]: Device, traffic, segments
        - Portfolio [170:190]: Project metadata
        """
        features = []
        
        # Visual/Design features [0:80]
        # Use basic counts and derived metrics from available data
        layout_depth = len(state.layout) if state.layout else 0
        component_count = len(state.components) if state.components else 0
        features.extend([
            min(layout_depth / 30.0, 1.0),  # Normalize depth
            min(component_count / 50.0, 1.0),  # Normalize count
            0.5,  # Placeholder: whitespace_ratio
        ])
        features.extend([0.5] * 77)  # Fill rest with neutral values
        
        # Content features [80:110]
        headline_len = len(state.headline) if state.headline else 0
        cta_primary_len = len(state.cta_primary) if state.cta_primary else 0
        features.extend([
            min(headline_len / 100.0, 1.0),
            min(cta_primary_len / 50.0, 1.0),
            1.0 if state.cta_primary else 0.0,
            1.0 if state.cta_secondary else 0.0,
        ])
        features.extend([0.5] * 26)
        
        # Performance features [110:130]
        # Use performance data if available, otherwise neutral values
        if state.performance:
            perf = state.performance
            features.extend([
                perf.lighthouse_performance / 100.0 if hasattr(perf, 'lighthouse_performance') else 0.8,
                perf.lighthouse_accessibility / 100.0 if hasattr(perf, 'lighthouse_accessibility') else 0.8,
                perf.lighthouse_best_practices / 100.0 if hasattr(perf, 'lighthouse_best_practices') else 0.8,
                perf.lighthouse_seo / 100.0 if hasattr(perf, 'lighthouse_seo') else 0.8,
            ])
        else:
            features.extend([0.8, 0.8, 0.8, 0.8])  # Default good scores
        features.extend([0.5] * 16)
        
        # Outcomes [130:140]
        outcomes_vec = self._extract_outcomes(state).tolist()
        features.extend(outcomes_vec)
        
        # Cognitive load [140:150]
        if state.cognitive_load:
            cog = state.cognitive_load
            features.extend([
                cog.clutter_score if hasattr(cog, 'clutter_score') else 0.5,
                cog.hierarchy_clarity if hasattr(cog, 'hierarchy_clarity') else 0.5,
                cog.visual_weight_balance if hasattr(cog, 'visual_weight_balance') else 0.5,
            ])
        else:
            features.extend([0.5, 0.5, 0.5])
        features.extend([0.5] * 7)
        
        # Context [150:170]
        features.extend([
            1.0 if state.device_type == 'desktop' else 0.5,
            1.0 if state.traffic_source == 'organic' else 0.5,
            1.0 if state.user_segment else 0.5,
        ])
        features.extend([0.5] * 17)
        
        # Portfolio [170:190]
        features.extend([
            1.0 if state.project_category else 0.5,
            min(len(state.tech_stack) / 5.0, 1.0) if state.tech_stack else 0.5,
            min(len(state.tags) / 10.0, 1.0) if state.tags else 0.5,
        ])
        features.extend([0.5] * 17)
        
        # Ensure exactly 190 features
        assert len(features) == 190, f"Expected 190 features, got {len(features)}"
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_outcomes(self, state: WebpageState) -> torch.Tensor:
        """
        Extract outcome metrics as 10D tensor
        
        Returns:
            [conversion_rate, bounce_rate, avg_time_normalized, scroll_depth,
             cta_click_rate, form_start_rate, form_completion_rate,
             return_visitor_rate, exit_rate, pages_per_session_normalized]
        """
        # Use defaults if outcomes not available (for seed data without real measurements)
        if not state.outcomes:
            # Return neutral/good default outcomes for award-winning designs
            return torch.tensor([
                0.05,  # conversion_rate: 5% (good)
                0.30,  # bounce_rate: 30% (good)
                0.60,  # avg_time_normalized: ~3min
                0.75,  # scroll_depth: 75%
                0.15,  # cta_click_rate: 15%
                0.10,  # form_start_rate: 10%
                0.70,  # form_completion_rate: 70%
                0.40,  # return_visitor_rate: 40%
                0.25,  # exit_rate: 25%
                0.60,  # pages_per_session_normalized: ~3 pages
            ], dtype=torch.float32)
        
        outcomes = state.outcomes
        
        # Helper to safely get and normalize values
        def safe_get(attr_name, default_val, normalizer=None):
            val = getattr(outcomes, attr_name, None)
            if val is None:
                return default_val
            if normalizer:
                return normalizer(val)
            return val
        
        # Extract available metrics, with defaults for missing values
        return torch.tensor([
            safe_get('conversion_rate', 0.05),
            safe_get('bounce_rate', 0.30),
            safe_get('avg_time_on_page', 180.0, lambda x: x / 300.0),  # Normalize ~5min
            safe_get('scroll_depth', 0.75),
            safe_get('cta_click_rate', 0.15),
            safe_get('form_start_rate', 0.10),
            safe_get('form_completion_rate', 0.70),
            safe_get('return_visitor_rate', 0.40),
            safe_get('exit_rate', 0.25),
            safe_get('pages_per_session', 3.0, lambda x: x / 5.0),  # Normalize ~5 pages
        ], dtype=torch.float32)

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_webpage_state(state: WebpageState, augmentation_id: int) -> WebpageState:
    """
    Apply random perturbations to create training variations
    
    Args:
        state: Original WebpageState
        augmentation_id: ID for deterministic augmentation (0-9)
    
    Returns:
        Augmented WebpageState
    """
    # For now, return original state
    # TODO: Implement augmentation strategies from design doc
    # - Component shuffling
    # - Emphasis jitter (±1 level)
    # - Spacing jitter (±1 level)
    # - Color palette shift (slight HSV perturbation)
    # - Outcome noise (±5% relative)
    
    return state


def load_seed_data(
    webalchemist_path: str = "c:/dev/projects/WebAlchemist",
    augment_factor: int = 5,
) -> List[WebpageState]:
    """
    Load Awwwards seed states with augmentation
    
    Args:
        webalchemist_path: Path to WebAlchemist project
        augment_factor: Number of augmentations per original state
    
    Returns:
        List of WebpageState objects (original + augmented)
    """
    print(f"Loading Awwwards seed data from {webalchemist_path}...")
    
    # Load original 24 states
    loader = AwwwardsLoader(webalchemist_path=webalchemist_path)
    original_states = loader.create_seed_states_from_patterns()
    
    print(f"✓ Loaded {len(original_states)} original states")
    
    # Apply augmentation
    all_states = original_states.copy()
    
    if augment_factor > 0:
        print(f"Applying {augment_factor}x augmentation...")
        for state in original_states:
            for aug_id in range(augment_factor):
                augmented = augment_webpage_state(state, aug_id)
                all_states.append(augmented)
        
        print(f"✓ Created {len(all_states)} total states ({len(original_states)} + {len(all_states) - len(original_states)} augmented)")
    
    return all_states


def create_datasets(
    states: List[WebpageState],
    train_ratio: float = 0.8,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Create train and validation datasets
    
    Args:
        states: List of WebpageState objects
        train_ratio: Fraction for training (rest for validation)
    
    Returns:
        train_dataset, val_dataset
    """
    # Create full dataset
    full_dataset = InMemoryWebpageDataset(states)
    
    # Split into train/val
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Split dataset: {train_size} train, {val_size} val")
    
    return train_dataset, val_dataset


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: VAELoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch
    
    Returns:
        Dictionary of average losses
    """
    model.train()
    
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'kl_divergence': 0.0,
        'outcome_prediction': 0.0,
        'consistency': 0.0,
    }
    
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for batch in progress_bar:
        # Move to device
        x = batch['state_vector'].to(device)
        outcomes_true = batch['outcomes'].to(device)
        
        # Forward pass
        x_recon, outcomes_pred, mu, logvar = model(x, deterministic=False)
        
        # Reparameterize for consistency loss
        from webpage_models import reparameterize
        z = reparameterize(mu, logvar)
        
        # Compute loss
        total_loss, loss_dict = loss_fn(
            model=model,
            x=x,
            x_recon=x_recon,
            outcomes_pred=outcomes_pred,
            outcomes_true=outcomes_true,
            mu=mu,
            logvar=logvar,
            z=z,
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        for key, value in loss_dict.items():
            total_losses[key] += value
        
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'recon': f"{loss_dict['reconstruction']:.4f}",
            'kl': f"{loss_dict['kl_divergence']:.4f}",
        })
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: VAELoss,
    device: str,
) -> Dict[str, float]:
    """
    Validate for one epoch
    
    Returns:
        Dictionary of average losses
    """
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'kl_divergence': 0.0,
        'outcome_prediction': 0.0,
        'consistency': 0.0,
    }
    
    num_batches = 0
    
    for batch in val_loader:
        # Move to device
        x = batch['state_vector'].to(device)
        outcomes_true = batch['outcomes'].to(device)
        
        # Forward pass (deterministic for validation)
        x_recon, outcomes_pred, mu, logvar = model(x, deterministic=True)
        z = mu  # Use mean for validation
        
        # Compute loss
        total_loss, loss_dict = loss_fn(
            model=model,
            x=x,
            x_recon=x_recon,
            outcomes_pred=outcomes_pred,
            outcomes_true=outcomes_true,
            mu=mu,
            logvar=logvar,
            z=z,
        )
        
        # Accumulate losses
        for key, value in loss_dict.items():
            total_losses[key] += value
        
        num_batches += 1
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_losses: Dict[str, float],
    val_losses: Dict[str, float],
    save_path: Path,
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    
    torch.save(checkpoint, save_path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: VAELoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: str,
    epochs: int,
    checkpoint_dir: Path,
    early_stopping_patience: int = 15,
    beta_kl_max: float = 0.5,
    beta_kl_anneal_epochs: int = 20,
):
    """
    Main training loop
    
    Args:
        model: VAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of epochs
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Epochs to wait for improvement
        beta_kl_max: Maximum β_kl value
        beta_kl_anneal_epochs: Epochs to anneal β_kl over
    """
    print(f"\nTraining VAE for {epochs} epochs on {device}...")
    print(f"β_kl annealing: 0.0 → {beta_kl_max} over {beta_kl_anneal_epochs} epochs")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'beta_kl_schedule': [],
        'learning_rates': [],
    }
    
    for epoch in range(1, epochs + 1):
        # Anneal β_kl
        beta_kl_current = anneal_beta_kl(epoch - 1, beta_kl_max, beta_kl_anneal_epochs)
        loss_fn.update_beta_kl(beta_kl_current)
        
        # Train
        train_losses = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        
        # Validate
        val_losses = validate_epoch(model, val_loader, loss_fn, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_losses'].append(train_losses)
        history['val_losses'].append(val_losses)
        history['beta_kl_schedule'].append(beta_kl_current)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_losses['total']:.4f} | "
              f"Val Loss: {val_losses['total']:.4f} | "
              f"β_kl: {beta_kl_current:.3f} | "
              f"LR: {current_lr:.6f}")
        print(f"  Train - Recon: {train_losses['reconstruction']:.4f}, "
              f"KL: {train_losses['kl_divergence']:.4f}, "
              f"Outcome: {train_losses['outcome_prediction']:.4f}")
        print(f"  Val   - Recon: {val_losses['reconstruction']:.4f}, "
              f"KL: {val_losses['kl_divergence']:.4f}, "
              f"Outcome: {val_losses['outcome_prediction']:.4f}")
        
        # Save checkpoint if best
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            
            save_path = checkpoint_dir / f"vae_best_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, save_path)
            print(f"  ✓ Saved best checkpoint (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
            break
        
        # Learning rate scheduler step
        scheduler.step()
        
        print()
    
    # Save final checkpoint
    final_path = checkpoint_dir / "vae_final.pt"
    save_checkpoint(model, optimizer, epoch, train_losses, val_losses, final_path)
    print(f"✓ Saved final checkpoint: {final_path}")
    
    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history: {history_path}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train VAE on Awwwards seed data")
    
    # Data
    parser.add_argument('--webalchemist-path', type=str, 
                       default="c:/dev/projects/WebAlchemist",
                       help="Path to WebAlchemist project")
    parser.add_argument('--augment-factor', type=int, default=5,
                       help="Number of augmentations per original state")
    
    # Model
    parser.add_argument('--latent-dim', type=int, default=32,
                       help="Latent space dimension")
    parser.add_argument('--dropout', type=float, default=0.1,
                       help="Dropout rate")
    
    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                       help="Batch size")
    parser.add_argument('--epochs', type=int, default=100,
                       help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help="Device to train on")
    
    # Loss weights
    parser.add_argument('--beta-recon', type=float, default=1.0,
                       help="Reconstruction loss weight")
    parser.add_argument('--beta-kl-max', type=float, default=0.5,
                       help="Maximum KL divergence weight")
    parser.add_argument('--beta-kl-anneal-epochs', type=int, default=20,
                       help="Epochs to anneal β_kl over")
    parser.add_argument('--beta-outcome', type=float, default=1.0,
                       help="Outcome prediction loss weight")
    parser.add_argument('--beta-consistency', type=float, default=0.2,
                       help="Consistency loss weight")
    
    # Optimization
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    
    # Output
    parser.add_argument('--output-dir', type=str, 
                       default="results/vae_seed_training",
                       help="Output directory for checkpoints")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VAE World-Model Training on Awwwards Seed Data")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_dir}")
    print()
    
    # Load seed data
    states = load_seed_data(
        webalchemist_path=args.webalchemist_path,
        augment_factor=args.augment_factor,
    )
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(states, train_ratio=0.8)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # Single worker for small dataset
        drop_last=True,  # Avoid BatchNorm issues with incomplete batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,  # Avoid BatchNorm issues with incomplete batches
    )
    
    print(f"✓ Created data loaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print()
    
    # Create model
    model = WebpageVAE(
        input_dim=190,
        latent_dim=args.latent_dim,
        outcome_dim=10,
        dropout=args.dropout,
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Created VAE model")
    print(f"  Total parameters: {total_params:,}")
    print()
    
    # Create loss function
    loss_fn = VAELoss(
        beta_recon=args.beta_recon,
        beta_kl=0.0,  # Will be annealed during training
        beta_outcome=args.beta_outcome,
        beta_consistency=args.beta_consistency,
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create learning rate scheduler
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
        beta_kl_max=args.beta_kl_max,
        beta_kl_anneal_epochs=args.beta_kl_anneal_epochs,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {min(h['total'] for h in history['val_losses']):.4f}")
    print(f"Final validation loss: {history['val_losses'][-1]['total']:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
