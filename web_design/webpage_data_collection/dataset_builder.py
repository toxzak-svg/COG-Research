"""
Dataset Builder
Assembles training datasets for VAE world-model and RNN self-model

Training data format:
- World-Model: (state_t, edit, state_t+1, outcome_delta)
- Self-Model: (latent_state_t, edit_history, next_edit, stability_metrics)

Dataset structure:
- Phase 1: 1000 sites × 30 days = ~30K state snapshots
- Phase 1: ~5K edit sequences (real designer changes)
- Target: 80/10/10 train/val/test split
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
import pickle

from .state_schema import (
    WebpageState, SiteSnapshot, EditRecord,
    OutcomeMetrics, LayoutTree, Component
)


class WebpageStateDataset(Dataset):
    """
    PyTorch Dataset for webpage state sequences
    
    Returns (state_t, edit, state_t+1, outcome_delta) tuples
    Used for training VAE world-model
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_sequences: Optional[int] = None,
        min_outcome_visitors: int = 50
    ):
        """
        Args:
            data_dir: Directory containing collected snapshots
            split: "train", "val", or "test"
            max_sequences: Maximum number of sequences to load (for debugging)
            min_outcome_visitors: Minimum visitors required for reliable outcomes
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.min_outcome_visitors = min_outcome_visitors
        
        # Load all site snapshots
        self.sequences = self._load_sequences(max_sequences)
        
        print(f"Loaded {len(self.sequences)} sequences for {split} split")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single training sample
        
        Returns:
            {
                'state_before': Vectorized state before edit
                'edit': Vectorized edit operation
                'state_after': Vectorized state after edit
                'outcome_delta': Outcome improvement (scalar)
                'state_before_raw': Original WebpageState object
                'state_after_raw': Original WebpageState object
                'edit_raw': Original EditRecord object
            }
        """
        sequence = self.sequences[idx]
        
        state_before = sequence['state_before']
        edit = sequence['edit']
        state_after = sequence['state_after']
        outcome_delta = sequence['outcome_delta']
        
        # Vectorize states and edit
        state_before_vec = self._vectorize_state(state_before)
        state_after_vec = self._vectorize_state(state_after)
        edit_vec = self._vectorize_edit(edit)
        
        return {
            'state_before': state_before_vec,
            'edit': edit_vec,
            'state_after': state_after_vec,
            'outcome_delta': torch.tensor(outcome_delta, dtype=torch.float32),
            'state_before_raw': state_before,
            'state_after_raw': state_after,
            'edit_raw': edit,
        }
    
    def _load_sequences(self, max_sequences: Optional[int]) -> List[Dict]:
        """
        Load edit sequences from collected site snapshots
        
        Each sequence is a (before_state, edit, after_state, outcome_delta) tuple
        """
        sequences = []
        
        # Load all site snapshots
        snapshots_dir = self.data_dir / "snapshots"
        if not snapshots_dir.exists():
            print(f"Warning: Snapshots directory not found: {snapshots_dir}")
            return sequences
        
        snapshot_files = list(snapshots_dir.glob("*_snapshot.json"))
        
        # Split into train/val/test (80/10/10)
        np.random.seed(42)
        np.random.shuffle(snapshot_files)
        
        n_train = int(0.8 * len(snapshot_files))
        n_val = int(0.1 * len(snapshot_files))
        
        if self.split == "train":
            snapshot_files = snapshot_files[:n_train]
        elif self.split == "val":
            snapshot_files = snapshot_files[n_train:n_train+n_val]
        elif self.split == "test":
            snapshot_files = snapshot_files[n_train+n_val:]
        
        # Load sequences from each snapshot
        for snapshot_file in snapshot_files:
            try:
                site_sequences = self._load_snapshot_sequences(snapshot_file)
                sequences.extend(site_sequences)
                
                if max_sequences and len(sequences) >= max_sequences:
                    break
            
            except Exception as e:
                print(f"Error loading snapshot {snapshot_file}: {e}")
                continue
        
        return sequences[:max_sequences] if max_sequences else sequences
    
    def _load_snapshot_sequences(self, snapshot_file: Path) -> List[Dict]:
        """
        Load edit sequences from a single site snapshot
        
        Returns list of (before, edit, after, delta) dicts
        """
        # In production: Load actual snapshot file
        # For now, placeholder
        
        # with open(snapshot_file, 'r') as f:
        #     snapshot_data = json.load(f)
        
        # Parse into SiteSnapshot object
        # snapshot = self._deserialize_snapshot(snapshot_data)
        
        # Build sequences from states and edits
        # sequences = []
        # for edit in snapshot.edits:
        #     before_state = self._find_state(snapshot.states, edit.before_state_id)
        #     after_state = self._find_state(snapshot.states, edit.after_state_id)
        #     
        #     if before_state and after_state:
        #         # Check if outcomes are reliable
        #         if (after_state.outcomes and 
        #             after_state.outcomes.unique_visitors >= self.min_outcome_visitors):
        #             
        #             outcome_delta = self._compute_outcome_delta(
        #                 before_state.outcomes,
        #                 after_state.outcomes
        #             )
        #             
        #             sequences.append({
        #                 'state_before': before_state,
        #                 'edit': edit,
        #                 'state_after': after_state,
        #                 'outcome_delta': outcome_delta
        #             })
        
        # Placeholder: Return empty list
        return []
    
    def _vectorize_state(self, state: WebpageState) -> torch.Tensor:
        """
        Convert WebpageState to fixed-size vector
        
        State vector includes:
        - Layout features (50 dims)
        - Component features (100 dims)
        - Styling features (30 dims)
        - Cognitive load metrics (10 dims)
        - Total: 190 dims (will be compressed to 32D by VAE)
        """
        # Layout features
        layout_features = self._extract_layout_features(state.layout)
        
        # Component features
        component_features = self._extract_component_features(state.components)
        
        # Styling features
        styling_features = self._extract_styling_features(state.styling)
        
        # Cognitive load features
        cognitive_features = self._extract_cognitive_features(state.cognitive_load)
        
        # Concatenate all features
        state_vec = torch.cat([
            torch.tensor(layout_features, dtype=torch.float32),
            torch.tensor(component_features, dtype=torch.float32),
            torch.tensor(styling_features, dtype=torch.float32),
            torch.tensor(cognitive_features, dtype=torch.float32),
        ])
        
        return state_vec
    
    def _extract_layout_features(self, layout: List[LayoutTree]) -> List[float]:
        """
        Extract fixed-size feature vector from layout tree
        
        Features (50 dims):
        - Element count (1)
        - Max depth (1)
        - Layout type histogram (5)
        - Padding level histogram (6)
        - Margin level histogram (6)
        - Gap level histogram (6)
        - Container ratio (1)
        - ... padding to 50
        """
        features = [0.0] * 50
        
        if not layout:
            return features
        
        # Element count
        features[0] = len(layout) / 100.0  # Normalize
        
        # Max depth
        max_depth = max([self._get_depth(elem, layout) for elem in layout], default=0)
        features[1] = max_depth / 10.0  # Normalize
        
        # Layout type histogram
        layout_types = [elem.layout_type.value for elem in layout]
        for i, ltype in enumerate(['block', 'flex', 'grid', 'absolute', 'inline']):
            features[2 + i] = layout_types.count(ltype) / max(len(layout), 1)
        
        # Padding level histogram
        padding_levels = [elem.padding_level for elem in layout]
        for level in range(6):
            features[7 + level] = padding_levels.count(level) / max(len(layout), 1)
        
        # Margin level histogram
        margin_levels = [elem.margin_level for elem in layout]
        for level in range(6):
            features[13 + level] = margin_levels.count(level) / max(len(layout), 1)
        
        # Gap level histogram  
        gap_levels = [elem.gap_level for elem in layout if elem.gap_level is not None]
        for level in range(6):
            features[19 + level] = gap_levels.count(level) / max(len(gap_levels), 1)
        
        # Container ratio
        containers = sum(1 for elem in layout if elem.is_container)
        features[25] = containers / max(len(layout), 1)
        
        return features
    
    def _extract_component_features(self, components: List[Component]) -> List[float]:
        """
        Extract fixed-size feature vector from components
        
        Features (100 dims):
        - Component count (1)
        - Component type histogram (10)
        - Emphasis level histogram (6)
        - Color role histogram (5)
        - Size level histogram (6)
        - Clickable ratio (1)
        - ... padding to 100
        """
        features = [0.0] * 100
        
        if not components:
            return features
        
        # Component count
        features[0] = len(components) / 50.0  # Normalize
        
        # Component type histogram
        comp_types = [c.component_type.value for c in components]
        for i, ctype in enumerate(['button', 'form', 'input', 'cta', 'card', 'hero', 
                                   'navigation', 'footer', 'image', 'text_block']):
            features[1 + i] = comp_types.count(ctype) / max(len(components), 1)
        
        # Emphasis level histogram
        emphasis_levels = [c.emphasis_level for c in components]
        for level in range(6):
            features[11 + level] = emphasis_levels.count(level) / max(len(components), 1)
        
        # Color role histogram
        color_roles = [c.color_role for c in components]
        for i, role in enumerate(['primary', 'secondary', 'accent', 'neutral', 'danger']):
            features[17 + i] = color_roles.count(role) / max(len(components), 1)
        
        # Size level histogram
        size_levels = [c.size_level for c in components]
        for level in range(6):
            features[22 + level] = size_levels.count(level) / max(len(components), 1)
        
        # Clickable ratio
        clickables = sum(1 for c in components if c.clickable)
        features[28] = clickables / max(len(components), 1)
        
        return features
    
    def _extract_styling_features(self, styling: Optional[dict]) -> List[float]:
        """
        Extract styling features
        
        Features (30 dims):
        - Color embeddings (15)
        - Font size/line height (2)
        - Border radius (1)
        - ... padding to 30
        """
        features = [0.0] * 30
        
        if not styling:
            return features
        
        # Color embeddings (RGB normalized)
        # Would extract RGB from hex colors
        # For now, placeholder
        
        # Font sizing
        features[15] = styling.get('base_font_size', 16) / 20.0
        features[16] = styling.get('line_height', 1.5) / 2.0
        
        # Border radius
        features[17] = styling.get('border_radius', 4) / 20.0
        
        return features
    
    def _extract_cognitive_features(self, cognitive_load: Optional[dict]) -> List[float]:
        """
        Extract cognitive load features
        
        Features (10 dims):
        - All CognitiveLoadMetrics fields
        """
        features = [0.0] * 10
        
        if not cognitive_load:
            return features
        
        features[0] = cognitive_load.get('visual_clutter_score', 0.5)
        features[1] = cognitive_load.get('color_variety_score', 0.5)
        features[2] = cognitive_load.get('hierarchy_depth', 5) / 10.0
        features[3] = cognitive_load.get('cta_visibility', 0.7)
        features[4] = cognitive_load.get('navigation_complexity', 0.5)
        features[5] = cognitive_load.get('content_density', 0.5)
        features[6] = cognitive_load.get('f_pattern_score', 0.7)
        
        return features
    
    def _vectorize_edit(self, edit: EditRecord) -> torch.Tensor:
        """
        Convert EditRecord to fixed-size vector
        
        Edit vector includes:
        - Edit type (one-hot, 20 dims)
        - Edit parameters (embeddings, 30 dims)
        - Total: 50 dims
        """
        features = [0.0] * 50
        
        # Edit type one-hot
        edit_types = [
            'add_component', 'remove_component', 'modify_component',
            'change_layout', 'change_styling', 'change_copy',
            'reorder_elements', 'change_colors', 'change_spacing',
            'change_sizing', 'add_animation', 'remove_animation',
            'change_cta', 'change_headline', 'change_image',
            'optimize_performance', 'improve_accessibility', 'add_interaction',
            'change_navigation', 'other'
        ]
        
        if edit.edit_type in edit_types:
            idx = edit_types.index(edit.edit_type)
            features[idx] = 1.0
        else:
            features[-1] = 1.0  # 'other'
        
        # Edit parameters (simplified embedding)
        # Would use more sophisticated encoding in production
        if edit.edit_params:
            # Placeholder: hash parameters to fixed positions
            param_str = json.dumps(edit.edit_params, sort_keys=True)
            param_hash = hash(param_str) % 30
            features[20 + param_hash] = 1.0
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_outcome_delta(
        self,
        before_outcomes: Optional[OutcomeMetrics],
        after_outcomes: Optional[OutcomeMetrics]
    ) -> float:
        """
        Compute single outcome delta score (primary outcome)
        
        Prioritizes conversion rate, falls back to engagement metrics
        """
        if not before_outcomes or not after_outcomes:
            return 0.0
        
        # Primary: Conversion rate delta
        if (before_outcomes.conversion_rate is not None and 
            after_outcomes.conversion_rate is not None):
            return after_outcomes.conversion_rate - before_outcomes.conversion_rate
        
        # Fallback: Time on page delta
        if (before_outcomes.avg_time_on_page is not None and
            after_outcomes.avg_time_on_page is not None):
            # Normalize to similar scale as conversion rate
            return (after_outcomes.avg_time_on_page - before_outcomes.avg_time_on_page) / 200.0
        
        # Fallback: Bounce rate delta (negative is good)
        if (before_outcomes.bounce_rate is not None and
            after_outcomes.bounce_rate is not None):
            return -(after_outcomes.bounce_rate - before_outcomes.bounce_rate)
        
        return 0.0
    
    def _get_depth(self, elem: LayoutTree, layout: List[LayoutTree]) -> int:
        """Compute depth of element in layout tree"""
        depth = 0
        current = elem
        
        while current.parent_id:
            depth += 1
            parent = next((e for e in layout if e.element_id == current.parent_id), None)
            if not parent:
                break
            current = parent
        
        return depth
    
    def _find_state(self, states: List[WebpageState], state_id: str) -> Optional[WebpageState]:
        """Find state by ID"""
        for state in states:
            if f"{state.site_id}_{state.timestamp}" == state_id:
                return state
        return None


class EditSequenceDataset(Dataset):
    """
    PyTorch Dataset for edit sequences
    
    Returns (state_history, edit_history, next_edit, stability) tuples
    Used for training RNN self-model
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sequence_length: int = 5,
        max_sequences: Optional[int] = None
    ):
        """
        Args:
            data_dir: Directory containing collected snapshots
            split: "train", "val", or "test"
            sequence_length: How many edits to use as history
            max_sequences: Maximum number of sequences to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        
        # Load edit sequences
        self.sequences = self._load_edit_sequences(max_sequences)
        
        print(f"Loaded {len(self.sequences)} edit sequences for {split} split")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single training sample
        
        Returns:
            {
                'latent_states': Sequence of latent states (seq_len, 32)
                'edit_history': Sequence of edits (seq_len, 50)
                'next_edit': Target edit to predict (50,)
                'stability_metrics': Target stability (3,)  [spectral_radius, perturbation_return, outcome_delta]
            }
        """
        sequence = self.sequences[idx]
        
        # Convert to tensors
        latent_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in sequence['latent_states']])
        edit_history = torch.stack([torch.tensor(e, dtype=torch.float32) for e in sequence['edit_history']])
        next_edit = torch.tensor(sequence['next_edit'], dtype=torch.float32)
        stability_metrics = torch.tensor(sequence['stability_metrics'], dtype=torch.float32)
        
        return {
            'latent_states': latent_states,
            'edit_history': edit_history,
            'next_edit': next_edit,
            'stability_metrics': stability_metrics,
        }
    
    def _load_edit_sequences(self, max_sequences: Optional[int]) -> List[Dict]:
        """
        Load multi-edit sequences from snapshots
        
        Each sequence is (state_0, edit_0, state_1, edit_1, ..., state_n)
        """
        # Placeholder: Would load from collected data
        # For now, return empty list
        return []


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset_type: str = "state"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders
    
    Args:
        data_dir: Directory with collected data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        dataset_type: "state" for world-model, "sequence" for self-model
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    DatasetClass = WebpageStateDataset if dataset_type == "state" else EditSequenceDataset
    
    train_dataset = DatasetClass(data_dir, split="train")
    val_dataset = DatasetClass(data_dir, split="val")
    test_dataset = DatasetClass(data_dir, split="test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
