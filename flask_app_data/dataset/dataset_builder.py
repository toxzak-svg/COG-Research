"""
Flask App Dataset Builder
Creates train/val/test datasets from collected Flask repositories
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import asdict

from .app_schema import AppState, EditRecord, Transition, SchemaState, EndpointState, TestState


class FlaskAppDataset(Dataset):
    """
    PyTorch Dataset for Flask app evolution
    
    Returns (state_t, edit, state_t+1, outcome) tuples
    Used for training VAE world-model
    """
    
    def __init__(
        self,
        data_dir: str = "flask_app_data/collected",
        split: str = "train",
        state_dim: int = 170,
        edit_dim: int = 32,
        max_sequences: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory containing collected app data
            split: "train", "val", or "test"
            state_dim: Dimension of state vectors
            edit_dim: Dimension of edit vectors
            max_sequences: Maximum number of sequences to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.state_dim = state_dim
        self.edit_dim = edit_dim
        
        # Load transitions
        self.transitions = self._load_transitions(max_sequences)
        
        print(f"Loaded {len(self.transitions)} transitions for {split} split")
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single training sample
        
        Returns:
            {
                'state_before': (state_dim,) tensor
                'edit': (edit_dim,) tensor
                'state_after': (state_dim,) tensor
                'outcome': scalar tensor (1.0 = tests passed)
            }
        """
        transition = self.transitions[idx]
        
        return {
            'state_before': torch.tensor(transition['state_before'], dtype=torch.float32),
            'edit': torch.tensor(transition['edit'], dtype=torch.float32),
            'state_after': torch.tensor(transition['state_after'], dtype=torch.float32),
            'outcome': torch.tensor(transition['outcome'], dtype=torch.float32),
            'test_pass_rate': torch.tensor(transition.get('test_pass_rate', 1.0), dtype=torch.float32),
        }
    
    def _load_transitions(self, max_sequences: Optional[int]) -> List[Dict]:
        """
        Load transitions from collected data
        
        Expected file structure:
        data_dir/
            repos/
                repo1/
                    transitions.json
                repo2/
                    transitions.json
        """
        transitions = []
        
        repos_dir = self.data_dir / "repos"
        if not repos_dir.exists():
            print(f"Warning: Repos directory not found: {repos_dir}")
            return self._create_synthetic_data(max_sequences)
        
        # Load from each repo
        for repo_dir in repos_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            
            transition_file = repo_dir / "transitions.json"
            if not transition_file.exists():
                continue
            
            try:
                with open(transition_file, 'r') as f:
                    repo_transitions = json.load(f)
                    transitions.extend(repo_transitions)
            except Exception as e:
                print(f"Error loading {transition_file}: {e}")
                continue
            
            if max_sequences and len(transitions) >= max_sequences:
                break
        
        # Split into train/val/test
        np.random.seed(42)
        np.random.shuffle(transitions)
        
        n = len(transitions)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        if self.split == "train":
            transitions = transitions[:n_train]
        elif self.split == "val":
            transitions = transitions[n_train:n_train+n_val]
        else:  # test
            transitions = transitions[n_train+n_val:]
        
        return transitions[:max_sequences] if max_sequences else transitions
    
    def _create_synthetic_data(self, max_sequences: Optional[int]) -> List[Dict]:
        """
        Create synthetic training data for development/testing
        
        This generates realistic-looking transitions based on common Flask patterns.
        In production, this would be replaced with real collected data.
        """
        print(f"Creating synthetic training data...")
        
        # Common table patterns
        tables = [
            ['id', 'email', 'password_hash', 'created_at'],
            ['id', 'name', 'description', 'created_at', 'updated_at'],
            ['id', 'title', 'content', 'author_id', 'created_at'],
        ]
        
        # Common edit sequences
        edit_templates = [
            # Add user email
            {
                'edit_type': 'add_column',
                'target': 'user',
                'field': 'email',
            },
            # Add auth token
            {
                'edit_type': 'add_column',
                'target': 'user',
                'field': 'auth_token',
            },
            # Add post title
            {
                'edit_type': 'add_column',
                'target': 'post',
                'field': 'title',
            },
            # Add endpoint
            {
                'edit_type': 'add_endpoint',
                'target': '/api/users',
                'field': 'methods',
            },
            # Add test
            {
                'edit_type': 'add_test',
                'target': 'test_user_model',
                'field': 'test_email_validation',
            },
        ]
        
        # Generate transitions
        transitions = []
        
        for _ in range(1000):  # Generate 1000 synthetic transitions
            # Random starting state
            table_idx = np.random.randint(0, len(tables))
            columns = tables[table_idx].copy()
            
            # Create before state
            state_before = self._create_synthetic_state(columns)
            
            # Random edit
            edit_template = edit_templates[np.random.randint(0, len(edit_templates))]
            
            # Apply edit to get after state
            columns_after = columns.copy()
            if edit_template['edit_type'] == 'add_column':
                new_col = edit_template['field']
                if new_col not in columns_after:
                    columns_after.append(new_col)
            
            state_after = self._create_synthetic_state(columns_after)
            
            # Create edit vector
            edit = self._create_synthetic_edit(edit_template)
            
            # Outcome (mostly positive in synthetic data)
            outcome = 1.0 if np.random.random() > 0.2 else 0.0
            
            transitions.append({
                'state_before': state_before,
                'edit': edit,
                'state_after': state_after,
                'outcome': outcome,
                'test_pass_rate': outcome,
            })
        
        # Split
        np.random.seed(42)
        np.random.shuffle(transitions)
        
        n = len(transitions)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        if self.split == "train":
            transitions = transitions[:n_train]
        elif self.split == "val":
            transitions = transitions[n_train:n_train+n_val]
        else:
            transitions = transitions[n_train+n_val:]
        
        print(f"Created {len(transitions)} synthetic transitions")
        
        return transitions[:max_sequences] if max_sequences else transitions
    
    def _create_synthetic_state(self, columns: List[str]) -> List[float]:
        """Create a synthetic state vector"""
        # Start with zeros
        state = [0.0] * self.state_dim
        
        # Set first table as present
        state[0] = 1.0
        
        # Set columns (simple encoding)
        for i, col in enumerate(columns[:10]):
            if i < 20:
                state[20 + i] = 1.0  # Column presence
        
        # Add some randomness
        noise = np.random.randn(self.state_dim) * 0.1
        state = [s + n for s, n in zip(state, noise)]
        
        return state
    
    def _create_synthetic_edit(self, template: Dict) -> List[float]:
        """Create a synthetic edit vector"""
        edit = [0.0] * self.edit_dim
        
        # Simple encoding based on edit type
        type_to_idx = {
            'add_column': 0,
            'add_endpoint': 1,
            'add_test': 2,
        }
        
        idx = type_to_idx.get(template['edit_type'], 0)
        edit[idx] = 1.0
        
        # Add hash of target
        target_hash = hash(template.get('target', '')) % self.edit_dim
        edit[target_hash] = max(edit[target_hash], 0.5)
        
        return edit


class EditSequenceDataset(Dataset):
    """
    PyTorch Dataset for edit sequences
    
    Returns (state_history, edit_history, next_edit, outcome) tuples
    Used for training RNN self-model
    """
    
    def __init__(
        self,
        data_dir: str = "flask_app_data/collected",
        split: str = "train",
        sequence_length: int = 5,
        state_dim: int = 170,
        edit_dim: int = 32,
        max_sequences: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory containing collected app data
            split: "train", "val", or "test"
            sequence_length: Number of edits in history
            state_dim: Dimension of state vectors
            edit_dim: Dimension of edit vectors
            max_sequences: Maximum number of sequences to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.edit_dim = edit_dim
        
        # Load sequences
        self.sequences = self._load_sequences(max_sequences)
        
        print(f"Loaded {len(self.sequences)} edit sequences for {split} split")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single training sample
        
        Returns:
            {
                'state_history': (seq_len, state_dim) tensor
                'edit_history': (seq_len, edit_dim) tensor  
                'next_edit': (edit_dim,) tensor
                'outcome': scalar tensor
            }
        """
        seq = self.sequences[idx]
        
        return {
            'state_history': torch.tensor(seq['state_history'], dtype=torch.float32),
            'edit_history': torch.tensor(seq['edit_history'], dtype=torch.float32),
            'next_edit': torch.tensor(seq['next_edit'], dtype=torch.float32),
            'outcome': torch.tensor(seq['outcome'], dtype=torch.float32),
        }
    
    def _load_sequences(self, max_sequences: Optional[int]) -> List[Dict]:
        """
        Load edit sequences from transitions
        
        Groups consecutive transitions into sequences
        """
        # For now, use synthetic sequences
        # In production, would load from real data
        
        sequences = []
        
        for _ in range(500):  # 500 sequences
            seq_len = self.sequence_length
            
            # Random state history
            state_history = []
            edit_history = []
            
            for _ in range(seq_len):
                state_history.append(np.random.randn(self.state_dim).tolist())
                edit_history.append(np.random.randn(self.edit_dim).tolist())
            
            # Next edit
            next_edit = np.random.randn(self.edit_dim).tolist()
            
            # Outcome
            outcome = 1.0 if np.random.random() > 0.2 else 0.0
            
            sequences.append({
                'state_history': state_history,
                'edit_history': edit_history,
                'next_edit': next_edit,
                'outcome': outcome,
            })
        
        # Split
        np.random.seed(42)
        np.random.shuffle(sequences)
        
        n = len(sequences)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        if self.split == "train":
            sequences = sequences[:n_train]
        elif self.split == "val":
            sequences = sequences[n_train:n_train+n_val]
        else:
            sequences = sequences[n_train+n_val:]
        
        return sequences[:max_sequences] if max_sequences else sequences


def create_flask_dataloaders(
    data_dir: str = "flask_app_data/collected",
    batch_size: int = 32,
    num_workers: int = 0,
    state_dim: int = 170,
    edit_dim: int = 32,
    sequence_length: int = 5,
    dataset_type: str = "transition",  # "transition" or "sequence"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders
    
    Args:
        data_dir: Directory with collected data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        state_dim: Dimension of state vectors
        edit_dim: Dimension of edit vectors
        sequence_length: Length of edit sequences
        dataset_type: "transition" for world-model, "sequence" for self-model
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    DatasetClass = FlaskAppDataset if dataset_type == "transition" else EditSequenceDataset
    
    # Add extra kwargs for sequence dataset
    kwargs = {}
    if dataset_type == "sequence":
        kwargs['sequence_length'] = sequence_length
    
    train_dataset = DatasetClass(
        data_dir=data_dir,
        split="train",
        state_dim=state_dim,
        edit_dim=edit_dim,
        **kwargs
    )
    
    val_dataset = DatasetClass(
        data_dir=data_dir,
        split="val",
        state_dim=state_dim,
        edit_dim=edit_dim,
        **kwargs
    )
    
    test_dataset = DatasetClass(
        data_dir=data_dir,
        split="test",
        state_dim=state_dim,
        edit_dim=edit_dim,
        **kwargs
    )
    
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


def main():
    """Demo: Test dataset loading"""
    print("Testing Flask App Dataset...")
    
    # Test transition dataset
    train_ds = FlaskAppDataset(split="train")
    print(f"Train dataset: {len(train_ds)} samples")
    
    sample = train_ds[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"state_before shape: {sample['state_before'].shape}")
    print(f"edit shape: {sample['edit'].shape}")
    print(f"state_after shape: {sample['state_after'].shape}")
    print(f"outcome: {sample['outcome']}")
    
    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, val_loader, test_loader = create_flask_dataloaders(
        batch_size=8,
        dataset_type="transition"
    )
    
    for batch in train_loader:
        print(f"Batch state_before shape: {batch['state_before'].shape}")
        print(f"Batch edit shape: {batch['edit'].shape}")
        break
    
    print("\n✓ Dataset loading works!")


if __name__ == "__main__":
    main()

