"""Data loaders for Timeseries-PILE forecasting datasets."""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import (
    normalize_data,
    denormalize_data,
    train_val_test_split,
    sliding_window_split,
)


# Dataset metadata
AUTOFORMER_DATASETS = {
    'ETTh1': {'freq': '1H', 'features': 7, 'target_col': 'OT'},
    'ETTh2': {'freq': '1H', 'features': 7, 'target_col': 'OT'},
    'ETTm1': {'freq': '15T', 'features': 7, 'target_col': 'OT'},
    'ETTm2': {'freq': '15T', 'features': 7, 'target_col': 'OT'},
    'electricity': {'freq': '1H', 'features': 321, 'target_col': 'MT_001'},
    'exchange_rate': {'freq': '1D', 'features': 8, 'target_col': None},
    'national_illness': {'freq': '1W', 'features': 7, 'target_col': 'OT'},
    'traffic': {'freq': '1H', 'features': 862, 'target_col': None},
    'weather': {'freq': '10T', 'features': 21, 'target_col': 'OT'},
}


def load_autoformer_dataset(
    dataset_name: str,
    data_root: str = 'data/Timeseries-PILE/forecasting/autoformer',
    normalize: str = 'standard',
    split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> Dict[str, np.ndarray]:
    """
    Load an Autoformer dataset from Timeseries-PILE.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ETTh1', 'weather')
        data_root: Root directory containing the datasets
        normalize: Normalization method ('standard', 'minmax', or 'none')
        split_ratios: Ratios for (train, val, test) splits
    
    Returns:
        Dictionary containing:
            - 'train': Training data
            - 'val': Validation data
            - 'test': Test data
            - 'norm_params': Normalization parameters
            - 'metadata': Dataset metadata
    """
    # Handle dataset name variations
    dataset_key = dataset_name.lower()
    if dataset_key not in AUTOFORMER_DATASETS:
        # Try with common aliases
        if dataset_key == 'electricity':
            dataset_key = 'electricity'
        elif dataset_key == 'illness' or dataset_key == 'ili':
            dataset_key = 'national_illness'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(AUTOFORMER_DATASETS.keys())}")
    
    metadata = AUTOFORMER_DATASETS[dataset_key]
    
    # Construct file path
    file_map = {
        'ETTh1': 'ETTh1.csv',
        'ETTh2': 'ETTh2.csv',
        'ETTm1': 'ETTm1.csv',
        'ETTm2': 'ETTm2.csv',
        'electricity': 'electricity.csv',
        'exchange_rate': 'exchange_rate.csv',
        'national_illness': 'national_illness.csv',
        'traffic': 'traffic.csv',
        'weather': 'weather.csv',
    }
    
    file_path = Path(data_root) / file_map[dataset_key]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Remove date column if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Convert to numpy
    data = df.values.astype(np.float32)
    
    # Split into train/val/test
    train_data, val_data, test_data = train_val_test_split(
        data, *split_ratios
    )
    
    # Normalize based on training data statistics
    if normalize != 'none':
        train_normalized, norm_params = normalize_data(train_data, method=normalize)
        
        # Apply same normalization to val and test
        if normalize == 'standard':
            val_normalized = (val_data - norm_params['mean']) / norm_params['std']
            test_normalized = (test_data - norm_params['mean']) / norm_params['std']
        elif normalize == 'minmax':
            val_normalized = (val_data - norm_params['min']) / (norm_params['max'] - norm_params['min'] + 1e-8)
            test_normalized = (test_data - norm_params['min']) / (norm_params['max'] - norm_params['min'] + 1e-8)
    else:
        train_normalized = train_data
        val_normalized = val_data
        test_normalized = test_data
        norm_params = {}
    
    return {
        'train': train_normalized,
        'val': val_normalized,
        'test': test_normalized,
        'norm_params': norm_params,
        'metadata': metadata,
    }


class ForecastingDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting with Timeseries-PILE data.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 24,
        stride: int = 1,
        include_target_in_input: bool = True,
    ):
        """
        Args:
            data: Time series data of shape (time, features)
            seq_len: Length of input sequence
            pred_len: Length of prediction horizon
            stride: Stride for sliding window
            include_target_in_input: If True, target features are in input
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.include_target_in_input = include_target_in_input
        
        # Pre-compute valid indices
        self.indices = self._compute_indices()
    
    def _compute_indices(self):
        """Compute valid starting indices for sequences."""
        max_idx = len(self.data) - self.seq_len - self.pred_len
        return list(range(0, max_idx + 1, self.stride))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            Tuple of (input_seq, target_seq) as torch tensors
            - input_seq: shape (seq_len, n_features)
            - target_seq: shape (pred_len, n_features)
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        pred_end_idx = end_idx + self.pred_len
        
        input_seq = self.data[start_idx:end_idx]
        target_seq = self.data[end_idx:pred_end_idx]
        
        return (
            torch.from_numpy(input_seq).float(),
            torch.from_numpy(target_seq).float(),
        )


class SequenceDataset(Dataset):
    """
    Simple sequential dataset for self-model experiments.
    Adapted for multivariate time series.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 50,
    ):
        """
        Args:
            data: Time series data of shape (time, features)
            seq_len: Length of sequences
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        """Create non-overlapping sequences."""
        n_sequences = len(self.data) // self.seq_len
        sequences = []
        
        for i in range(n_sequences):
            start_idx = i * self.seq_len
            end_idx = start_idx + self.seq_len
            sequences.append(self.data[start_idx:end_idx])
        
        return np.array(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            Single sequence of shape (seq_len, n_features)
        """
        return torch.from_numpy(self.sequences[idx]).float()


def create_dataloaders(
    dataset_name: str,
    seq_len: int = 96,
    pred_len: int = 24,
    batch_size: int = 32,
    data_root: str = 'data/Timeseries-PILE/forecasting/autoformer',
    normalize: str = 'standard',
    num_workers: int = 0,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for a forecasting dataset.
    
    Args:
        dataset_name: Name of the dataset
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        batch_size: Batch size
        data_root: Root directory for datasets
        normalize: Normalization method
        num_workers: Number of DataLoader workers
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders and 'metadata'
    """
    # Load data
    data_dict = load_autoformer_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        normalize=normalize,
    )
    
    # Create datasets
    train_dataset = ForecastingDataset(
        data_dict['train'],
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1,
    )
    
    val_dataset = ForecastingDataset(
        data_dict['val'],
        seq_len=seq_len,
        pred_len=pred_len,
        stride=pred_len,  # Non-overlapping for validation
    )
    
    test_dataset = ForecastingDataset(
        data_dict['test'],
        seq_len=seq_len,
        pred_len=pred_len,
        stride=pred_len,  # Non-overlapping for testing
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'metadata': data_dict['metadata'],
        'norm_params': data_dict['norm_params'],
    }
