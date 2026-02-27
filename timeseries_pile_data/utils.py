"""Utility functions for time series data processing."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def normalize_data(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
    """
    Normalize time series data.
    
    Args:
        data: Input data of shape (time, features)
        method: Normalization method ('standard', 'minmax', or 'none')
    
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == 'none':
        return data, {}
    
    if method == 'standard':
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-8
        normalized = (data - mean) / std
        params = {'mean': mean, 'std': std, 'method': 'standard'}
    
    elif method == 'minmax':
        min_val = data.min(axis=0, keepdims=True)
        max_val = data.max(axis=0, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(data: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize time series data.
    
    Args:
        data: Normalized data
        params: Normalization parameters from normalize_data
    
    Returns:
        Denormalized data
    """
    method = params.get('method', 'none')
    
    if method == 'standard':
        return data * params['std'] + params['mean']
    elif method == 'minmax':
        return data * (params['max'] - params['min']) + params['min']
    else:
        return data


def create_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Create time-based features from timestamps.
    
    Args:
        timestamps: DatetimeIndex of timestamps
    
    Returns:
        Array of shape (time, n_features) with cyclical time features
    """
    features = []
    
    # Hour of day (cyclical)
    hour = timestamps.hour
    features.append(np.sin(2 * np.pi * hour / 24))
    features.append(np.cos(2 * np.pi * hour / 24))
    
    # Day of week (cyclical)
    dayofweek = timestamps.dayofweek
    features.append(np.sin(2 * np.pi * dayofweek / 7))
    features.append(np.cos(2 * np.pi * dayofweek / 7))
    
    # Day of month (cyclical)
    day = timestamps.day
    features.append(np.sin(2 * np.pi * day / 31))
    features.append(np.cos(2 * np.pi * day / 31))
    
    # Month of year (cyclical)
    month = timestamps.month
    features.append(np.sin(2 * np.pi * month / 12))
    features.append(np.cos(2 * np.pi * month / 12))
    
    return np.stack(features, axis=-1)


def sliding_window_split(
    data: np.ndarray,
    seq_len: int,
    pred_len: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series.
    
    Args:
        data: Input data of shape (time, features)
        seq_len: Length of input sequence
        pred_len: Length of prediction sequence
        stride: Stride for sliding window
    
    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    n_samples = (len(data) - seq_len - pred_len) // stride + 1
    
    inputs = []
    targets = []
    
    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        pred_end_idx = end_idx + pred_len
        
        inputs.append(data[start_idx:end_idx])
        targets.append(data[end_idx:pred_end_idx])
    
    return np.array(inputs), np.array(targets)


def train_val_test_split(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into train/val/test sets.
    
    Args:
        data: Input data
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
