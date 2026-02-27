"""
Experiment configuration for real-world time series benchmarks using Timeseries-PILE.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TimeseriesExperimentConfig:
    """Configuration for time series experiments on Timeseries-PILE datasets."""
    
    # Dataset configuration
    dataset_name: str = 'ETTh1'
    data_root: str = 'data/Timeseries-PILE/forecasting/autoformer'
    seq_len: int = 96  # Input sequence length
    pred_len: int = 24  # Prediction horizon
    normalize: str = 'standard'  # 'standard', 'minmax', or 'none'
    
    # Model configuration
    model_type: str = 'self_model'  # 'self_model' or 'world_model'
    state_dim: int = 8  # Latent state dimension
    hidden_dim: int = 64  # Hidden dimension for RNN/MLP
    n_layers: int = 2  # Number of layers
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10  # Early stopping patience
    grad_clip: float = 1.0
    
    # VAE configuration (for world_model)
    vae_latent_dim: int = 16  # Increased from 8 for better capacity
    vae_hidden_dim: int = 128  # Increased from 64 for better capacity
    kl_weight: float = 0.1  # Reduced from 1.0 to prevent KL collapse
    kl_warmup_epochs: int = 20  # Increased warmup period
    recon_loss: str = 'mse'  # 'mse' or 'bce'
    
    # Optimization
    optimizer: str = 'adam'  # 'adam' or 'adamw'
    scheduler: str = 'cosine'  # 'cosine', 'step', or 'none'
    warmup_epochs: int = 5
    
    # Evaluation
    eval_horizons: List[int] = field(default_factory=lambda: [24, 48, 96, 192])
    eval_freq: int = 1  # Evaluate every N epochs
    
    # Logging and checkpointing
    output_dir: str = 'results/timeseries_pile'
    experiment_name: Optional[str] = None
    save_checkpoints: bool = True
    log_interval: int = 100
    
    # Reproducibility
    seed: int = 42
    device: str = 'cpu'  # 'cpu', 'cuda', or 'cuda:0'
    num_workers: int = 0
    
    def __post_init__(self):
        """Validate and set derived fields."""
        if self.experiment_name is None:
            self.experiment_name = (
                f"{self.dataset_name}_{self.model_type}_"
                f"seed{self.seed}_h{self.hidden_dim}"
            )


# Preset configurations for common benchmarks
BENCHMARK_CONFIGS = {
    # ETT datasets (Electricity Transformer Temperature)
    'ETTh1_short': TimeseriesExperimentConfig(
        dataset_name='ETTh1',
        seq_len=96,
        pred_len=24,
        state_dim=4,
        hidden_dim=32,
        vae_latent_dim=16,  # Updated for better performance
        vae_hidden_dim=128,
    ),
    'ETTh1_long': TimeseriesExperimentConfig(
        dataset_name='ETTh1',
        seq_len=96,
        pred_len=192,
        state_dim=8,
        hidden_dim=64,
    ),
    'ETTm1_short': TimeseriesExperimentConfig(
        dataset_name='ETTm1',
        seq_len=96,
        pred_len=24,
        state_dim=4,
        hidden_dim=32,
    ),
    
    # Weather dataset
    'weather_short': TimeseriesExperimentConfig(
        dataset_name='weather',
        seq_len=96,
        pred_len=24,
        state_dim=8,
        hidden_dim=64,
    ),
    'weather_long': TimeseriesExperimentConfig(
        dataset_name='weather',
        seq_len=96,
        pred_len=192,
        state_dim=16,
        hidden_dim=128,
    ),
    
    # Exchange rate dataset
    'exchange_short': TimeseriesExperimentConfig(
        dataset_name='exchange_rate',
        seq_len=96,
        pred_len=24,
        state_dim=4,
        hidden_dim=32,
    ),
    
    # ILI (Influenza-like illness) dataset
    'illness_short': TimeseriesExperimentConfig(
        dataset_name='national_illness',
        seq_len=36,
        pred_len=24,
        state_dim=4,
        hidden_dim=32,
        batch_size=16,  # Smaller dataset
    ),
}


def get_config(preset_name: str) -> TimeseriesExperimentConfig:
    """Get a preset configuration by name."""
    if preset_name not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(BENCHMARK_CONFIGS.keys())}"
        )
    return BENCHMARK_CONFIGS[preset_name]
