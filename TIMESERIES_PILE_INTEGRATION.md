# Timeseries-PILE Integration Guide

## Overview

This project now includes integration with the [Timeseries-PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE) dataset, a large collection of over 5+ public time-series databases from diverse domains for foundation model pre-training and evaluation.

## Dataset Location

The Timeseries-PILE dataset is located at: `data/Timeseries-PILE/`

It contains three major categories:
- **forecasting/**: Long-horizon and short-horizon forecasting datasets
  - `autoformer/`: ETT (Electricity Transformer Temperature), Weather, Exchange Rate, ILI, Electricity, Traffic
  - `monash/`: 58 short-horizon forecasting datasets
- **classification/**: UCR/UEA time series classification archive (159 datasets)
- **anomaly_detection/**: TSB-UAD anomaly benchmark (1980 univariate time series)

## Quick Start

### 1. Data Loaders

The project includes a unified data loader module at `timeseries_pile_data/`:

```python
from timeseries_pile_data import create_dataloaders

# Create train/val/test loaders for ETTh1
dataloaders = create_dataloaders(
    dataset_name='ETTh1',
    seq_len=96,        # Input sequence length
    pred_len=24,       # Forecast horizon
    batch_size=32,
    normalize='standard',
)

train_loader = dataloaders['train']
val_loader = dataloaders['val'] 
test_loader = dataloaders['test']
metadata = dataloaders['metadata']
```

**Available Datasets:**
- `etth1`, `etth2` - Hourly Electricity Transformer Temperature
- `ettm1`, `ettm2` - 15-minute Electricity Transformer Temperature
- `weather` - Weather measurements (21 features)
- `exchange_rate` - Exchange rates (8 currencies)
- `national_illness` - ILI (Influenza-like illness)
- `electricity` - Electricity consumption (321 clients)
- `traffic` - Traffic measurements (862 sensors)

### 2. Training Self-Model on Real-World Data

```bash
# Train with default preset
python experiments/train_timeseries_self_model.py --preset ETTh1_short

# Custom configuration
python experiments/train_timeseries_self_model.py \
  --dataset ETTh1 \
  --seq-len 96 \
  --pred-len 24 \
  --hidden-dim 64 \
  --epochs 100 \
  --seed 42
```

### 3. Training World-Model (VAE) on Real-World Data

```bash
# Train with default preset
python experiments/train_timeseries_world_model.py --preset ETTh1_short

# Custom configuration
python experiments/train_timeseries_world_model.py \
  --dataset weather \
  --seq-len 96 \
  --pred-len 192 \
  --latent-dim 16 \
  --epochs 100 \
  --seed 42
```

### 4. Running Benchmark Comparison

Compare self-model vs world-model across multiple seeds:

```bash
python experiments/benchmark_timeseries_comparison.py \
  --dataset ETTh1 \
  --preset ETTh1_short \
  --seeds 42 43 44 45 46 \
  --device cpu
```

This will:
1. Train both self-model and world-model for each seed
2. Aggregate results across seeds
3. Generate a comparison report at `results/timeseries_pile/comparisons/`

## Preset Configurations

Available in `experiments/timeseries_config.py`:

| Preset | Dataset | Seq Len | Pred Len | State Dim | Hidden Dim |
|--------|---------|---------|----------|-----------|------------|
| `ETTh1_short` | ETTh1 | 96 | 24 | 4 | 32 |
| `ETTh1_long` | ETTh1 | 96 | 192 | 8 | 64 |
| `ETTm1_short` | ETTm1 | 96 | 24 | 4 | 32 |
| `weather_short` | weather | 96 | 24 | 8 | 64 |
| `weather_long` | weather | 96 | 192 | 16 | 128 |
| `exchange_short` | exchange_rate | 96 | 24 | 4 | 32 |
| `illness_short` | national_illness | 36 | 24 | 4 | 32 |

## Results Directory Structure

```
results/timeseries_pile/
├── ETTh1_self_model_seed42_h32/
│   ├── config.json
│   ├── history.json
│   ├── best_model.pth
│   └── final_model.pth
├── ETTh1_world_model_seed42_h32/
│   ├── config.json
│   ├── history.json
│   ├── best_model.pth
│   └── final_model.pth
└── comparisons/
    ├── ETTh1_comparison.json
    └── ETTh1_comparison.md
```

## Key Features

### Data Loader Features
- **Automatic normalization**: Standard, MinMax, or None
- **Train/val/test splits**: Configurable ratios (default: 70/10/20)
- **Sliding window sequences**: For overlapping time series samples
- **Time features**: Cyclical encoding of hour, day, month (optional)

### Training Features
- **Early stopping**: Patience-based with validation monitoring
- **Gradient clipping**: Prevents exploding gradients
- **Learning rate scheduling**: Cosine annealing or step decay
- **Checkpoint saving**: Best and final models
- **Comprehensive logging**: Training history in JSON format

### Comparison Features
- **Multi-seed aggregation**: Mean, std, min, max across seeds
- **Automatic report generation**: Markdown and JSON outputs
- **Statistical analysis**: Effect sizes and improvement percentages

## Next Steps

### Immediate
1. Run comprehensive benchmarks on all ETT datasets
2. Compare with published baseline results
3. Analyze failure modes on different dataset characteristics

### Research Questions
1. Does self-model-first superiority from synthetic data generalize to real-world data?
2. How do models perform on different forecast horizons (24, 96, 192, 336)?
3. What role does dataset dimensionality play (7 features vs 321 features)?
4. How do models handle different temporal resolutions (10-minute, hourly, daily, weekly)?

### Potential Improvements
1. **Model Architecture**:
   - Add attention mechanisms
   - Multi-resolution temporal modeling
   - Better handling of multivariate correlations

2. **Training**:
   - Implement curriculum learning (shorter to longer horizons)
   - Add auxiliary losses (trend, seasonality)
   - Better normalization strategies for time series

3. **Evaluation**:
   - Multiple metrics (MSE, MAE, MAPE, MASE)
   - Quantile forecasting for uncertainty
   - Stability analysis over longer rollouts

## References

- **Timeseries-PILE**: Goswami et al. (2024). "MOMENT: A Family of Open Time-series Foundation Models". ICML 2024.
- **Autoformer datasets**: Zhou et al. (2021). "Autoformer: Decomposition Transformers with Auto-Correlation". NeurIPS 2021.
- **Monash Archive**: Godahewa et al. (2021). "Monash Time Series Forecasting Archive".
- **UCR Archive**: Dau et al. (2018). "The UCR Time Series Archive".

## Citation

If you use this integration in your research, please cite both the original datasets and MOMENT:

```bibtex
@inproceedings{goswami2024moment,
  title={MOMENT: A Family of Open Time-series Foundation Models},
  author={Mononito Goswami and Konrad Szafer and Arjun Choudhry and Yifu Cai and Shuo Li and Artur Dubrawski},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
