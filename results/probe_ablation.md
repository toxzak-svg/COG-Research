# Probe Ablation Study

Dataset: ar1
Hidden Dimension: 128
Random Seed: 42

## Configuration Robustness

Testing sensitivity to probe design choices:

### Bootstrap Samples

| Samples | One-Step MSE | Rollout Div | Spectral Radius | Perturbation Return | **Overall Winner** |
|---------|--------------|-------------|-----------------|---------------------|--------------------|
| 100 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 500 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 2000 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 2000 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 2000 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 2000 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 5000 | ✓S | ✓W | ✓S | ✓S | **Self-first** |

### Probe Points

| Points | One-Step MSE | Rollout Div | Spectral Radius | Perturbation Return | **Overall Winner** |
|--------|--------------|-------------|-----------------|---------------------|--------------------|
| 16 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 32 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 128 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 128 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 128 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 128 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 256 | ✓S | ✓W | ✓S | ✓S | **Self-first** |

### Perturbation Scale (ε)

| Epsilon | One-Step MSE | Rollout Div | Spectral Radius | Perturbation Return | **Overall Winner** |
|---------|--------------|-------------|-----------------|---------------------|--------------------|
| 0.001 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 0.01 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 0.01 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 0.01 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 0.01 | ✓S | ✓W | ✓S | ✓S | **Self-first** |
| 0.1 | ✓S | ✓W | ✓S | ✓S | **Self-first** |

## Summary

- Self-model-first wins: 12/12 configurations
- World-model-first wins: 0/12 configurations
- Ties: 0/12 configurations

## Interpretation

**Key:** ✓S = Self-model-first better on this metric, ✓W = World-model-first better, — = No clear winner

✅ **Strong robustness**: Self-model-first wins consistently across all parameter variations.
