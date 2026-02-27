# Probe Ablation Study

Dataset: vanderpol
Hidden Dimension: 128
Random Seed: 42

## Configuration Robustness

Testing sensitivity to probe design choices:

### Baseline Configuration

- Bootstrap samples: 2000
- Probe points: 128
- Finite-diff epsilon: 1e-02


### Bootstrap Sample Variations

| Bootstrap | one_step_mse | rollout_div | spectral_radius | return_rate |
|-----------|--------------|-------------|-----------------|-------------|
| 100 | N/A | N/A | N/A | N/A |
| 500 | N/A | N/A | N/A | N/A |
| 2000 | N/A | N/A | N/A | N/A |
| 5000 | N/A | N/A | N/A | N/A |

### Probe Points Variations

| Points | one_step_mse | rollout_div | spectral_radius | return_rate |
|--------|--------------|-------------|-----------------|-------------|
| 16 | N/A | N/A | N/A | N/A |
| 32 | N/A | N/A | N/A | N/A |
| 128 | N/A | N/A | N/A | N/A |
| 256 | N/A | N/A | N/A | N/A |

### Epsilon Variations

| Epsilon | one_step_mse | rollout_div | spectral_radius | return_rate |
|---------|--------------|-------------|-----------------|-------------|
| 1e-03 | N/A | N/A | N/A | N/A |
| 1e-02 | N/A | N/A | N/A | N/A |
| 1e-01 | N/A | N/A | N/A | N/A |

## Interpretation

**Key:** ✓S = Self-model-first wins, ✓W = World-model-first wins, — = No clear winner

If conclusions are consistent across parameter variations, this indicates robustness.
