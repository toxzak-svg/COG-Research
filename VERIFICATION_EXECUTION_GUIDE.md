# Verification Experiments - Execution Guide

This guide explains how to run and extend the verification experiments.

## Quick Start

### Run Full Verification Suite (AR1)

```bash
# 1. Ordering Hypothesis Probe
python experiments/ordering_hypothesis_probe.py --dataset ar1

# 2. Probe Ablation Study
python experiments/probe_ablation.py --dataset ar1

# 3. Mechanistic Deep-Dive
python experiments/mechanistic_deepdive.py --dataset ar1

# 4. Multi-Seed Comparison (long-running)
python experiments/multi_seed_comparison.py --dataset ar1 --seeds 10

# 5. Stress Test
python experiments/stress_test.py --dataset ar1

# 6. Generate Visualizations
python experiments/visualization.py --dataset ar1
```

## Extending to New Datasets

### Step 1: Train World-Model Baselines

The verification experiments require both self-model and world-model checkpoints for comparison.

```bash
# Train world-model baselines for damped and vanderpol
python experiments/train_world_model_baselines.py \
    --datasets damped vanderpol \
    --hidden-dims 16 32 64 128 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --epochs 50
```

This will create checkpoints in:
```
results/
  damped/
    world_model_first/
      16/seed_0.pth, seed_1.pth, ...
      32/seed_0.pth, ...
      ...
  vanderpol/
    world_model_first/
      ...
```

### Step 2: Run Verification Experiments

```bash
# Run ordering hypothesis probe
python experiments/ordering_hypothesis_probe.py --dataset damped
python experiments/ordering_hypothesis_probe.py --dataset vanderpol

# Run probe ablation
python experiments/probe_ablation.py --dataset damped
python experiments/probe_ablation.py --dataset vanderpol

# Run mechanistic deep-dive
python experiments/mechanistic_deepdive.py --dataset damped
python experiments/mechanistic_deepdive.py --dataset vanderpol

# Generate visualizations
python experiments/visualization.py --dataset damped
python experiments/visualization.py --dataset vanderpol
```

## Experiment Details

### 1. Ordering Hypothesis Probe

**Purpose**: Test counterfactual generalization with matched random seeds

**Key Parameters**:
- `--dataset`: Dataset to use (ar1, damped, vanderpol)
- `--seed`: Random seed for probe (default: 42)
- `--bootstrap-samples`: Bootstrap samples for CI (default: 2000)
- `--probe-points`: Number of points to probe (default: 128)

**Outputs**:
- `results/ordering_hypothesis_probe.json` - Full results with statistics
- `plots/ordering_hypothesis_probe.md` - Human-readable summary

**Metrics Compared**:
- One-step MSE (lower is better)
- Rollout divergence (lower is better)
- Spectral radius (stability measure)
- Perturbation return rate (recovery speed)

### 2. Probe Ablation Study

**Purpose**: Test robustness to probe design choices

**Key Parameters**:
- `--dataset`: Dataset to use
- `--hidden-dim`: Hidden dimension to test (default: 128)

**Ablation Dimensions**:
- Bootstrap samples: {100, 500, 2000, 5000}
- Probe points: {16, 32, 128, 256}
- Finite-diff epsilon: {1e-3, 1e-2, 1e-1}

**Outputs**:
- `results/probe_ablation.json` - Full ablation results
- `plots/probe_ablation.md` - Summary table

### 3. Mechanistic Deep-Dive

**Purpose**: Understand latent dynamics and error modes

**Key Parameters**:
- `--dataset`: Dataset to use
- `--hidden-dim`: Hidden dimension (default: 128)
- `--seed`: Model seed (default: 0)

**Analyses**:
- **Self-Model**:
  - Jacobian spectral analysis (stability)
  - Error distribution (where model fails)
  - Generalization to longer horizons
- **World-Model**:
  - Latent transition eigenvalues
  - Fixed point analysis
  - Latent space geometry (PCA)
  - Reconstruction error modes

**Outputs**:
- `results/mechanistic_deepdive.json` - Detailed analysis
- `plots/mechanistic_deepdive.md` - Summary report

### 4. Multi-Seed Comparison

**Purpose**: Verify robustness across random initializations

**Key Parameters**:
- `--datasets`: Datasets to test
- `--hidden-dims`: Hidden dimensions to test
- `--seeds`: Number of seeds (default: 10)
- `--epochs`: Training epochs (default: 50)

**Note**: This is a long-running experiment (hours to days depending on configuration).

### 5. Stress Test

**Purpose**: Test with longer sequences and varied dimensions

**Key Parameters**:
- `--dataset`: Dataset to use
- `--seq-lengths`: Sequence lengths to test (default: [50, 100, 200])
- `--state-dims`: State dimensions to test (default: [2, 4])
- `--hidden-dims`: Hidden dimensions (default: [16, 32, 64, 128])

**Known Issue**: Currently fails with variable state dimensions. Requires architecture fix.

### 6. Visualization Suite

**Purpose**: Generate comprehensive plots

**Key Parameters**:
- `--dataset`: Dataset to visualize
- `--plot-types`: Types of plots (default: all)

**Generated Plots**:
- One-step MSE comparison
- Rollout divergence comparison
- Spectral radius comparison
- Perturbation return rate
- Parameter efficiency plots

## Results Interpretation

### Statistical Significance

All experiments use bootstrap confidence intervals (default: 2000 samples) to assess statistical significance.

**Interpreting CI**:
- If CI excludes zero → statistically significant difference
- Effect size (Cohen's d > 0.8) → large effect
- Cliff's Delta = -1.0 or 1.0 → perfect separation

### Winner Determination

Experiments automatically determine "winner" based on:
- **Lower-is-better**: one_step_mse, rollout_divergence, spectral_radius
- **Higher-is-better**: perturbation_return_rate

### Robustness Indicators

A finding is "robust" if:
✓ CI excludes zero across all configurations
✓ Effect size remains large (|d| > 0.8)
✓ Winner is consistent across ablations

## Troubleshooting

### "Dataset not found in results"

**Cause**: No trained models exist for the dataset/paradigm combination.

**Solution**: 
1. Check if checkpoints exist: `results/{dataset}/{paradigm}/{hidden_dim}/seed_*.pth`
2. Train missing models using `train_world_model_baselines.py`
3. Or run the original training scripts to create self-model baselines

### "Shape mismatch" errors in stress test

**Cause**: Self-model architecture expects fixed input dimensions.

**Solution**: Modify `minimal_self_model/models/self_model.py` to accept variable `obs_dim` parameter.

### Experiments run but produce empty results

**Cause**: No models to compare (need both self-model and world-model checkpoints).

**Solution**: Ensure both paradigms have trained checkpoints before running comparison experiments.

## Directory Structure

```
results/
  {dataset}/
    self_model_first/{hidden_dim}/seed_{N}.pth
    world_model_first/{hidden_dim}/seed_{N}.pth
  ordering_hypothesis_probe.json
  probe_ablation.json
  mechanistic_deepdive.json
  aggregated_results.json

plots/
  {dataset}_one_step_mse.png
  {dataset}_rollout_divergence.png
  {dataset}_spectral_radius.png
  {dataset}_perturbation_return.png
  {dataset}_efficiency_*.png
  ordering_hypothesis_probe.md
  probe_ablation.md
  mechanistic_deepdive.md
  {dataset}_summary.md
```

## Citation

If you use these verification experiments, please cite:

```bibtex
@misc{cog2026,
  title={Verification Experiments for Self-Model-First Learning},
  author={COG Research},
  year={2026},
  note={Rigorous statistical validation of counterfactual generalization hypothesis}
}
```

## Contact

For questions or issues, please open an issue in the GitHub repository.
