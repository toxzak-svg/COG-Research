# Verification Experiments: Replication Guide

## Overview

This document provides comprehensive instructions for replicating the verification experiments that test the robustness and generalizability of the self-model-first vs world-model-first comparison.

**Verification Dimensions:**
1. ✅ Multiple datasets (AR(1), damped oscillator, Van der Pol)
2. ✅ Probe design ablations (bootstrap samples, probe points, epsilon)
3. ✅ Mechanistic deep-dive (latent structure, error modes, generalization)
4. ✅ Stress tests (longer sequences, higher dimensions)
5. ✅ External replication (independent hardware/environment)

## Prerequisites

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd Cog

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0  # Optional, for some analyses
```

### Hardware Requirements

- **Minimum**: CPU with 4GB RAM
- **Recommended**: GPU with CUDA support for faster training
- **Disk Space**: ~500MB for datasets and results

## Dataset Generation

### 1. AR(1) Dataset (Already Generated)

AR(1) is the baseline dataset. If you need to regenerate:

```bash
python deterministic_data/deterministic_dataset.py --dataset ar1 --n-samples 1400 --horizon 50
```

### 2. Damped Oscillator

```bash
python deterministic_data/deterministic_dataset.py --dataset damped --n-samples 1400 --horizon 50
```

### 3. Van der Pol Oscillator

```bash
python deterministic_data/deterministic_dataset.py --dataset vanderpol --n-samples 1400 --horizon 50
```

**Expected Output:** Three files per dataset in `data/deterministic/`:
- `{dataset}_train.pt`
- `{dataset}_val.pt`
- `{dataset}_test.pt`

## Verification Experiment 1: Multi-Dataset Replication

### Goal
Verify that self-model-first vs world-model-first conclusions hold across different dynamical systems.

### Commands

```bash
# Run on all datasets with 10 seeds each
python experiments/multi_seed_comparison.py \
  --datasets ar1,damped,vanderpol \
  --n-seeds 10 \
  --hidden-dims 16,32,64,128 \
  --epochs 50 \
  --device auto

# Run probe analysis on each dataset
python experiments/ordering_hypothesis_probe.py --dataset ar1
python experiments/ordering_hypothesis_probe.py --dataset damped
python experiments/ordering_hypothesis_probe.py --dataset vanderpol
```

### Expected Runtime
- **CPU**: ~6-8 hours per dataset (10 seeds × 4 hidden dims × 2 paradigms)
- **GPU**: ~2-3 hours per dataset

### Validation Criteria

**Results Directory Structure:**
```
results/
├── ar1/
│   ├── self_model_first/
│   │   ├── 16/seed_0.json, seed_0.pth, ..., seed_9.pth
│   │   ├── 32/...
│   │   ├── 64/...
│   │   └── 128/...
│   └── world_model_first/
│       └── (same structure)
├── damped/
└── vanderpol/
```

**Probe Results:**
- JSON: `results/ordering_hypothesis_probe_{dataset}.json`
- Markdown: `plots/ordering_hypothesis_probe_{dataset}.md`

**Key Metrics to Check:**
1. All 10 seeds completed successfully per configuration
2. Consistent winner patterns across datasets
3. CI excludes zero for key metrics (one_step_mse, rollout_divergence_50)

## Verification Experiment 2: Probe Design Ablation

### Goal
Test robustness of conclusions to probe hyperparameters.

### Commands

```bash
# Run ablation on AR(1) with hidden_dim=128
python experiments/probe_ablation.py \
  --dataset ar1 \
  --hidden-dim 128 \
  --device auto

# Run on other datasets
python experiments/probe_ablation.py --dataset damped --hidden-dim 128
python experiments/probe_ablation.py --dataset vanderpol --hidden-dim 128
```

### Ablation Grid

**Bootstrap Samples:** 100, 500, 2000, 5000
**Probe Points:** 16, 32, 128, 256
**Finite-Diff Epsilon:** 1e-3, 1e-2, 1e-1

### Expected Runtime
- **Per dataset**: ~10-15 minutes on CPU

### Validation Criteria

**Check:** Winner consistency across parameter variations

Example from output:
```markdown
### Bootstrap Sample Variations
| Bootstrap | one_step_mse | rollout_div | spectral_radius | return_rate |
|-----------|--------------|-------------|-----------------|-------------|
| 100       | ✓S           | ✓W          | ✓S              | ✓S          |
| 500       | ✓S           | ✓W          | ✓S              | ✓S          |
| 2000      | ✓S           | ✓W          | ✓S              | ✓S          |
| 5000      | ✓S           | ✓W          | ✓S              | ✓S          |
```

**Pass Criteria:** ≥90% consistency in winner across variations.

## Verification Experiment 3: Mechanistic Deep-Dive

### Goal
Understand *why* one paradigm wins by analyzing latent structure, error modes, and generalization.

### Commands

```bash
# Deep-dive on specific model (seed 0, hidden_dim 128)
python experiments/mechanistic_deepdive.py \
  --dataset ar1 \
  --hidden-dim 128 \
  --seed 0 \
  --max-horizon 100 \
  --device auto

# Run on multiple seeds for robustness
for seed in 0 1 2 3 4; do
  python experiments/mechanistic_deepdive.py \
    --dataset ar1 \
    --hidden-dim 128 \
    --seed $seed \
    --output-json results/mechanistic_deepdive_seed${seed}.json
done
```

### Expected Runtime
- **Per model**: ~5 minutes on CPU

### Key Analyses

1. **Latent Transition Structure (World-Model)**
   - Eigenvalue spectrum of transition matrix A
   - Fixed point analysis
   - Effective rank of latent space

2. **Jacobian Dynamics (Self-Model)**
   - Spectral radius distribution
   - Local stability at test points

3. **Error Modes**
   - Distribution of prediction errors
   - High-error sample identification (top 10%)

4. **Generalization to Longer Horizons**
   - MSE and divergence at horizons 10, 20, 30, 50, 100

### Validation Criteria

**Self-Model**: Should show low, consistent spectral radius (<1.5 typically)
**World-Model**: May show higher spectral radius but with structured latent space

## Verification Experiment 4: Stress Tests

### Goal
Identify breaking points: test with longer sequences and higher dimensions.

### Commands

```bash
# Stress test on AR(1) with varying sequence lengths
python experiments/stress_test.py \
  --dataset ar1 \
  --sequence-lengths 50,100,200 \
  --state-dims 2,4 \
  --hidden-dims 16,32,64,128 \
  --epochs 50 \
  --device auto

# Quick stress test (fewer configs)
python experiments/stress_test.py \
  --dataset ar1 \
  --sequence-lengths 50,100 \
  --state-dims 2 \
  --hidden-dims 32,64 \
  --epochs 30
```

### Expected Runtime
- **Full stress test**: ~4-6 hours on CPU
- **Quick test**: ~1 hour on CPU

### Validation Criteria

1. **Convergence Rate**: Check `converged` field in results
2. **Training Time Scaling**: Should scale sub-quadratically with sequence length
3. **Failure Modes**: Note which configurations fail (if any)

**Example Output:**
```json
{
  "sequence_length": 200,
  "hidden_dim": 128,
  "paradigm": "self_model",
  "train_time_seconds": 145.2,
  "converged": true,
  "mean_spectral_radius": 1.23
}
```

## Verification Experiment 5: External Replication

### Goal
Independent verification on different hardware/environment.

### Protocol

#### Step 1: Clean Environment

```bash
# Start from scratch
rm -rf results/ plots/ordering_hypothesis_probe_*.md
python3 -m venv .venv_replication
source .venv_replication/bin/activate
pip install -r requirements.txt
```

#### Step 2: Run Minimal Replication

```bash
# Single dataset, 5 seeds, 2 hidden dims (faster)
python experiments/multi_seed_comparison.py \
  --datasets ar1 \
  --n-seeds 5 \
  --hidden-dims 32,64 \
  --epochs 50 \
  --device cpu

# Run probe
python experiments/ordering_hypothesis_probe.py \
  --dataset ar1 \
  --hidden-dims 32,64 \
  --bootstrap-samples 1000
```

#### Step 3: Document Environment

```bash
# Save environment info
python -c "import torch, numpy, sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'NumPy: {numpy.__version__}')"

# Save hardware info
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CPU count: {torch.get_num_threads()}')"
```

Save this output to `replication_environment.txt`.

#### Step 4: Compare Results

**Key Metrics to Compare:**
- Final validation MSE for seed_0, hidden_dim=32, self-model-first
- Spectral radius for same configuration
- Winner determination from probe (self vs world for each metric)

**Tolerance:**
- MSE: Within 10% (due to numerical differences)
- Spectral radius: Within 5%
- Winner: Must match (binary agreement)

### Replication Checklist

- [ ] Environment setup successful
- [ ] All datasets generated
- [ ] Training completed for all seeds
- [ ] Probe analysis ran without errors
- [ ] Winner patterns match original results
- [ ] Results JSON files validate (parseable)
- [ ] Markdown summaries generated

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size or hidden dimensions
```bash
python experiments/multi_seed_comparison.py --batch-size 16 --hidden-dims 16,32
```

### Issue: Training Diverges (NaN losses)

**Cause:** Learning rate too high or numerical instability

**Solution:** Lower learning rate in script or use gradient clipping

### Issue: Probe Reports "No matched seeds"

**Cause:** Training didn't complete for all seeds

**Solution:** Check `results/{dataset}/{paradigm}/{hidden_dim}/` for missing seed files

### Issue: Different Results on Different Hardware

**Expected:** Small numerical differences (within 5-10%)

**Action:** Document hardware differences and compare trends, not exact values

## Result Validation Script

Create `scripts/validate_replication.py`:

```python
import json
from pathlib import Path

def validate_results():
    """Validate that all expected result files exist and are parseable."""
    datasets = ["ar1", "damped", "vanderpol"]
    paradigms = ["self_model_first", "world_model_first"]
    hidden_dims = [16, 32, 64, 128]
    n_seeds = 10
    
    results_dir = Path("results")
    
    missing = []
    for dataset in datasets:
        for paradigm in paradigms:
            for hidden_dim in hidden_dims:
                for seed in range(n_seeds):
                    json_path = results_dir / dataset / paradigm / str(hidden_dim) / f"seed_{seed}.json"
                    if not json_path.exists():
                        missing.append(str(json_path))
    
    if missing:
        print(f"Missing {len(missing)} result files:")
        for m in missing[:10]:
            print(f"  - {m}")
        return False
    else:
        print("✓ All expected result files present")
        return True

if __name__ == "__main__":
    validate_results()
```

Run with:
```bash
python scripts/validate_replication.py
```

## Reporting Results

When reporting replication results, include:

1. **Environment Details**
   - OS, Python version, PyTorch version
   - CPU/GPU specs
   - Random seeds used

2. **Quantitative Comparison**
   - Table of key metrics vs original
   - Correlation plots (if available)

3. **Qualitative Agreement**
   - Do winner patterns match?
   - Are stability trends consistent?

4. **Deviations**
   - Note any configurations that failed
   - Document any unexpected behavior

## Expected Outputs Summary

After completing all verification experiments:

```
results/
├── ar1/, damped/, vanderpol/         # Multi-seed results
├── ordering_hypothesis_probe_*.json  # Probe results per dataset
├── probe_ablation.json               # Ablation study
├── mechanistic_deepdive_seed*.json   # Deep-dive per seed
├── stress_test.json                  # Stress test results
└── aggregated_results.json           # Summary statistics

plots/
├── ordering_hypothesis_probe_ar1.md
├── ordering_hypothesis_probe_damped.md
├── ordering_hypothesis_probe_vanderpol.md
├── probe_ablation.md
├── mechanistic_deepdive.md
└── stress_test.md
```

## Citation

If you replicate these experiments in your work, please cite:

```
@misc{cog-replication-2026,
  title={Verification Experiments for Self-Model-First Learning},
  year={2026},
  note={Replication package}
}
```

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify your environment matches prerequisites
3. Run smoke tests: `python -m unittest discover -s tests`
4. Open an issue with full error logs and environment details

---

**Last Updated:** 2026-02-27
**Version:** 1.0
