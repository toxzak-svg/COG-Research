# Verification Experiments Execution Plan

**Date:** 2026-02-27  
**Status:** In Progress

## Overview

This document tracks the execution of comprehensive verification experiments for the self-model-first vs world-model-first comparison.

## Execution Status

### ✅ Phase 1: Dataset Preparation
- [x] AR(1) dataset (already existed)
- [x] Damped oscillator dataset (already existed)
- [x] Van der Pol oscillator dataset (newly generated)

### 🔄 Phase 2: Multi-Seed Experiments
- [x] AR(1): 10 seeds × 4 hidden dims × 2 paradigms (COMPLETED)
- 🔄 Damped: Currently running (in progress)
- 🔄 Van der Pol: Currently running (in progress)

**Expected Completion:** ~4-6 hours total (CPU)

### 🔄 Phase 3: Probe Analysis
- [x] AR(1) probe analysis (completed previously)
- ⏳ Damped probe analysis (pending multi-seed completion)
- ⏳ Van der Pol probe analysis (pending multi-seed completion)

### 🔄 Phase 4: Ablation Studies
- 🔄 AR(1) probe ablation (currently running)
- ⏳ Damped ablation (queued)
- ⏳ Van der Pol ablation (queued)

**Current Progress:** Bootstrap/probe point/epsilon variations being tested

### 🔄 Phase 5: Mechanistic Analysis
- 🔄 AR(1) deep-dive seed_0 (currently running)
- ⏳ AR(1 deep-dive seeds 1-4 (queued)
- ⏳ Other datasets (queued)

**Analyses:** Latent structure, error modes, generalization to longer horizons

### 🔄 Phase 6: Stress Tests
- 🔄 AR(1) quick stress test (currently running)
  - Sequence lengths: 50, 100
  - Hidden dims: 32, 64
  - State dim: 2
- ⏳ Full stress test (queued)
  - Sequence lengths: 50, 100, 200
  - Hidden dims: 16, 32, 64, 128
  - State dims: 2, 4

### ✅ Phase 7: Documentation
- [x] Replication guide created (VERIFICATION_REPLICATION.md)
- [x] Execution plan documented (this file)

## Scripts Created

### New Analysis Tools
1. **experiments/probe_ablation.py**
   - Tests sensitivity to bootstrap samples (100-5000)
   - Tests sensitivity to probe points (16-256)
   - Tests sensitivity to finite-diff epsilon (1e-3 to 1e-1)

2. **experiments/mechanistic_deepdive.py**
   - Latent transition eigenvalue analysis
   - Fixed point detection
   - Error mode characterization
   - Generalization to longer horizons (up to 200 steps)

3. **experiments/stress_test.py**
   - Scalability testing
   - Longer sequence training (50-200 timesteps)
   - Higher dimensional states (2-4D)
   - Training time and convergence tracking

4. **VERIFICATION_REPLICATION.md**
   - Complete replication protocol
   - Environment setup instructions
   - Validation criteria
   - Troubleshooting guide

## Currently Running Processes

### Terminal Status
- **Terminal 1**: Damped oscillator multi-seed experiments
  - Progress: ~40% complete (processing hidden_dim=32)
  
- **Terminal 2**: Van der Pol multi-seed experiments
  - Progress: ~30% complete (processing hidden_dim=32)
  
- **Terminal 3**: AR(1) probe ablation study
  - Testing 12 different configurations
  
- **Terminal 4**: AR(1) mechanistic deep-dive
  - Analyzing seed_0, hidden_dim=128
  
- **Terminal 5**: AR(1) stress test (quick)
  - Testing 8 configurations (2 seq_len × 2 hidden × 2 paradigms)

## Next Steps

### Immediate (< 1 hour)
1. ✅ Monitor running processes for completion
2. ⏳ Check ablation study results
3. ⏳ Check mechanistic deep-dive results
4. ⏳ Check stress test results

### Short-term (1-6 hours)
1. ⏳ Wait for damped/vanderpol experiments to complete
2. ⏳ Run probe analysis on damped dataset
3. ⏳ Run probe analysis on vanderpol dataset
4. ⏳ Generate comparison plots across all datasets

### Medium-term (6-24 hours)
1. ⏳ Run full stress test with all configurations
2. ⏳ Run mechanistic deep-dive on multiple seeds
3. ⏳ Run ablation studies on damped/vanderpol
4. ⏳ Consolidate all results into final report

## Expected Outputs

### Results Files
```
results/
├── ar1/                              [✅ Complete]
├── damped/                           [🔄 In Progress]
├── vanderpol/                        [🔄 In Progress]
├── ordering_hypothesis_probe_ar1.json      [✅ Exists]
├── ordering_hypothesis_probe_damped.json   [⏳ Pending]
├── ordering_hypothesis_probe_vanderpol.json [⏳ Pending]
├── probe_ablation.json               [🔄 Running]
├── mechanistic_deepdive.json         [🔄 Running]
└── stress_test.json                  [🔄 Running]
```

### Report Files
```
plots/
├── ordering_hypothesis_probe_ar1.md        [✅ Exists]
├── ordering_hypothesis_probe_damped.md     [⏳ Pending]
├── ordering_hypothesis_probe_vanderpol.md  [⏳ Pending]
├── probe_ablation.md                 [🔄 Running]
├── mechanistic_deepdive.md           [🔄 Running]
└── stress_test.md                    [🔄 Running]
```

## Validation Criteria

### Multi-Seed Experiments
- ✅ All 10 seeds complete per configuration
- ✅ JSON files contain all required metrics
- ✅ .pth checkpoint files saved

### Probe Analysis
- Winner consistency across datasets
- CI excludes zero for key differentiating metrics
- Bootstrap estimates stable

### Ablation Studies
- ≥90% winner agreement across parameter variations
- Mechanism probes show consistent patterns

### Mechanistic Analysis
- Latent structure characterized
- Error distributions documented
- Generalization curves computed

### Stress Tests
- Convergence tracked for all configs
- Training time scaling documented
- Failure points identified (if any)

## Commands Reference

### Check Running Processes
```bash
# PowerShell
Get-Process python | Select-Object Id,ProcessName,StartTime
```

### Check Results
```bash
# List completed seed files
ls results/ar1/self_model_first/128/
ls results/damped/self_model_first/128/
ls results/vanderpol/self_model_first/128/
```

### Run Remaining Analyses (after multi-seed complete)
```bash
# Probe analysis
python experiments/ordering_hypothesis_probe.py --dataset damped
python experiments/ordering_hypothesis_probe.py --dataset vanderpol

# Full stress test
python experiments/stress_test.py \
  --dataset ar1 \
  --sequence-lengths 50,100,200 \
  --state-dims 2,4 \
  --hidden-dims 16,32,64,128 \
  --epochs 50
```

## Resource Usage

### Estimated Total Compute Time
- Multi-seed experiments: ~18 hours CPU (3 datasets × 6 hours)
- Probe analyses: ~30 minutes CPU (3 datasets × 10 min)
- Ablation studies: ~30 minutes CPU (12 configs × 2.5 min)
- Mechanistic deep-dive: ~25 minutes CPU (5 seeds × 5 min)
- Stress tests: ~4 hours CPU (full test)

**Total:** ~23 hours CPU sequential, or ~8 hours with parallelization

### Disk Space
- Raw checkpoints: ~200MB per dataset
- Results JSON: ~10MB total
- **Total:** ~700MB

## Notes

- All experiments use fixed random seeds for reproducibility
- CPU training is intentionally used for replication consistency
- Bootstrap samples = 500-2000 provides stable estimates
- Longer sequence stress tests may reveal instability modes

## Contact

For issues or questions about this execution plan, see VERIFICATION_REPLICATION.md troubleshooting section.

---

**Last Updated:** 2026-02-27 (auto-generated during execution)
