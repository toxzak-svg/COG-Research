# COG Research — Next Steps TODO

Generated from analysis of repo innovations and gaps.

## 🔴 CRITICAL PATH (Weeks 1–4)

### Fix Bugs
- [x] Fix stress test `compute_jacobian` call — wrong positional arg (`device` passed as `input_seq`) and wrong tensor shape ✅
- [ ] Fix stress test variable `state_dim` support in self-model (add input projection layer)

### Multi-Dataset Validation
- [ ] Run full 5-seed benchmark on ETTm1
- [ ] Run full 5-seed benchmark on Weather
- [ ] Run full 5-seed benchmark on Traffic
- [ ] Train world-model baselines for Damped oscillator
- [ ] Train world-model baselines for VanderPol
- [ ] Run cross-dataset pattern analysis

## 🟠 HIGH IMPACT (Weeks 5–12)

### Architecture Advances
- [x] Write training script for HierarchicalSelfModel (Level 0 → Level 1 → Level 2) ✅ `experiments/train_hierarchical_self_model.py`
- [ ] Evaluate HierarchicalSelfModel vs flat SelfModel on long-horizon predictions
- [ ] Implement β-TCVAE / Factor-VAE for disentangled latent factors
- [ ] Design and implement compositional generalization test suite
  - Train on (A→B, C→D), test on (A→D, C→B)
  - Zero-shot evaluation on unseen dataset combinations

### Interpretability
- [ ] Visualize learned latent dimensions from VAE
- [ ] Test causal intervention: perturb single latent dim, measure output change
- [ ] Add latent space PCA visualization to training pipeline

## 🟡 PRODUCT PATH (Weeks 8–16)

### App Builder Validation
- [ ] Collect 100 Flask repos from GitHub with CI/CD history
- [ ] Build `(app_state_before, schema_diff) → test_outcome` dataset
- [ ] Train world-model on real app data (target: ≥75% accuracy)
- [ ] Validate spectral radius predicts stable edit sequences
- [ ] Build VS Code extension prototype (stability monitor in status bar)
- [ ] Implement online learning pipeline (experience replay + periodic retraining)

## 🟢 RESEARCH FRONTIER (Months 3–12)

### AGI Capabilities
- [ ] Implement MAML integration for few-shot adaptation
  - Train on 5 datasets, test on 1 new with 100 samples
  - Compare: Does self-model adapt faster than world-model?
- [ ] Implement causal self-modeling (`do(action) → state_{t+1}`)
- [ ] Design intervention datasets for causal discovery
- [ ] Implement RecursiveSelfModel (Theory of Mind)
- [ ] Implement CuriositySelfModel (information-gain driven exploration)
- [ ] Implement SelfGeneratingSelfModel (self-defined objectives)

## 📄 PUBLICATION

- [ ] Write research paper draft (core finding: self-model-first wins 12/12)
- [ ] Complete multi-dataset validation section (need 5/6 datasets)
- [ ] Add information-theoretic analysis (mutual information between latent and observations)
- [ ] Add sample complexity analysis (how few samples needed for convergence?)
- [ ] Submit to NeurIPS / ICML

---

## ✅ COMPLETED (this session)

- [x] Stress test `compute_jacobian` bug fixed (hidden state extraction + correct arg order)
- [x] `experiments/train_hierarchical_self_model.py` created — trains flat, hierarchical L0/L1, and multiscale models with comparison report
- [x] `TODO.md` created with full prioritized roadmap

## ✅ COMPLETED (prior work)

- [x] Core hypothesis validated on AR1 (12/12 wins, Cohen's d > 200, Cliff's Δ = -1.0)
- [x] ETTh1 real-world benchmark (self-model 0.0936 vs world-model 544 → fixed to 0.246)
- [x] ETTm1 quick test (self-model 2.1× better)
- [x] HierarchicalSelfModel implemented (3 levels)
- [x] MultiTimescaleSelfModel implemented (horizons: 1, 5, 10, 24, 48)
- [x] VAE bugs fixed (Sigmoid→Linear, KL clamping, 2200× improvement)
- [x] App builder demo working (run_demo.py)
- [x] Jacobian spectral analysis module (jacobian_spectral.py)
- [x] Perturbation return rate module (perturbation_return.py)
- [x] Timeseries-PILE integration (13M+ time series)
