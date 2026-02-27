# Experiment Execution Status

**Updated:** 2026-02-27 12:58 PM

## Active Experiments

### 1. Timeseries-PILE ETTh1 Benchmark
**Status:** 🔄 RUNNING  
**Started:** 12:49 PM  
**Progress:** 1/10 experiments complete (10%)  
**Expected completion:** ~3-4 hours (around 4:00 PM)

**Configuration:**
- Dataset: ETTh1 (hourly electricity transformer temperature)
- Models: Self-Model (RNN) vs World-Model (VAE)
- Seeds: 42, 43, 44, 45, 46 (5 seeds)
- Epochs: 100 per model
- Total experiments: 10 (5 seeds × 2 models)

**Current status:**
- ✅ world_model seed 42: COMPLETE (val_loss: 592.865)
- ▶️ self_model seed 42: RUNNING (39/100 epochs, val_loss: improving)
- ⏸️ Seeds 43-46: QUEUED

**Process:** PID 185092, CPU: 171s, Memory: 335 MB  
**Estimated completion of seed 42:** ~12 more minutes

---

### 2. Damped Oscillator Verification (world-model-first)
**Status:** 🔄 RUNNING  
**Started:** 12:53 PM  
**Progress:** 0/11 experiments complete (0%)  
**Expected completion:** ~2-3 hours (around 3:30 PM)

**Configuration:**
- System: Damped oscillator (deterministic dynamics)
- Paradigm: World-model-first (VAE)
- Seeds: 10-20 (11 seeds)
- Hidden dims: [16, 32, 64, 128] (4 parameter sweeps)
- Epochs: 50 per model
- Total experiments: 44 (11 seeds × 4 hidden_dims)

**▶️ Training in progress: Epoch 20/50
- Hidden dim 16, seed 10: RUNNING
- Remaining seeds: 11-20 in queue

**Process:** PID 148320, CPU: 1560s, Memory: 287 MB  
**Estimated completion:** ~1.5 hours remaining
**Process:** PID 148320, CPU: 349s, Memory: 287 MB

---

## Deliverables

### When ETTh1 Benchmark Completes:
1. ✅ Training histories for 10 experiments
2. ✅ Aggregated comparison report (JSON + Markdown)
3. ✅ Statistical analysis (mean/std/min/max)
4. ✅ Winner determination with effect size
5. 📊 Visualization plots (training curves, comparison bars, seed variance)

### When Damped Verification Completes:
1. ✅ Multi-seed validation of world-model-first paradigm
2. ✅ Parameter sweep results (16, 32, 64, 128 hidden dims)
3. ✅ Stability analysis metrics (spectral radius, return rate)
4. ✅ Comparison with existing self-model-first results

---

## Monitoring

**Check progress:**
```powershell
python experiments/monitor_progress.py --interval 0
```

**Continuous monitoring (updates every 60s):**
```powershell
python experiments/monitor_progress.py --interval 60
```

**Check process status:**
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, @{Name='MemMB';Expression={[math]::Round($_.WorkingSet/1MB)}}
```

---

## Visualization (After Completion)

**Generate plots for ETTh1:**
```powershell
python experiments/visualize_timeseries_results.py --dataset ETTh1
```

**View results:**
- Comparison report: `results/timeseries_pile/comparisons/ETTh1_comparison.md`
- Training curves: `plots/timeseries/ETTh1_training_curves.png`
- Bar chart: `plots/timeseries/ETTh1_comparison.png`
- Seed variance: `plots/timeseries/ETTh1_seed_variance.png`

---

## Next Steps (After Current Experiments)

1. **Analyze ETTh1 results** - Does self-model superiority from synthetic data hold on real-world multivariate time series?

2. **Complete damped verification** - Finish remaining seeds (0-9) for world-model-first paradigm

3. **Expand benchmarks:**
   - ETTm1 (15-minute resolution)
   - Weather (21 features)
   - Exchange Rate (8 currencies)
   - Traffic (862 sensors)

4. **Vanderpol verification** - Run full multi-seed validation for both paradigms

5. **Cross-dataset analysis** - Identify patterns across synthetic and real-world data

---

## Research Questions Being Addressed

1. **Generalization to real data:** Does self-model-first superiority observed in AR1 synthetic data extend to real-world multivariate time series?

2. **Scale effects:** How do the architectures compare when:
   - Number of features varies (7 in ETT, 21 in Weather)
   - Sequence length changes (96 vs 336)
   - Prediction horizon extends (24, 48, 96, 192)

3. **Computational efficiency:** What is the parameter efficiency trade-off?
   - Self-model: ~1.5K parameters
   - World-model: ~191K parameters (127x larger)

4. **Stability:** Do stability properties from deterministic systems transfer to noisy real-world data?
