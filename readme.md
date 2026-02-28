# Project: Imagination-First Learning and Minimal Self-Model First Architectures

## Overview
This project explores two innovative concepts in machine learning:

1. **Imagination-First Learning**: A framework for generating counterfactual worlds using a family of transition operators \(T_\theta\) with internal coherence constraints.
2. **Minimal Self-Model First Architectures**: A framework where world modeling emerges as a perturbation to self-prediction in a dynamical system \(S_\theta\).

## Project Structure

```
Cog/
├── imagination_first_learning/   # Imagination-First Learning framework
├── minimal_self_model/          # Minimal Self-Model First Architectures
├── experiments/                  # Main experiment scripts (timeseries, world models)
├── data/                        # Datasets (ETT, Traffic, Weather, Timeseries-PILE, etc.)
├── deterministic_data/          # Deterministic dataset generators
├── results/                     # Training results and checkpoints
├── checkpoints/                 # Saved model checkpoints
├── tests/                       # Unit and smoke tests
├── scripts/                     # Utility scripts (analysis, cleanup, etc.)
├── plots/                       # Visualization and analysis results
├── stability_analysis/          # Stability analysis experiments
├── web_design/                  # Web design & aesthetics projects (separated)
└── requirements.txt             # Python dependencies
```

## Recent Updates

### Timeseries-PILE Integration (2026-02-27)
- ✅ Integrated [Timeseries-PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE) dataset (13M+ time series from 5+ databases)
- ✅ Created unified data loaders for forecasting, classification, and anomaly detection datasets
- ✅ Adapted self-model and world-model for real-world multivariate time series
- ✅ Built benchmark comparison framework for self-model vs world-model evaluation
- ✅ Validated on ETT datasets with working training pipelines

**See [TIMESERIES_PILE_INTEGRATION.md](TIMESERIES_PILE_INTEGRATION.md) for details**

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (Current Prototypes)

### Imagination-First Learning (VAE)

Generate synthetic data:

```bash
python3 imagination_first_learning/data/generate_synthetic_data.py
```

Train VAE (config-driven defaults, CLI overrides supported):

```bash
python3 imagination_first_learning/experiments/train_vae.py
```

Notes:
- Default synthetic generator now uses a structured `latent_mixture` process (learnable).
- To reproduce the old unstructured/noise-floor behavior, use `python3 imagination_first_learning/data/generate_synthetic_data.py --process iid`.
- For continuous synthetic data, the trainer defaults to `--recon-loss mse` with KL warmup.

Evaluate VAE:

```bash
python3 imagination_first_learning/experiments/evaluate_vae.py
```

### Minimal Self-Model First Architectures

Generate sequential synthetic data:

```bash
python3 minimal_self_model/data/generate_synthetic_data.py
```

Notes:
- Default process is now `ar1` (learnable dynamics).
- To reproduce the old noise-floor behavior, use `python3 minimal_self_model/data/generate_synthetic_data.py --process iid`.

Train self-model + fit VAE latent transition baseline:

```bash
python3 minimal_self_model/experiments/train_self_model.py
```

Fair VAE baseline for the self-model experiment (train VAE on the same sequential observations):

```bash
python3 minimal_self_model/data/flatten_observations.py
python3 imagination_first_learning/experiments/train_vae.py --data-path minimal_self_model/data/synthetic_observations.npy --output-path minimal_self_model/experiments/artefacts/vae_on_seq_obs.pth --latent-dim 2 --recon-loss mse --kl-warmup-epochs 20
python3 minimal_self_model/experiments/train_self_model.py --vae-model-path minimal_self_model/experiments/artefacts/vae_on_seq_obs.pth --latent-transition-path minimal_self_model/experiments/artefacts/vae_on_seq_obs_transition.npz
```

Evaluate self-model and VAE latent-transition rollouts:

```bash
python3 minimal_self_model/experiments/evaluate_self_model.py
```

To compare directly with the validation metric printed at the end of training, evaluate the same split:

```bash
python3 minimal_self_model/experiments/evaluate_self_model.py --split val
```

If using the fair sequential-observation VAE baseline above, pass the same VAE checkpoint and latent transition file:

```bash
python3 minimal_self_model/experiments/evaluate_self_model.py --split val --vae-model-path minimal_self_model/experiments/artefacts/vae_on_seq_obs.pth --latent-transition-path minimal_self_model/experiments/artefacts/vae_on_seq_obs_transition.npz
```

### Smoke Tests

```bash
python3 -m unittest discover -s tests -p 'test_*_smoke.py'
```

## Phased Plan

### Phase 1: Formalization and Research
- **Imagination-First Learning**:
  - Formalize the concept of "imagination" as a family of transition operators \(T_\theta\).
  - Define internal coherence constraints for \(T_\theta\).
  - Identify methods to project \(T_\theta\) onto the true transition operator \(T_*\).
  - Research existing work on counterfactual reasoning, generative models, and constraint satisfaction.

- **Minimal Self-Model First Architectures**:
  - Formalize the self-model as a dynamical system \(S_\theta\).
  - Define how world modeling emerges as a perturbation to self-prediction.
  - Research existing work on self-supervised learning, neural ODEs, and dynamical systems.

**Deliverables**:
- Mathematical framework for both concepts.
- Literature review of related work.
- Initial hypotheses for both approaches.

### Phase 2: Prototyping
- **Imagination-First Learning**:
  - Build a prototype generative model (e.g., VAE, diffusion model) to generate counterfactual worlds.
  - Implement internal coherence constraints in the generative process.
  - Test the model’s ability to generate internally consistent counterfactuals.

- **Minimal Self-Model First Architectures**:
  - Build a prototype self-model using RNNs or neural ODEs.
  - Train the model to predict its own internal state transitions.
  - Introduce external perturbations and observe the emergence of world modeling.

**Deliverables**:
- Working prototypes for both approaches.
- Initial results on synthetic datasets.

### Phase 3: Evaluation and Refinement
- **Imagination-First Learning**:
  - Evaluate the generative model’s ability to align \(T_\theta\) with \(T_*\) using real-world data.
  - Measure the model’s generalization to unseen scenarios and its robustness to distribution shifts.

- **Minimal Self-Model First Architectures**:
  - Evaluate the self-model’s ability to generalize to new environments and tasks.
  - Compare its performance with traditional world-model-first architectures.

**Deliverables**:
- Performance metrics for both approaches.
- Insights into failure modes and areas for improvement.

### Phase 4: Application and Scaling
- **Imagination-First Learning**:
  - Apply the imagination-first approach to a specific domain (e.g., robotics, game AI).
  - Scale the system to handle more complex counterfactual scenarios.

- **Minimal Self-Model First Architectures**:
  - Apply the self-model-first approach to a specific domain (e.g., autonomous agents, reinforcement learning).
  - Scale the system to handle more complex environments and tasks.

**Deliverables**:
- Domain-specific applications for both approaches.
- Roadmap for scaling and further development.

### Phase 5: Publish and Monetize
- Publish research papers on the theoretical frameworks and experimental results.
- Identify potential applications and markets for commercialization.
- Develop a pitch for funding or partnerships.

**Deliverables**:
- Research papers.
- Funding-ready technical roadmap.
- Initial commercialization strategy.

## Current Progress

### Imagination-First Learning
- Framework: Defined \(T_\theta\) as a family of transition operators with internal coherence constraints.
- Projection Operators: Explored methods to align \(T_\theta\) with \(T_*\).
- Prototype Implemented:
  - VAE model (`imagination_first_learning/models/vae.py`) with training and evaluation scripts.
  - Config-driven experiment scripts with checkpoint metadata, validation split, and device selection.
  - Synthetic data generator with reproducible seeds and CLI parameters.
  - Smoke tests for VAE forward/loss/backward behavior.

### Minimal Self-Model First Architectures
- Framework: Defined the self-model as a dynamical system \(S_\theta\) that predicts its own internal state transitions.
- Perturbation Process: Explored how world modeling emerges as a perturbation to self-prediction.
- Prototype Implemented:
  - RNN self-model with train/eval scripts for one-step and rollout MSE metrics.
  - VAE latent-transition baseline integration for comparison.
  - Config-driven experiment defaults and checkpoint metadata support.
  - Synthetic sequential data generator and smoke tests.

## Next Steps
- Add a reproducible experiment runner (saved configs per run, timestamped output directories).
- Add visualization notebooks/scripts (VAE latent plots, rollout error curves, sample reconstructions).
- Introduce more structured synthetic environments (controlled dynamics, interventions, counterfactual labels).
- Add stronger tests (checkpoint round-trip, script smoke runs with tiny data, metric regression checks).
- Start defining task-specific internal coherence constraints for the imagination-first objective beyond plain reconstruction.


