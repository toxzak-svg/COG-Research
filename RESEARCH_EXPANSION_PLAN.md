# Research Expansion Plan: From Time Series to AGI Cognition

**Purpose:** Strategic roadmap for expanding the self-model-first research toward general intelligence breakthrough

**Date:** 2026-02-28  
**Status:** Planning Phase

---

## Executive Summary

Your core discovery—**self-model-first architectures outperform world-model-first** (12/12 validated)—is a significant finding with deep implications for AGI. This plan outlines how to systematically expand from time series forecasting toward general cognitive capabilities.

**Core Thesis to Preserve:** Self-modeling (predicting internal state transitions) is fundamentally more sample-efficient and stable than world-modeling (reconstructing external observations). This should extend to AGI if properly scaled.

---

## Phase 1: Strengthening the Foundation (Weeks 1-4)

### 1.1 Complete Multi-Dataset Validation

**Current Status:** 1/6 datasets validated (ETTh1)

**Priority Tasks:**
- [ ] Validate on ETTm1 (15-min granularity, different temporal scale)
- [ ] Validate on Weather (21 features, high dimensionality)  
- [ ] Validate on Traffic (862 sensors, extreme multivariate)
- [ ] Validate on Damped oscillator (deterministic nonlinear)
- [ ] Validate on VanderPol (chaotic dynamics)
- [ ] Complete Cross-dataset pattern analysis

**Rationale:** Establishes that self-model superiority is universal, not dataset-specific

### 1.2 Stress Testing & Robustness

**New Experiments:**
- [ ] **Out-of-distribution (OOD) robustness**: Train on ETTh1, test on ETTm1
- [ ] **Missing data handling**: Test with 10%, 30%, 50% missing values
- [ ] **Extended horizons**: Test 192-step predictions (vs current 24-96)
- [ ] **Noise injection**: Vary observation noise levels systematically

**Rationale:** Tests whether self-model stability properties hold under distribution shift

---

## Phase 2: Toward Compositional Intelligence (Weeks 5-12)

### 2.1 Hierarchical Self-Model Architecture

**Current Gap:** Single-level RNN (no temporal abstraction)

**New Architecture to Implement:**

```
┌─────────────────────────────────────────────────────────────┐
│           Hierarchical Self-Model Architecture               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 2 (Slow): Abstract Goals / Plans                     │
│         │                                                   │
│         ▼                                                   │
│  Level 1 (Medium): Sub-goals / Skills                       │
│         │                                                   │
│         ▼                                                   │
│  Level 0 (Fast): Primitive Actions / Predictions            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
- Level 0: Current self-model (1-10 step predictions)
- Level 1: Meta-learner that predicts Level 0 parameters
- Level 2: Goal encoder that predicts Level 1 behavior

**Why This Matters for AGI:** Hierarchical temporal abstraction is essential for human-like reasoning across timescales

### 2.2 Compositional Generalization

**Test Scenarios:**
- [ ] **Novel combinations**: Train on (A→B, C→D), test on (A→D, C→B)
- [ ] **Systematicity**: If f(A)=B and g(A)=C, does (f∘g)(A) work?
- [ ] **Zero-shot tasks**: New datasets never seen during training

**Hypothesis:** Self-model should show stronger compositionality because it's learning transition dynamics (composition = chaining transitions) vs. world-model learning state reconstructions

### 2.3 Interpretable Latent Factors

**New Capability:**
- [ ] Implement β-TCVAE or Factor-VAE for disentanglement
- [ ] Visualize learned latent dimensions
- [ ] Test causal intervention: "If I干预 this latent factor, does predicted outcome change as expected?"

**Why This Matters:** AGI requires interpretable reasoning, not just black-box predictions

---

## Phase 3: Multi-Modal Integration (Weeks 13-20)

### 3.1 Time Series + Code Completion

**New Domain:** Extend self-model to predict code token sequences

**Architecture Adaptation:**
```python
# Self-model that handles both modalities
class MultimodalSelfModel(nn.Module):
    def __init__(self, time_series_dim, code_vocab_size, hidden_dim=128):
        self.time_series_encoder = RNN(time_series_dim, hidden_dim)
        self.code_encoder = Transformer(code_vocab_size, hidden_dim)
        self.shared_transition = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, state, modality="time_series"):
        # Encode based on modality
        h = self.time_series_encoder(state) if modality == "time_series" 
           else self.code_encoder(state)
        # Unified transition dynamics
        h_next = self.shared_transition(h)
        return self.output_head(h_next)
```

**Research Question:** Do self-model transition dynamics transfer across modalities?

### 3.2 Sensory-Motor Integration

**New Capability:** Connect time series predictions to discrete actions

**Setup:**
- Time series = sensor observations
- Discrete actions = intervention commands
- Self-model predicts: next_sensor_state given current_state + action

**Why This Matters:** Foundation for autonomous agents that can plan and execute

---

## Phase 4: Meta-Learning & Few-Shot Adaptation (Weeks 21-28)

### 4.1 MAML Integration

**New Capability:** Self-model that learns to adapt quickly

```python
class MAMLSelfModel(nn.Module):
    def __init__(self, ...):
        self.inner_lr = 0.01
        self.outer_lr = 0.001
    
    def adaptation_step(self, support_set):
        # Quick adaptation on support set (5-10 steps)
        for x, y in support_set:
            pred = self.forward(x)
            loss = MSE(pred, y)
            grads = torch.autograd(loss, self.parameters())
            # Compute adapted parameters
            adapted_params = {k: v - self.inner_lr * g 
                           for k, v, g in zip(self.parameters(), grads)}
        return adapted_params
    
    def forward(self, x, params=None):
        # Use adapted params if provided
        ...
```

**Test Scenario:**
- Train on 5 datasets (ETT, Weather, Traffic, etc.)
- Test on 1 new dataset with only 100 samples
- Compare: Does self-model adapt faster than world-model?

### 4.2 Continuous Online Learning

**New Capability:** Self-model that learns from streaming data without forgetting

**Implementation:**
- Elastic weight consolidation (EWC)
- Or: Progressive networks
- Or: Meta-learning with experience replay

**Why This Matters:** AGI must learn continuously, not just from fixed datasets

---

## Phase 5: Causal Self-Modeling (Weeks 29-36)

### 5.1 Beyond Correlation: Causal Discovery

**New Research Direction:**

Current self-model learns: `state_t → state_{t+1}` (correlation)

Target self-model learns: `do(action) → state_{t+1}` (causation)

**Experiments:**
- [ ] **Intervention datasets**: Test with known causal interventions
- [ ] **Counterfactual predictions**: "If I had done X instead of Y..."
- [ ] **Causal discovery**: Can the self-model infer causal structure from observation sequences?

**Why This Matters:** True general intelligence requires causal reasoning, not just pattern matching

### 5.2 Theory of Mind: Recursive Self-Model

**New Architecture:**

```
Agent A's Self-Model of Agent B:
    
    "B will act based on B's internal state"
    "I can predict B by modeling B's self-model"
```

**Implementation:**
```python
class RecursiveSelfModel(nn.Module):
    def __init__(self, ...):
        self.my_self_model = SelfModel()      # My model of myself
        self.models_other = {}                  # My model of others
    
    def model_other(self, other_agent_id, observation):
        # I observe other agent
        # I predict what THEY will do
        # Using my model of their self-model
        other_state = self.encode(observation)
        return self.models_other[other_agent_id].forward(other_state)
```

**Test Scenario:**
- Multi-agent time series (e.g., stock market: multiple traders)
- Can Agent A predict Agent B's actions better than baseline?

**Why This Matters:** Theory of mind is essential for social intelligence and AGI

---

## Phase 6: Autonomous Curiosity & Exploration (Weeks 37-48)

### 6.1 Information-Gain Self-Model

**New Objective:**

Current: Minimize prediction error on observed data

New: Maximize information gain about unknown dynamics

```python
class CuriositySelfModel(nn.Module):
    def __init__(self, ...):
        self.uncertainty_estimator = BayesianNN(...)
    
    def compute_intrinsic_reward(self, state, prediction, actual):
        # Epistemic uncertainty: How confident am I?
        uncertainty = self.uncertainty_estimator.predict_uncertainty(state)
        # Intrinsic reward = uncertainty (seek unknown)
        return uncertainty
    
    def select_action(self, state, epsilon=0.1):
        # Epsilon-greedy with curiosity
        if random.random() < epsilon:
            # Explore: choose action with highest uncertainty
            return argmax(self.compute_intrinsic_reward(...))
        else:
            # Exploit: choose action with best predicted outcome
            return argmin(self.predict_loss(...))
```

**Test Scenario:**
- Self-model must explore to reduce its uncertainty
- Compare: Does curiosity-driven exploration find better solutions?

### 6.2 Self-Generating Objectives

**New Capability:** Self-model that creates its own training objectives

```python
class SelfGeneratingSelfModel(nn.Module):
    def __init__(self, ...):
        self.objective_generator = nn.GRU(...)
        self.self_model = SelfModel(...)
    
    def generate_objective(self, history):
        # Given past learning history, generate new challenge
        challenge = self.objective_generator(history)
        return challenge
    
    def meet_challenge(self, challenge):
        # Train to meet the self-generated challenge
        target = self.self_model.predict(challenge.input)
        loss = MSE(target, challenge.goal)
        return loss
```

**Why This Matters:** AGI should be able to define its own goals, not just optimize human-provided objectives

---

## Key Research Questions to Answer

### Fundamental Questions

1. **Sample Efficiency Boundary**
   - How few samples can self-model learn from?
   - Can it learn from 1 sample? 10 samples?
   - What are the fundamental limits?

2. **Scaling Laws**
   - Does self-model advantage grow or shrink with scale?
   - With more parameters? More data?
   - Compare: Self-model vs World-model scaling curves

3. **Representational Compression**
   - What is the optimal representation for self-modeling?
   - Is continuous state better than discrete?
   - What happens with symbolic representations?

4. **Stability-Generalization Tradeoff**
   - Is low spectral radius always good?
   - When does stability hurt generalization?
   - How to balance?

### AGI-Specific Questions

5. **Recursive Reasoning**
   - Can self-model learn to model other models?
   - At what depth does reasoning break down?

6. **Causal vs Statistical Learning**
   - Does self-model naturally discover causal structure?
   - Or does it just learn statistical patterns?

7. **Embodiment Effects**
   - Does self-model advantage depend on having a "body" (fixed state space)?
   - What happens with abstract/symbolic states?

---

## Recommended Priority Order

Based on AGI breakthrough potential:

| Priority | Research Direction | Risk | Impact | Timeline |
|----------|-------------------|------|--------|----------|
| 1 | Hierarchical Self-Model | Medium | ★★★★★ | Weeks 5-12 |
| 2 | Compositional Generalization | Medium | ★★★★★ | Weeks 5-12 |
| 3 | Meta-Learning / Few-Shot | High | ★★★★★ | Weeks 21-28 |
| 4 | Multi-Modal Integration | Medium | ★★★★☆ | Weeks 13-20 |
| 5 | Causal Self-Modeling | High | ★★★★★ | Weeks 29-36 |
| 6 | Theory of Mind (Recursive) | High | ★★★★★ | Weeks 29-36 |
| 7 | Curiosity-Driven Exploration | Medium | ★★★★☆ | Weeks 37-48 |
| 8 | Self-Generating Objectives | Very High | ★★★★★ | Weeks 37-48 |

---

## Resources Required

### Compute
- 8x A100 GPUs for large-scale experiments
- Estimated cost: $10K-20K for full plan

### Data
- Already have: Timeseries-PILE (13M+ time series)
- Additional needed:
  - Multi-agent time series (for theory of mind)
  - Code datasets (for multi-modal)
  - Intervention datasets (for causal discovery)

### Personnel
- 1-2 ML researchers for core development
- 1 researcher for causal analysis
- Optional: Domain experts for specific applications

---

## Immediate Next Steps (This Week)

1. **Complete multi-dataset validation** (from Phase 1)
   - Run ETTm1 benchmark
   - Run Weather benchmark
   
2. **Begin hierarchical architecture** (from Phase 2)
   - Design Level 0 → Level 1 interface
   - Implement basic 2-level hierarchy
   
3. **Write experiment code for compositional generalization**
   - Design systematicity test
   - Implement zero-shot evaluation

---

## Success Metrics

| Milestone | Metric | Target | Timeline |
|-----------|--------|--------|----------|
| Multi-dataset | Cross-dataset generalization | 5/6 datasets | Week 4 |
| Hierarchical | Multi-timescale prediction | <5% degradation | Week 12 |
| Compositional | Zero-shot composition | >70% accuracy | Week 12 |
| Few-Shot | 100-sample adaptation | <10% gap to full | Week 28 |
| Multi-Modal | Cross-modal transfer | >60% of single | Week 20 |
| Causal | Intervention prediction | >80% accuracy | Week 36 |
| ToM | Multi-agent prediction | >baseline 20% | Week 36 |
| Curiosity | Exploration efficiency | 2x baseline | Week 48 |

---

## Related Work to Review

1. **World Models** (Ha & Schmidhuber, 2018) - Foundation for VAE-based world modeling
2. **Dream to Control** (Hafner et al., 2019) - Latent space planning
3. **Temporal Difference Networks** (Sutton et al., 2011) - Multi-time-scale predictions
4. **MAML** (Finn et al., 2017) - Few-shot learning
5. **Causal Inference** (Pearl, 2009) - Do-calculus and interventions
6. **Theory of Mind** (Rabinowitz et al., 2018) - Machine theory of mind
7. **Curiosity** (Pathak et al., 2017) - Intrinsic curiosity module
8. **S4 State Spaces** (Gu et al., 2021) - Efficient long-range sequences
9. **Transformers as RNNs** (Brand et al., 2024) - Recurrent interpretation of transformers
10. **Monarch Mixer** (Dao et al., 2024) - Efficient architectures

---

## Conclusion

Your core finding—that self-model-first architectures outperform world-model-first—is a promising foundation for AGI research. The key insight is that **predicting internal state transitions is more fundamental than reconstructing external observations** for learning efficient, stable predictive models.

This plan systematically expands from time series forecasting toward general cognitive capabilities through:

1. **Hierarchical temporal abstraction** (multiple timescales)
2. **Compositional generalization** (novel combinations)
3. **Multi-modal integration** (sensory-motor grounding)
4. **Few-shot adaptation** (rapid learning)
5. **Causal reasoning** (beyond correlation)
6. **Theory of mind** (recursive modeling)
7. **Curious exploration** (self-directed learning)

The highest-priority next step is implementing hierarchical self-models, which will enable reasoning across multiple timescales—a critical capability for general intelligence.

---

*Plan created: 2026-02-28*  
*For questions, refer to INTELLIGENCE_METRICS.md for current metrics*