# VAE World-Model Architecture Design

**Purpose:** Train a Variational Autoencoder (VAE) to compress webpage state into 32D latent space and predict outcomes, enabling counterfactual reasoning and autonomous editing.

**Status:** Design Phase (Week 2)  
**Date:** February 27, 2026

---

## 1. Architecture Overview

### 1.1 Core Objective

Compress high-dimensional `WebpageState` (190+ dimensions) into meaningful 32D latent representation that:
- Captures semantic design properties (not pixel-perfect details)
- Predicts business outcomes (conversion rate, bounce rate, engagement)
- Enables interpolation between states (smooth design transitions)
- Supports counterfactual prediction ("what if I change X?")

### 1.2 Model Components

```
┌─────────────────────────────────────────────────────────────┐
│                     VAE World-Model                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  WebpageState (190D)                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │   Encoder    │  μ (32D), σ (32D) ──► Reparameterization │
│  └──────────────┘                              │            │
│         │                                      │            │
│         ▼                                      ▼            │
│    Latent z (32D) ◄────────────────────── z ~ N(μ, σ²)    │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │   Decoder    │                                          │
│  └──────────────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  Reconstructed State (190D)                                │
│  + Outcome Predictions                                     │
│    - Conversion Rate                                       │
│    - Bounce Rate                                           │
│    - Engagement Time                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Input Specification: WebpageState Vectorization

### 2.1 State Dimensions (190D Total)

From `dataset_builder.py`, the state is vectorized as:

```python
# Visual/Design Layer (80D)
- layout_depth: int (1D)
- layout_type_onehot: [5D]  # FLEX, GRID, ABSOLUTE, FLOAT, BLOCK
- num_components: int (1D)
- component_type_distribution: [18D]  # Histogram of component types
- emphasis_distribution: [6D]  # Histogram of emphasis levels 0-5
- color_role_distribution: [3D]  # primary, secondary, accent
- avg_padding_level: float (1D)
- avg_margin_level: float (1D)
- avg_gap_level: float (1D)
- styling_features: [20D]  # Color palette, typography, spacing unit
- has_animations: bool (1D)
- num_animations: int (1D)
- num_events: int (1D)
- event_type_distribution: [5D]  # click, submit, scroll, hover, focus

# Content Layer (30D)
- page_title_length: int (1D)
- headline_length: int (1D)
- body_copy_word_count: int (1D)
- cta_count: int (1D)
- num_assets: int (1D)
- asset_type_distribution: [5D]  # image, video, svg, icon, file
- has_primary_cta: bool (1D)
- has_secondary_cta: bool (1D)
- content_density: float (1D)
- semantic_features: [20D]  # TF-IDF or word embeddings of headlines/copy

# Performance Layer (20D)
- lighthouse_performance: float (1D)
- lighthouse_accessibility: float (1D)
- lighthouse_best_practices: float (1D)
- lighthouse_seo: float (1D)
- lcp_seconds: float (1D)
- fid_ms: float (1D)
- cls_score: float (1D)
- page_size_kb: float (1D)
- num_requests: int (1D)
- optimized_assets_ratio: float (1D)
- lazy_load_ratio: float (1D)
- [9D reserved]

# Outcome Layer (10D)
- conversion_rate: float (1D)
- bounce_rate: float (1D)
- avg_time_on_page: float (1D)
- scroll_depth_avg: float (1D)
- cta_click_rate: float (1D)
- form_start_rate: float (1D)
- form_completion_rate: float (1D)
- [3D reserved]

# Cognitive Load Layer (10D)
- visual_clutter_score: float (1D)
- hierarchy_depth: int (1D)
- cta_visibility: float (1D)
- f_pattern_score: float (1D)
- navigation_complexity: float (1D)
- content_density: float (1D)
- [4D reserved]

# Context Layer (20D)
- device_type_onehot: [3D]  # desktop, mobile, tablet
- traffic_source_onehot: [4D]  # organic, paid, direct, social
- user_segment_onehot: [3D]  # new, returning, high_intent
- time_of_day_onehot: [4D]  # morning, afternoon, evening, night
- [6D reserved]

# Portfolio/Metadata Layer (20D)
- project_category_onehot: [5D]  # saas, ecommerce, portfolio, blog, landing
- tech_stack_features: [10D]  # One-hot or count for common tech
- tag_features: [5D]  # Top 5 tag categories
```

**Total: 190D** (with room to grow to 256D)

### 2.2 Normalization Strategy

```python
# Continuous features: StandardScaler (mean=0, std=1)
- All float metrics (conversion_rate, bounce_rate, etc.)
- Performance scores (Lighthouse, LCP, FID, CLS)
- Cognitive load scores

# Count features: Log1p + StandardScaler
- num_components, num_animations, num_events, num_assets
- page_size_kb, num_requests

# Categorical features: One-hot encoding (already done)
- layout_type, device_type, traffic_source, etc.

# Distributions: L2 normalization
- component_type_distribution, emphasis_distribution
- Ensures sum = 1.0, normalized vector
```

---

## 3. Encoder Architecture

### 3.1 Network Design

**Goal:** Compress 190D state → 32D latent representation

```python
class WebpageVAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 190,
        latent_dim: int = 32,
        hidden_dims: List[int] = [128, 96, 64],
    ):
        """
        Encoder: WebpageState → Latent Distribution
        
        Architecture:
            Input (190D)
              ↓
            Linear(190 → 128) + BatchNorm + ReLU + Dropout(0.1)
              ↓
            Linear(128 → 96) + BatchNorm + ReLU + Dropout(0.1)
              ↓
            Linear(96 → 64) + BatchNorm + ReLU + Dropout(0.1)
              ↓
            ┌─────────────────┬─────────────────┐
            ↓                 ↓                 
        Linear(64 → 32)   Linear(64 → 32)
        μ (mean)          log_σ² (variance)
        """
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 190) - Vectorized webpage state
        
        Returns:
            mu: (batch_size, 32) - Latent mean
            logvar: (batch_size, 32) - Latent log-variance
        """
```

### 3.2 Design Rationale

- **Progressive compression:** 190 → 128 → 96 → 64 → 32
  - Gradual dimensionality reduction prevents information bottleneck
  - Each layer learns abstractions at different granularities

- **BatchNorm + ReLU + Dropout:**
  - BatchNorm: Stabilizes training, reduces internal covariate shift
  - ReLU: Non-linearity for learning complex mappings
  - Dropout (0.1): Light regularization, prevents overfitting on small seed dataset

- **Two-head output:**
  - `μ` (mean): Point estimate of latent representation
  - `log_σ²` (log-variance): Uncertainty estimate
  - Using log-variance ensures σ² > 0 (numerical stability)

### 3.3 Latent Space Properties

**What should the 32D latent capture?**

Dimensions should correspond to semantic design properties:
1. **Visual Style (8D):** Color palette, typography, spacing density
2. **Layout Structure (6D):** Grid vs. flex, hierarchy depth, component arrangement
3. **Content Strategy (6D):** CTA prominence, copy length, media richness
4. **Performance Profile (4D):** Speed, optimization level, asset efficiency
5. **Outcome Quality (4D):** Conversion potential, engagement level, UX quality
6. **Context (4D):** Device type, audience segment, traffic source

**Verification:** After training, we'll test semantic consistency via:
- Interpolation: z₁ → z₂ should produce visually coherent intermediate states
- Arithmetic: z_hero + z_minimal - z_complex should yield simplified hero
- Clustering: Similar designs should cluster in latent space

---

## 4. Decoder Architecture

### 4.1 Network Design

**Goal:** Reconstruct 190D state from 32D latent + predict outcomes

```python
class WebpageVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        output_dim: int = 190,
        hidden_dims: List[int] = [64, 96, 128],
    ):
        """
        Decoder: Latent → Reconstructed State + Outcomes
        
        Architecture:
            Latent z (32D)
              ↓
            Linear(32 → 64) + BatchNorm + ReLU
              ↓
            Linear(64 → 96) + BatchNorm + ReLU
              ↓
            Linear(96 → 128) + BatchNorm + ReLU
              ↓
            ┌────────────────────────────┬──────────────────────┐
            ↓                            ↓                      
        Linear(128 → 190)           Linear(128 → 10)
        Reconstructed State         Outcome Predictions
        (mixed activations)         (sigmoid/linear)
        """
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch_size, 32) - Latent representation
        
        Returns:
            x_recon: (batch_size, 190) - Reconstructed state
            outcomes: (batch_size, 10) - Predicted outcomes
                [conversion_rate, bounce_rate, time_on_page,
                 scroll_depth, cta_click_rate, form_start_rate,
                 form_completion_rate, ...]
        """
```

### 4.2 Output Activation Functions

Different activation functions for different feature types:

```python
# Reconstruction head (190D)
x_recon = decoder_hidden  # (batch, 128)

# Split by feature type
recon_components = []

# 1. Continuous normalized features (0-1 range) → Sigmoid
continuous_mask = [0:100]  # Visual, performance, cognitive scores
recon_components.append(torch.sigmoid(self.recon_continuous(x_recon)))

# 2. Count features (non-negative integers) → Softplus
count_mask = [100:120]  # num_components, num_animations, etc.
recon_components.append(F.softplus(self.recon_counts(x_recon)))

# 3. One-hot features (categorical) → Softmax per category
onehot_groups = [
    (120, 125),  # layout_type (5 classes)
    (125, 128),  # device_type (3 classes)
    # ... etc
]
for start, end in onehot_groups:
    recon_components.append(F.softmax(self.recon_onehot[start:end](x_recon), dim=-1))

# 4. Distribution features (sums to 1) → Softmax
dist_groups = [
    (140, 158),  # component_type_distribution (18 classes)
    # ... etc
]
for start, end in dist_groups:
    recon_components.append(F.softmax(self.recon_dist[start:end](x_recon), dim=-1))

x_recon = torch.cat(recon_components, dim=-1)

# Outcome prediction head (10D)
outcomes = torch.sigmoid(self.outcome_head(x_recon))  # All outcomes in [0, 1]
```

### 4.3 Design Rationale

- **Mirror encoder:** Symmetric architecture (64 → 96 → 128) aids gradient flow
- **Separate outcome head:** Direct supervision on business metrics
- **Mixed activations:** Respects different feature types (counts, probabilities, continuous)

---

## 5. Loss Function

### 5.1 Total Loss Formulation

```python
total_loss = β_recon * L_reconstruction 
           + β_kl * L_kl_divergence 
           + β_outcome * L_outcome_prediction
           + β_consistency * L_consistency
```

### 5.2 Component Losses

#### **5.2.1 Reconstruction Loss**

```python
def reconstruction_loss(x_recon, x_true, feature_masks):
    """
    Weighted reconstruction loss accounting for feature importance
    """
    loss = 0.0
    
    # Visual/Design features (high weight)
    visual_mask = feature_masks['visual']  # [0:80]
    loss += 2.0 * F.mse_loss(x_recon[:, visual_mask], x_true[:, visual_mask])
    
    # Content features (high weight)
    content_mask = feature_masks['content']  # [80:110]
    loss += 2.0 * F.mse_loss(x_recon[:, content_mask], x_true[:, content_mask])
    
    # Performance features (medium weight)
    perf_mask = feature_masks['performance']  # [110:130]
    loss += 1.0 * F.mse_loss(x_recon[:, perf_mask], x_true[:, perf_mask])
    
    # Outcome features (low weight - predicted separately)
    outcome_mask = feature_masks['outcomes']  # [130:140]
    loss += 0.5 * F.mse_loss(x_recon[:, outcome_mask], x_true[:, outcome_mask])
    
    # Categorical features (cross-entropy)
    for cat_slice in feature_masks['categorical_groups']:
        loss += F.cross_entropy(
            x_recon[:, cat_slice], 
            torch.argmax(x_true[:, cat_slice], dim=-1)
        )
    
    return loss / len(feature_masks)  # Normalize by number of groups
```

**Weighting Rationale:**
- Visual/content features: High weight (2.0) - Core to design understanding
- Performance: Medium weight (1.0) - Important but derivative
- Outcomes: Low weight (0.5) - Handled by separate outcome head

#### **5.2.2 KL Divergence Loss**

```python
def kl_divergence_loss(mu, logvar):
    """
    KL(q(z|x) || p(z)) where p(z) = N(0, I)
    
    = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

**Purpose:** Regularize latent space to be standard normal N(0, I)
- Enables sampling: z ~ N(0, I) produces valid states
- Enables interpolation: Linear paths in z-space are meaningful
- Prevents posterior collapse: Forces encoder to use all 32 dimensions

**β-VAE Annealing:**
```python
# Start with β = 0.0, anneal to β = 0.5 over 20 epochs
β_kl = min(0.5, epoch / 20 * 0.5)
```
- Allows encoder to learn good representations first
- Gradually enforces Gaussian prior
- Prevents KL term from dominating early in training

#### **5.2.3 Outcome Prediction Loss**

```python
def outcome_prediction_loss(outcomes_pred, outcomes_true):
    """
    Direct supervision on business outcomes
    """
    # Conversion rate, bounce rate, etc. (all in [0, 1])
    mse_loss = F.mse_loss(outcomes_pred, outcomes_true)
    
    # Ranking loss: Ensure relative ordering is correct
    # Higher conversion rate samples should have higher predictions
    ranking_loss = 0.0
    for i in range(len(outcomes_true)):
        for j in range(i+1, len(outcomes_true)):
            if outcomes_true[i, 0] > outcomes_true[j, 0]:  # i has higher conversion
                # Penalize if prediction disagrees
                ranking_loss += F.relu(outcomes_pred[j, 0] - outcomes_pred[i, 0])
    
    return mse_loss + 0.1 * ranking_loss
```

**Purpose:** Learn which design properties correlate with good outcomes
- MSE: Accurate point predictions
- Ranking: Relative ordering (which design is better?)

#### **5.2.4 Consistency Loss**

```python
def consistency_loss(z, x_recon):
    """
    Ensure latent space has desired properties
    """
    # Smooth latent manifold: Nearby z should have nearby reconstructions
    z_noise = z + torch.randn_like(z) * 0.1
    x_recon_noise = decoder(z_noise)
    
    smoothness_loss = F.mse_loss(x_recon, x_recon_noise)
    
    return smoothness_loss
```

**Purpose:** Regularize latent space for interpolation quality

### 5.3 Loss Weight Schedule

```python
# Initial training (Epochs 1-10)
β_recon = 1.0
β_kl = 0.0  # Disable KL initially
β_outcome = 0.5
β_consistency = 0.1

# Mid training (Epochs 11-30)
β_recon = 1.0
β_kl = 0.0 → 0.5  # Linear annealing
β_outcome = 1.0
β_consistency = 0.2

# Late training (Epochs 31+)
β_recon = 1.0
β_kl = 0.5
β_outcome = 1.0
β_consistency = 0.3
```

---

## 6. Training Strategy

### 6.1 Dataset: Awwwards Seed States

**Phase 1:** Train on 24 Awwwards patterns
- Small dataset, high quality
- Risk of overfitting
- Mitigation: Strong regularization (dropout, KL, data augmentation)

**Data Augmentation:**
```python
def augment_webpage_state(state: WebpageState) -> WebpageState:
    """
    Apply random perturbations to create training variations
    """
    augmented = state.copy()
    
    # 1. Component shuffling (30% chance)
    if random.random() < 0.3:
        random.shuffle(augmented.components)
    
    # 2. Emphasis jitter (±1 level)
    for comp in augmented.components:
        comp.emphasis_level = np.clip(
            comp.emphasis_level + random.choice([-1, 0, 1]),
            0, 5
        )
    
    # 3. Spacing jitter (±1 level)
    for layout in augmented.layout:
        layout.padding_level = np.clip(layout.padding_level + random.choice([-1, 0, 1]), 0, 5)
        layout.margin_level = np.clip(layout.margin_level + random.choice([-1, 0, 1]), 0, 5)
    
    # 4. Color palette shift (slight HSV perturbation)
    if augmented.styling:
        # Shift hue ±10 degrees, saturation ±5%, value ±5%
        augmented.styling.primary_color = perturb_color(
            augmented.styling.primary_color,
            hue_delta=random.uniform(-10, 10)
        )
    
    # 5. Outcome noise (±5% relative)
    if augmented.outcomes:
        augmented.outcomes.conversion_rate *= random.uniform(0.95, 1.05)
        augmented.outcomes.bounce_rate *= random.uniform(0.95, 1.05)
    
    return augmented
```

**Augmentation Result:** 24 base states × 10 augmentations = 240 effective training samples

### 6.2 Training Configuration

```python
# Model
latent_dim = 32
hidden_dims_encoder = [128, 96, 64]
hidden_dims_decoder = [64, 96, 128]

# Optimization
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

# Training
batch_size = 8  # Small batches for small dataset
epochs = 100
early_stopping_patience = 15

# Regularization
dropout = 0.1
β_kl_max = 0.5
gradient_clip = 1.0
```

### 6.3 Training Loop

```python
for epoch in range(epochs):
    # Anneal KL weight
    β_kl = min(β_kl_max, epoch / 20 * β_kl_max)
    
    for batch in train_loader:
        # Forward pass
        x = batch['state_vector']  # (batch, 190)
        mu, logvar = encoder(x)
        z = reparameterize(mu, logvar)
        x_recon, outcomes_pred = decoder(z)
        
        # Loss computation
        loss_recon = reconstruction_loss(x_recon, x, feature_masks)
        loss_kl = kl_divergence_loss(mu, logvar)
        loss_outcome = outcome_prediction_loss(outcomes_pred, batch['outcomes'])
        loss_consist = consistency_loss(z, x_recon)
        
        total_loss = (1.0 * loss_recon + 
                     β_kl * loss_kl + 
                     1.0 * loss_outcome + 
                     0.2 * loss_consist)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    
    # Validation
    val_loss = validate(model, val_loader)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, f'vae_epoch_{epoch}.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break
    
    scheduler.step()
```

### 6.4 Validation Strategy

**Train/Val Split:** 80/20 (19 train, 5 validation)
- Small validation set acceptable for seed data
- Later will expand with partner site data

**Validation Metrics:**
```python
val_metrics = {
    'reconstruction_mse': compute_mse(x_recon, x_true),
    'outcome_mae': compute_mae(outcomes_pred, outcomes_true),
    'latent_space_variance': z.std(dim=0).mean(),  # Should be ~1.0
    'kl_divergence': kl_loss.item(),
    'interpolation_quality': test_interpolation(model, val_states),
}
```

---

## 7. Latent Space Validation

### 7.1 Reconstruction Quality

**Test:** Feed state through encoder → decoder, measure reconstruction accuracy

```python
def test_reconstruction(model, test_states):
    """
    Metrics:
    - Component type accuracy (% correct)
    - Layout structure similarity (tree edit distance)
    - Color palette error (ΔE color distance)
    - Outcome prediction error (MAE)
    """
    results = []
    
    for state in test_states:
        x = vectorize_state(state)
        mu, logvar = model.encoder(x)
        z = mu  # Use mean (no sampling)
        x_recon, outcomes_pred = model.decoder(z)
        
        # Devectorize and compare
        state_recon = devectorize_state(x_recon)
        
        results.append({
            'component_accuracy': compare_components(state, state_recon),
            'layout_similarity': compute_tree_similarity(state.layout, state_recon.layout),
            'color_error': compute_color_distance(state.styling, state_recon.styling),
            'outcome_mae': abs(state.outcomes.conversion_rate - outcomes_pred[0].item()),
        })
    
    return pd.DataFrame(results)
```

**Expected Results:**
- Component accuracy: >90% (should preserve component types)
- Layout similarity: >0.85 (tree structure mostly intact)
- Color error: <10 ΔE (perceptually similar colors)
- Outcome MAE: <2% (within ±2% conversion rate)

### 7.2 Interpolation Quality

**Test:** Interpolate between two states, verify intermediate states are coherent

```python
def test_interpolation(model, state_a, state_b, num_steps=10):
    """
    Interpolate: z_a → z_b
    Check: Are intermediate states visually coherent?
    """
    x_a = vectorize_state(state_a)
    x_b = vectorize_state(state_b)
    
    mu_a, _ = model.encoder(x_a.unsqueeze(0))
    mu_b, _ = model.encoder(x_b.unsqueeze(0))
    
    # Linear interpolation in latent space
    alphas = torch.linspace(0, 1, num_steps)
    intermediate_states = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * mu_a + alpha * mu_b
        x_recon, outcomes = model.decoder(z_interp)
        intermediate_states.append(devectorize_state(x_recon[0]))
    
    # Check coherence
    coherence_scores = []
    for i in range(len(intermediate_states) - 1):
        # Adjacent states should be similar
        similarity = compute_state_similarity(
            intermediate_states[i], 
            intermediate_states[i+1]
        )
        coherence_scores.append(similarity)
    
    # Visualization
    visualize_interpolation_sequence(intermediate_states)
    
    return np.mean(coherence_scores)
```

**Expected Results:**
- Coherence score: >0.90 (adjacent states are similar)
- Visual check: Smooth transitions (no sudden jumps)
- Outcome trajectory: Monotonic or smooth (not erratic)

### 7.3 Semantic Clustering

**Test:** Do similar designs cluster in latent space?

```python
def test_semantic_clustering(model, states_by_category):
    """
    Check if similar designs cluster together
    
    Categories:
    - landing pages
    - saas dashboards
    - portfolio sites
    - ecommerce product pages
    """
    # Encode all states
    latents = []
    labels = []
    
    for category, states in states_by_category.items():
        for state in states:
            x = vectorize_state(state)
            mu, _ = model.encoder(x.unsqueeze(0))
            latents.append(mu.squeeze(0).detach().numpy())
            labels.append(category)
    
    latents = np.array(latents)
    
    # Dimensionality reduction for visualization
    from sklearn.manifold import TSNE
    latents_2d = TSNE(n_components=2, perplexity=5).fit_transform(latents)
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    for category in set(labels):
        mask = [l == category for l in labels]
        plt.scatter(
            latents_2d[mask, 0], 
            latents_2d[mask, 1],
            label=category,
            alpha=0.6
        )
    plt.legend()
    plt.title("Latent Space Visualization (t-SNE)")
    plt.savefig("latent_space_clustering.png")
    
    # Compute silhouette score (cluster quality)
    from sklearn.metrics import silhouette_score
    score = silhouette_score(latents, labels)
    
    return score
```

**Expected Results:**
- Silhouette score: >0.3 (moderate clustering)
- Visual: Distinct clusters for different site types
- Outcome gradient: High-conversion sites in one region

### 7.4 Arithmetic Operations

**Test:** Do latent space operations have semantic meaning?

```python
def test_latent_arithmetic(model, states):
    """
    Test operations like:
    - z_minimal + (z_rich - z_average) = more visual richness
    - z_hero_layout + (z_grid_layout - z_flex_layout) = hero with grid
    """
    # Example: Add visual richness
    z_minimal = encode(states['minimal_landing'])
    z_rich = encode(states['rich_portfolio'])
    z_average = encode(states['average_saas'])
    
    z_result = z_minimal + (z_rich - z_average)
    state_result = decode(z_result)
    
    # Check: Does result have more visual elements than minimal?
    assert len(state_result.components) > len(states['minimal_landing'].components)
    
    visualize_comparison([
        states['minimal_landing'],
        state_result,
        states['rich_portfolio']
    ])
```

---

## 8. Integration with Self-Model

### 8.1 Control Loop Architecture

Once VAE is trained, integrate with RNN self-model for autonomous editing:

```
┌─────────────────────────────────────────────────────────────┐
│                   Autonomous Editing Loop                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Goal: "Increase conversion rate"                     │
│       │                                                     │
│       ▼                                                     │
│  WebpageState (current)                                    │
│       │                                                     │
│       ▼─────────────────────────────────────┐              │
│  VAE Encoder                                │              │
│       │                                      │              │
│       ▼                                      │              │
│  Latent z_t (32D)                            │              │
│       │                                      │              │
│       ▼                                      │              │
│  Self-Model RNN (propose edits)              │              │
│       │                                      │              │
│       ▼                                      │              │
│  Edit candidates: Δz₁, Δz₂, ..., Δzₙ        │              │
│       │                                      │              │
│       ▼                                      │              │
│  VAE Decoder (predict outcomes)              │              │
│       │                                      │              │
│       ▼                                      │              │
│  Predicted states + outcomes                 │              │
│       │                                      │              │
│       ▼                                      │              │
│  Select best edit: argmax(conversion_rate)   │              │
│       │                                      │              │
│       ▼                                      │              │
│  Stability check: σ(Jacobian) < 1?          │              │
│       │                                      │              │
│       ├─ Yes ──► Apply edit ─────────────────┘              │
│       │                                                     │
│       └─ No ──► Reject (unstable)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Edit Representation

**Edit Vector (50D):**
```python
edit_vector = [
    edit_type_onehot: [10D],  # add_component, modify_layout, change_color, etc.
    target_component_id: [5D],  # Which component to modify
    delta_latent: [32D],  # Change in latent space
    edit_magnitude: [1D],  # Strength of edit (0-1)
    confidence: [1D],  # Model's confidence
    expected_outcome_delta: [1D],  # Predicted conversion rate change
]
```

### 8.3 Training Self-Model (Future Work)

**Phase 2-3:** Once we have edit history from 1000 sites:

```python
# Dataset: (state_t, edit, state_t+1, outcome_delta)
# Train RNN to predict:
#   - Which edit to apply given current state + goal
#   - Expected outcome improvement
#   - Stability metrics (spectral radius)

self_model = RNNEditPolicy(
    state_dim=32,  # VAE latent
    edit_dim=50,
    hidden_dim=64,
    num_layers=2
)

# Loss: Imitation learning + outcome prediction
loss = imitation_loss(edit_pred, edit_true) + \
       outcome_loss(outcome_pred, outcome_true) + \
       stability_loss(spectral_radius)
```

---

## 9. Expected Results

### 9.1 Phase 1 (Seed Data Training)

**Week 2 Goals:**
- ✅ Train VAE on 24 Awwwards states
- ✅ Achieve reconstruction MSE < 0.1
- ✅ Demonstrate smooth interpolation
- ✅ Verify semantic clustering (silhouette > 0.3)

**Deliverables:**
- Trained VAE checkpoint (.pt file)
- Validation report (reconstruction quality, interpolation, clustering)
- Latent space visualization (t-SNE plot)

### 9.2 Phase 2 (Partner Data Training)

**Week 4-8 Goals:**
- Train VAE on 1000+ site snapshots
- Improve outcome prediction (MAE < 1%)
- Learn fine-grained design patterns
- Validate transfer from seed data

**Expected Improvements:**
- Reconstruction MSE: 0.1 → 0.05
- Outcome MAE: 2% → 1%
- Latent space coverage: Better generalization

### 9.3 Phase 3 (Self-Model Integration)

**Month 2-3 Goals:**
- Train RNN self-model on edit sequences
- Demonstrate autonomous editing
- Validate stability guarantees (σ < 1)
- Deploy for A/B testing

---

## 10. Implementation Timeline

### Week 2: VAE Architecture + Training

**Day 1-2:** Model implementation
- [ ] Create `webpage_models/vae_world_model.py`
- [ ] Implement encoder (190D → 32D)
- [ ] Implement decoder (32D → 190D + outcomes)
- [ ] Implement loss functions

**Day 3-4:** Training script
- [ ] Create `experiments/train_vae_seed_data.py`
- [ ] Integrate Awwwards loader
- [ ] Implement data augmentation
- [ ] Training loop with β-annealing

**Day 5-6:** Validation
- [ ] Create `experiments/validate_latent_space.py`
- [ ] Test reconstruction quality
- [ ] Test interpolation smoothness
- [ ] Visualize latent space (t-SNE)

**Day 7:** Documentation
- [ ] Document latent dimensions
- [ ] Save trained checkpoint
- [ ] Generate validation report

---

## 11. Success Criteria

**Minimum Viable VAE:**
- ✅ Reconstruction MSE < 0.15
- ✅ Outcome MAE < 3%
- ✅ Interpolation coherence > 0.85
- ✅ Silhouette score > 0.25

**Production-Ready VAE:**
- ✅ Reconstruction MSE < 0.05
- ✅ Outcome MAE < 1%
- ✅ Interpolation coherence > 0.95
- ✅ Silhouette score > 0.4
- ✅ Validated on 100+ real sites

---

## 12. References

**Research Foundation:**
- `AI_APP_BUILDER_DESIGN.md` - Core architecture
- `DYNAMICS_TRANSFER_ANALYSIS.md` - Transfer learning analysis
- Demo validation results (spectral radius < 1 proven)

**Implementation Assets:**
- `webpage_data_collection/state_schema.py` - State representation
- `webpage_data_collection/dataset_builder.py` - Vectorization
- `webpage_data_collection/awwwards_loader.py` - Seed data

**Similar Work:**
- β-VAE: Higgins et al. (2017) - Disentangled representations
- Conditional VAE: Sohn et al. (2015) - Conditioned generation
- World Models: Ha & Schmidhuber (2018) - VAE for RL environments
