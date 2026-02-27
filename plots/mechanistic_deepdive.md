# Mechanistic Deep-Dive Analysis

**Dataset**: ar1
**Hidden Dimension**: 128
**Model Seed**: 0

## Self-Model Analysis

### Jacobian Dynamics
- Spectral radius: 1.2299 ± 0.0007
- Range: [1.2262, 1.2304]
- Frobenius norm: 6.6034 ± 0.0030

### Error Distribution
- Mean error: 0.126520
- Median error: 0.118999
- 95th percentile: 0.246709
- Max error: 0.424185
- High-error samples (top 10%): 980 (10.0%)

### Generalization (Longer Horizons)

| Horizon | MSE | Divergence |
|---------|-----|------------|
| 10 | 0.026412 | 0.191620 |
| 100 | 0.053875 | 0.286957 |
| 20 | 0.043038 | 0.251526 |
| 30 | 0.050802 | 0.273357 |
| 50 | 0.057217 | 0.291999 |

## World-Model Analysis

### Latent Transition Structure
- Max eigenvalue modulus: 0.8676
- Min eigenvalue modulus: 0.0019
- Fixed point exists: stable
- Fixed point norm: 5594.3818

### Latent Space Geometry
- Latent dimension: 128
- Effective rank: 7.01
- PC1 cumulative variance: 0.4314
- PC2 cumulative variance: 0.5976
- PC3 cumulative variance: 0.7016

### Reconstruction Error Distribution
- Mean error: 0.165456
- Median error: 0.156267
- 95th percentile: 0.324346

### Generalization (Longer Sequences)

| Horizon | MSE | Divergence |
|---------|-----|------------|
| 10 | 0.026292 | 0.204284 |
| 100 | 0.018836 | 0.171378 |
| 20 | 0.020729 | 0.181163 |
| 30 | 0.020167 | 0.176637 |
| 50 | 0.020682 | 0.178233 |

## Key Insights

1. **Stability**: Compare spectral radii and eigenvalue moduli between paradigms
2. **Error Modes**: Identify where failures occur (distributional outliers)
3. **Generalization**: Track performance degradation with longer horizons
4. **Latent Structure**: Examine effective dimensionality and geometry
