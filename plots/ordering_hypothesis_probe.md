# Ordering Hypothesis Probe (ar1)

This report compares **Self-Model-First** vs **World-Model-First** across matched seeds and hidden dimensions.

## Attribution Summary

### Hidden Dim 16

- Matched seed pairs: 10

| Metric | Self Mean | World Mean | Δ(Self-World) | 95% CI | Winner |
|---|---:|---:|---:|---:|---|
| one_step_mse | 0.0176305 | 0.0177181 | -8.75632e-05 | [-0.00127293, 0.00090304] | self_model_first_better |
| rollout_divergence_50 | 0.0533468 | 0.0190297 | 0.0343171 | [0.0305465, 0.0393755] | world_model_first_better |
| spectral_radius | 1.09222 | 181.95 | -180.857 | [-391.007, -32.532] | self_model_first_better |
| perturbation_return_rate | 1.18139 | 0.29539 | 0.885999 | [0.701006, 1.05787] | self_model_first_better |

Mechanism probes:
- Local gain (mean): self=0.522508, world=3.02543e-07, ratio(world/self)=5.79021e-07
- World latent covariance condition (mean): 13.2696
- World transition spectral norm (mean): 181.95

### Hidden Dim 32

- Matched seed pairs: 10

| Metric | Self Mean | World Mean | Δ(Self-World) | 95% CI | Winner |
|---|---:|---:|---:|---:|---|
| one_step_mse | 0.0132252 | 0.0177138 | -0.00448861 | [-0.00523088, -0.00355103] | self_model_first_better |
| rollout_divergence_50 | 0.0528195 | 0.0189958 | 0.0338237 | [0.0308787, 0.037054] | world_model_first_better |
| spectral_radius | 1.1433 | 949.398 | -948.255 | [-1258.79, -726.801] | self_model_first_better |
| perturbation_return_rate | 0.814183 | 0.220814 | 0.593369 | [0.449588, 0.757054] | self_model_first_better |

Mechanism probes:
- Local gain (mean): self=0.653896, world=1.08083e-05, ratio(world/self)=1.6529e-05
- World latent covariance condition (mean): 8.71785
- World transition spectral norm (mean): 949.398

### Hidden Dim 64

- Matched seed pairs: 10

| Metric | Self Mean | World Mean | Δ(Self-World) | 95% CI | Winner |
|---|---:|---:|---:|---:|---|
| one_step_mse | 0.0105706 | 0.0177191 | -0.00714854 | [-0.00728696, -0.00698635] | self_model_first_better |
| rollout_divergence_50 | 0.0492209 | 0.0189943 | 0.0302266 | [0.0291706, 0.031535] | world_model_first_better |
| spectral_radius | 1.20298 | 2671.77 | -2670.57 | [-3078.03, -2241.32] | self_model_first_better |
| perturbation_return_rate | 0.416958 | 0.135203 | 0.281754 | [0.160579, 0.378578] | self_model_first_better |

Mechanism probes:
- Local gain (mean): self=0.802111, world=2.03059e-05, ratio(world/self)=2.53156e-05
- World latent covariance condition (mean): 21.5578
- World transition spectral norm (mean): 2671.77

### Hidden Dim 128

- Matched seed pairs: 10

| Metric | Self Mean | World Mean | Δ(Self-World) | 95% CI | Winner |
|---|---:|---:|---:|---:|---|
| one_step_mse | 0.0101652 | 0.0177102 | -0.00754498 | [-0.00755203, -0.00753875] | self_model_first_better |
| rollout_divergence_50 | 0.048414 | 0.0189638 | 0.0294501 | [0.0285813, 0.0302709] | world_model_first_better |
| spectral_radius | 1.24734 | 5343.16 | -5341.91 | [-6096.5, -4598.83] | self_model_first_better |
| perturbation_return_rate | 0.229163 | -0.0324342 | 0.261597 | [0.133195, 0.400291] | self_model_first_better |

Mechanism probes:
- Local gain (mean): self=0.890595, world=0, ratio(world/self)=0
- World latent covariance condition (mean): 34.7602
- World transition spectral norm (mean): 5343.16

## Notes

- Lower-is-better metrics: one_step_mse, rollout_divergence_50, spectral_radius
- Higher-is-better metric: perturbation_return_rate
- CI is computed on paired seed differences via bootstrap resampling