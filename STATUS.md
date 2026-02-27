# Status

Last updated: 2026-02-25

## Snapshot

- Workspace: `/mnt/c/dev/Cog`
- Repo state: initialized for Git tracking in this session (see staged files below after init)
- Smoke tests: `python3 -m unittest discover -s tests -p 'test_*_smoke.py'` passed (4 tests)
- Current focus: expand test coverage beyond smoke (metric regression checks; shared checkpoint helper consolidation)

## What Exists (Validated)

- VAE and self-model train/eval scripts with config-driven defaults and checkpoint metadata
- Self-model eval includes one-step and rollout metrics plus split reconstruction from checkpoint metadata
- Root checkpoints (`vae_model.pth`, `self_model.pth`) use metadata-wrapped checkpoint format (`model_state_dict`, `config`, `training`)

## Gaps / Risks

- Counterfactual probes are still ad hoc research diagnostics (not yet integrated into experiment train/eval pipelines or tests)
- Test coverage now includes model smoke + tiny-data CLI script smoke + checkpoint round-trip checks; metric regression checks are still missing

## Active Work

- [done] Refactored `counterfactual_generalization_test.py`:
  - metadata-wrapped and legacy checkpoint loading supported
  - model dims inferred from checkpoint `config` / weights
  - SciPy dependency removed (`numpy.trapezoid` used for AUDC)
  - plotting is optional (`matplotlib` only required when plotting flags are used)
  - CLI subcommands added (`shift-sweep`, `perturbation-return`, `lyapunov`)
- [done] Added `tests/test_experiment_scripts_smoke.py`:
  - tiny-data VAE train/eval CLI smoke test
  - tiny-data self-model train/eval CLI smoke test
  - checkpoint metadata / latent-transition artifact round-trip assertions
- [in progress] Add metric regression checks (stable tiny fixtures / expected ranges)

## Validation (2026-02-25)

- `python3 counterfactual_generalization_test.py shift-sweep --num-samples 8 --sequence-length 6 --device cpu` ✅
- `python3 counterfactual_generalization_test.py perturbation-return --num-samples 8 --sequence-length 6 --device cpu` ✅
- `python3 counterfactual_generalization_test.py lyapunov --num-samples 8 --sequence-length 6 --device cpu` ✅
- Earlier smoke tests still pass: `python3 -m unittest discover -s tests -p 'test_*_smoke.py'` (4 tests)
- `python3 -m unittest tests/test_experiment_scripts_smoke.py -v` ✅ (2 tests; CLI train/eval round-trip)
- `python3 -m unittest discover -s tests -p 'test_*_smoke.py'` ✅ (6 tests total)

## Next Priorities

1. Add metric regression checks (tiny deterministic fixtures, expected metric ranges/tolerances).
2. Move shared checkpoint-load/infer helpers into a reusable utility module to avoid duplication across scripts.
3. Add a reproducible experiment runner with timestamped output directories and saved run config.

## Update Practice

- Append a dated note here when a milestone lands (new script/tooling/tests, dataset changes, or eval baselines).
- Keep this file committed so repo status is visible without reading terminal history.
