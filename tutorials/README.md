# BERNN tutorials

Start with the minimal examples, then move to optimized configurations for production runs.

## Quick start: Minimal examples (5 min)

[minimal_examples.ipynb](minimal_examples.ipynb)

Four concise end-to-end examples showing both trainers with and without pooled data structures:

1. `TrainAEClassifierHoldout` + `pools=False`
2. `TrainAEClassifierHoldout` + `pools=True`
3. `TrainAEThenClassifierHoldout` + `pools=False`
4. `TrainAEThenClassifierHoldout` + `pools=True`

**Runtime:** ~1-2 min per example on CPU (minimal config).

## Production: TrainAEClassifierHoldout optimized (20+ min)

[optimized_classifier_holdout_all_configs.ipynb](optimized_classifier_holdout_all_configs.ipynb)

Thorough grid covering all valid cross-validation and cross-test configurations:

- `cross_validation=False`, `cross_test=False` (standard holdout)
- `cross_validation=True`, `cross_test=False` (full cross-validation)
- `cross_validation=True`, `cross_test=True` (cross-validation with transductive test)

Each configuration is tested with `pools=False` and `pools=True` (6 variants total).

Shows explicit optimization parameters like `n_trials`, `fixed_hyperparams`, `n_repeats`, layer sizing, and training schedule.

**Runtime:** ~15+ min total on CPU with n_trials=5 (set `run_all=False` to preview).

## Production: TrainAEThenClassifierHoldout optimized (20+ min)

[optimized_ae_then_classifier_holdout_all_configs.ipynb](optimized_ae_then_classifier_holdout_all_configs.ipynb)

Identical structure to the classifier-holdout variant above but for the AE-then-classifier workflow.

All valid cross-validation and cross-test configurations tested with both pool modes.

**Runtime:** ~15+ min total on CPU with n_trials=5 (set `run_all=False` to preview).

## Important notes

- `cross_test=True` can only be used when `cross_validation=True`.
- Both optimized notebooks have `run_all=False` by default; set to `True` to execute full runs.
- Dataset used: `../data/benchmark/intensities.csv` (auto-loaded from repo).
- All notebooks use CPU by default; change `device='cpu'` to `device='cuda:0'` for GPU.
- Mandatory data contract:
  - `groups_train` is required.
  - `groups_test` is required when `X_test` is provided.
