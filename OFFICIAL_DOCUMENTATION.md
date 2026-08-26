# BERNN Official Documentation

This document is the short index for BERNN documentation.

## Start here

1. Quick install and basic usage: [README.md](README.md)
2. Parameter and API reference: [TRAINING_PARAMETERS.md](TRAINING_PARAMETERS.md)
3. Tutorials start here: [tutorials/README.md](tutorials/README.md)
4. Minimal runnable examples notebook: [tutorials/minimal_examples.ipynb](tutorials/minimal_examples.ipynb)
5. Optimized all-config notebooks:
   - [tutorials/optimized_classifier_holdout_all_configs.ipynb](tutorials/optimized_classifier_holdout_all_configs.ipynb)
   - [tutorials/optimized_ae_then_classifier_holdout_all_configs.ipynb](tutorials/optimized_ae_then_classifier_holdout_all_configs.ipynb)

## What each doc contains

- [README.md](README.md)
  - `pip install bernn`
  - Minimal API usage example
  - Most important tuning parameters
- [TRAINING_PARAMETERS.md](TRAINING_PARAMETERS.md)
  - Exhaustive `TrainingConfig` field documentation
  - Constructor flags and precedence rules
  - Runtime `fit` and `fit_predict` contracts
  - Optimization parameter keys
- [tutorials/README.md](tutorials/README.md)
  - Recommended order: start with minimal, move to optimized
  - Runtime estimates for each notebook
  - Data contract and configuration notes
- [tutorials/minimal_examples.ipynb](tutorials/minimal_examples.ipynb)
  - 4 concise examples in one notebook:
  - `TrainAEClassifierHoldout` with `pools=False`
  - `TrainAEClassifierHoldout` with `pools=True`
  - `TrainAEThenClassifierHoldout` with `pools=False`
  - `TrainAEThenClassifierHoldout` with `pools=True`
- [tutorials/optimized_classifier_holdout_all_configs.ipynb](tutorials/optimized_classifier_holdout_all_configs.ipynb)
  - Thorough `TrainAEClassifierHoldout` optimized grid across all valid `cross_validation` and `cross_test` settings and both pool modes.
- [tutorials/optimized_ae_then_classifier_holdout_all_configs.ipynb](tutorials/optimized_ae_then_classifier_holdout_all_configs.ipynb)
  - Thorough `TrainAEThenClassifierHoldout` optimized grid across all valid `cross_validation` and `cross_test` settings and both pool modes.

## Legacy

Older long-form README content is preserved in [LEGACY_README.md](LEGACY_README.md).
