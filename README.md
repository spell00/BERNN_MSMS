# 

# BERNN-MSMS

Minimal README for quick usage.

Longer historical content is kept in [LEGACY_README.md](LEGACY_README.md).

## Install

```bash
pip install bernn
```

## Basic usage

```python
from bernn import TrainAEClassifierHoldout

trainer_cls = TrainAEClassifierHoldout
trainer = trainer_cls(config=bernn_config, log_metrics=True, keep_models=False)

# Train and predict in one call
preds_encoded = trainer.fit_predict(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
    groups_train=batches_train,
    groups_test=batches_test,
    cross_validation=False,
    cross_test=False,
)

# Decode predictions back to original labels
preds = trainer.predict(X_test)
```

Important runtime contract:

- groups_train is mandatory.
- If X_test is provided, groups_test is mandatory.

## Important parameters

Focus on these first:

- optimize_hyperparams: enable/disable Ax optimization.
- n_trials: number of optimization trials.
- fixed_hyperparams: force values and remove them from search.
- n_repeats: number of holdout repeats.
- n_layers, layer1: classifier depth and width seed.
- dloss: domain loss mode.
- warmup, n_epochs: core training schedule.
- device: cpu/cuda target.
- scaler, bs: preprocessing and batch size.

## Official documentation

- Full reference: [OFFICIAL_DOCUMENTATION.md](OFFICIAL_DOCUMENTATION.md)
- Full parameter catalog: [TRAINING_PARAMETERS.md](TRAINING_PARAMETERS.md)
- Minimal runnable examples notebook (4 variants): [tutorials/minimal_examples.ipynb](tutorials/minimal_examples.ipynb)
- Optimized all-config notebook (TrainAEClassifierHoldout): [tutorials/optimized_classifier_holdout_all_configs.ipynb](tutorials/optimized_classifier_holdout_all_configs.ipynb)
- Optimized all-config notebook (TrainAEThenClassifierHoldout): [tutorials/optimized_ae_then_classifier_holdout_all_configs.ipynb](tutorials/optimized_ae_then_classifier_holdout_all_configs.ipynb)
- Historical CLI-heavy guide: [LEGACY_README.md](LEGACY_README.md)
