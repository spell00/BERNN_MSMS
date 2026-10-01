# BERNN-MSMS

Batch-effect-aware representation learning and classification with an
estimator-style `fit(...); predict(...)` interface.

## Install

```bash
pip install bernn
```

## Basic usage

```python
from bernn import TrainAEClassifierHoldout
from bernn.config.training_config import TrainingConfig

config = TrainingConfig(
    n_epochs=100,
    optimize_hyperparams=False,
    device="cpu",
)
model = TrainAEClassifierHoldout(config=config, log_metrics=False)

# Inductive fit: training data only.
model.fit(X_train, y_train, groups_train=batch_train)
y_pred = model.predict(X_new, groups_test=batch_new)
```

Read the complete [BERNN usage guide](TUTORIAL.md) for input shapes, inductive
and fully transductive fitting, prediction, and hyperparameter guidance.

Important runtime contract:

- `groups_train` is mandatory.
- If `X_valid` or `X_test` is supplied, its matching batch vector is mandatory.
- BERNN hyperparameters are dataset-dependent. The examples are interface
  demonstrations, not universal performance-optimal defaults.

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
- num_workers: PyTorch DataLoader subprocesses; defaults to 0.

## Documentation

- [Usage tutorial](TUTORIAL.md)

## Single-cell batch integration

The `feature/openproblems-single-cell` branch adds an AnnData adapter for
single-cell batch integration while keeping BERNN's matrix-based training API
unchanged.

```python
from bernn.single_cell import fit_transform_anndata

embedding = fit_transform_anndata(
    adata,
    layer="normalized",
    batch_key="batch",
    label_key="cell_type",
    n_hvg=2000,
)
adata.obsm["X_emb"] = embedding
```

The adapter uses `var["hvg_score"]` when available, trains BERNN with
`obs["batch"]` as the domain label and `obs["cell_type"]` as the supervised
biological label, and returns the integrated latent representation.
