# BERNN Training Parameter Reference

This document is the exhaustive parameter reference for BERNN training APIs.

It covers:
- `TrainingConfig` fields.
- Direct constructor arguments for holdout trainers.
- Non-config constructor flags.
- Runtime data arguments used by `fit` and `fit_predict`.
- Hyperparameters consumed during Ax/optimization runs.

## 1) Parameter Sources And Precedence

Supported training entrypoints:
- `TrainAEThenClassifierHoldout`
- `TrainAEClassifierHoldout`

The holdout constructors accept:
- `config` as `TrainingConfig`, `dict`, legacy args object, or `None`.
- Direct overrides for common parameters: `n_epochs`, `dloss`, `variational`, `n_layers`, `layer1`, `n_repeats`, `warmup`, `device`, `kan`, `scaler`, `bs`, `n_trials`.
- `**kwargs` filtered to valid `TrainingConfig` fields.

Precedence rule:
1. Direct constructor overrides win when provided (non-`None`).
2. Values in `config` are used next.
3. Remaining values fall back to `TrainingConfig` defaults.

## 2) Complete `TrainingConfig` Field Reference

### Model architecture
- `dloss: str = "inverseTriplet"`
  - Domain-loss mode. Allowed values: `revTriplet`, `revDANN`, `DANN`, `inverseTriplet`, `normae`, `no`.
- `variational: bool = False`
  - Enables VAE behavior when `True`.
- `tied_weights: bool = False`
  - Enables tied encoder/decoder weights.
- `use_mapping: bool = True`
  - Enables batch mapping in reconstruction path.
- `n_layers: int = 1`
  - Number of classifier hidden layers.
- `layer1: int | None = None`
  - Optional explicit width seed for first layer. Deeper defaults are auto-derived by halving with floor at 16.

### Training loop
- `n_epochs: int = 1000`
  - Number of training epochs.
- `n_repeats: int = 1`
  - Number of holdout repeats.
- `early_stop: int = 50`
  - Early-stop patience for main training.
- `early_warmup_stop: int = 50`
  - Early-stop patience during warmup.
- `train_after_warmup: bool = False`
  - Continue AE/domain learning after warmup when `True`.
- `warmup_after_warmup: bool = False`
  - Run extra warmup-style phase after warmup when `True`.
- `warmup: int = 100`
  - Warmup epochs.
- `device: str = "cpu"`
  - Device identifier. Common values: `cpu`, `cuda`, `cuda:0`, `cuda:1`, ...
- `use_sigmoid: bool = False`
  - Apply sigmoid activation at AE output.

### Loss and regularization
- `rec_loss: str = "l1"`
  - Reconstruction loss type.
- `classif_loss: str = "ce"`
  - Classification loss type.
- `threshold: float = 0.0`
  - Generic threshold value used in some regularization/filter paths.
- `kan: bool = False`
  - Enables KAN-specific behavior.
- `use_l1: bool = True`
  - Enables L1 regularization path.
- `prune_network: bool = True`
  - Enables pruning path.
- `clip_val: float = 1.0`
  - Gradient clipping max norm/value depending on implementation path.
- `update_grid: bool = True`
  - Enables grid update behavior in KAN paths.

### Data processing
- `embeddings_meta: int = 0`
  - Embedding size for metadata branch.
- `groupkfold: bool = True`
  - Use grouped split strategy when possible.
- `log1p: bool = True`
  - Apply log1p transform to inputs.
- `scaler: str = "standard"`
  - Input scaling strategy.

### Experiment tracking
- `exp_id: str = "bernn_training"`
  - Experiment identifier (for example, MLflow experiment name).
- `model_name: str = "ae_then_classifier_holdout"`
  - Model/training variant identifier.

### Logging and evaluation
- `random_recs: bool = False`
  - Legacy reconstruction sampling behavior.
- `predict_tests: bool = False`
  - Whether to predict test partitions in some flows.
- `n_agg: int = 1`
  - Number of trailing values used for stable validation summaries.

### Batching and dataloaders
- `bs: int = 32`
  - Batch size.
- `bdisc: bool = True`
  - Batch discriminator toggle in relevant model paths.

### Hyperparameter optimization controls
- `n_trials: int = 1`
  - Number of optimization trials.
- `random: bool = True`
  - Random-search style mode toggle in relevant entrypoints.
- `scheduler: str = "ReductionLROnPlateau"`
  - LR scheduler name. Common values include `CosineAnnealingLR`, `ReductionLROnPlateau`, `CosineAnnealingWarmRestarts`.
- `optimize_hyperparams: bool = True`
  - If `False`, optimization parameter space is empty.
- `fixed_hyperparams: dict[str, Any] | None = None`
  - Parameter values forced as fixed (removed from search space).

### Legacy compatibility fields
- `triplet_dloss: bool = True`
  - Legacy switch used by older paths.
- `prune_threshold: float = 0.0`
  - Legacy pruning threshold.
- `prune_neurites_threshold: float = 0.0`
  - Legacy neurite pruning threshold.
- `berm: str = "bernn"`
  - Legacy BERM/BERRN mode value.
- `disc_b_warmup: int = 0`
  - Discriminator warmup control.
- `update_grid_warmup: int = 0`
  - Grid update warmup control.
- `remove_zeros: bool = False`
  - Legacy input filtering behavior.
- `batches: object | None = None`
  - Legacy external batches container.
- `pool_metrics_enc: object | None = None`
  - Optional encoded-space pooled metrics object.
- `pool_metrics_rec: object | None = None`
  - Optional reconstructed-space pooled metrics object.

## 3) Direct Holdout Constructor Overrides

These parameters are available directly in both holdout constructors:
- `n_epochs`
- `dloss`
- `variational`
- `n_layers`
- `layer1`
- `n_repeats`
- `warmup`
- `device`
- `kan`
- `scaler`
- `bs`
- `n_trials`

These are convenience overrides for the corresponding `TrainingConfig` fields.

## 4) Non-Config Constructor Flags (Holdout Trainers)

Shared constructor flags:
- `fix_thres: float = -1`
  - If in `[0, 1)`, forces threshold hyperparameter; otherwise threshold remains learnable/default.
- `log_metrics: bool = False`
  - Enables keeping batch-effect metrics.
- `keep_models: bool = True`
  - Save model artifacts.
- `log_inputs: bool = False`
  - Log input projections/metrics.
- `log_plots: bool = False`
  - Plot PCA/UMAP/CCA/LDA diagnostics.
- `log_tb: bool = False`
  - Enable TensorBoard logging.
- `log_mlflow: bool = False`
  - Enable MLflow logging.
- `log_dvclive: bool = False`
  - Enable DVC Live logging.
- `groupkfold: bool = True`
  - Split behavior toggle for grouped folds.
- `pools: bool = False`
  - Enable pooled sample structures.
- `load_tb: bool = False`
  - Reload historical TensorBoard runs.

Additional constructor input:
- `**kwargs`
  - Filtered to valid `TrainingConfig` fields only.

## 5) Runtime Training API Arguments (`TrainAE`)

### `fit(...)`
- `X_train`
- `y_train`
- `groups_train=None`
- `X_test=None`
- `y_test=None`
- `groups_test=None`
- `params=None`
- `cross_validation=False`
- `cross_test=False`
- `val_size=0.2`
- `**kwargs`

### `fit_predict(...)`
- `X_train`
- `y_train`
- `X_test=None`
- `y_test=None`
- `groups_train=None`
- `groups_test=None`
- `params=None`
- `cross_validation=False`
- `cross_test=False`
- `val_size=0.2`
- `**kwargs`

### `predict(...)`
- `X`

Important data contract:
- Training batch IDs are mandatory: `groups_train` (or `groups`) must be provided.
- Test batch IDs are mandatory when test data is provided: if `X_test` is passed, `groups_test` must be passed.

## 6) Optimization Parameter Keys Used At Runtime

During optimization/training, BERNN may consume the following parameter keys in `params` dictionaries:
- `lr`
- `dropout`
- `wd`
- `margin`
- `smoothing`
- `scaler`
- `gamma`
- `beta`
- `zeta`
- `nu`
- `thres`
- `l1`
- `reg_entropy`
- `prune_threshold`
- `warmup`
- `disc_b_warmup`
- `triplet_margin`
- `knn_n_neighbors`
- `n_layers`
- `layer1`, `layer2`, ..., `layerN`

Notes:
- If `n_layers` is greater than 1 and deeper layers are missing, defaults are auto-derived by halving each step with floor at 16.
- If optimization is disabled (`optimize_hyperparams=False`), Ax search space is emptied.
- Any parameter in `fixed_hyperparams` (and explicit `layer1`) is removed from search space and injected as fixed.
