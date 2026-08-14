# Using BERNN

BERNN follows the familiar estimator sequence `fit(...); predict(...)`, with
batch identifiers supplied alongside the feature matrix. Install the released
package with:

```bash
python -m pip install bernn==1.0.2
```

## Inputs

- `X`: a sample-by-feature numeric matrix with shape `(n_samples, n_features)`.
- `y`: one biological class label per sample.
- `groups`: one acquisition batch, site, run, or technical-domain identifier
  per sample.
- Train, validation, and test matrices must have the same feature columns in
  the same order.

## Inductive fitting

Use inductive mode when fitting must not see validation or future test spectra.

```python
from bernn import TrainAEClassifierHoldout
from bernn.config.training_config import TrainingConfig

config = TrainingConfig(
    n_epochs=100,
    optimize_hyperparams=False,
    device="cpu",
)
model = TrainAEClassifierHoldout(config=config, log_metrics=False)

model.fit(X_train, y_train, groups_train=batch_train)
y_pred = model.predict(X_new, groups_test=batch_new)
```

## Fully transductive fitting

Supply explicit validation and cross-test matrices when spectra from all three
splits should participate in reconstruction and batch-domain learning. In the
evaluation protocol used by BE_leaderboard, each spectrum appears exactly once
in that unsupervised/domain pool, only training labels contribute supervised
gradients, validation controls model selection, and cross-test labels are used
for monitoring and evaluation.

```python
model.fit(
    X_train,
    y_train,
    groups_train=batch_train,
    X_valid=X_valid,
    y_valid=y_valid,
    groups_valid=batch_valid,
    X_test=X_test,
    y_test=y_test,
    groups_test=batch_test,
)
y_test_pred = model.predict(X_test, groups_test=batch_test)
```

Passing validation or test data changes the learning setting from inductive to
transductive. Report that choice explicitly when publishing results.

## Hyperparameters

The configuration above demonstrates the API; it is not a universal best
configuration. Batch structure, sample size, feature count, class balance, and
signal strength all affect suitable architecture and optimization settings.
For competitive results, select hyperparameters using training/validation data
only, or use a recommendation system that has been independently validated on
held-out datasets. Do not use cross-test performance for model selection.

The most important starting parameters are `dloss`, `n_layers`, `layer1`,
`warmup`, `n_epochs`, `scaler`, `bs`, and `device`. Set
`optimize_hyperparams=True` only when the associated search cost and validation
protocol are appropriate for the experiment.

## Prediction

`predict` returns class predictions for a sample-by-feature matrix. Supply the
matching batch vector when predicting spectra from known or new acquisition
domains:

```python
y_pred = model.predict(X_new, groups_test=batch_new)
```

Keep the feature order identical to training, and apply the same upstream
feature-generation rules to every split.
