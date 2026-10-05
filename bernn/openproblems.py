"""OpenProblems-oriented training helpers for BERNN.

This module deliberately keeps OpenProblems itself as an optional integration.
BERNN owns model fitting and Optuna optimization; callers provide ``score_fn``
to evaluate a candidate embedding with the benchmark (or another objective).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import gc
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

from .config.training_config import TrainingConfig
from .dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout
from .single_cell import _dense_frame, _select_feature_indices


class _OpenProblemsGroupedWarmupTrainer(TrainAEClassifierHoldout):
    """OpenProblems grouped trainer that snapshots the model after warmup.

    This is intentionally isolated to the OpenProblems integration. Legacy BERNN
    checkpointing and validation-MCC selection remain unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._openproblems_warmup_state: Optional[dict[str, Any]] = None
        self._openproblems_warmup_epoch: Optional[int] = None

    def warmup_loop(
        self, optimizer_ae, scheduler, ae, celoss, loader, triplet_loss, mseloss,
        warmup, epoch, optimizer_b, values, loggers, loaders, run, mapping=True,
    ):
        result = super().warmup_loop(
            optimizer_ae, scheduler, ae, celoss, loader, triplet_loss, mseloss,
            warmup, epoch, optimizer_b, values, loggers, loaders, run, mapping,
        )
        # Capture the state *after* every true warmup epoch. The last snapshot is
        # therefore the exact model state at the warmup -> supervised boundary,
        # including the early-stop boundary if warmup terminates early.
        if warmup:
            self._openproblems_warmup_state = {
                name: copy.deepcopy(module.state_dict())
                for name, module in self._iter_torch_modules()
            }
            self._openproblems_warmup_epoch = int(epoch)
        return result

    def restore_openproblems_warmup_state(self):
        if not self._openproblems_warmup_state:
            raise RuntimeError("No post-warmup OpenProblems state was captured")
        modules = dict(self._iter_torch_modules())
        for name, state in self._openproblems_warmup_state.items():
            if name not in modules:
                raise RuntimeError(f"Cannot restore post-warmup model: missing module '{name}'")
            modules[name].load_state_dict(state)


class _OpenProblemsFinalEpochTrainer(TrainAEClassifierHoldout):
    """OpenProblems-only trainer behavior without changing BERNN legacy trainers.

    BERNN's legacy holdout trainer selects/restores checkpoints by validation MCC.
    OpenProblems optimization must not use BERNN metrics for model selection, so
    this subclass captures the final epoch state and restores that state after
    the parent fit has completed its own bookkeeping.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._openproblems_last_state: Optional[dict[str, Any]] = None
        self._openproblems_last_epoch: Optional[int] = None

    def _notify_epoch(self, payload: Mapping[str, Any]):
        # Capture state after each completed joint-training epoch. Do not use any
        # BERNN validation metric to decide whether this state is better.
        self._openproblems_last_state = {
            name: copy.deepcopy(module.state_dict())
            for name, module in self._iter_torch_modules()
        }
        self._openproblems_last_epoch = int(payload.get("epoch", -1))

    def restore_openproblems_final_state(self):
        if not self._openproblems_last_state:
            raise RuntimeError("No final OpenProblems training state was captured")
        modules = dict(self._iter_torch_modules())
        for name, state in self._openproblems_last_state.items():
            if name in modules:
                modules[name].load_state_dict(state)


def fit_openproblems_once(
    adata: Any,
    *,
    layer: str = "normalized",
    batch_key: str = "batch",
    label_key: str = "cell_type",
    n_hvg: Optional[int] = 2000,
    dloss: str = "inverseTriplet",
    n_epochs: int = 200,
    warmup: int = 20,
    early_warmup_stop: int = 50,
    batch_size: int = 256,
    n_layers: int = 2,
    layer1: int = 256,
    scaler: str = "standard",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    margin: float = 1.0,
    smoothing: float = 0.1,
    nu: float = 1.0,
    random_state: int = 1,
    num_workers: int = 0,
    device: Optional[str] = None,
    return_trainer: bool = False,
):
    """Fit BERNN on the full OpenProblems dataset with no holdout split.

    This path is intentionally separate from BERNN's legacy fit APIs. Every row
    is used for fitting. BERNN's internal metrics may still be computed for
    bookkeeping by the inherited trainer, but they do not control early stopping,
    checkpoint selection, trial ranking, or the returned embedding. The returned
    model is always the final completed epoch.
    """
    import torch

    if batch_key not in adata.obs:
        raise KeyError(f"AnnData obs is missing required batch key {batch_key!r}")
    if label_key not in adata.obs:
        raise KeyError(f"AnnData obs is missing required label key {label_key!r}")

    feature_idx = _select_feature_indices(adata, n_hvg)
    frame = _dense_frame(adata, feature_idx, layer)
    labels = adata.obs[label_key].astype(str).to_numpy()
    batches = adata.obs[batch_key].astype(str).to_numpy()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainingConfig(
        optimize_hyperparams=False, dloss=dloss, class_triplet=False,
        variational=False, kan=False, n_layers=int(n_layers), layer1=int(layer1),
        tied_weights=False, use_mapping=True, rec_loss="l1", scaler=scaler,
        log1p=False, use_l1=True, prune_network=False, update_grid=False,
        n_epochs=int(n_epochs), warmup=int(warmup), n_repeats=1,
        bs=int(batch_size), num_workers=int(num_workers), groupkfold=True,
        device=device, dataset=str(getattr(adata, "uns", {}).get("dataset_id", "openproblems")),
        exp_id="bernn_openproblems_exact",
        # Disable MCC-driven early stopping for this dedicated path.
        early_stop=int(n_epochs) + 1,
        early_warmup_stop=int(early_warmup_stop),
    )
    config.lr=float(learning_rate); config.wd=float(weight_decay)
    config.dropout=float(dropout); config.margin=float(margin)
    config.smoothing=float(smoothing); config.nu=float(nu)
    config.thres=0.0; config.gamma=0.1; config.beta=0.0

    trainer = _OpenProblemsFinalEpochTrainer(
        config=config, groupkfold=True, pools=False, keep_models=False,
        log_inputs=False, log_plots=False, log_tb=False, log_mlflow=False,
        log_dvclive=False,
    )
    params = {
        "lr": float(learning_rate), "dropout": float(dropout),
        "wd": float(weight_decay), "margin": float(margin),
        "smoothing": float(smoothing), "scaler": scaler, "gamma": 0.1,
        "beta": 0.0, "nu": float(nu), "thres": 0.0,
        "prune_threshold": 0.0, "warmup": int(warmup), "l1": 0.0,
        "reg_entropy": 0.0, "layer1": int(layer1),
        "n_layers": int(n_layers),
    }
    for idx in range(2, int(n_layers) + 1):
        params[f"layer{idx}"] = max(16, int(layer1) // (2 ** (idx - 1)))

    # No external validation/test and no internal split: every row is fit data.
    trainer.fit(
        frame, labels, groups_train=batches, params=params,
        internal_validation=False,
    )
    # Parent fit restores best-valid-MCC for legacy semantics. Replace it with the
    # final-epoch state for this OpenProblems-only path.
    trainer.restore_openproblems_final_state()
    embedding = trainer.transform(frame, groups_test=batches).astype(np.float32, copy=False)
    if return_trainer:
        return embedding, trainer
    return embedding


def fit_openproblems_grouped_once(
    adata: Any,
    *,
    layer: str = "normalized",
    batch_key: str = "batch",
    label_key: str = "cell_type",
    n_hvg: Optional[int] = 2000,
    dloss: str = "inverseTriplet",
    variational: bool = False,
    kan: bool = False,
    class_triplet: bool = False,
    class_triplet_w: float = 1.0,
    rec_loss: str = "l1",
    gamma: float = 0.1,
    beta: float = 0.0,
    l1: float = 0.0,
    reg_entropy: float = 0.0,
    thres: float = 0.0,
    n_epochs: int = 200,
    warmup: int = 20,
    early_stop: int = 50,
    early_warmup_stop: int = 50,
    batch_size: int = 256,
    n_layers: int = 2,
    layer1: int = 256,
    scaler: str = "standard",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    margin: float = 1.0,
    smoothing: float = 0.1,
    nu: float = 1.0,
    random_state: int = 1,
    num_workers: int = 0,
    device: Optional[str] = None,
    n_splits: int = 5,
    epoch_callback: Any = None,
    return_trainer: bool = False,
):
    """Fit BERNN with true grouped train/valid/test splits for OpenProblems.

    This is the legacy-style ``gkf=1`` mode: batches are split into disjoint
    train/valid/test groups, and BERNN restores the checkpoint with the best
    validation MCC. The returned embedding is then produced for the complete
    transductive dataset. Train/valid/test MCCs are recomputed afterward through
    the same public prediction path so their values are directly comparable.
    """
    if batch_key not in adata.obs:
        raise KeyError(f"AnnData obs is missing required batch key {batch_key!r}")
    if label_key not in adata.obs:
        raise KeyError(f"AnnData obs is missing required label key {label_key!r}")
    if int(n_splits) < 3:
        raise ValueError("n_splits must be >= 3 for train/valid/test grouped mode")

    feature_idx = _select_feature_indices(adata, n_hvg)
    frame = _dense_frame(adata, feature_idx, layer)
    labels = adata.obs[label_key].astype(str).to_numpy()
    batches = adata.obs[batch_key].astype(str).to_numpy()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainingConfig(
        optimize_hyperparams=False, dloss=dloss, class_triplet=bool(class_triplet),
        class_triplet_w=float(class_triplet_w), variational=bool(variational), kan=bool(kan),
        n_layers=int(n_layers), layer1=int(layer1), tied_weights=False, use_mapping=True,
        rec_loss=str(rec_loss), scaler=scaler,
        log1p=False, use_l1=True, prune_network=False, update_grid=False,
        n_epochs=int(n_epochs), warmup=int(warmup), n_repeats=int(n_splits),
        train_after_warmup=True,
        bs=int(batch_size), num_workers=int(num_workers), groupkfold=True,
        device=device, dataset=str(getattr(adata, "uns", {}).get("dataset_id", "openproblems")),
        exp_id="bernn_openproblems_grouped",
        early_stop=int(early_stop),
        early_warmup_stop=int(early_warmup_stop),
    )
    config.lr=float(learning_rate); config.wd=float(weight_decay)
    config.dropout=float(dropout); config.margin=float(margin)
    config.smoothing=float(smoothing); config.nu=float(nu)
    config.thres=float(thres); config.gamma=float(gamma); config.beta=float(beta); config.l1=float(l1); config.reg_entropy=float(reg_entropy)

    trainer = _OpenProblemsGroupedWarmupTrainer(
        config=config, groupkfold=True, pools=False, keep_models=False,
        log_inputs=False, log_plots=False, log_tb=False, log_mlflow=False,
        log_dvclive=False, epoch_callback=epoch_callback,
    )
    trainer.seed = int(random_state)
    params = {
        "lr": float(learning_rate), "dropout": float(dropout),
        "wd": float(weight_decay), "margin": float(margin),
        "smoothing": float(smoothing), "scaler": scaler, "gamma": float(gamma),
        "beta": float(beta), "nu": float(nu), "thres": float(thres),
        "prune_threshold": 0.0, "warmup": int(warmup), "l1": float(l1),
        "reg_entropy": float(reg_entropy), "layer1": int(layer1), "n_layers": int(n_layers),
    }
    for idx in range(2, int(n_layers) + 1):
        params[f"layer{idx}"] = max(16, int(layer1) // (2 ** (idx - 1)))

    trainer.fit(
        frame, labels, groups_train=batches, params=params,
        internal_validation=True,
    )

    split_metrics: dict[str, float] = {}
    split_batches: dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        split_frame = trainer.data["inputs"][split]
        indices = np.asarray(split_frame.index, dtype=int)
        y_true = labels[indices]
        batch_true = batches[indices]
        y_pred = trainer.predict(frame.iloc[indices], batches_test=batch_true)
        split_metrics[f"{split}_mcc"] = float(
            matthews_corrcoef(y_true.astype(str), np.asarray(y_pred).astype(str))
        )
        split_batches[split] = sorted(set(batch_true.tolist()))

    trainer.openproblems_split_metrics = split_metrics
    trainer.openproblems_split_batches = split_batches

    # Current behavior: score the restored best-validation-MCC checkpoint as the
    # end-of-training candidate. Preserve it exactly, then temporarily restore
    # the post-warmup snapshot to generate a second full-dataset embedding.
    embedding = trainer.transform(frame, groups_test=batches).astype(np.float32, copy=False)
    end_state = {
        name: copy.deepcopy(module.state_dict())
        for name, module in trainer._iter_torch_modules()
    }
    trainer.restore_openproblems_warmup_state()
    trainer.openproblems_after_warmup_embedding = trainer.transform(
        frame, groups_test=batches
    ).astype(np.float32, copy=False)
    modules = dict(trainer._iter_torch_modules())
    for name, state in end_state.items():
        if name in modules:
            modules[name].load_state_dict(state)
    trainer.openproblems_after_warmup_epoch = trainer._openproblems_warmup_epoch

    if return_trainer:
        return embedding, trainer
    return embedding


ScoreResult = float | Mapping[str, float]
ScoreFn = Callable[[np.ndarray], ScoreResult]
ParamSuggester = Callable[[Any], Mapping[str, Any]]
TrialCallback = Callable[[Any, Mapping[str, float]], None]


@dataclass
class OpenProblemsFitResult:
    """Result returned by :func:`fit_openproblems`."""

    embedding: np.ndarray
    trainer: Any
    best_params: dict[str, Any]
    best_score: float
    study: Any
    best_trial_number: int
    best_trial_metrics: dict[str, float] = field(default_factory=dict)


def _default_openproblems_params(
    trial: Any,
    *,
    max_warmup: Optional[int] = None,
) -> dict[str, Any]:
    """Conservative BERNN search space for single-cell batch integration.

    The default space intentionally keeps the batch/domain loss fixed. Changing
    ``dloss`` can materially change the optimization problem and should be an
    explicit experiment via ``param_suggester``.
    """

    warmup_choices = [0, 10, 20, 50]
    if max_warmup is not None:
        warmup_choices = [value for value in warmup_choices if value < int(max_warmup)]
        if not warmup_choices:
            warmup_choices = [0]

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.30, step=0.05),
        "margin": trial.suggest_float("margin", 0.5, 2.0, step=0.25),
        "smoothing": trial.suggest_float("smoothing", 0.0, 0.20, step=0.05),
        "nu": trial.suggest_float("nu", 0.25, 2.0, log=True),
        "scaler": trial.suggest_categorical("scaler", ["standard", "robust", "standard_per_batch", "robust_per_batch"]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "layer1": trial.suggest_categorical("layer1", [128, 256, 512, 1024]),
        "warmup": trial.suggest_categorical("warmup", warmup_choices),
    }


def _normalize_score_result(
    result: ScoreResult,
    *,
    objective_key: str,
) -> tuple[float, dict[str, float]]:
    """Normalize scalar/dict scorer outputs to an objective and metric mapping."""

    if isinstance(result, Mapping):
        metrics: dict[str, float] = {}
        for key, value in result.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if objective_key in metrics:
            score = metrics[objective_key]
        elif "score" in metrics:
            score = metrics["score"]
        elif "mean_score" in metrics:
            score = metrics["mean_score"]
        elif len(metrics) == 1:
            score = next(iter(metrics.values()))
        else:
            raise ValueError(
                "score_fn returned multiple metrics but no objective metric. "
                f"Return {objective_key!r}, 'score', or 'mean_score'."
            )
    else:
        score = float(result)
        metrics = {objective_key: score}

    if not np.isfinite(score):
        raise ValueError(f"score_fn returned a non-finite objective: {score!r}")
    return float(score), metrics


def fit_openproblems(
    adata: Any,
    score_fn: ScoreFn,
    *,
    n_trials: int = 20,
    trial_epochs: int = 40,
    final_epochs: int = 200,
    timeout: Optional[float] = None,
    objective_key: str = "score",
    direction: str = "maximize",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    sampler: Any = None,
    pruner: Any = None,
    param_suggester: Optional[ParamSuggester] = None,
    enqueue_params: Optional[list[Mapping[str, Any]]] = None,
    on_trial_complete: Optional[TrialCallback] = None,
    random_state: int = 1,
    retrain_best: bool = True,
    **fit_kwargs: Any,
) -> OpenProblemsFitResult:
    """Optimize BERNN for an OpenProblems-style embedding objective.

    ``score_fn`` receives each candidate embedding and may return either a
    scalar objective or a mapping containing ``objective_key``. This keeps the
    official OpenProblems/scIB stack outside BERNN while making that benchmark
    score the quantity Optuna actually optimizes.

    HPO uses ``trial_epochs`` as a lower-fidelity budget. Once the study
    completes, BERNN retrains once with the best hyperparameters for
    ``final_epochs``.
    """

    if score_fn is None:
        raise TypeError("fit_openproblems requires score_fn(embedding)")
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1")
    if int(trial_epochs) < 1 or int(final_epochs) < 1:
        raise ValueError("trial_epochs and final_epochs must be >= 1")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_openproblems requires Optuna. Install BERNN with its optimization "
            "dependencies or run: pip install optuna"
        ) from exc

    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            seed=int(random_state),
            multivariate=True,
        )
    if pruner is None:
        # BERNN currently exposes the OpenProblems score only after a trial fit.
        # Pruning at that point would discard a completed trial without saving
        # compute. Callers can pass a pruner once an intermediate scorer/hook is
        # available.
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )

    for known_params in enqueue_params or []:
        study.enqueue_trial(dict(known_params), skip_if_exists=True)

    if param_suggester is None:
        suggest = lambda trial: _default_openproblems_params(
            trial, max_warmup=int(trial_epochs)
        )
    else:
        suggest = param_suggester
    base_fit_kwargs = dict(fit_kwargs)
    base_fit_kwargs.pop("return_trainer", None)
    base_fit_kwargs.pop("n_epochs", None)

    def objective(trial: Any) -> float:
        trial_params = dict(suggest(trial))
        trial_fit_kwargs = dict(base_fit_kwargs)
        trial_fit_kwargs.update(trial_params)
        trial_fit_kwargs["n_epochs"] = int(trial_epochs)
        trial_fit_kwargs.setdefault("random_state", int(random_state))

        trainer = None
        embedding = None
        metrics: dict[str, float] = {}
        try:
            embedding, trainer = fit_openproblems_once(
                adata,
                return_trainer=True,
                **trial_fit_kwargs,
            )
            score, metrics = _normalize_score_result(
                score_fn(embedding),
                objective_key=objective_key,
            )

            for key, value in metrics.items():
                if np.isfinite(value):
                    trial.set_user_attr(f"metric/{key}", float(value))
            trial.set_user_attr("trial_epochs", int(trial_epochs))
            if trainer is not None:
                valid_mcc = getattr(trainer, "best_valid_mcc", None)
                if valid_mcc is None:
                    valid_mcc = getattr(trainer, "best_mcc", None)
                try:
                    if valid_mcc is not None and np.isfinite(float(valid_mcc)):
                        trial.set_user_attr("bernn/best_valid_mcc", float(valid_mcc))
                except (TypeError, ValueError):
                    pass

            trial.report(score, step=int(trial_epochs))
            if trial.should_prune():
                raise optuna.TrialPruned()

            if on_trial_complete is not None:
                on_trial_complete(trial, dict(metrics))
            return float(score)
        finally:
            del embedding
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study.optimize(objective, n_trials=int(n_trials), timeout=timeout)

    best_params = dict(base_fit_kwargs)
    best_params.update(study.best_trial.params)
    best_trial_metrics = {
        key.removeprefix("metric/"): float(value)
        for key, value in study.best_trial.user_attrs.items()
        if key.startswith("metric/")
    }

    if not retrain_best:
        raise ValueError(
            "retrain_best=False is not supported yet because trial trainers are "
            "released between trials to keep GPU memory bounded."
        )

    final_fit_kwargs = dict(best_params)
    final_fit_kwargs["n_epochs"] = int(final_epochs)
    final_fit_kwargs.setdefault("random_state", int(random_state))
    embedding, trainer = fit_openproblems_once(
        adata,
        return_trainer=True,
        **final_fit_kwargs,
    )

    return OpenProblemsFitResult(
        embedding=np.asarray(embedding, dtype=np.float32),
        trainer=trainer,
        best_params=best_params,
        best_score=float(study.best_value),
        study=study,
        best_trial_number=int(study.best_trial.number),
        best_trial_metrics=best_trial_metrics,
    )


__all__ = [
    "OpenProblemsFitResult",
    "fit_openproblems_once",
    "fit_openproblems",
]
