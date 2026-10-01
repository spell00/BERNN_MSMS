"""OpenProblems-oriented training helpers for BERNN.

This module deliberately keeps OpenProblems itself as an optional integration.
BERNN owns model fitting and Optuna optimization; callers provide ``score_fn``
to evaluate a candidate embedding with the benchmark (or another objective).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch

from .single_cell import fit_transform_anndata


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
            embedding, trainer = fit_transform_anndata(
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
    embedding, trainer = fit_transform_anndata(
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
    "fit_openproblems",
]
