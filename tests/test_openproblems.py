import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import bernn.openproblems as op


optuna = pytest.importorskip("optuna")


def test_fit_openproblems_optimizes_external_score_and_retrains(monkeypatch):
    calls = []

    def fake_fit_transform(adata, *, return_trainer=False, **kwargs):
        calls.append(dict(kwargs))
        value = float(kwargs.get("dropout", 0.0))
        embedding = np.full((4, 2), value, dtype=np.float32)
        trainer = SimpleNamespace(best_valid_mcc=0.75)
        return (embedding, trainer) if return_trainer else embedding

    monkeypatch.setattr(op, "fit_openproblems_once", fake_fit_transform)

    def suggest(trial):
        return {"dropout": trial.suggest_categorical("dropout", [0.0, 0.2])}

    sampler = optuna.samplers.GridSampler({"dropout": [0.0, 0.2]})
    result = op.fit_openproblems(
        object(),
        lambda embedding: {
            "score": float(embedding.mean()),
            "graph_score": float(embedding.mean()) + 0.1,
        },
        n_trials=2,
        trial_epochs=2,
        final_epochs=10,
        sampler=sampler,
        param_suggester=suggest,
        device="cpu",
    )

    assert result.best_score == pytest.approx(0.2)
    assert result.best_params["dropout"] == pytest.approx(0.2)
    assert result.best_trial_metrics["graph_score"] == pytest.approx(0.3)
    assert result.embedding.mean() == pytest.approx(0.2)
    assert len(calls) == 3
    assert [call["n_epochs"] for call in calls[:2]] == [2, 2]
    assert calls[-1]["n_epochs"] == 10
    assert calls[-1]["dropout"] == pytest.approx(0.2)


def test_normalize_score_result_requires_named_objective_for_multiple_metrics():
    with pytest.raises(ValueError, match="no objective metric"):
        op._normalize_score_result({"ari": 0.5, "nmi": 0.6}, objective_key="score")

    score, metrics = op._normalize_score_result(
        {"openproblems": 0.7, "ari": 0.5},
        objective_key="openproblems",
    )
    assert score == pytest.approx(0.7)
    assert metrics["ari"] == pytest.approx(0.5)


def test_fit_openproblems_enqueues_known_baseline(monkeypatch):
    calls = []

    def fake_fit_transform(adata, *, return_trainer=False, **kwargs):
        calls.append(dict(kwargs))
        value = float(kwargs.get("dropout", 0.0))
        emb = np.full((3, 2), value, dtype=np.float32)
        trainer = SimpleNamespace(best_valid_mcc=0.1)
        return (emb, trainer) if return_trainer else emb

    monkeypatch.setattr(op, "fit_openproblems_once", fake_fit_transform)

    def suggest(trial):
        return {"dropout": trial.suggest_categorical("dropout", [0.0, 0.2])}

    result = op.fit_openproblems(
        object(),
        lambda emb: float(emb.mean()),
        n_trials=1,
        trial_epochs=2,
        final_epochs=3,
        param_suggester=suggest,
        enqueue_params=[{"dropout": 0.2}],
        device="cpu",
    )
    assert result.best_score == pytest.approx(0.2)
    assert result.study.trials[0].params["dropout"] == pytest.approx(0.2)
    assert calls[0]["dropout"] == pytest.approx(0.2)


def test_grouped_openproblems_captures_and_restores_post_warmup_state(monkeypatch):
    import torch
    from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout

    trainer = op._OpenProblemsGroupedWarmupTrainer.__new__(op._OpenProblemsGroupedWarmupTrainer)
    trainer._openproblems_warmup_state = None
    trainer._openproblems_warmup_epoch = None
    trainer.ae = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.ae.weight.fill_(1.0)

    def fake_parent_warmup(self, *args, **kwargs):
        with torch.no_grad():
            self.ae.weight.fill_(2.5)
        return 1, self.ae, True

    monkeypatch.setattr(TrainAEClassifierHoldout, "warmup_loop", fake_parent_warmup)
    trainer.warmup_loop(
        None, None, trainer.ae, None, None, None, None,
        True, 7, None, None, None, None, None, True,
    )

    assert trainer._openproblems_warmup_epoch == 7
    assert float(trainer._openproblems_warmup_state["ae"]["weight"].item()) == pytest.approx(2.5)

    with torch.no_grad():
        trainer.ae.weight.fill_(9.0)
    trainer.restore_openproblems_warmup_state()
    assert float(trainer.ae.weight.item()) == pytest.approx(2.5)


def test_grouped_openproblems_exposes_early_stop_parameter():
    sig = inspect.signature(op.fit_openproblems_grouped_once)
    assert sig.parameters["early_stop"].default == 50


def test_grouped_openproblems_exposes_epoch_callback_parameter():
    sig = inspect.signature(op.fit_openproblems_grouped_once)
    assert "epoch_callback" in sig.parameters
    assert sig.parameters["epoch_callback"].default is None


def test_openproblems_exposes_warmup_early_stop_parameters():
    grouped = inspect.signature(op.fit_openproblems_grouped_once)
    full = inspect.signature(op.fit_openproblems_once)
    assert grouped.parameters["early_warmup_stop"].default == 50
    assert full.parameters["early_warmup_stop"].default == 50
