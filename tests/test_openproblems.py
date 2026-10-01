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
