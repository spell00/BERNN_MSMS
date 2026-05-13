from pathlib import Path

import numpy as np


def test_readme_includes_pip_install_quickstart():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "pip install bernn" in readme
    assert "## Basic usage" in readme


def test_official_docs_index_mentions_pip_quickstart():
    docs = Path("OFFICIAL_DOCUMENTATION.md").read_text(encoding="utf-8")
    assert "pip install bernn" in docs
    assert "README.md" in docs


def test_readme_quickstart_api_smoke(monkeypatch):
    from bernn import TrainAEClassifierHoldout
    from bernn.config.training_config import TrainingConfig

    calls = {"fit_predict": 0, "predict": 0}

    def fake_fit_predict(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        groups_train=None,
        groups_test=None,
        cross_validation=False,
        cross_test=False,
        **kwargs,
    ):
        calls["fit_predict"] += 1
        assert X_test is not None
        assert y_test is not None
        assert groups_train is not None
        assert groups_test is not None
        assert cross_validation is False
        assert cross_test is False
        return np.array([0] * len(X_test))

    def fake_predict(self, X):
        calls["predict"] += 1
        return np.array([0] * len(X))

    monkeypatch.setattr(TrainAEClassifierHoldout, "fit_predict", fake_fit_predict)
    monkeypatch.setattr(TrainAEClassifierHoldout, "predict", fake_predict)

    bernn_config = TrainingConfig(
        optimize_hyperparams=False,
        n_trials=1,
        n_repeats=1,
        n_epochs=2,
        warmup=1,
        n_layers=1,
        layer1=64,
        device="cpu",
        dloss="inverseTriplet",
        scaler="standard",
        bs=8,
    )

    X_train = np.random.randn(10, 4)
    y_train = np.array([0, 1] * 5)
    X_test = np.random.randn(4, 4)
    y_test = np.array([0, 1, 0, 1])
    batches_train = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    batches_test = np.array([0, 1, 0, 1])

    # Mirror README quickstart structure.
    trainer_cls = TrainAEClassifierHoldout
    trainer = trainer_cls(config=bernn_config, log_metrics=True, keep_models=False)

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

    preds = trainer.predict(X_test)

    assert preds_encoded.shape[0] == len(X_test)
    assert preds.shape[0] == len(X_test)
    assert calls["fit_predict"] == 1
    assert calls["predict"] == 1
