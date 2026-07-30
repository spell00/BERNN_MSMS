import numpy as np
import pandas as pd
import pytest
import torch
import types
from types import SimpleNamespace

from bernn.config.training_config import TrainingConfig
from bernn.dl.models.pytorch.utils.dataset import get_loaders_no_pool
from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout
from bernn.dl.train.train_ae_then_classifier_holdout import TrainAEThenClassifierHoldout
from bernn.utils.data_getters import get_data


def _args(**overrides):
    ns = SimpleNamespace(
        device="cpu",
        random_recs=0,
        predict_tests=0,
        early_stop=2,
        early_warmup_stop=-1,
        train_after_warmup=0,
        threshold=0.0,
        n_epochs=2,
        n_trials=1,
        rec_loss="l1",
        tied_weights=0,
        random=1,
        variational=0,
        zinb=0,
        use_mapping=1,
        bdisc=0,
        n_repeats=1,
        dloss="inverseTriplet",
        remove_zeros=0,
        n_meta=0,
        embeddings_meta=0,
        groupkfold=0,
        n_layers=2,
        kan=0,
        use_l1=0,
        clip_val=1.0,
        log_metrics=0,
        log_plots=0,
        prune_network=0,
        dataset="custom",
        csv_file="mock.csv",
        log1p=1,
        berm="none",
        pool=0,
        strategy="none",
        best_features_file="",
        n_features=-1,
        bad_batches="",
        controls="",
        exp_id="unit_test",
        warmup_after_warmup=0,
        bs=8,
        n_agg=1,
        update_grid=0,
        prune_threshold=0.0,
        scheduler="ReduceLROnPlateau",
        path=".",
        log_tb=0,

        log_mlflow=0,
        keep_models=0,
        log_inputs=0,
        classif_loss="ce",
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _small_no_pool_data():
    n_train, n_valid, n_test = 6, 4, 4
    n_total = n_train + n_valid + n_test
    feature_cols = [f"f{i}" for i in range(5)]
    meta_cols = ["m0", "m1"]

    def frame(n, cols, offset):
        return pd.DataFrame(np.arange(offset, offset + n * len(cols)).reshape(n, len(cols)), columns=cols)

    data = {
        "inputs": {
            "train": frame(n_train, feature_cols, 0),
            "valid": frame(n_valid, feature_cols, 100),
            "test": frame(n_test, feature_cols, 200),
        },
        "meta": {
            "train": frame(n_train, meta_cols, 300),
            "valid": frame(n_valid, meta_cols, 400),
            "test": frame(n_test, meta_cols, 500),
        },
        "names": {
            "train": np.array([f"train_{i}" for i in range(n_train)]),
            "valid": np.array([f"valid_{i}" for i in range(n_valid)]),
            "test": np.array([f"test_{i}" for i in range(n_test)]),
        },
        "cats": {
            "train": np.array([0, 1, 0, 1, 0, 1]),
            "valid": np.array([0, 1, 0, 1]),
            "test": np.array([1, 0, 1, 0]),
        },
        "batches": {
            "train": np.array([0, 0, 1, 1, 0, 1]),
            "valid": np.array([0, 1, 0, 1]),
            "test": np.array([1, 1, 0, 0]),
        },
        "sets": {
            "train": np.array(["train"] * n_train),
            "valid": np.array(["valid"] * n_valid),
            "test": np.array(["test"] * n_test),
        },
    }
    data["inputs"]["all"] = pd.concat([data["inputs"]["train"], data["inputs"]["valid"], data["inputs"]["test"]], axis=0)
    data["meta"]["all"] = pd.concat([data["meta"]["train"], data["meta"]["valid"], data["meta"]["test"]], axis=0)
    data["names"]["all"] = np.concatenate([data["names"]["train"], data["names"]["valid"], data["names"]["test"]])
    data["cats"]["all"] = np.concatenate([data["cats"]["train"], data["cats"]["valid"], data["cats"]["test"]])
    data["batches"]["all"] = np.concatenate([data["batches"]["train"], data["batches"]["valid"], data["batches"]["test"]])
    data["sets"]["all"] = np.concatenate([data["sets"]["train"], data["sets"]["valid"], data["sets"]["test"]])
    return data


@pytest.mark.unit
def test_custom_get_data_builds_aligned_sets(tmp_path):
    csv_path = tmp_path / "mock.csv"
    rows = []
    for idx in range(24):
        rows.append(
            {
                "names": f"sample_{idx}",
                "labels": f"label_{idx % 3}",
                "batches": f"batch_{idx % 2}",
                "feat0": float(idx),
                "feat1": float(idx + 1),
                "feat2": float(idx + 2),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    args = _args(csv_file=csv_path.name, path=str(tmp_path), pool=0)
    data, _, _ = get_data(str(tmp_path), args, seed=0)

    for split in ("train", "valid", "test", "all"):
        assert len(data["inputs"][split]) == len(data["sets"][split])
        assert len(data["labels"][split]) == len(data["sets"][split])
        assert len(data["batches"][split]) == len(data["sets"][split])


@pytest.mark.unit
def test_get_loaders_no_pool_uses_train_sets_length():
    data = _small_no_pool_data()
    samples_weights = {
        "train": np.ones(len(data["inputs"]["train"])),
        "valid": np.ones(len(data["inputs"]["valid"])),
        "test": np.ones(len(data["inputs"]["test"])),
    }

    loaders = get_loaders_no_pool(data, random_recs=0, samples_weights=samples_weights, triplet_dloss="inverseTriplet", bs=2, device="cpu")

    assert len(loaders["train"].dataset) == len(data["inputs"]["train"])
    assert len(loaders["all"].dataset) == len(data["inputs"]["all"])


@pytest.mark.unit
def test_train_ae_then_classifier_holdout_accepts_training_config(tmp_path):
    config = TrainingConfig(
        device="cpu",
        kan=False,
        use_l1=False,
        prune_network=False,
        groupkfold=False,
        log1p=True,
    )

    trainer = TrainAEThenClassifierHoldout(
        config=config,
        path=str(tmp_path),
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )

    assert trainer.config is config
    assert trainer.args is config


@pytest.mark.unit
def test_holdout_trainer_predict_proba_available(tmp_path):
    config = TrainingConfig(
        device="cpu",
        kan=False,
        use_l1=False,
        prune_network=False,
        groupkfold=False,
        log1p=True,
        bs=4,
    )

    trainer = TrainAEThenClassifierHoldout(
        config=config,
        path=str(tmp_path),
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )

    class FakeAE(torch.nn.Module):
        def predict_proba(self, x):
            n = int(x.shape[0])
            return np.tile(np.array([[0.2, 0.8]], dtype=np.float32), (n, 1))

    trainer.ae = FakeAE()
    X = np.zeros((5, 3), dtype=np.float32)
    probs = trainer.predict_proba(X)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.unit
def test_prepare_data_normalizes_equivalent_string_labels(tmp_path):
    config = TrainingConfig(
        device="cpu",
        kan=False,
        use_l1=False,
        prune_network=False,
        groupkfold=False,
        log1p=True,
        bs=4,
    )

    trainer = TrainAEThenClassifierHoldout(
        config=config,
        path=str(tmp_path),
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )

    X_train = pd.DataFrame(np.random.randn(8, 4))
    y_train = np.array(["0", "1", "0", "1", "0", "1", "0", "1"])
    groups_train = np.array(["b0", "b0", "b1", "b1", "b0", "b1", "b0", "b1"])

    X_test = pd.DataFrame(np.random.randn(4, 4))
    y_test = np.array(["0.0", "1.0", "0.0", "1.0"])
    groups_test = np.array(["b0", "b1", "b0", "b1"])

    trainer._prepare_data(
        X=X_train,
        y=y_train,
        groups=groups_train,
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        cross_validation=False,
        cross_test=False,
        val_size=0.5,
    )

    assert set(np.unique(trainer.data["labels"]["test"])) <= set(np.unique(trainer.data["labels"]["train"]))


@pytest.mark.unit
def test_prepare_data_normalizes_mixed_string_labels(tmp_path):
    config = TrainingConfig(
        device="cpu",
        kan=False,
        use_l1=False,
        prune_network=False,
        groupkfold=False,
        log1p=True,
        bs=4,
    )

    trainer = TrainAEThenClassifierHoldout(
        config=config,
        path=str(tmp_path),
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )

    X_train = pd.DataFrame(np.random.randn(8, 4))
    y_train = np.array(["QC", "0.0", "QC", "1.0", "QC", "0.0", "QC", "1.0"])
    groups_train = np.array(["b0", "b0", "b1", "b1", "b0", "b1", "b0", "b1"])

    X_test = pd.DataFrame(np.random.randn(4, 4))
    y_test = np.array(["QC", "0", "QC", "1"])
    groups_test = np.array(["b0", "b1", "b0", "b1"])

    trainer._prepare_data(
        X=X_train,
        y=y_train,
        groups=groups_train,
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        cross_validation=False,
        cross_test=False,
        val_size=0.5,
    )

    assert set(np.unique(trainer.data["labels"]["test"])) <= set(np.unique(trainer.data["labels"]["train"]))


@pytest.mark.unit
def test_prepare_data_numeric_train_string_test_labels(tmp_path):
    """Regression test for leaderboard issue where train is numeric but test is string."""
    config = TrainingConfig(
        device="cpu",
        kan=False,
        use_l1=False,
        prune_network=False,
        groupkfold=False,
        log1p=True,
        bs=4,
    )

    trainer = TrainAEThenClassifierHoldout(
        config=config,
        path=str(tmp_path),
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )

    X_train = pd.DataFrame(np.random.randn(8, 4))
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # numeric
    groups_train = np.array(["b0", "b0", "b1", "b1", "b0", "b1", "b0", "b1"])

    X_test = pd.DataFrame(np.random.randn(4, 4))
    y_test = np.array(["0", "1", "0", "1"])  # string
    groups_test = np.array(["b0", "b1", "b0", "b1"])

    trainer._prepare_data(
        X=X_train,
        y=y_train,
        groups=groups_train,
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        cross_validation=False,
        cross_test=False,
        val_size=0.5,
    )

    assert set(np.unique(trainer.data["labels"]["test"])) <= set(np.unique(trainer.data["labels"]["train"]))




@pytest.mark.unit
def test_holdout_fit_accepts_external_validation_and_unlabeled_test(tmp_path):
    config = TrainingConfig(
        device="cpu",
        kan=False,
        use_l1=False,
        prune_network=False,
        groupkfold=False,
        n_epochs=1,
        warmup=0,
        n_repeats=1,
        bs=4,
        optimize_hyperparams=False,
    )
    trainer = TrainAEClassifierHoldout(
        config=config,
        path=str(tmp_path),
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        log_dvclive=False,
    )
    X = pd.DataFrame(np.arange(24).reshape(12, 2), columns=["f0", "f1"])
    y = pd.Series(["a", "b", "c"] * 4)
    groups = pd.Series(["batch0", "batch1"] * 6)

    with pytest.raises(TypeError, match="requires external X_valid/y_valid"):
        trainer.fit(X, y, groups_train=groups, X_test=X.copy())
    with pytest.raises(TypeError, match="no longer accepts a test matrix"):
        trainer.fit_predict(X, y, X.copy(), groups_train=groups)

    X_valid = pd.DataFrame(np.arange(8).reshape(4, 2), columns=["f0", "f1"])
    y_valid = pd.Series(["a", "b", "c", "a"])
    groups_valid = pd.Series(["valid0", "valid0", "valid1", "valid1"])
    X_test = pd.DataFrame(np.arange(10).reshape(5, 2), columns=["f0", "f1"])
    groups_test = pd.Series(["test0", "test0", "test1", "test1", "test1"])

    trainer._prepare_data(
        X=X,
        y=y,
        groups=groups,
        X_valid=X_valid,
        y_valid=y_valid,
        groups_valid=groups_valid,
        X_test=X_test,
        groups_test=groups_test,
        internal_validation=False,
    )

    assert trainer._no_internal_validation is True
    assert len(trainer.data["inputs"]["train"]) == len(X)
    assert len(trainer.data["inputs"]["valid"]) == len(X_valid)
    assert len(trainer.data["inputs"]["test"]) == len(X_test)
    assert set(trainer.data["labels"]["test"]) == {-1}
    assert set(trainer.data["sets"]["train"]) == {"train"}
    assert set(trainer.data["sets"]["valid"]) == {"valid"}
    assert set(trainer.data["sets"]["test"]) == {"test"}

    trainer._prepare_data(X=X, y=y, groups=groups, internal_validation=False)

    assert trainer._no_internal_validation is True
    assert len(trainer.data["inputs"]["train"]) == len(X)
    assert len(trainer.data["inputs"]["valid"]) == len(X)
    assert len(trainer.data["inputs"]["test"]) == len(X)
    assert set(trainer.data["sets"]["train"]) == {"train"}
    assert set(trainer.data["sets"]["valid"]) == {"valid"}
    assert set(trainer.data["sets"]["test"]) == {"test"}


@pytest.mark.unit
def test_public_split_mcc_scores_prepared_inputs_without_rescaling():
    trainer = TrainAEClassifierHoldout.__new__(TrainAEClassifierHoldout)
    trainer.args = _args(device="cpu", scaler="standard")
    trainer.columns = None
    trainer._label_encoder = None
    trainer.n_cats = 2

    class BadScaler:
        def transform(self, X):
            return np.asarray(X) + 1000

    class TinyAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Identity()
            self.classifier = torch.nn.Identity()

    trainer.scaler = BadScaler()
    trainer.ae = TinyAE()
    trainer.data = {
        "inputs": {"valid": pd.DataFrame({"f0": [0.0, 1.0, 0.0, 1.0]})},
        "labels": {"valid": np.array([0, 1, 0, 1])},
        "batches": {"valid": np.array([0, 0, 1, 1])},
    }

    def fake_logits(self, data, batch_ids=None):
        assert float(data.max()) < 10.0
        preds = (data[:, 0] > 0.5).long()
        return torch.nn.functional.one_hot(preds, num_classes=2).float()

    trainer._predict_logits_from_batch = types.MethodType(fake_logits, trainer)

    assert trainer._score_public_split_mcc("valid") == pytest.approx(1.0)


@pytest.mark.unit
def test_holdout_prepare_data_legacy_internal_validation_still_splits(tmp_path):
    config = TrainingConfig(
        device="cpu",
        kan=False,
        use_l1=False,
        prune_network=False,
        groupkfold=False,
        n_epochs=1,
        warmup=0,
        n_repeats=3,
        bs=4,
        optimize_hyperparams=False,
    )
    trainer = TrainAEClassifierHoldout(
        config=config,
        path=str(tmp_path),
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        log_dvclive=False,
    )
    X = pd.DataFrame(np.arange(90).reshape(30, 3))
    y = pd.Series(["a", "b", "c"] * 10)
    groups = pd.Series(["batch0"] * len(X))

    trainer._prepare_data(X=X, y=y, groups=groups, internal_validation=True)

    assert trainer._no_internal_validation is False
    assert len(trainer.data["inputs"]["train"]) < len(X)
    assert len(trainer.data["inputs"]["valid"]) > 0
    assert len(trainer.data["inputs"]["test"]) > 0
