import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from bernn.config.training_config import TrainingConfig
from bernn.dl.models.pytorch.utils.dataset import get_loaders_no_pool
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
        log_neptune=0,
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
        csv_file="mock.csv",
        dataset="custom",
        device="cpu",
        kan=False,
        use_l1=False,
        prune_network=False,
        pool=False,
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
        log_neptune=True,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )

    assert trainer.config is config
    assert trainer.args is config
    assert trainer.log_neptune is False
    assert trainer.path == str(tmp_path)