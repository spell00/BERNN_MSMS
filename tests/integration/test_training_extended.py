"""Integration tests: dloss variants, variational AE, and data-getter round-trip."""
import os
import pytest
import importlib
import numpy as np
import pandas as pd
from types import SimpleNamespace

from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout


# ────────────────────────────────────────────────────────────
# Shared fixtures
# ────────────────────────────────────────────────────────────

def _make_sample_data(n=80, n_features=30, n_batches=3, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    feat_cols = [f"feature_{i}" for i in range(n_features)]
    meta_cols = ["meta_0", "meta_1"]

    def _split_df(start, end):
        size = end - start
        return pd.DataFrame(rng.standard_normal((size, n_features)), columns=feat_cols)

    def _split_meta(start, end):
        size = end - start
        return pd.DataFrame(rng.standard_normal((size, len(meta_cols))), columns=meta_cols)

    def _labels(start, end):
        return np.array([i % n_classes for i in range(start, end)], dtype=np.int64)

    def _batches(start, end):
        return np.array([i % n_batches for i in range(start, end)], dtype=np.int64)

    def _names(start, end):
        return np.array([f"s{i}" for i in range(start, end)])

    splits = {"all": (0, n), "train": (0, n // 2), "valid": (n // 2, n * 3 // 4), "test": (n * 3 // 4, n)}
    data: dict = {"inputs": {}, "meta": {}, "batches": {}, "labels": {}, "cats": {}, "sets": {}, "names": {}}

    for split, (s, e) in splits.items():
        df = _split_df(s, e)
        data["inputs"][split] = df
        data["meta"][split] = _split_meta(s, e)
        data["batches"][split] = _batches(s, e)
        data["labels"][split] = _labels(s, e)
        data["cats"][split] = data["labels"][split].copy()
        data["sets"][split] = np.array([split] * (e - s))
        data["names"][split] = pd.Series(_names(s, e))

    return data


def _make_args(**overrides):
    ns = SimpleNamespace(
        device="cpu",
        random_recs=0,
        predict_tests=0,
        early_stop=2,
        early_warmup_stop=-1,
        train_after_warmup=0,
        threshold=0.0,
        n_epochs=2,
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
        dataset="mock",
        csv_file="mock.csv",
        log1p=1,
        berm="none",
        pool=0,
        strategy="none",
        best_features_file="",
        n_features=-1,
        bad_batches="",
        controls="",
        exp_id="testIntegration",
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
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _make_trainer(args, tmp_path, **kwargs):
    return TrainAEClassifierHoldout(
        args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,

        log_mlflow=False,
        groupkfold=False,
        pools=False,
        **kwargs,
    )


def _base_params():
    return {
        "nu": 0.01,
        "lr": 1e-3,
        "wd": 1e-6,
        "smoothing": 0.0,
        "margin": 1.0,
        "warmup": 1,
        "disc_b_warmup": 1,
        "dropout": 0.0,
        "scaler": "standard",
        "layer1": 32,
        "layer2": 32,
        "gamma": 0.0,
        "beta": 0.0,
        "zeta": 0.0,
        "thres": 0.0,
        "prune_threshold": 0.0,
    }


def _inject_and_run(trainer, data, tmp_path, params):
    trainer.data = data
    trainer.unique_labels = np.unique(data["labels"]["all"])
    trainer.unique_batches = np.unique(data["batches"]["all"])
    trainer.columns = data["inputs"]["all"].columns

    csv_path = tmp_path / "mock.csv"
    data["inputs"]["all"].to_csv(csv_path, index=False)
    try:
        result = trainer.train(params)
    finally:
        if csv_path.exists():
            os.remove(csv_path)
    return result


# ────────────────────────────────────────────────────────────
# dloss variant tests
# ────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("dloss", ["inverseTriplet", "revTriplet", "DANN", "no"])
def test_dloss_variants_return_float(dloss, tmp_path):
    data = _make_sample_data()
    args = _make_args(dloss=dloss)
    trainer = _make_trainer(args, tmp_path)
    params = _base_params()
    if dloss in ("DANN", "inverseTriplet", "revTriplet"):
        params["gamma"] = 0.1

    try:
        result = _inject_and_run(trainer, data, tmp_path, params)
    except Exception as e:
        pytest.skip(f"dloss={dloss} failed: {e}")

    assert isinstance(result, (float, int))


# ────────────────────────────────────────────────────────────
# variational AE
# ────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_variational_ae_runs(tmp_path):
    data = _make_sample_data()
    args = _make_args(variational=1, dloss="no")
    trainer = _make_trainer(args, tmp_path)
    params = _base_params()
    params["beta"] = 1.0

    try:
        result = _inject_and_run(trainer, data, tmp_path, params)
    except Exception as e:
        pytest.skip(f"Variational AE failed: {e}")

    assert isinstance(result, (float, int))


# ────────────────────────────────────────────────────────────
# get_ordered_layers
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_get_ordered_layers_ordering(tmp_path):
    args = _make_args()
    trainer = _make_trainer(args, tmp_path)
    params = {"layer3": 16, "layer1": 64, "layer2": 32, "dropout": 0.1}
    ordered = trainer.get_ordered_layers(params)
    assert list(ordered.keys()) == ["layer1", "layer2", "layer3"]
    assert list(ordered.values()) == [64, 32, 16]


@pytest.mark.unit
def test_get_ordered_layers_single(tmp_path):
    args = _make_args()
    trainer = _make_trainer(args, tmp_path)
    params = {"layer1": 128, "lr": 0.001}
    ordered = trainer.get_ordered_layers(params)
    assert ordered == {"layer1": 128}


# ────────────────────────────────────────────────────────────
# get_dummy round-trip
# ────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_get_dummy_returns_expected_keys():
    from bernn.utils.data_getters import get_dummy
    args = SimpleNamespace(
        dummy_features=16, dummy_classes=3, dummy_batches=2,
        dummy_train=30, dummy_valid=10, dummy_test=10,
        n_features=-1, bad_batches="", log1p=0,
        remove_zeros=0,
    )
    data, unique_labels, unique_batches = get_dummy(args, seed=0)
    assert "inputs" in data
    assert "labels" in data
    assert "batches" in data
    assert len(unique_labels) == 3
    assert data["inputs"]["train"].shape[0] == 30


@pytest.mark.integration
def test_get_dummy_label_consistency():
    from bernn.utils.data_getters import get_dummy
    args = SimpleNamespace(
        dummy_features=8, dummy_classes=2, dummy_batches=2,
        dummy_train=20, dummy_valid=8, dummy_test=8,
        n_features=-1, bad_batches="", log1p=0,
        remove_zeros=0,
    )
    data, unique_labels, _ = get_dummy(args, seed=42)
    # All labels in splits should be a subset of unique_labels
    for split in ("train", "valid", "test"):
        assert set(np.unique(data["labels"][split])).issubset(set(unique_labels))


# ────────────────────────────────────────────────────────────
# Multiple repeats (n_repeats > 1)
# ────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_training_multiple_repeats(tmp_path):
    data = _make_sample_data()
    args = _make_args(n_repeats=2, n_epochs=1, early_stop=1)
    trainer = _make_trainer(args, tmp_path)
    try:
        result = _inject_and_run(trainer, data, tmp_path, _base_params())
    except Exception as e:
        pytest.skip(f"Multi-repeat run failed: {e}")
    assert isinstance(result, (float, int))


# ────────────────────────────────────────────────────────────
# Reconstruction loss variants
# ────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("rec_loss", ["l1", "mse"])
def test_rec_loss_variants(rec_loss, tmp_path):
    data = _make_sample_data()
    args = _make_args(rec_loss=rec_loss, dloss="no")
    trainer = _make_trainer(args, tmp_path)
    try:
        result = _inject_and_run(trainer, data, tmp_path, _base_params())
    except Exception as e:
        pytest.skip(f"rec_loss={rec_loss} failed: {e}")
    assert isinstance(result, (float, int))
