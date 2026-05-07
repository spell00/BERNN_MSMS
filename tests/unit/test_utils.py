"""Unit tests for bernn/utils/utils.py and bernn/utils/metrics.py."""
import numpy as np
import pandas as pd
import pytest

from bernn.utils.utils import get_unique_labels, scale_data, to_csv
from bernn.utils.metrics import calculate_aic, calculate_bic


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _make_tabular_data(n=40, n_features=5, n_batches=2, seed=0):
    """Return a minimal data dict with inputs/meta/batches/labels/names/cats."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    all_df = pd.DataFrame(rng.standard_normal((n, n_features)), columns=cols)
    train_df = all_df.iloc[: n // 2].copy()
    valid_df = all_df.iloc[n // 2 : n * 3 // 4].copy()
    test_df = all_df.iloc[n * 3 // 4 :].copy()

    meta_cols = ["m0", "m1"]
    all_meta = pd.DataFrame(rng.standard_normal((n, 2)), columns=meta_cols)
    train_meta = all_meta.iloc[: n // 2].copy()
    valid_meta = all_meta.iloc[n // 2 : n * 3 // 4].copy()
    test_meta = all_meta.iloc[n * 3 // 4 :].copy()

    batches_all = np.array([f"b{i % n_batches}" for i in range(n)])

    data = {
        "inputs": {"all": all_df, "train": train_df, "valid": valid_df, "test": test_df},
        "meta": {"all": all_meta, "train": train_meta, "valid": valid_meta, "test": test_meta},
        "batches": {
            "all": batches_all,
            "train": batches_all[: n // 2],
            "valid": batches_all[n // 2 : n * 3 // 4],
            "test": batches_all[n * 3 // 4 :],
        },
        "labels": {
            "all": np.array([f"l{i % 2}" for i in range(n)]),
            "train": np.array([f"l{i % 2}" for i in range(n // 2)]),
            "valid": np.array([f"l{i % 2}" for i in range(n // 4)]),
            "test": np.array([f"l{i % 2}" for i in range(n // 4)]),
        },
    }
    return data


# ────────────────────────────────────────────────────────────
# get_unique_labels
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_get_unique_labels_preserves_order():
    labels = ["b", "a", "b", "c", "a"]
    result = get_unique_labels(labels)
    assert list(result) == ["b", "a", "c"]


@pytest.mark.unit
def test_get_unique_labels_single():
    result = get_unique_labels(["x"])
    assert list(result) == ["x"]


@pytest.mark.unit
def test_get_unique_labels_all_same():
    result = get_unique_labels(["z", "z", "z"])
    assert list(result) == ["z"]


@pytest.mark.unit
def test_get_unique_labels_returns_ndarray():
    result = get_unique_labels(["a", "b"])
    assert isinstance(result, np.ndarray)


# ────────────────────────────────────────────────────────────
# scale_data — simple scalers (no per-batch)
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("scale", ["standard", "minmax", "robust", "l1", "l2"])
def test_scale_data_returns_df(scale):
    data = _make_tabular_data()
    result, scaler = scale_data(scale, data)
    assert isinstance(result["inputs"]["all"], pd.DataFrame)
    assert result["inputs"]["all"].shape == (40, 5)


@pytest.mark.unit
def test_scale_data_none_passthrough():
    data = _make_tabular_data()
    original_vals = data["inputs"]["all"].values.copy()
    result, scaler = scale_data("none", data)
    # Values unchanged
    np.testing.assert_array_equal(result["inputs"]["all"].values, original_vals)
    assert scaler == "none"


@pytest.mark.unit
def test_scale_data_standard_zero_mean():
    """After standard scaling the all-split should be ~zero mean."""
    data = _make_tabular_data(n=200, n_features=10, seed=42)
    result, _ = scale_data("standard", data)
    means = result["inputs"]["all"].mean(axis=0).values
    np.testing.assert_allclose(means, 0.0, atol=1e-6)


@pytest.mark.unit
def test_scale_data_minmax_range():
    """After minmax scaling values in [0,1] for the all-split."""
    data = _make_tabular_data(n=100, n_features=8, seed=7)
    result, _ = scale_data("minmax", data)
    vals = result["inputs"]["all"].values
    assert vals.min() >= -1e-9
    assert vals.max() <= 1.0 + 1e-9


@pytest.mark.unit
def test_scale_data_binarize():
    """After binarize, inputs contain only 0/1."""
    data = _make_tabular_data()
    result, _ = scale_data("binarize", data)
    vals = result["inputs"]["all"].values
    assert set(np.unique(vals)).issubset({0, 1})


@pytest.mark.unit
@pytest.mark.parametrize("scale", ["robust_minmax", "standard_minmax", "l1_minmax", "l2_minmax"])
def test_scale_data_pipeline_variants(scale):
    data = _make_tabular_data()
    result, scaler = scale_data(scale, data)
    assert isinstance(result["inputs"]["all"], pd.DataFrame)


# ────────────────────────────────────────────────────────────
# calculate_aic / calculate_bic
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_calculate_aic_formula():
    from math import log
    n, mse, k = 100, 0.5, 3
    expected = n * log(mse) + 2 * k
    assert calculate_aic(n, mse, k) == pytest.approx(expected)


@pytest.mark.unit
def test_calculate_bic_formula():
    from math import log
    n, mse, k = 100, 0.5, 3
    expected = n * log(mse) + k * log(n)
    assert calculate_bic(n, mse, k) == pytest.approx(expected)


@pytest.mark.unit
def test_aic_increases_with_params():
    aic1 = calculate_aic(100, 0.5, 2)
    aic2 = calculate_aic(100, 0.5, 5)
    assert aic2 > aic1


# ────────────────────────────────────────────────────────────
# to_csv — round-trip produces files
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_to_csv_creates_files(tmp_path):
    n = 10
    n_enc = 8
    feat_cols = pd.Index([f"f{i}" for i in range(n_enc)])
    rng = np.random.default_rng(0)
    lists = {
        "train": {
            "encoded_values": [rng.standard_normal((n, n_enc))],
            "labels": [np.array([f"l{i%2}" for i in range(n)])],
            "domains": [np.array([f"b{i%2}" for i in range(n)])],
            "names": [np.array([f"s{i}" for i in range(n)])],
            "classes": [np.arange(n) % 2],
            "rec_values": [rng.standard_normal((n, n_enc))],
        },
        "valid": {
            "encoded_values": [],
            "labels": [],
            "domains": [],
            "names": [],
            "classes": [],
            "rec_values": [],
        },
    }
    columns = feat_cols
    to_csv(lists, str(tmp_path), columns)
    # At least the encoded CSV for train should be written
    enc_files = list(tmp_path.glob("*enc*"))
    assert len(enc_files) >= 1
