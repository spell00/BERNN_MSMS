"""
tests/unit/test_ae_head_classifier.py
=======================================
Unit tests for head_classifier.py — covers all supported head types
(including XGBoost when installed) with synthetic embeddings.
"""
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from bernn.dl.train.head_classifier import (
    HEAD_TYPES,
    HEAD_TYPES_NO_XGB,
    _HAS_XGB,
    cv_score_head,
    fit_and_score_head,
    sweep_all_heads,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """60 samples, 16-dim embeddings, 3 balanced classes (string labels)."""
    rng = np.random.RandomState(0)
    # Make mildly separable blobs so MCC is non-trivially zero
    means = {"A": [1, 0] * 8, "B": [0, 1] * 8, "C": [-1, -1] * 8}
    Xs, ys = [], []
    for label, mu in means.items():
        X_class = rng.randn(20, 16) + np.array(mu)
        Xs.append(X_class)
        ys.extend([label] * 20)
    return np.vstack(Xs), np.array(ys)


@pytest.fixture
def synthetic_data_encoded(synthetic_data):
    """Same data but with integer-encoded labels (0/1/2)."""
    X, y = synthetic_data
    le = LabelEncoder()
    return X, le.fit_transform(y)


# ---------------------------------------------------------------------------
# fit_and_score_head — string labels
# ---------------------------------------------------------------------------

SKLEARN_HEADS = [h for h in HEAD_TYPES if h != "xgboost"]


@pytest.mark.parametrize("head_type", SKLEARN_HEADS)
def test_fit_and_score_string_labels(synthetic_data, head_type):
    """All sklearn/prototype heads accept string labels without error."""
    X, y = synthetic_data
    split = 45
    head, tr_mcc, vl_mcc = fit_and_score_head(
        X[:split], y[:split], X[split:], y[split:], head_type, {}
    )
    assert head is not None
    assert -1.0 <= tr_mcc <= 1.0, f"train_mcc out of range for {head_type}: {tr_mcc}"
    assert -1.0 <= vl_mcc <= 1.0, f"valid_mcc out of range for {head_type}: {vl_mcc}"


@pytest.mark.skipif(not _HAS_XGB, reason="xgboost not installed")
def test_fit_and_score_xgboost_string_labels(synthetic_data):
    """XGBoost head accepts string labels (label-encoded internally)."""
    X, y = synthetic_data
    split = 45
    head, tr_mcc, vl_mcc = fit_and_score_head(
        X[:split], y[:split], X[split:], y[split:], "xgboost", {}
    )
    assert head is not None
    assert -1.0 <= tr_mcc <= 1.0, f"train_mcc out of range: {tr_mcc}"
    assert -1.0 <= vl_mcc <= 1.0, f"valid_mcc out of range: {vl_mcc}"


@pytest.mark.skipif(not _HAS_XGB, reason="xgboost not installed")
def test_fit_and_score_xgboost_int_labels(synthetic_data_encoded):
    """XGBoost head also accepts integer labels directly."""
    X, y = synthetic_data_encoded
    split = 45
    head, tr_mcc, vl_mcc = fit_and_score_head(
        X[:split], y[:split], X[split:], y[split:], "xgboost", {}
    )
    assert -1.0 <= vl_mcc <= 1.0


# ---------------------------------------------------------------------------
# cv_score_head
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head_type", SKLEARN_HEADS)
def test_cv_score_returns_valid_dict(synthetic_data, head_type):
    """cv_score_head returns a dict with required keys and valid float values."""
    X, y = synthetic_data
    res = cv_score_head(X, y, head_type=head_type, head_params={}, n_splits=3)
    for key in ("mean_valid_mcc", "std_valid_mcc", "mean_train_mcc", "fold_valid_mccs"):
        assert key in res, f"Missing key '{key}' for head={head_type}"
    assert -1.0 <= res["mean_valid_mcc"] <= 1.0 or np.isnan(res["mean_valid_mcc"])
    assert len(res["fold_valid_mccs"]) == 3


@pytest.mark.skipif(not _HAS_XGB, reason="xgboost not installed")
def test_cv_score_xgboost_string_labels(synthetic_data):
    """cv_score_head works for xgboost with string labels."""
    X, y = synthetic_data
    res = cv_score_head(X, y, head_type="xgboost", head_params={}, n_splits=3)
    assert "mean_valid_mcc" in res
    assert not np.isnan(res["mean_valid_mcc"]), "XGBoost cv_score returned NaN"
    assert len(res["fold_valid_mccs"]) == 3


def test_cv_score_insufficient_samples():
    """cv_score_head with only 2 samples per class (n_splits=3) returns nan gracefully."""
    rng = np.random.RandomState(1)
    X = rng.randn(6, 8)
    y = np.array(["A", "A", "B", "B", "C", "C"])
    res = cv_score_head(X, y, head_type="random_forest", head_params={}, n_splits=3)
    # Should not raise; returns nan when StratifiedKFold can't split
    assert "mean_valid_mcc" in res
    assert np.isnan(res["mean_valid_mcc"]), "Expected nan for too-few-samples edge case"


# ---------------------------------------------------------------------------
# sweep_all_heads
# ---------------------------------------------------------------------------

def test_sweep_all_heads_returns_sorted(synthetic_data):
    """sweep_all_heads returns a dict sorted descending by mean_valid_mcc."""
    X, y = synthetic_data
    results = sweep_all_heads(X, y, n_splits=3)
    assert isinstance(results, dict)
    mccs = [v.get("mean_valid_mcc", float("-inf")) for v in results.values()]
    assert mccs == sorted(mccs, reverse=True), "Results not sorted descending"


def test_sweep_all_heads_covers_all_types(synthetic_data):
    """sweep_all_heads runs every head type (excluding xgboost if not installed)."""
    X, y = synthetic_data
    results = sweep_all_heads(X, y, n_splits=3)
    expected = set(HEAD_TYPES if _HAS_XGB else HEAD_TYPES_NO_XGB)
    assert set(results.keys()) == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_binary_classification():
    """All heads work with binary (2-class) string labels."""
    rng = np.random.RandomState(7)
    X = rng.randn(40, 8) + np.array([rng.choice([-2, 2])] * 8)
    y = np.array(["pos"] * 20 + ["neg"] * 20)
    for ht in ["random_forest", "logistic_regression", "knn", "prototype_mean"]:
        res = cv_score_head(X, y, head_type=ht, head_params={}, n_splits=3)
        assert not np.isnan(res["mean_valid_mcc"]), f"{ht} returned NaN for binary"


def test_predict_head_returns_correct_classes(synthetic_data):
    """Fitted head.predict returns only labels seen in training."""
    X, y = synthetic_data
    from bernn.dl.train.head_classifier import _make_head
    from sklearn.ensemble import RandomForestClassifier
    head, _, _ = fit_and_score_head(X[:45], y[:45], X[45:], y[45:], "random_forest", {})
    preds = head.predict(X[45:])
    assert set(preds).issubset(set(y)), "Predictions contain unseen classes"
