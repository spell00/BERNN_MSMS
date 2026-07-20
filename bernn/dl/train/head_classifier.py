#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
head_classifier.py
==================
Classification heads that are fully trained on frozen AE encoder embeddings.

Supported head types
--------------------
  xgboost            XGBoost gradient-boosted trees
  random_forest      sklearn RandomForestClassifier
  linear_svc         sklearn LinearSVC (fast linear SVM)
  svc_rbf            sklearn SVC with RBF kernel
  logistic_regression  sklearn LogisticRegression (saga solver, L1/L2)
  knn                sklearn KNeighborsClassifier
  gradient_boosting  sklearn GradientBoostingClassifier
  prototype_mean     Nearest class-mean prototype (cosine or L2)
  prototype_kmeans   K-means prototypes per class (like OtiteNet)

Evaluation
----------
All heads are evaluated with StratifiedKFold(n_splits=3) on the provided
training embeddings, and the mean MCC across folds is returned.  This is the
same cv=3 criterion used by the Optuna sweep in train_ae_head_sweep.py.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: All head types that this module supports.
HEAD_TYPES = [
    "xgboost",
    "random_forest",
    "linear_svc",
    "svc_rbf",
    "logistic_regression",
    "knn",
    "gradient_boosting",
    "prototype_mean",
    "prototype_kmeans",
]

#: Head types that do NOT require the optional xgboost package.
HEAD_TYPES_NO_XGB = [h for h in HEAD_TYPES if h != "xgboost"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_head(head_type: str, head_params: Dict[str, Any]):
    """Instantiate the requested classifier (not yet fitted)."""
    if head_type == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return xgb.XGBClassifier(
            n_estimators=int(head_params.get("n_estimators", 200)),
            max_depth=int(head_params.get("max_depth", 6)),
            learning_rate=float(head_params.get("learning_rate", 0.1)),
            reg_alpha=float(head_params.get("reg_alpha", 1e-3)),
            reg_lambda=float(head_params.get("reg_lambda", 1.0)),
            subsample=float(head_params.get("subsample", 0.8)),
            colsample_bytree=float(head_params.get("colsample_bytree", 0.8)),
            tree_method=head_params.get("tree_method", "hist"),
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=-1,
        )

    if head_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(head_params.get("n_estimators", 200)),
            max_features=head_params.get("max_features", "sqrt"),
            min_samples_split=int(head_params.get("min_samples_split", 2)),
            min_samples_leaf=int(head_params.get("min_samples_leaf", 1)),
            criterion=head_params.get("criterion", "gini"),
            oob_score=bool(head_params.get("oob_score", False)),
            class_weight="balanced",
            n_jobs=-1,
        )

    if head_type == "linear_svc":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(
            C=float(head_params.get("C", 1.0)),
            tol=float(head_params.get("tol", 1e-4)),
            max_iter=int(head_params.get("max_iter", 1000)),
            class_weight="balanced",
        )
        # Wrap in calibration so predict_proba is available if needed
        return CalibratedClassifierCV(base, cv=2)

    if head_type == "svc_rbf":
        from sklearn.svm import SVC
        return SVC(
            C=float(head_params.get("C", 1.0)),
            kernel="rbf",
            gamma=head_params.get("gamma", "scale"),
            probability=True,
            class_weight="balanced",
            max_iter=int(head_params.get("max_iter", 1000)),
        )

    if head_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=float(head_params.get("C", 1.0)),
            penalty=head_params.get("penalty", "l2"),
            solver="saga",
            max_iter=int(head_params.get("max_iter", 1000)),
            class_weight="balanced",
            n_jobs=-1,
        )

    if head_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=int(head_params.get("n_neighbors", 5)),
            metric=head_params.get("metric", "euclidean"),
            weights=head_params.get("weights", "distance"),
            n_jobs=-1,
        )

    if head_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=int(head_params.get("n_estimators", 100)),
            max_depth=int(head_params.get("max_depth", 3)),
            learning_rate=float(head_params.get("learning_rate", 0.1)),
            subsample=float(head_params.get("subsample", 1.0)),
        )

    raise ValueError(f"Unknown head_type: {head_type!r}. Choose from {HEAD_TYPES}")


# ---------------------------------------------------------------------------
# Prototype heads
# ---------------------------------------------------------------------------

class PrototypeMeanClassifier:
    """Nearest class-mean classifier (L2 or cosine distance)."""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.prototypes_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrototypeMeanClassifier":
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes_ = le.classes_
        self.prototypes_ = np.array(
            [X[y_enc == c].mean(axis=0) for c in range(len(self.classes_))]
        )
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            P_norm = self.prototypes_ / (
                np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-12
            )
            # cosine *similarity* → negate for distance
            return 1.0 - X_norm @ P_norm.T
        # L2
        return np.sum(
            (X[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = self._distances(X).argmin(axis=1)
        return self.classes_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class PrototypeKMeansClassifier:
    """K-means per-class prototypes (like OtiteNet prototype_kmeans)."""

    def __init__(self, n_components: int = 1, metric: str = "cosine"):
        self.n_components = n_components
        self.metric = metric
        self.prototypes_: Optional[np.ndarray] = None
        self.proto_labels_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrototypeKMeansClassifier":
        from sklearn.cluster import KMeans

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes_ = le.classes_
        protos, labels = [], []
        for cls_idx, cls_label in enumerate(self.classes_):
            X_cls = X[y_enc == cls_idx]
            k = min(self.n_components, len(X_cls))
            if k <= 1 or len(X_cls) < 2:
                protos.append(X_cls.mean(axis=0))
                labels.append(cls_label)
            else:
                km = KMeans(n_clusters=k, n_init=5, random_state=42)
                km.fit(X_cls)
                for c in km.cluster_centers_:
                    protos.append(c)
                    labels.append(cls_label)
        self.prototypes_ = np.array(protos)
        self.proto_labels_ = np.array(labels)
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            P_norm = self.prototypes_ / (
                np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-12
            )
            return 1.0 - X_norm @ P_norm.T
        return np.sum(
            (X[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = self._distances(X).argmin(axis=1)
        return self.proto_labels_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def _make_prototype_head(head_type: str, head_params: Dict[str, Any]):
    metric = head_params.get("proto_metric", "cosine")
    if head_type == "prototype_mean":
        return PrototypeMeanClassifier(metric=metric)
    if head_type == "prototype_kmeans":
        return PrototypeKMeansClassifier(
            n_components=int(head_params.get("proto_components", 1)),
            metric=metric,
        )
    raise ValueError(f"Unknown prototype head type: {head_type!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_and_score_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    head_type: str,
    head_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, float, float]:
    """
    Fit a classification head on (X_train, y_train) and score on (X_valid, y_valid).

    Returns
    -------
    head : fitted classifier
    train_mcc : float
    valid_mcc : float
    """
    head_params = head_params or {}
    is_proto = head_type.startswith("prototype_")

    if is_proto:
        head = _make_prototype_head(head_type, head_params)
    else:
        head = _make_head(head_type, head_params)

    # XGBoost requires integer-encoded labels; encode internally and decode preds.
    le: Optional[LabelEncoder] = None
    if head_type == "xgboost":
        le = LabelEncoder()
        y_train_fit = le.fit_transform(y_train)
        y_valid_eval = le.transform(y_valid)
    else:
        y_train_fit = y_train
        y_valid_eval = y_valid

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        head.fit(X_train, y_train_fit)
        train_preds_raw = head.predict(X_train)
        valid_preds_raw = head.predict(X_valid)

    # Decode XGBoost integer predictions back to original label space for MCC.
    if le is not None:
        train_preds = le.inverse_transform(train_preds_raw.astype(int))
        valid_preds = le.inverse_transform(valid_preds_raw.astype(int))
    else:
        train_preds = train_preds_raw
        valid_preds = valid_preds_raw

    train_mcc = float(matthews_corrcoef(y_train, train_preds))
    valid_mcc = float(matthews_corrcoef(y_valid, valid_preds))
    return head, train_mcc, valid_mcc


def cv_score_head(
    X: np.ndarray,
    y: np.ndarray,
    head_type: str,
    head_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate a head type via StratifiedKFold cross-validation.

    Parameters
    ----------
    X          : embedding matrix (n_samples, n_features)
    y          : label array (n_samples,)
    head_type  : one of HEAD_TYPES
    head_params: dict of hyperparameters for the head
    n_splits   : number of CV folds (default 3, matching Optuna sweep criterion)
    random_state : random seed

    Returns
    -------
    dict with keys: mean_valid_mcc, std_valid_mcc, mean_train_mcc, fold_valid_mccs
    """
    head_params = head_params or {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Guard: not enough samples per class for the requested n_splits.
    try:
        splits = list(skf.split(X, y))
    except ValueError:
        return {
            "mean_valid_mcc": float("nan"),
            "std_valid_mcc": float("nan"),
            "mean_train_mcc": float("nan"),
            "fold_valid_mccs": [],
        }

    valid_mccs, train_mccs = [], []
    for fold_train_idx, fold_valid_idx in splits:
        X_tr, y_tr = X[fold_train_idx], y[fold_train_idx]
        X_vl, y_vl = X[fold_valid_idx], y[fold_valid_idx]

        # Skip folds where a class is missing in the training split
        if len(np.unique(y_tr)) < 2:
            continue

        _, tr_mcc, vl_mcc = fit_and_score_head(
            X_tr, y_tr, X_vl, y_vl, head_type, head_params
        )
        valid_mccs.append(vl_mcc)
        train_mccs.append(tr_mcc)

    if not valid_mccs:
        return {
            "mean_valid_mcc": float("nan"),
            "std_valid_mcc": float("nan"),
            "mean_train_mcc": float("nan"),
            "fold_valid_mccs": [],
        }

    return {
        "mean_valid_mcc": float(np.mean(valid_mccs)),
        "std_valid_mcc": float(np.std(valid_mccs)),
        "mean_train_mcc": float(np.mean(train_mccs)),
        "fold_valid_mccs": valid_mccs,
    }


def sweep_all_heads(
    X: np.ndarray,
    y: np.ndarray,
    head_params_by_type: Optional[Dict[str, Dict[str, Any]]] = None,
    n_splits: int = 3,
    include_xgboost: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Score every head type (cv=3) and return a summary dict.

    Useful for quick baseline comparisons on frozen AE embeddings.
    """
    head_params_by_type = head_params_by_type or {}
    head_types = HEAD_TYPES if include_xgboost else HEAD_TYPES_NO_XGB

    results = {}
    for ht in head_types:
        params = head_params_by_type.get(ht, {})
        try:
            res = cv_score_head(X, y, ht, params, n_splits=n_splits)
            results[ht] = res
        except Exception as exc:
            results[ht] = {"error": str(exc), "mean_valid_mcc": float("nan")}

    # Sort by mean_valid_mcc descending
    results = dict(
        sorted(results.items(), key=lambda kv: kv[1].get("mean_valid_mcc", float("-inf")), reverse=True)
    )
    return results
