#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter search spaces for sklearn / XGBoost classifiers used in the
BERNN head sweep (train_ae_head_sweep.py) and the legacy GP sweep
(sklearn_train3_gp2.py).

All classifiers are now active (none commented out).  XGBoost is included with
a richer parameter space that mirrors what Optuna can suggest.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from skopt.space import Real, Integer, Categorical

# ---------------------------------------------------------------------------
# Individual search spaces (skopt)
# ---------------------------------------------------------------------------

sgd_space = [
    Integer(1, 20000, "uniform", name="features_cutoff"),
    Integer(1, 1000,  "uniform", name="max_iter"),
    Real(1e-4, 1,     "uniform", name="alpha"),
    Categorical(["log", "modified_huber"],  name="loss"),
    Categorical(["l2", "l1", "elasticnet"], name="penalty"),
    Categorical(["balanced"],               name="class_weight"),
    Categorical([True, False],              name="fit_intercept"),
]

xgb_space = [
    Integer(50, 500,   "uniform",     name="n_estimators"),
    Integer(3, 10,     "uniform",     name="max_depth"),
    Real(1e-3, 0.5,    "log-uniform", name="learning_rate"),
    Real(1e-8, 10.0,   "log-uniform", name="reg_alpha"),
    Real(1e-8, 10.0,   "log-uniform", name="reg_lambda"),
    Real(0.5, 1.0,     "uniform",     name="subsample"),
    Real(0.5, 1.0,     "uniform",     name="colsample_bytree"),
    Categorical(["hist", "gpu_hist"], name="tree_method"),
]

rfc_space = [
    Integer(1, 20000, "uniform", name="features_cutoff"),
    Integer(1, 100,   "uniform", name="max_features"),
    Integer(2, 10,    "uniform", name="min_samples_split"),
    Integer(1, 10,    "uniform", name="min_samples_leaf"),
    Integer(1, 1000,  "uniform", name="n_estimators"),
    Categorical(["gini", "entropy"], name="criterion"),
    Categorical([True, False],       name="oob_score"),
    Categorical(["balanced"],        name="class_weight"),
]

kn_space = [
    Integer(1, 20000, "uniform", name="features_cutoff"),
    Integer(1, 31,    "uniform", name="n_neighbors"),
    Categorical(["euclidean", "manhattan", "cosine"], name="metric"),
    Categorical(["uniform", "distance"],              name="weights"),
]

logreg_space = [
    Integer(1, 20000,    "uniform", name="features_cutoff"),
    Integer(100, 20000,  "uniform", name="max_iter"),
    Real(1e-3, 1e4,      "uniform", name="C"),
    Categorical(["saga"],           name="solver"),
    Categorical(["l1", "l2"],       name="penalty"),
    Categorical([True, False],      name="fit_intercept"),
    Categorical(["balanced"],       name="class_weight"),
]

linsvc_space = [
    Integer(1, 20000, "uniform",     name="features_cutoff"),
    Real(1e-4, 1,     "log-uniform", name="tol"),
    Integer(1, 1000,  "uniform",     name="max_iter"),
    Categorical(["l2"],              name="penalty"),
    Real(1e-3, 1e4,   "uniform",     name="C"),
    Categorical(["balanced"],        name="class_weight"),
]

svc_space = [
    Integer(1, 20000, "uniform", name="features_cutoff"),
    Integer(1, 1000,  "uniform", name="max_iter"),
    Real(1e-3, 1e3,   "uniform", name="C"),
    Categorical(["balanced"],    name="class_weight"),
    Categorical(["linear"],      name="kernel"),
]

svc_rbf_space = [
    Integer(1, 20000, "uniform", name="features_cutoff"),
    Integer(1, 1000,  "uniform", name="max_iter"),
    Real(1e-3, 1e3,   "uniform", name="C"),
    Categorical(["balanced"],    name="class_weight"),
    Categorical(["rbf"],         name="kernel"),
]

gbc_space = [
    Integer(50, 500,  "uniform",     name="n_estimators"),
    Integer(3, 10,    "uniform",     name="max_depth"),
    Real(1e-3, 0.5,   "log-uniform", name="learning_rate"),
    Real(0.5, 1.0,    "uniform",     name="subsample"),
]

# ---------------------------------------------------------------------------
# Active model registry (all controls enabled, XGBoost included)
# ---------------------------------------------------------------------------

# Import XGBoost lazily so the module still loads without it installed.
try:
    import xgboost as xgb
    _XGB_CLS = xgb.XGBClassifier
except ImportError:
    _XGB_CLS = None

models = {
    "XGBoostClassifier":      [_XGB_CLS, xgb_space],
    "RandomForestClassifier": [RandomForestClassifier, rfc_space],
    "LinearSVC":              [LinearSVC, linsvc_space],
    "LogisticRegression":     [LogisticRegression, logreg_space],
    "KNeighbors":             [KNeighborsClassifier, kn_space],
    "SVCLinear":              [SVC, svc_space],
    "SVCRbf":                 [SVC, svc_rbf_space],
    "GradientBoosting":       [GradientBoostingClassifier, gbc_space],
    "SGDClassifier":          [SGDClassifier, sgd_space],
}

# Remove XGBoost entry if the library isn't available.
if _XGB_CLS is None:
    models.pop("XGBoostClassifier", None)
