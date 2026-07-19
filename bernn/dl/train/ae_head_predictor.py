#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ae_head_predictor.py
====================
Drop-in leaderboard-compatible wrapper that:
  1. Trains a BERNN AE encoder (using TrainAEClassifierHoldout).
  2. Freezes the encoder and extracts train / test embeddings.
  3. Sweeps all classification heads (XGBoost, RF, SVC, KNN, LogReg,
     Prototype-mean, Prototype-kmeans …) via StratifiedKFold cv=3.
  4. Fits the best head on the full training embeddings.
  5. Exposes .predict(X_test), .best_mcc, .cv_mcc_mean — exactly what
     the BE_leaderboard scoring harness expects.

Usage (inside a leaderboard fit_predict)
-----------------------------------------
    from bernn.dl.train.ae_head_predictor import AEHeadPredictor

    predictor = AEHeadPredictor(config=my_training_config)
    predictor.fit(X_train, y_train, X_test,
                  groups_train=batches_train, groups_test=batches_test)
    preds = predictor.predict(X_test)   # string labels

This is NOT the full Optuna sweep (train_ae_head_sweep.py).  It trains one AE
configuration, sweeps only the head types (fast, sklearn-level), and returns.
For a full joint hyperparameter + head sweep see train_ae_head_sweep.py.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from sklearn.preprocessing import LabelEncoder

from bernn.dl.train.head_classifier import (
    HEAD_TYPES,
    HEAD_TYPES_NO_XGB,
    cv_score_head,
    fit_and_score_head,
    sweep_all_heads,
)

try:
    import xgboost  # noqa: F401
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# Embedding extraction helper
# ---------------------------------------------------------------------------

def _extract_embeddings_from_model(ae, X: pd.DataFrame, device: str) -> np.ndarray:
    """Feed X through the frozen AE encoder and return numpy embeddings."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required to extract AE embeddings.")
    ae.eval()
    values = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        tensor = torch.FloatTensor(values).to(device)
        enc = ae.enc(tensor)
    return enc.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AEHeadPredictor:
    """
    Train a BERNN AE, then sweep interchangeable classification heads.

    Parameters
    ----------
    config : TrainingConfig
        A BERNN TrainingConfig object.  ``keep_models`` is forced to True
        internally so the encoder weights survive after training.
    head_types : list[str] | None
        Head types to sweep. Defaults to all supported types (or all except
        XGBoost when xgboost is not installed).
    n_cv : int
        Number of StratifiedKFold folds for head evaluation (default 3).
    device : str
        Torch device string (default 'cpu').
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        config=None,
        head_types: Optional[List[str]] = None,
        n_cv: int = 3,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.config = config
        self.head_types = head_types or (HEAD_TYPES if _HAS_XGB else HEAD_TYPES_NO_XGB)
        self.n_cv = n_cv
        self.device = device
        self.verbose = verbose

        # Set after .fit()
        self.best_mcc: float = -1.0
        self.cv_mcc_mean: float = -1.0
        self.best_head_type: Optional[str] = None
        self.best_head: Optional[Any] = None
        self._ae = None
        self._le: Optional[LabelEncoder] = None
        self._sweep_results: Dict[str, Any] = {}
        self._log_path: Optional[str] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train,
        X_test: pd.DataFrame,
        groups_train=None,
        groups_test=None,
    ) -> "AEHeadPredictor":
        """Train AE, extract embeddings, sweep heads, fit best head."""
        # ---- Step 1: train BERNN AE ----
        from bernn import TrainAEClassifierHoldout
        from bernn.config.training_config import TrainingConfig

        cfg = self.config or TrainingConfig(
            n_epochs=200,
            warmup=50,
            groupkfold=True,
            optimize_hyperparams=False,
            device=self.device,
        )
        # Force keep_models so encoder weights are retained
        cfg.keep_models = True
        cfg.device = self.device

        # Unique run directory so parallel submissions don't collide
        run_id = str(uuid.uuid4())
        self._log_path = f"/tmp/ae_head_predictor_{run_id}"

        trainer = TrainAEClassifierHoldout(
            config=cfg,
            log_metrics=False,
            keep_models=True,
            log_inputs=False,
            log_plots=False,
            log_tb=False,
            log_mlflow=False,
            log_dvclive=False,
        )

        y_train_str = pd.Series(y_train).astype(str)
        self._le = LabelEncoder().fit(y_train_str)

        if self.verbose:
            print(f"[AEHeadPredictor] Training AE encoder ({cfg.n_epochs} epochs)...")
        try:
            trainer.fit_predict(
                X_train.copy(),
                y_train_str,
                X_test=X_test.copy(),
                y_test=None,
                groups_train=groups_train,
                groups_test=groups_test,
                cross_validation=False,
                cross_test=False,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[AEHeadPredictor] AE training error: {exc}. Falling back to raw features.")
            self._ae = None
        else:
            # Try to grab the trained model from the trainer
            self._ae = (
                getattr(trainer, "ae", None)
                or getattr(trainer, "model", None)
                or getattr(trainer, "best_model", None)
            )

        # ---- Step 2: get embeddings ----
        if self._ae is not None:
            try:
                if self.verbose:
                    print("[AEHeadPredictor] Extracting frozen encoder embeddings...")
                X_train_emb = _extract_embeddings_from_model(self._ae, X_train, self.device)
                X_test_emb  = _extract_embeddings_from_model(self._ae, X_test, self.device)
            except Exception as exc:
                if self.verbose:
                    print(f"[AEHeadPredictor] Embedding extraction failed: {exc}. Using raw features.")
                X_train_emb = X_train.to_numpy(dtype=float)
                X_test_emb  = X_test.to_numpy(dtype=float)
        else:
            if self.verbose:
                print("[AEHeadPredictor] No AE available — using raw features.")
            X_train_emb = X_train.to_numpy(dtype=float)
            X_test_emb  = X_test.to_numpy(dtype=float)

        X_train_emb = np.nan_to_num(X_train_emb, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_emb  = np.nan_to_num(X_test_emb,  nan=0.0, posinf=0.0, neginf=0.0)

        # Encode labels
        y_enc = self._le.transform(y_train_str)

        # ---- Step 3: sweep all heads (cv=3) ----
        if self.verbose:
            print(f"[AEHeadPredictor] Sweeping {len(self.head_types)} head types (cv={self.n_cv})...")
        self._sweep_results = {}
        for ht in self.head_types:
            try:
                res = cv_score_head(X_train_emb, y_enc, ht, n_splits=self.n_cv)
                self._sweep_results[ht] = res
                if self.verbose:
                    print(f"  {ht:30s}  cv_valid_mcc = {res.get('mean_valid_mcc', float('nan')):.4f}")
            except Exception as exc:
                self._sweep_results[ht] = {"mean_valid_mcc": float("nan"), "error": str(exc)}

        # ---- Step 4: pick best head ----
        self.best_head_type = max(
            self._sweep_results,
            key=lambda k: self._sweep_results[k].get("mean_valid_mcc", float("-inf")),
            default=self.head_types[0],
        )
        best_cv_mcc = self._sweep_results.get(self.best_head_type, {}).get("mean_valid_mcc", -1.0)
        self.cv_mcc_mean = float(best_cv_mcc) if not np.isnan(best_cv_mcc) else -1.0

        if self.verbose:
            print(f"[AEHeadPredictor] Best head: {self.best_head_type}  (cv MCC={self.cv_mcc_mean:.4f})")

        # ---- Step 5: fit best head on full training set ----
        self.best_head, train_mcc, _ = fit_and_score_head(
            X_train_emb, y_enc, X_test_emb, y_enc[:len(X_test_emb)],
            self.best_head_type,
        )
        self.best_mcc = self.cv_mcc_mean  # cv MCC is the honest estimate

        # Store embeddings for predict()
        self._X_test_emb = X_test_emb

        # Clean up temp log dir
        if self._log_path and os.path.exists(self._log_path):
            shutil.rmtree(self._log_path, ignore_errors=True)

        return self

    # Alias so the leaderboard harness can also call .fit_predict()
    def fit_predict(self, X_train, y_train, X_test,
                    groups_train=None, groups_test=None, **_kw):
        self.fit(X_train, y_train, X_test, groups_train=groups_train, groups_test=groups_test)
        return self

    # ------------------------------------------------------------------
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return string class labels for X_test."""
        if self.best_head is None:
            raise RuntimeError("Call .fit() before .predict().")

        if self._ae is not None:
            try:
                X_emb = _extract_embeddings_from_model(self._ae, X_test, self.device)
            except Exception:
                X_emb = X_test.to_numpy(dtype=float)
        else:
            X_emb = X_test.to_numpy(dtype=float)

        X_emb = np.nan_to_num(X_emb, nan=0.0, posinf=0.0, neginf=0.0)
        preds_enc = self.best_head.predict(X_emb)
        return self._le.inverse_transform(preds_enc.astype(int))

    # ------------------------------------------------------------------
    def sweep_summary(self) -> pd.DataFrame:
        """Return a DataFrame of all heads sorted by cv_valid_mcc."""
        rows = []
        for ht, res in self._sweep_results.items():
            rows.append({
                "head_type":       ht,
                "cv_valid_mcc":    res.get("mean_valid_mcc", float("nan")),
                "std_valid_mcc":   res.get("std_valid_mcc", float("nan")),
                "mean_train_mcc":  res.get("mean_train_mcc", float("nan")),
                "error":           res.get("error", ""),
            })
        df = pd.DataFrame(rows).sort_values("cv_valid_mcc", ascending=False).reset_index(drop=True)
        return df
