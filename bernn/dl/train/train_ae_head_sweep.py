#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ae_head_sweep.py
======================
Optuna sweep: jointly optimise AE encoder hyperparameters **and** the
classification head type/hyperparameters.

Design
------
Each Optuna trial:
  1. Trains the AE encoder with the trial's hyperparameters (layer sizes,
     learning rate, dropout, domain-alignment loss weight, etc.).
  2. Freezes the encoder and extracts train/valid embeddings.
  3. Trains the head suggested by the trial (XGBoost, RandomForest,
     LinearSVC, LogisticRegression, KNN, SVC-RBF, GradientBoosting,
     prototype_mean, prototype_kmeans) on the embeddings.
  4. Evaluates using StratifiedKFold(n_splits=3) on the training embeddings
     and reports mean valid_mcc — the same criterion used by the Optuna sweep
     in the classical AE-then-classifier workflow.

The AE architecture follows the existing TrainAE pattern from
``train_ae_then_classifier_holdout.py``.  No gradients flow through the
classification head; the encoder is fully frozen during head fitting.

Usage
-----
    python -m bernn.dl.train.train_ae_head_sweep \
        --dataset prostate \
        --path ./data/ \
        --n_trials 200 \
        --n_epochs 300 \
        --n_cv 3 \
        --exp_id bernn_head_sweep_v1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

from bernn.dl.models.pytorch.aedann import AutoEncoder2 as AutoEncoder
from bernn.dl.models.pytorch.aedann import ReverseLayerF
from bernn.dl.models.pytorch.utils.dataset import get_loaders, get_loaders_no_pool
from bernn.dl.models.pytorch.utils.utils import (
    get_empty_dicts,
    get_empty_traces,
    get_optimizer,
    to_categorical,
    compute_class_triplet,
)
from bernn.dl.train.head_classifier import (
    HEAD_TYPES,
    HEAD_TYPES_NO_XGB,
    cv_score_head,
    fit_and_score_head,
    sweep_all_heads,
)
from bernn.utils.mlflow_compat import mlflow

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
from bernn.utils.data_getters import load_data_for_args

try:
    import xgboost  # noqa: F401
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    ae: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, labels) for the entire loader, encoder frozen."""
    X, y, _batches = extract_embeddings_labels_batches(ae, loader, device)
    return X, y


@torch.no_grad()
def extract_embeddings_labels_batches(
    ae: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, labels, batches) for the entire loader."""
    ae.eval()
    all_enc, all_labels, all_batches = [], [], []
    for batch in loader:
        inputs, _names, labels, domain, to_rec = batch[:5]
        inputs = inputs.to(device).float()
        to_rec = to_rec.to(device).float()
        enc, _rec, _zinb, _kld = ae(inputs, to_rec, domain, sampling=False)
        all_enc.append(enc.float().cpu().numpy())
        all_labels.extend(labels if isinstance(labels, list) else labels.tolist())
        if hasattr(domain, "detach"):
            all_batches.extend(domain.detach().cpu().numpy().tolist())
        else:
            all_batches.extend(list(domain))
    if not all_enc:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))
    return np.concatenate(all_enc, axis=0), np.array(all_labels), np.array(all_batches)


def _predict_with_optional_label_decoder(head, X: np.ndarray) -> np.ndarray:
    preds = head.predict(X)
    le = getattr(head, "_bernn_label_encoder", None)
    if le is not None:
        preds = le.inverse_transform(np.asarray(preds).astype(int))
    return np.asarray(preds)


def _classification_metrics(prefix: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute broad classification metrics for one split."""
    out: Dict[str, float] = {}
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
        )
        out[f"{prefix}_mcc"] = float(matthews_corrcoef(y_true, y_pred))
        out[f"{prefix}_accuracy"] = float(accuracy_score(y_true, y_pred))
        out[f"{prefix}_balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out[f"{prefix}_precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        out[f"{prefix}_precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        out[f"{prefix}_recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        out[f"{prefix}_recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        out[f"{prefix}_f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        out[f"{prefix}_f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception:
        pass
    return out


def _embedding_batch_effect_metrics(*splits) -> Dict[str, float]:
    """Batch-effect metrics on frozen embeddings; lower is better."""
    metrics = {
        "batch_silhouette": 1.0,
        "batch_centroid_dispersion": 1.0,
        "batch_nbe": 1.0,
        "batch_entropy": 0.0,
        "batch_entropy_loss": 1.0,
        "batch_nmi": 1.0,
        "batch_ari": 1.0,
        "batch_nri": 1.0,
        "batch_ami": 1.0,
        "batch_lisi": 1.0,
        "batch_ilisi": 1.0,
        "batch_metric_samples": 0.0,
    }
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            adjusted_rand_score,
            normalized_mutual_info_score,
            silhouette_score,
        )
        from sklearn.neighbors import NearestNeighbors

        xs, bs = [], []
        for X, batches in splits:
            if X is None or len(X) == 0:
                continue
            xs.append(np.asarray(X, dtype=float))
            bs.append(np.asarray(batches).astype(str))
        if not xs:
            return metrics
        X_all = np.nan_to_num(np.concatenate(xs, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        batches_all = np.concatenate(bs, axis=0)
        metrics["batch_metric_samples"] = float(len(X_all))
        unique_batches = np.unique(batches_all)
        if len(X_all) < 3 or len(unique_batches) < 2:
            return metrics

        n_components = min(20, X_all.shape[1], max(1, X_all.shape[0] - 1))
        values = PCA(n_components=n_components).fit_transform(X_all) if X_all.shape[1] > n_components else X_all

        _, counts = np.unique(batches_all, return_counts=True)
        if counts.min() >= 2:
            sil = float(silhouette_score(values, batches_all))
            metrics["batch_silhouette"] = float(np.clip(sil, 0.0, 1.0))

        centroids = []
        for batch in sorted(unique_batches):
            mask = batches_all == batch
            centroids.append(values[mask].mean(axis=0))
        if len(centroids) >= 2:
            centroid_arr = np.vstack(centroids)
            diffs = centroid_arr[:, None, :] - centroid_arr[None, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            tri = dists[np.triu_indices(len(centroids), k=1)]
            if tri.size:
                centroid_disp = float(np.mean(tri))
                global_center = np.mean(values, axis=0)
                sample_disp = float(np.mean(np.linalg.norm(values - global_center, axis=1)))
                denom = centroid_disp + sample_disp
                metrics["batch_centroid_dispersion"] = float(np.clip(centroid_disp / denom, 0.0, 1.0)) if denom > 0 else 0.0

        n_clusters = min(max(2, len(unique_batches)), len(values))
        if n_clusters >= 2:
            cluster_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(values)
            _, batch_codes = np.unique(batches_all, return_inverse=True)
            metrics["batch_nmi"] = float(np.clip(normalized_mutual_info_score(batch_codes, cluster_labels), 0.0, 1.0))
            ari = float(adjusted_rand_score(batch_codes, cluster_labels))
            ami = float(adjusted_mutual_info_score(batch_codes, cluster_labels))
            metrics["batch_ari"] = ari
            metrics["batch_nri"] = float(np.clip(max(0.0, ari), 0.0, 1.0))
            metrics["batch_ami"] = float(np.clip(max(0.0, ami), 0.0, 1.0))

            try:
                k = min(30, max(2, len(values) - 1))
                nn = NearestNeighbors(n_neighbors=k + 1).fit(values)
                neigh = nn.kneighbors(values, return_distance=False)[:, 1:]
                inv_scores = []
                for row in neigh:
                    neighbor_batches = batch_codes[row]
                    cts = np.bincount(neighbor_batches, minlength=len(unique_batches)).astype(float)
                    probs = cts / max(cts.sum(), 1.0)
                    inv_scores.append(1.0 / max(float(np.sum(probs ** 2)), 1e-12))
                if inv_scores:
                    ilisi = float(np.mean(inv_scores))
                    metrics["batch_ilisi"] = ilisi
                    metrics["batch_lisi"] = ilisi
            except Exception:
                pass

            denom = np.log(max(len(unique_batches), 2))
            entropies, weights = [], []
            if denom > 0:
                for cluster_id in sorted(np.unique(cluster_labels)):
                    mask = cluster_labels == cluster_id
                    cluster_batches = batches_all[mask]
                    if len(cluster_batches) == 0:
                        continue
                    _, cts = np.unique(cluster_batches, return_counts=True)
                    probs = cts.astype(float) / float(cts.sum())
                    probs = probs[probs > 0]
                    entropy = -float(np.sum(probs * np.log(probs))) / float(denom)
                    entropies.append(entropy)
                    weights.append(float(len(cluster_batches)))
                if entropies and np.sum(weights) > 0:
                    mean_entropy = float(np.average(entropies, weights=np.asarray(weights, dtype=float)))
                    metrics["batch_entropy"] = float(np.clip(mean_entropy, 0.0, 1.0))
                    metrics["batch_nbe"] = float(np.clip(1.0 - mean_entropy, 0.0, 1.0))
                    metrics["batch_entropy_loss"] = metrics["batch_nbe"]
    except Exception:
        pass
    return metrics


# ---------------------------------------------------------------------------
# Optuna parameter suggestion helpers
# ---------------------------------------------------------------------------

def _suggest_ae_params(trial: "optuna.Trial", args) -> Dict[str, Any]:
    """Suggest AE architecture / training hyperparameters."""
    p: Dict[str, Any] = {}
    p["layer1"]    = trial.suggest_int("layer1",    8, 256, log=True)
    p["layer2"]    = trial.suggest_int("layer2",    4, 64,  log=True)
    p["lr"]        = trial.suggest_float("lr",       1e-4, 1e-1, log=True)
    p["wd"]        = trial.suggest_float("wd",       1e-8, 1e-4, log=True)
    p["nu"]        = trial.suggest_float("nu",       1e-4, 1e2,  log=False)
    p["dropout"]   = trial.suggest_float("dropout",  0.0,  0.5)
    p["smoothing"] = trial.suggest_float("smoothing", 0.0, 0.2)
    p["margin"]    = trial.suggest_float("margin",   1e-4, 10.0)
    p["warmup"]    = trial.suggest_int("warmup",     1,    100)
    # binarize excluded: BCEWithLogitsLoss on unbinarized decoder outputs
    # causes device-side CUDA asserts that poison the context for the whole study.
    p["scaler"]    = trial.suggest_categorical("scaler", ["robust", "standard"])
    p["ncols"]     = trial.suggest_int("ncols",      20,   10000)
    p["gamma"]     = 0.0
    p["beta"]      = 0.0
    if args.dloss in ["revTriplet", "revDANN", "DANN", "inverseTriplet", "normae"]:
        p["gamma"] = trial.suggest_float("gamma", 1e-8, 1e-4, log=True)
    if getattr(args, "variational", False):
        p["beta"] = trial.suggest_float("beta", 1e-2, 1e2, log=True)
    return p


def _suggest_head_params(trial: "optuna.Trial", head_type: str) -> Dict[str, Any]:
    """Suggest head-specific hyperparameters conditioned on head type."""
    p: Dict[str, Any] = {}

    if head_type == "xgboost":
        p["n_estimators"]    = trial.suggest_int("xgb_n_estimators", 50, 500, step=50)
        p["max_depth"]       = trial.suggest_int("xgb_max_depth", 3, 10)
        p["learning_rate"]   = trial.suggest_float("xgb_lr", 1e-3, 0.5, log=True)
        p["reg_alpha"]       = trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True)
        p["reg_lambda"]      = trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True)
        p["subsample"]       = trial.suggest_float("xgb_subsample", 0.5, 1.0)
        p["colsample_bytree"]= trial.suggest_float("xgb_colsample", 0.5, 1.0)
        p["tree_method"]     = "hist"

    elif head_type == "random_forest":
        p["n_estimators"]      = trial.suggest_int("rf_n_estimators", 50, 500, step=50)
        p["max_features"]      = trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None])
        p["min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 10)
        p["min_samples_leaf"]  = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        p["criterion"]         = trial.suggest_categorical("rf_criterion", ["gini", "entropy"])
        p["oob_score"]         = trial.suggest_categorical("rf_oob_score", [True, False])

    elif head_type == "linear_svc":
        p["C"]       = trial.suggest_float("linsvc_C", 1e-3, 1e4, log=True)
        p["tol"]     = trial.suggest_float("linsvc_tol", 1e-5, 1e-2, log=True)
        p["max_iter"] = trial.suggest_int("linsvc_max_iter", 200, 2000, step=100)

    elif head_type == "svc_rbf":
        p["C"]       = trial.suggest_float("svcrbf_C", 1e-3, 1e3, log=True)
        p["max_iter"] = trial.suggest_int("svcrbf_max_iter", 100, 1000, step=100)

    elif head_type == "logistic_regression":
        p["C"]        = trial.suggest_float("lr_C", 1e-3, 1e4, log=True)
        p["penalty"]  = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
        p["max_iter"] = trial.suggest_int("lr_max_iter", 100, 2000, step=100)

    elif head_type == "knn":
        p["n_neighbors"] = trial.suggest_int("knn_k", 1, 31, step=2)
        p["metric"]      = trial.suggest_categorical("knn_metric", ["euclidean", "manhattan", "cosine"])
        p["weights"]     = trial.suggest_categorical("knn_weights", ["uniform", "distance"])

    elif head_type == "gradient_boosting":
        p["n_estimators"]  = trial.suggest_int("gbc_n_estimators", 50, 300, step=50)
        p["max_depth"]     = trial.suggest_int("gbc_max_depth", 3, 8)
        p["learning_rate"] = trial.suggest_float("gbc_lr", 1e-3, 0.5, log=True)
        p["subsample"]     = trial.suggest_float("gbc_subsample", 0.5, 1.0)

    elif head_type == "prototype_mean":
        p["proto_metric"] = trial.suggest_categorical("proto_mean_metric", ["cosine", "euclidean"])

    elif head_type == "prototype_kmeans":
        p["proto_components"] = trial.suggest_int("proto_k", 1, 5)
        p["proto_metric"]     = trial.suggest_categorical("proto_km_metric", ["cosine", "euclidean"])

    return p


# ---------------------------------------------------------------------------
# AE training logic (lightweight, adapted from TrainAE holdout)
# ---------------------------------------------------------------------------

class AEHeadSweepTrainer:
    """
    Trains an AE encoder using Optuna-suggested hyperparameters, then fits
    and evaluates interchangeable classification heads on the frozen embeddings.
    """

    def __init__(self, args, path: str, unique_labels: List[str],
                 unique_batches: List[str], data: Dict, n_cv: int = 3):
        self.args = args
        self.path = path
        self.unique_labels  = unique_labels
        self.unique_batches = unique_batches
        self.data  = data
        self.n_cv  = n_cv
        self.best_valid_mcc = float("-inf")
        self.best_head_type: Optional[str] = None
        self.best_head_params: Optional[Dict] = None

    # ------------------------------------------------------------------
    def _build_ae(self, layer1: int, layer2: int) -> AutoEncoder:
        args = self.args
        n_features = self.data["inputs"]["train"].shape[1]
        ae = AutoEncoder(
            n_features,
            n_batches=len(self.unique_batches),
            nb_classes=len(self.unique_labels),
            mapper=getattr(args, "use_mapping", False),
            variational=getattr(args, "variational", False),
            layer1=layer1,
            layer2=layer2,
            dropout=0.0,
            n_layers=2,
            prune_threshold=0.0,
            conditional=False,
            add_noise=0,
            tied_weights=getattr(args, "tied_weights", False),
            update_grid=False,
            device=args.device,
        ).to(args.device)
        return ae

    def _get_losses(self, ae, params):
        import torch.nn as nn
        scale   = params.get("scaler", "standard")
        dloss   = getattr(self.args, "dloss", "inverseTriplet")
        rec_loss = getattr(self.args, "rec_loss", "mse")

        sceloss = nn.CrossEntropyLoss(label_smoothing=params.get("smoothing", 0.0))
        celoss  = nn.CrossEntropyLoss()
        mseloss = nn.MSELoss() if rec_loss == "mse" else nn.L1Loss()
        if scale == "binarize":
            mseloss = nn.BCELoss()
        margin = params.get("margin", 1.0)
        if dloss == "revTriplet":
            triplet_loss = nn.TripletMarginLoss(margin, p=2, swap=True)
        else:
            triplet_loss = nn.TripletMarginLoss(max(margin, 1e-6), p=2, swap=False)
        return sceloss, celoss, mseloss, triplet_loss

    # ------------------------------------------------------------------
    def _train_ae(self, ae, params, loaders, trial_num=None, wandb_run=None):
        """Train the AE encoder. Returns (best_valid_mcc, epoch_metrics_list).

        Fixes
        -----
        * Per-epoch stdout: rec / d / c loss + valid_mcc each epoch.
        * Label-encoding bug: batch labels may be int-encoded already; handle
          both string and integer labels when computing MCC.
        * W&B per-epoch logging when wandb_run is provided.
        """
        from bernn.dl.models.pytorch.utils.utils import get_optimizer
        from sklearn.metrics import matthews_corrcoef
        from sklearn.preprocessing import LabelEncoder as LE

        args       = self.args
        nu, lr, wd = params["nu"], params["lr"], params["wd"]
        optimizer_ae = get_optimizer(ae, lr, wd, "adam")
        optimizer_c  = get_optimizer(ae.classifier, nu * lr, wd, "adam")

        sceloss, celoss, mseloss, triplet_loss = self._get_losses(ae, params)
        dloss      = getattr(args, "dloss", "inverseTriplet")
        n_epochs   = getattr(args, "n_epochs", 200)
        early_stop = getattr(args, "early_stop", 30)

        # Label encoder — handles both string and int-encoded batch labels
        le = LE().fit(self.unique_labels)

        best_valid_mcc     = float("-inf")
        early_stop_counter = 0
        epoch_metrics      = []
        tag = f"[trial {trial_num}]" if trial_num is not None else "[AE]"

        for epoch in range(n_epochs):
            # ── TRAIN ──────────────────────────────────────────────────────
            ae.train()
            epoch_rec, epoch_d, epoch_c, n_batches = 0.0, 0.0, 0.0, 0
            for batch in loaders.get("train", []):
                raw = batch[:11] if len(batch) >= 11 else (*batch, *([None] * max(0, 11 - len(batch))))
                inputs, _names, labels, domain, to_rec, _not_rec,                     pos_to_rec, neg_to_rec, pos_batch, neg_batch, _ = raw
                if inputs is None:
                    break
                inputs = inputs.to(args.device).float()
                to_rec = to_rec.to(args.device).float() if to_rec is not None else inputs

                optimizer_ae.zero_grad()
                optimizer_c.zero_grad()

                enc, rec, _zinb, _kld = ae(inputs, to_rec, domain, sampling=True)
                _rec_mean = rec["mean"] if isinstance(rec, dict) else rec
                rec_val   = _rec_mean[-1] if isinstance(_rec_mean, (list, tuple)) else _rec_mean

                if enc.abs().sum() == 0:
                    continue

                # BCEWithLogitsLoss needs target in [0,1]; clamp when binarize
                _rec_target = to_rec.clamp(0.0, 1.0) if params.get("scaler") == "binarize" else to_rec
                rec_loss_val = mseloss(rec_val, _rec_target)

                gamma  = params.get("gamma", 0.0)
                d_loss = torch.tensor(0.0, device=args.device)
                if gamma > 0 and dloss != "no":
                    if dloss in ["revTriplet", "inverseTriplet"]:
                        if pos_batch is not None and neg_batch is not None:
                            pb = pos_batch.to(args.device).float()
                            nb = neg_batch.to(args.device).float()
                            pos_enc, _, _, _ = ae(pb, pb, domain, sampling=True)
                            neg_enc, _, _, _ = ae(nb, nb, domain, sampling=True)
                            if dloss == "revTriplet":
                                d_loss = triplet_loss(
                                    ReverseLayerF.apply(enc, 1),
                                    ReverseLayerF.apply(pos_enc, 1),
                                    ReverseLayerF.apply(neg_enc, 1),
                                )
                            else:
                                d_loss = triplet_loss(enc, pos_enc, neg_enc)

                cats    = to_categorical(labels, len(self.unique_labels)).to(args.device)
                c_loss  = sceloss(ae.classifier(enc), cats.argmax(1))

                loss = rec_loss_val + gamma * d_loss + c_loss
                if getattr(args, "class_triplet", False) and pos_to_rec is not None:
                    loss = loss + getattr(args, "class_triplet_w", 1.0) * compute_class_triplet(
                        ae, enc, pos_to_rec, neg_to_rec, domain, args.device,
                        margin=max(float(params.get("margin", 1.0)), 1e-6), mapping=False,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                optimizer_ae.step()
                optimizer_c.step()

                epoch_rec += float(rec_loss_val.item())
                epoch_d   += float(d_loss.item()) if hasattr(d_loss, "item") else float(d_loss)
                epoch_c   += float(c_loss.item())
                n_batches += 1

            if n_batches:
                epoch_rec /= n_batches
                epoch_d   /= n_batches
                epoch_c   /= n_batches

            # ── VALID ──────────────────────────────────────────────────────
            ae.eval()
            all_preds, all_labels_val = [], []
            with torch.no_grad():
                for batch in loaders.get("valid", []):
                    inputs, _names, labels, domain, to_rec = batch[:5]
                    inputs = inputs.to(args.device).float()
                    to_rec = to_rec.to(args.device).float() if to_rec is not None else inputs
                    enc, _, _, _ = ae(inputs, to_rec, domain, sampling=False)
                    p = ae.classifier(enc).argmax(1).cpu().numpy()
                    all_preds.extend(p.tolist())
                    all_labels_val.extend(labels if isinstance(labels, list) else labels.tolist())

            valid_mcc = float("-inf")
            ae_epoch_metrics = {}
            if len(np.unique(all_labels_val)) > 1 and all_preds:
                try:
                    preds_enc  = np.array(all_preds, dtype=int)
                    labels_arr = np.array(all_labels_val)
                    # Batch labels may be raw strings OR already int-encoded —
                    # use LabelEncoder only when they're strings.
                    if labels_arr.dtype.kind in ("U", "S", "O"):
                        labels_enc = le.transform(labels_arr)
                    else:
                        labels_enc = labels_arr.astype(int)
                    ae_epoch_metrics = _classification_metrics("valid", labels_enc, preds_enc)
                    valid_mcc = ae_epoch_metrics.get("valid_mcc", float("-inf"))
                except Exception as _exc:
                    valid_mcc = float("-inf")

            # ── Print epoch ────────────────────────────────────────────────
            print(
                f"{tag} epoch={epoch+1:4d}/{n_epochs}  "
                f"rec={epoch_rec:.4f}  d={epoch_d:.4f}  c={epoch_c:.4f}  "
                f"valid_mcc={valid_mcc:+.4f}  best={best_valid_mcc:+.4f}  "
                f"es={early_stop_counter}/{early_stop}",
                flush=True,
            )

            # ── W&B per-epoch ──────────────────────────────────────────────
            if wandb_run is not None:
                try:
                    payload = {
                        "epoch":        epoch,
                        "ae/rec_loss":  epoch_rec,
                        "ae/d_loss":    epoch_d,
                        "ae/c_loss":    epoch_c,
                        "ae/valid_mcc": valid_mcc,
                        "ae/best_mcc":  best_valid_mcc,
                    }
                    for key, value in ae_epoch_metrics.items():
                        payload[f"ae/{key}"] = value
                    wandb_run.log(payload)
                except Exception:
                    pass

            epoch_row = {
                "epoch": epoch, "rec": epoch_rec, "d": epoch_d,
                "c": epoch_c, "valid_mcc": valid_mcc,
            }
            epoch_row.update(ae_epoch_metrics)
            epoch_metrics.append(epoch_row)

            if valid_mcc > best_valid_mcc:
                best_valid_mcc     = valid_mcc
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop:
                    print(f"{tag} early stop at epoch {epoch+1} (patience={early_stop})", flush=True)
                    break

        return best_valid_mcc, epoch_metrics
    # ------------------------------------------------------------------
    def objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective: suggest AE params + head, train AE, fit head, return cv MCC."""
        import gc

        args = self.args
        ae_params = _suggest_ae_params(trial, args)

        # Head type suggestion — include XGBoost only if installed
        available_heads = HEAD_TYPES if _HAS_XGB else HEAD_TYPES_NO_XGB
        head_type  = trial.suggest_categorical("head_type", available_heads)
        head_params = _suggest_head_params(trial, head_type)

        # Build AE and get loaders
        ae = self._build_ae(ae_params["layer1"], ae_params["layer2"])

        # Apply scaler to data
        self.args.scaler = ae_params.get("scaler", "standard")
        self.args.ncols  = ae_params.get("ncols", -1)

        import copy as _copy
        from bernn.utils.utils import scale_data
        data = _copy.deepcopy(self.data)
        data, _ = scale_data(self.args.scaler, data, args.device)
        for g in list(data["inputs"].keys()):
            data["inputs"][g] = data["inputs"][g].round(4)

        dloss = getattr(args, "dloss", "inverseTriplet")
        bs = getattr(args, "bs", 32)

        # Build class-balanced sample weights (train) + uniform (valid/test).
        # get_loaders_no_pool uses samples_weights[split] directly for
        # WeightedRandomSampler — passing None crashes with NoneType subscript.
        _samples_weights: dict = {}
        for _g in ("train", "valid", "test"):
            if _g not in data.get("cats", {}):
                continue
            _cats_g = np.asarray(data["cats"][_g])
            if _g == "train" and len(_cats_g):
                _cls, _cnt = np.unique(_cats_g, return_counts=True)
                _w = {int(c): 1.0 / max(int(n), 1) for c, n in zip(_cls, _cnt)}
                _samples_weights[_g] = [_w[int(c)] for c in _cats_g]
            else:
                _samples_weights[_g] = [1.0] * len(_cats_g)

        try:
            loaders = get_loaders(data, False, _samples_weights, dloss, None, None, bs, args.device)
        except Exception:
            loaders = get_loaders_no_pool(data, False, _samples_weights, dloss, None, None, bs, args.device)

        # Step 1: train AE encoder
        # Guard: flush the CUDA queue with synchronize() to surface any pending
        # async device-side assert from a previous trial before we allocate.
        if torch.cuda.is_available() and str(getattr(args, "device", "")).startswith("cuda"):
            try:
                torch.cuda.synchronize(args.device)
            except RuntimeError as _cuda_guard_err:
                trial.set_user_attr("ae_error", f"CUDA context poisoned: {_cuda_guard_err}")
                return float("-inf")

        # -- optional W&B run per trial --
        _wandb_run = None
        if _HAS_WANDB and getattr(args, "log_wandb", False):
            try:
                _wandb_run = _wandb.init(
                    project=getattr(args, "wandb_project", "bernn_head_sweep"),
                    entity=getattr(args, "wandb_entity", None) or None,
                    name=f"{args.exp_id}_trial{trial.number}",
                    group=args.exp_id,
                    config={**ae_params, "head_type": head_type, **head_params,
                            "dloss": getattr(args, "dloss", ""),
                            "class_triplet": getattr(args, "class_triplet", False)},
                    reinit=True,
                )
            except Exception:
                _wandb_run = None

        try:
            _ae_mcc, _epoch_metrics = self._train_ae(
                ae, ae_params, loaders,
                trial_num=trial.number,
                wandb_run=_wandb_run,
            )
        except Exception as exc:
            trial.set_user_attr("ae_error", str(exc))
            if _wandb_run is not None:
                try: _wandb_run.finish(exit_code=1)
                except Exception: pass
            return float("-inf")

        # Step 2: freeze encoder, extract embeddings
        ae.eval()
        for param in ae.enc.parameters():
            param.requires_grad = False
        for param in ae.dec.parameters():
            param.requires_grad = False

        try:
            X_train, y_train, b_train = extract_embeddings_labels_batches(ae, loaders["train"], args.device)
            X_valid, y_valid, b_valid = extract_embeddings_labels_batches(ae, loaders["valid"], args.device)
            X_test, y_test, b_test = extract_embeddings_labels_batches(ae, loaders["test"], args.device)
        except Exception as exc:
            trial.set_user_attr("embed_error", str(exc))
            return float("-inf")

        if len(X_train) == 0 or len(X_valid) == 0:
            return float("-inf")

        # Metric contract: public valid_* metrics are cross-validation metrics.
        # Single held-out validation metrics are logged as holdout_valid_*.
        try:
            fitted_head, train_mcc, holdout_valid_mcc = fit_and_score_head(
                X_train, y_train, X_valid, y_valid, head_type, head_params
            )
            train_preds = _predict_with_optional_label_decoder(fitted_head, X_train)
            valid_preds = _predict_with_optional_label_decoder(fitted_head, X_valid)
            if len(X_test):
                test_preds = _predict_with_optional_label_decoder(fitted_head, X_test)
                test_metrics = _classification_metrics("test", y_test, test_preds)
                test_mcc = float(test_metrics.get("test_mcc", float("nan")))
            else:
                test_preds = np.empty((0,))
                test_metrics = {"test_mcc": float("nan")}
                test_mcc = float("nan")
            split_metrics = {}
            split_metrics.update(_classification_metrics("train", y_train, train_preds))
            split_metrics.update(_classification_metrics("holdout_valid", y_valid, valid_preds))
            split_metrics.update(test_metrics)
            # Preserve these exact primary names even if sklearn metric helpers
            # were unavailable for some reason. valid_mcc is filled from CV below.
            split_metrics["train_mcc"] = float(train_mcc)
            split_metrics["holdout_valid_mcc"] = float(holdout_valid_mcc)
            split_metrics["test_mcc"] = float(test_mcc)
        except Exception as exc:
            trial.set_user_attr("head_error", str(exc))
            return float("-inf")

        cv_metrics = {}
        try:
            cv_result = cv_score_head(
                X_train, y_train,
                head_type=head_type,
                head_params=head_params,
                n_splits=self.n_cv,
                random_state=42,
            )
            valid_mcc = float(cv_result.get("mean_valid_mcc", float("nan")))
            head_cv_train_mcc = float(cv_result.get("mean_train_mcc", float("nan")))
            for _k, _v in cv_result.items():
                if _k.startswith("mean_valid_"):
                    cv_metrics[_k.replace("mean_valid_", "valid_")] = float(_v)
                elif _k.startswith("std_valid_"):
                    cv_metrics[_k.replace("std_valid_", "valid_") + "_std"] = float(_v)
                elif _k.startswith("mean_train_"):
                    cv_metrics[_k.replace("mean_train_", "cv_train_")] = float(_v)
                elif _k.startswith("std_train_"):
                    cv_metrics[_k.replace("std_train_", "cv_train_") + "_std"] = float(_v)
            cv_metrics["valid_mcc"] = valid_mcc
            cv_metrics["cv_mcc_mean"] = valid_mcc
            cv_metrics["cv_mcc_std"] = float(cv_result.get("std_valid_mcc", float("nan")))
            cv_metrics["cv_n_folds_ok"] = float(len(cv_result.get("fold_valid_mccs", [])))
        except Exception:
            valid_mcc = float("nan")
            head_cv_train_mcc = float("nan")

        batch_metrics = _embedding_batch_effect_metrics(
            (X_train, b_train), (X_valid, b_valid), (X_test, b_test)
        )

        split_metrics.update(cv_metrics)
        for _k, _v in split_metrics.items():
            trial.set_user_attr(_k, _v)
        trial.set_user_attr("head_cv_train_mcc", head_cv_train_mcc)
        trial.set_user_attr("ae_classifier_mcc", _ae_mcc)
        try:
            trial.set_user_attr("epoch_metrics_json", json.dumps(_epoch_metrics))
        except Exception:
            pass
        for _k, _v in batch_metrics.items():
            trial.set_user_attr(_k, _v)
        trial.set_user_attr("head_type", head_type)
        trial.set_user_attr("head_params", json.dumps(head_params))

        if valid_mcc > self.best_valid_mcc:
            self.best_valid_mcc = valid_mcc
            self.best_head_type = head_type
            self.best_head_params = head_params

        # MLflow logging (optional)
        if getattr(args, "log_mlflow", False):
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params({**ae_params, "head_type": head_type, **head_params})
                    for _k, _v in split_metrics.items():
                        mlflow.log_metric(_k, _v)
                    mlflow.log_metric("head_cv_train_mcc", head_cv_train_mcc)
                    mlflow.log_metric("ae_classifier_mcc", _ae_mcc)
                    for _k, _v in batch_metrics.items():
                        mlflow.log_metric(_k, _v)
            except Exception:
                pass

        # W&B: final trial metrics then close run
        if _wandb_run is not None:
            try:
                _wandb_run.log({
                    **{f"trial/{_k}": _v for _k, _v in split_metrics.items()},
                    "trial/head_cv_train_mcc": head_cv_train_mcc,
                    "trial/ae_classifier_mcc": _ae_mcc,
                    "trial/head_type": head_type,
                    **{f"trial/{_k}": _v for _k, _v in batch_metrics.items()},
                })
                _wandb_run.finish()
            except Exception:
                pass

        # Per-trial CSV run log
        try:
            import csv as _csv
            _log_dir  = getattr(args, "run_log_dir", "logs/head_sweep")
            os.makedirs(_log_dir, exist_ok=True)
            _csv_path = os.path.join(_log_dir, f"{args.exp_id}_trial_log.csv")
            _row = {
                "trial":        trial.number,
                "timestamp":    datetime.now().isoformat(timespec="seconds"),
                **split_metrics,
                "head_cv_train_mcc": head_cv_train_mcc,
                "ae_classifier_mcc": _ae_mcc,
                **batch_metrics,
                "head_type":    head_type,
                "dloss":        getattr(args, "dloss", ""),
                "class_triplet": int(getattr(args, "class_triplet", False)),
                "variational":  int(getattr(args, "variational", False)),
                **{k: v for k, v in ae_params.items()},
                **{f"head_{k}": v for k, v in head_params.items()},
            }
            _write_header = not os.path.exists(_csv_path)
            with open(_csv_path, "a", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(_row.keys()))
                if _write_header:
                    _w.writeheader()
                _w.writerow(_row)
        except Exception:
            pass

        # Free GPU memory
        del ae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return valid_mcc if not np.isnan(valid_mcc) else float("-inf")

    # ------------------------------------------------------------------
    def run_sweep(self, n_trials: int = 100, study_name: Optional[str] = None,
                  storage: Optional[str] = None,
                  direction: str = "maximize") -> "optuna.Study":
        """Launch the Optuna sweep and return the study."""
        if not _HAS_OPTUNA:
            raise ImportError("optuna is required: pip install optuna")

        study_name = study_name or f"bernn_head_sweep_{self.args.exp_id}"
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=max(10, n_trials // 10))
        pruner  = optuna.pruners.MedianPruner(n_warmup_steps=5)

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        _sweep_label = (study_name or f"bernn_{getattr(self.args, 'exp_id', '?')}")[:30]

        def _trial_cb(study_cb, trial_cb):
            if trial_cb.state.name not in ("COMPLETE", "PRUNED"):
                return
            cv    = trial_cb.value if trial_cb.value is not None else float("nan")
            head  = trial_cb.user_attrs.get("head_type", "?")
            test_m = trial_cb.user_attrs.get("test_mcc", float("nan"))
            ae_m = trial_cb.user_attrs.get("ae_classifier_mcc", float("-inf"))
            try:
                best = study_cb.best_value
            except Exception:
                best = float("nan")
            ae_str   = f"{ae_m:+.3f}" if ae_m not in (float("-inf"), float("inf")) else "-inf"
            test_str = f"{test_m:.3f}" if test_m == test_m else "nan"  # nan check
            print(
                f"[{_sweep_label}] trial {trial_cb.number+1:4d}/{n_trials}  "
                f"valid_mcc={cv:7.4f}  head={head:<22s}"
                f"test={test_str}  ae_cls={ae_str}  best={best:.4f}",
                flush=True,
            )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            gc_after_trial=True,
            catch=(Exception,),
            callbacks=[_trial_cb],
        )
        return study


# ---------------------------------------------------------------------------
# Result summary helper
# ---------------------------------------------------------------------------

def print_study_summary(study: "optuna.Study") -> None:
    print("\n" + "=" * 60)
    print("BERNN Head Sweep – Best trial")
    print("=" * 60)
    t = study.best_trial
    print(f"  Trial #{t.number}  valid_mcc = {t.value:.4f}")
    print(f"  head_type     = {t.user_attrs.get('head_type', 'unknown')}")
    print(f"  test_mcc      = {t.user_attrs.get('test_mcc', 'n/a')}")
    print(f"  ae_classifier_mcc = {t.user_attrs.get('ae_classifier_mcc', 'n/a')}")
    print("\nBest params:")
    for k, v in t.params.items():
        print(f"    {k}: {v}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERNN AE + head type Optuna sweep (cv=3 valid_mcc objective)"
    )
    # Data
    parser.add_argument("--dataset",        type=str, default="prostate")
    parser.add_argument("--path",           type=str, default="./data/")
    parser.add_argument("--csv_file",       type=str, default="unique_genes.csv")
    parser.add_argument("--features_to_keep", type=str, default="features_proteins.csv")
    parser.add_argument("--bad_batches",    type=str, default="")
    parser.add_argument("--remove_zeros",   type=int, default=1)
    parser.add_argument("--groupkfold",     type=int, default=1)
    parser.add_argument("--log1p",          type=int, default=1)
    parser.add_argument("--pool",           type=int, default=0,
                        help="Use pooled/QC samples (custom get_data pool branch)")
    parser.add_argument("--zinb",           type=int, default=0,
                        help="ZINB reconstruction (used by amide/mice getters)")
    # Model
    parser.add_argument("--dloss",          type=str, default="inverseTriplet",
                        choices=["revDANN", "DANN", "inverseTriplet", "revTriplet", "normae", "no"])
    parser.add_argument("--class_triplet",  type=int, default=0,
                        help="Add a class-based triplet loss on embeddings (combinable with dloss)")
    parser.add_argument("--class_triplet_w", type=float, default=1.0,
                        help="Weight of the class-based triplet loss")
    parser.add_argument("--variational",    type=int, default=0)
    parser.add_argument("--tied_weights",   type=int, default=0)
    parser.add_argument("--rec_loss",       type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--device",         type=str, default="cuda:0")
    # Accepted for CLI-parity with train_ae_then_classifier_holdout.py
    parser.add_argument("--train_after_warmup", type=int, default=0)
    parser.add_argument("--bdisc",          type=int, default=1)
    parser.add_argument("--use_mapping",    type=int, default=1)
    parser.add_argument("--kan",            type=int, default=0)
    # Training
    parser.add_argument("--n_epochs",       type=int, default=200)
    parser.add_argument("--early_stop",     type=int, default=30)
    parser.add_argument("--bs",             type=int, default=32)
    parser.add_argument("--n_cv",           type=int, default=3,
                        help="Number of CV folds for head evaluation (same as Optuna sweep criterion)")
    # Sweep
    parser.add_argument("--n_trials",       type=int, default=100)
    parser.add_argument("--n_repeats",      type=int, default=1,
                        help="Accepted for CLI-parity; the sweep evaluates via n_cv folds")
    parser.add_argument("--exp_id",         type=str, default="bernn_head_sweep")
    parser.add_argument("--storage",        type=str, default=None,
                        help="Optuna storage URL, e.g. sqlite:///sweep.db")
    parser.add_argument("--study_name",     type=str, default=None)
    # Logging
    parser.add_argument("--log_mlflow",     type=int, default=0)
    parser.add_argument("--log_wandb",      type=int, default=0,
                        help="Enable Weights & Biases logging (0/1)")
    parser.add_argument("--wandb_project",  type=str, default="bernn_head_sweep",
                        help="W&B project name")
    parser.add_argument("--wandb_entity",   type=str, default="",
                        help="W&B entity/team (leave empty for default)")
    parser.add_argument("--run_log_dir",    type=str, default="logs/head_sweep",
                        help="Directory for per-trial CSV log files")

    args = parser.parse_args()
    args.variational  = bool(args.variational)
    args.tied_weights = bool(args.tied_weights)
    args.groupkfold   = bool(args.groupkfold)
    args.log1p        = bool(args.log1p)
    args.remove_zeros = bool(args.remove_zeros)
    args.pool         = bool(args.pool)
    args.zinb         = bool(args.zinb)
    args.class_triplet = bool(args.class_triplet)

    if not torch.cuda.is_available() or args.device.startswith("cpu"):
        args.device = "cpu"

    # Load data — getters read options off the args object and dispatch on
    # args.dataset (alzheimer/amide/mice → dedicated; else generic custom CSV).
    data, unique_labels, unique_batches = load_data_for_args(args.path, args)

    # Optional: create MLflow experiment
    if args.log_mlflow:
        try:
            mlflow.create_experiment(args.exp_id)
        except Exception:
            pass

    trainer = AEHeadSweepTrainer(
        args=args,
        path=args.path,
        unique_labels=unique_labels,
        unique_batches=unique_batches,
        data=data,
        n_cv=args.n_cv,
    )

    study = trainer.run_sweep(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
    )

    print_study_summary(study)

    # Save best params to JSON
    out_path = f"logs/head_sweep/{args.exp_id}_best_params.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best = {
        "best_valid_mcc": study.best_value,
        "best_params":       study.best_trial.params,
        "best_head_type":    study.best_trial.user_attrs.get("head_type"),
        "test_mcc":          study.best_trial.user_attrs.get("test_mcc"),
        "ae_classifier_mcc": study.best_trial.user_attrs.get("ae_classifier_mcc"),
    }
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest params saved to: {out_path}")
