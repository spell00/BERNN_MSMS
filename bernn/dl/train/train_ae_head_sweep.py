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
    ae.eval()
    all_enc, all_labels = [], []
    for batch in loader:
        inputs, _names, labels, domain, to_rec = batch[:5]
        inputs = inputs.to(device).float()
        to_rec  = to_rec.to(device).float()
        enc, _rec, _zinb, _kld = ae(inputs, to_rec, domain, sampling=False)
        all_enc.append(enc.float().cpu().numpy())
        all_labels.extend(labels if isinstance(labels, list) else labels.tolist())
    if not all_enc:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(all_enc, axis=0), np.array(all_labels)


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
    p["margin"]    = trial.suggest_float("margin",   0.0,  10.0)
    p["warmup"]    = trial.suggest_int("warmup",     1,    100)
    p["scaler"]    = trial.suggest_categorical("scaler", ["robust", "standard", "binarize"])
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
            triplet_loss = nn.TripletMarginLoss(0, p=2, swap=False)
        return sceloss, celoss, mseloss, triplet_loss

    # ------------------------------------------------------------------
    def _train_ae(self, ae, params, loaders):
        """Run the warmup + classification training loop. Returns best_mcc."""
        from bernn.dl.models.pytorch.utils.utils import get_optimizer, get_empty_traces

        args = self.args
        nu, lr, wd = params["nu"], params["lr"], params["wd"]
        optimizer_type = "adam"
        optimizer_ae = get_optimizer(ae, lr, wd, optimizer_type)
        optimizer_c  = get_optimizer(ae.classifier, nu * lr, wd, optimizer_type)

        sceloss, celoss, mseloss, triplet_loss = self._get_losses(ae, params)
        dloss = getattr(args, "dloss", "inverseTriplet")

        import torch.nn as nn
        best_valid_mcc = float("-inf")
        early_stop_counter = 0
        early_stop = getattr(args, "early_stop", 30)

        for epoch in range(getattr(args, "n_epochs", 200)):
            # ---- TRAIN ----
            ae.train()
            for batch in loaders.get("train", []):
                inputs, _names, labels, domain, to_rec, not_to_rec, \
                    pos_to_rec, neg_to_rec, pos_batch, neg_batch, _ = batch[:11] if len(batch) >= 11 else (*batch, *([None]*max(0, 11-len(batch))))
                if inputs is None:
                    break
                inputs = inputs.to(args.device).float()
                to_rec = to_rec.to(args.device).float() if to_rec is not None else inputs

                optimizer_ae.zero_grad()
                optimizer_c.zero_grad()
                enc, rec, zinb_loss, kld = ae(inputs, to_rec, domain, sampling=True)
                rec_val = rec["mean"] if isinstance(rec, dict) else rec

                if enc.abs().sum() == 0:
                    continue

                # Reconstruction loss
                rec_loss_val = mseloss(rec_val, to_rec)

                # Domain alignment
                gamma = params.get("gamma", 0.0)
                d_loss = torch.tensor(0.0, device=args.device)
                if gamma > 0 and dloss != "no":
                    reverse = ReverseLayerF.apply(enc, gamma)
                    if dloss in ["revTriplet", "inverseTriplet"]:
                        if pos_batch is not None and neg_batch is not None:
                            pos_batch = pos_batch.to(args.device).float()
                            neg_batch = neg_batch.to(args.device).float()
                            pos_enc, _, _, _ = ae(pos_batch, pos_batch, domain, sampling=True)
                            neg_enc, _, _, _ = ae(neg_batch, neg_batch, domain, sampling=True)
                            if dloss == "revTriplet":
                                d_loss = triplet_loss(
                                    ReverseLayerF.apply(enc, 1),
                                    ReverseLayerF.apply(pos_enc, 1),
                                    ReverseLayerF.apply(neg_enc, 1),
                                )
                            else:
                                d_loss = triplet_loss(enc, pos_enc, neg_enc)

                # Classifier loss
                cats = to_categorical(labels, len(self.unique_labels)).to(args.device)
                preds = ae.classifier(enc)
                c_loss = sceloss(preds, cats.argmax(1))

                loss = rec_loss_val + gamma * d_loss + c_loss
                # Additive class-based triplet loss on embeddings
                if getattr(args, "class_triplet", False) and pos_to_rec is not None:
                    loss = loss + getattr(args, "class_triplet_w", 1.0) * compute_class_triplet(
                        ae, enc, pos_to_rec, neg_to_rec, domain, args.device,
                        margin=params.get("margin", 1.0), mapping=False,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                optimizer_ae.step()
                optimizer_c.step()

            # ---- VALID ----
            ae.eval()
            all_preds, all_labels_val = [], []
            with torch.no_grad():
                for batch in loaders.get("valid", []):
                    inputs, _names, labels, domain, to_rec = batch[:5]
                    inputs = inputs.to(args.device).float()
                    to_rec = to_rec.to(args.device).float() if to_rec is not None else inputs
                    enc, _, _, _ = ae(inputs, to_rec, domain, sampling=False)
                    preds = ae.classifier(enc).argmax(1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels_val.extend(labels if isinstance(labels, list) else labels.tolist())

            if len(np.unique(all_labels_val)) > 1 and all_preds:
                from sklearn.preprocessing import LabelEncoder as LE
                le = LE().fit(self.unique_labels)
                try:
                    preds_enc  = np.array(all_preds)
                    labels_enc = le.transform(all_labels_val)
                    from sklearn.metrics import matthews_corrcoef
                    valid_mcc = matthews_corrcoef(labels_enc, preds_enc)
                except Exception:
                    valid_mcc = float("-inf")

                if valid_mcc > best_valid_mcc:
                    best_valid_mcc = valid_mcc
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop:
                        break

        return best_valid_mcc

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
        try:
            loaders = get_loaders(data, False, None, dloss, None, None, bs, args.device)
        except Exception:
            loaders = get_loaders_no_pool(data, False, None, dloss, None, None, bs, args.device)

        # Step 1: train AE encoder
        try:
            _ae_mcc = self._train_ae(ae, ae_params, loaders)
        except Exception as exc:
            trial.set_user_attr("ae_error", str(exc))
            return float("-inf")

        # Step 2: freeze encoder, extract embeddings
        ae.eval()
        for param in ae.enc.parameters():
            param.requires_grad = False
        for param in ae.dec.parameters():
            param.requires_grad = False

        try:
            X_train, y_train = extract_embeddings(ae, loaders["train"], args.device)
            X_valid, y_valid = extract_embeddings(ae, loaders["valid"], args.device)
        except Exception as exc:
            trial.set_user_attr("embed_error", str(exc))
            return float("-inf")

        if len(X_train) == 0 or len(X_valid) == 0:
            return float("-inf")

        # Step 3: cv=3 head evaluation on training embeddings
        cv_result = cv_score_head(
            X_train, y_train,
            head_type=head_type,
            head_params=head_params,
            n_splits=self.n_cv,
            random_state=42,
        )
        cv_mcc = cv_result["mean_valid_mcc"]

        # Also compute held-out valid_mcc for logging
        try:
            _, _, held_valid_mcc = fit_and_score_head(
                X_train, y_train, X_valid, y_valid, head_type, head_params
            )
        except Exception:
            held_valid_mcc = float("nan")

        trial.set_user_attr("cv_valid_mcc",     cv_mcc)
        trial.set_user_attr("held_valid_mcc",   held_valid_mcc)
        trial.set_user_attr("ae_valid_mcc",     _ae_mcc)
        trial.set_user_attr("head_type",        head_type)
        trial.set_user_attr("head_params",      json.dumps(head_params))

        if cv_mcc > self.best_valid_mcc:
            self.best_valid_mcc  = cv_mcc
            self.best_head_type  = head_type
            self.best_head_params = head_params

        # MLflow logging (optional)
        if getattr(args, "log_mlflow", False):
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params({**ae_params, "head_type": head_type, **head_params})
                    mlflow.log_metric("cv_valid_mcc",   cv_mcc)
                    mlflow.log_metric("held_valid_mcc", held_valid_mcc)
                    mlflow.log_metric("ae_valid_mcc",   _ae_mcc)
            except Exception:
                pass

        # Free GPU memory
        del ae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cv_mcc if not np.isnan(cv_mcc) else float("-inf")

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

        study.optimize(
            self.objective,
            n_trials=n_trials,
            gc_after_trial=True,
            catch=(Exception,),
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
    print(f"  Trial #{t.number}  cv_valid_mcc = {t.value:.4f}")
    print(f"  head_type     = {t.user_attrs.get('head_type', 'unknown')}")
    print(f"  held_valid_mcc = {t.user_attrs.get('held_valid_mcc', 'n/a')}")
    print(f"  ae_valid_mcc  = {t.user_attrs.get('ae_valid_mcc', 'n/a')}")
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
        "best_cv_valid_mcc": study.best_value,
        "best_params":       study.best_trial.params,
        "best_head_type":    study.best_trial.user_attrs.get("head_type"),
        "held_valid_mcc":    study.best_trial.user_attrs.get("held_valid_mcc"),
        "ae_valid_mcc":      study.best_trial.user_attrs.get("ae_valid_mcc"),
    }
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest params saved to: {out_path}")
