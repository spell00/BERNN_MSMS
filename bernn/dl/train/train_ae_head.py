#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ae_head.py
================
Train an AE encoder with domain adaptation (DANN / inverseTriplet / normae)
**without** any neural-network classification head.

Every epoch the training loop is:
  1. Forward pass: enc, rec = AE(inputs)
  2. Loss: reconstruction + gamma * domain-alignment  (NO classification loss)
  3. Backward + optimizer step
  4. Extract train/valid embeddings (no grad)
  5. Fit the chosen sklearn/XGBoost head on train embeddings
  6. Score valid embeddings with the head → track best valid MCC

Head type is a first-class hyperparameter (shown first in the params dict).
No warmup phase — the domain-alignment loss runs from epoch 0.

Two-stage Optuna sweep (train_ae_head_sweep.py):
  Stage 1: sweep head_type + AE params   → find best head family
  Stage 2: fix head_type, sweep head hyperparameters + AE params
"""

from __future__ import annotations

import os
import uuid
import warnings
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef as MCC

from bernn.dl.train.train_ae import TrainAE
from bernn.dl.train.head_classifier import (
    HEAD_TYPES,
    HEAD_TYPES_NO_XGB,
    fit_and_score_head,
)
from bernn.dl.models.pytorch.utils.utils import (
    get_optimizer,
    get_empty_traces,
)
from bernn.dl.models.pytorch.utils.dataset import get_loaders, get_loaders_no_pool

try:
    import xgboost  # noqa: F401
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# --------------------------------------------------------------------------
# Head-parameter key prefixes — extracted from the flat params dict
# --------------------------------------------------------------------------
_HEAD_PARAM_PREFIXES = {
    "xgboost":             "xgb_",
    "random_forest":       "rf_",
    "linear_svc":          "linsvc_",
    "svc_rbf":             "svcrbf_",
    "logistic_regression": "logreg_",
    "knn":                 "knn_",
    "gradient_boosting":   "gbc_",
    "prototype_mean":      "proto_mean_",
    "prototype_kmeans":    "proto_km_",
}


def _extract_head_params(params: Dict[str, Any], head_type: str) -> Dict[str, Any]:
    """Pull head-specific keys out of the flat Optuna/Ax params dict."""
    prefix = _HEAD_PARAM_PREFIXES.get(head_type, f"{head_type}_")
    result = {}
    for k, v in params.items():
        if k.startswith(prefix):
            result[k[len(prefix):]] = v
    # Common aliases
    if head_type == "prototype_mean" and "proto_metric" not in result:
        result["proto_metric"] = params.get("proto_mean_metric", "cosine")
    if head_type == "prototype_kmeans":
        if "proto_components" not in result:
            result["proto_components"] = params.get("proto_km_k", 1)
        if "proto_metric" not in result:
            result["proto_metric"] = params.get("proto_km_metric", "cosine")
    return result


# --------------------------------------------------------------------------
# Embedding extraction helper (no grad, eval mode)
# --------------------------------------------------------------------------

@torch.no_grad()
def _get_embeddings_and_labels(
    ae: nn.Module,
    loader,
    device: str,
    unique_labels: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, integer-encoded labels) for the full loader."""
    ae.eval()
    label_encoder = LabelEncoder().fit(unique_labels)
    all_enc, all_cats = [], []
    for batch in loader:
        inputs, _names, labels, domain, to_rec = batch[:5]
        inputs = inputs.to(device).float()
        to_rec  = to_rec.to(device).float() if to_rec is not None else inputs
        enc, _rec, _zinb, _kld = ae(inputs, to_rec, domain, sampling=False)
        all_enc.append(enc.float().cpu().numpy())
        cats = labels.detach().int().cpu().numpy()
        all_cats.append(cats)
    if not all_enc:
        return np.empty((0,)), np.empty((0,), dtype=int)
    X = np.concatenate(all_enc, axis=0)
    y = np.concatenate(all_cats, axis=0)
    return X, y


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------

class TrainAEHead(TrainAE):
    """
    AE encoder + interchangeable sklearn/XGBoost classification head.

    The head is fit from scratch every epoch on the current frozen embeddings.
    No neural-network classifier, no warmup phase.

    Parameters (in params dict, head_type shown first)
    ---------------------------------------------------
    head_type      : one of HEAD_TYPES (e.g. 'xgboost', 'linear_svc', ...)
    lr, wd, nu     : AE optimizer hyperparameters
    dropout, smoothing, margin, gamma, beta : standard BERNN AE knobs
    layer1, n_layers : AE architecture
    scaler         : input scaling
    <prefix>_*     : head-specific hyperparameters (see _HEAD_PARAM_PREFIXES)
    """

    def train(self, params: Dict[str, Any]) -> float:
        """
        Optuna/Ax objective.

        Returns best valid MCC achieved by the sklearn/XGBoost head across all
        epochs.  Also stored in ``self.best_mcc``.
        """
        # ---- 0. Resolve head type (FIRST in output) ----
        head_type = str(params.get("head_type", "linear_svc"))
        head_params = _extract_head_params(params, head_type)

        # Print head_type FIRST so it's the first visible line in the log
        print(f"\nhead_type: {head_type}")
        params = self.make_params(params)

        args = self.args
        start = datetime.now()

        # ---- 1. Build AE (no classifier needed but AutoEncoder2 carries one;
        #         we simply never call it or backprop through it) ----
        from bernn.dl.models.pytorch.aedann import AutoEncoder2 as AutoEncoder
        from bernn.dl.models.pytorch.aedann import ReverseLayerF

        sceloss, celoss, mseloss, triplet_loss = self.get_losses(
            params.get("scaler", "standard"),
            params.get("smoothing", 0.0),
            params.get("margin", 1.0),
            args.dloss,
        )

        gamma    = params.get("gamma", 0.0)
        beta     = params.get("beta",  0.0)
        lr       = params.get("lr",    1e-3)
        wd       = params.get("wd",    1e-5)

        # load data + make loaders
        self.make_samples_weights()
        try:
            loaders = get_loaders(
                self.data,
                getattr(args, "random_recs", 0),
                self.samples_weights,
                args.dloss,
                None, None,
                bs=args.bs,
            )
        except Exception:
            loaders = get_loaders_no_pool(
                self.data,
                getattr(args, "random_recs", 0),
                self.samples_weights,
                args.dloss,
                None, None,
                bs=args.bs,
            )

        # Build fresh AE per rep
        self.load_autoencoder()  # sets self.ae via the parent class
        ae = self.ae
        optimizer_ae = get_optimizer(ae, lr, wd, "adam")

        # Freeze the classifier layers — we never train them
        if hasattr(ae, "classifier"):
            for p in ae.classifier.parameters():
                p.requires_grad = False

        best_valid_mcc = float("-inf")
        best_head      = None
        early_stop_counter = 0
        early_stop = getattr(args, "early_stop", 30)

        self.complete_log_path = f"logs/ae_head/{str(uuid.uuid4())}"
        os.makedirs(self.complete_log_path, exist_ok=True)
        print(f"See results using: tensorboard --logdir={self.complete_log_path} --port=6006")

        # ---- 2. Per-epoch training loop ----
        for epoch in range(args.n_epochs):
            if early_stop_counter >= early_stop:
                print(f"  Early stop at epoch {epoch}")
                break

            # --- 2a. AE + domain alignment (warmup_loop pattern, no classifier) ---
            self.warmup_loop(
                optimizer_ae, None, ae, celoss,
                loaders.get("all", loaders.get("train")),
                triplet_loss, mseloss,
                warmup=True, epoch=epoch,
                optimizer_b=None, values={}, loggers={},
                loaders=loaders, run=None,
                mapping=getattr(args, "use_mapping", True),
            )

            # --- 2b. Extract embeddings (eval, no grad) ---
            X_tr, y_tr = _get_embeddings_and_labels(
                ae, loaders["train"], args.device, self.unique_labels
            )
            X_vl, y_vl = _get_embeddings_and_labels(
                ae, loaders["valid"], args.device, self.unique_labels
            )
            if len(X_tr) == 0 or len(X_vl) == 0:
                continue
            if len(np.unique(y_tr)) < 2:
                continue

            # --- 2c. Fit head + score ---
            try:
                head, train_mcc, valid_mcc = fit_and_score_head(
                    X_tr, y_tr, X_vl, y_vl, head_type, head_params
                )
            except Exception as exc:
                print(f"  [head] epoch {epoch}: {type(exc).__name__}: {exc}")
                continue

            print(
                f"  Epoch {epoch:4d} | head={head_type} | "
                f"train_mcc={train_mcc:.4f} | valid_mcc={valid_mcc:.4f}"
            )

            if valid_mcc > best_valid_mcc:
                best_valid_mcc = valid_mcc
                best_head      = head
                early_stop_counter = 0
                torch.save(
                    ae.state_dict(),
                    os.path.join(self.complete_log_path, "best_ae.pth"),
                )
                print(f"  *** NEW BEST valid_mcc={valid_mcc:.4f} (head={head_type}) ***")
            else:
                early_stop_counter += 1

        # Expose for the scoring harness + Optuna
        self.best_mcc      = best_valid_mcc
        self.best_head     = best_head
        self.best_head_type = head_type

        print(
            f"\nDuration: {datetime.now() - start} | "
            f"best valid_mcc={best_valid_mcc:.4f} | head={head_type}"
        )
        return best_valid_mcc


# --------------------------------------------------------------------------
# Two-stage Optuna sweep
# --------------------------------------------------------------------------

def _ae_head_params_stage1(trial, args) -> Dict[str, Any]:
    """
    Stage 1: sweep head_type + AE hyperparameters.
    Head-specific hyperparameters are suggested conditionally.
    """
    # Head type FIRST — shows up first in params dict print
    available = HEAD_TYPES if _HAS_XGB else HEAD_TYPES_NO_XGB
    head_type = trial.suggest_categorical("head_type", available)

    p: Dict[str, Any] = {"head_type": head_type}

    # AE hyperparameters
    p["lr"]        = trial.suggest_float("lr",        1e-4, 1e-2, log=True)
    p["wd"]        = trial.suggest_float("wd",        1e-8, 1e-4, log=True)
    p["dropout"]   = trial.suggest_float("dropout",   0.0,  0.5)
    p["smoothing"] = trial.suggest_float("smoothing", 0.0,  0.2)
    p["margin"]    = trial.suggest_float("margin",    0.0,  10.0)
    p["scaler"]    = trial.suggest_categorical("scaler", ["robust", "standard"])
    p["layer1"]    = trial.suggest_int("layer1", 32, 512, log=True)
    p["n_layers"]  = trial.suggest_int("n_layers", 1, 3)
    p["gamma"]     = 0.0
    p["beta"]      = 0.0
    if getattr(args, "dloss", "inverseTriplet") in {
        "revTriplet", "revDANN", "DANN", "inverseTriplet", "normae"
    }:
        p["gamma"] = trial.suggest_float("gamma", 1e-4, 1e2, log=True)
    if getattr(args, "variational", False):
        p["beta"] = trial.suggest_float("beta", 1e-2, 1e2, log=True)

    # Head-specific hyperparameters (conditional on head_type)
    p.update(_suggest_head_hyperparams(trial, head_type))

    return p


def _ae_head_params_stage2(trial, args, head_type: str) -> Dict[str, Any]:
    """
    Stage 2: head_type is FIXED; tune head hyperparameters + AE params.
    """
    p: Dict[str, Any] = {"head_type": head_type}
    p["lr"]        = trial.suggest_float("lr",        1e-4, 1e-2, log=True)
    p["wd"]        = trial.suggest_float("wd",        1e-8, 1e-4, log=True)
    p["dropout"]   = trial.suggest_float("dropout",   0.0,  0.5)
    p["smoothing"] = trial.suggest_float("smoothing", 0.0,  0.2)
    p["margin"]    = trial.suggest_float("margin",    0.0,  10.0)
    p["scaler"]    = trial.suggest_categorical("scaler", ["robust", "standard"])
    p["layer1"]    = trial.suggest_int("layer1", 32, 512, log=True)
    p["n_layers"]  = trial.suggest_int("n_layers", 1, 3)
    p["gamma"]     = 0.0
    p["beta"]      = 0.0
    if getattr(args, "dloss", "inverseTriplet") in {
        "revTriplet", "revDANN", "DANN", "inverseTriplet", "normae"
    }:
        p["gamma"] = trial.suggest_float("gamma", 1e-4, 1e2, log=True)
    p.update(_suggest_head_hyperparams(trial, head_type))
    return p


def _suggest_head_hyperparams(trial, head_type: str) -> Dict[str, Any]:
    """Suggest head-specific hyperparameters, prefixed so they are unambiguous."""
    p: Dict[str, Any] = {}
    if head_type == "xgboost":
        p["xgb_n_estimators"]  = trial.suggest_int("xgb_n_estimators", 50, 500, step=50)
        p["xgb_max_depth"]     = trial.suggest_int("xgb_max_depth", 3, 10)
        p["xgb_learning_rate"] = trial.suggest_float("xgb_learning_rate", 1e-3, 0.5, log=True)
        p["xgb_reg_alpha"]     = trial.suggest_float("xgb_reg_alpha",  1e-8, 10.0, log=True)
        p["xgb_reg_lambda"]    = trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True)
        p["xgb_subsample"]     = trial.suggest_float("xgb_subsample",  0.5, 1.0)
        p["xgb_colsample"]     = trial.suggest_float("xgb_colsample",  0.5, 1.0)
    elif head_type == "random_forest":
        p["rf_n_estimators"]      = trial.suggest_int("rf_n_estimators", 50, 500, step=50)
        p["rf_max_features"]      = trial.suggest_categorical("rf_max_features", ["sqrt", "log2"])
        p["rf_min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 10)
        p["rf_min_samples_leaf"]  = trial.suggest_int("rf_min_samples_leaf",  1, 10)
        p["rf_criterion"]         = trial.suggest_categorical("rf_criterion", ["gini", "entropy"])
    elif head_type == "linear_svc":
        p["linsvc_C"]        = trial.suggest_float("linsvc_C",   1e-3, 1e4, log=True)
        p["linsvc_tol"]      = trial.suggest_float("linsvc_tol", 1e-5, 1e-2, log=True)
        p["linsvc_max_iter"] = trial.suggest_int("linsvc_max_iter", 200, 2000, step=100)
    elif head_type == "svc_rbf":
        p["svcrbf_C"]        = trial.suggest_float("svcrbf_C",   1e-3, 1e3, log=True)
        p["svcrbf_max_iter"] = trial.suggest_int("svcrbf_max_iter", 100, 1000, step=100)
    elif head_type == "logistic_regression":
        p["logreg_C"]        = trial.suggest_float("logreg_C",   1e-3, 1e4, log=True)
        p["logreg_penalty"]  = trial.suggest_categorical("logreg_penalty", ["l1", "l2"])
        p["logreg_max_iter"] = trial.suggest_int("logreg_max_iter", 100, 2000, step=100)
    elif head_type == "knn":
        p["knn_n_neighbors"] = trial.suggest_int("knn_n_neighbors", 1, 31, step=2)
        p["knn_metric"]      = trial.suggest_categorical("knn_metric", ["euclidean", "manhattan", "cosine"])
        p["knn_weights"]     = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
    elif head_type == "gradient_boosting":
        p["gbc_n_estimators"]  = trial.suggest_int("gbc_n_estimators", 50, 300, step=50)
        p["gbc_max_depth"]     = trial.suggest_int("gbc_max_depth", 3, 8)
        p["gbc_learning_rate"] = trial.suggest_float("gbc_learning_rate", 1e-3, 0.5, log=True)
        p["gbc_subsample"]     = trial.suggest_float("gbc_subsample", 0.5, 1.0)
    elif head_type == "prototype_mean":
        p["proto_mean_metric"] = trial.suggest_categorical("proto_mean_metric", ["cosine", "euclidean"])
    elif head_type == "prototype_kmeans":
        p["proto_km_k"]      = trial.suggest_int("proto_km_k", 1, 5)
        p["proto_km_metric"] = trial.suggest_categorical("proto_km_metric", ["cosine", "euclidean"])
    return p


def run_two_stage_sweep(
    trainer: TrainAEHead,
    n_trials_stage1: int = 50,
    n_trials_stage2: int = 50,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
) -> Tuple["optuna.Study", "optuna.Study"]:
    """
    Stage 1: sweep head_type + AE params.
    Stage 2: fix best head_type, sweep head params + AE params.
    Returns (stage1_study, stage2_study).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna is required: pip install optuna")

    args = trainer.args

    # ---- Stage 1 ----
    print("\n" + "=" * 60)
    print("Stage 1: sweeping head_type + AE hyperparameters")
    print("=" * 60)

    s1_name = (study_name or f"ae_head_{args.exp_id}") + "_stage1"
    study1 = optuna.create_study(
        study_name=s1_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=max(10, n_trials_stage1 // 5)),
        storage=storage,
        load_if_exists=True,
    )

    def obj_stage1(trial):
        params = _ae_head_params_stage1(trial, args)
        return trainer.train(params)

    study1.optimize(obj_stage1, n_trials=n_trials_stage1, gc_after_trial=True, catch=(Exception,))

    best_head_type = study1.best_trial.params.get("head_type", "linear_svc")
    print(f"\nStage 1 done. Best head: {best_head_type}  MCC={study1.best_value:.4f}")

    # ---- Stage 2 ----
    print("\n" + "=" * 60)
    print(f"Stage 2: tuning {best_head_type} hyperparameters + AE params")
    print("=" * 60)

    s2_name = (study_name or f"ae_head_{args.exp_id}") + f"_stage2_{best_head_type}"
    study2 = optuna.create_study(
        study_name=s2_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=max(10, n_trials_stage2 // 5)),
        storage=storage,
        load_if_exists=True,
    )

    def obj_stage2(trial):
        params = _ae_head_params_stage2(trial, args, best_head_type)
        return trainer.train(params)

    study2.optimize(obj_stage2, n_trials=n_trials_stage2, gc_after_trial=True, catch=(Exception,))

    print(f"\nStage 2 done. Best {best_head_type} MCC={study2.best_value:.4f}")
    return study1, study2


# --------------------------------------------------------------------------
# Alias for import
# --------------------------------------------------------------------------
TrainAEWithHead = TrainAEHead


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="BERNN AE + sklearn/XGBoost head sweep (no neural classifier)"
    )
    parser.add_argument("--dataset",        type=str, default="prostate")
    parser.add_argument("--path",           type=str, default="./data/")
    parser.add_argument("--csv_file",       type=str, default="unique_genes.csv")
    parser.add_argument("--features_to_keep", type=str, default="features_proteins.csv")
    parser.add_argument("--bad_batches",    type=str, default="")
    parser.add_argument("--remove_zeros",   type=int, default=1)
    parser.add_argument("--groupkfold",     type=int, default=1)
    parser.add_argument("--log1p",          type=int, default=1)
    parser.add_argument("--dloss",          type=str, default="inverseTriplet")
    parser.add_argument("--variational",    type=int, default=0)
    parser.add_argument("--n_epochs",       type=int, default=200)
    parser.add_argument("--early_stop",     type=int, default=30)
    parser.add_argument("--bs",             type=int, default=32)
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument("--n_trials_s1",    type=int, default=50,  help="Stage-1 trials (head_type sweep)")
    parser.add_argument("--n_trials_s2",    type=int, default=50,  help="Stage-2 trials (head param tuning)")
    parser.add_argument("--exp_id",         type=str, default="ae_head_sweep")
    parser.add_argument("--storage",        type=str, default=None)
    parser.add_argument("--study_name",     type=str, default=None)

    args = parser.parse_args()
    args.variational  = bool(args.variational)
    args.groupkfold   = bool(args.groupkfold)
    args.log1p        = bool(args.log1p)
    args.remove_zeros = bool(args.remove_zeros)
    args.use_mapping  = True
    args.kan          = False
    args.use_l1       = False
    args.prune_network = False
    args.tied_weights  = False
    args.n_layers      = 1
    args.layer1        = 256
    args.n_agg         = 1
    args.bdisc         = True
    args.update_grid   = False
    args.warmup        = 0  # no warmup phase

    if not torch.cuda.is_available() or args.device.startswith("cpu"):
        args.device = "cpu"

    from bernn.utils.data_getters import get_data
    data, unique_labels, unique_batches = get_data(
        path=args.path, dataset=args.dataset, csv_file=args.csv_file,
        features_to_keep=args.features_to_keep, bad_batches=args.bad_batches,
        remove_zeros=args.remove_zeros, groupkfold=args.groupkfold, log1p=args.log1p,
    )

    trainer = TrainAEHead(
        args, args.path,
        fix_thres=-1, load_tb=False, log_metrics=False, keep_models=False,
        log_inputs=False, log_plots=False, log_tb=False, log_mlflow=False,
        groupkfold=args.groupkfold, pools=True,
    )
    trainer.data          = data
    trainer.unique_labels  = unique_labels
    trainer.unique_batches = unique_batches

    s1, s2 = run_two_stage_sweep(
        trainer,
        n_trials_stage1=args.n_trials_s1,
        n_trials_stage2=args.n_trials_s2,
        study_name=args.study_name,
        storage=args.storage,
    )

    out = {
        "stage1_best_mcc":       s1.best_value,
        "stage1_best_head_type": s1.best_trial.params.get("head_type"),
        "stage1_best_params":    s1.best_trial.params,
        "stage2_best_mcc":       s2.best_value,
        "stage2_best_params":    s2.best_trial.params,
    }
    out_path = f"logs/head_sweep/{args.exp_id}_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")
