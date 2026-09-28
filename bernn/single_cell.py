"""Single-cell adapters for BERNN.

The functions in this module keep the core BERNN trainer matrix-based while
providing a thin AnnData-facing API suitable for scRNA-seq benchmarks.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .config.training_config import TrainingConfig
from .dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout


def _select_feature_indices(adata: Any, n_hvg: Optional[int]) -> np.ndarray:
    """Return feature indices, preferring OpenProblems hvg_score metadata."""
    n_vars = int(adata.shape[1])
    if n_hvg is None or int(n_hvg) <= 0 or int(n_hvg) >= n_vars:
        return np.arange(n_vars, dtype=int)

    n_hvg = int(n_hvg)
    var = adata.var
    if "hvg_score" in var:
        scores = pd.to_numeric(var["hvg_score"], errors="coerce").to_numpy()
        scores = np.nan_to_num(scores, nan=-np.inf)
        return np.argsort(scores)[::-1][:n_hvg].astype(int)

    if "hvg" in var:
        hvg = np.flatnonzero(np.asarray(var["hvg"], dtype=bool))
        if len(hvg) >= n_hvg:
            return hvg[:n_hvg].astype(int)

    return np.arange(n_hvg, dtype=int)


def _dense_frame(adata: Any, feature_idx: np.ndarray, layer: str) -> pd.DataFrame:
    """Extract an AnnData matrix as a float32 DataFrame without gene-name assumptions."""
    if layer in ("X", None):
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"AnnData layer {layer!r} is not available")
        matrix = adata.layers[layer]

    matrix = matrix[:, feature_idx]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D expression matrix, got shape {values.shape}")

    # Integer column ids are deliberate: duplicated gene symbols must never make
    # BERNN's fitted-column alignment ambiguous at transform time.
    return pd.DataFrame(values, columns=np.arange(values.shape[1]))


def _monitor_indices(
    labels: np.ndarray,
    batches: np.ndarray,
    fraction: float,
    max_cells: int,
    random_state: int,
) -> np.ndarray:
    """Small deterministic monitor subset while keeping every cell in training."""
    n = len(labels)
    if n == 0:
        return np.array([], dtype=int)

    target = max(32, int(np.ceil(n * max(float(fraction), 0.0))))
    target = min(n, max(1, int(max_cells)), target)
    rng = np.random.default_rng(random_state)
    selected = set(rng.choice(n, size=target, replace=False).tolist())

    # Ensure the monitor contains every observed cell type and batch at least once.
    for values in (labels, batches):
        for value in np.unique(values):
            selected.add(int(np.flatnonzero(values == value)[0]))

    return np.asarray(sorted(selected), dtype=int)


def fit_transform_anndata(
    adata: Any,
    *,
    layer: str = "normalized",
    batch_key: str = "batch",
    label_key: str = "cell_type",
    n_hvg: Optional[int] = 2000,
    dloss: str = "inverseTriplet",
    n_epochs: int = 200,
    warmup: int = 20,
    batch_size: int = 256,
    n_layers: int = 2,
    layer1: int = 256,
    scaler: str = "standard",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    margin: float = 1.0,
    smoothing: float = 0.1,
    nu: float = 1.0,
    validation_fraction: float = 0.05,
    max_validation_cells: int = 10_000,
    random_state: int = 1,
    num_workers: int = 0,
    device: Optional[str] = None,
    return_trainer: bool = False,
):
    """Fit BERNN to a labelled AnnData object and return a batch-integrated embedding.

    All cells are used for training. A small overlapping monitor subset is passed
    to BERNN for validation/test bookkeeping, avoiding BERNN's legacy behavior of
    triplicating the full matrix into train/valid/test monitor splits.
    """
    if batch_key not in adata.obs:
        raise KeyError(f"AnnData obs is missing required batch key {batch_key!r}")
    if label_key not in adata.obs:
        raise KeyError(f"AnnData obs is missing required label key {label_key!r}")

    feature_idx = _select_feature_indices(adata, n_hvg)
    frame = _dense_frame(adata, feature_idx, layer)
    labels = adata.obs[label_key].astype(str).to_numpy()
    batches = adata.obs[batch_key].astype(str).to_numpy()

    if len(frame) != len(labels) or len(frame) != len(batches):
        raise ValueError("Expression rows, cell labels, and batch labels must have equal length")
    if len(np.unique(batches)) < 2:
        raise ValueError("BERNN single-cell integration requires at least two batches")
    if len(np.unique(labels)) < 2:
        raise ValueError("BERNN supervised integration requires at least two cell types")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_id = str(getattr(adata, "uns", {}).get("dataset_id", "single_cell"))
    config = TrainingConfig(
        optimize_hyperparams=False,
        dloss=dloss,
        class_triplet=False,
        variational=False,
        kan=False,
        n_layers=int(n_layers),
        layer1=int(layer1),
        tied_weights=False,
        use_mapping=True,
        rec_loss="l1",
        scaler=scaler,
        log1p=False,
        use_l1=True,
        prune_network=False,
        update_grid=False,
        n_epochs=int(n_epochs),
        warmup=int(warmup),
        n_repeats=1,
        bs=int(batch_size),
        num_workers=int(num_workers),
        groupkfold=True,
        device=device,
        dataset=dataset_id,
        exp_id="bernn_single_cell",
    )

    # Stable non-HPO parameters used by BERNN when _train(params=None).
    config.lr = float(learning_rate)
    config.wd = float(weight_decay)
    config.dropout = float(dropout)
    config.margin = float(margin)
    config.smoothing = float(smoothing)
    config.nu = float(nu)
    config.thres = 0.0
    config.gamma = 0.1
    config.beta = 0.1

    trainer = TrainAEClassifierHoldout(
        config=config,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=True,
        pools=False,
    )

    monitor_idx = _monitor_indices(
        labels,
        batches,
        fraction=validation_fraction,
        max_cells=max_validation_cells,
        random_state=random_state,
    )
    trainer.fit(
        frame,
        labels,
        X_valid=frame.iloc[monitor_idx],
        y_valid=labels[monitor_idx],
        X_test=frame.iloc[monitor_idx],
        y_test=labels[monitor_idx],
        groups_train=batches,
        groups_valid=batches[monitor_idx],
        groups_test=batches[monitor_idx],
    )
    embedding = trainer.transform(frame, groups_test=batches).astype(np.float32, copy=False)

    trainer.single_cell_feature_indices_ = feature_idx
    trainer.single_cell_batch_key_ = batch_key
    trainer.single_cell_label_key_ = label_key
    trainer.single_cell_input_layer_ = layer

    if return_trainer:
        return embedding, trainer
    return embedding
