"""Unit tests for model utilities: losses, get_optimizer, to_categorical, get_empty_dicts,
get_loaders, and MSDataset3."""
import pytest
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace

from bernn.dl.models.pytorch.utils.losses import (
    softmax_mse_loss,
    softmax_kl_loss,
    symmetric_mse_loss,
    get_losses,
)
from bernn.dl.models.pytorch.utils.utils import (
    get_optimizer,
    to_categorical,
    get_empty_dicts,
    get_empty_traces,
)
from bernn.dl.models.pytorch.utils.dataset import MSDataset3


# ────────────────────────────────────────────────────────────
# losses.py
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_softmax_mse_loss_same_inputs_near_zero():
    x = torch.randn(4, 3)
    loss = softmax_mse_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_softmax_mse_loss_different_inputs_positive():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    loss = softmax_mse_loss(a, b)
    assert loss.item() > 0


@pytest.mark.unit
def test_softmax_kl_loss_same_inputs_near_zero():
    x = torch.randn(4, 3)
    loss = softmax_kl_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


@pytest.mark.unit
def test_symmetric_mse_loss_same_inputs_zero():
    x = torch.randn(4, 5)
    loss = symmetric_mse_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_symmetric_mse_loss_symmetric():
    a = torch.randn(4, 3)
    b = torch.randn(4, 3)
    assert symmetric_mse_loss(a, b).item() == pytest.approx(symmetric_mse_loss(b, a).item(), rel=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("rec_loss,dloss,classif_loss", [
    ("l1", "no", "ce"),
    ("mse", "revTriplet", "ce"),
    ("l1", "inverseTriplet", "ce"),
])
def test_get_losses_returns_four_objects(rec_loss, dloss, classif_loss):
    args = SimpleNamespace(rec_loss=rec_loss, dloss=dloss, classif_loss=classif_loss)
    result = get_losses("standard", 0.0, 1.0, args)
    assert len(result) == 4


@pytest.mark.unit
def test_get_losses_binarize_uses_bce():
    """When scale=='binarize', mseloss should be BCELoss."""
    import torch.nn as nn
    args = SimpleNamespace(rec_loss="mse", dloss="no", classif_loss="ce")
    _, _, mseloss, _ = get_losses("binarize", 0.0, 1.0, args)
    assert isinstance(mseloss, nn.BCELoss)


# ────────────────────────────────────────────────────────────
# utils.py — get_optimizer
# ────────────────────────────────────────────────────────────

def _tiny_model():
    return torch.nn.Linear(4, 2)


@pytest.mark.unit
@pytest.mark.parametrize("opt_type", ["adam", "radam", "adamw", "rmsprop", "sgd"])
def test_get_optimizer_creates_optimizer(opt_type):
    model = _tiny_model()
    opt = get_optimizer(model, 1e-3, 1e-5, opt_type)
    assert isinstance(opt, torch.optim.Optimizer)


@pytest.mark.unit
def test_get_optimizer_lr_is_set():
    model = _tiny_model()
    lr = 5e-4
    opt = get_optimizer(model, lr, 0.0, "adam")
    assert opt.param_groups[0]["lr"] == pytest.approx(lr)


# ────────────────────────────────────────────────────────────
# utils.py — to_categorical
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_to_categorical_shape():
    y = torch.tensor([0, 1, 2])
    result = to_categorical(y, 3)
    assert result.shape == (3, 3)


@pytest.mark.unit
def test_to_categorical_one_hot():
    y = torch.tensor([0, 2])
    result = to_categorical(y, 3)
    assert result[0, 0].item() == 1
    assert result[1, 2].item() == 1
    assert result[0, 1].item() == 0


# ────────────────────────────────────────────────────────────
# utils.py — get_empty_dicts / get_empty_traces
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_get_empty_dicts_has_expected_splits():
    # get_empty_dicts returns (values, best_values, best_lists, best_traces)
    result = get_empty_dicts()
    assert isinstance(result, tuple)
    values = result[0]
    for split in ["train", "valid", "test"]:
        assert split in values


@pytest.mark.unit
def test_get_empty_traces_is_dict():
    # get_empty_traces returns (lists, traces) tuple
    result = get_empty_traces()
    assert isinstance(result, tuple)
    assert len(result) == 2
    lists = result[0]
    assert isinstance(lists, dict)


# ────────────────────────────────────────────────────────────
# MSDataset3 — basic construction and __getitem__
# ────────────────────────────────────────────────────────────

@pytest.fixture
def ms_dataset():
    n, f = 20, 10
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"ft{i}" for i in range(f)])
    # meta must be a DataFrame so MSDataset3's .iloc branch succeeds
    meta = pd.DataFrame(np.zeros((n, 1)), columns=["meta0"])
    names = np.array([f"s{i}" for i in range(n)])
    labels = np.array([f"l{i % 2}" for i in range(n)])
    batches = np.array([f"b{i % 3}" for i in range(n)])
    sets = np.zeros(n, dtype=int)
    return MSDataset3(
        data, meta, names=names, labels=labels, batches=batches, sets=sets,
        random_recs=False, triplet_dloss=False
    )


@pytest.mark.unit
def test_msdataset3_len(ms_dataset):
    assert len(ms_dataset) == 20


@pytest.mark.unit
def test_msdataset3_getitem_returns_tensor(ms_dataset):
    item = ms_dataset[0]
    # item is a tuple; first element should be a tensor-like
    assert item is not None
