from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bernn.dl.train.train_ae import TrainAE


def test_public_split_metrics_uses_classifier_cats_for_13_classes():
    trainer = TrainAE.__new__(TrainAE)
    encoded = np.arange(13, dtype=int)
    ordered = sorted(encoded, key=str)
    label_map = {label: idx for idx, label in enumerate(ordered)}
    cats = np.array([label_map[label] for label in encoded], dtype=int)

    # For >=11 classes these spaces differ; this is the DKD failure mode.
    assert not np.array_equal(encoded, cats)

    trainer.data = {
        "labels": {"test": encoded.copy()},
        "cats": {"test": cats.copy()},
        "inputs": {"test": pd.DataFrame({"class_id": cats.astype(float)})},
        "batches": {"test": np.zeros(13, dtype=int)},
    }
    trainer.args = SimpleNamespace(scaler=None, device="cpu", use_mapping=True)
    trainer.scaler = None
    trainer.columns = ["class_id"]
    trainer.batch_map_ = {0: 0}
    trainer.unique_batches = np.array([0])
    trainer._label_encoder = SimpleNamespace()  # Must not be used by this scorer.
    trainer.ae = SimpleNamespace(
        enc=SimpleNamespace(eval=lambda: None),
        classifier=SimpleNamespace(eval=lambda: None),
    )
    trainer._predict_logits_from_batch = lambda data, batch_ids=None: torch.nn.functional.one_hot(
        data[:, 0].long(), num_classes=13
    ).float()

    result = trainer._score_public_split_metrics("test")
    assert result["mcc"] == pytest.approx(1.0)
    assert result["acc"] == pytest.approx(1.0)


def test_predict_decodes_classifier_cat_order_for_13_string_classes():
    from sklearn.preprocessing import LabelEncoder
    import torch.nn as nn

    class DummyAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Identity()
            self.classifier = nn.Identity()

    original = np.array([f'class_{i:02d}' for i in range(13)])
    le = LabelEncoder().fit(original)
    encoded = np.arange(13, dtype=int)
    cat_order = np.array(sorted(encoded, key=str), dtype=int)

    trainer = TrainAE.__new__(TrainAE)
    trainer.ae = DummyAE()
    trainer.args = SimpleNamespace(bs=32, num_workers=0, device='cpu', scaler=None, use_mapping=True)
    trainer.scaler = None
    trainer.columns = ['dummy']
    trainer.unique_labels = cat_order
    trainer._label_encoder = le
    trainer._prepare_prediction_matrix = lambda X, **kwargs: (
        (pd.DataFrame(X).copy(), np.zeros(len(X), dtype=int))
        if kwargs.get('return_batch_ids') else pd.DataFrame(X).copy()
    )
    logits = torch.eye(13, dtype=torch.float32)
    trainer._predict_logits_from_batch = lambda data, batch_ids=None: logits[: len(data)]

    X = pd.DataFrame({'dummy': np.zeros(13)})
    predicted = trainer.predict(X, batches_test=np.zeros(13, dtype=int))
    expected = le.inverse_transform(cat_order)
    assert np.array_equal(predicted, expected)
