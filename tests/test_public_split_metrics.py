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
