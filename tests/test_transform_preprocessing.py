from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout


class EchoAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Identity()
        self.classifier = torch.nn.Identity()

    def forward(self, x, to_rec, batches=None, sampling=False, mapping=True):
        return [x, {"mean": x}, torch.zeros(1), torch.zeros(1)]


def test_transform_reuses_fitted_preprocessing_and_scaler():
    trainer = TrainAEClassifierHoldout.__new__(TrainAEClassifierHoldout)
    trainer.args = SimpleNamespace(
        bs=2,
        num_workers=0,
        device="cpu",
        scaler="standard",
        log1p=True,
        use_mapping=True,
    )
    trainer.ae = EchoAE()
    trainer.columns = ["a", "b"]

    train = pd.DataFrame([[0.0, 3.0], [3.0, 7.0], [8.0, 15.0]], columns=trainer.columns)
    train_log = np.log1p(train.to_numpy())
    trainer.scaler = StandardScaler().fit(train_log)

    raw = pd.DataFrame([[1.0, 3.0], [8.0, 7.0]], columns=trainer.columns)
    expected = trainer.scaler.transform(np.log1p(raw.to_numpy()))
    expected = np.round(expected, 4).astype(np.float32)

    transformed = trainer.transform(raw)
    np.testing.assert_allclose(transformed, expected, rtol=0, atol=1e-6)
