from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

import bernn.single_cell as sc


class DummyTrainer:
    def __init__(self, config, **kwargs):
        self.config = config
        self.fit_args = None
        self.transform_args = None

    def fit(self, X, y, **kwargs):
        self.fit_args = (X.copy(), np.asarray(y), kwargs)
        return self

    def transform(self, X, **kwargs):
        self.transform_args = (X.copy(), kwargs)
        return np.asarray(X.iloc[:, :2], dtype=np.float32)


def test_fit_transform_anndata_uses_hvgs_and_all_cells(monkeypatch):
    monkeypatch.setattr(sc, "TrainAEClassifierHoldout", DummyTrainer)

    obs = pd.DataFrame(
        {
            "cell_type": ["a", "a", "b", "b", "a", "b"],
            "batch": ["x", "y", "x", "y", "x", "y"],
        }
    )
    var = pd.DataFrame(
        {
            "hvg_score": [0.1, 3.0, 1.0, 2.0],
            "hvg": [False, True, True, True],
        }
    )
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    adata = SimpleNamespace(
        X=sparse.csr_matrix(values),
        layers={"normalized": sparse.csr_matrix(values)},
        obs=obs,
        var=var,
        uns={"dataset_id": "toy"},
        shape=values.shape,
    )

    emb, trainer = sc.fit_transform_anndata(
        adata,
        n_hvg=3,
        n_epochs=1,
        batch_size=2,
        device="cpu",
        return_trainer=True,
    )

    X_fit, y_fit, fit_kwargs = trainer.fit_args
    assert X_fit.shape == (6, 3)
    np.testing.assert_array_equal(y_fit, obs["cell_type"].to_numpy())
    np.testing.assert_array_equal(fit_kwargs["groups_train"], obs["batch"].to_numpy())
    assert set(trainer.single_cell_feature_indices_.tolist()) == {1, 2, 3}
    assert emb.shape == (6, 2)
