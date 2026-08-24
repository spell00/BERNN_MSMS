import numpy as np
import pandas as pd
import pytest

from bernn.config.training_config import TrainingConfig
from bernn.dl.models.pytorch.utils import dataset as dataset_module


def _split(rows, split):
    return {
        "inputs": pd.DataFrame(np.arange(rows * 3, dtype=float).reshape(rows, 3)),
        "names": np.asarray([f"{split}_{index}" for index in range(rows)]),
        "cats": np.asarray([index % 2 for index in range(rows)]),
        "batches": np.asarray([f"batch_{index % 2}" for index in range(rows)]),
        "sets": np.asarray([split] * rows),
    }


def test_training_config_accepts_num_workers():
    config = TrainingConfig(num_workers=2)

    assert config.num_workers == 2


def test_training_config_rejects_negative_num_workers():
    with pytest.raises(ValueError, match="num_workers must be >= 0"):
        TrainingConfig(num_workers=-1)


def test_no_pool_loaders_forward_num_workers(monkeypatch):
    splits = {name: _split(4, name) for name in ("train", "valid", "test")}
    data = {
        key: {name: split[key] for name, split in splits.items()}
        for key in ("inputs", "names", "cats", "batches", "sets")
    }
    for key in data:
        if key == "inputs":
            data[key]["all"] = pd.concat(list(data[key].values()), ignore_index=True)
        else:
            data[key]["all"] = np.concatenate(list(data[key].values()))

    calls = []

    class RecordingDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs
            calls.append(kwargs)

    monkeypatch.setattr(dataset_module, "DataLoader", RecordingDataLoader)
    weights = {name: np.ones(len(splits[name]["cats"])) for name in splits}

    loaders = dataset_module.get_loaders_no_pool(
        data,
        random_recs=False,
        samples_weights=weights,
        triplet_dloss="DANN",
        bs=2,
        device="cpu",
        num_workers=2,
    )

    assert loaders
    assert calls
    assert all(call["num_workers"] == 2 for call in calls)
