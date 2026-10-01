import numpy as np

from bernn.dl.models.pytorch.utils.utils import LogConfusionMatrix


def _group_payload(n=1024):
    return {
        "preds": [np.column_stack([np.ones(n), np.zeros(n)])],
        "classes": [np.zeros(n, dtype=int)],
        "encoded_values": [np.zeros((n, 2), dtype=np.float32)],
        "rec_values": [np.zeros((n, 2), dtype=np.float32)],
        "cats": [np.zeros(n, dtype=int)],
        "domains": [np.zeros(n, dtype=int)],
    }


def test_confusion_matrix_logger_handles_batch_size_over_1000(tmp_path):
    logger = LogConfusionMatrix(str(tmp_path))
    payload = {
        "train": _group_payload(1024),
        "valid": _group_payload(1024),
        "test": _group_payload(1024),
    }

    logger.add(payload)

    assert len(logger.preds["train"]) == 1
    assert logger.preds["train"][0].shape == (1024,)
