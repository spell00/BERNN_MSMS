import inspect

import pytest

from bernn.dl.train.train_ae import TrainAE
from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout
from bernn.dl.train.train_ae_then_classifier_holdout import TrainAEThenClassifierHoldout


@pytest.mark.unit
def test_holdout_trainers_expose_epoch_callback():
    assert "epoch_callback" in inspect.signature(TrainAEClassifierHoldout.__init__).parameters
    assert "epoch_callback" in inspect.signature(TrainAEThenClassifierHoldout.__init__).parameters


@pytest.mark.unit
def test_epoch_callback_receives_copy_and_exceptions_propagate():
    trainer = TrainAE.__new__(TrainAE)
    seen = []
    trainer.epoch_callback = seen.append
    payload = {"epoch": 3, "valid_mcc": 0.71}
    trainer._notify_epoch(payload)
    assert seen == [payload]
    assert seen[0] is not payload

    def stop(_payload):
        raise RuntimeError("stop trial")

    trainer.epoch_callback = stop
    with pytest.raises(RuntimeError, match="stop trial"):
        trainer._notify_epoch(payload)
