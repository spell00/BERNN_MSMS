"""Unit tests for TrainAE helpers in bernn/dl/train/train_ae.py."""
import pytest
import numpy as np
import torch
from types import SimpleNamespace

from bernn.dl.train.train_ae import TrainAE


def _minimal_args(**overrides):
    ns = SimpleNamespace(
        device="cpu",
        random_recs=0,
        predict_tests=0,
        early_stop=5,
        early_warmup_stop=-1,
        train_after_warmup=0,
        train_only_warmup=0,
        threshold=0.0,
        n_epochs=1,
        rec_loss="l1",
        tied_weights=0,
        random=1,
        variational=0,
        zinb=0,
        use_mapping=1,
        bdisc=0,
        n_repeats=1,
        dloss="inverseTriplet",
        remove_zeros=0,
        n_meta=0,
        embeddings_meta=0,
        groupkfold=0,
        n_layers=2,
        kan=0,
        use_l1=0,
        clip_val=1.0,
        log_metrics=0,
        log_plots=0,
        prune_network=0,
        dataset="mock",
        csv_file="mock.csv",
        log1p=1,
        berm="none",
        pool=0,
        strategy="none",
        best_features_file="",
        n_features=-1,
        bad_batches="",
        controls="",
        exp_id="test",
        warmup_after_warmup=0,
        bs=8,
        n_agg=1,
        update_grid=0,
        prune_threshold=0.0,
        scheduler="ReduceLROnPlateau",
        path=".",
        log_tb=0,

        log_mlflow=0,
        keep_models=0,
        log_inputs=0,
        classif_loss="ce",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def trainer(tmp_path):
    args = _minimal_args()
    return TrainAE(
        args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )


# ────────────────────────────────────────────────────────────
# fill_missing_params_with_default
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_fill_missing_params_adds_defaults(trainer):
    """A namespace missing optional keys gets them filled from defaults."""
    args = _minimal_args()
    # Remove an attribute that has a default
    if hasattr(args, "n_trials"):
        delattr(args, "n_trials")
    filled = trainer.fill_missing_params_with_default(args)
    assert hasattr(filled, "n_trials")


@pytest.mark.unit
def test_fill_missing_params_preserves_given_value(trainer):
    """Explicitly set values should not be overwritten by defaults."""
    args = _minimal_args(n_epochs=999)
    filled = trainer.fill_missing_params_with_default(args)
    assert filled.n_epochs == 999


@pytest.mark.unit
def test_fill_missing_params_returns_namespace(trainer):
    args = _minimal_args()
    result = trainer.fill_missing_params_with_default(args)
    assert hasattr(result, "__dict__")


# ────────────────────────────────────────────────────────────
# make_params
# ────────────────────────────────────────────────────────────

def _base_params():
    return {
        "gamma": 1.0,
        "beta": 1.0,
        "zeta": 1.0,
        "thres": 0.5,
        "scaler": "standard",
        "warmup": 1,
        "disc_b_warmup": 1,
        "dropout": 0.0,
        "prune_threshold": 0.0,
        "l1": 0.01,
        "reg_entropy": 0.001,
    }


@pytest.mark.unit
def test_make_params_gamma_zeroed_when_no_dann(trainer):
    """gamma should be 0 when dloss is not a DANN variant."""
    trainer.args.dloss = "no"
    params = _base_params()
    trainer.make_params(params)
    assert params["gamma"] == 0


@pytest.mark.unit
def test_make_params_gamma_preserved_when_dann(trainer):
    """gamma should keep its value when dloss is DANN."""
    trainer.args.dloss = "DANN"
    trainer.args.variational = 0
    params = _base_params()
    params["gamma"] = 0.5
    trainer.make_params(params)
    assert params["gamma"] == 0.5


@pytest.mark.unit
def test_make_params_beta_zeroed_nonvariational(trainer):
    trainer.args.variational = 0
    params = _base_params()
    trainer.make_params(params)
    assert params["beta"] == 0


@pytest.mark.unit
def test_make_params_l1_zeroed_when_use_l1_off(trainer):
    trainer.args.use_l1 = 0
    params = _base_params()
    trainer.make_params(params)
    assert params["l1"] == 0


@pytest.mark.unit
def test_make_params_fix_thres_applied(tmp_path):
    """When fix_thres is in [0,1), params['thres'] should equal fix_thres."""
    args = _minimal_args()
    t = TrainAE(
        args, path=str(tmp_path), fix_thres=0.3,
        load_tb=False, log_metrics=False, keep_models=False,
        log_inputs=False, log_plots=False, log_tb=False, log_mlflow=False, groupkfold=False, pools=False,
    )
    params = _base_params()
    t.make_params(params)
    assert params["thres"] == pytest.approx(0.3)


@pytest.mark.unit
def test_make_params_no_fix_thres(trainer):
    """When fix_thres=-1, params['thres'] should be 0."""
    params = _base_params()
    trainer.make_params(params)
    assert params["thres"] == 0


# ────────────────────────────────────────────────────────────
# default_params / all_params completeness
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_default_params_has_required_keys(trainer):
    required = {"n_epochs", "dloss", "variational", "rec_loss", "bs", "kan"}
    assert required.issubset(set(trainer.all_params.keys()))


# ────────────────────────────────────────────────────────────
# binarize_labels (imported from train_ae module)
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_binarize_labels_produces_binary(tmp_path):
    """binarize_labels should turn controls into 0, others into 1."""
    from bernn.dl.train.train_ae import binarize_labels
    import numpy as np

    data = {
        "labels": {
            "all": np.array(["ctrl", "case", "ctrl", "case"]),
            "train": np.array(["ctrl", "case"]),
            "valid": np.array(["ctrl"]),
            "test": np.array(["case"]),
        },
        "cats": {
            "all": np.array(["ctrl", "case", "ctrl", "case"]),
            "train": np.array(["ctrl", "case"]),
            "valid": np.array(["ctrl"]),
            "test": np.array(["case"]),
        },
    }
    result = binarize_labels(data, controls=["ctrl"])
    assert set(result["labels"]["all"]).issubset({0, 1})
    assert result["labels"]["all"][0] == 0   # ctrl → 0
    assert result["labels"]["all"][1] == 1   # case → 1


class _TinyAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(2, 2)
        self.dec = torch.nn.Linear(2, 2)
        self.classifier = torch.nn.Linear(2, 2)

    def forward(self, x, to_rec, batches=None, sampling=False, **kwargs):
        encoded = self.enc(x)
        reconstructed = {"mean": self.dec(encoded)}
        return [encoded, reconstructed, torch.zeros(1)]

    def predict(self, x):
        return self.classifier(self.enc(x)).argmax(1).detach().cpu().numpy()

    def predict_proba(self, x):
        return torch.softmax(self.classifier(self.enc(x)), dim=1).detach().cpu().numpy()


@pytest.fixture
def inference_trainer(trainer):
    trainer.ae = _TinyAE()
    trainer.args.device = "cpu"
    trainer.args.bs = 2
    return trainer


@pytest.mark.unit
def test_get_encoded_inputs_returns_bottleneck(inference_trainer):
    X = np.ones((3, 2), dtype=np.float32)
    encoded = inference_trainer.get_encoded_inputs(X)
    assert encoded.shape == (3, 2)


@pytest.mark.unit
def test_get_reconstructed_inputs_returns_original_feature_shape(inference_trainer):
    X = np.ones((3, 2), dtype=np.float32)
    reconstructed = inference_trainer.get_reconstructed_inputs(X)
    assert reconstructed.shape == (3, 2)


@pytest.mark.unit
def test_infer_can_return_predictions_and_representations(inference_trainer):
    X = np.ones((3, 2), dtype=np.float32)
    result = inference_trainer.infer(X, return_representations=True)
    assert set(["predictions", "encoded", "reconstructed", "probabilities"]).issubset(result)
    assert result["predictions"].shape == (3,)
    assert result["encoded"].shape == (3, 2)
    assert result["reconstructed"].shape == (3, 2)
    assert result["probabilities"].shape == (3, 2)


@pytest.mark.unit
def test_validate_and_save_best_checkpoint_keeps_in_memory_state(inference_trainer, tmp_path):
    inference_trainer.complete_log_path = str(tmp_path)
    assert inference_trainer._validate_and_save_best_checkpoint(inference_trainer.ae, epoch=4, valid_mcc=0.25)
    assert inference_trainer.best_state_dicts is not None
    assert inference_trainer.best_epoch == 4
