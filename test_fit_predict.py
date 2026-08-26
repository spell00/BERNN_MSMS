"""
Test for fit_predict workflow:
  - TrainAEThenClassifierHoldout (with pools)
  - TrainAEClassifierHoldout     (without pools)
Both use synthetic data with groupkfold=True.
"""
import numpy as np
import pandas as pd
from bernn.dl.train.train_ae_then_classifier_holdout import TrainAEThenClassifierHoldout
from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout
import torch
import os
import numpy.testing as npt
from sklearn.metrics import matthews_corrcoef as MCC


def make_args(dataset_name, use_pools):
    """Build a minimal MockArgs compatible with all train() accessors."""
    class MockArgs:
        pass

    a = MockArgs()
    # Core training
    a.n_epochs = 2
    a.warmup = 1
    a.n_repeats = 1
    a.bs = 4
    a.lr = 1e-3
    a.wd = 1e-5
    a.layer1 = 16
    a.layer2 = 8
    a.layer3 = 16
    a.dropout = 0.0
    a.n_layers = 1
    a.clip_val = 1.0
    a.optimizer_type = 'adam'

    # Regularisation / loss
    a.dloss = 'inverseTriplet'
    a.rec_loss = 'mse'
    a.variational = 0
    a.tied_weights = 0
    a.random_recs = 0
    a.train_after_warmup = 1
    a.threshold = 0.0
    a.bdisc = 0
    a.use_l1 = 0
    a.prune_network = 0
    a.kan = 0
    a.update_grid = 0
    a.update_grid_warmup = 0
    a.use_sigmoid = 0
    a.use_mapping = 0
    a.embeddings_meta = 0
    a.n_agg = 1

    # Early stopping
    a.early_stop = 10
    a.early_warmup_stop = 0

    # Batch / domain
    a.bad_batches = ''
    a.remove_zeros = 0
    a.groupkfold = True
    a.pools = use_pools

    # Data paths (not used in in-memory flow but accessed for logging)
    a.csv_file = 'dummy.csv'
    a.best_features_file = 'dummy.tsv'
    a.dataset = dataset_name
    a.predict_tests = 0

    # Logging
    a.scaler = 'standard'
    a.model_name = 'test_model'
    a.scheduler = 'ReductionLROnPlateau'
    a.device = 'cpu'
    a.exp_id = 'test_exp'

    # Additional fields accessed directly
    a.berm = 'no'
    a.triplet_dloss = 'inverseTriplet'
    a.prune_threshold = 0.0
    a.prune_neurites_threshold = 0.0
    a.use_l1 = 0
    a.warmup_after_warmup = 0
    return a


def run_test(model_class, use_pools, dataset_name):
    print(f"\n{'='*60}")
    print(f"Testing {model_class.__name__}")
    print(f"  pools={use_pools}, dataset={dataset_name}")
    print(f"{'='*60}")

    np.random.seed(42)
    n_features = 10
    n_train = 20
    n_test = 6  # 3 per class

    # Interleave labels so both classes appear in every batch/split
    X_train = pd.DataFrame(np.random.randn(n_train, n_features))
    y_train = np.tile([0, 1], n_train // 2)                 # [0,1,0,1,...] - both classes in each half
    groups = np.array([0] * (n_train // 2) + [1] * (n_train // 2))   # batch 0: first 10, batch 1: last 10
    X_test = pd.DataFrame(np.random.randn(n_test, n_features))

    args = make_args(dataset_name, use_pools)

    model = model_class(
        config=args,
        groupkfold=True,
        pools=use_pools,
        log_mlflow=False,
        log_tb=False,
    )

    # Test with y_test=None to verify fix for reported TypeError
    # New API: call fit(...) then predict(...) separately
    model.fit(X_train, y_train, groups_train=groups)
    preds = model.predict(X_test, groups_test=groups[:len(X_test)])
    assert len(preds) == n_test, f"Expected length {n_test} but got {len(preds)}"
    print(f"\n✓ {model_class.__name__} passed! Predictions shape: {np.array(preds).shape}")
    return preds


def main():
    # Test 1: TrainAEThenClassifierHoldout with pools
    run_test(TrainAEThenClassifierHoldout, use_pools=True, dataset_name='dataset_with_pools')

    # Test 2: TrainAEClassifierHoldout without pools
    run_test(TrainAEClassifierHoldout, use_pools=False, dataset_name='dataset_no_pools')

    # Regression test: ensure checkpoint best_mcc equals restored-model MCC
    print('\nRunning regression: checkpoint best_mcc equals restored-model MCC')
    np.random.seed(123)
    n_features = 8
    n_train = 40
    X_train = pd.DataFrame(np.random.randn(n_train, n_features))
    y_train = np.tile([0, 1], n_train // 2)
    groups_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))

    # Use an external validation split (not identical to train)
    X_valid = pd.DataFrame(np.random.randn(8, n_features))
    y_valid = np.tile([0, 1], 4)
    groups_valid = np.array([0] * 4 + [1] * 4)

    args = make_args('regression_dataset', use_pools=False)
    model = TrainAEClassifierHoldout(config=args, groupkfold=True, pools=False, log_mlflow=False, log_tb=False)
    # Provide explicit validation and test splits to avoid empty-test dataset issues
    X_test = pd.DataFrame(np.random.randn(4, n_features))
    y_test = np.tile([0, 1], 2)
    groups_test = np.array([0, 1, 0, 1])

    model.fit(
        X_train, y_train,
        X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test,
        groups_train=groups_train, groups_valid=groups_valid, groups_test=groups_test
    )

    # BERNN's holdout trainer saves model state to `model_{rep}_state.pth`.
    saved_path = os.path.join(model.complete_log_path, f"model_{model.rep-1}_state.pth")
    assert os.path.exists(saved_path), f"Expected saved model state at {saved_path}"

    # Temporarily load the saved state into the AE, compute MCC on the validation
    # set, then restore the current in-memory state.
    current_state = model.ae.state_dict()
    saved_state = torch.load(saved_path, map_location='cpu')
    try:
        model.ae.load_state_dict(saved_state)
    except Exception:
        # Some saved files might be plain state_dicts; try that first.
        model.ae.load_state_dict(saved_state)

    preds_ckpt = model.predict(X_valid, groups_test=groups_valid)
    ckpt_mcc = float(MCC(pd.Series(y_valid).astype(str), pd.Series(preds_ckpt).astype(str)))

    # Restore in-memory AE
    model.ae.load_state_dict(current_state)

    # The fit() call also recomputes and stores restored valid MCC into model.best_mcc
    npt.assert_allclose(ckpt_mcc, float(model.best_mcc), atol=1e-6,
                        err_msg="Epoch-saved checkpoint MCC does not match restored-model MCC")

    print("\n\n✓✓ All tests passed!")


if __name__ == "__main__":
    main()
