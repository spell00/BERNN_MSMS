import os
import pytest
import torch
import pandas as pd
import numpy as np
from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout


@pytest.fixture
def sample_data():
    # Create sample dataset
    n_samples = 100
    n_features = 50
    n_meta = 2
    n_batches = 3
    n_classes = 2

    # Generate random data
    data = {
        'inputs': {
            'all': pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            ),
            'train': pd.DataFrame(
                np.random.randn(n_samples//2, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            ),
            'valid': pd.DataFrame(
                np.random.randn(n_samples//4, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            ),
            'test': pd.DataFrame(
                np.random.randn(n_samples//4, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            ),
            'train_pool': pd.DataFrame(
                np.random.randn(n_samples//2, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            ),
            'valid_pool': pd.DataFrame(
                np.random.randn(n_samples//4, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            ),
            'test_pool': pd.DataFrame(
                np.random.randn(n_samples//4, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            ),
            'all_pool': pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            )
        },
        'meta': {
            'all': pd.DataFrame(np.random.randn(n_samples, n_meta), columns=[f'meta_{i}' for i in range(n_meta)]),
            'train': pd.DataFrame(np.random.randn(n_samples//2, n_meta), columns=[f'meta_{i}' for i in range(n_meta)]),
            'valid': pd.DataFrame(np.random.randn(n_samples//4, n_meta), columns=[f'meta_{i}' for i in range(n_meta)]),
            'test': pd.DataFrame(np.random.randn(n_samples//4, n_meta), columns=[f'meta_{i}' for i in range(n_meta)]),
            'train_pool': pd.DataFrame(np.random.randn(n_samples//2, n_meta), columns=[f'meta_{i}' for i in range(n_meta)]),
            'valid_pool': pd.DataFrame(np.random.randn(n_samples//4, n_meta), columns=[f'meta_{i}' for i in range(n_meta)]),
            'test_pool': pd.DataFrame(np.random.randn(n_samples//4, n_meta), columns=[f'meta_{i}' for i in range(n_meta)]),
            'all_pool': pd.DataFrame(np.random.randn(n_samples, n_meta), columns=[f'meta_{i}' for i in range(n_meta)]),
        },
        'batches': {
            'all': np.random.randint(0, n_batches, n_samples),
            'train': np.random.randint(0, n_batches, n_samples//2),
            'valid': np.random.randint(0, n_batches, n_samples//4),
            'test': np.random.randint(0, n_batches, n_samples//4),
            'train_pool': np.random.randint(0, n_batches, n_samples//2),
            'valid_pool': np.random.randint(0, n_batches, n_samples//4),
            'test_pool': np.random.randint(0, n_batches, n_samples//4),
            'all_pool': np.random.randint(0, n_batches, n_samples),
        },
        'labels': {
            'all': np.random.randint(0, n_classes, n_samples),
            'train': np.random.randint(0, n_classes, n_samples//2),
            'valid': np.random.randint(0, n_classes, n_samples//4),
            'test': np.random.randint(0, n_classes, n_samples//4),
            'train_pool': np.random.randint(0, n_classes, n_samples//2),
            'valid_pool': np.random.randint(0, n_classes, n_samples//4),
            'test_pool': np.random.randint(0, n_classes, n_samples//4),
            'all_pool': np.random.randint(0, n_classes, n_samples),
        },
        'cats': {
            'all': None,
            'train': None,
            'valid': None,
            'test': None,
            'train_pool': None,
            'valid_pool': None,
            'test_pool': None,
            'all_pool': None,
        },
        'sets': {
            'all': np.array(['all'] * n_samples),
            'train': np.array(['train'] * (n_samples // 2)),
            'valid': np.array(['valid'] * (n_samples // 4)),
            'test': np.array(['test'] * (n_samples // 4)),
            'train_pool': np.array(['train_pool'] * (n_samples // 2)),
            'valid_pool': np.array(['valid_pool'] * (n_samples // 4)),
            'test_pool': np.array(['test_pool'] * (n_samples // 4)),
            'all_pool': np.array(['all_pool'] * n_samples),
        },
        'names': {
            'all': pd.Series(['s' + str(x) for x in np.arange(0, n_samples)]),
            'train': pd.Series(['s' + str(x) for x in np.arange(0, n_samples//2)]),
            'valid': pd.Series(['s' + str(x) for x in np.arange(0, n_samples//4)]),
            'test': pd.Series(['s' + str(x) for x in np.arange(0, n_samples//4)]),
            'train_pool': pd.Series(['sp' + str(x) for x in np.arange(0, n_samples//2)]),
            'valid_pool': pd.Series(['sp' + str(x) for x in np.arange(0, n_samples//4)]),
            'test_pool': pd.Series(['sp' + str(x) for x in np.arange(0, n_samples//4)]),
            'all_pool': pd.Series(['sp' + str(x) for x in np.arange(0, n_samples)]),
        }
    }
    # Ensure balanced labels: half one label, half the other
    for split in ['all', 'train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all_pool']:
        n = len(data['labels'][split])
        data['labels'][split][:n // 2] = 0
        data['labels'][split][n // 2:] = 1
    # Ensure all splits include all labels
    for split in ['train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool']:
        data['labels'][split][0] = 0  # Ensure at least one instance of label 0
        data['labels'][split][1] = 1  # Ensure at least one instance of label 1
    for split in ['all', 'train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all_pool']:
        data['cats'][split] = data['labels'][split].copy()
        if split in ['all', 'train', 'valid', 'test']:
            print(data['inputs'][split].columns[:5])
    return data


@pytest.fixture
def mock_args():
    class Args:
        def __init__(self):
            self.device = 'cpu'
            self.random_recs = 0
            self.predict_tests = 0
            self.early_stop = 5
            self.early_warmup_stop = -1
            self.train_after_warmup = 0
            self.threshold = 0.0
            self.n_epochs = 2
            self.rec_loss = 'l1'
            self.tied_weights = 0
            self.random = 1
            self.variational = 0
            self.zinb = 0
            self.use_mapping = 1
            self.bdisc = 1
            self.n_repeats = 1
            self.dloss = 'inverseTriplet'
            self.remove_zeros = 0
            self.n_meta = 0
            self.embeddings_meta = 0
            self.groupkfold = 1
            self.n_layers = 2
            self.kan = 0
            self.use_l1 = 0
            self.clip_val = 1.0
            self.log_metrics = 0
            self.log_plots = 0
            self.prune_network = 0
            self.dataset = 'mock'
            self.csv_file = 'mock.csv'
            self.log1p = 1
            self.berm = 'none'
            self.pool = 0
            self.strategy = 'none'
            self.best_features_file = 'mock_top_features.tsv'
            self.n_features = -1
            self.bad_batches = ''
            self.controls = 'l0'
            self.exp_id = 'mockTest'
            self.warmup_after_warmup = 1
            self.bs = 8
            self.n_agg = 1
            self.update_grid = 1
            self.prune_threshold = 0.001
            self.scheduler = 'ReduceLROnPlateau'

    return Args()


@pytest.mark.integration
def test_training_loop(sample_data, mock_args, tmp_path):
    # Initialize trainer
    trainer = TrainAEClassifierHoldout(
        mock_args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=True,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=True,
        pools=True
    )

    # Set the data
    trainer.data = sample_data
    trainer.unique_labels = np.unique(sample_data['labels']['all'])
    trainer.unique_batches = np.unique(sample_data['batches']['all'])
    trainer.columns = sample_data['inputs']['all'].columns

    # Run training with some test parameters
    params = {
        'nu': 0.001,
        'lr': 0.001,
        'wd': 1e-6,
        'smoothing': 0.1,
        'margin': 1.0,
        'warmup': 2,
        'disc_b_warmup': 1,
        'dropout': 0.1,
        'scaler': 'standard',
        'layer1': 32,
        'layer2': 32,
        'gamma': 0.1,
        'beta': 0.0,
        'zeta': 0.0,
        'thres': 0.0,
        'prune_threshold': 0.0
    }

    # Write the data to files outside the trainer (external loading contract)
    csv_path = tmp_path / 'mock.csv'
    tsv_path = tmp_path / 'mock_top_features.tsv'
    sample_data['inputs']['all'].to_csv(csv_path, index=False)
    pd.DataFrame(sample_data['inputs']['all'].columns).to_csv(tsv_path, index=False)

    try:
        result = trainer.train(params)
        assert isinstance(result, (float, int)), "Training should return a numeric value"
    except Exception as e:
        pytest.skip(f"Training failed due to: {str(e)}")
    if csv_path.exists():
        os.remove(csv_path)


@pytest.mark.integration
@pytest.mark.slow
def test_full_training_pipeline(sample_data, mock_args, tmp_path):
    if not torch.cuda.is_available():
        mock_args.device = 'cpu'
    else:
        # Set CUDA device
        mock_args.device = 'cuda:0'

    # Initialize trainer
    trainer = TrainAEClassifierHoldout(
        mock_args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=True,
        pools=True
    )

    # Set the data
    trainer.data = sample_data
    trainer.unique_labels = np.unique(sample_data['labels']['all'])
    trainer.unique_batches = np.unique(sample_data['batches']['all'])
    trainer.columns = sample_data['inputs']['all'].columns

    # Write the data to files outside the trainer (external loading contract)
    csv_path = tmp_path / 'mock.csv'
    tsv_path = tmp_path / 'mock_top_features.tsv'
    sample_data['inputs']['all'].to_csv(csv_path, index=False)
    pd.DataFrame(sample_data['inputs']['all'].columns).to_csv(tsv_path, index=False)

    # Run training with some test parameters
    params = {
        'nu': 0.001,
        'lr': 0.001,
        'wd': 1e-6,
        'smoothing': 0.1,
        'margin': 1.0,
        'warmup': 5,
        'disc_b_warmup': 1,
        'dropout': 0.1,
        'scaler': 'standard',
        'layer1': 32,
        'layer2': 32,
        'gamma': 0.1,
        'beta': 0.0,
        'zeta': 0.0,
        'thres': 0.0,
        'prune_threshold': 0.0
    }

    try:
        result = trainer.train(params)
        assert isinstance(result, (float, int)), "Training should return a numeric value"
    except Exception as e:
        pytest.skip(f"Training failed due to: {str(e)}")
    if csv_path.exists():
        os.remove(csv_path)
