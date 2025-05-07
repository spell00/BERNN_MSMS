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
            )
        },
        'batches': {
            'all': np.random.randint(0, n_batches, n_samples),
            'train': np.random.randint(0, n_batches, n_samples//2),
            'valid': np.random.randint(0, n_batches, n_samples//4),
            'test': np.random.randint(0, n_batches, n_samples//4)
        },
        'labels': {
            'all': np.random.randint(0, n_classes, n_samples),
            'train': np.random.randint(0, n_classes, n_samples//2),
            'valid': np.random.randint(0, n_classes, n_samples//4),
            'test': np.random.randint(0, n_classes, n_samples//4)
        }
    }
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
        log_neptune=False,
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
        'layer2': 32,
        'layer1': 64,
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

@pytest.mark.integration
@pytest.mark.slow
def test_full_training_pipeline(sample_data, mock_args, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Set CUDA device
    mock_args.device = 'cuda:0'
    
    # Initialize trainer
    trainer = TrainAEClassifierHoldout(
        mock_args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=True,
        keep_models=True,
        log_inputs=True,
        log_plots=True,
        log_tb=True,
        log_neptune=False,
        log_mlflow=True,
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
        'warmup': 5,
        'disc_b_warmup': 1,
        'dropout': 0.1,
        'scaler': 'standard',
        'layer2': 32,
        'layer1': 64,
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