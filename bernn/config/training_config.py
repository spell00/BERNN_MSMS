"""Training configuration dataclass for BERNN models."""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class TrainingConfig:
    """Configuration class for TrainAEThenClassifierHoldout.

    This dataclass contains all the configuration parameters for the training process.
    It provides type hints, default values, and clear documentation for each parameter.
    """

    # Data configuration
    csv_file: str = 'unique_genes.csv'
    best_features_file: str = ''
    dataset: str = 'custom'
    strategy: str = 'CU_DEM'  # only for alzheimer dataset
    bad_batches: str = ''  # e.g. '0;23;22;21;20;19;18;17;16;15'
    remove_zeros: bool = False

    # Model architecture
    dloss: str = 'inverseTriplet'  # one of revDANN, DANN, inverseTriplet, revTriplet, normae
    variational: bool = False
    zinb: bool = False  # TODO resolve problems, do not use
    tied_weights: bool = False
    use_mapping: bool = True  # Use batch mapping for reconstruct
    n_layers: int = 2  # N layers for classifier

    # Training configuration
    n_epochs: int = 1000
    n_repeats: int = 5
    early_stop: int = 50
    early_warmup_stop: int = -1
    train_after_warmup: bool = False
    warmup: int = 100  # Set during training
    device: str = 'cuda:0'
    use_sigmoid: bool = False  # Use sigmoid activation in the last layer of the AE

    # Loss and regularization
    rec_loss: str = 'l1'
    threshold: float = 0.0
    kan: bool = True
    use_l1: bool = True
    prune_network: bool = True
    clip_val: float = 1.0
    update_grid: bool = True

    # Data processing
    n_meta: int = 0
    embeddings_meta: int = 0
    groupkfold: bool = True
    log1p: bool = True  # log1p the data? Should be 0 with zinb
    scaler: str = 'standard'  # Set during training

    # Experiment tracking
    exp_id: str = 'default_ae_then_classifier'
    model_name: str = 'ae_then_classifier_holdout'  # Set during training

    # Logging and evaluation
    random_recs: bool = False  # TODO to deprecate, no longer used
    predict_tests: bool = False
    n_agg: int = 5  # Number of trailing values to get stable valid values

    # Batch processing
    bs: int = 32  # Batch size
    bdisc: bool = True
    pool: bool = False  # only for alzheimer dataset

    # Hyperparameter optimization
    n_trials: int = 100
    random: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dloss not in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae', 'no']:
            raise ValueError(f"Invalid dloss: {self.dloss}. Must be one of: revTriplet, revDANN, DANN, inverseTriplet, normae, no")

        if self.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
            # Allow any device string that starts with 'cuda:'
            if not (self.device == 'cpu' or self.device.startswith('cuda')):
                raise ValueError(f"Invalid device: {self.device}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create TrainingConfig from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args) -> 'TrainingConfig':
        """Create TrainingConfig from an argparse.Namespace or similar object."""
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = args

        # Only include keys that are valid for TrainingConfig
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_args = {k: v for k, v in args_dict.items() if k in valid_keys}

        return cls(**filtered_args)
