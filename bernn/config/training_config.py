"""Training configuration dataclass for BERNN models."""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class TrainingConfig:
    """Configuration contract for BERNN holdout trainers.

    Core concepts
    - Layer depth is controlled by ``n_layers``.
    - ``layer1`` is the optional explicit seed width.
    - ``layer2+`` are intentionally not required in config and can be auto-derived
        at runtime by the trainer (halving each step, floor at 16).
    - Hyperparameter optimization can be switched on/off with
        ``optimize_hyperparams``.
    - Any value declared in ``fixed_hyperparams`` (and explicit ``layer1``)
        is treated as fixed and removed from Ax search space.

    This dataclass is also backward-compatible with legacy training paths by
    keeping optional compatibility attributes as explicit fields.
    """

    # Data loading/configuration is intentionally external to this dataclass.
    # Defaults are intentionally aligned with BERNN minimal trainer usage.

    # Model architecture
    dloss: str = 'inverseTriplet'  # one of revDANN, DANN, inverseTriplet, revTriplet, normae
    variational: bool = False
    tied_weights: bool = False
    use_mapping: bool = True  # Use batch mapping for reconstruct
    n_layers: int = 1  # N layers for classifier
    layer1: Optional[int] = None  # Optional explicit first hidden size; deeper layers auto-derived

    # Training configuration
    n_epochs: int = 1000
    n_repeats: int = 1
    early_stop: int = 50
    early_warmup_stop: int = 50
    train_after_warmup: bool = False
    warmup_after_warmup: bool = False
    warmup: int = 100  # Set during training
    device: str = 'cpu'
    use_sigmoid: bool = False  # Use sigmoid activation in the last layer of the AE

    # Loss and regularization
    rec_loss: str = 'l1'
    classif_loss: str = 'ce'
    threshold: float = 0.0
    kan: bool = False
    use_l1: bool = True
    prune_network: bool = True
    clip_val: float = 1.0
    update_grid: bool = True

    # Data processing
    embeddings_meta: int = 0
    groupkfold: bool = True
    log1p: bool = True
    scaler: str = 'standard'  # Set during training

    # Experiment tracking
    exp_id: str = 'bernn_training'
    model_name: str = 'ae_then_classifier_holdout'  # Set during training

    # Logging and evaluation
    random_recs: bool = False  # TODO to deprecate, no longer used
    predict_tests: bool = False
    n_agg: int = 1  # Number of trailing values to get stable valid values

    # Batch processing
    bs: int = 32  # Batch size
    bdisc: bool = True

    # Hyperparameter optimization
    n_trials: int = 1
    random: bool = True
    scheduler: str = 'ReductionLROnPlateau'  # Set during training, one of 'CosineAnnealingLR', 'ReductionLROnPlateau', 'CosineAnnealingWarmRestarts'
    optimize_hyperparams: bool = True
    fixed_hyperparams: Optional[dict[str, Any]] = None

    # Compatibility attributes expected by some legacy training paths.
    triplet_dloss: bool = True
    prune_threshold: float = 0.0
    prune_neurites_threshold: float = 0.0
    berm: str = 'bernn'
    disc_b_warmup: int = 0
    update_grid_warmup: int = 0
    remove_zeros: bool = False
    batches: Optional[object] = None
    pool_metrics_enc: Optional[object] = None
    pool_metrics_rec: Optional[object] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dloss not in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae', 'no']:
            raise ValueError(f"Invalid dloss: {self.dloss}. Must be one of: revTriplet, revDANN, DANN, inverseTriplet, normae, no")

        if self.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
            # Allow any device string that starts with 'cuda:'
            if not (self.device == 'cpu' or self.device.startswith('cuda')):
                raise ValueError(f"Invalid device: {self.device}")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'TrainingConfig':
        """Create a ``TrainingConfig`` from a plain dictionary.

        Unknown keys are not filtered in this method and will raise via dataclass
        construction. Use ``from_args`` when permissive filtering is desired.
        """
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args: Any) -> 'TrainingConfig':
        """Create a ``TrainingConfig`` from an argparse-like object.

        This helper is intentionally permissive: it filters out unknown fields
        instead of failing, which preserves compatibility with broader CLI
        namespaces and legacy argument objects.
        """
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = args

        # Only include keys that are valid for TrainingConfig
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_args = {k: v for k, v in args_dict.items() if k in valid_keys}

        return cls(**filtered_args)

    def get_fixed_hyperparams(self) -> dict[str, Any]:
        """Return explicit hyperparameter overrides that must stay fixed.

        Sources of fixed values
        - ``fixed_hyperparams`` dictionary provided by the user.
        - Explicit ``layer1`` value when set (treated as fixed seed width).

        Returns:
            Mapping of hyperparameter names to concrete fixed values that should
            be injected into each trial and excluded from optimization.
        """
        fixed = dict(self.fixed_hyperparams or {})
        if self.layer1 is not None:
            fixed['layer1'] = int(self.layer1)
        return fixed

    def filter_optimizable_parameters(self, parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter Ax parameter space according to optimization settings.

        Rules:
        - If ``optimize_hyperparams`` is ``False``, returns an empty list.
        - Parameters whose names appear in ``get_fixed_hyperparams()`` are
          removed from the search space.

        Args:
            parameters: Ax parameter definitions.

        Returns:
            Filtered list of parameters that remain eligible for optimization.
        """
        if not self.optimize_hyperparams:
            return []

        fixed_names = set(self.get_fixed_hyperparams().keys())
        return [p for p in parameters if p.get('name') not in fixed_names]
