"""Deep Learning modules for BERNN.

This subpackage contains the deep learning models and training code.
"""

# Model definitions
from .models.pytorch import (
    AutoEncoder2,
    SHAPAutoEncoder2,
    KANAutoencoder2,
    SHAPKANAutoencoder2,
)

# Training modules
from .train import (
    TrainAE,
    TrainAEClassifierHoldout,
    TrainAEThenClassifierHoldout,
)

# KAN modules
from .train.pytorch.ekan import KANLinear, KAN

__all__ = [
    # Models
    "AutoEncoder2",
    "SHAPAutoEncoder2",
    "KANAutoencoder2",
    "SHAPKANAutoencoder2",
    
    # Training
    "TrainAE",
    "TrainAEClassifierHoldout",
    "TrainAEThenClassifierHoldout",
    
    # KAN
    "KANLinear",
    "KAN"
]