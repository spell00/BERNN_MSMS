"""BERNN: Batch Effect Removal Neural Networks for Tandem Mass Spectrometry.

This package provides tools for removing batch effects from mass spectrometry data
using deep learning approaches.
"""

__version__ = "0.2.2"
__author__ = "Simon Pelletier"
__license__ = "MIT"

# Import configuration
from . import config

# Import other modules
from .utils import *
from .dl import models

# Import models explicitly so they are available for direct import
from .dl.models.pytorch import (
    AutoEncoder2,
    SHAPAutoEncoder2,
    KANAutoencoder2,
    SHAPKANAutoencoder2,
    AutoEncoder3,
    SHAPAutoEncoder3,
    KANAutoencoder3,
    SHAPKANAutoencoder3,
)

# Optionally expose trainers
try:
    from .dl.train import (
        TrainAE,
        TrainAEClassifierHoldout, 
        TrainAEThenClassifierHoldout,
    )
except Exception:
    TrainAE = TrainAEClassifierHoldout = TrainAEThenClassifierHoldout = None

__all__ = [
    # Training
    "TrainAE",
    "TrainAEClassifierHoldout",
    "TrainAEThenClassifierHoldout",

    # Models
    "AutoEncoder2",
    "SHAPAutoEncoder2",
    "KANAutoencoder2",
    "SHAPKANAutoencoder2",
    "AutoEncoder3",
    "SHAPAutoEncoder3",
    "KANAutoencoder3",
    "SHAPKANAutoencoder3",

    # KAN
    "KANLinear",
    "KAN"
]
