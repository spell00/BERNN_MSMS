"""BERNN: Batch Effect Removal Neural Networks for Tandem Mass Spectrometry.

This package provides tools for removing batch effects from mass spectrometry data
using deep learning approaches.
"""

__version__ = "0.1.0"
__author__ = "Simon Pelletier"
__license__ = "MIT"

from bernn.dl.train.train_ae import TrainAE
from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout
from bernn.dl.train.train_ae_then_classifier_holdout import TrainAEThenClassifierHoldout

__all__ = [
    "TrainAE",
    "TrainAEClassifierHoldout", 
    "TrainAEThenClassifierHoldout"
]
