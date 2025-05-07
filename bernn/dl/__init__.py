"""Deep Learning modules for BERNN.

This subpackage contains the deep learning models and training code.
"""

from bernn.dl.models.pytorch.aedann import AutoEncoder2, SHAPAutoEncoder2
from bernn.dl.models.pytorch.aeekandann import KANAutoencoder2, SHAPKANAutoencoder2
# from bernn.dl.models.pytorch.aekandann import KANAutoencoder3, SHAPKANAutoencoder3

__all__ = [
    "AutoEncoder2",
    "SHAPAutoEncoder2",
    "KANAutoencoder2",
    "SHAPKANAutoencoder2",
    # "KANAutoencoder3",
    # "SHAPKANAutoencoder3"
]