def test_python_imports():
    """Test that all required Python packages can be imported."""
    import numpy
    import torch
    import tensorflow
    import mlflow
    import pandas
    import sklearn
    import matplotlib
    import seaborn
    import scipy
    import tqdm
    import joblib
    import ax
    import pycombat
    import torchvision
    import tensorboardX
    import tensorboard
    import psutil
    import skimage
    import nibabel
    import mpmath
    import patsy
    import umap
    import shapely
    import numba
    import rpy2
    import openpyxl
    import xgboost
    import torch_geometric
    import neptune
    import fastapi
    import threadpoolctl
    # import protobuf
    import shap

def test_r_imports():
    """Test that all required R packages can be imported."""
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    
    # Import R packages
    # harmony = importr('harmony')
    # sva = importr('sva')
    # zinbwave = importr('zinbwave')
    # lisi = importr('lisi')
    # gPCA = importr('gPCA')
    # WaveICA = importr('WaveICA')
