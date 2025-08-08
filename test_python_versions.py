import sys
import numpy
import torch

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    import bernn
except ImportError:
    bernn = None

def test_python_versions():
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {numpy.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    if tf:
        print(f"TensorFlow version: {tf.__version__}")
    else:
        print("TensorFlow not available")
    
    if mlflow:
        print(f"MLflow version: {mlflow.__version__}")
    else:
        print("MLflow not available")
    
    if bernn:
        print(f"BERNN version: {bernn.__version__}")
    else:
        print("BERNN not available")

if __name__ == "__main__":
    test_python_versions()