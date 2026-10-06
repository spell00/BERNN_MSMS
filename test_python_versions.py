import importlib.metadata
import os
import sys

import numpy
import torch

_SKIP_TENSORFLOW_IMPORT = os.environ.get(
    "BERNN_SKIP_TENSORFLOW_IMPORT", ""
).strip().lower() in {"1", "true", "yes", "on"}

if _SKIP_TENSORFLOW_IMPORT:
    tf = None
    try:
        tensorflow_version = importlib.metadata.version("tensorflow")
    except importlib.metadata.PackageNotFoundError:
        tensorflow_version = None
else:
    try:
        import tensorflow as tf
        tensorflow_version = tf.__version__
    except ImportError:
        tf = None
        tensorflow_version = None

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

    if tensorflow_version:
        suffix = " (package metadata; import skipped for CPU-only CI)" if _SKIP_TENSORFLOW_IMPORT else ""
        print(f"TensorFlow version: {tensorflow_version}{suffix}")
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
