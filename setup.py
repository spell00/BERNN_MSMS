#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 2021
@author: Simon Pelletier
"""

from setuptools import setup, find_packages
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get Python version info
python_version = sys.version_info
is_python312_or_later = python_version >= (3, 12)
is_python313_or_later = python_version >= (3, 13)

# Define core/minimal requirements (compatible with Python 3.11+)
# These are carefully chosen to minimize conflicts
if is_python313_or_later:
    # For Python 3.13+ - use newer versions and more flexible constraints
    minimal_requirements = [
        "scikit-learn>=1.6.0",  # Force newer scikit-learn on Python 3.13+
        "pandas>=2.2.0",  # Python 3.13 compatible
        "scikit-optimize>=0.9.0",
        "matplotlib>=3.7.0",  # Python 3.13 compatible
        "seaborn>=0.12.2",
        "tabulate>=0.9.0",
        "scipy>=1.11.0",  # Python 3.13 compatible
        "tqdm",
        "joblib>=1.3.0",  # Python 3.13 compatible
        "psutil>=5.9.4",
        "scikit-image>=0.21.0",  # Python 3.13 compatible
        "nibabel",
        "mpmath>=1.3.0",
        "patsy>=0.5.3",
        "umap-learn>=0.5.3",
        "shapely",
        "numba>=0.58.0",  # Python 3.13 compatible
        "openpyxl>=3.0.10",
        "xgboost>=1.7.0",  # Python 3.13 compatible
        "importlib-metadata>=6.0.0",
        "threadpoolctl>=3.1.0",
        "protobuf>=6.31.1,<7",  # Keep consistent with TensorFlow ecosystem pins
        "requests>=2.31.0,<3.0.0",
        "PyYAML>=6.0.1",
        "python-dateutil>=2.8.2",
        "nbformat>=5.9.2",
        "statsmodels",
    ]
elif is_python312_or_later:
    # For Python 3.12+ - use newer versions that support Python 3.12
    minimal_requirements = [
        "scikit-learn>=1.6.0",  # Force newer scikit-learn on Python 3.12+
        "pandas>=2.2.0",  # Python 3.12 compatible
        "scikit-optimize>=0.9.0",
        "matplotlib>=3.7.0",  # Python 3.12 compatible
        "seaborn>=0.12.2",
        "tabulate>=0.9.0",
        "scipy>=1.11.0",  # Python 3.12 compatible
        "tqdm",
        "joblib>=1.3.0",  # Python 3.12 compatible
        "psutil>=5.9.4",
        "scikit-image>=0.21.0",  # Python 3.12 compatible
        "nibabel",
        "mpmath>=1.3.0",
        "patsy>=0.5.3",
        "umap-learn>=0.5.3",
        "shapely",
        "numba>=0.58.0",  # Python 3.12 compatible
        "openpyxl>=3.0.10",
        "xgboost>=1.7.0",  # Python 3.12 compatible
        "importlib-metadata>=6.0.0",
        "threadpoolctl>=3.1.0",
        "protobuf>=6.31.1,<7",
        "requests>=2.31.0,<3.0.0",
        "PyYAML>=6.0.1",
        "python-dateutil>=2.8.2",
        "nbformat>=5.9.2",
        "statsmodels",
    ]
else:
    # For Python 3.11 - use compatible versions
    minimal_requirements = [
        "scikit-learn>=1.6.0",  # Force newer scikit-learn on Python 3.11+
        "pandas>=1.5.3,<3.0.0",  # Align with stable requirements profile
        "scikit-optimize>=0.9.0",
        "matplotlib>=3.6.3",
        "seaborn>=0.12.2",
        "tabulate>=0.9.0",
        "scipy>=1.9.3,<2.0.0",  # Avoid forced downgrades during editable install
        "tqdm",
        "joblib>=1.2.0",
        "psutil>=5.9.4",
        "scikit-image",
        "nibabel",
        "mpmath>=1.3.0",
        "patsy>=0.5.3",
        "umap-learn>=0.5.3",
        "shapely",
        "numba>=0.57.1",
        "openpyxl>=3.0.10",
        "xgboost>=1.7.0",
        "importlib-metadata>=3.7.0,<8",
        "threadpoolctl>=3.1.0",
        "protobuf>=6.31.1,<7",  # Required by newer TensorFlow releases
        "requests>=2.31.0,<3.0.0",  # Compatible with most packages
        "PyYAML>=6.0.1",
        "python-dateutil>=2.8.2",
        "nbformat>=5.9.2",
        "statsmodels",
    ]

# Optional packages that may cause conflicts - separate from core
optional_requirements = [
    "shap",  # Can be problematic with some TensorFlow versions
    "pytest",
    "pytest-cov",
    "cython>=0.29.21",
]

# R integration (separate due to potential conflicts)
r_integration_requirements = [
    "rpy2>=3.6.0",  # Only if R integration is needed
]

# Development tools (separate due to version conflicts)
dev_tools_requirements = [
    "jedi>=0.18.2",  # Conflicts with older spyder versions
]

# Packages with specific compatibility issues
compatibility_requirements = [
    "FuzzyTM>=0.4.0",
    "blosc2>=2.0.0,<3.0.0",
    "llvmlite>=0.40.1",
    "pycombat",
]

# Web/API packages (separate due to fastapi conflicts)
web_requirements = [
    "fastapi>=0.89.1,<0.103.0",  # Compatible with current environment
    "websocket-client>=1.8.0",  # Updated for selenium compatibility
    "platformdirs>=3.11.0,<4.2.0",  # Current environment compatible
]

# Deep learning dependencies - version-specific with conflict resolution
if is_python313_or_later:
    # For Python 3.13+ - use working combination with matching versions
    deep_learning_requirements = [
        "torch>=2.1.0",  # PyTorch generally supports newer Python versions faster
        "torchvision>=0.16.0",
        "torch-geometric",
        # Use TensorFlow 2.20.0rc0 with matching estimator
        "tensorflow>=2.20.0rc0",  # Use release candidate for Python 3.13
        # Don't specify tensorflow-estimator version - let TensorFlow handle it
        "typing-extensions>=4.9.0",
        "numpy>=1.24,<2.3",
        "six>=1.16.0",
    ]
elif is_python312_or_later:
    # For Python 3.12+ - use latest versions
    deep_learning_requirements = [
        "torch>=2.1.0",  # Python 3.12 compatible
        "torchvision>=0.16.0",
        "torch-geometric",
        "tensorflow>=2.15.0",  # Python 3.12 compatible
        "tensorflow-estimator>=2.15.0",
        "typing-extensions>=4.9.0",  # Updated for compatibility
        "numpy<2.3,>=1.24",
        "six>=1.16.0",  # Updated for compatibility
    ]
else:
    # For Python 3.11 - use newer versions
    deep_learning_requirements = [
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torch-geometric",
        "tensorflow>=2.15.0",
        "tensorflow-estimator>=2.15.0",
        "typing-extensions>=4.9.0",
        "numpy>=1.24",
    ]

# Experiment tracking dependencies - version-specific with conflict resolution
if is_python313_or_later:
    experiment_tracking_requirements = [
        # Don't specify tensorboard version - let TensorFlow handle it
        "tensorboardX",
        "mlflow[extras]>=2.12.1",  # Python 3.13 should be compatible
        "sqlalchemy>=2.0.0",
        "urllib3>=1.26.7",
    ]
elif is_python312_or_later:
    experiment_tracking_requirements = [
        "tensorboard>=2.15.0",
        "tensorboard-data-server>=0.7.0",
        "tensorboardX",
        "mlflow[extras]>=2.12.1",  # Python 3.12 compatible
        "sqlalchemy>=2.0.0",
        "urllib3>=1.26.7",
    ]
else:
    experiment_tracking_requirements = [
        "tensorboard>=2.15.0",
        "tensorboard-data-server>=0.7.0",
        "tensorboardX",
        "mlflow[extras]>=2.15.0",
        "sqlalchemy>=1.4.0",
        "urllib3>=1.26.7",
    ]

# Notebook dependencies - version-specific
if is_python313_or_later:
    notebook_requirements = [
        "notebook>=7.0.0",  # Python 3.13 compatible
        "ipywidgets>=8.0.0",
        "jupyterlab>=4.0.0",
    ]
elif is_python312_or_later:
    notebook_requirements = [
        "notebook>=7.0.0",  # Python 3.12 compatible
        "ipywidgets>=8.0.0",
        "jupyterlab>=4.0.0",
    ]
else:
    notebook_requirements = [
        "notebook>=7.0.0",
        "ipywidgets>=8.0.0",
        "jupyterlab>=4.0.0",
    ]

# Additional tools (with conflict resolution)
# Note: ax-platform has very strict version requirements that conflict with modern packages
if is_python313_or_later:
    tools_requirements = [
        # Exclude ax-platform for Python 3.13+ due to conflicts
        "packaging>=21.0",
        "python-dateutil>=2.8.2",
        "PyYAML>=6.0.1",
        "optuna>=3.0.0",  # Alternative optimization library
    ]
elif is_python312_or_later:
    tools_requirements = [
        # Exclude ax-platform for Python 3.12+ due to conflicts
        "ax-platform>=1.0.0",  # Causes too many conflicts
        "packaging>=21.0",
        "python-dateutil>=2.8.2",
        "PyYAML>=6.0.1",
        "optuna>=3.0.0",  # Alternative optimization library
    ]
else:
    tools_requirements = [
        # For Python 3.11, use ax-platform with caution
        "ax-platform>=0.3.0,<0.4.0",  # More constrained to avoid conflicts
        "packaging>=21.0,<24.0",  # Constrained to avoid ax conflicts
        "python-dateutil>=2.8.2,<2.9.0",
        "PyYAML>=6.0.1,<7.0.0",
    ]

# Conflicting packages - separate install option
conflict_prone_requirements = [
    "ax-platform",  # Has many version conflicts
]

# Additional problematic packages that should be separated
external_tool_requirements = [
    "spyder>=5.0.0",  # Newer version to avoid PyQt conflicts
    "selenium>=4.15.0,<4.25.0",  # Avoid typing-extensions conflicts
    "spotdl>=4.2.0,<4.2.5",  # Avoid fastapi/pydantic conflicts
]

# Web development packages with stricter constraints
web_dev_requirements = [
    "fastapi>=0.103.0,<0.104.0",  # spotdl requirement
    "pydantic>=2.6.4,<3.0.0",  # Modern pydantic for spotdl
    "platformdirs>=4.2.0,<5.0.0",  # spotdl requirement
]

# Type checking and extensions with updated constraints
typing_requirements = [
    "typing-extensions>=4.9.0",  # Updated for modern compatibility
]

# Python 3.13 minimal ML setup without TensorFlow (for early compatibility)
python313_ml_minimal_requirements = [
    "torch>=2.1.0",
    "torchvision>=0.16.0", 
    "torch-geometric",
    # Skip TensorFlow initially until stable release
    "scikit-learn>=1.6.0",
    "typing-extensions>=4.9.0",
    "numpy>=1.24,<2.3",
]

# Python 3.13 safe ML setup with stable TensorFlow (avoiding release candidates)
python313_ml_stable_requirements = [
    "torch>=2.1.0",
    "torchvision>=0.16.0", 
    "torch-geometric",
    "tensorflow>=2.15.0",  # Use latest stable instead of RC
    "scikit-learn>=1.6.0",
    "typing-extensions>=4.9.0",
    "numpy>=1.24,<2.3",
    # Let TensorFlow manage its own dependencies
]

# Special packages that need careful handling
special_requirements = [
    "pykan",  # May have its own conflicts
]

setup(
    name='bernn',
    version='0.6.18',
    packages=find_packages(),
    url='https://github.com/spell00/BERNN_MSMS',  # Replace with actual repo URL
    license='MIT',  # Choose appropriate license
    author='Simon Pelletier',
    author_email='',  # Add your email if you want
    description='Batch Effect Removal Neural Networks for Tandem Mass Spectrometry',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Avoid emitting License-File metadata for broader tool compatibility
    license_files=[],
    python_requires='>=3.10',
    install_requires=minimal_requirements,
    extras_require={
        # Basic installs
        'minimal': [],  # Just the base requirements
        'core-extended': optional_requirements + compatibility_requirements,

        # Feature-based installs
        'deep-learning': deep_learning_requirements,
        'experiment-tracking': experiment_tracking_requirements,
        'notebooks': notebook_requirements,
        'tools': tools_requirements,
        # Python 3.11 specific install option
        'python311-plus': deep_learning_requirements + experiment_tracking_requirements,
        'tools-with-ax': tools_requirements + conflict_prone_requirements,  # Include ax-platform
        'web': web_requirements,
        'web-dev': web_dev_requirements,  # Modern web dev with spotdl compatibility
        'external-tools': external_tool_requirements,  # spyder, selenium, spotdl
        'typing': typing_requirements,  # Updated typing-extensions
        'r-integration': r_integration_requirements,
        'dev-tools': dev_tools_requirements,
        'special': special_requirements,

        # Combined installs
        'ml-full': deep_learning_requirements + experiment_tracking_requirements,
        'analysis': notebook_requirements + tools_requirements + special_requirements,
        'analysis-with-ax': notebook_requirements + tools_requirements + conflict_prone_requirements + special_requirements,
        'development': optional_requirements + dev_tools_requirements + web_requirements,
        'modern-web': web_dev_requirements + typing_requirements,  # For spotdl compatibility
        'ide-tools': external_tool_requirements + typing_requirements,  # For spyder, selenium
        'python313-ml-minimal': python313_ml_minimal_requirements,  # Python 3.13 ML without TensorFlow
        'python313-ml-stable': python313_ml_stable_requirements,  # Python 3.13 ML with stable TensorFlow
        'python313-minimal-safe': [  # Python 3.13 absolutely minimal (avoid all complex dependencies)
            "torch>=2.1.0",
            "torchvision>=0.16.0", 
            "torch-geometric",
            "scikit-learn>=1.6.0",
            "pandas>=2.2.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.2",
            "numpy>=1.24,<2.3",
            "scipy>=1.11.0",
            "jupyter>=1.0.0",  # Simple jupyter instead of full jupyterlab
        ],
        'python313-safe': (  # Python 3.13 with safe packages only
            optional_requirements +
            compatibility_requirements +
            python313_ml_minimal_requirements +
            notebook_requirements +
            special_requirements
        ),

        # Full installs
        'full-no-ax': (
            optional_requirements +
            compatibility_requirements +
            deep_learning_requirements +
            experiment_tracking_requirements +
            notebook_requirements +
            tools_requirements +  # Without ax-platform
            web_requirements +
            r_integration_requirements +
            special_requirements
        ),
        'full': (  # Full install WITH ax-platform (may have conflicts)
            optional_requirements +
            compatibility_requirements +
            deep_learning_requirements +
            experiment_tracking_requirements +
            notebook_requirements +
            tools_requirements +
            conflict_prone_requirements +
            web_requirements +
            r_integration_requirements +
            special_requirements
        ),
        'full-safe': (  # Full install without potentially conflicting packages
            optional_requirements +
            deep_learning_requirements +
            experiment_tracking_requirements +
            notebook_requirements +
            special_requirements
        ),

        # Python version-specific installs
        'py311-plus': [] if python_version < (3, 11) else (
            deep_learning_requirements + experiment_tracking_requirements
        ),
        'py312-plus': [] if not is_python312_or_later else (
            deep_learning_requirements + experiment_tracking_requirements
        ),
        'py313-plus': [] if not is_python313_or_later else (
            deep_learning_requirements + experiment_tracking_requirements
        ),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        'console_scripts': [
            'bernn-train-ae=bernn.dl.train.train_ae:main',
            'bernn-train-ae-classifier=bernn.dl.train.train_ae_classifier_holdout:main',
            'bernn-train-ae-then-classifier=bernn.dl.train.train_ae_then_classifier_holdout:main',
        ],
    }
)


