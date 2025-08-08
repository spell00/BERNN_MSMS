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
is_python39_or_earlier = python_version < (3, 10)
is_python312_or_later = python_version >= (3, 12)

# Define core/minimal requirements (compatible with Python 3.8+)
# These are carefully chosen to minimize conflicts
if is_python312_or_later:
    # For Python 3.12+ - use newer versions that support Python 3.12
    minimal_requirements = [
        "scikit-learn>=1.3.0",  # Python 3.12 compatible
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
        "shapely>=2.0.0",
        "numba>=0.58.0",  # Python 3.12 compatible
        "openpyxl>=3.0.10",
        "xgboost>=1.7.0",  # Python 3.12 compatible
        "importlib-metadata>=6.0.0",
        "threadpoolctl>=3.1.0",
        "protobuf>=4.21.0,<5.0.0",
        "requests>=2.31.0,<3.0.0",
        "PyYAML>=6.0.1",
        "python-dateutil>=2.8.2",
        "nbformat>=5.9.2",
        "statsmodels",
    ]
else:
    # For Python 3.8-3.11 - use compatible versions
    minimal_requirements = [
        "scikit-learn>=1.0.2,<1.2.0",  # Compatible with most packages
        "pandas>=1.4.4,<1.6.0",  # Avoid conflicts with pyfume
        "scikit-optimize>=0.9.0",
        "matplotlib>=3.6.3",
        "seaborn>=0.12.2",
        "tabulate>=0.9.0",
        "scipy>=1.9.1,<1.11.0",  # Balance between compatibility and features
        "tqdm",
        "joblib>=1.2.0",
        "psutil>=5.9.4",
        "scikit-image",
        "nibabel",
        "mpmath>=1.3.0",
        "patsy>=0.5.3",
        "umap-learn>=0.5.3",
        "shapely>=2.0.0",
        "numba>=0.57.1",
        "openpyxl>=3.0.10",
        "xgboost>=1.0.0,<2.0.0",
        "importlib-metadata>=3.7.0,<8",
        "threadpoolctl>=3.1.0",
        "protobuf>=3.20.3,<5.0.0",  # Avoid conflicts with newer versions
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
    "rpy2>=3.5.7",  # Only if R integration is needed
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
if is_python312_or_later:
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
elif is_python39_or_earlier:
    # For Python 3.8-3.9 - use TensorFlow 2.13 but with flexible typing-extensions
    deep_learning_requirements = [
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "torch-geometric",
        "tensorflow>=2.13.0,<2.14.0",  # TensorFlow 2.13 for Python 3.8 compatibility
        "tensorflow-estimator>=2.13.0,<2.14.0",
        "typing-extensions>=4.5.0",  # Flexible constraint - let pip resolve
        "numpy<2.3,>=1.24",  # Compatible with TensorFlow 2.13
        "gast>=0.2.1,<=0.4.0",  # TensorFlow requirement
    ]
else:
    # For Python 3.10-3.11 - use newer versions
    deep_learning_requirements = [
        "torch>=2.1.0",  # Newer version for Python 3.10+
        "torchvision>=0.16.0",
        "torch-geometric",
        "tensorflow>=2.15.0",  # Newer TensorFlow for Python 3.10+
        "tensorflow-estimator>=2.15.0",
        "typing-extensions>=4.9.0",  # Updated for compatibility
        "numpy>=1.24",  # Newer numpy for Python 3.10+
    ]

# Experiment tracking dependencies - version-specific with conflict resolution
if is_python312_or_later:
    experiment_tracking_requirements = [
        "tensorboard>=2.15.0",
        "tensorboard-data-server>=0.7.0",
        "tensorboardX",
        "neptune",
        "mlflow[extras]>=2.12.1",  # Python 3.12 compatible
        "sqlalchemy>=2.0.0",
        "urllib3>=1.26.7",
    ]
elif is_python39_or_earlier:
    experiment_tracking_requirements = [
        "tensorboard>=2.13.0,<2.14.0",  # Compatible with TensorFlow 2.13
        "tensorboard-data-server>=0.7.0,<0.8.0",
        "tensorboardX",
        "neptune",
        "mlflow[extras]>=2.12.1,<2.13.0",  # Avoid SQLAlchemy conflicts
        "sqlalchemy>=1.4.0,<3.0.0",  # Compatible with MLflow
        "urllib3>=1.26.7,<2.0.0",  # Avoid databricks-cli conflicts
    ]
else:
    experiment_tracking_requirements = [
        "tensorboard>=2.15.0",
        "tensorboard-data-server>=0.7.0",
        "tensorboardX",
        "neptune",
        "mlflow[extras]>=2.15.0",
        "sqlalchemy>=1.4.0",
        "urllib3>=1.26.7",
    ]

# Notebook dependencies - version-specific
if is_python312_or_later:
    notebook_requirements = [
        "notebook>=7.0.0",  # Python 3.12 compatible
        "ipywidgets>=8.0.0",
        "jupyterlab>=4.0.0",
    ]
elif is_python39_or_earlier:
    notebook_requirements = [
        "notebook==6.5.6",
        "ipywidgets==7.7.5",
        "jupyterlab==3.6.6",
    ]
else:
    notebook_requirements = [
        "notebook>=6.5.6",
        "ipywidgets>=7.7.5",
        "jupyterlab>=3.6.6",
    ]

# Additional tools (with conflict resolution)
# Note: ax-platform has very strict version requirements that conflict with modern packages
if is_python312_or_later:
    tools_requirements = [
        # Exclude ax-platform for Python 3.12+ due to conflicts
        "ax-platform>=1.0.0",  # Causes too many conflicts
        "packaging>=21.0",
        "python-dateutil>=2.8.2",
        "PyYAML>=6.0.1",
        "optuna>=3.0.0",  # Alternative optimization library
    ]
elif is_python39_or_earlier:
    tools_requirements = [
        "ax-platform>=0.2.10,<0.3.0",  # Pinned for compatibility
        "packaging>=20.1,<=23.2",  # Avoid ax conflicts
        "python-dateutil>=2.8.1,<=2.8.2",  # Avoid ax conflicts
        "PyYAML>=5.1.2,<=6.0.1",  # Avoid ax conflicts
    ]
else:
    tools_requirements = [
        # For Python 3.10-3.11, use ax-platform with caution
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

# Python 3.8 specific requirements for environments with modern packages
python38_modern_requirements = [
    "typing-extensions>=4.6.0,<4.10.0",  # For environments with modern packages
]

# Python 3.8 minimal ML setup without conflicting experiment tracking
python38_ml_minimal_requirements = [
    "torch>=2.0.1",
    "torchvision>=0.15.2",
    "torch-geometric",
    # Skip TensorFlow to avoid typing-extensions conflicts
    "scikit-learn>=1.0.2,<1.2.0",
    "typing-extensions>=4.6.0,<4.10.0",
]

# Special packages that need careful handling
special_requirements = [
    "pykan",  # May have its own conflicts
]

setup(
    name='bernn',
    version='0.2.2',
    packages=find_packages(),
    url='https://github.com/username/BERNN_MSMS',  # Replace with actual repo URL
    license='MIT',  # Choose appropriate license
    author='Simon Pelletier',
    author_email='',  # Add your email if you want
    description='Batch Effect Removal Neural Networks for Tandem Mass Spectrometry',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
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
        'tools-with-ax': tools_requirements + conflict_prone_requirements,  # Include ax-platform
        'web': web_requirements,
        'web-dev': web_dev_requirements,  # Modern web dev with spotdl compatibility
        'external-tools': external_tool_requirements,  # spyder, selenium, spotdl
        'typing': typing_requirements,  # Updated typing-extensions
        'python38-modern': python38_modern_requirements,  # Python 3.8 with modern packages
        'python38-ml-minimal': python38_ml_minimal_requirements,  # Python 3.8 ML without TensorFlow
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
        'python38-full': (  # Python 3.8 with modern typing-extensions but selective ML
            optional_requirements +
            compatibility_requirements +
            python38_ml_minimal_requirements +  # PyTorch but no TensorFlow
            notebook_requirements +
            tools_requirements +
            python38_modern_requirements +
            special_requirements
        ),
        'python38-tensorflow': (  # Python 3.8 with TensorFlow (may have typing conflicts)
            optional_requirements +
            deep_learning_requirements +  # Includes TensorFlow 2.13
            experiment_tracking_requirements +
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
        'py38': [] if not (python_version >= (3, 8) and python_version < (3, 9)) else (
            deep_learning_requirements + experiment_tracking_requirements + python38_modern_requirements
        ),
        'py39': [] if not is_python39_or_earlier else (
            deep_learning_requirements + experiment_tracking_requirements
        ),
        'py310-plus': [] if is_python39_or_earlier else (
            deep_learning_requirements + experiment_tracking_requirements
        ),
        'py312-plus': [] if not is_python312_or_later else (
            deep_learning_requirements + experiment_tracking_requirements
        ),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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


