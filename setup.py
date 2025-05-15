#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 2021
@author: Simon Pelletier
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name='bernn',
    version='0.1.7',
    packages=find_packages(),
    url='https://github.com/username/BERNN_MSMS',  # Replace with actual repo URL
    license='MIT',  # Choose appropriate license
    author='Simon Pelletier',
    author_email='',  # Add your email if you want
    description='Batch Effect Removal Neural Networks for Tandem Mass Spectrometry',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
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
