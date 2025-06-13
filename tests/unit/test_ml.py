"""
Unit tests for the ML module.
"""
import pytest
import numpy as np
from bernn.ml import import_data

def test_data_import():
    """Test data import functionality."""
    # Test loading sample data
    data = import_data.load_sample_data()
    assert data is not None
    assert isinstance(data, dict)
    assert 'features' in data
    assert 'labels' in data

def test_data_preprocessing():
    """Test data preprocessing in ML module."""
    # Test data cleaning
    raw_data = np.random.randn(100, 10)
    cleaned_data = import_data.clean_data(raw_data)
    assert cleaned_data.shape == raw_data.shape
    assert not np.isnan(cleaned_data).any()

def test_feature_engineering():
    """Test feature engineering functions."""
    # Test feature extraction
    data = np.random.randn(100, 10)
    features = import_data.extract_features(data)
    assert features.shape[0] == data.shape[0]
    assert features.shape[1] >= data.shape[1]  # Should have at least original features 