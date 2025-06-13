"""
Unit tests for the utils module.
"""
import pytest
import numpy as np
import pandas as pd
from bernn.utils import utils

def test_data_validation(sample_data):
    """Test data validation functions."""
    # Test with valid data
    assert utils.validate_data(sample_data) is True
    
    # Test with invalid data (NaN values)
    invalid_data = sample_data.copy()
    invalid_data.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        utils.validate_data(invalid_data)

def test_data_preprocessing(sample_data):
    """Test data preprocessing functions."""
    # Test normalization
    normalized_data = utils.normalize_data(sample_data)
    assert normalized_data.shape == sample_data.shape
    assert np.allclose(normalized_data.mean(), 0, atol=1e-10)
    assert np.allclose(normalized_data.std(), 1, atol=1e-10)

def test_feature_selection(sample_data, sample_labels):
    """Test feature selection functions."""
    # Test feature importance calculation
    importance = utils.calculate_feature_importance(sample_data, sample_labels)
    assert len(importance) == sample_data.shape[1]
    assert all(0 <= imp <= 1 for imp in importance) 