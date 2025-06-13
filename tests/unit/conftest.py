"""
Common test fixtures and configurations for BERNN-MSMS unit tests.
"""
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = np.random.randn(100, 10)
    return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(10)])

@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=100) 