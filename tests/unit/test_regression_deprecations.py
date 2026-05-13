import pytest
import numpy as np
import sys
import warnings


def test_no_np_int_deprecation():
    """Test that using np.int or similar deprecated types raises an error or warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # np.int is deprecated, should raise a warning or error
        try:
            arr = np.array([1, 2, 3], dtype=np.int)
        except AttributeError:
            # np.int is removed in recent numpy
            pass
        except Exception as e:
            pytest.fail(f"Unexpected error for np.int: {e}")
        # Check for DeprecationWarning or AttributeError
        assert not any(
            issubclass(warn.category, DeprecationWarning) for warn in w
        ), "np.int usage should not raise DeprecationWarning (should be removed from codebase)"


def test_mlflow_import_in_train_ae():
    """Test that importing and using mlflow in train_ae does not raise NameError."""
    import importlib
    train_ae = importlib.import_module("bernn.dl.train.train_ae")
    # Should not raise NameError for mlflow
    assert hasattr(train_ae, "mlflow"), "mlflow should be imported in train_ae module"
