"""
Unit tests for the DL module.
"""
import pytest
import numpy as np
import torch
from bernn.dl import models

def test_model_initialization():
    """Test model initialization."""
    # Test model creation
    model = models.BERNNModel(input_dim=10, hidden_dim=64, output_dim=2)
    assert isinstance(model, torch.nn.Module)
    assert model.input_dim == 10
    assert model.hidden_dim == 64
    assert model.output_dim == 2

def test_model_forward_pass():
    """Test model forward pass."""
    model = models.BERNNModel(input_dim=10, hidden_dim=64, output_dim=2)
    x = torch.randn(32, 10)  # batch size of 32
    output = model(x)
    assert output.shape == (32, 2)
    assert not torch.isnan(output).any()

def test_model_training_step():
    """Test model training step."""
    model = models.BERNNModel(input_dim=10, hidden_dim=64, output_dim=2)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    loss = model.training_step(x, y)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss) 