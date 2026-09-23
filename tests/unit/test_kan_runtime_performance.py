"""Regression tests for KAN runtime acceleration settings."""

from types import SimpleNamespace

import pytest
import torch

from bernn.config.training_config import TrainingConfig
from bernn.dl.models.pytorch.ekan.src.efficient_kan.kan import KANLinear
from bernn.dl.train.train_ae import TrainAE


@pytest.mark.unit
def test_kan_activity_counters_are_device_buffers_not_checkpoint_state():
    layer = KANLinear(4, 3, device="cpu")
    assert layer.counts.device == layer.base_weight.device
    assert layer.n.device == layer.base_weight.device
    assert "counts" not in layer.state_dict()
    assert "n" not in layer.state_dict()

    x = torch.randn(5, 4)
    _ = layer(x)

    assert layer.n.item() == pytest.approx(1.0)
    assert torch.count_nonzero(layer.counts).item() > 0

    layer.restore_counts()
    assert layer.n.item() == 0
    assert torch.count_nonzero(layer.counts).item() == 0


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_kan_initializes_directly_on_cuda():
    layer = KANLinear(4, 3, device="cuda")
    assert layer.base_weight.is_cuda
    assert layer.spline_weight.is_cuda
    assert layer.grid.is_cuda
    assert layer.counts.is_cuda


@pytest.mark.unit
def test_runtime_defaults_keep_bf16_and_optional_accelerators_off():
    cfg = TrainingConfig()
    assert cfg.train_only_warmup is False
    assert cfg.precision == "bf16"
    assert cfg.tf32 is False
    assert cfg.torch_compile is False
    assert cfg.torch_compile_mode == "default"
    assert cfg.cpu_threads == 0


@pytest.mark.unit
def test_fp32_disables_autocast_even_when_requested_device_is_cuda():
    trainer = object.__new__(TrainAE)
    trainer.args = SimpleNamespace(device="cuda", precision="fp32")
    context = trainer.autocast_context()
    assert context.__class__.__name__ == "nullcontext"
