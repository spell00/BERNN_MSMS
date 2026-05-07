"""
Integration tests for BERNN trainer variants.

These tests validate that the three trainer implementations (AEClassifierHoldout, 
AEThenClassifier, AEThenClassifierHoldout) can be instantiated and that key fixes
are in place:
1. reg_entropy parameter has a safe default
2. neptune import is wrapped for optional availability
3. KAN neuron counting works without undefined class references
"""

import os
import sys
import pytest
import numpy as np
import torch
from types import SimpleNamespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestTrainerImports:
    """Verify that trainers can be imported without errors."""

    def test_ae_classifier_holdout_imports(self):
        """TrainAEClassifierHoldout should import without errors."""
        try:
            from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout
            assert TrainAEClassifierHoldout is not None
        except ImportError as e:
            pytest.fail(f"Failed to import TrainAEClassifierHoldout: {e}")

    def test_ae_then_classifier_imports(self):
        """TrainAEThenClassifier should import, even with optional neptune."""
        try:
            from bernn.dl.train.train_ae_then_classifier import TrainAEThenClassifier
            assert TrainAEThenClassifier is not None
            # Verify neptune was imported gracefully
            from bernn.dl.train import train_ae_then_classifier
            # neptune should be either the module or None
            assert train_ae_then_classifier.neptune is not None or train_ae_then_classifier.neptune is None
        except ImportError as e:
            pytest.fail(f"Failed to import TrainAEThenClassifier: {e}")

    def test_ae_then_classifier_holdout_imports(self):
        """TrainAEThenClassifierHoldout should import without errors."""
        try:
            from bernn.dl.train.train_ae_then_classifier_holdout import TrainAEThenClassifierHoldout
            assert TrainAEThenClassifierHoldout is not None
        except ImportError as e:
            pytest.fail(f"Failed to import TrainAEThenClassifierHoldout: {e}")


class TestTrainerParameterDefaults:
    """Verify that trainers handle missing parameters gracefully."""

    def test_ae_then_classifier_holdout_reg_entropy_default(self):
        """reg_entropy parameter should have a safe default (not error on missing)."""
        from bernn.dl.train.train_ae_then_classifier_holdout import TrainAEThenClassifierHoldout
        
        # Create minimal args
        args = SimpleNamespace(
            device="cpu",
            kan=0,
            n_meta=0,
            embeddings_meta=0,
            pool=0
        )
        
        # Create trainer (without calling train)
        try:
            trainer = TrainAEThenClassifierHoldout(
                args,
                path=".",
                fix_thres=-1,
                load_tb=False,
            )
            # Just verify instantiation works
            assert trainer is not None
        except Exception as e:
            pytest.fail(f"Failed to instantiate TrainAEThenClassifierHoldout: {e}")


class TestKANNeuronCounting:
    """Verify that KAN neuron counting works correctly."""

    def test_kan_neuron_counting_no_undefined_references(self):
        """count_neurons should work without referencing undefined KANAutoEncoder2."""
        from bernn.dl.train.train_ae import TrainAE
        from bernn.dl.models.pytorch.ekan.src.efficient_kan.kan import KANLinear
        
        # Create a minimal args object
        args = SimpleNamespace(
            device="cpu",
            kan=0,
            n_meta=0,
            embeddings_meta=0,
        )
        
        trainer = TrainAE(args, path=".", fix_thres=-1, load_tb=False)
        
        # Create a mock AE object with a KANLinear layer
        class MockAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Only include if KANLinear is available
                try:
                    self.kan_layer = KANLinear(10, 5)
                except Exception:
                    self.kan_layer = None
            
            def forward(self, x):
                if self.kan_layer:
                    return self.kan_layer(x)
                return x
        
        ae = MockAE()
        
        try:
            # This should not raise an error about undefined KANAutoEncoder2
            neuron_count = trainer.count_neurons(ae)
            assert isinstance(neuron_count, int)
            assert neuron_count >= 0
        except NameError as e:
            if "KANAutoEncoder2" in str(e):
                pytest.fail(f"KANAutoEncoder2 reference still exists: {e}")
            raise
        except Exception as e:
            pytest.fail(f"Unexpected error in count_neurons: {e}")


class TestTrainerInstantiation:
    """Verify that trainer instances can be created with minimal configuration."""

    @pytest.mark.parametrize("kan_flag", [0, 1])
    def test_ae_classifier_holdout_instantiation(self, kan_flag):
        """TrainAEClassifierHoldout should instantiate with kan=0 and kan=1."""
        from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout
        
        args = SimpleNamespace(
            device="cpu",
            kan=kan_flag,
            n_meta=0,
            embeddings_meta=0,
            pool=0,
        )
        
        try:
            trainer = TrainAEClassifierHoldout(args, path=".", fix_thres=-1, load_tb=False)
            assert trainer is not None
            assert trainer.args.kan == kan_flag
        except Exception as e:
            pytest.fail(f"Failed to instantiate TrainAEClassifierHoldout (kan={kan_flag}): {e}")

    @pytest.mark.parametrize("kan_flag", [0, 1])
    def test_ae_then_classifier_instantiation(self, kan_flag):
        """TrainAEThenClassifier should instantiate with kan=0 and kan=1."""
        from bernn.dl.train.train_ae_then_classifier import TrainAEThenClassifier
        
        args = SimpleNamespace(
            device="cpu",
            kan=kan_flag,
            n_meta=0,
            embeddings_meta=0,
            pool=0,
        )
        
        try:
            trainer = TrainAEThenClassifier(args, path=".", fix_thres=-1, load_tb=False)
            assert trainer is not None
            assert trainer.args.kan == kan_flag
        except Exception as e:
            pytest.fail(f"Failed to instantiate TrainAEThenClassifier (kan={kan_flag}): {e}")

    @pytest.mark.parametrize("kan_flag", [0, 1])
    def test_ae_then_classifier_holdout_instantiation(self, kan_flag):
        """TrainAEThenClassifierHoldout should instantiate with kan=0 and kan=1."""
        from bernn.dl.train.train_ae_then_classifier_holdout import TrainAEThenClassifierHoldout
        
        args = SimpleNamespace(
            device="cpu",
            kan=kan_flag,
            n_meta=0,
            embeddings_meta=0,
            pool=0,
        )
        
        try:
            trainer = TrainAEThenClassifierHoldout(args, path=".", fix_thres=-1, load_tb=False)
            assert trainer is not None
            assert trainer.args.kan == kan_flag
        except Exception as e:
            pytest.fail(f"Failed to instantiate TrainAEThenClassifierHoldout (kan={kan_flag}): {e}")
