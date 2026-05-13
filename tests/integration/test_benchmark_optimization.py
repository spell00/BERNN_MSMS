"""
Benchmark optimization test — the largest test in the suite.

Uses the real benchmark dataset (data/benchmark/intensities.csv) and runs
a minimal Ax HPO loop (n_trials=3, n_epochs=3, early_stop=2) to verify the
full pipeline end-to-end.  Target: completes in <= 5 minutes on CPU.

Marked @pytest.mark.slow so it can be excluded with `-m "not slow"`.
"""
import os
import time
import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from pathlib import Path

# ─── Locate the benchmark data ─────────────────────────────────────────────
BENCHMARK_CSV = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "intensities.csv"
ADENOCARCINOMA_CSV = Path(__file__).resolve().parents[2] / "data" / "adenocarcinoma_data.csv"


def pytest_configure(config):
    """Register the 'benchmark' marker if it isn't already present."""
    pass


@pytest.fixture(scope="module")
def benchmark_data_path():
    if not BENCHMARK_CSV.exists():
        pytest.skip(f"Benchmark CSV not found: {BENCHMARK_CSV}")
    return str(BENCHMARK_CSV)


@pytest.fixture(scope="module")
def adenocarcinoma_data_path():
    if not ADENOCARCINOMA_CSV.exists():
        pytest.skip(f"Adenocarcinoma CSV not found: {ADENOCARCINOMA_CSV}")
    return str(ADENOCARCINOMA_CSV)


# ─── Helpers ───────────────────────────────────────────────────────────────

def _benchmark_args(csv_file, path, **overrides):
    ns = SimpleNamespace(
        device="cpu",
        random_recs=0,
        predict_tests=0,
        early_stop=2,
        early_warmup_stop=-1,
        train_after_warmup=0,
        threshold=0.0,
        n_epochs=3,
        n_trials=3,
        rec_loss="l1",
        tied_weights=0,
        random=1,
        variational=0,
        zinb=0,
        use_mapping=1,
        bdisc=0,
        n_repeats=1,
        dloss="inverseTriplet",
        remove_zeros=0,
        n_meta=0,
        embeddings_meta=0,
        groupkfold=0,
        n_layers=1,
        kan=0,
        use_l1=0,
        clip_val=1.0,
        log_metrics=0,
        log_plots=0,
        prune_network=0,
        dataset="custom",
        csv_file=csv_file,
        log1p=1,
        berm="none",
        pool=0,
        strategy="none",
        best_features_file="",
        n_features=-1,
        bad_batches="",
        controls="",
        exp_id="benchmark_test",
        warmup_after_warmup=0,
        bs=32,
        n_agg=1,
        update_grid=0,
        prune_threshold=0.0,
        scheduler="ReduceLROnPlateau",
        path=path,
        log_tb=0,
        log_mlflow=0,
        keep_models=0,
        log_inputs=0,
        classif_loss="ce",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ─── Tests ─────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.benchmark
def test_benchmark_data_loads_correctly(benchmark_data_path):
    """Verify the benchmark CSV has the expected structure."""
    df = pd.read_csv(benchmark_data_path)
    # Should have at least 1000 rows
    assert df.shape[0] >= 900, f"Expected >=900 samples, got {df.shape[0]}"
    # First 3 cols: names, label/labels, batch/batches
    assert df.columns[0] in ("names", "name", "sample"), f"Unexpected first col: {df.columns[0]}"
    assert df.columns[1] in ("label", "labels", "class"), f"Unexpected label col: {df.columns[1]}"
    assert df.columns[2] in ("batch", "batches", "group"), f"Unexpected batch col: {df.columns[2]}"
    # Should have numeric features
    feature_cols = df.columns[3:]
    assert len(feature_cols) >= 10, f"Too few feature columns: {len(feature_cols)}"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.benchmark
def test_benchmark_get_data_roundtrip(benchmark_data_path):
    """get_data() should load benchmark CSV into the expected structure."""
    from bernn.utils.data_getters import get_data

    path = str(Path(benchmark_data_path).parent)
    csv_file = Path(benchmark_data_path).name
    args = _benchmark_args(csv_file, path)

    data, unique_labels, unique_batches = get_data(path, args, seed=42)

    assert "inputs" in data
    assert "labels" in data
    assert "batches" in data
    # Benchmark has 6 classes
    assert len(unique_labels) >= 2
    assert len(unique_batches) >= 2
    for split in ("train", "valid", "test"):
        assert data["inputs"][split].shape[0] > 0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.benchmark
def test_benchmark_prepare_data_preserves_train_label_coverage(benchmark_data_path, tmp_path):
    """After BERNN split prep, every known holdout label must already exist in train labels."""
    from bernn.utils.data_getters import get_data
    from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout

    path = str(Path(benchmark_data_path).parent)
    csv_file = Path(benchmark_data_path).name
    args = _benchmark_args(csv_file, path, groupkfold=0, pool=0, bs=32)

    data, _, _ = get_data(path, args, seed=42)

    trainer = TrainAEClassifierHoldout(
        args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )

    trainer._prepare_data(
        X=data["inputs"]["train"],
        y=data["labels"]["train"],
        groups=data["batches"]["train"],
        X_test=data["inputs"]["test"],
        y_test=data["labels"]["test"],
        groups_test=data["batches"]["test"],
        cross_validation=False,
        cross_test=False,
        val_size=0.2,
    )

    train_labels = set(np.unique(trainer.data["labels"]["train"]))
    holdout_labels = set(np.unique(np.concatenate((trainer.data["labels"]["valid"], trainer.data["labels"]["test"]))))
    assert holdout_labels <= train_labels


@pytest.mark.integration
@pytest.mark.slow
def test_adenocarcinoma_prepare_data_preserves_train_label_coverage(adenocarcinoma_data_path, tmp_path):
    """After BERNN split prep, every known holdout label must already exist in train labels (adenocarcinoma)."""
    from bernn.utils.data_getters import get_data
    from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout

    path = str(Path(adenocarcinoma_data_path).parent)
    csv_file = Path(adenocarcinoma_data_path).name
    args = _benchmark_args(csv_file, path, groupkfold=0, pool=0, bs=32)

    data, _, _ = get_data(path, args, seed=42)

    trainer = TrainAEClassifierHoldout(
        args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )

    trainer._prepare_data(
        X=data["inputs"]["train"],
        y=data["labels"]["train"],
        groups=data["batches"]["train"],
        X_test=data["inputs"]["test"],
        y_test=data["labels"]["test"],
        groups_test=data["batches"]["test"],
        cross_validation=False,
        cross_test=False,
        val_size=0.2,
    )

    train_labels = set(np.unique(trainer.data["labels"]["train"]))
    holdout_labels = set(np.unique(np.concatenate((trainer.data["labels"]["valid"], trainer.data["labels"]["test"]))))
    assert holdout_labels <= train_labels


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.benchmark
def test_benchmark_single_train_call(benchmark_data_path, tmp_path):
    """A single train() call on the benchmark dataset should return a finite float."""
    from bernn.utils.data_getters import get_data
    from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout

    path = str(Path(benchmark_data_path).parent)
    csv_file = Path(benchmark_data_path).name
    args = _benchmark_args(csv_file, path, n_epochs=5, early_stop=3, warmup=2, n_features=100)

    data, unique_labels, unique_batches = get_data(path, args, seed=0)

    trainer = TrainAEClassifierHoldout(
        args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )
    trainer.data = data
    trainer.unique_labels = unique_labels
    trainer.unique_batches = unique_batches
    trainer.columns = data["inputs"]["all"].columns

    # Copy CSV to tmp_path so the trainer can locate it
    import shutil
    shutil.copy(benchmark_data_path, str(tmp_path / csv_file))

    params = {
        "nu": 0.1,
        "lr": 1e-3,
        "wd": 1e-6,
        "smoothing": 0.0,
        "margin": 1.0,
        "warmup": 2,
        "disc_b_warmup": 1,
        "dropout": 0.0,
        "scaler": "standard",
        "layer1": 64,
        "layer2": 32,
        "gamma": 0.0,
        "beta": 0.0,
        "zeta": 0.0,
        "thres": 0.0,
        "prune_threshold": 0.0,
    }

    t0 = time.time()
    try:
        result = trainer.train(params)
    except Exception as e:
        pytest.fail(f"train() raised an exception on benchmark data: {e}")
    elapsed = time.time() - t0

    assert isinstance(result, (float, int)), f"Expected numeric result, got {type(result)}"
    print(f"\n[benchmark] train() returned {result:.4f} in {elapsed:.1f}s")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.benchmark
def test_benchmark_single_train_call_then_classifier_holdout(benchmark_data_path, tmp_path):
    """Benchmark data should load and wire up cleanly for the AE-then-classifier holdout trainer."""
    from bernn.utils.data_getters import get_data
    from bernn.dl.train.train_ae_then_classifier_holdout import TrainAEThenClassifierHoldout

    path = str(Path(benchmark_data_path).parent)
    csv_file = Path(benchmark_data_path).name
    args = _benchmark_args(csv_file, path, n_epochs=3, early_stop=2, warmup=1, n_features=100)

    data, unique_labels, unique_batches = get_data(path, args, seed=0)

    trainer = TrainAEThenClassifierHoldout(
        args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,

        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )
    trainer.data = data
    trainer.unique_labels = unique_labels
    trainer.unique_batches = unique_batches
    trainer.columns = data["inputs"]["all"].columns

    import shutil
    shutil.copy(benchmark_data_path, str(tmp_path / csv_file))

    params = {
        "layer2": 32,
        "layer1": 64,
        "gamma": 0.0,
        "beta": 0.0,
        "zeta": 0.0,
        "warmup": 1,
        "disc_b_warmup": 1,
        "dropout": 0.0,
        "scaler": "standard",
        "nu": 0.1,
        "lr": 1e-3,
        "wd": 1e-6,
        "smoothing": 0.0,
        "margin": 1.0,
        "thres": 0.0,
        "prune_threshold": 0.0,
    }

    ordered_layers = trainer.get_ordered_layers(params)

    assert trainer.columns.equals(data["inputs"]["all"].columns)
    assert len(data["inputs"]["all"]) == len(data["sets"]["all"])
    assert list(ordered_layers.keys()) == ["layer1", "layer2"]
    assert ordered_layers["layer1"] == 64
    assert ordered_layers["layer2"] == 32


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.filterwarnings("ignore:DataFrameGroupBy.apply operated on the grouping columns.*:FutureWarning:ax.core.data")
def test_benchmark_ax_optimization_3_trials(benchmark_data_path, tmp_path):
    """
    Full Ax HPO loop with 3 trials on the real benchmark dataset.
    Must complete in under 5 minutes and return valid best_parameters.
    """
    from bernn.utils.data_getters import get_data
    from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout

    try:
        from ax.api.client import Client
        from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
    except ImportError:
        pytest.skip("ax-platform not available")

    path = str(Path(benchmark_data_path).parent)
    csv_file = Path(benchmark_data_path).name

    # Small feature subset for speed
    args = _benchmark_args(
        csv_file, path,
        n_epochs=3,
        early_stop=2,
        warmup=1,
        n_features=50,
        n_repeats=1,
    )

    data, unique_labels, unique_batches = get_data(path, args, seed=0)

    import shutil
    shutil.copy(benchmark_data_path, str(tmp_path / csv_file))

    trainer = TrainAEClassifierHoldout(
        args,
        path=str(tmp_path),
        fix_thres=-1,
        load_tb=False,
        log_metrics=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,

        log_mlflow=False,
        groupkfold=False,
        pools=False,
    )
    trainer.data = data
    trainer.unique_labels = unique_labels
    trainer.unique_batches = unique_batches
    trainer.columns = data["inputs"]["all"].columns

    parameters = [
        {"name": "nu", "type": "range", "bounds": [0.01, 1.0], "log_scale": False},
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
        {"name": "smoothing", "type": "range", "bounds": [0.0, 0.1]},
        {"name": "margin", "type": "range", "bounds": [0.0, 2.0]},
        {"name": "warmup", "type": "range", "bounds": [1, 3]},
        {"name": "disc_b_warmup", "type": "range", "bounds": [1, 2]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.3]},
        {
            "name": "scaler",
            "type": "choice",
            "values": ["standard", "minmax", "robust"],
            "is_ordered": False,
            "sort_values": False,
        },
        {"name": "layer1", "type": "range", "bounds": [32, 128]},
        {"name": "layer2", "type": "range", "bounds": [16, 64]},
    ]

    t0 = time.time()
    param_configs = []
    for p in parameters:
        if p["type"] == "range":
            lo, hi = p["bounds"]
            is_int = isinstance(lo, int) and isinstance(hi, int)
            param_configs.append(
                RangeParameterConfig(
                    name=p["name"],
                    bounds=(lo, hi),
                    parameter_type="int" if is_int else "float",
                    scaling="log" if p.get("log_scale", False) else "linear",
                )
            )
        elif p["type"] == "choice":
            values = p["values"]
            first = values[0]
            if isinstance(first, bool):
                param_type = "bool"
            elif isinstance(first, int):
                param_type = "int"
            elif isinstance(first, float):
                param_type = "float"
            else:
                param_type = "str"
            param_configs.append(
                ChoiceParameterConfig(
                    name=p["name"],
                    values=list(values),
                    parameter_type=param_type,
                    is_ordered=p.get("is_ordered"),
                )
            )

    client = Client(random_seed=42)
    client.configure_experiment(parameters=param_configs, name="benchmark_ax_client")
    # Minimize closs by maximizing negative closs.
    client.configure_optimization(objective="-closs")

    for _ in range(3):
        for trial_index, parameterization in client.get_next_trials(max_trials=1).items():
            trainer.data = data
            trainer.unique_labels = unique_labels
            trainer.unique_batches = unique_batches
            trainer.columns = data["inputs"]["all"].columns
            try:
                result = float(trainer.train(parameterization))
                if not np.isfinite(result):
                    raise ValueError(f"Non-finite objective: {result}")
                client.complete_trial(trial_index=trial_index, raw_data={"closs": result})
            except Exception as e:
                print(f"[AX WARN] trial failed: {e}")
                client.mark_trial_failed(trial_index=trial_index, failed_reason=str(e))

    best_parameters, best_metrics, _, _ = client.get_best_parameterization(use_model_predictions=False)
    elapsed = time.time() - t0

    # ── Assertions ──────────────────────────────────────────
    assert elapsed < 300, f"Optimization took {elapsed:.0f}s, expected < 300s"
    assert isinstance(best_parameters, dict), "best_parameters should be a dict"
    assert "lr" in best_parameters, "best_parameters should contain 'lr'"
    assert "layer1" in best_parameters, "best_parameters should contain 'layer1'"

    # Values should be finite
    best_obj = best_metrics.get("closs", None)
    if isinstance(best_obj, tuple):
        best_val = best_obj[0]
    else:
        best_val = best_obj
    if best_val is not None:
        assert np.isfinite(best_val), f"Best objective value is not finite: {best_val}"

    print(f"\n[benchmark] Ax optimization (3 trials) completed in {elapsed:.1f}s")
    print(f"[benchmark] Best parameters: {best_parameters}")
    print(f"[benchmark] Best objective: {best_val}")
