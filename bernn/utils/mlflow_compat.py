"""Compatibility wrapper for optional MLflow logging.

BERNN should be importable and testable even when MLflow or one of MLflow's
optional transitive dependencies is broken in the runtime environment. Modules
that only log to MLflow can import ``mlflow`` from here and safely call common
logging APIs; when MLflow is unavailable, calls become no-ops.
"""

from __future__ import annotations


class _NoOpRun:
    info = None
    data = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _NoOpMLflow:
    """Small no-op stand-in covering the MLflow APIs BERNN uses."""

    def active_run(self):
        return None

    def start_run(self, *args, **kwargs):
        return _NoOpRun()

    def end_run(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop


try:  # Import can fail from optional dependency skew, not only ImportError.
    import mlflow as mlflow  # type: ignore
    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - depends on external environment
    mlflow = _NoOpMLflow()
    MLFLOW_AVAILABLE = False
