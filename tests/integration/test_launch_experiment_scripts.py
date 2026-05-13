import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

TRAIN_AE_CLF = ROOT / "bernn/dl/train/train_ae_classifier_holdout.py"
TRAIN_AE_THEN = ROOT / "bernn/dl/train/train_ae_then_classifier_holdout.py"


@pytest.mark.integration
def test_train_entrypoints_exist_and_are_python_files():
    for train_script in (TRAIN_AE_CLF, TRAIN_AE_THEN):
        assert train_script.exists()
        assert train_script.suffix == ".py"


def _extract_declared_flags(train_script: Path) -> set[str]:
    text = train_script.read_text(encoding="utf-8")
    declared = set()
    for match in re.finditer(r"parser\.add_argument\('\-\-([a-zA-Z0-9_]+)'", text):
        declared.add(match.group(1))
    return declared


@pytest.mark.integration
def test_expected_logging_flags_are_supported_by_trainers():
    expected_flags = {"log_mlflow", "log_tb"}

    declared_ae_clf = _extract_declared_flags(TRAIN_AE_CLF)
    declared_ae_then = _extract_declared_flags(TRAIN_AE_THEN)

    missing_ae_clf = sorted(expected_flags - declared_ae_clf)
    missing_ae_then = sorted(expected_flags - declared_ae_then)

    assert missing_ae_clf == [], f"Missing expected flags in {TRAIN_AE_CLF.name}: {missing_ae_clf}"
    assert missing_ae_then == [], f"Missing expected flags in {TRAIN_AE_THEN.name}: {missing_ae_then}"


@pytest.mark.integration
def test_python_entrypoints_prefer_mlflow_defaults():
    text_ae_clf = TRAIN_AE_CLF.read_text(encoding="utf-8")
    text_ae_then = TRAIN_AE_THEN.read_text(encoding="utf-8")

    assert "parser.add_argument('--log_mlflow', type=int, default=1" in text_ae_clf
    assert "parser.add_argument('--log_mlflow', type=int, default=1" in text_ae_then
