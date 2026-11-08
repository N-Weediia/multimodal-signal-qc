"""Validation checks for the delegation-trial inventory and notes artifacts."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest


TRIAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "experiments"
    / "delegation_trial_20260901"
)
INVENTORY_PATH = TRIAL_DIR / "delegation_trial_inventory.csv"
NOTES_PATH = TRIAL_DIR / "delegation_trial_notes.md"

REQUIRED_COLUMNS = {
    "subject",
    "session",
    "movement_fnirs",
    "movement_emg",
    "movement_eeg",
    "eye_report",
    "four_diagnosis",
}
BOOLEAN_COLUMNS = {
    "movement_fnirs",
    "movement_emg",
    "movement_eeg",
    "eye_report",
    "four_diagnosis",
}
SUBJECT_PATTERN = re.compile(r"^2026\d{6}$")
ALLOWED_BOOLEAN_VALUES = {"True", "False", "0", "1"}


def test_delegation_trial_inventory_is_structurally_valid() -> None:
    """Validate the generated inventory once it is available."""
    if not INVENTORY_PATH.exists():
        pytest.skip(f"delegation-trial inventory not generated: {INVENTORY_PATH}")

    with INVENTORY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(fieldnames)
        assert not missing, f"inventory is missing required columns: {sorted(missing)}"
        assert len(fieldnames) == len(set(fieldnames)), "inventory contains duplicate column names"

        rows = list(reader)

    assert rows, "inventory must contain at least one data row"
    subject_sessions: list[tuple[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        subject = (row.get("subject") or "").strip()
        session = (row.get("session") or "").strip()
        assert SUBJECT_PATTERN.fullmatch(subject), (
            f"row {row_number} has invalid subject {subject!r}; expected ^2026\\d{{6}}$"
        )
        assert session, f"row {row_number} has an empty session"
        subject_sessions.append((subject, session))

        for column in BOOLEAN_COLUMNS:
            value = (row.get(column) or "").strip()
            assert value in ALLOWED_BOOLEAN_VALUES, (
                f"row {row_number}, column {column} has invalid boolean value {value!r}"
            )

    assert len(subject_sessions) == len(set(subject_sessions)), (
        "subject/session combinations must be unique"
    )


def test_delegation_trial_notes_define_model_exclusion_and_process_metrics() -> None:
    """Ensure the notes document records the intended modeling boundary."""
    if not NOTES_PATH.exists():
        pytest.skip(f"delegation-trial notes not generated: {NOTES_PATH}")

    notes = NOTES_PATH.read_text(encoding="utf-8-sig")
    normalized = re.sub(r"\s+", "", notes).lower()

    eye_total_markers = (
        "眼动总分",
        "眼动报告总分",
        "eye_total_score",
        "eyetotalscore",
        "eye total score",
    )
    exclusion_markers = (
        "不入模",
        "不进入模型",
        "不纳入模型",
        "不作为模型",
        "notincludedinthemodel",
        "excludedfromthemodel",
    )
    has_eye_total = any(marker.lower() in normalized for marker in eye_total_markers)
    has_exclusion = any(marker.lower() in normalized for marker in exclusion_markers)
    assert has_eye_total and has_exclusion, (
        "notes must explicitly state that the eye-movement total score is not entered into the model"
    )

    has_process_metrics = (
        "量化过程指标" in normalized
        or ("量化" in normalized and "过程指标" in normalized)
        or "quantifiedprocessmetric" in normalized
        or "quantitativeprocessmetric" in normalized
    )
    assert has_process_metrics, "notes must mention quantitative process metrics"
