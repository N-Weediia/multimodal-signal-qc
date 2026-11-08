from __future__ import annotations

import csv
from pathlib import Path

import pytest


LEDGER_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase1_20260901"
LEDGER_FILES = {
    "identity": "subject_identity.csv",
    "sessions": "subject_session_ledger.csv",
    "modality": "modality_availability.csv",
    "eye": "eye_report_inventory.csv",
    "four": "four_diagnosis_inventory.csv",
}


def _read(name: str) -> list[dict[str, str]]:
    path = LEDGER_DIR / LEDGER_FILES[name]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_phase1_ledgers_have_expected_coverage_and_identity() -> None:
    missing = [str(LEDGER_DIR / filename) for filename in LEDGER_FILES.values() if not (LEDGER_DIR / filename).is_file()]
    if missing:
        pytest.skip("phase-one ledger(s) not generated: " + ", ".join(missing))

    identity = _read("identity")
    sessions = _read("sessions")
    modality = _read("modality")
    eye = _read("eye")
    four = _read("four")

    assert len(identity) == 14
    assert len(sessions) == 20
    assert sum(row.get("paired_baseline_sec_test") == "1" for row in identity) == 5
    excluded = [row for row in sessions if row.get("analysis_inclusion") == "exclude_duplicate"]
    assert len(excluded) == 1
    assert excluded[0]["subject"] == "2026081403"
    assert excluded[0]["session"] == "sec"
    assert excluded[0]["duplicate_session_of"] == "sec_test"

    assert len(modality) == 120
    assert len(eye) == 25
    assert len({(row["subject"], row["session"]) for row in eye}) == 15
    assert len(four) == 12
    assert len({(row["subject"], row["session"]) for row in four}) == 12

    assert all(row.get("name_match_status") == "match" for row in identity)
    assert all(row.get("name_match_status") == "match" for row in modality)
