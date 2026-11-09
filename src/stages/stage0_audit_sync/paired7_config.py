"""Frozen configuration for the 2026-09-03 paired-seven refresh."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_REPRO_ROOT = PROJECT_ROOT / "data_repro_20260903"
RAW_DATA_ROOT = LOCAL_REPRO_ROOT if LOCAL_REPRO_ROOT.is_dir() else Path(r"J:\2026二康\数据")
CANONICAL_ROOT = PROJECT_ROOT / "outputs" / "analysis" / "20260903_all_cases_canonical"
RUN_ROOT = PROJECT_ROOT / "outputs" / "analysis" / "paired7_20260903"
SEGMENT_EEG_ROOT = PROJECT_ROOT / "outputs" / "segment_eeg_quality_paired7_20260903"
RELEASE_ROOT = PROJECT_ROOT / "outputs" / "release_20260903_paired7"

PAIRED_SUBJECTS = (
    "2026080901",
    "2026081401",
    "2026081403",
    "2026081501",
    "2026081502",
    "2026081503",
    "2026081701",
)
SESSIONS = ("base_line", "sec_test")
TASKS = ("摸头", "摸膝", "摸腰")


def phase_root(number: int | str) -> Path:
    return RUN_ROOT / f"phase{number}"
