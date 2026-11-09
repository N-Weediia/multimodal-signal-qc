"""Project paths, session structures, and protocol definitions."""

from __future__ import annotations

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_REPRO_DIR = PROJECT_ROOT / "data_repro_20260903"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

# Known sessions & modalities
KNOWN_SESSIONS = ("base_line", "sec_test", "thi_test", "sec")
KNOWN_MODALITIES = ("EEG", "EMG", "FNIRS")

# Movement tasks & markers
MOVEMENT_TASKS = ("摸头", "摸膝", "摸腰", "握拳")
TASK_MARKER_MAP = {
    "摸头": 1,
    "摸膝": 2,
    "摸腰": 3,
    "握拳": 4,
}
