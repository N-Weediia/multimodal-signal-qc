from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("import_raw_subset", ROOT / "scripts" / "import_raw_subset.py")
assert SPEC and SPEC.loader
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)


def test_parse_subjects_rejects_non_numeric_and_selects_only_numeric(tmp_path: Path) -> None:
    (tmp_path / "123").mkdir()
    (tmp_path / "abc").mkdir()
    assert IMPORTER.parse_subjects("all", tmp_path) == ["123"]
    with pytest.raises(ValueError):
        IMPORTER.parse_subjects("123,../escape", tmp_path)


def test_allowed_extensions_include_metadata_and_emg_container_formats() -> None:
    assert {".json", ".hpf"}.issubset(IMPORTER.ALLOWED_EXTENSIONS)


def test_dry_run_preserves_destination_and_filters_extensions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "data"
    (source / "1001" / "EEG").mkdir(parents=True)
    (source / "1001" / "EEG" / "record.bdf").write_bytes(b"raw")
    (source / "1001" / "EEG" / "ignore.bin").write_bytes(b"ignore")

    records = IMPORTER.plan_copy(source, destination, ["1001"], apply=False)

    assert [record.status for record in records] == ["planned"]
    assert records[0].destination_relative_path == "1001/EEG/record.bdf"
    assert not destination.exists()


def test_conflict_is_never_overwritten_and_valid_copy_hash_is_verified(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "data"
    (source / "1001").mkdir(parents=True)
    (source / "1001" / "same.csv").write_bytes(b"source")
    (source / "1001" / "conflict.csv").write_bytes(b"source")
    (destination / "1001").mkdir(parents=True)
    (destination / "1001" / "same.csv").write_bytes(b"source")
    (destination / "1001" / "conflict.csv").write_bytes(b"different")

    records = IMPORTER.plan_copy(source, destination, ["1001"], apply=True)
    statuses = {Path(record.source_relative_path).name: record.status for record in records}

    assert statuses == {"conflict.csv": "conflict", "same.csv": "existing_same_hash"}
    assert (destination / "1001" / "conflict.csv").read_bytes() == b"different"
    assert hashlib.sha256((destination / "1001" / "same.csv").read_bytes()).hexdigest() == next(record.source_sha256 for record in records if record.source_relative_path.endswith("same.csv"))
