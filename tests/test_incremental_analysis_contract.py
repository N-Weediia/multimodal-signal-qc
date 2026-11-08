from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from scripts.build_incremental_analysis_contract import build_contract
from preprocess import PREPROCESS_PIPELINE_VERSION, _preprocess_cache_is_current, _preprocess_parameter_hash, _sha256_file


def test_contract_expands_movement_recordings_without_filename_task_inference() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "data"
        recording = root / "P01" / "FNIRS" / "运动" / "copied_静息态.nirs"
        recording.parent.mkdir(parents=True)
        recording.write_bytes(b"nirs")
        output = Path(directory) / "out"
        contract = build_contract(root, output, Path(directory) / "missing_analysis")
        with (output / "task_qc_ledger.csv").open(encoding="utf-8-sig", newline="") as handle:
            ledger = list(csv.DictReader(handle))
    assert contract["rows"] == 3
    assert {row["task_name"] for row in ledger} == {"摸头", "摸膝", "摸腰"}
    assert all(row["task"] == "运动任务" for row in ledger)


def test_contract_records_emg_mapping_and_missing_policy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "data"
        recording = root / "P01" / "base_line" / "EMG" / "record.csv"
        recording.parent.mkdir(parents=True)
        recording.write_text('T[s],"Avanti sensor 2: EMG 2 (IM)"\n0,0\n', encoding="utf-8")
        output = Path(directory) / "out"
        contract = build_contract(root, output, Path(directory) / "missing_analysis")
    assert contract["emg"]["expected_sensor_ids"] == [2, 3, 4, 5, 9, 10, 11]
    assert "sensor 1" in contract["emg"]["missing_sensor_policy"]


def test_preprocess_cache_requires_source_and_parameter_hashes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.bin"
        source.write_bytes(b"stable")
        destination = root / "out"
        destination.mkdir()
        (destination / "preprocessing_summary.json").write_text(
            '{"source_sha256": "' + _sha256_file(source) + '", '
            '"preprocess_pipeline_version": "' + PREPROCESS_PIPELINE_VERSION + '", '
            '"preprocess_parameter_hash": "' + _preprocess_parameter_hash() + '"}',
            encoding="utf-8",
        )
        assert _preprocess_cache_is_current(destination, source)
        source.write_bytes(b"changed")
        assert not _preprocess_cache_is_current(destination, source)
