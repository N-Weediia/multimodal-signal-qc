from pathlib import Path

from scripts.segment_eeg_quality import movement_recording


def test_movement_recording_accepts_movement_task_folder_and_largest_bdf(tmp_path: Path) -> None:
    folder = tmp_path / "2026081502" / "base_line" / "EEG" / "运动任务"
    folder.mkdir(parents=True)
    (folder / "evt.bdf").write_bytes(b"event")
    (folder / "data.1.bdf").write_bytes(b"short")
    expected = folder / "formal.bdf"
    expected.write_bytes(b"longest-recording")

    assert movement_recording(tmp_path, "2026081502", "base_line") == expected


def test_movement_recording_returns_none_without_motor_directory(tmp_path: Path) -> None:
    (tmp_path / "2026081502" / "base_line" / "EEG" / "静息态").mkdir(parents=True)

    assert movement_recording(tmp_path, "2026081502", "base_line") is None
