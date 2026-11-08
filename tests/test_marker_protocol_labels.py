from __future__ import annotations

import numpy as np

from explore_202608_cases import EventSeries, infer_emg_marker_rows, infer_movement_event_labels


def test_movement_markers_use_confirmed_protocol_labels() -> None:
    times = np.arange(26, dtype=float)
    labels = infer_movement_event_labels(times, "2026081403")

    assert labels[0]["inference_status"] == "observed"
    assert labels[2]["task_name"] == "摸头"
    assert labels[2]["inference_status"] == "protocol_labeled"
    assert labels[8]["task_name"] == "摸膝"
    assert labels[14]["task_name"] == "摸腰"
    assert labels[20]["task_name"] == "握拳"
    assert labels[20]["inference_status"] == "protocol_labeled"


def test_emg_markers_are_projected_from_reference_stream_frame_zero() -> None:
    reference = EventSeries(np.asarray([100.0, 101.0, 102.0]), "fNIRS observed")
    secondary = EventSeries(np.asarray([200.0, 201.0, 202.0]), "EEG observed")
    rows = infer_emg_marker_rows(
        "2026081403",
        reference,
        np.arange(5000, dtype=float) / 1000.0,
        1000.0,
        secondary,
    )

    assert rows[0]["emg_frame_index"] == 0
    assert rows[2]["task_name"] == "摸头"
    assert rows[2]["emg_frame_index"] == 2000
    assert rows[2]["alignment_status"] == "inferred_from_eeg_or_fnirs_marker"
    assert rows[2]["secondary_marker_disagreement_ms"] == 0.0
