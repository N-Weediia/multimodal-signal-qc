"""Synthetic guards for the reviewed MBLL branch; no participant data needed."""
from __future__ import annotations

import numpy as np
import pytest

import mbll_relative_sensitivity_reviewed_20260908 as reviewed


def _natural_design(dpf: np.ndarray) -> np.ndarray:
    return np.asarray([
        [reviewed.EXTINCTION_E_NATURAL[730.0]["hbo"], reviewed.EXTINCTION_E_NATURAL[730.0]["hbr"]],
        [reviewed.EXTINCTION_E_NATURAL[850.0]["hbo"], reviewed.EXTINCTION_E_NATURAL[850.0]["hbr"]],
    ]) * dpf[:, None]


def test_known_hbo_hbr_is_recovered_from_natural_od_once() -> None:
    dpf = np.asarray([6.0, 6.0])
    expected_uM = np.asarray([2.25, -0.75])
    distance_cm = np.asarray([5.0])
    od = (_natural_design(dpf) @ (expected_uM * 1e-6) * distance_cm[0])[None, :, None]
    recovered, _ = reviewed.unmix_natural_od(od, dpf, distance_cm)
    np.testing.assert_allclose(recovered[0, :, 0], expected_uM, rtol=1e-10, atol=1e-10)


def test_exact_wavelength_order_and_decadic_to_natural_conversion() -> None:
    assert reviewed.EXTINCTION_E_DEC[730.0] == {"hbo": 390.0, "hbr": 1102.2}
    assert reviewed.EXTINCTION_E_DEC[850.0] == {"hbo": 1058.0, "hbr": 691.32}
    assert reviewed.EXTINCTION_E_NATURAL[730.0]["hbo"] == pytest.approx(np.log(10.0) * 390.0)
    reviewed.validate_wavelength_order([(730.0, 850.0), (730.0, 850.0)])
    with pytest.raises(ValueError, match="Expected raw column order"):
        reviewed.validate_wavelength_order([(850.0, 730.0)])
    # If a caller nonetheless swapped the OD wavelength axis downstream, it
    # cannot silently reproduce the known concentration truth.
    dpf = np.asarray([6.0, 6.0])
    truth = np.asarray([1.5, -0.5])
    od = (_natural_design(dpf) @ (truth * 1e-6) * 5.0)[None, :, None]
    swapped, _ = reviewed.unmix_natural_od(od[:, ::-1, :], dpf, np.asarray([5.0]))
    assert not np.allclose(swapped[0, :, 0], truth)


def test_gap_is_not_restored_to_filter_or_event_mask() -> None:
    fs = 10.0
    time = np.arange(600) / fs
    values = np.sin(2 * np.pi * 0.05 * time)[:, None]
    mask = np.ones_like(values, dtype=bool)
    mask[250:258] = False
    filtered, usable, _ = reviewed.filter_masked(values, mask, fs, 0.01, 0.2)
    assert not usable[250:258].any()
    assert np.isnan(filtered[250:258]).all()
    # The excluded gap cannot become an event statistic even if it lies in the window.
    labels = [{"event_role": "action_block_start", "task_name": "摸头", "marker_index": 3, "timestamp_seconds": 25.2}]
    rows = reviewed.event_rows_for_signal(
        {"subject": "S", "session": "base_line"}, time, filtered, usable,
        np.asarray([True]), labels, "test", "signal",
    )
    assert rows[0]["status"] == "coverage_or_channel_gate_failed"
    assert np.isnan(rows[0]["delta"])


def test_eye_or_grip_labels_cannot_enter_completed_movement_branch() -> None:
    assert not reviewed.is_completed_movement_action({"event_role": "action_block_start", "task_name": "眼动"})
    assert not reviewed.is_completed_movement_action({"event_role": "action_block_start", "task_name": "握拳"})
    assert reviewed.is_completed_movement_action({"event_role": "action_block_start", "task_name": "摸腰"})
