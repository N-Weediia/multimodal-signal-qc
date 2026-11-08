"""Dependency-free equivalent smoke checks for the reviewed MBLL branch.

Use this runner when the analysis interpreter intentionally has no pytest.
It covers the same four invariants as the pytest suite without installing
anything: natural-log MBLL recovery, explicit wavelength-order rejection,
gap-mask restoration, and eye/grip isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mbll_relative_sensitivity_reviewed_20260908 as reviewed


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    dpf = np.asarray([6.0, 6.0])
    truth = np.asarray([2.25, -0.75])
    design = np.asarray([
        [reviewed.EXTINCTION_E_NATURAL[730.0]["hbo"], reviewed.EXTINCTION_E_NATURAL[730.0]["hbr"]],
        [reviewed.EXTINCTION_E_NATURAL[850.0]["hbo"], reviewed.EXTINCTION_E_NATURAL[850.0]["hbr"]],
    ]) * dpf[:, None]
    od = (design @ (truth * 1e-6) * 5.0)[None, :, None]
    recovered, _ = reviewed.unmix_natural_od(od, dpf, np.asarray([5.0]))
    expect(np.allclose(recovered[0, :, 0], truth, rtol=1e-10, atol=1e-10), "known HbO/HbR recovery failed")
    try:
        reviewed.validate_wavelength_order([(850.0, 730.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("reversed wavelengths were accepted")
    fs, samples = 10.0, 600
    time = np.arange(samples) / fs
    values = np.sin(2 * np.pi * 0.05 * time)[:, None]
    mask = np.ones_like(values, dtype=bool); mask[250:258] = False
    filtered, usable, _ = reviewed.filter_masked(values, mask, fs, 0.01, 0.2)
    expect(not usable[250:258].any() and np.isnan(filtered[250:258]).all(), "gap was restored after filtering")
    labels = [{"event_role": "action_block_start", "task_name": "摸头", "marker_index": 3, "timestamp_seconds": 25.2}]
    rows = reviewed.event_rows_for_signal({"subject": "S", "session": "base_line"}, time, filtered, usable, np.asarray([True]), labels, "test", "signal")
    expect(rows[0]["status"] == "coverage_or_channel_gate_failed", "masked gap entered event estimate")
    expect(not reviewed.is_completed_movement_action({"event_role": "action_block_start", "task_name": "眼动"}), "eye label entered movement branch")
    expect(not reviewed.is_completed_movement_action({"event_role": "action_block_start", "task_name": "握拳"}), "grip extension entered movement branch")
    print("smoke checks passed: known recovery; wavelength rejection; gap mask; eye/grip isolation")


if __name__ == "__main__":
    main()
