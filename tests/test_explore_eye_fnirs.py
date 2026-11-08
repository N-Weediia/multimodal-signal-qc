from pathlib import Path

import numpy as np
from scipy.io import savemat

from explore_202608_cases import EventSeries, eye_fnirs_features


def _write_eye_fnirs(path: Path, channels: int = 20, poor_first: bool = False) -> None:
    sample_rate = 11.0
    time = np.arange(660) / sample_rate
    columns = []
    measurements = []
    src_pos = []
    det_pos = []
    for index in range(channels):
        src = det = index + 1
        src_pos.append([float(index), 0.0, 0.0])
        det_pos.append([float(index), 1.0, 0.0])
        pulse = np.sin(2 * np.pi * 1.2 * time)
        a = 200.0 * (1 + 0.01 * pulse)
        b = 180.0 * (1 + 0.012 * pulse)
        if poor_first and index == 0:
            a = np.zeros_like(a)
            b = np.zeros_like(b)
        columns.extend((a, b))
        measurements.extend(([src, det, 1, 1], [src, det, 1, 2]))
    sd = {
        "Lambda": np.array([730.0, 850.0]),
        "SrcPos": np.asarray(src_pos),
        "DetPos": np.asarray(det_pos),
        "MeasList": np.asarray(measurements),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"d": np.column_stack(columns), "t": time, "SD": sd, "ml": sd["MeasList"]})


def test_eye_fnirs_features_are_quality_gated_and_nonhaemodynamic(tmp_path: Path) -> None:
    path = tmp_path / "P" / "眼动任务" / "record.nirs"
    _write_eye_fnirs(path)
    features, extras = eye_fnirs_features(path, EventSeries(np.array([2.0]), "unknown"))

    assert features["eye_fnirs_known_ir_interference"] == 1
    assert features["eye_fnirs_occlusion_status"] == "PARTIAL_REPORTED"
    assert features["eye_fnirs_event_count_observed"] == 1
    assert features["eye_fnirs_event_semantics"].startswith("unknown")
    assert 0.0 <= features["eye_fnirs_analysis_sample_coverage"] <= 1.0
    assert extras["analysis_sample_mask"].shape == (20, 660)
    assert extras["wavelength_valid_sample_mask"].shape == (20, 2, 660)
    assert extras["motion_artifact_mask"].shape == (20, 660)
    assert extras["repair_mask"].shape == (20, 660)
    np.testing.assert_array_equal(
        extras["analysis_sample_mask"],
        extras["wavelength_valid_sample_mask"].all(axis=1)
        & ~extras["motion_artifact_mask"]
        & ~extras["repair_mask"],
    )
    assert not np.any(np.isfinite(extras["proxy"]) & ~extras["analysis_sample_mask"].T)
    assert not any("hbo" in key.lower() or "hbr" in key.lower() for key in features)


def test_eye_fnirs_poor_channels_stay_excluded(tmp_path: Path) -> None:
    path = tmp_path / "P" / "眼动任务" / "record.nirs"
    _write_eye_fnirs(path, poor_first=True)
    features, extras = eye_fnirs_features(path)

    assert features["eye_fnirs_quality_poor_channels"] >= 1
    assert features["eye_fnirs_eligible_channel_count"] == 19
    assert not np.any(extras["analysis_sample_mask"][0])
    assert np.all(~np.isfinite(extras["proxy"][:, 0]))
    assert not bool(extras["proxy_channel_gate"][0])
