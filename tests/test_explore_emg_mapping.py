from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from emg_qc import EMGSeries
from explore_202608_cases import emg_features


def test_exploratory_emg_features_reject_unresolved_mapping() -> None:
    time = np.arange(4000, dtype=float) / 1000.0
    series = [
        EMGSeries(
            sensor_id=sensor_id,
            label=f"sensor {sensor_id}",
            muscle="",
            time=time,
            values=np.sin(2 * np.pi * (20 + sensor_id) * time) + 0.01 * sensor_id,
            sample_rate_hz=1000.0,
        )
        for sensor_id in (1, 2, 3, 4, 5, 9, 10, 11)
    ]
    fake_metrics = SimpleNamespace(line_noise_power_percent=1.0)
    with patch("explore_202608_cases.read_emg_csv", return_value=series), patch(
        "explore_202608_cases.assess_emg_channel", return_value=fake_metrics
    ), pytest.raises(ValueError, match="映射未确认"):
        emg_features("unused.csv")


def test_exploratory_emg_features_do_not_re_admit_unknown_quality_on_short_fallback() -> None:
    time = np.arange(5000, dtype=float) / 1000.0
    series = [
        EMGSeries(2, "sensor 2", "", time, np.sin(2 * np.pi * 80 * time), 1000.0, muscle_index=1),
        EMGSeries(3, "sensor 3", "", time, np.sin(2 * np.pi * 90 * time), 1000.0, muscle_index=2),
    ]
    metrics = [
        SimpleNamespace(line_noise_power_percent=40.0, status="GOOD", quality_score=90.0),
        SimpleNamespace(line_noise_power_percent=1.0, status="UNKNOWN", quality_score=float("nan")),
    ]
    with patch("explore_202608_cases.read_emg_csv", return_value=series), patch(
        "explore_202608_cases.assess_emg_channel", side_effect=metrics
    ):
        features, extras = emg_features("unused.csv")
    assert extras["sensor_ids"].tolist() == [2]
    assert extras["excluded_physical_sensor_ids"].tolist() == [3]
    assert "emg_muscle_01_rms" in features
    assert "emg_muscle_02_rms" not in features
