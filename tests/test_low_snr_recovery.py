import json

import numpy as np

import low_snr_recovery as recovery

from emg_qc import EMGSeries
from low_snr_recovery import asr_like, derivative_repair, emg_summary, hampel_filter, harmonic_cancel, kalman_smooth


def test_hampel_replaces_isolated_spike_only():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    x[100] += 20
    cleaned, mask = hampel_filter(x, window=9, n_sigma=5)
    assert mask[100]
    assert abs(cleaned[100] - np.sin(np.linspace(0, 4 * np.pi, 200))[100]) < 0.2
    assert mask.sum() <= 3


def test_derivative_repair_removes_step_without_flattening_trend():
    t = np.linspace(0, 10, 500)
    x = 0.1 * t
    x[250:] += 4
    repaired, mask = derivative_repair(x, n_sigma=6)
    assert mask[250]
    assert np.std(np.diff(repaired)) < 0.03
    assert abs(repaired[-1] - x[-1] + 4) < 0.2


def test_harmonic_cancel_preserves_nonharmonic_component():
    fs = 1000.0
    t = np.arange(20_000) / fs
    clean = np.sin(2 * np.pi * 23 * t)
    noisy = clean + 2.5 * np.sin(2 * np.pi * 50 * t + 0.4)
    output = harmonic_cancel(noisy, fs, upper=450, block_seconds=10)
    assert np.corrcoef(clean, output)[0, 1] > 0.98
    assert np.std(output - clean) < 0.05


def test_kalman_smooth_reduces_observation_noise():
    rng = np.random.default_rng(2)
    clean = np.sin(np.linspace(0, 3 * np.pi, 300))
    noisy = clean + 0.25 * rng.normal(size=clean.size)
    smoothed = kalman_smooth(noisy)
    assert np.std(smoothed - clean) < np.std(noisy - clean)


def test_asr_like_shrinks_transient_subspace():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(8, 2000))
    data[0, 900:1100] += 30
    cleaned, affected = asr_like(data, sample_rate=100.0, cutoff=3)
    assert affected > 0
    assert np.ptp(cleaned[0, 900:1100]) < np.ptp(data[0, 900:1100])


def test_eye_task_guard_is_scoped_to_movement_inventory_metadata(tmp_path, monkeypatch):
    eye_path = tmp_path / "P" / "FNIRS" / "眼动" / "record.nirs"
    monkeypatch.setattr(recovery.expanded, "load_identity", lambda _: {"P": "Name"})
    monkeypatch.setattr(recovery.expanded, "discover_sessions", lambda _: {"P": ["base_line"]})
    monkeypatch.setattr(recovery.expanded, "canonicalize_sessions", lambda sessions: (sessions, []))
    monkeypatch.setattr(
        recovery.expanded,
        "robust_find_recording",
        lambda subject, session, modality, task: eye_path if modality == "FNIRS" else None,
    )

    output = tmp_path / "out"
    recovery.run(output)

    metadata = json.loads((output / "low_snr_recovery_metadata.json").read_text(encoding="utf-8"))
    eye_limitations = [item for item in metadata["limitations"] if "eye-tracker" in item.lower()]
    assert eye_limitations
    assert "movement-task fNIRS only" in eye_limitations[0]
    assert "main pipeline" in eye_limitations[0]
    assert "excluded because" not in eye_limitations[0].lower()


def test_emg_sensitivity_summary_excludes_poor_channels(monkeypatch, tmp_path):
    time = np.arange(4000, dtype=float) / 1000.0
    series = [
        EMGSeries(2, "sensor 2", "", time, np.sin(2 * np.pi * 80 * time), 1000.0, muscle_index=1),
        EMGSeries(3, "sensor 3", "", time, np.sin(2 * np.pi * 90 * time), 1000.0, muscle_index=2),
    ]
    monkeypatch.setattr("low_snr_recovery.read_emg_csv", lambda path, max_seconds: series)
    monkeypatch.setattr(
        "low_snr_recovery.assess_emg_channel",
        lambda item: type("Metric", (), {"status": "POOR" if item.sensor_id == 2 else "GOOD"})(),
    )
    summary = emg_summary(tmp_path / "2026000001" / "base_line" / "EMG.csv")
    assert summary["emg_channels"] == 1
    assert summary["emg_channels_observed"] == 2
    assert summary["emg_quality_eligible_channels"] == 1
    assert summary["emg_quality_poor_channels"] == 1
