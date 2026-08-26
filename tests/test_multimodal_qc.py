from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import savemat

from emg_qc import EMGSeries, SENSOR_MUSCLES, assess_emg_channel, read_emg_csv
from fnirs_qc import FNIRSChannelSeries, assess_fnirs_channel, is_infrared_eye_tracking_task, read_fnirs
from preprocess import _notch_harmonics, _resample, eeg_bad_channels, preprocess_fnirs
from main import (
    ACTIVE_EEG_CHANNELS,
    AUXILIARY_CHANNELS,
    classify_channels,
    discover_recordings,
    output_path_for,
    recording_modality,
    task_label_for,
    write_batch_inventory,
)


class EMGQualityTests(unittest.TestCase):
    def test_corrected_sensor_sequence_maps_to_seven_muscles(self) -> None:
        self.assertEqual(
            SENSOR_MUSCLES,
            {
                2: "斜方肌（上部）",
                3: "三角肌前束",
                4: "三角肌中束",
                5: "肱二头肌",
                9: "肱三头肌",
                10: "肱桡肌",
                11: "尺侧腕屈肌",
            },
        )

    def test_clean_wideband_series_is_good(self) -> None:
        rng = np.random.default_rng(42)
        sample_rate = 1000.0
        time = np.arange(10_000) / sample_rate
        values = 2e-5 * np.sin(2 * np.pi * 83 * time) + rng.normal(scale=2e-6, size=time.size)
        metrics = assess_emg_channel(
            EMGSeries(2, "Avanti sensor 2: EMG 2 (IM)", "三角肌前束", time, values, sample_rate)
        )
        self.assertEqual(metrics.status, "GOOD")
        self.assertEqual(metrics.missing_sample_percent, 0.0)
        self.assertEqual(metrics.flat_epoch_percent, 0.0)

    def test_csv_reader_keeps_emg_and_ignores_imu_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text(
                'X[s],"Avanti sensor 2: EMG 2 (IM)",X[s],"Avanti sensor 2: Acc 2.X (IM)"\n'
                "0.000,0.1,0.000,1\n0.001,0.2,0.010,2\n0.002,0.3,0.020,3\n",
                encoding="utf-8",
            )
            series = read_emg_csv(path, None)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].sensor_id, 2)
        self.assertEqual(series[0].muscle, "斜方肌（上部）")
        np.testing.assert_allclose(series[0].values, [0.1, 0.2, 0.3])


class FNIRSQualityTests(unittest.TestCase):
    def test_eye_tracking_task_is_marked_as_infrared_interference(self) -> None:
        self.assertTrue(is_infrared_eye_tracking_task(Path("受试者/眼动任务/record.nirs")))
        self.assertFalse(is_infrared_eye_tracking_task(Path("受试者/运动任务/record.nirs")))

    def _channel(self, coupled: bool) -> tuple[FNIRSChannelSeries, float]:
        rng = np.random.default_rng(7)
        sample_rate = 11.0
        time = np.arange(660) / sample_rate
        cardiac = np.sin(2 * np.pi * 1.2 * time)
        a = 200 * (1 + 0.01 * cardiac) + rng.normal(scale=0.1, size=time.size)
        if coupled:
            b = 180 * (1 + 0.012 * cardiac) + rng.normal(scale=0.1, size=time.size)
        else:
            b = 180 * (1 + 0.012 * np.sin(2 * np.pi * 1.7 * time)) + rng.normal(scale=0.1, size=time.size)
        return FNIRSChannelSeries(1, 1, 1, "CH01 (S1-D1)", 730, 850, a, b, (0.0, 0.0)), sample_rate

    def test_sci_distinguishes_coupled_wavelengths(self) -> None:
        good_channel, sample_rate = self._channel(coupled=True)
        bad_channel, _ = self._channel(coupled=False)
        good = assess_fnirs_channel(good_channel, sample_rate)
        bad = assess_fnirs_channel(bad_channel, sample_rate)
        self.assertGreater(good.scalp_coupling_index, 0.95)
        self.assertEqual(good.status, "GOOD")
        self.assertLess(bad.scalp_coupling_index, 0.5)
        self.assertLess(bad.quality_score, good.quality_score)

    def test_nirs_reader_pairs_wavelengths_by_source_detector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.nirs"
            time = np.arange(110) / 11.0
            data = np.column_stack([
                100 + np.sin(time), 200 + np.sin(time),
                110 + np.sin(time), 210 + np.sin(time),
            ])
            sd = {
                "Lambda": np.array([730, 850]),
                "SrcPos": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                "DetPos": np.array([[1.0, 0.0, 0.0]]),
                "MeasList": np.array([[1, 1, 1, 1], [2, 1, 1, 1], [1, 1, 1, 2], [2, 1, 1, 2]]),
            }
            savemat(path, {"d": data, "t": time, "SD": sd, "ml": sd["MeasList"]})
            recording = read_fnirs(path, None)
        self.assertEqual(len(recording.channels), 2)
        self.assertEqual(recording.channels[0].label, "CH01 (S1-D1)")
        np.testing.assert_allclose(recording.channels[0].intensity_b, data[:, 2])


class DiscoveryTests(unittest.TestCase):
    def test_32_channel_montage_excludes_disabled_scalp_electrodes(self) -> None:
        all_scalp = [
            "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
            "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4", "C5", "C6",
            "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", "Pz", "P3", "P4", "P5", "P6", "P7",
            "P8", "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2",
        ]
        labels = all_scalp + sorted(AUXILIARY_CHANNELS)
        active, auxiliary, disabled = classify_channels(labels)
        self.assertEqual([labels[index] for index in active], list(ACTIVE_EEG_CHANNELS))
        self.assertEqual(len(active), 31)
        self.assertEqual(len(disabled), 28)
        self.assertEqual(len(auxiliary), 5)

    def test_single_avanti_csv_is_detected_outside_emg_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.csv"
            path.write_text('X[s],"Avanti sensor 2: EMG 2 (IM)"\n', encoding="utf-8")
            self.assertEqual(recording_modality(path), "EMG")

    def test_discovery_and_output_paths_are_modality_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            eeg = root / "P" / "EEG" / "S" / "静息态" / "data.bdf"
            emg = root / "P" / "EMG" / "session.csv"
            fnirs = root / "P" / "FNIRS" / "S" / "record_静息态.nirs"
            for path in (eeg, emg, fnirs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            recordings = discover_recordings(root, root / "P")
            self.assertEqual({recording_modality(path) for path in recordings}, {"EEG", "EMG", "FNIRS"})
            self.assertEqual(output_path_for(eeg, root, Path("results")), Path("results/P/EEG/S/静息态"))
            self.assertEqual(output_path_for(emg, root, Path("results")), Path("results/P/EMG/session"))
            self.assertEqual(
                output_path_for(fnirs, root, Path("results")),
                Path("results/P/FNIRS/S/静息态/record_静息态"),
            )

    def test_repeated_exports_have_distinct_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            eeg = root / "P" / "EEG" / "运动" / "data.bdf"
            eeg_continuation = root / "P" / "EEG" / "运动" / "data.1.bdf"
            fnirs = root / "P" / "FNIRS" / "眼动" / "first_眼动.nirs"
            fnirs_repeat = root / "P" / "FNIRS" / "眼动" / "second_眼动.nirs"
            for path in (eeg, eeg_continuation, fnirs, fnirs_repeat):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            destinations = {
                output_path_for(path, root, Path("results"))
                for path in (eeg, eeg_continuation, fnirs, fnirs_repeat)
            }
            self.assertEqual(len(destinations), 4)
            self.assertEqual(task_label_for(eeg), "运动任务")
            self.assertEqual(task_label_for(fnirs), "眼动任务")

    def test_batch_inventory_keeps_unassessable_recordings_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            results = Path(directory) / "results"
            (root / "P").mkdir(parents=True)
            (root / "Q").mkdir()
            results.mkdir()
            outcomes = [
                {
                    "subject": "P", "task": "运动任务", "modality": "EEG", "status": "FAILED",
                    "session_status": "UNASSESSABLE", "mean_quality_score": None,
                    "source_relative_path": "P/EEG/运动/data.1.bdf", "source_filename": "data.1.bdf",
                    "source_size_bytes": 12, "source_modified_at": "2026-08-22T00:00:00+08:00",
                    "result_directory": "results/P/EEG/运动/data.1", "message": "记录不足 2 s",
                }
            ]
            write_batch_inventory(root, outcomes, results)
            payload = json.loads((results / "data_inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["failed_count"], 1)
            self.assertEqual(payload["subjects"]["P"]["failed_count"], 1)
            self.assertEqual(payload["subjects"]["Q"]["recording_count"], 0)
            self.assertIn("记录不足 2 s", (results / "batch_report.md").read_text(encoding="utf-8"))


class PreprocessingTests(unittest.TestCase):
    def test_harmonic_notch_reduces_50_hz_without_changing_sample_count(self) -> None:
        sample_rate = 1000.0
        time = np.arange(20_000) / sample_rate
        noisy = np.sin(2 * np.pi * 10 * time) + 3 * np.sin(2 * np.pi * 50 * time)
        cleaned = _notch_harmonics(noisy[None, :], sample_rate, 45.0)[0]
        frequencies, before = signal.welch(noisy, fs=sample_rate, nperseg=4000)
        _, after = signal.welch(cleaned, fs=sample_rate, nperseg=4000)
        line_bin = np.argmin(np.abs(frequencies - 50))
        self.assertEqual(cleaned.size, noisy.size)
        self.assertLess(after[line_bin], before[line_bin] * 0.02)

    def test_direct_signal_failures_mark_eeg_channel_bad(self) -> None:
        sample_rate = 1000.0
        time = np.arange(10_000) / sample_rate
        usable = 20 * np.sin(2 * np.pi * 10 * time)
        flat = np.zeros_like(time)
        bad, reasons = eeg_bad_channels(np.stack([usable, flat]), ["Fz", "Cz"], sample_rate)
        self.assertFalse(bad[0])
        self.assertTrue(bad[1])
        self.assertIn("Cz", reasons)

    def test_eye_tracking_fnirs_is_excluded_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "out"
            summary = preprocess_fnirs(Path("subject/眼动任务/missing.nirs"), destination)
        self.assertEqual(summary["processing_status"], "EXCLUDED_KNOWN_IR_INTERFERENCE")

    def test_resample_supports_avanti_fractional_rate(self) -> None:
        values = np.sin(2 * np.pi * 20 * np.arange(2_223) / 1111.111111)[None, :]
        resampled, rate = _resample(values, 1111.111111, 1000.0)
        self.assertEqual(rate, 1000.0)
        self.assertGreater(resampled.shape[1], 1_990)
        self.assertLess(resampled.shape[1], 2_010)


if __name__ == "__main__":
    unittest.main()
