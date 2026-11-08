from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import savemat

from emg_qc import (
    EMGSeries,
    EXPECTED_SENSOR_IDS,
    SENSOR_MUSCLES,
    SENSOR_TO_MUSCLE_INDEX,
    assess_emg_channel,
    read_emg_csv,
    resolve_emg_mapping,
)
from fnirs_qc import (
    FNIRSChannelSeries,
    analyze_fnirs_recording,
    assess_fnirs_channel,
    build_fnirs_sample_masks,
    is_infrared_eye_tracking_task,
    read_fnirs,
)
from preprocess import (
    _bandpass,
    _notch_harmonics,
    _resample,
    _robust_scalp_reference,
    audit_batch_provenance,
    detect_eeg_gel_bridges,
    eeg_bad_channels,
    preprocess_emg,
    preprocess_fnirs,
)
from main import (
    ACTIVE_EEG_CHANNELS,
    AUXILIARY_CHANNELS,
    FULL_EEG_CHANNELS,
    assert_writable_output_path,
    clamp_eeg_disconnect_epochs,
    classify_channels,
    discover_recordings,
    eeg_configuration_for_recording,
    output_path_for,
    recording_modality,
    session_for,
    task_label_for,
    write_batch_inventory,
)


class EMGQualityTests(unittest.TestCase):
    def test_physical_sensor_ids_map_to_stable_logical_muscles(self) -> None:
        self.assertEqual(EXPECTED_SENSOR_IDS, (2, 3, 4, 5, 9, 10, 11))
        self.assertEqual(SENSOR_TO_MUSCLE_INDEX[2], 1)
        self.assertEqual(SENSOR_TO_MUSCLE_INDEX[9], 5)
        self.assertEqual(SENSOR_TO_MUSCLE_INDEX[11], 7)
        self.assertEqual(set(SENSOR_MUSCLES), set(EXPECTED_SENSOR_IDS))
        self.assertEqual(SENSOR_MUSCLES[2], "斜方肌（上部）")
        self.assertEqual(SENSOR_MUSCLES[11], "尺侧腕屈肌")

    def test_clean_wideband_series_is_good(self) -> None:
        rng = np.random.default_rng(42)
        sample_rate = 1000.0
        time = np.arange(10_000) / sample_rate
        values = 2e-5 * np.sin(2 * np.pi * 83 * time) + rng.normal(scale=2e-6, size=time.size)
        metrics = assess_emg_channel(
            EMGSeries(2, "Avanti sensor 2: EMG 2 (IM)", "斜方肌（上部）", time, values, sample_rate)
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
        self.assertEqual(series[0].muscle_index, 1)
        np.testing.assert_allclose(series[0].values, [0.1, 0.2, 0.3])

    def test_confirmed_shifted_record_maps_to_logical_muscles(self) -> None:
        mapping = resolve_emg_mapping(
            [1, 2, 3, 4, 5, 9, 10], Path("data_repro_20260903/2026080901/base_line/EMG/data.csv")
        )
        self.assertEqual(mapping["mapping_status"], "record_override")
        self.assertTrue(mapping["mapping_valid_for_primary_features"])
        self.assertEqual(mapping["mapping_missing_sensor_ids"], [])
        self.assertEqual(mapping["mapping_unexpected_sensor_ids"], [])
        self.assertEqual(mapping["canonical_missing_sensor_ids"], [11])
        self.assertEqual(mapping["physical_to_logical"]["1"], 1)
        self.assertEqual(mapping["physical_to_canonical"]["1"], 2)
        self.assertEqual(mapping["logical_labels"]["1"], "斜方肌（上部）")

    def test_shifted_like_record_without_override_stays_unresolved(self) -> None:
        mapping = resolve_emg_mapping([1, 2, 3, 4, 5, 9, 10], Path("other/2026081401/base_line/EMG/data.csv"))
        self.assertEqual(mapping["mapping_status"], "unresolved")
        self.assertFalse(mapping["mapping_valid_for_primary_features"])
        self.assertEqual(mapping["mapping_unexpected_sensor_ids"], [1])

    def test_duplicate_sensor_ids_stay_unresolved(self) -> None:
        mapping = resolve_emg_mapping([2, 2, 3, 4, 5, 9, 10, 11], Path("2026081401/base_line/data.csv"))
        self.assertEqual(mapping["mapping_status"], "unresolved")
        self.assertFalse(mapping["mapping_valid_for_primary_features"])


class FNIRSQualityTests(unittest.TestCase):
    def test_eye_tracking_task_is_marked_as_infrared_interference(self) -> None:
        self.assertTrue(is_infrared_eye_tracking_task(Path("受试者/眼动任务/record.nirs")))
        self.assertFalse(is_infrared_eye_tracking_task(Path("受试者/运动任务/record.nirs")))
        self.assertFalse(is_infrared_eye_tracking_task(Path("受试者/运动任务/record_眼动.nirs")))

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

    def test_complete_64_cap_uses_all_scalp_channels(self) -> None:
        labels = list(FULL_EEG_CHANNELS) + sorted(AUXILIARY_CHANNELS)
        active, auxiliary, disabled = classify_channels(labels, FULL_EEG_CHANNELS)
        self.assertEqual([labels[index] for index in active], list(FULL_EEG_CHANNELS))
        self.assertEqual(len(active), 59)
        self.assertEqual(len(disabled), 0)
        self.assertEqual(len(auxiliary), 5)

    def test_only_two_baseline_records_use_reduced_montage(self) -> None:
        special = Path(r"J:\2026二康\数据\2026081701\base_line\EEG\静息态\data.bdf")
        ordinary = Path(r"J:\2026二康\数据\2026081701\sec_test\EEG\静息态\data.bdf")
        channels, _ = eeg_configuration_for_recording(special)
        self.assertEqual(channels, ACTIVE_EEG_CHANNELS)
        channels, _ = eeg_configuration_for_recording(ordinary)
        self.assertEqual(channels, FULL_EEG_CHANNELS)

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
            self.assertEqual(task_label_for(emg), "运动任务")
            self.assertEqual(
                output_path_for(fnirs, root, Path("results")),
                Path("results/P/FNIRS/S/未识别任务/record_静息态"),
            )

    def test_task_label_uses_parent_directory_not_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            movement = root / "P" / "FNIRS" / "运动" / "record_静息态.nirs"
            movement.parent.mkdir(parents=True, exist_ok=True)
            movement.touch()
            self.assertEqual(task_label_for(movement), "运动任务")
            self.assertEqual(
                output_path_for(movement, root, Path("results")),
                Path("results/P/FNIRS/运动/运动任务/record_静息态"),
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
    def _write_eye_fnirs(self, path: Path, channels: int = 20) -> None:
        sample_rate = 11.0
        time = np.arange(660) / sample_rate
        data_columns = []
        measurements = []
        source_positions = []
        detector_positions = []
        cardiac = np.sin(2 * np.pi * 1.2 * time)
        for index in range(channels):
            source = index + 1
            detector = index + 1
            source_positions.append([float(index), 0.0, 0.0])
            detector_positions.append([float(index), 1.0, 0.0])
            a = 200.0 * (1.0 + 0.01 * cardiac)
            b = 180.0 * (1.0 + 0.012 * cardiac)
            if index == 0:
                a = a.copy()
                b = b.copy()
                a[30] = 0.0
                b[30] = 0.0
            data_columns.extend((a, b))
            measurements.extend(([source, detector, 1, 1], [source, detector, 1, 2]))
        sd = {
            "Lambda": np.array([730.0, 850.0]),
            "SrcPos": np.asarray(source_positions),
            "DetPos": np.asarray(detector_positions),
            "MeasList": np.asarray(measurements),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        savemat(path, {"d": np.column_stack(data_columns), "t": time, "SD": sd, "ml": sd["MeasList"]})

    def test_emg_preprocess_uses_common_elapsed_duration_and_exports_quality_masks(self) -> None:
        # Two sensors with slightly different sample rates must be cropped by
        # elapsed time, not by the shortest row count.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.csv"
            rows = ['T[s],"Avanti sensor 2: EMG 2 (IM)",T[s],"Avanti sensor 3: EMG 3 (IM)"']
            for index in range(3_001):
                t_a = index / 1000.0
                t_b = index / 1001.0
                value_a = 1e-3 * np.sin(2 * np.pi * 80 * t_a)
                value_b = 1e-3 * np.sin(2 * np.pi * 80 * t_b)
                rows.append(f"{t_a:.9f},{value_a:.9g},{t_b:.9f},{value_b:.9g}")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            destination = root / "out"
            summary = preprocess_emg(path, destination)
            with np.load(destination / "emg_cleaned.npz") as payload:
                data_shape = payload["data_source_unit"].shape
                raw_shape = payload["data_raw_source_unit"].shape
                score_shape = payload["channel_quality_score"].shape
                gate_shape = payload["channel_signal_gate"].shape
                window_shape = payload["window_mask"].shape
                output_duration = float(payload["time_seconds"][-1])
                sensor_ids = payload["sensor_ids"].tolist()
                muscle_indices = payload["muscle_indices"].tolist()
                missing_sensor_ids = payload["missing_sensor_ids"].tolist()
                unexpected_sensor_ids = payload["unexpected_sensor_ids"].tolist()
        self.assertEqual(data_shape[0], 2)
        self.assertEqual(score_shape, (2,))
        self.assertEqual(raw_shape, data_shape)
        self.assertEqual(gate_shape, (2,))
        self.assertEqual(window_shape[0], 2)
        self.assertAlmostEqual(float(summary["common_duration_seconds"]), 3.0, places=2)
        self.assertAlmostEqual(output_duration, float(summary["common_duration_seconds"]), places=6)
        self.assertEqual(sensor_ids, [2, 3])
        self.assertEqual(muscle_indices, [1, 2])
        self.assertEqual(missing_sensor_ids, [4, 5, 9, 10, 11])
        self.assertEqual(unexpected_sensor_ids, [])
        self.assertEqual(summary["missing_sensor_ids"], [4, 5, 9, 10, 11])

    def test_emg_unexpected_sensor_is_retained_but_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.csv"
            path.write_text(
                'T[s],"Avanti sensor 1: EMG 1 (IM)",T[s],"Avanti sensor 2: EMG 2 (IM)"\n'
                + "\n".join(
                    f"{i / 1000:.3f},{0.001 * np.sin(2 * np.pi * 83 * i / 1000):.9g},"
                    f"{i / 1000:.3f},{0.002 * np.sin(2 * np.pi * 83 * i / 1000):.9g}"
                    for i in range(3001)
                )
                + "\n",
                encoding="utf-8",
            )
            destination = root / "out"
            summary = preprocess_emg(path, destination)
            with np.load(destination / "emg_cleaned.npz") as payload:
                sensor_ids = payload["sensor_ids"].tolist()
                gates = payload["channel_signal_gate"].tolist()
        self.assertEqual(sensor_ids, [1, 2])
        self.assertEqual(summary["unexpected_sensor_ids"], [1])
        # A mixed/unknown record is unresolved as a whole; retain both raw
        # channels for audit, but do not admit either to primary features.
        self.assertEqual(gates, [False, False])

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

    def test_eye_tracking_fnirs_is_retained_with_interference_and_sample_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "subject" / "眼动任务" / "record.nirs"
            self._write_eye_fnirs(path)
            destination = root / "out"
            summary = preprocess_fnirs(path, destination)
            with np.load(destination / "fnirs_cleaned_od.npz") as payload:
                self.assertEqual(payload["optical_density"].shape, (20, 2, 660))
                self.assertEqual(payload["repair_mask"].shape, (20, 660))
                self.assertTrue(bool(payload["repair_mask"][0, 30]))
                self.assertFalse(bool(payload["analysis_sample_mask"][0, 30]))
        self.assertEqual(summary["processing_status"], "READY_WITH_MASKS_KNOWN_IR_INTERFERENCE")
        self.assertTrue(summary["known_ir_interference"])
        self.assertEqual(summary["eye_tracker_occlusion"]["status"], "PARTIAL_REPORTED")
        self.assertEqual(summary["analysis_branch"], "EYE_IR_DEGRADED_MASKED")

    def test_eye_tracking_fnirs_qc_writes_degraded_report_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "subject" / "眼动任务" / "record.nirs"
            self._write_eye_fnirs(path)
            destination = root / "qc"
            summary = analyze_fnirs_recording(path, destination, None)
            with np.load(destination / "fnirs_qc_masks.npz") as payload:
                self.assertEqual(payload["analysis_sample_mask"].shape, (20, 660))
            report = (destination / "quality_report.md").read_text(encoding="utf-8")
        self.assertTrue(summary["known_ir_interference"])
        self.assertEqual(summary["session_status"], "CAUTION")
        self.assertIn("fnirs_qc_masks.npz", report)
        self.assertIn("`PARTIAL_REPORTED`", report)

    def test_resample_supports_avanti_fractional_rate(self) -> None:
        values = np.sin(2 * np.pi * 20 * np.arange(2_223) / 1111.111111)[None, :]
        resampled, rate = _resample(values, 1111.111111, 1000.0)
        self.assertEqual(rate, 1000.0)
        self.assertGreater(resampled.shape[1], 1_990)
        self.assertLess(resampled.shape[1], 2_010)

    def test_output_path_guard_blocks_readonly_drive_and_data_root(self) -> None:
        with self.assertRaises(PermissionError):
            assert_writable_output_path(Path(r"J:\2026二康\forbidden_out"))
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory) / "raw_data"
            raw_root.mkdir()
            with self.assertRaises(PermissionError):
                assert_writable_output_path(raw_root / "nested", raw_root)

    def test_clamp_eeg_disconnect_and_gel_bridge_trimmed_reference(self) -> None:
        rng = np.random.default_rng(11)
        fs = 250.0
        time = np.arange(int(30 * fs)) / fs
        labels = ["Fp1", "Fz", "Cz", "Pz", "O1", "O2", "C3", "C4"]
        clean = np.stack([12.0 * np.sin(2 * np.pi * (8.0 + 0.4 * idx) * time) + rng.normal(0, 1.0, time.size) for idx in range(len(labels))]).astype(np.float32)
        clean[1] = clean[0] + rng.normal(0, 0.02, time.size).astype(np.float32)
        unplugged = clean.copy()
        unplugged[:, -400:] += 380_000.0
        clamped, disc_mask = clamp_eeg_disconnect_epochs(unplugged, None, None, None, fs)
        self.assertTrue(bool(disc_mask[-1]))
        self.assertFalse(bool(disc_mask[2]))
        filtered = _bandpass(_notch_harmonics(clamped, fs, 45.0), fs, (0.5, 45.0))
        self.assertLess(float(np.std(filtered[2, 500:4000])), 25.0)
        cids, clusters = detect_eeg_gel_bridges(filtered, labels, np.zeros(len(labels), dtype=bool), fs, valid_epoch_mask=~disc_mask)
        self.assertEqual(clusters, [["Fp1", "Fz"]])
        popped = clean.copy()
        popped[7, 1000:1200] += 900.0
        ref = _robust_scalp_reference(popped, np.zeros(len(labels), dtype=bool), cids)
        self.assertLess(float(np.max(np.abs(ref[:, 1000:1200]))), 25.0)

    def test_emg_active_span_preserves_duration_despite_early_dead_sensor_and_masks_wireless_gap(self) -> None:
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "2026080901" / "base_line" / "EMG" / "sample.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = ['T[s],"Avanti sensor 1: EMG 1 (IM)",T[s],"Avanti sensor 2: EMG 2 (IM)",T[s],"Avanti sensor 3: EMG 3 (IM)"']
            t2_list = [i / 1000.0 for i in range(6001) if not (2000 < i < 3000)]
            for idx in range(6001):
                t1 = f"{idx / 1000.0:.6f},0.0" if idx <= 1000 else ","
                if idx < len(t2_list):
                    t2_val = t2_list[idx]
                    burst = 12.0 if 4.0 <= t2_val <= 5.0 else 1.0
                    v2 = burst * (1e-4 * np.sin(2 * np.pi * 85.0 * t2_val) + rng.normal(0, 2e-5))
                    t2 = f"{t2_val:.6f},{v2:.8g}"
                else:
                    t2 = ","
                t3_val = idx / 1000.0
                burst3 = 10.0 if 4.0 <= t3_val <= 5.0 else 1.0
                v3 = burst3 * (1e-4 * np.sin(2 * np.pi * 92.0 * t3_val) + rng.normal(0, 2e-5))
                rows.append(f"{t1},{t2},{t3_val:.6f},{v3:.8g}")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            out = root / "out"
            summary = preprocess_emg(path, out)
            with np.load(out / "emg_cleaned.npz") as payload:
                dur = float(payload["time_seconds"][-1])
                s_valid = payload["sample_valid_mask"]
                w_mask = payload["window_mask"]
        self.assertGreater(dur, 5.9)
        self.assertEqual(summary["early_terminated_sensor_ids"], [1])
        self.assertEqual(summary["gap_repaired_sensor_ids"], [2])
        self.assertFalse(bool(s_valid[0, 3000]))
        self.assertFalse(bool(s_valid[1, 2500]))
        self.assertFalse(bool(w_mask[1, 2]))
        self.assertTrue(bool(w_mask[2, 4]))

    def test_fnirs_long_dropout_preserves_motion_detection_and_dilates_ringing_edges(self) -> None:
        fs = 10.0
        time = np.arange(600) / fs
        pulse = np.sin(2 * np.pi * 1.2 * time)
        a = 200.0 * (1.0 + 0.01 * pulse)
        b = 180.0 * (1.0 + 0.012 * pulse)
        a[100:120] = 0.0
        b[100:120] = 0.0
        a[420:] *= 0.65
        b[420:] *= 0.65
        ch = FNIRSChannelSeries(1, 1, 1, "CH01 (S1-D1)", 730.0, 850.0, a, b, (0.0, 0.0))
        _, motion, _, analysis = build_fnirs_sample_masks(ch, fs, time=time)
        self.assertTrue(bool(motion[90]))
        self.assertTrue(bool(motion[125]))
        self.assertTrue(bool(motion[420]))
        self.assertFalse(bool(analysis[420]))

    def test_batch_provenance_quarantines_duplicates_and_thi_test_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            p_orig = root / "2026090201" / "base_line" / "EMG" / "2026090201_test.csv"
            p_dup = root / "2026091901" / "base_line" / "EMG" / "2026090201_test.csv"
            p_rest = root / "2026081801" / "thi_test" / "EEG" / "静息态" / "data.bdf"
            p_eye_dup = root / "2026081801" / "thi_test" / "EEG" / "眼动" / "data.bdf"
            csv_body = 'X[s],"Avanti sensor 2: EMG 2 (IM)"\n0.0,0.1\n0.001,0.2\n'
            for item, content in ((p_orig, csv_body), (p_dup, csv_body), (p_rest, "BDF_BYTES"), (p_eye_dup, "BDF_BYTES")):
                item.parent.mkdir(parents=True, exist_ok=True)
                item.write_text(content, encoding="utf-8")
            self.assertEqual(session_for(p_rest, root), "thi_test")
            _, exclusions, _ = audit_batch_provenance([p_orig, p_dup, p_rest, p_eye_dup], root)
            self.assertEqual(exclusions[p_dup][0], "EXCLUDED_CROSS_SUBJECT_DUPLICATE")
            self.assertNotIn(p_orig, exclusions)
            self.assertEqual(exclusions[p_eye_dup][0], "EXCLUDED_CROSS_TASK_DUPLICATE")
            self.assertNotIn(p_rest, exclusions)


if __name__ == "__main__":
    unittest.main()
