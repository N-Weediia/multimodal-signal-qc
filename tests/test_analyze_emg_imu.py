import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_emg_imu import (
    _imu_proxy,
    _stream_summary,
    analyze_file,
    parse_header,
)


class EmgImuProxyTests(unittest.TestCase):
    def _write_csv(self, path: Path, zero_imu: bool = False) -> None:
        header = [
            "X[s]",
            '"Avanti sensor 2: EMG 2 (IM)"',
            "X[s]",
            '"Avanti sensor 2: Acc 2.X (IM)"',
            "X[s]",
            '"Avanti sensor 2: Acc 2.Y (IM)"',
            "X[s]",
            '"Avanti sensor 2: Acc 2.Z (IM)"',
            "X[s]",
            '"Avanti sensor 2: Gyro 2.X (IM)"',
            "X[s]",
            '"Avanti sensor 2: Gyro 2.Y (IM)"',
            "X[s]",
            '"Avanti sensor 2: Gyro 2.Z (IM)"',
            "X[s]",
            '"Avanti sensor 2: Mag 2.X (IM)"',
            "X[s]",
            '"Avanti sensor 2: Mag 2.Y (IM)"',
            "X[s]",
            '"Avanti sensor 2: Mag 2.Z (IM)"',
        ]
        rows = [header]
        for i in range(300):
            t = i / 100.0
            # Gravity vector points along +Z, so roll/pitch should be near 0.
            acc = (0.0, 0.0, 0.0) if zero_imu else (0.0, 0.0, 1.0)
            gyro = (0.0, 0.0, 0.0)
            mag = (0.0, 0.0, 0.0)
            row = []
            for value in (0.001, *acc, *gyro, *mag):
                row.extend((f"{t:.4f}", f"{value:.6f}"))
            rows.append(row)
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)

    def test_inventory_and_proxy_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            self._write_csv(path)
            fields = parse_header(path)
            self.assertEqual(sorted(fields), [2])
            self.assertEqual(sorted(fields[2]), ["Acc", "EMG", "Gyro", "Mag"])
            summary, proxy, inventory = analyze_file(path)
        acc_summary = next(row for row in summary if row["stream"] == "Acc")
        self.assertAlmostEqual(acc_summary["sample_rate_hz"], 100.0, places=5)
        self.assertEqual(acc_summary["valid_rows"], 300)
        self.assertEqual(acc_summary["status"], "OK")
        self.assertEqual(inventory["stream_pair_counts"]["2"]["Acc"], 3)
        self.assertGreater(len(proxy), 0)
        self.assertAlmostEqual(proxy[0]["roll_proxy_deg"], 0.0, places=6)
        self.assertAlmostEqual(proxy[0]["pitch_proxy_deg"], 0.0, places=6)
        self.assertEqual(proxy[0]["gravity_consistent_fraction"], 1.0)
        self.assertEqual(proxy[0]["tilt_quality_gate"], "PASS_SOURCE_UNIT_HEURISTIC")
        self.assertEqual(proxy[0]["orientation_status"], "TILT_PROXY_ONLY")
        self.assertEqual(proxy[0]["yaw_status"], "UNAVAILABLE_MAG_UNCALIBRATED")
        self.assertEqual(proxy[0]["gyro_status"], "ZERO_OR_UNINFORMATIVE")
        self.assertNotIn("yaw_deg", proxy[0])
        self.assertEqual(inventory["interpretation"]["orientation_status"], "TILT_PROXY_ONLY")

    def test_all_zero_acceleration_is_not_claimed_as_gravity_tilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero.csv"
            self._write_csv(path, zero_imu=True)
            summary, proxy, _ = analyze_file(path)
        acc_summary = next(row for row in summary if row["stream"] == "Acc")
        self.assertEqual(acc_summary["zero_row_percent"], 100.0)
        self.assertEqual(acc_summary["status"], "ALL_ZERO")
        self.assertTrue(proxy)
        self.assertEqual(proxy[0]["gravity_consistent_fraction"], 0.0)
        self.assertEqual(proxy[0]["tilt_proxy_status"], "NO_GRAVITY_SIGNAL")
        self.assertEqual(proxy[0]["tilt_quality_gate"], "FAIL_NO_GRAVITY_SIGNAL")
        self.assertEqual(proxy[0]["dynamic_accel_rms_source_unit"], 0.0)

    def test_stream_summary_empty_is_explicit(self) -> None:
        result = _stream_summary(np.empty((0, 2)), "Acc")
        self.assertEqual(result["valid_rows"], 0)
        self.assertTrue(np.isnan(result["sample_rate_hz"]))


if __name__ == "__main__":
    unittest.main()
