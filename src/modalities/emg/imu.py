"""Inventory Avanti EMG CSV IMU fields and compute conservative kinematic proxies.

The export contains asynchronous time/value pairs.  This script keeps the
source timestamps, estimates each stream rate, and reports acceleration-based
tilt only as a quasi-static proxy.  It never interprets EMG as kinematics and
does not integrate gyro into an absolute pose (yaw/drift are unobservable
without calibration and a declared frame convention).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


STREAMS = ("EMG", "Acc", "Gyro", "Mag")
SENSOR_RE = re.compile(r"Avanti sensor\s+(\d+):\s*(EMG|Acc|Gyro|Mag)\s+\d+", re.I)
# The export does not declare acceleration units.  This is deliberately a
# source-unit heuristic (validated against the observed ~1-unit norm), not a
# conversion to SI gravity.
GRAVITY_BAND_SOURCE_UNIT = (0.75, 1.25)
MIN_GRAVITY_CONSISTENT_FRACTION = 0.5


def parse_header(path: Path) -> dict[int, dict[str, list[tuple[int, int, str]]]]:
    """Return sensor -> stream -> [(timestamp column, value column, label)]."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    result: dict[int, dict[str, list[tuple[int, int, str]]]] = {}
    for value_col, label in enumerate(header):
        match = SENSOR_RE.search(label or "")
        if not match or value_col == 0:
            continue
        sensor = int(match.group(1))
        stream = {"emg": "EMG", "acc": "Acc", "gyro": "Gyro", "mag": "Mag"}[match.group(2).lower()]
        result.setdefault(sensor, {}).setdefault(stream, []).append(
            (value_col - 1, value_col, label.strip())
        )
    # Preserve X/Y/Z order as exported.  Duplicate channels are retained and
    # flagged in the summary rather than silently dropped.
    return result


def _rate(time: np.ndarray) -> float:
    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    d = np.diff(t)
    d = d[d > 0]
    return float(1.0 / np.median(d)) if d.size else float("nan")


def _stream_arrays(path: Path, columns: list[tuple[int, int, str]], chunksize: int = 100_000):
    """Read one stream's xyz pairs in chunks, preserving asynchronous clocks."""
    usecols = [index for pair in columns for index in pair[:2]]
    values: list[np.ndarray] = []
    for chunk in pd.read_csv(
        path,
        header=None,
        skiprows=1,
        usecols=usecols,
        names=[f"c{i}" for i in range(len(usecols))],
        chunksize=chunksize,
        na_values=("", "nan", "NaN"),
    ):
        values.append(chunk.to_numpy(dtype=float))
    if not values:
        return np.empty((0, len(columns) * 2), dtype=float)
    return np.concatenate(values, axis=0)


def _stream_summary(arr: np.ndarray, stream: str) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {"stream": stream, "columns": int(arr.shape[1] // 2)}
    if arr.size == 0:
        out.update(valid_rows=0, duration_s=float("nan"), sample_rate_hz=float("nan"), status="NO_VALID_ROWS")
        return out
    # A row is valid only when all axes have timestamp and value.
    t = arr[:, 0::2]
    x = arr[:, 1::2]
    valid = np.isfinite(t).all(axis=1) & np.isfinite(x).all(axis=1)
    if not valid.any():
        out.update(valid_rows=0, duration_s=float("nan"), sample_rate_hz=float("nan"), status="NO_VALID_ROWS")
        return out
    tv = t[valid, 0]
    xv = x[valid]
    rate = _rate(tv)
    duration = float(tv[-1] - tv[0]) if tv.size > 1 else 0.0
    out.update(
        valid_rows=int(valid.sum()),
        duration_s=duration,
        sample_rate_hz=rate,
        zero_row_percent=float(np.mean(np.all(xv == 0, axis=1)) * 100),
        # Rows after a slower stream ends are NaN because all streams are
        # exported side-by-side; this is occupancy, not within-stream loss.
        csv_row_occupancy_percent=float(np.mean(valid) * 100),
        min_source_unit=float(np.min(xv)),
        max_source_unit=float(np.max(xv)),
        status="ALL_ZERO" if bool(np.all(xv == 0)) else "OK",
    )
    return out


def _imu_proxy(sensor: int, acc: np.ndarray, gyro: np.ndarray, window_s: float = 1.0) -> list[dict[str, float | int]]:
    """Compute acceleration tilt and dynamic intensity in fixed 1 s windows."""
    if acc.size == 0:
        return []
    ta = acc[:, 0::2]
    av = acc[:, 1::2]
    valid_a = np.isfinite(ta).all(axis=1) & np.isfinite(av).all(axis=1)
    if not valid_a.any():
        return []
    ta, av = ta[valid_a, 0], av[valid_a]
    tg = gyro[:, 0::2] if gyro.size else np.empty((0, 0))
    gv = gyro[:, 1::2] if gyro.size else np.empty((0, 0))
    valid_g = np.isfinite(tg).all(axis=1) & np.isfinite(gv).all(axis=1) if gyro.size else np.zeros(0, bool)
    tg, gv = (tg[valid_g, 0], gv[valid_g]) if valid_g.any() else (np.empty(0), np.empty((0, 3)))
    rate = _rate(ta)
    if not np.isfinite(rate) or rate <= 0:
        return []
    start, stop = float(ta[0]), float(ta[-1])
    rows: list[dict[str, float | int]] = []
    nwin = int(np.floor((stop - start) / window_s)) + 1
    for wi in range(nwin):
        lo, hi = start + wi * window_s, start + (wi + 1) * window_s
        mask = (ta >= lo) & (ta < hi)
        if mask.sum() < max(5, int(rate * 0.25)):
            continue
        a = av[mask]
        norm = np.linalg.norm(a, axis=1)
        # Tilt is meaningful only when acceleration is dominated by gravity.
        gravity_fraction = float(np.mean((norm >= GRAVITY_BAND_SOURCE_UNIT[0]) & (norm <= GRAVITY_BAND_SOURCE_UNIT[1])))
        roll = np.degrees(np.arctan2(a[:, 1], a[:, 2]))
        pitch = np.degrees(np.arctan2(-a[:, 0], np.sqrt(a[:, 1] ** 2 + a[:, 2] ** 2)))
        if gv.size:
            gm = (tg >= lo) & (tg < hi)
            gnorm = np.linalg.norm(gv[gm], axis=1) if gm.any() else np.empty(0)
        else:
            gnorm = np.empty(0)
        rows.append(
            {
                "sensor_id": sensor,
                "window_start_s": lo,
                "window_end_s": hi,
                "accel_valid_samples": int(mask.sum()),
                "accel_norm_median_source_unit": float(np.median(norm)),
                "accel_norm_iqr_source_unit": float(np.percentile(norm, 75) - np.percentile(norm, 25)),
                "gravity_consistent_fraction": gravity_fraction,
                "tilt_proxy_status": "OK" if gravity_fraction >= MIN_GRAVITY_CONSISTENT_FRACTION else "NO_GRAVITY_SIGNAL",
                "tilt_quality_gate": "PASS_SOURCE_UNIT_HEURISTIC" if gravity_fraction >= MIN_GRAVITY_CONSISTENT_FRACTION else "FAIL_NO_GRAVITY_SIGNAL",
                # These statuses prevent downstream consumers from treating
                # dynamic rate or magnetic channels as an integrated pose.
                "orientation_status": "TILT_PROXY_ONLY",
                "yaw_status": "UNAVAILABLE_MAG_UNCALIBRATED",
                "roll_proxy_deg": float(np.median(roll)),
                "pitch_proxy_deg": float(np.median(pitch)),
                "dynamic_accel_rms_source_unit": float(np.sqrt(np.mean((norm - np.median(norm)) ** 2))),
                "gyro_norm_rms_source_unit": float(np.sqrt(np.mean(gnorm**2))) if gnorm.size else float("nan"),
                "gyro_valid_samples": int(gnorm.size),
                "gyro_status": (
                    "NO_GYRO_SIGNAL"
                    if not gnorm.size
                    else "ZERO_OR_UNINFORMATIVE"
                    if bool(np.all(gnorm == 0))
                    else "DYNAMIC_RATE_ONLY"
                ),
            }
        )
    return rows


def analyze_file(path: Path) -> tuple[list[dict], list[dict], dict]:
    fields = parse_header(path)
    summary_rows: list[dict] = []
    proxy_rows: list[dict] = []
    for sensor, streams in sorted(fields.items()):
        arrays: dict[str, np.ndarray] = {}
        for stream in STREAMS:
            if stream in streams:
                arrays[stream] = _stream_arrays(path, streams[stream])
                item = _stream_summary(arrays[stream], stream)
                item.update(source_file=str(path), sensor_id=sensor, field_labels="; ".join(p[2] for p in streams[stream]))
                summary_rows.append(item)
        if "Acc" in arrays:
            for row in _imu_proxy(sensor, arrays["Acc"], arrays.get("Gyro", np.empty((0, 0)))):
                row["source_file"] = str(path)
                proxy_rows.append(row)
    inventory = {
        "source_file": str(path),
        "sensor_ids": sorted(fields),
        "streams_by_sensor": {str(s): sorted(v) for s, v in sorted(fields.items())},
        "stream_pair_counts": {str(s): {k: len(v) for k, v in sorted(st.items())} for s, st in sorted(fields.items())},
        "interpretation": {
            "acc": "Acc X/Y/Z; source unit and body-frame axis directions are not declared in CSV",
            "gyro": "Gyro X/Y/Z; source unit/range convention is not declared (values resemble angular-rate export)",
            "mag": "Mag X/Y/Z; magnetic field units and calibration are not declared",
            "pose_limit": "roll/pitch are gravity-referenced proxies only; no absolute yaw or calibrated segment pose",
            "orientation_status": "TILT_PROXY_ONLY",
            "tilt_quality_gate": {
                "accel_norm_source_unit_band": list(GRAVITY_BAND_SOURCE_UNIT),
                "minimum_fraction": MIN_GRAVITY_CONSISTENT_FRACTION,
                "meaning": "heuristic source-unit consistency; not an SI gravity/unit conversion",
            },
            "gyro_status": "DYNAMIC_RATE_ONLY; unit/range and bias are unknown; no integration",
            "yaw_status": "UNAVAILABLE_MAG_UNCALIBRATED; no hard/soft-iron calibration or body-frame heading",
        },
    }
    return summary_rows, proxy_rows, inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/imu_posture_20260904"))
    args = parser.parse_args()
    files = sorted(args.data_root.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files under {args.data_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict] = []
    all_proxy: list[dict] = []
    inventories: list[dict] = []
    for path in files:
        summary, proxy, inventory = analyze_file(path)
        all_summary.extend(summary)
        all_proxy.extend(proxy)
        inventories.append(inventory)
        print(f"{path}: sensors={inventory['sensor_ids']}")
    pd.DataFrame(all_summary).to_csv(args.output_root / "imu_stream_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_proxy).to_csv(args.output_root / "imu_kinematic_proxy_1s.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "imu_field_inventory.json").write_text(json.dumps(inventories, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = args.output_root / "README.md"
    readme.write_text(
        "# EMG CSV IMU field inventory and conservative proxy\n\n"
        "CSV headers expose Avanti `Acc`, `Gyro`, and `Mag` X/Y/Z pairs in addition to EMG. "
        "The script estimates stream rates from each stream's own `X[s]` timestamps. "
        "`csv_row_occupancy_percent` is the fraction of side-by-side CSV rows populated by a stream; "
        "low values are expected for slower streams and are not interpreted as packet loss. "
        "`imu_kinematic_proxy_1s.csv` reports gravity-based roll/pitch proxies and dynamic "
        "acceleration/angular-rate intensity. These are not calibrated absolute poses: axis "
        "directions, units, sensor-to-segment mounting, magnetic calibration, and synchronization "
        "are not declared in the export; gyro integration and yaw are therefore intentionally omitted. "
        "The tilt gate is a source-unit norm heuristic (0.75--1.25, at least 50% of samples), not a "
        "claim that values are SI g. `orientation_status=TILT_PROXY_ONLY`, "
        "`yaw_status=UNAVAILABLE_MAG_UNCALIBRATED`, and `gyro_status=DYNAMIC_RATE_ONLY` are explicit "
        "downstream gates. `status=ALL_ZERO` and `tilt_proxy_status=NO_GRAVITY_SIGNAL` identify unusable zero channels.\n\n"
        "Reproduce from the full root: `python scripts/analyze_emg_imu.py --data-root data --output-root outputs/imu_posture_20260904`\n"
        "Reproduce the paired clean subset: `python scripts/analyze_emg_imu.py --data-root data_repro_20260903 --output-root outputs/imu_posture_repro_20260903`\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(all_summary)} stream rows and {len(all_proxy)} 1-s proxy rows to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
