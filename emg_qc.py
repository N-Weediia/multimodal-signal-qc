"""Offline quality screening for Delsys/Avanti surface-EMG CSV exports."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, signal


LINE_FREQUENCY_HZ = 50.0
EMG_BAND_HZ = (20.0, 450.0)
EMG_EPOCH_SECONDS = 1.0
EMG_WINDOW_MIN_WEIGHT = 0.25

# The user-confirmed acquisition table is authoritative for the current
# dataset. It contains seven physical Avanti sensors; the gaps 6--8 are not
# missing logical channels and must not trigger any channel renumbering.
ACTUAL_SENSOR_MUSCLES = {
    2: "斜方肌（上部）",
    3: "三角肌前束",
    4: "三角肌中束",
    5: "肱二头肌",
    9: "肱三头肌",
    10: "肱桡肌",
    11: "尺侧腕屈肌",
}
EXPECTED_SENSOR_IDS = tuple(ACTUAL_SENSOR_MUSCLES)
MAPPING_PROFILE_STANDARD = "standard_v1"
MAPPING_PROFILE_UNRESOLVED = "unresolved_v1"

# Logical index follows the order in the confirmed table, not the numeric
# value of the physical sensor ID (sensor 9 is logical muscle 5).
SENSOR_TO_MUSCLE_INDEX = {
    sensor_id: index for index, sensor_id in enumerate(EXPECTED_SENSOR_IDS, start=1)
}
SENSOR_MUSCLES = dict(ACTUAL_SENSOR_MUSCLES)
# Compatibility name retained for older reports; new reports also expose the
# explicit actual_sensor_muscles field below.
LEGACY_SENSOR_MUSCLES = dict(ACTUAL_SENSOR_MUSCLES)


def muscle_index_for_sensor(sensor_id: int) -> int | None:
    """Return the table-order logical muscle index for a physical sensor."""
    return SENSOR_TO_MUSCLE_INDEX.get(int(sensor_id))


def muscle_label_for_sensor(sensor_id: int) -> str:
    """Return an anatomical/placeholder label without renumbering channels."""
    sensor_id = int(sensor_id)
    if sensor_id in SENSOR_MUSCLES:
        return SENSOR_MUSCLES[sensor_id]
    return f"unconfigured_sensor_{sensor_id}"


@dataclass
class EMGSeries:
    sensor_id: int
    label: str
    muscle: str
    time: np.ndarray
    values: np.ndarray
    sample_rate_hz: float
    muscle_index: int | None = None
    canonical_sensor_id: int | None = None
    mapping_profile_id: str = MAPPING_PROFILE_UNRESOLVED
    mapping_status: str = "unresolved"
    mapping_confidence: str = "低"
    mapping_basis: str = ""
    mapping_valid_for_primary_features: bool = False


@dataclass
class EMGChannelMetrics:
    channel: str
    sensor_id: int
    target_muscle: str
    sample_rate_hz: float
    duration_seconds: float
    valid_samples: int
    missing_sample_percent: float
    nonmonotonic_timestamp_count: int
    maximum_timestamp_gap_ms: float
    zero_sample_percent: float
    flat_epoch_percent: float
    repeated_extreme_percent: float
    rms_source_unit: float
    robust_std_source_unit: float
    line_noise_power_percent: float
    low_frequency_power_percent: float
    quality_score: float
    status: str
    reasons: str
    muscle_index: int | None = None
    canonical_sensor_id: int | None = None
    mapping_profile_id: str = MAPPING_PROFILE_UNRESOLVED
    mapping_status: str = "unresolved"
    mapping_confidence: str = "低"
    mapping_valid_for_primary_features: bool = False


def resolve_emg_mapping(sensor_ids: list[int] | tuple[int, ...] | set[int], recording: Path | None = None) -> dict[str, Any]:
    """Resolve physical sensor IDs to stable logical muscle indices.

    Standard records retain the confirmed physical IDs.  Partial or shifted
    exports remain unresolved rather than being silently reinterpreted.
    """
    raw_ids = [int(value) for value in sensor_ids]
    observed = tuple(sorted(set(raw_ids)))
    duplicate_ids = len(raw_ids) != len(set(raw_ids))
    canonical = tuple(sorted(EXPECTED_SENSOR_IDS))
    if duplicate_ids:
        profile_id = MAPPING_PROFILE_UNRESOLVED
        status = "unresolved"
        confidence = "低"
        basis = "EMG 表头包含重复物理 sensor ID，无法建立一一映射"
        physical_to_logical = {}
        physical_to_canonical = {}
        labels = {}
        valid = False
        expected = canonical
    elif observed == canonical:
        profile_id = MAPPING_PROFILE_STANDARD
        status = "standard"
        confidence = "高"
        basis = "用户确认的标准物理 sensor 映射表"
        physical_to_logical = {sensor_id: SENSOR_TO_MUSCLE_INDEX[sensor_id] for sensor_id in observed}
        physical_to_canonical = {sensor_id: sensor_id for sensor_id in observed}
        labels = {sensor_id: SENSOR_MUSCLES[sensor_id] for sensor_id in observed}
        valid = True
        expected = canonical
    else:
        profile_id = MAPPING_PROFILE_STANDARD if set(observed).issubset(set(EXPECTED_SENSOR_IDS)) else MAPPING_PROFILE_UNRESOLVED
        status = "standard_partial" if profile_id == MAPPING_PROFILE_STANDARD else "unresolved"
        confidence = "中" if profile_id == MAPPING_PROFILE_STANDARD else "低"
        basis = "观测通道属于标准表的子集；未发生自动重排" if profile_id == MAPPING_PROFILE_STANDARD else "观测通道与已确认映射不一致，等待记录级核对"
        physical_to_logical = {sensor_id: SENSOR_TO_MUSCLE_INDEX[sensor_id] for sensor_id in observed if sensor_id in SENSOR_TO_MUSCLE_INDEX}
        physical_to_canonical = {sensor_id: sensor_id for sensor_id in physical_to_logical}
        labels = {sensor_id: SENSOR_MUSCLES[sensor_id] for sensor_id in physical_to_logical}
        valid = profile_id == MAPPING_PROFILE_STANDARD
        expected = canonical
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    canonical_missing = sorted(set(canonical) - set(observed))
    canonical_unexpected = sorted(set(observed) - set(canonical))
    return {
        "mapping_profile_id": profile_id,
        "mapping_status": status,
        "mapping_confidence": confidence,
        "mapping_basis": basis,
        "mapping_source": (
            "user_confirmed_record_override"
            if status == "record_override"
            else "record_review_required"
            if status == "unresolved"
            else "confirmed_sensor_table"
        ),
        "mapping_valid_for_primary_features": valid,
        "physical_sensor_ids": list(observed),
        "mapping_expected_sensor_ids": list(expected),
        "mapping_missing_sensor_ids": missing,
        "mapping_unexpected_sensor_ids": unexpected,
        "canonical_expected_sensor_ids": list(canonical),
        "canonical_missing_sensor_ids": canonical_missing,
        "canonical_unexpected_sensor_ids": canonical_unexpected,
        "physical_to_logical": {str(key): value for key, value in sorted(physical_to_logical.items())},
        "physical_to_canonical": {str(key): value for key, value in sorted(physical_to_canonical.items())},
        "logical_labels": {str(key): labels[key] for key in sorted(labels)},
    }


def _sensor_columns(header: list[str]) -> list[tuple[int, int, int, str]]:
    columns: list[tuple[int, int, int, str]] = []
    for signal_index, name in enumerate(header):
        match = re.search(r"Avanti sensor\s+(\d+):\s*EMG\s+\d+", name, re.IGNORECASE)
        if match and signal_index > 0:
            columns.append((int(match.group(1)), signal_index - 1, signal_index, name.strip()))
    if not columns:
        raise ValueError("CSV 中未找到 Avanti EMG 通道。")
    return columns


def _estimate_max_rows(path: Path, columns: list[tuple[int, int, int, str]], max_seconds: float | None) -> int | None:
    if max_seconds is None:
        return None
    times: dict[int, list[float]] = {sensor_id: [] for sensor_id, *_ in columns}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row_index, row in enumerate(reader):
            for sensor_id, time_index, _, _ in columns:
                if time_index < len(row) and row[time_index]:
                    try:
                        times[sensor_id].append(float(row[time_index]))
                    except ValueError:
                        pass
            if row_index >= 31:
                break
    rates = []
    for values in times.values():
        differences = np.diff(values)
        differences = differences[differences > 0]
        if differences.size:
            rates.append(1.0 / float(np.median(differences)))
    if not rates:
        return None
    return max(2, int(math.ceil(max_seconds * max(rates))) + 64)


def read_emg_csv(path: Path, max_seconds: float | None) -> list[EMGSeries]:
    """Read only time/EMG column pairs and preserve each sensor's timestamps."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    columns = _sensor_columns(header)
    mapping = resolve_emg_mapping([sensor_id for sensor_id, *_ in columns], path)
    usecols = [column for _, time_index, signal_index, _ in columns for column in (time_index, signal_index)]
    raw = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=usecols,
        filling_values=np.nan,
        invalid_raise=False,
        max_rows=_estimate_max_rows(path, columns, max_seconds),
    )
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]

    series: list[EMGSeries] = []
    for pair_index, (sensor_id, _, _, label) in enumerate(columns):
        time = raw[:, pair_index * 2]
        values = raw[:, pair_index * 2 + 1]
        valid = np.isfinite(time) & np.isfinite(values)
        time = time[valid]
        values = values[valid]
        if time.size < 3:
            raise ValueError(f"传感器 {sensor_id} 的有效 EMG 样本不足。")
        if max_seconds is not None:
            keep = time <= time[0] + max_seconds
            time = time[keep]
            values = values[keep]
        positive_differences = np.diff(time)
        positive_differences = positive_differences[positive_differences > 0]
        if positive_differences.size == 0:
            raise ValueError(f"传感器 {sensor_id} 的时间戳无法确定采样率。")
        sample_rate = 1.0 / float(np.median(positive_differences))
        series.append(
            EMGSeries(
                sensor_id=sensor_id,
                label=label,
                muscle=mapping["logical_labels"].get(str(sensor_id), muscle_label_for_sensor(sensor_id)),
                time=time,
                values=values,
                sample_rate_hz=sample_rate,
                muscle_index=mapping["physical_to_logical"].get(str(sensor_id)),
                canonical_sensor_id=mapping["physical_to_canonical"].get(str(sensor_id)),
                mapping_profile_id=mapping["mapping_profile_id"],
                mapping_status=mapping["mapping_status"],
                mapping_confidence=mapping["mapping_confidence"],
                mapping_basis=mapping["mapping_basis"],
                mapping_valid_for_primary_features=mapping["mapping_valid_for_primary_features"],
            )
        )
    return series


def _band_power(frequencies: np.ndarray, spectrum: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies <= high)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(integrate.trapezoid(spectrum[mask], frequencies[mask]))


def assess_emg_channel(series: EMGSeries) -> EMGChannelMetrics:
    time = series.time
    values = series.values
    differences = np.diff(time)
    positive = differences[differences > 0]
    nominal_dt = 1.0 / series.sample_rate_hz
    duration = max(float(time[-1] - time[0] + nominal_dt), nominal_dt)
    expected_samples = max(1, int(round(duration * series.sample_rate_hz)))
    missing_percent = max(0.0, (expected_samples - values.size) / expected_samples * 100.0)
    nonmonotonic_count = int(np.count_nonzero(differences <= 0))
    maximum_gap_ms = float(np.max(positive) * 1000.0) if positive.size else float("nan")

    centered = signal.detrend(values, type="linear")
    robust_std = float(np.median(np.abs(centered - np.median(centered))) * 1.4826)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    zero_percent = float(np.mean(values == 0) * 100.0)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    repeated_extreme = (
        100.0
        / max(1, values.size)
        * max(
            0,
            int(np.count_nonzero(values == minimum)) - 1,
            int(np.count_nonzero(values == maximum)) - 1,
        )
    )

    epoch_samples = max(2, int(round(EMG_EPOCH_SECONDS * series.sample_rate_hz)))
    epoch_count = values.size // epoch_samples
    if epoch_count:
        epochs = values[:epoch_count * epoch_samples].reshape(epoch_count, epoch_samples)
        tolerance = max(np.finfo(float).eps * max(1.0, float(np.max(np.abs(values)))) * 10.0, 1e-15)
        flat_percent = float(np.mean(np.ptp(epochs, axis=1) <= tolerance) * 100.0)
    else:
        flat_percent = 0.0

    nperseg = min(max(256, int(round(series.sample_rate_hz * 2))), values.size)
    frequencies, spectrum = signal.welch(centered, fs=series.sample_rate_hz, nperseg=nperseg)
    upper = min(EMG_BAND_HZ[1], series.sample_rate_hz * 0.45)
    broadband_power = _band_power(frequencies, spectrum, EMG_BAND_HZ[0], upper)
    line_power = 0.0
    harmonic = LINE_FREQUENCY_HZ
    while harmonic <= upper:
        line_power += _band_power(frequencies, spectrum, harmonic - 1.0, harmonic + 1.0)
        harmonic += LINE_FREQUENCY_HZ
    line_percent = 100.0 * line_power / max(broadband_power, np.finfo(float).tiny)
    total_power = _band_power(frequencies, spectrum, 0.5, upper)
    low_frequency_percent = 100.0 * _band_power(frequencies, spectrum, 0.5, 20.0) / max(
        total_power, np.finfo(float).tiny
    )

    score = 100.0
    reasons: list[str] = []
    if missing_percent >= 5.0 or nonmonotonic_count > 0:
        score -= 50
        reasons.append(f"估计丢样 {missing_percent:.2f}% 或时间戳不单调 {nonmonotonic_count} 次")
    elif missing_percent >= 0.5:
        score -= 20
        reasons.append(f"估计丢样 {missing_percent:.2f}%")
    elif np.isfinite(maximum_gap_ms) and maximum_gap_ms >= 500.0:
        score -= 15
        reasons.append(f"存在最大采样间隔跳变 {maximum_gap_ms:.1f} ms")
    if flat_percent >= 10.0 or robust_std == 0:
        score -= 50
        reasons.append(f"平线分段 {flat_percent:.1f}%")
    elif flat_percent > 0:
        score -= 15
        reasons.append(f"平线分段 {flat_percent:.1f}%")
    if repeated_extreme >= 1.0:
        score -= 40
        reasons.append(f"重复极值采样 {repeated_extreme:.2f}%")
    elif repeated_extreme >= 0.1:
        score -= 15
        reasons.append(f"重复极值采样 {repeated_extreme:.2f}%")
    if line_percent >= 85.0:
        score -= 50
        reasons.append(f"50 Hz 及谐波主导功率占比极高 {line_percent:.1f}%（疑似电极悬空/脱落）")
    elif line_percent >= 20.0:
        score -= 30
        reasons.append(f"50 Hz 及谐波功率占 EMG 频带 {line_percent:.1f}%")
    elif line_percent >= 5.0:
        score -= 15
        reasons.append(f"50 Hz 及谐波功率占 EMG 频带 {line_percent:.1f}%")
    if values.size < series.sample_rate_hz * 2:
        score -= 30
        reasons.append("有效记录不足 2 s")
    score = max(0.0, score)
    status = "GOOD" if score >= 80 else "CAUTION" if score >= 55 else "POOR"
    return EMGChannelMetrics(
        channel=series.label,
        sensor_id=series.sensor_id,
        target_muscle=series.muscle,
        sample_rate_hz=round(series.sample_rate_hz, 6),
        duration_seconds=duration,
        valid_samples=int(values.size),
        missing_sample_percent=missing_percent,
        nonmonotonic_timestamp_count=nonmonotonic_count,
        maximum_timestamp_gap_ms=maximum_gap_ms,
        zero_sample_percent=zero_percent,
        flat_epoch_percent=flat_percent,
        repeated_extreme_percent=repeated_extreme,
        rms_source_unit=rms,
        robust_std_source_unit=robust_std,
        line_noise_power_percent=line_percent,
        low_frequency_power_percent=low_frequency_percent,
        quality_score=score,
        status=status,
        reasons="；".join(reasons) if reasons else "未超过质量阈值",
        muscle_index=series.muscle_index,
        canonical_sensor_id=series.canonical_sensor_id,
        mapping_profile_id=series.mapping_profile_id,
        mapping_status=series.mapping_status,
        mapping_confidence=series.mapping_confidence,
        mapping_valid_for_primary_features=series.mapping_valid_for_primary_features,
    )


def emg_window_quality(
    values: np.ndarray,
    sample_rate_hz: float,
    channel_score: float = 100.0,
    window_seconds: float = EMG_EPOCH_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative per-window validity mask and soft weight.

    The mask only rejects windows with non-finite samples, a flat segment, or
    a robust amplitude excursion.  The score remains continuous so CAUTION
    channels can be retained with lower influence downstream.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    size = max(1, int(round(float(sample_rate_hz) * window_seconds)))
    count = int(np.ceil(values.size / size)) if values.size else 0
    mask = np.zeros(count, dtype=bool)
    weights = np.zeros(count, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return mask, weights
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(finite))
    scale = max(scale, np.finfo(float).eps)
    base = float(np.clip(channel_score, 0.0, 100.0) / 100.0)
    for index in range(count):
        chunk = values[index * size : (index + 1) * size]
        finite_chunk = chunk[np.isfinite(chunk)]
        if finite_chunk.size < max(3, int(0.5 * size)):
            continue
        chunk_center = float(np.median(finite_chunk))
        chunk_scale = float(1.4826 * np.median(np.abs(finite_chunk - chunk_center)))
        amplitude_z = float(np.median(np.abs(finite_chunk - center)) / scale)
        flat = chunk_scale <= max(np.finfo(float).eps, scale * 1e-4)
        # Weight mildly down-weights unusually energetic windows and strongly
        # down-weights windows dominated by an amplitude excursion.
        excursion = max(0.0, amplitude_z - 6.0)
        local = float(np.exp(-0.18 * excursion))
        weight = float(np.clip(base * local, 0.0, 1.0))
        weights[index] = weight
        mask[index] = bool(not flat and weight >= EMG_WINDOW_MIN_WEIGHT)
    return mask, weights


def _write_metrics(metrics: list[EMGChannelMetrics], destination: Path) -> None:
    rows = [asdict(metric) for metric in metrics]
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _filtered_emg(values: np.ndarray, sample_rate: float) -> np.ndarray:
    upper = min(EMG_BAND_HZ[1], sample_rate * 0.45)
    if upper <= EMG_BAND_HZ[0]:
        return signal.detrend(values)
    sos = signal.butter(4, (EMG_BAND_HZ[0], upper), btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, signal.detrend(values)).astype(float)


def _plot_preview(series: list[EMGSeries], destination: Path) -> None:
    fig, axes = plt.subplots(len(series), 1, figsize=(12, max(6, 1.55 * len(series))), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, item in zip(axes, series):
        points = min(item.values.size, int(round(item.sample_rate_hz * 5.0)))
        filtered = _filtered_emg(item.values[:points], item.sample_rate_hz)
        axis.plot(item.time[:points] - item.time[0], filtered, color="#006d77", linewidth=0.65)
        axis.set_ylabel(f"{item.sensor_id}\n源单位")
        axis.grid(alpha=0.2)
    axes[0].set_title("sEMG 前 5 秒波形（20-450 Hz，幅值单位沿用源文件）")
    axes[-1].set_xlabel("时间（s）")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_spectra(series: list[EMGSeries], destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for item in series:
        centered = signal.detrend(item.values)
        frequencies, spectrum = signal.welch(
            centered,
            fs=item.sample_rate_hz,
            nperseg=min(max(256, int(round(item.sample_rate_hz * 2))), item.values.size),
        )
        mask = (frequencies >= 5) & (frequencies <= min(500, item.sample_rate_hz * 0.48))
        axis.semilogy(frequencies[mask], spectrum[mask], linewidth=0.9, label=f"传感器 {item.sensor_id}")
    for harmonic in np.arange(LINE_FREQUENCY_HZ, 501, LINE_FREQUENCY_HZ):
        axis.axvline(harmonic, color="#b71c1c", alpha=0.18, linewidth=0.7)
    axis.set(title="sEMG 功率谱（红线为 50 Hz 及谐波）", xlabel="频率（Hz）", ylabel="功率谱密度（源单位²/Hz）")
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_quality(metrics: list[EMGChannelMetrics], destination: Path) -> None:
    labels = [f"{metric.sensor_id}: {metric.target_muscle}" for metric in metrics]
    scores = [metric.quality_score for metric in metrics]
    colors = ["#2a9d8f" if score >= 80 else "#e9c46a" if score >= 55 else "#e76f51" for score in scores]
    fig, axis = plt.subplots(figsize=(10, max(4.5, 0.65 * len(metrics))), constrained_layout=True)
    y = np.arange(len(metrics))
    axis.barh(y, scores, color=colors)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set(xlim=(0, 100), xlabel="质量分（0-100）", title="sEMG 传感器质量")
    axis.axvline(55, color="#666666", linestyle="--", linewidth=0.8)
    axis.axvline(80, color="#666666", linestyle="--", linewidth=0.8)
    for index, score in enumerate(scores):
        axis.text(min(score + 1, 96), index, f"{score:.0f}", va="center", fontsize=9)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def analyze_emg_recording(recording: Path, output_directory: Path, max_seconds: float | None) -> dict[str, Any]:
    print(f"读取 sEMG {recording}")
    series = read_emg_csv(recording, max_seconds)
    metrics = [assess_emg_channel(item) for item in series]
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_metrics(metrics, output_directory / "emg_channel_metrics.csv")
    _plot_quality(metrics, output_directory / "channel_quality.png")
    _plot_preview(series, output_directory / "signal_preview.png")
    _plot_spectra(series, output_directory / "power_spectra.png")

    actual_sensors = {item.sensor_id for item in series}
    mapping = resolve_emg_mapping(sorted(actual_sensors), recording)
    missing_sensors = mapping["mapping_missing_sensor_ids"]
    unconfigured_sensors = mapping["mapping_unexpected_sensor_ids"]
    status_counts = {status: sum(metric.status == status for metric in metrics) for status in ("GOOD", "CAUTION", "POOR")}
    mean_score = float(np.mean([metric.quality_score for metric in metrics]))
    session_status = "POOR" if status_counts["POOR"] else "CAUTION" if status_counts["CAUTION"] else "GOOD"
    if (missing_sensors or unconfigured_sensors) and session_status == "GOOD":
        session_status = "CAUTION"
    summary = {
        "modality": "EMG",
        "recording": str(recording),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_seconds_analyzed": round(max(metric.duration_seconds for metric in metrics), 3),
        "sensor_count": len(metrics),
        "sensor_ids": sorted(actual_sensors),
        "configured_sensor_ids": sorted(EXPECTED_SENSOR_IDS),
        "sensor_muscle_mapping": {
            str(sensor_id): {
                "muscle_index": mapping["physical_to_logical"].get(str(sensor_id)),
                "canonical_sensor_id": mapping["physical_to_canonical"].get(str(sensor_id)),
                "label": mapping["logical_labels"].get(str(sensor_id), muscle_label_for_sensor(sensor_id)),
                "label_role": "user_confirmed_record_mapping" if mapping["mapping_status"] == "record_override" else "user_confirmed_actual_mapping",
            }
            for sensor_id in sorted(actual_sensors)
        },
        "actual_sensor_muscles": {str(sensor_id): label for sensor_id, label in sorted(ACTUAL_SENSOR_MUSCLES.items())},
        "legacy_sensor_muscles": {str(sensor_id): label for sensor_id, label in sorted(LEGACY_SENSOR_MUSCLES.items())},
        "expected_sensor_ids": mapping["mapping_expected_sensor_ids"],
        "mapping_profile_id": mapping["mapping_profile_id"],
        "mapping_status": mapping["mapping_status"],
        "mapping_confidence": mapping["mapping_confidence"],
        "mapping_basis": mapping["mapping_basis"],
        "mapping_source": mapping["mapping_source"],
        "mapping_valid_for_primary_features": mapping["mapping_valid_for_primary_features"],
        "physical_to_logical": mapping["physical_to_logical"],
        "physical_to_canonical": mapping["physical_to_canonical"],
        "canonical_expected_sensor_ids": mapping["canonical_expected_sensor_ids"],
        "canonical_missing_sensor_ids": mapping["canonical_missing_sensor_ids"],
        "canonical_unexpected_sensor_ids": mapping["canonical_unexpected_sensor_ids"],
        "missing_configured_sensor_ids": missing_sensors,
        "unconfigured_sensor_ids": unconfigured_sensors,
        "source_amplitude_unit": "CSV 未声明；保留源数值单位",
        "quality_counts": status_counts,
        "mean_quality_score": round(mean_score, 2),
        "session_status": session_status,
        "thresholds": {
            "emg_band_hz": EMG_BAND_HZ,
            "epoch_seconds": EMG_EPOCH_SECONDS,
            "line_frequency_hz": LINE_FREQUENCY_HZ,
            "missing_sample_caution_percent": 0.5,
            "missing_sample_poor_percent": 5.0,
            "line_noise_caution_percent": 5.0,
            "maximum_timestamp_gap_caution_ms": 500.0,
            "line_noise_poor_percent": 85.0,
        },
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    poor = [str(metric.sensor_id) for metric in metrics if metric.status == "POOR"]
    caution = [str(metric.sensor_id) for metric in metrics if metric.status == "CAUTION"]
    report = [
        f"# sEMG 数据质量报告：{recording.stem}", "",
        f"- 数据文件：`{recording}`",
        f"- 实际评估时长：{summary['duration_seconds_analyzed']:.1f} s",
        f"- 实际传感器：{', '.join(map(str, sorted(actual_sensors)))}",
        f"- 平均质量分：{mean_score:.1f}/100；会话状态：`{session_status}`",
        f"- 状态统计：良好 {status_counts['GOOD']}，需关注 {status_counts['CAUTION']}，较差 {status_counts['POOR']}", "",
        "## 校正后的肌肉映射", "",
        f"- 映射配置：`{mapping['mapping_profile_id']}`；状态：`{mapping['mapping_status']}`；置信度：{mapping['mapping_confidence']}。",
        f"- 映射依据：{mapping['mapping_basis']}。",
        "- 主特征按逻辑肌肉索引 1--7 对齐；原始物理 sensor ID 仅用于追溯。",
        "- 标准映射为物理 sensor 2、3、4、5、9、10、11；该记录的前移映射为观测 sensor 1、2、3、4、5、9、10。",
        f"- 缺少校正映射中的传感器：{', '.join(map(str, missing_sensors)) or '无'}。",
        f"- 未在校正映射中定义的传感器：{', '.join(map(str, unconfigured_sensors)) or '无'}。", "",
        "## 离线质量规则", "",
        "- 逐传感器检查时间戳丢样/倒退、整段平线、重复极值，以及 50 Hz 与谐波在 20-450 Hz EMG 频带中的功率比例。",
        "- 低于 20 Hz 的功率比例只作运动/电极移动污染提示，不直接扣分，因为本文件没有任务事件，无法区分主动动作与伪迹。",
        "- CSV 表头没有声明幅值单位，因此 RMS 和稳健标准差保留源文件单位，不使用假定的 uV 绝对阈值。",
        "- 本报告是离线采集筛查，不能替代原始波形目视检查、肌肉定位复核和预先规定的伪迹处理。", "",
        "## 需要复核的传感器", "",
        f"- 较差（POOR）：{', '.join(poor) or '无'}",
        f"- 需关注（CAUTION）：{', '.join(caution) or '无'}", "",
        "逐传感器证据见 `emg_channel_metrics.csv`；波形、频谱和质量条形图见同目录 PNG 文件。",
    ]
    (output_directory / "quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"  sEMG：良好 {status_counts['GOOD']}，需关注 {status_counts['CAUTION']}，较差 {status_counts['POOR']} 个传感器")
    return summary
