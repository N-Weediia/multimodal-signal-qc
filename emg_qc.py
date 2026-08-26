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
SENSOR_MUSCLES = {
    2: "斜方肌（上部）",
    3: "三角肌前束",
    4: "三角肌中束",
    5: "肱二头肌",
    9: "肱三头肌",
    10: "肱桡肌",
    11: "尺侧腕屈肌",
}


@dataclass
class EMGSeries:
    sensor_id: int
    label: str
    muscle: str
    time: np.ndarray
    values: np.ndarray
    sample_rate_hz: float


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
                muscle=SENSOR_MUSCLES.get(sensor_id, "未在校正采集映射中定义"),
                time=time,
                values=values,
                sample_rate_hz=sample_rate,
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
    repeated_extreme = float(max(np.mean(values == minimum), np.mean(values == maximum)) * 100.0)

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
    if line_percent >= 20.0:
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
    )


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
    configured_sensors = set(SENSOR_MUSCLES)
    missing_sensors = sorted(configured_sensors - actual_sensors)
    unconfigured_sensors = sorted(actual_sensors - configured_sensors)
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
        "configured_sensor_ids": sorted(configured_sensors),
        "sensor_muscle_mapping": {str(sensor_id): SENSOR_MUSCLES[sensor_id] for sensor_id in sorted(configured_sensors)},
        "mapping_basis": "现场校正：以实际单元序列 2、3、4、5、9、10、11 依次对应七块目标肌肉",
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
            "line_noise_poor_percent": 20.0,
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
        "- 根据现场校正，附件中的七块目标肌肉整体后移到实际采集单元序列 2、3、4、5、9、10、11。",
        "- 对应关系：2 号斜方肌（上部），3 号三角肌前束，4 号三角肌中束，5 号肱二头肌，9 号肱三头肌，10 号肱桡肌，11 号尺侧腕屈肌。",
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
