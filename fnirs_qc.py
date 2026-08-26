"""Offline quality screening for NirSmart/Homer-style fNIRS ``.nirs`` files."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat


SCI_BAND_HZ = (0.5, 2.5)
FNIRS_EPOCH_SECONDS = 2.0
SCI_CAUTION = 0.80
SCI_POOR = 0.50
CV_CAUTION_PERCENT = 7.5
CV_POOR_PERCENT = 15.0
MOTION_CAUTION_PERCENT = 5.0
MOTION_POOR_PERCENT = 20.0
MOTION_MINIMUM_OD_STEP = 0.03
MOTION_MAD_MULTIPLIER = 12.0
MOTION_PADDING_SECONDS = 0.5


@dataclass
class FNIRSChannelSeries:
    channel_number: int
    source: int
    detector: int
    label: str
    wavelength_a_nm: float
    wavelength_b_nm: float
    intensity_a: np.ndarray
    intensity_b: np.ndarray
    position: tuple[float, float]


@dataclass
class FNIRSRecording:
    time: np.ndarray
    sample_rate_hz: float
    channels: list[FNIRSChannelSeries]
    source_positions: np.ndarray
    detector_positions: np.ndarray


@dataclass
class FNIRSChannelMetrics:
    channel: str
    channel_number: int
    source: int
    detector: int
    wavelength_a_nm: float
    wavelength_b_nm: float
    median_intensity_a: float
    median_intensity_b: float
    invalid_or_nonpositive_percent: float
    coefficient_of_variation_a_percent: float
    coefficient_of_variation_b_percent: float
    scalp_coupling_index: float
    motion_artifact_percent: float
    flat_epoch_percent: float
    repeated_extreme_percent: float
    quality_score: float
    status: str
    reasons: str


def is_infrared_eye_tracking_task(path: Path) -> bool:
    task_text = f"{path.parent.name} {path.stem}".casefold()
    return "眼动" in task_text or "eye" in task_text


def _as_two_dimensional_positions(value: Any, name: str) -> np.ndarray:
    positions = np.asarray(value, dtype=float)
    if positions.ndim == 1 and positions.size >= 2:
        positions = positions[None, :]
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"{name} 坐标格式无效。")
    return positions[:, :2]


def read_fnirs(path: Path, max_seconds: float | None) -> FNIRSRecording:
    document = loadmat(path, variable_names=["d", "t", "ml", "SD"], simplify_cells=True)
    missing = [key for key in ("d", "t", "SD") if key not in document]
    if missing:
        raise ValueError(f"NIRS 文件缺少必要变量：{', '.join(missing)}")
    data = np.asarray(document["d"], dtype=float)
    time = np.ravel(np.asarray(document["t"], dtype=float))
    if data.ndim != 2:
        raise ValueError("NIRS 变量 d 必须为二维矩阵。")
    if data.shape[0] != time.size and data.shape[1] == time.size:
        data = data.T
    if data.shape[0] != time.size:
        raise ValueError("NIRS 时间向量与强度矩阵长度不一致。")
    if time.size < 4:
        raise ValueError("NIRS 有效样本不足。")
    differences = np.diff(time)
    positive_differences = differences[differences > 0]
    if positive_differences.size == 0:
        raise ValueError("NIRS 时间向量无法确定采样率。")
    sample_rate = 1.0 / float(np.median(positive_differences))
    if max_seconds is not None:
        keep = time <= time[0] + max_seconds
        time = time[keep]
        data = data[keep]

    sd = document["SD"]
    if not isinstance(sd, dict):
        raise ValueError("NIRS SD 探头结构无法解析。")
    wavelengths = np.ravel(np.asarray(sd.get("Lambda"), dtype=float))
    measurement_list = np.asarray(sd.get("MeasList", document.get("ml")), dtype=int)
    if measurement_list.ndim != 2 or measurement_list.shape[1] < 4:
        raise ValueError("NIRS MeasList 格式无效。")
    if measurement_list.shape[0] != data.shape[1]:
        raise ValueError("NIRS MeasList 行数与强度通道数不一致。")
    source_positions = _as_two_dimensional_positions(sd.get("SrcPos"), "光源")
    detector_positions = _as_two_dimensional_positions(sd.get("DetPos"), "探测器")

    pair_order: list[tuple[int, int]] = []
    columns_by_pair: dict[tuple[int, int], dict[int, int]] = {}
    for column, row in enumerate(measurement_list):
        pair = (int(row[0]), int(row[1]))
        wavelength_index = int(row[3])
        if pair not in columns_by_pair:
            pair_order.append(pair)
            columns_by_pair[pair] = {}
        columns_by_pair[pair][wavelength_index] = column

    channels: list[FNIRSChannelSeries] = []
    for channel_number, pair in enumerate(pair_order, start=1):
        wavelength_columns = columns_by_pair[pair]
        if len(wavelength_columns) != 2:
            raise ValueError(f"S{pair[0]}-D{pair[1]} 未包含两个波长。")
        ordered_indices = sorted(wavelength_columns)
        if ordered_indices[0] < 1 or ordered_indices[-1] > wavelengths.size:
            raise ValueError("MeasList 中的波长索引超出 Lambda 范围。")
        column_a = wavelength_columns[ordered_indices[0]]
        column_b = wavelength_columns[ordered_indices[1]]
        source = pair[0]
        detector = pair[1]
        midpoint = (
            source_positions[source - 1, :2] + detector_positions[detector - 1, :2]
        ) / 2.0
        channels.append(
            FNIRSChannelSeries(
                channel_number=channel_number,
                source=source,
                detector=detector,
                label=f"CH{channel_number:02d} (S{source}-D{detector})",
                wavelength_a_nm=float(wavelengths[ordered_indices[0] - 1]),
                wavelength_b_nm=float(wavelengths[ordered_indices[1] - 1]),
                intensity_a=data[:, column_a],
                intensity_b=data[:, column_b],
                position=(float(midpoint[0]), float(midpoint[1])),
            )
        )
    return FNIRSRecording(
        time=time,
        sample_rate_hz=sample_rate,
        channels=channels,
        source_positions=source_positions,
        detector_positions=detector_positions,
    )


def _fill_invalid(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(result) & (result > 0)
    if np.count_nonzero(valid) < 2:
        return np.full_like(result, np.nan)
    indices = np.arange(result.size)
    result[~valid] = np.interp(indices[~valid], indices[valid], result[valid])
    return result


def _scalp_coupling_index(a: np.ndarray, b: np.ndarray, sample_rate: float) -> float:
    upper = min(SCI_BAND_HZ[1], sample_rate * 0.45)
    if upper <= SCI_BAND_HZ[0] or a.size < sample_rate * 4:
        return float("nan")
    normalized = []
    for values in (a, b):
        standard_deviation = float(np.std(values))
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            return float("nan")
        normalized.append((values - np.mean(values)) / standard_deviation)
    sos = signal.butter(3, (SCI_BAND_HZ[0], upper), btype="bandpass", fs=sample_rate, output="sos")
    filtered_a = signal.sosfiltfilt(sos, normalized[0])
    filtered_b = signal.sosfiltfilt(sos, normalized[1])
    denominator = np.sqrt(np.sum(filtered_a ** 2) * np.sum(filtered_b ** 2))
    if denominator == 0:
        return float("nan")
    return float(np.sum(filtered_a * filtered_b) / denominator)


def _motion_artifact_percent(a: np.ndarray, b: np.ndarray, sample_rate: float) -> float:
    motion = np.zeros(a.size, dtype=bool)
    for values in (a, b):
        optical_density = -np.log(values / np.median(values))
        derivative = np.abs(np.diff(optical_density, prepend=optical_density[0]))
        median = float(np.median(derivative))
        mad = float(np.median(np.abs(derivative - median)) * 1.4826)
        # A small robust threshold alone can repeatedly flag the normal pulse
        # upstroke. Requiring a 0.03 OD sample-to-sample step keeps the detector
        # focused on abrupt optode motion while the MAD term adapts to noisy data.
        threshold = max(
            MOTION_MINIMUM_OD_STEP,
            median + max(MOTION_MAD_MULTIPLIER * mad, np.finfo(float).eps),
        )
        motion |= derivative > threshold
    padding = max(1, int(round(sample_rate * MOTION_PADDING_SECONDS)))
    expanded = np.convolve(motion.astype(int), np.ones(2 * padding + 1, dtype=int), mode="same") > 0
    return float(np.mean(expanded) * 100.0)


def assess_fnirs_channel(channel: FNIRSChannelSeries, sample_rate: float) -> FNIRSChannelMetrics:
    raw_a = channel.intensity_a
    raw_b = channel.intensity_b
    valid = np.isfinite(raw_a) & np.isfinite(raw_b) & (raw_a > 0) & (raw_b > 0)
    invalid_percent = float((1.0 - np.mean(valid)) * 100.0)
    a = _fill_invalid(raw_a)
    b = _fill_invalid(raw_b)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        median_a = median_b = cv_a = cv_b = sci = motion_percent = float("nan")
        flat_percent = repeated_extreme = 100.0
    else:
        median_a = float(np.median(a))
        median_b = float(np.median(b))
        cv_a = float(np.std(a) / max(abs(np.mean(a)), np.finfo(float).tiny) * 100.0)
        cv_b = float(np.std(b) / max(abs(np.mean(b)), np.finfo(float).tiny) * 100.0)
        sci = _scalp_coupling_index(a, b, sample_rate)
        motion_percent = _motion_artifact_percent(a, b, sample_rate)

        epoch_samples = max(2, int(round(FNIRS_EPOCH_SECONDS * sample_rate)))
        epoch_count = a.size // epoch_samples
        if epoch_count:
            epochs_a = a[:epoch_count * epoch_samples].reshape(epoch_count, epoch_samples)
            epochs_b = b[:epoch_count * epoch_samples].reshape(epoch_count, epoch_samples)
            tolerance_a = max(np.finfo(float).eps * max(1.0, float(np.max(np.abs(a)))) * 10.0, 1e-12)
            tolerance_b = max(np.finfo(float).eps * max(1.0, float(np.max(np.abs(b)))) * 10.0, 1e-12)
            flat = (np.ptp(epochs_a, axis=1) <= tolerance_a) | (np.ptp(epochs_b, axis=1) <= tolerance_b)
            flat_percent = float(np.mean(flat) * 100.0)
        else:
            flat_percent = 0.0
        # One minimum and maximum are inevitable in any finite signal. Only
        # additional samples at exactly the same extreme indicate piling up.
        repeated_extreme = 100.0 / a.size * max(
            np.count_nonzero(a == np.min(a)) - 1,
            np.count_nonzero(a == np.max(a)) - 1,
            np.count_nonzero(b == np.min(b)) - 1,
            np.count_nonzero(b == np.max(b)) - 1,
        )

    score = 100.0
    reasons: list[str] = []
    if invalid_percent >= 5.0:
        score -= 50
        reasons.append(f"缺失或非正强度 {invalid_percent:.2f}%")
    elif invalid_percent >= 0.5:
        score -= 20
        reasons.append(f"缺失或非正强度 {invalid_percent:.2f}%")
    worst_cv = max(cv_a, cv_b) if np.isfinite(cv_a) and np.isfinite(cv_b) else float("inf")
    if worst_cv > CV_POOR_PERCENT:
        score -= 30
        reasons.append(f"双波长最大 CV {worst_cv:.1f}%")
    elif worst_cv > CV_CAUTION_PERCENT:
        score -= 15
        reasons.append(f"双波长最大 CV {worst_cv:.1f}%")
    if not np.isfinite(sci) or sci < SCI_POOR:
        score -= 45
        reasons.append("SCI 无效" if not np.isfinite(sci) else f"SCI {sci:.2f}")
    elif sci < SCI_CAUTION:
        score -= 20
        reasons.append(f"SCI {sci:.2f}")
    if motion_percent >= MOTION_POOR_PERCENT:
        score -= 30
        reasons.append(f"自适应运动突变覆盖 {motion_percent:.1f}%")
    elif motion_percent >= MOTION_CAUTION_PERCENT:
        score -= 15
        reasons.append(f"自适应运动突变覆盖 {motion_percent:.1f}%")
    if flat_percent >= 10.0:
        score -= 40
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
    score = max(0.0, score)
    status = "GOOD" if score >= 80 else "CAUTION" if score >= 55 else "POOR"
    return FNIRSChannelMetrics(
        channel=channel.label,
        channel_number=channel.channel_number,
        source=channel.source,
        detector=channel.detector,
        wavelength_a_nm=channel.wavelength_a_nm,
        wavelength_b_nm=channel.wavelength_b_nm,
        median_intensity_a=median_a,
        median_intensity_b=median_b,
        invalid_or_nonpositive_percent=invalid_percent,
        coefficient_of_variation_a_percent=cv_a,
        coefficient_of_variation_b_percent=cv_b,
        scalp_coupling_index=sci,
        motion_artifact_percent=motion_percent,
        flat_epoch_percent=flat_percent,
        repeated_extreme_percent=repeated_extreme,
        quality_score=score,
        status=status,
        reasons="；".join(reasons) if reasons else "未超过质量阈值",
    )


def _write_metrics(metrics: list[FNIRSChannelMetrics], destination: Path) -> None:
    rows = [asdict(metric) for metric in metrics]
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_quality_map(recording: FNIRSRecording, metrics: list[FNIRSChannelMetrics], destination: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    positions = np.asarray([channel.position for channel in recording.channels])
    scores = np.asarray([metric.quality_score for metric in metrics])
    points = axis.scatter(
        positions[:, 0], positions[:, 1], c=scores, cmap="RdYlGn", vmin=0, vmax=100,
        s=220, edgecolors="#303030", linewidths=0.6, zorder=3,
    )
    axis.scatter(recording.source_positions[:, 0], recording.source_positions[:, 1], marker="^", s=80, color="#1d3557", label="光源", zorder=2)
    axis.scatter(recording.detector_positions[:, 0], recording.detector_positions[:, 1], marker="s", s=62, color="#8d99ae", label="探测器", zorder=2)
    for channel, position in zip(recording.channels, positions):
        axis.text(position[0], position[1], str(channel.channel_number), ha="center", va="center", fontsize=7)
    axis.set(title=title, xlabel="探头模板 X（cm）", ylabel="探头模板 Y（cm）", aspect="equal")
    axis.invert_yaxis()
    axis.legend(loc="upper right")
    axis.grid(alpha=0.18)
    fig.colorbar(points, ax=axis, label="质量分（0-100）")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_preview(recording: FNIRSRecording, metrics: list[FNIRSChannelMetrics], destination: Path) -> None:
    metric_by_number = {metric.channel_number: metric for metric in metrics}
    selected = sorted(recording.channels, key=lambda item: metric_by_number[item.channel_number].quality_score)[:6]
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, max(7, 1.6 * len(selected))), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    seconds = min(60.0, float(recording.time[-1] - recording.time[0]))
    points = max(2, min(recording.time.size, int(round(seconds * recording.sample_rate_hz))))
    for axis, channel in zip(axes, selected):
        a = _fill_invalid(channel.intensity_a[:points])
        b = _fill_invalid(channel.intensity_b[:points])
        od_a = -np.log(a / np.median(a))
        od_b = -np.log(b / np.median(b))
        relative_time = recording.time[:points] - recording.time[0]
        axis.plot(relative_time, od_a, color="#457b9d", linewidth=0.8, label=f"{channel.wavelength_a_nm:.0f} nm")
        axis.plot(relative_time, od_b, color="#e76f51", linewidth=0.8, label=f"{channel.wavelength_b_nm:.0f} nm")
        axis.set_ylabel(f"CH{channel.channel_number:02d}\nOD")
        axis.grid(alpha=0.2)
    axes[0].set_title("质量分最低的 6 个 fNIRS 通道（光密度变化）")
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("时间（s）")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_spectra(recording: FNIRSRecording, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for channel in recording.channels:
        a = _fill_invalid(channel.intensity_a)
        optical_density = -np.log(a / np.median(a))
        frequencies, spectrum = signal.welch(
            optical_density,
            fs=recording.sample_rate_hz,
            nperseg=min(max(64, int(round(recording.sample_rate_hz * 20))), optical_density.size),
        )
        mask = frequencies <= min(3.0, recording.sample_rate_hz * 0.48)
        axis.semilogy(frequencies[mask], spectrum[mask], color="#457b9d", alpha=0.18, linewidth=0.7)
    axis.axvspan(SCI_BAND_HZ[0], min(SCI_BAND_HZ[1], recording.sample_rate_hz * 0.45), color="#e9c46a", alpha=0.22, label="SCI 心搏频段")
    axis.set(title="fNIRS 730 nm 光密度功率谱", xlabel="频率（Hz）", ylabel="功率谱密度（OD²/Hz）")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def analyze_fnirs_recording(path: Path, output_directory: Path, max_seconds: float | None) -> dict[str, Any]:
    print(f"读取 fNIRS {path}")
    recording = read_fnirs(path, max_seconds)
    metrics = [assess_fnirs_channel(channel, recording.sample_rate_hz) for channel in recording.channels]
    infrared_eye_tracker_interference = is_infrared_eye_tracking_task(path)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_metrics(metrics, output_directory / "fnirs_channel_metrics.csv")
    quality_title = f"fNIRS 通道质量：{path.stem}"
    if infrared_eye_tracker_interference:
        quality_title += "（已知红外眼动仪干扰）"
    _plot_quality_map(recording, metrics, output_directory / "channel_quality_map.png", quality_title)
    _plot_preview(recording, metrics, output_directory / "signal_preview.png")
    _plot_spectra(recording, output_directory / "power_spectra.png")

    status_counts = {status: sum(metric.status == status for metric in metrics) for status in ("GOOD", "CAUTION", "POOR")}
    mean_score = float(np.mean([metric.quality_score for metric in metrics]))
    poor_fraction = status_counts["POOR"] / len(metrics)
    session_status = "POOR" if poor_fraction >= 0.20 or mean_score < 55 else "CAUTION" if status_counts["POOR"] or status_counts["CAUTION"] else "GOOD"
    if infrared_eye_tracker_interference and session_status == "GOOD":
        session_status = "CAUTION"
    wavelengths = sorted({metric.wavelength_a_nm for metric in metrics} | {metric.wavelength_b_nm for metric in metrics})
    companion_hcx = path.with_suffix(".hcx")
    summary = {
        "modality": "FNIRS",
        "recording": str(path),
        "companion_hcx": str(companion_hcx) if companion_hcx.exists() else None,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_seconds_analyzed": round(float(recording.time[-1] - recording.time[0] + 1.0 / recording.sample_rate_hz), 3),
        "sample_rate_hz": round(recording.sample_rate_hz, 6),
        "source_count": int(recording.source_positions.shape[0]),
        "detector_count": int(recording.detector_positions.shape[0]),
        "physical_channel_count": len(metrics),
        "wavelengths_nm": wavelengths,
        "quality_counts": status_counts,
        "mean_quality_score": round(mean_score, 2),
        "session_status": session_status,
        "known_interference": {
            "present": infrared_eye_tracker_interference,
            "source": "红外眼动仪发射光" if infrared_eye_tracker_interference else None,
            "basis": "现场采集说明（用户确认）" if infrared_eye_tracker_interference else None,
            "interpretation": (
                "通道异常不能仅归因于头发遮挡、探头耦合或运动；当前离线指标无法单独分离眼动仪红外光贡献。"
                if infrared_eye_tracker_interference else None
            ),
        },
        "thresholds": {
            "sci_band_hz": SCI_BAND_HZ,
            "sci_caution": SCI_CAUTION,
            "sci_poor": SCI_POOR,
            "coefficient_of_variation_caution_percent": CV_CAUTION_PERCENT,
            "coefficient_of_variation_poor_percent": CV_POOR_PERCENT,
            "motion_caution_percent": MOTION_CAUTION_PERCENT,
            "motion_poor_percent": MOTION_POOR_PERCENT,
            "motion_minimum_od_step": MOTION_MINIMUM_OD_STEP,
            "motion_mad_multiplier": MOTION_MAD_MULTIPLIER,
            "motion_padding_seconds_each_side": MOTION_PADDING_SECONDS,
            "epoch_seconds": FNIRS_EPOCH_SECONDS,
        },
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    poor = [f"CH{metric.channel_number:02d}" for metric in metrics if metric.status == "POOR"]
    caution = [f"CH{metric.channel_number:02d}" for metric in metrics if metric.status == "CAUTION"]
    interference_report = []
    if infrared_eye_tracker_interference:
        interference_report = [
            "## 已知红外眼动仪干扰", "",
            "- 本眼动任务同步使用红外眼动仪，其发射光会干扰 fNIRS 原始光强采集。该背景由现场采集信息确认，不是程序从波形中反推的结论。",
            "- 因此，低 SCI、高 CV 或光密度阶跃不能仅归因于头发遮挡、探头耦合或头部运动；现有记录本身无法把眼动仪红外光贡献单独分离出来。",
            "- 通道分数保留这种实际污染的影响，不进行数值补偿。该任务的 fNIRS 结果应按已知光学混杂处理，并避免与无红外眼动仪任务直接作生理差异解释。", "",
        ]
    report = [
        f"# fNIRS 数据质量报告：{path.stem}", "",
        f"- 数据文件：`{path}`",
        f"- 配对 HCX：`{companion_hcx}`" if companion_hcx.exists() else "- 配对 HCX：未找到",
        f"- 实际评估时长：{summary['duration_seconds_analyzed']:.1f} s；采样率：{recording.sample_rate_hz:.3f} Hz",
        f"- 探头结构：{summary['source_count']} 个光源、{summary['detector_count']} 个探测器、{len(metrics)} 个物理通道",
        f"- 波长：{', '.join(f'{value:.0f} nm' for value in wavelengths)}",
        f"- 平均质量分：{mean_score:.1f}/100；会话状态：`{session_status}`",
        f"- 状态统计：良好 {status_counts['GOOD']}，需关注 {status_counts['CAUTION']}，较差 {status_counts['POOR']}", "",
        "## 与采集方法的对应", "",
        "- 附件要求在正式采集前完成预采集和自动增益，并将红色/灰色节点调整到绿色或黄色；头发遮挡是重点排查原因。",
        "- 本程序基于正式记录的双波长原始光强离线评估，不能从未公开的 HCX 状态码可靠还原厂商界面颜色，因此不会把离线分数称为厂商节点状态。", "",
    ] + interference_report + [
        "## 离线质量规则", "",
        f"- 每个源-探测器物理通道配对 {wavelengths[0]:.0f}/{wavelengths[1]:.0f} nm，检查缺失/非正强度、双波长 CV、{SCI_BAND_HZ[0]:.1f}-{SCI_BAND_HZ[1]:.1f} Hz 心搏耦合 SCI、运动突变、平线和重复极值。",
        f"- SCI < {SCI_CAUTION:.2f} 或任一波长 CV > {CV_CAUTION_PERCENT:.1f}% 进入复核；SCI < {SCI_POOR:.2f} 或 CV > {CV_POOR_PERCENT:.1f}% 给予更重扣分。",
        f"- 运动突变要求相邻样本变化至少 {MOTION_MINIMUM_OD_STEP:.2f} OD，并采用每通道自适应导数阈值；检出点向前后各扩展约 {MOTION_PADDING_SECONDS:.1f} s，仅用于筛查，不代替正式运动伪迹校正。",
        "- 本报告评估原始光强质量，不对 HbO/HbR 生理结果、激活方向或脑区功能作解释。", "",
        "## 需要复核的通道", "",
        f"- 较差（POOR）：{', '.join(poor) or '无'}",
        f"- 需关注（CAUTION）：{', '.join(caution) or '无'}", "",
        "逐通道证据见 `fnirs_channel_metrics.csv`；探头质量图、低分通道光密度波形和频谱见同目录 PNG 文件。",
    ]
    (output_directory / "quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"  fNIRS：良好 {status_counts['GOOD']}，需关注 {status_counts['CAUTION']}，较差 {status_counts['POOR']} 个通道")
    return summary
