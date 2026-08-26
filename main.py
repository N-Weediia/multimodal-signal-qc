"""EEG、sEMG 与 fNIRS 多模态采集质量评估。

在工程目录下运行：
    python main.py --overwrite

程序默认评估 ``data/傅开强/`` 下的 BDF、Avanti CSV 与 NirSmart NIRS 文件。
通过 ``--subject`` 指定其他受试者目录。结果写入 ``results/``，不会改动原始记录。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy import signal

from emg_qc import analyze_emg_recording
from fnirs_qc import analyze_fnirs_recording


def configure_matplotlib_font() -> None:
    """Use a CJK-capable system font when a task directory has Chinese text."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_font()


# -------- Settings intended for local adjustment --------
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_RESULTS_ROOT = Path("results")
# An empty subject means "all subject folders".  This is the safe default for
# this project because a successful QC run should not silently omit newly
# acquired participants.
DEFAULT_SUBJECT = ""
LINE_FREQUENCY_HZ = 50.0
BANDPASS_HZ = (0.5, 45.0)
EPOCH_SECONDS = 2.0
ARTIFACT_PEAK_TO_PEAK_UV = 200.0
FLATLINE_PEAK_TO_PEAK_UV = 2.0
RESIDUAL_LINE_CAUTION_UV = 20.0
RESIDUAL_LINE_POOR_UV = 50.0
LINE_EPOCH_CAUTION_PERCENT = 5.0
LINE_EPOCH_POOR_PERCENT = 20.0
COMMON_LINE_CAUTION_UV = 50.0
POOR_IMPEDANCE_KOHM = 50.0
CAUTION_IMPEDANCE_KOHM = 20.0
AUXILIARY_CHANNELS = {"ECG", "HEOR", "HEOL", "VEOU", "VEOL"}

# The 32-channel cap configuration contains 31 scored scalp electrodes plus
# the system reference/ground, which is not exported as a scalp BDF channel.
ACTIVE_EEG_CHANNELS = (
    "Fpz", "Fp1", "Fp2",
    "Fz", "F3", "F4", "F7", "F8",
    "FCz", "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8",
    "POz", "Oz", "O1", "O2",
)
EEG_CONFIGURATION_NAME = "32通道缩减方案（31个头皮导联+参考/地电极）"


# Approximate 10-10 scalp positions. They are used only to make the quality
# map easy to inspect; quantitative metrics always come from the BDF signals.
POSITION_MAP = {
    "Fp1": (-0.35, 1.05), "Fpz": (0.0, 1.12), "Fp2": (0.35, 1.05),
    "AF7": (-0.72, 0.82), "AF3": (-0.28, 0.77), "AF4": (0.28, 0.77), "AF8": (0.72, 0.82),
    "F7": (-0.91, 0.52), "F5": (-0.62, 0.47), "F3": (-0.35, 0.42), "F1": (-0.12, 0.39),
    "Fz": (0.0, 0.39), "F2": (0.12, 0.39), "F4": (0.35, 0.42), "F6": (0.62, 0.47), "F8": (0.91, 0.52),
    "FT7": (-1.03, 0.17), "FC5": (-0.72, 0.14), "FC3": (-0.43, 0.12), "FC1": (-0.15, 0.10),
    "FCz": (0.0, 0.10), "FC2": (0.15, 0.10), "FC4": (0.43, 0.12), "FC6": (0.72, 0.14), "FT8": (1.03, 0.17),
    "T7": (-1.10, -0.18), "C5": (-0.74, -0.18), "C3": (-0.43, -0.18), "C1": (-0.15, -0.18),
    "Cz": (0.0, -0.18), "C2": (0.15, -0.18), "C4": (0.43, -0.18), "C6": (0.74, -0.18), "T8": (1.10, -0.18),
    "TP7": (-1.04, -0.48), "CP5": (-0.72, -0.45), "CP3": (-0.43, -0.45), "CP1": (-0.15, -0.45),
    "CP2": (0.15, -0.45), "CP4": (0.43, -0.45), "CP6": (0.72, -0.45), "TP8": (1.04, -0.48),
    "P7": (-0.91, -0.72), "P5": (-0.62, -0.68), "P3": (-0.35, -0.68), "Pz": (0.0, -0.68),
    "P4": (0.35, -0.68), "P6": (0.62, -0.68), "P8": (0.91, -0.72),
    "PO7": (-0.69, -0.93), "PO5": (-0.48, -0.88), "PO3": (-0.26, -0.91), "POz": (0.0, -0.94),
    "PO4": (0.26, -0.91), "PO6": (0.48, -0.88), "PO8": (0.69, -0.93),
    "O1": (-0.25, -1.10), "Oz": (0.0, -1.15), "O2": (0.25, -1.10),
}


@dataclass
class BDFHeader:
    labels: list[str]
    physical_dimensions: list[str]
    physical_minimums: np.ndarray
    physical_maximums: np.ndarray
    digital_minimums: np.ndarray
    digital_maximums: np.ndarray
    samples_per_record: list[int]
    header_bytes: int
    record_count: int
    record_duration_seconds: float

    @property
    def sample_rate(self) -> float:
        values = {n / self.record_duration_seconds for n in self.samples_per_record}
        if len(values) != 1:
            raise ValueError("This BDF has unequal channel sample rates, which this program does not support.")
        return values.pop()


@dataclass
class ChannelMetrics:
    channel: str
    channel_type: str
    impedance_kohm: float | None
    raw_50hz_amplitude_uv: float
    rereferenced_50hz_amplitude_uv: float
    line_noise_reduction_percent: float
    rereferenced_high_line_epoch_percent: float
    rereferenced_robust_std_uv: float
    rereferenced_median_peak_to_peak_uv: float
    rereferenced_artifact_epoch_percent: float
    rereferenced_flat_epoch_percent: float
    clipping_percent: float
    quality_score: float
    status: str
    reasons: str


def fixed_text(block: bytes) -> str:
    return block.decode("latin-1", errors="replace").strip()


def parse_bdf_header(path: Path) -> BDFHeader:
    """Read an EDF/BDF fixed and signal header without loading signal data."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError("File is too small to be a valid BDF/EDF file.")
        header_bytes = int(fixed[184:192].decode("ascii").strip())
        record_count = int(fixed[236:244].decode("ascii").strip())
        record_duration = float(fixed[244:252].decode("ascii").strip())
        channel_count = int(fixed[252:256].decode("ascii").strip())
        extension = handle.read(header_bytes - 256)
    if len(extension) != (header_bytes - 256):
        raise ValueError("BDF signal header is incomplete.")

    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    fields: list[list[str]] = []
    cursor = 0
    for width in widths:
        fields.append([
            fixed_text(extension[cursor + index * width: cursor + (index + 1) * width])
            for index in range(channel_count)
        ])
        cursor += width * channel_count
    return BDFHeader(
        labels=fields[0],
        physical_dimensions=fields[2],
        physical_minimums=np.asarray(fields[3], dtype=float),
        physical_maximums=np.asarray(fields[4], dtype=float),
        digital_minimums=np.asarray(fields[5], dtype=float),
        digital_maximums=np.asarray(fields[6], dtype=float),
        samples_per_record=[int(value) for value in fields[8]],
        header_bytes=header_bytes,
        record_count=record_count,
        record_duration_seconds=record_duration,
    )


def decode_bdf_24bit(chunk: np.ndarray) -> np.ndarray:
    """Convert little-endian signed 24-bit BDF samples to int32."""
    values = (
        chunk[..., 0].astype(np.int32)
        | (chunk[..., 1].astype(np.int32) << 8)
        | (chunk[..., 2].astype(np.int32) << 16)
    )
    values[values >= 0x800000] -= 0x1000000
    return values


def read_bdf(path: Path, max_seconds: float | None) -> tuple[BDFHeader, np.ndarray, np.ndarray]:
    """Load BDF data as microvolts, returning header, physical data, digital data."""
    header = parse_bdf_header(path)
    sample_rates = {round(n / header.record_duration_seconds, 8) for n in header.samples_per_record}
    if len(sample_rates) != 1:
        raise ValueError("Only BDF files whose channels share one sample rate are supported.")
    samples_per_record = header.samples_per_record[0]
    selected_records = header.record_count
    if max_seconds is not None:
        selected_records = min(selected_records, math.ceil(max_seconds / header.record_duration_seconds))
    bytes_per_record = 3 * sum(header.samples_per_record)
    expected_bytes = selected_records * bytes_per_record
    with path.open("rb") as handle:
        handle.seek(header.header_bytes)
        packed = np.frombuffer(handle.read(expected_bytes), dtype=np.uint8)
    if packed.size != expected_bytes:
        raise ValueError(f"BDF ended early: expected {expected_bytes} data bytes, read {packed.size}.")

    packed = packed.reshape(selected_records, bytes_per_record)
    total_samples = selected_records * samples_per_record
    digital = np.empty((len(header.labels), total_samples), dtype=np.int32)
    offset = 0
    for channel_index, channel_samples in enumerate(header.samples_per_record):
        width = 3 * channel_samples
        values = decode_bdf_24bit(packed[:, offset:offset + width].reshape(selected_records, channel_samples, 3))
        digital[channel_index] = values.reshape(-1)
        offset += width
    scale = (header.physical_maximums - header.physical_minimums) / (
        header.digital_maximums - header.digital_minimums
    )
    physical = (digital.astype(np.float32) - header.digital_minimums[:, None]) * scale[:, None] + header.physical_minimums[:, None]
    return header, physical.astype(np.float32), digital


def normalize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", label).upper()


def load_impedances(record_information: Path | None) -> dict[str, float]:
    if record_information is None or not record_information.exists():
        return {}
    try:
        document = json.loads(record_information.read_text(encoding="utf-8-sig"))
        impedance = document.get("ImpedanceData", {})
        labels = impedance.get("Item1", [])
        values = impedance.get("Item2", [])
        return {normalize_label(str(label)): float(value) for label, value in zip(labels, values)}
    except (OSError, ValueError, TypeError) as exc:
        print(f"  Warning: could not read impedance values ({exc}).", file=sys.stderr)
        return {}


def clean_for_artifact_qc(data: np.ndarray, sample_rate: float) -> np.ndarray:
    """Remove DC and line noise before judging non-line artifacts.

    The narrow 50 Hz component is measured separately. This prevents a shared
    line signal from making every EEG channel look like a high-amplitude motion
    artifact, while preserving the clinical EEG below 45 Hz for quality checks.
    """
    high, low = BANDPASS_HZ
    sos = signal.butter(4, (high, low), btype="bandpass", fs=sample_rate, output="sos")
    detrended = signal.detrend(data, axis=-1, type="linear")
    notch_b, notch_a = signal.iirnotch(LINE_FREQUENCY_HZ, Q=30, fs=sample_rate)
    without_line = signal.filtfilt(notch_b, notch_a, detrended, axis=-1)
    return signal.sosfiltfilt(sos, without_line, axis=-1).astype(np.float32)


def line_statistics(
    data: np.ndarray, sample_rate: float, frequency: float = LINE_FREQUENCY_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Return median 50 Hz amplitude and percent of high-line 2-s windows.

    Computing the sinusoid in each short window avoids phase cancellation over
    long records and detects intermittent mains contamination.
    """
    epoch_samples = max(1, int(round(EPOCH_SECONDS * sample_rate)))
    epoch_count = data.shape[-1] // epoch_samples
    if epoch_count == 0:
        raise ValueError("Recording is shorter than one quality-control epoch.")
    time = np.arange(epoch_samples) / sample_rate
    kernel = np.exp(-2j * np.pi * frequency * time)
    amplitudes = np.empty((data.shape[0], epoch_count), dtype=float)
    for row in range(data.shape[0]):
        epochs = data[row, :epoch_count * epoch_samples].reshape(epoch_count, epoch_samples)
        detrended = signal.detrend(epochs, axis=-1, type="linear")
        amplitudes[row] = 2.0 * np.abs(detrended @ kernel) / epoch_samples
    return np.median(amplitudes, axis=1), np.mean(amplitudes >= RESIDUAL_LINE_POOR_UV, axis=1) * 100


def assess_channel(
    label: str,
    rereferenced_filtered: np.ndarray,
    digital: np.ndarray,
    digital_min: float,
    digital_max: float,
    sample_rate: float,
    impedance: float | None,
    raw_line_amplitude_uv: float,
    rereferenced_line_amplitude_uv: float,
    rereferenced_high_line_epoch_percent: float,
) -> ChannelMetrics:
    epoch_samples = max(1, int(round(EPOCH_SECONDS * sample_rate)))
    epoch_count = rereferenced_filtered.size // epoch_samples
    usable = rereferenced_filtered[:epoch_count * epoch_samples].reshape(epoch_count, epoch_samples)
    peak_to_peak = np.ptp(usable, axis=1)
    artifact_fraction = float(np.mean(peak_to_peak > ARTIFACT_PEAK_TO_PEAK_UV) * 100)
    flat_fraction = float(np.mean(peak_to_peak < FLATLINE_PEAK_TO_PEAK_UV) * 100)
    robust_std = float(np.median(np.abs(rereferenced_filtered - np.median(rereferenced_filtered))) * 1.4826)
    clipping = float(np.mean((digital <= digital_min) | (digital >= digital_max)) * 100)

    reasons: list[str] = []
    score = 100.0
    if artifact_fraction >= 20:
        score -= 45
        reasons.append(f"{artifact_fraction:.1f}% 分段峰峰值超过 {ARTIFACT_PEAK_TO_PEAK_UV:.0f} uV")
    elif artifact_fraction >= 5:
        score -= 20
        reasons.append(f"{artifact_fraction:.1f}% 分段峰峰值超过 {ARTIFACT_PEAK_TO_PEAK_UV:.0f} uV")
    if flat_fraction >= 10:
        score -= 45
        reasons.append(f"{flat_fraction:.1f}% 分段接近平线")
    elif flat_fraction > 0:
        score -= 15
        reasons.append(f"{flat_fraction:.1f}% 分段接近平线")
    if rereferenced_high_line_epoch_percent >= LINE_EPOCH_POOR_PERCENT:
        score -= 25
        reasons.append(
            f"{rereferenced_high_line_epoch_percent:.1f}% 分段重参考后 {LINE_FREQUENCY_HZ:.0f} Hz 残留超过 "
            f"{RESIDUAL_LINE_POOR_UV:.0f} uV"
        )
    elif rereferenced_high_line_epoch_percent >= LINE_EPOCH_CAUTION_PERCENT:
        score -= 10
        reasons.append(
            f"{rereferenced_high_line_epoch_percent:.1f}% 分段重参考后 {LINE_FREQUENCY_HZ:.0f} Hz 残留超过 "
            f"{RESIDUAL_LINE_POOR_UV:.0f} uV"
        )
    elif rereferenced_line_amplitude_uv >= RESIDUAL_LINE_CAUTION_UV:
        score -= 10
        reasons.append(f"重参考后 {LINE_FREQUENCY_HZ:.0f} Hz 残留中位振幅 {rereferenced_line_amplitude_uv:.1f} uV")
    if clipping > 0:
        score -= 50
        reasons.append(f"{clipping:.3f}% 采样点达到数字量程上限")
    if impedance is not None and impedance >= POOR_IMPEDANCE_KOHM:
        score -= 25
        reasons.append(f"记录阻抗 {impedance:.1f} kOhm 偏高")
    elif impedance is not None and impedance >= CAUTION_IMPEDANCE_KOHM:
        score -= 10
        reasons.append(f"记录阻抗 {impedance:.1f} kOhm 需关注")
    score = max(0.0, score)
    status = "GOOD" if score >= 80 else "CAUTION" if score >= 55 else "POOR"
    return ChannelMetrics(
        channel=label,
        channel_type="EEG",
        impedance_kohm=impedance,
        raw_50hz_amplitude_uv=raw_line_amplitude_uv,
        rereferenced_50hz_amplitude_uv=rereferenced_line_amplitude_uv,
        line_noise_reduction_percent=float(100 * (1 - rereferenced_line_amplitude_uv / max(raw_line_amplitude_uv, 1e-12))),
        rereferenced_high_line_epoch_percent=rereferenced_high_line_epoch_percent,
        rereferenced_robust_std_uv=robust_std,
        rereferenced_median_peak_to_peak_uv=float(np.median(peak_to_peak)),
        rereferenced_artifact_epoch_percent=artifact_fraction,
        rereferenced_flat_epoch_percent=flat_fraction,
        clipping_percent=clipping,
        quality_score=score,
        status=status,
        reasons="；".join(reasons) if reasons else "未超过质量阈值",
    )


def write_channel_csv(metrics: list[ChannelMetrics], destination: Path) -> None:
    rows = [asdict(metric) for metric in metrics]
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_quality_map(metrics: list[ChannelMetrics], destination: Path, title: str) -> None:
    located = [metric for metric in metrics if metric.channel in POSITION_MAP]
    missing = [metric.channel for metric in metrics if metric.channel not in POSITION_MAP]
    fig, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    head = plt.Circle((0, -0.02), 1.0, fill=False, linewidth=2, color="#263238")
    axis.add_patch(head)
    axis.plot((-0.12, 0, 0.12), (0.96, 1.14, 0.96), color="#263238", linewidth=2)
    axis.plot((-1.02, -1.15, -1.02), (0.08, 0, -0.08), color="#263238", linewidth=2)
    axis.plot((1.02, 1.15, 1.02), (0.08, 0, -0.08), color="#263238", linewidth=2)
    x_values = [POSITION_MAP[metric.channel][0] for metric in located]
    y_values = [POSITION_MAP[metric.channel][1] for metric in located]
    scores = [metric.quality_score for metric in located]
    points = axis.scatter(x_values, y_values, c=scores, cmap="RdYlGn", vmin=0, vmax=100, s=620, edgecolors="#303030", linewidths=0.7)
    for metric in located:
        x, y = POSITION_MAP[metric.channel]
        axis.text(x, y, metric.channel, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(points, ax=axis, label="重参考后质量分（0-100）", shrink=0.75)
    axis.set_title(title)
    if missing:
        axis.text(0, -1.42, "未绘制导联：" + ", ".join(missing), ha="center", fontsize=8)
    axis.set(xlim=(-1.35, 1.35), ylim=(-1.5, 1.32), aspect="equal")
    axis.axis("off")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_preview(filtered: np.ndarray, labels: list[str], sample_rate: float, destination: Path, title: str) -> None:
    selected = [index for index, label in enumerate(labels) if label in {"Fp1", "Fz", "Cz", "Pz", "O1", "O2"}]
    if not selected:
        selected = list(range(min(6, len(labels))))
    seconds = min(10.0, filtered.shape[1] / sample_rate)
    points = int(seconds * sample_rate)
    clip = filtered[selected, :points]
    offsets = np.arange(len(selected))[::-1] * 120
    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    time = np.arange(points) / sample_rate
    for row, (index, offset) in enumerate(zip(selected, offsets)):
        axis.plot(time, clip[row] + offset, linewidth=0.6, color="#174a7e")
        axis.text(-0.15, offset, labels[index], ha="right", va="center", fontsize=9)
    axis.set(title=title, xlabel="时间（s）", ylabel="重参考、去工频后的信号 + 偏移（uV）", xlim=(0, seconds))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_psd(filtered: np.ndarray, labels: list[str], sample_rate: float, destination: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for index, label in enumerate(labels):
        frequencies, spectrum = signal.welch(filtered[index], fs=sample_rate, nperseg=min(int(sample_rate * 4), filtered.shape[1]))
        mask = frequencies <= 60
        axis.semilogy(frequencies[mask], spectrum[mask], alpha=0.25, linewidth=0.7)
    axis.axvline(LINE_FREQUENCY_HZ, color="#b71c1c", linestyle="--", linewidth=1, label=f"{LINE_FREQUENCY_HZ:.0f} Hz")
    axis.set(title=title, xlabel="频率（Hz）", ylabel="功率谱密度（uV^2/Hz）", xlim=(0, 60))
    axis.legend()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def classify_channels(labels: list[str]) -> tuple[list[int], list[int], list[int]]:
    """Keep disabled high-impedance electrodes out of scoring and referencing."""
    eeg_indices: list[int] = []
    auxiliary_indices: list[int] = []
    disabled_eeg_indices: list[int] = []
    active_labels = {normalize_label(label) for label in ACTIVE_EEG_CHANNELS}
    for index, label in enumerate(labels):
        normalized = normalize_label(label)
        if normalized in AUXILIARY_CHANNELS or label.strip().upper() == "STATUS":
            auxiliary_indices.append(index)
        elif normalized in active_labels:
            eeg_indices.append(index)
        else:
            disabled_eeg_indices.append(index)
    return eeg_indices, auxiliary_indices, disabled_eeg_indices


def analyze_recording(recording: Path, output_directory: Path, max_seconds: float | None) -> dict[str, Any]:
    print(f"读取 {recording}")
    header, raw, digital = read_bdf(recording, max_seconds)
    sample_rate = header.sample_rate
    if raw.shape[1] < int(math.ceil(EPOCH_SECONDS * sample_rate)):
        raise ValueError(
            f"EEG 记录仅 {raw.shape[1] / sample_rate:.3f} s，短于 {EPOCH_SECONDS:.1f} s 质量评估窗口；"
            "已在批处理清单中保留为无法判定。"
        )
    eeg_indices, auxiliary_indices, disabled_eeg_indices = classify_channels(header.labels)
    present_labels = {normalize_label(header.labels[index]) for index in eeg_indices}
    missing_configured_channels = [
        label for label in ACTIVE_EEG_CHANNELS if normalize_label(label) not in present_labels
    ]
    if not eeg_indices:
        raise ValueError(
            f"BDF中没有找到配置的32通道头皮导联；期望标签：{', '.join(ACTIVE_EEG_CHANNELS)}"
        )
    labels = [header.labels[index] for index in eeg_indices]
    impedance_path = recording.with_name("recordInformation.json")
    impedances = load_impedances(impedance_path)
    raw_eeg = raw[eeg_indices]
    # A sample-wise median reference is robust to a minority of noisy or
    # disconnected electrodes (it tolerates up to half the channels being bad)
    # while still tracking the common-mode 50 Hz that appears on every channel.
    # A trimmed mean is easily contaminated once several channels are bad, which
    # then spreads their noise to every channel during re-referencing. QC-only.
    robust_reference = np.median(raw_eeg, axis=0)
    rereferenced_raw = raw_eeg - robust_reference
    rereferenced_filtered = clean_for_artifact_qc(rereferenced_raw, sample_rate)
    raw_line, _ = line_statistics(raw_eeg, sample_rate)
    rereferenced_line, rereferenced_high_line_epochs = line_statistics(rereferenced_raw, sample_rate)

    metrics = [
        assess_channel(
            label=header.labels[index], rereferenced_filtered=rereferenced_filtered[row], digital=digital[index],
            digital_min=header.digital_minimums[index], digital_max=header.digital_maximums[index],
            sample_rate=sample_rate,
            impedance=impedances.get(normalize_label(header.labels[index])),
            raw_line_amplitude_uv=float(raw_line[row]), rereferenced_line_amplitude_uv=float(rereferenced_line[row]),
            rereferenced_high_line_epoch_percent=float(rereferenced_high_line_epochs[row]),
        )
        for row, index in enumerate(eeg_indices)
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    write_channel_csv(metrics, output_directory / "channel_metrics.csv")
    plot_quality_map(metrics, output_directory / "channel_quality_map.png", f"EEG 导联质量：{recording.parent.name}")
    plot_preview(rereferenced_filtered, labels, sample_rate, output_directory / "signal_preview.png", f"10 秒稳健参考、去工频 EEG 波形：{recording.parent.name}")
    plot_psd(signal.detrend(rereferenced_raw, axis=-1), labels, sample_rate, output_directory / "power_spectra.png", f"稳健参考后 EEG 功率谱（保留 {LINE_FREQUENCY_HZ:.0f} Hz 供质控）")

    status_counts = {status: sum(metric.status == status for metric in metrics) for status in ("GOOD", "CAUTION", "POOR")}
    poor_fraction = status_counts["POOR"] / len(metrics)
    session_status = "POOR" if poor_fraction >= 0.20 else "CAUTION" if status_counts["POOR"] or status_counts["CAUTION"] else "GOOD"
    if missing_configured_channels and session_status == "GOOD":
        session_status = "CAUTION"
    summary = {
        "modality": "EEG",
        "recording": str(recording),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_seconds_analyzed": round(raw.shape[1] / sample_rate, 3),
        "sample_rate_hz": sample_rate,
        "bdf_channel_count": len(header.labels),
        "eeg_configuration": EEG_CONFIGURATION_NAME,
        "configured_scalp_channel_count": len(ACTIVE_EEG_CHANNELS),
        "eeg_channel_count": len(eeg_indices),
        "configured_scalp_channels": list(ACTIVE_EEG_CHANNELS),
        "missing_configured_scalp_channels": missing_configured_channels,
        "disabled_eeg_channels": [header.labels[index] for index in disabled_eeg_indices],
        "auxiliary_channels_excluded": [header.labels[index] for index in auxiliary_indices],
        "quality_reference": "仅由启用头皮导联构造的逐采样中位数参考（仅用于质量评估）",
        "raw_50hz_median_amplitude_uv": round(float(np.median(raw_line)), 2),
        "rereferenced_50hz_median_amplitude_uv": round(float(np.median(rereferenced_line)), 2),
        "rereferenced_high_line_epoch_median_percent": round(float(np.median(rereferenced_high_line_epochs)), 2),
        "common_line_noise_present": bool(np.median(raw_line) >= COMMON_LINE_CAUTION_UV),
        "quality_counts": status_counts,
        "mean_quality_score": round(float(np.mean([metric.quality_score for metric in metrics])), 2),
        "session_status": session_status,
        "thresholds": {
            "bandpass_hz": BANDPASS_HZ,
            "epoch_seconds": EPOCH_SECONDS,
            "artifact_peak_to_peak_uv": ARTIFACT_PEAK_TO_PEAK_UV,
            "flatline_peak_to_peak_uv": FLATLINE_PEAK_TO_PEAK_UV,
            "line_frequency_hz": LINE_FREQUENCY_HZ,
            "residual_line_caution_uv": RESIDUAL_LINE_CAUTION_UV,
            "residual_line_poor_uv": RESIDUAL_LINE_POOR_UV,
            "line_epoch_caution_percent": LINE_EPOCH_CAUTION_PERCENT,
            "line_epoch_poor_percent": LINE_EPOCH_POOR_PERCENT,
            "reference_method": "median",
            "impedance_caution_kohm": CAUTION_IMPEDANCE_KOHM,
            "impedance_poor_kohm": POOR_IMPEDANCE_KOHM,
        },
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    poor = [metric.channel for metric in metrics if metric.status == "POOR"]
    caution = [metric.channel for metric in metrics if metric.status == "CAUTION"]
    if summary["common_line_noise_present"]:
        acquisition_note = (
            f"检测到原始参考下的强共模 {LINE_FREQUENCY_HZ:.0f} Hz 工频。该问题通常与参考、地线、"
            "屏蔽或供电环境有关；已通过稳健公共参考从逐导联评分中分离，但仍应在后续采集中排查。"
        )
    else:
        acquisition_note = f"原始参考下未检出显著的共模 {LINE_FREQUENCY_HZ:.0f} Hz 工频。"
    report = [
        f"# EEG 采集质量报告：{recording.parent.name}", "",
        f"- 数据文件：`{recording}`", f"- 实际评估时长：{summary['duration_seconds_analyzed']:.1f} s",
        f"- 采样率：{sample_rate:.0f} Hz", f"- EEG配置：{EEG_CONFIGURATION_NAME}",
        f"- 已评估启用头皮 EEG 导联：{len(eeg_indices)} 个",
        f"- 配置中未找到的头皮导联：{', '.join(missing_configured_channels) or '无'}",
        f"- 未启用的头皮 EEG 导联（不参与评分）：{', '.join(summary['disabled_eeg_channels']) or '无'}",
        f"- 未纳入 EEG 评分的辅助导联：{', '.join(summary['auxiliary_channels_excluded']) or '无'}",
        "- 逐导联评分参考：仅由启用头皮导联构造的逐采样中位数参考（仅用于质量评估，不指定后续正式分析参考）",
        f"- 平均质量分：{summary['mean_quality_score']:.1f}/100；会话状态：`{session_status}`",
        f"- 状态统计：良好 {status_counts['GOOD']}，需关注 {status_counts['CAUTION']}，较差 {status_counts['POOR']}", "",
        "## 采集参考状态", "",
        f"- 原始参考下 {LINE_FREQUENCY_HZ:.0f} Hz 中位振幅：{summary['raw_50hz_median_amplitude_uv']:.1f} uV",
        f"- 稳健重参考后 {LINE_FREQUENCY_HZ:.0f} Hz 中位残留：{summary['rereferenced_50hz_median_amplitude_uv']:.1f} uV",
        f"- 导联层面高残余工频分段比例的中位数：{summary['rereferenced_high_line_epoch_median_percent']:.1f}%",
        f"- {acquisition_note}", "",
        "## 状态说明", "",
        "- `GOOD（良好）`：未超过重要质量阈值。",
        "- `CAUTION（需关注）`：使用前应目视检查；经伪迹处理后可能仍可使用。",
        "- `POOR（较差）`：建议尽可能重采，或在分析前预先规定排除/插值规则。",
        "- 原始参考下的共模工频单独报告，不会直接使所有导联判为较差；逐导联状态基于稳健重参考后的残余工频、非工频伪迹、平线、量程饱和和阻抗。",
        "- 本评估不因脑卒中后常见的慢波增多、侧化或局部振幅改变扣分；这些需在临床/研究分析阶段结合病灶与任务解释。",
        "- 阻抗值直接读取厂商 JSON；仅当采集软件显示单位确为 kOhm 时才按 kOhm 解读，作出硬件判断前请核对设备显示单位。",
        "- 本报告用于采集质量筛查，不构成临床解释，不能替代人工波形检查。", "",
        "## 需要复核的导联", "",
        f"- 较差（POOR）：{', '.join(poor) or '无'}",
        f"- 需关注（CAUTION）：{', '.join(caution) or '无'}", "",
        "逐导联数值证据见 `channel_metrics.csv`；头皮图、波形和频谱见同目录 PNG 文件。",
    ]
    (output_directory / "quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"  良好 {status_counts['GOOD']}，需关注 {status_counts['CAUTION']}，较差 {status_counts['POOR']} 个 EEG 导联")
    return summary


def recording_modality(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".bdf" and path.name.lower() != "evt.bdf":
        return "EEG"
    if suffix == ".csv":
        if any(part.upper() == "EMG" for part in path.parts):
            return "EMG"
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                header = handle.readline()
            if "Avanti sensor" in header and ": EMG " in header:
                return "EMG"
        except OSError:
            return None
    if suffix == ".nirs":
        return "FNIRS"
    return None


def discover_recordings(data_root: Path, input_path: Path | None) -> list[Path]:
    if input_path is not None:
        if input_path.is_file():
            if input_path.suffix.lower() == ".hcx" and input_path.with_suffix(".nirs").exists():
                input_path = input_path.with_suffix(".nirs")
            return [input_path] if recording_modality(input_path) else []
        search_root = input_path
    else:
        search_root = data_root
    recordings = [path for path in search_root.rglob("*") if path.is_file() and recording_modality(path)]
    modality_order = {"EEG": 0, "EMG": 1, "FNIRS": 2}
    return sorted(recordings, key=lambda path: (modality_order[recording_modality(path)], str(path)))


TASK_LABELS = (
    ("静息态", ("静息", "rest")),
    ("运动任务", ("运动", "movement", "motor")),
    ("眼动任务", ("眼动", "eye")),
)


def task_label_for(recording: Path) -> str:
    """Return a stable task label without changing the source folder names."""
    searchable = " ".join(part.casefold() for part in (*recording.parts, recording.stem))
    for label, tokens in TASK_LABELS:
        if any(token.casefold() in searchable for token in tokens):
            return label
    return "未识别任务"


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*\x00-\x1f]', "_", value).strip(". ")
    return cleaned or "unnamed_recording"


def output_path_for(recording: Path, data_root: Path, results_root: Path) -> Path:
    """Return a unique result directory for exactly one source recording.

    Vendors may create ``data.1.bdf`` beside ``data.bdf`` and operators may
    repeat a task without renaming its fNIRS export.  A result directory must
    therefore include the source filename whenever a collision is possible.
    """
    modality = recording_modality(recording)
    try:
        base = results_root / recording.parent.relative_to(data_root)
    except ValueError:
        base = results_root / (modality or "UNKNOWN")
    if modality == "EEG":
        bdf_exports = [path for path in recording.parent.glob("*.bdf") if path.name.casefold() != "evt.bdf"]
        return base / _safe_path_component(recording.stem) if len(bdf_exports) > 1 else base
    if modality == "FNIRS":
        return base / task_label_for(recording) / _safe_path_component(recording.stem)
    return base / recording.stem


def subject_for(recording: Path, data_root: Path) -> str:
    try:
        return recording.relative_to(data_root).parts[0]
    except (ValueError, IndexError):
        return "外部输入"


def recording_metadata(recording: Path, data_root: Path, output_directory: Path) -> dict[str, Any]:
    stat = recording.stat()
    try:
        relative_path = recording.relative_to(data_root)
    except ValueError:
        relative_path = recording
    return {
        "subject": subject_for(recording, data_root),
        "task": task_label_for(recording),
        "source_relative_path": str(relative_path),
        "source_filename": recording.name,
        "source_size_bytes": stat.st_size,
        "source_modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "result_directory": str(output_directory),
    }


def write_batch_inventory(
    data_root: Path, outcomes: list[dict[str, Any]], results_root: Path,
) -> None:
    """Write the complete audit trail, including records that were not scored."""
    rows = sorted(outcomes, key=lambda item: (item["subject"], item["modality"], item["task"], item["source_relative_path"]))
    fields = [
        "subject", "task", "modality", "status", "session_status", "mean_quality_score",
        "source_relative_path", "source_filename", "source_size_bytes", "source_modified_at",
        "result_directory", "message",
    ]
    with (results_root / "recording_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    subjects = sorted(path.name for path in data_root.iterdir() if path.is_dir()) if data_root.exists() else []
    by_subject: dict[str, dict[str, Any]] = {}
    for subject in subjects:
        subject_rows = [row for row in rows if row["subject"] == subject]
        present_modalities = sorted({row["modality"] for row in subject_rows})
        by_subject[subject] = {
            "recording_count": len(subject_rows),
            "modalities_with_supported_files": present_modalities,
            "modalities_not_found": sorted({"EEG", "EMG", "FNIRS"} - set(present_modalities)),
            "task_counts": {
                task: sum(row["task"] == task for row in subject_rows)
                for task, _ in TASK_LABELS
            },
            "success_count": sum(row["status"] == "SUCCESS" for row in subject_rows),
            "failed_count": sum(row["status"] == "FAILED" for row in subject_rows),
            "skipped_count": sum(row["status"] == "SKIPPED" for row in subject_rows),
        }
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "subject_count": len(subjects),
        "recording_count": len(rows),
        "success_count": sum(row["status"] == "SUCCESS" for row in rows),
        "failed_count": sum(row["status"] == "FAILED" for row in rows),
        "skipped_count": sum(row["status"] == "SKIPPED" for row in rows),
        "subjects": by_subject,
        "records": rows,
        "interpretation_boundary": (
            "未发现的模态仅表示当前 data 目录下没有受支持的原始文件，不推断为采集遗漏；"
            "FAILED 表示程序无法完成离线质量评分，必须在使用前人工复核。"
        ),
    }
    (results_root / "data_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# 批次采集质量总览", "",
        f"- 受试者目录：{len(subjects)}；发现记录：{len(rows)}；成功评分：{payload['success_count']}；"
        f"无法评分：{payload['failed_count']}；跳过：{payload['skipped_count']}。",
        "- 每个原始文件都有独立结果目录；同一任务的重复导出不会相互覆盖。",
        "- ‘未发现模态’只描述当前目录，不推断采集遗漏。‘无法评分’需要人工核对原始文件和采集日志。", "",
        "| 受试者 | 记录数 | 已发现模态 | 成功 | 无法评分 | 跳过 |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for subject, item in by_subject.items():
        report.append(
            f"| {subject} | {item['recording_count']} | {', '.join(item['modalities_with_supported_files']) or '无'} | "
            f"{item['success_count']} | {item['failed_count']} | {item['skipped_count']} |"
        )
    failed = [row for row in rows if row["status"] == "FAILED"]
    if failed:
        report.extend(["", "## 无法评分的记录", ""])
        report.extend(f"- `{row['source_relative_path']}`：{row['message']}" for row in failed)
    (results_root / "batch_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 EEG、sEMG 与 fNIRS 记录的采集质量。")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="存放受试者文件夹的根目录。")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT, help="要评估的受试者文件夹名称；默认评估 data 下全部受试者。")
    parser.add_argument("--input", type=Path, help="指定单个 BDF/CSV/NIRS/HCX 文件或目录；指定后将忽略 --subject。")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="报告和图片的输出根目录。")
    parser.add_argument("--max-seconds", type=float, help="每个文件仅评估前 N 秒，用于快速检查。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有的对应结果目录。")
    args = parser.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive.")
    selected_input = args.input
    if selected_input is None and args.subject:
        selected_input = args.data_root / args.subject
    recordings = discover_recordings(args.data_root, selected_input)
    if not recordings:
        print(f"在 {selected_input or args.data_root} 下未找到支持的 EEG、sEMG 或 fNIRS 记录。", file=sys.stderr)
        return 2
    outcomes: list[dict[str, Any]] = []
    for recording in recordings:
        modality = recording_modality(recording)
        if modality is None:
            continue
        destination = output_path_for(recording, args.data_root, args.results_root)
        metadata = recording_metadata(recording, args.data_root, destination)
        common_outcome = {"modality": modality, **metadata}
        if destination.exists() and args.overwrite:
            shutil.rmtree(destination)
        if destination.exists() and any(destination.iterdir()):
            print(f"跳过 {recording}：结果目录 {destination} 已存在（使用 --overwrite 覆盖）。")
            outcomes.append({
                **common_outcome,
                "status": "SKIPPED",
                "session_status": None,
                "mean_quality_score": None,
                "message": "结果目录已存在；未使用 --overwrite。",
            })
            continue
        try:
            if modality == "EEG":
                summary = analyze_recording(recording, destination, args.max_seconds)
            elif modality == "EMG":
                summary = analyze_emg_recording(recording, destination, args.max_seconds)
            elif modality == "FNIRS":
                summary = analyze_fnirs_recording(recording, destination, args.max_seconds)
            else:
                raise ValueError(f"不支持的记录格式：{recording}")
            summary.update(metadata)
            (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            outcomes.append({
                **common_outcome,
                "status": "SUCCESS",
                "session_status": summary.get("session_status"),
                "mean_quality_score": summary.get("mean_quality_score"),
                "message": "",
            })
        except Exception as exc:  # Keep later recordings analyzable if one export is corrupt.
            print(f"处理 {recording} 时出错：{exc}", file=sys.stderr)
            outcomes.append({
                **common_outcome,
                "status": "FAILED",
                "session_status": "UNASSESSABLE",
                "mean_quality_score": None,
                "message": str(exc),
            })
    args.results_root.mkdir(parents=True, exist_ok=True)
    write_batch_inventory(args.data_root, outcomes, args.results_root)
    index = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "recording_count": len(outcomes),
        "success_count": sum(item["status"] == "SUCCESS" for item in outcomes),
        "failed_count": sum(item["status"] == "FAILED" for item in outcomes),
        "skipped_count": sum(item["status"] == "SKIPPED" for item in outcomes),
        "records": outcomes,
    }
    (args.results_root / "analysis_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"已处理 {len(outcomes)} 段记录：成功 {index['success_count']}，"
        f"无法评分 {index['failed_count']}，跳过 {index['skipped_count']}。结果目录：{args.results_root.resolve()}"
    )
    return 0 if index["success_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
