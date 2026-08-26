"""Conservative, auditable preprocessing for the recordings in ``data/``.

This module never changes the vendor exports.  It creates analysis-ready
arrays together with channel and time-window masks that make every rejection
and interpolation explicit.  It deliberately does not try to reconstruct a
flat EEG channel or to remove the known infrared eye-tracker contamination
from fNIRS recordings.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from emg_qc import EMG_BAND_HZ, EMG_EPOCH_SECONDS, read_emg_csv
from fnirs_qc import _fill_invalid, is_infrared_eye_tracking_task, read_fnirs
from main import (
    ACTIVE_EEG_CHANNELS,
    EPOCH_SECONDS,
    POSITION_MAP,
    discover_recordings,
    line_statistics,
    normalize_label,
    output_path_for,
    read_bdf,
    recording_metadata,
    recording_modality,
    task_label_for,
)


EEG_BAND_HZ = (0.5, 45.0)
EEG_TARGET_SAMPLE_RATE_HZ = 250.0
EEG_EPOCH_SECONDS = EPOCH_SECONDS
EEG_BAD_ARTIFACT_PERCENT = 20.0
EEG_BAD_FLAT_PERCENT = 10.0
EEG_ARTIFACT_PEAK_TO_PEAK_UV = 200.0
EEG_FLAT_PEAK_TO_PEAK_UV = 2.0
FNIRS_BAND_HZ = (0.01, 0.2)
FNIRS_BAD_SCI = 0.50
FNIRS_BAD_CV_PERCENT = 15.0
LINE_FREQUENCY_HZ = 50.0


def _as_list(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def _epoch_percent(values: np.ndarray, sample_rate: float, predicate: float, lower: bool) -> float:
    samples = int(round(EEG_EPOCH_SECONDS * sample_rate))
    count = values.size // samples
    if not count:
        return 100.0
    peak_to_peak = np.ptp(values[:count * samples].reshape(count, samples), axis=1)
    return float(np.mean(peak_to_peak < predicate if lower else peak_to_peak > predicate) * 100.0)


def eeg_bad_channels(raw_eeg: np.ndarray, labels: list[str], sample_rate: float) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Reject channels only for direct signal failures, not ambiguous impedances.

    Strong common 50 Hz is handled by filtering rather than being treated as a
    reason to remove every channel.  This keeps the decision tied to failures
    that cannot be repaired by a linear filter: prolonged flatline, clipping,
    or persistent high-amplitude transients.
    """
    bad = np.zeros(len(labels), dtype=bool)
    reasons: dict[str, list[str]] = {label: [] for label in labels}
    for index, label in enumerate(labels):
        artifact = _epoch_percent(raw_eeg[index], sample_rate, EEG_ARTIFACT_PEAK_TO_PEAK_UV, lower=False)
        flat = _epoch_percent(raw_eeg[index], sample_rate, EEG_FLAT_PEAK_TO_PEAK_UV, lower=True)
        if artifact >= EEG_BAD_ARTIFACT_PERCENT:
            bad[index] = True
            reasons[label].append(f"{artifact:.1f}% 2 秒窗峰峰值超过 {EEG_ARTIFACT_PEAK_TO_PEAK_UV:.0f} uV")
        if flat >= EEG_BAD_FLAT_PERCENT:
            bad[index] = True
            reasons[label].append(f"{flat:.1f}% 2 秒窗接近平线")
    return bad, {label: item for label, item in reasons.items() if item}


def _notch_harmonics(values: np.ndarray, sample_rate: float, upper_frequency: float) -> np.ndarray:
    result = signal.detrend(values, axis=-1, type="linear")
    # A 45-Hz EEG analysis band still needs a 50-Hz notch before the bandpass:
    # finite-order filters leave transition-band leakage in strongly contaminated
    # recordings.  EMG additionally receives its in-band harmonics.
    harmonic_limit = max(upper_frequency, LINE_FREQUENCY_HZ)
    for frequency in np.arange(LINE_FREQUENCY_HZ, harmonic_limit + 0.1, LINE_FREQUENCY_HZ):
        if frequency >= sample_rate * 0.45:
            break
        b, a = signal.iirnotch(frequency, Q=35, fs=sample_rate)
        result = signal.filtfilt(b, a, result, axis=-1)
    return result.astype(np.float32)


def _bandpass(values: np.ndarray, sample_rate: float, band: tuple[float, float]) -> np.ndarray:
    upper = min(band[1], sample_rate * 0.45)
    if upper <= band[0]:
        raise ValueError(f"采样率 {sample_rate:.3f} Hz 不支持 {band[0]}-{band[1]} Hz 带通。")
    sos = signal.butter(4, (band[0], upper), btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, values, axis=-1).astype(np.float32)


def _inverse_distance_interpolate(
    data: np.ndarray, labels: list[str], bad: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Interpolate only a minority of failed EEG channels from known positions."""
    result = data.copy()
    good_indices = [i for i, label in enumerate(labels) if not bad[i] and label in POSITION_MAP]
    interpolated: list[str] = []
    if len(good_indices) < 4:
        return result, interpolated
    for index, label in enumerate(labels):
        if not bad[index] or label not in POSITION_MAP:
            continue
        x, y = POSITION_MAP[label]
        distances = np.asarray([
            max(np.hypot(x - POSITION_MAP[labels[good]][0], y - POSITION_MAP[labels[good]][1]), 0.05)
            for good in good_indices
        ])
        nearest = np.argsort(distances)[:min(4, len(good_indices))]
        source_indices = [good_indices[item] for item in nearest]
        weights = 1.0 / distances[nearest] ** 2
        weights /= weights.sum()
        result[index] = np.tensordot(weights, result[source_indices], axes=(0, 0))
        interpolated.append(label)
    return result, interpolated


def eeg_valid_epoch_mask(data: np.ndarray, sample_rate: float) -> np.ndarray:
    samples = int(round(EEG_EPOCH_SECONDS * sample_rate))
    count = data.shape[1] // samples
    if not count:
        return np.zeros(0, dtype=bool)
    peak_to_peak = np.ptp(data[:, :count * samples].reshape(data.shape[0], count, samples), axis=2)
    # Bad epochs are retained in the file, but downstream analysis can use this
    # explicit mask instead of silently analyzing movement transients.
    return np.all(peak_to_peak <= EEG_ARTIFACT_PEAK_TO_PEAK_UV, axis=0)


def _resample(values: np.ndarray, source_rate: float, target_rate: float) -> tuple[np.ndarray, float]:
    if abs(source_rate - target_rate) < 1e-6:
        return values, source_rate
    ratio = Fraction(target_rate / source_rate).limit_denominator(10_000)
    return signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32), target_rate


def _line_amplitude_median(values: np.ndarray, sample_rate: float) -> float:
    if values.shape[1] < int(sample_rate * EEG_EPOCH_SECONDS):
        return float("nan")
    amplitudes, _ = line_statistics(values, sample_rate)
    return float(np.median(amplitudes))


def _save_npz(destination: Path, **arrays: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def _plot_eeg_preview(data: np.ndarray, labels: list[str], sample_rate: float, destination: Path) -> None:
    selected = [labels.index(label) for label in ("Fp1", "Fz", "Cz", "Pz", "O1", "O2") if label in labels]
    if not selected:
        selected = list(range(min(6, len(labels))))
    points = min(data.shape[1], int(sample_rate * 10))
    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    offsets = np.arange(len(selected))[::-1] * 80
    time = np.arange(points) / sample_rate
    for row, offset in zip(selected, offsets):
        axis.plot(time, data[row, :points] + offset, linewidth=0.55)
        axis.text(-0.1, offset, labels[row], ha="right", va="center", fontsize=8)
    axis.set(title="预处理后 EEG（10 秒，0.5-45 Hz）", xlabel="时间（s）", ylabel="uV + 偏移")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def preprocess_eeg(recording: Path, destination: Path) -> dict[str, Any]:
    header, physical, digital = read_bdf(recording, None)
    sample_rate = header.sample_rate
    label_to_index = {normalize_label(label): index for index, label in enumerate(header.labels)}
    missing = [label for label in ACTIVE_EEG_CHANNELS if normalize_label(label) not in label_to_index]
    if missing:
        raise ValueError("缺少配置 EEG 导联：" + ", ".join(missing))
    indices = [label_to_index[normalize_label(label)] for label in ACTIVE_EEG_CHANNELS]
    labels = list(ACTIVE_EEG_CHANNELS)
    raw = physical[indices].astype(np.float32)
    # Detect failures in the same order as QC: form a robust reference first,
    # then filter.  Filtering each raw channel before referencing can create a
    # separate detrending ramp in an otherwise flat channel and hide its failure.
    robust_referenced = raw - np.median(raw, axis=0, keepdims=True)
    residual_for_detection = _bandpass(
        _notch_harmonics(robust_referenced, sample_rate, EEG_BAND_HZ[1]), sample_rate, EEG_BAND_HZ,
    )
    bad, bad_reasons = eeg_bad_channels(residual_for_detection, labels, sample_rate)
    if np.count_nonzero(~bad) < 16:
        raise ValueError("直接信号失败的 EEG 导联过多，少于 16 个可用导联；不生成伪恢复数据。")
    pre_line = _line_amplitude_median(raw[~bad], sample_rate)
    # Build the analysis reference from directly usable electrodes only, then reconstruct only
    # local failed electrodes for display/topography.  Their names remain in
    # the interpolation mask and must not be treated as observed data.
    analysis_referenced = raw - np.mean(raw[~bad], axis=0, keepdims=True)
    filtered = _bandpass(
        _notch_harmonics(analysis_referenced, sample_rate, EEG_BAND_HZ[1]), sample_rate, EEG_BAND_HZ,
    )
    cleaned, interpolated = _inverse_distance_interpolate(filtered, labels, bad)
    cleaned, output_rate = _resample(cleaned, sample_rate, EEG_TARGET_SAMPLE_RATE_HZ)
    valid_epochs = eeg_valid_epoch_mask(cleaned[~bad], output_rate)
    post_line = _line_amplitude_median(cleaned[~bad], output_rate)
    destination.mkdir(parents=True, exist_ok=True)
    _save_npz(
        destination / "eeg_cleaned.npz",
        data_uv=cleaned,
        labels=np.asarray(labels, dtype="U8"),
        sample_rate_hz=np.asarray(output_rate),
        bad_channel_mask=bad,
        interpolated_channel_mask=np.asarray([label in interpolated for label in labels]),
        valid_2s_epoch_mask=valid_epochs,
    )
    _plot_eeg_preview(cleaned, labels, output_rate, destination / "signal_preview_cleaned.png")
    return {
        "modality": "EEG",
        "processing_status": "READY_WITH_MASKS",
        "source_sample_rate_hz": sample_rate,
        "output_sample_rate_hz": output_rate,
        "bandpass_hz": EEG_BAND_HZ,
        "notch_hz": [LINE_FREQUENCY_HZ],
        "bad_channels_direct_signal": [labels[i] for i in np.flatnonzero(bad)],
        "bad_channel_reasons": bad_reasons,
        "interpolated_channels": interpolated,
        "valid_2s_epochs": int(np.count_nonzero(valid_epochs)),
        "total_2s_epochs": int(valid_epochs.size),
        "line_noise_median_before_uv": round(pre_line, 3),
        "line_noise_median_after_uv": round(post_line, 3),
        "line_noise_reduction_percent": round(100 * (1 - post_line / max(pre_line, np.finfo(float).eps)), 2),
        "analysis_rule": "下游统计必须排除 bad_channel_mask 和 valid_2s_epoch_mask=False 的数据；插值通道仅可用于可视化或明确声明的空间操作。",
    }


def _robust_derivative_repair(optical_density: np.ndarray, sample_rate: float) -> tuple[np.ndarray, float]:
    """Repair abrupt fNIRS motion spikes while preserving slow hemodynamics."""
    derivative = np.diff(optical_density, prepend=optical_density[0])
    median = float(np.median(derivative))
    mad = float(np.median(np.abs(derivative - median)) * 1.4826)
    threshold = max(0.03, 8.0 * max(mad, np.finfo(float).eps))
    spike = np.abs(derivative - median) > threshold
    repaired = optical_density.copy()
    # Correct discontinuities by removing the detected derivative residual from
    # all subsequent samples.  This preserves within-segment dynamics instead
    # of smoothing the entire physiological time series.
    corrections = np.where(spike, derivative - median, 0.0)
    repaired -= np.cumsum(corrections)
    return repaired, float(np.mean(spike) * 100.0)


def _fnirs_channel_quality(channel: Any, sample_rate: float) -> tuple[bool, list[str]]:
    a = channel.intensity_a
    b = channel.intensity_b
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    reasons: list[str] = []
    if np.mean(valid) < 0.95:
        reasons.append("非正或缺失原始光强超过 5%")
        return False, reasons
    a = _fill_invalid(a)
    b = _fill_invalid(b)
    cv = max(np.std(a) / max(abs(np.mean(a)), np.finfo(float).eps), np.std(b) / max(abs(np.mean(b)), np.finfo(float).eps)) * 100
    band_high = min(2.5, sample_rate * 0.45)
    sos = signal.butter(3, (0.5, band_high), btype="bandpass", fs=sample_rate, output="sos")
    fa = signal.sosfiltfilt(sos, (a - a.mean()) / max(a.std(), np.finfo(float).eps))
    fb = signal.sosfiltfilt(sos, (b - b.mean()) / max(b.std(), np.finfo(float).eps))
    sci = float(np.corrcoef(fa, fb)[0, 1])
    if sci < FNIRS_BAD_SCI:
        reasons.append(f"SCI {sci:.2f} < {FNIRS_BAD_SCI:.2f}")
    if cv > FNIRS_BAD_CV_PERCENT:
        reasons.append(f"双波长最大 CV {cv:.1f}% > {FNIRS_BAD_CV_PERCENT:.1f}%")
    return not reasons, reasons


def preprocess_fnirs(recording: Path, destination: Path) -> dict[str, Any]:
    if is_infrared_eye_tracking_task(recording):
        destination.mkdir(parents=True, exist_ok=True)
        return {
            "modality": "FNIRS",
            "processing_status": "EXCLUDED_KNOWN_IR_INTERFERENCE",
            "analysis_rule": "眼动仪红外发射光与 fNIRS 原始光强混杂，当前记录不输出伪恢复光密度。",
        }
    source = read_fnirs(recording, None)
    kept: list[np.ndarray] = []
    labels: list[str] = []
    rejected: dict[str, list[str]] = {}
    spike_rates: dict[str, float] = {}
    for channel in source.channels:
        usable, reasons = _fnirs_channel_quality(channel, source.sample_rate_hz)
        if not usable:
            rejected[channel.label] = reasons
            continue
        wavelengths = []
        per_wavelength_spikes = []
        for values in (channel.intensity_a, channel.intensity_b):
            intensity = _fill_invalid(values)
            od = -np.log(intensity / np.median(intensity))
            repaired, spike_rate = _robust_derivative_repair(od, source.sample_rate_hz)
            wavelengths.append(_bandpass(repaired[None, :], source.sample_rate_hz, FNIRS_BAND_HZ)[0])
            per_wavelength_spikes.append(spike_rate)
        kept.append(np.stack(wavelengths))
        labels.append(channel.label)
        spike_rates[channel.label] = round(float(np.mean(per_wavelength_spikes)), 3)
    if len(kept) < 20:
        raise ValueError("可用 fNIRS 通道少于 20 个，无法形成可靠分析集。")
    data = np.stack(kept).astype(np.float32)
    destination.mkdir(parents=True, exist_ok=True)
    _save_npz(
        destination / "fnirs_cleaned_od.npz",
        optical_density=data,
        labels=np.asarray(labels, dtype="U20"),
        wavelengths_nm=np.asarray([source.channels[0].wavelength_a_nm, source.channels[0].wavelength_b_nm]),
        time_seconds=source.time - source.time[0],
        sample_rate_hz=np.asarray(source.sample_rate_hz),
    )
    return {
        "modality": "FNIRS",
        "processing_status": "READY_CHANNEL_FILTERED",
        "output": "双波长去噪光密度；未转换 HbO/HbR，避免在未确认 DPF、探头距离和消光系数设置时引入模型假设。",
        "bandpass_hz": FNIRS_BAND_HZ,
        "kept_channels": labels,
        "rejected_channels": rejected,
        "motion_spike_percent_after_detection": spike_rates,
        "analysis_rule": "下游分析仅使用 kept_channels；不对被剔除的物理通道作空间插值。",
    }


def preprocess_emg(recording: Path, destination: Path) -> dict[str, Any]:
    series = read_emg_csv(recording, None)
    labels = [item.label for item in series]
    target_rate = min(1000.0, min(item.sample_rate_hz for item in series))
    shortest = min(item.values.size for item in series)
    cleaned = []
    pre_line = []
    post_line = []
    for item in series:
        values = item.values[:shortest]
        before = signal.detrend(values)
        pre_line.append(_line_amplitude_median(before[None, :], item.sample_rate_hz))
        filtered = _bandpass(_notch_harmonics(values[None, :], item.sample_rate_hz, EMG_BAND_HZ[1]), item.sample_rate_hz, EMG_BAND_HZ)[0]
        filtered, output_rate = _resample(filtered[None, :], item.sample_rate_hz, target_rate)
        filtered = filtered[0]
        post_line.append(_line_amplitude_median(filtered[None, :], output_rate))
        cleaned.append(filtered)
    length = min(item.size for item in cleaned)
    data = np.stack([item[:length] for item in cleaned]).astype(np.float32)
    destination.mkdir(parents=True, exist_ok=True)
    _save_npz(
        destination / "emg_cleaned.npz",
        data_source_unit=data,
        labels=np.asarray(labels, dtype="U80"),
        sensor_ids=np.asarray([item.sensor_id for item in series]),
        sample_rate_hz=np.asarray(target_rate),
        time_seconds=np.arange(length) / target_rate,
    )
    return {
        "modality": "EMG",
        "processing_status": "READY_CONTINUOUS",
        "bandpass_hz": EMG_BAND_HZ,
        "notch_hz": list(np.arange(LINE_FREQUENCY_HZ, EMG_BAND_HZ[1] + 0.1, LINE_FREQUENCY_HZ)),
        "source_amplitude_unit": "未声明，保持源文件单位",
        "source_line_noise_median": round(float(np.nanmedian(pre_line)), 6),
        "output_line_noise_median": round(float(np.nanmedian(post_line)), 6),
        "analysis_rule": "CSV 未提供任务事件；输出仅为连续信号，任何分段必须用外部同步日志确定。",
    }


def _preprocess_output_path(recording: Path, data_root: Path, output_root: Path) -> Path:
    return output_path_for(recording, data_root, output_root)


def _write_index(output_root: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["subject", "task", "modality", "source_relative_path", "status", "output_directory", "message"]
    with (output_root / "preprocessing_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "preprocessing_index.json").write_text(json.dumps({
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "records": rows,
        "boundary": "READY 表示已生成保守预处理数组和掩码，不表示临床有效或可跳过人工波形复核。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成可追溯的 EEG、sEMG、fNIRS 预处理数据。")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--subject", default="", help="默认处理 data 下全部受试者；可指定一个目录名。")
    parser.add_argument("--input", type=Path, help="单个文件或目录；指定后忽略 --subject。")
    parser.add_argument("--output-root", type=Path, default=Path("preprocessed_20260822"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_path = args.input if args.input else (args.data_root / args.subject if args.subject else None)
    recordings = discover_recordings(args.data_root, input_path)
    if not recordings:
        parser.error("未发现可处理记录。")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for recording in recordings:
        modality = recording_modality(recording)
        if modality is None:
            continue
        destination = _preprocess_output_path(recording, args.data_root, args.output_root)
        metadata = recording_metadata(recording, args.data_root, destination)
        if destination.exists() and args.overwrite:
            shutil.rmtree(destination)
        if destination.exists() and any(destination.iterdir()):
            rows.append({**metadata, "modality": modality, "status": "SKIPPED", "output_directory": str(destination), "message": "目录已存在"})
            continue
        try:
            if modality == "EEG":
                summary = preprocess_eeg(recording, destination)
            elif modality == "EMG":
                summary = preprocess_emg(recording, destination)
            else:
                summary = preprocess_fnirs(recording, destination)
            summary.update(metadata)
            (destination / "preprocessing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({**metadata, "modality": modality, "status": summary["processing_status"], "output_directory": str(destination), "message": ""})
            print(f"{modality} {recording}: {summary['processing_status']}")
        except Exception as exc:
            rows.append({**metadata, "modality": modality, "status": "FAILED", "output_directory": str(destination), "message": str(exc)})
            print(f"{modality} {recording}: FAILED: {exc}")
    _write_index(args.output_root, rows)
    return 0 if any(row["status"].startswith("READY") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
