"""Experimental recovery layer for the low-SNR multimodal recordings.

This module is intentionally separate from :mod:`preprocess`.  The existing
pipeline is the auditable, conservative analysis stream.  The functions here
create *sensitivity* versions that can be compared against it without
overwriting the source files or silently treating reconstructed samples as
observations.

The design follows the actual failure modes found in this project:

* fNIRS: isolated derivative spikes, slow drift and common-mode physiology;
* sEMG: non-stationary 50-Hz harmonics and impulsive samples;
* EEG: large transient subspace excursions and line interference.

The output of the command is a table of before/after diagnostics and event
features, not a claim that any signal has been clinically recovered.  Every
experimental feature is labelled with the method used and can be rejected if
the paired sensitivity analysis is not stable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal

from emg_qc import EXPECTED_SENSOR_IDS, assess_emg_channel, read_emg_csv, resolve_emg_mapping
from fnirs_qc import _fill_invalid, is_infrared_eye_tracking_task, read_fnirs
from main import (
    FULL_EEG_CHANNELS,
    eeg_configuration_for_recording,
    read_bdf,
)
import expand_all_cases as expanded
from explore_202608_cases import infer_movement_event_labels, read_fnirs_events


LINE_HZ = 50.0


def mad(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(values - center), axis=axis, keepdims=keepdims) * 1.4826


def robust_z(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values, axis=axis, keepdims=True)
    scale = np.maximum(mad(values, axis=axis, keepdims=True), 1e-9)
    return (values - center) / scale


def hampel_filter(values: np.ndarray, window: int = 11, n_sigma: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated outliers using a rolling median and MAD.

    Unlike a global z-score this keeps a genuine slow hemodynamic change and
    only changes samples that disagree with their local neighbourhood.
    """
    x = np.asarray(values, dtype=float).copy()
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False
    half = max(1, int(window) // 2)
    mask = np.zeros_like(x, dtype=bool)
    for row in range(x.shape[0]):
        series = x[row]
        padded = np.pad(series, (half, half), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
        local_median = np.nanmedian(windows, axis=1)
        local_mad = np.nanmedian(np.abs(windows - local_median[:, None]), axis=1) * 1.4826
        threshold = np.maximum(n_sigma * local_mad, 1e-10)
        bad = ~np.isfinite(series) | (np.abs(series - local_median) > threshold)
        series[bad] = local_median[bad]
        mask[row] = bad
    return (x[0] if squeeze else x), (mask[0] if squeeze else mask)


def derivative_repair(values: np.ndarray, n_sigma: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """TDDR-like correction using a robust derivative threshold.

    The correction removes a detected step from all subsequent samples.  This
    avoids the ringing introduced by replacing a discontinuity with a long
    moving average.
    """
    x = np.asarray(values, dtype=float).copy()
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False
    repaired = x.copy()
    spike = np.zeros_like(x, dtype=bool)
    for row in range(x.shape[0]):
        derivative = np.diff(x[row], prepend=x[row, 0])
        center = np.nanmedian(derivative)
        scale = max(float(mad(derivative)), 1e-10)
        bad = np.abs(derivative - center) > n_sigma * scale
        repaired[row] -= np.cumsum(np.where(bad, derivative - center, 0.0))
        spike[row] = bad
    return (repaired[0] if squeeze else repaired), (spike[0] if squeeze else spike)


def kalman_smooth(values: np.ndarray, innovation_sigma: float = 6.0) -> np.ndarray:
    """Robust local-level Kalman smoother for slow physiological signals."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False
    out = np.empty_like(x)
    for row, series in enumerate(x):
        if series.size < 4:
            out[row] = series
            continue
        diffs = np.diff(series)
        process = max(float(mad(diffs)) ** 2, 1e-10)
        measurement = max(float(mad(series - np.nanmedian(series))) ** 2 * 0.01, process * 0.1, 1e-10)
        level = float(series[0])
        variance = measurement
        forward = np.empty(series.size, dtype=float)
        variances = np.empty(series.size, dtype=float)
        forward[0] = level
        variances[0] = variance
        for index in range(1, series.size):
            prediction = level
            prediction_variance = variance + process
            innovation = float(series[index] - prediction)
            limit = innovation_sigma * np.sqrt(prediction_variance + measurement)
            if abs(innovation) > limit:
                innovation = np.sign(innovation) * limit
            gain = prediction_variance / (prediction_variance + measurement)
            level = prediction + gain * innovation
            variance = (1.0 - gain) * prediction_variance
            forward[index] = level
            variances[index] = variance
        # A short RTS-style backward pass makes the output continuous at
        # isolated repaired points while retaining the causal estimate above.
        smooth = forward.copy()
        for index in range(series.size - 2, -1, -1):
            gain = variances[index] / max(variances[index] + process, 1e-10)
            smooth[index] += gain * (smooth[index + 1] - forward[index])
        out[row] = smooth
    return out[0] if squeeze else out


def harmonic_cancel(values: np.ndarray, sample_rate: float, upper: float = 450.0, block_seconds: float = 20.0) -> np.ndarray:
    """Remove non-stationary line harmonics by blockwise least squares.

    A fixed notch can distort nearby EMG content.  Fitting sine/cosine pairs in
    short blocks lets amplitude and phase drift while leaving the rest of the
    spectrum untouched.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False
    result = x.copy()
    block = max(int(round(block_seconds * sample_rate)), 32)
    frequencies = np.arange(LINE_HZ, min(upper, sample_rate * 0.45) + 0.1, LINE_HZ)
    if not frequencies.size:
        return result[0] if squeeze else result
    for start in range(0, x.shape[1], block):
        stop = min(x.shape[1], start + block)
        if stop - start < 16:
            continue
        time = np.arange(stop - start, dtype=float) / sample_rate
        design = np.column_stack([v for f in frequencies for v in (np.sin(2 * np.pi * f * time), np.cos(2 * np.pi * f * time))])
        design = np.column_stack((np.ones(time.size), time, design))
        segment = x[:, start:stop]
        coefficients, *_ = np.linalg.lstsq(design, segment.T, rcond=None)
        baseline_fit = (np.column_stack((np.ones(time.size), time)) @ coefficients[:2]).T
        result[:, start:stop] -= (design @ coefficients).T - baseline_fit
    return result[0] if squeeze else result


def safe_bandpass(values: np.ndarray, sample_rate: float, low: float, high: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    upper = min(high, sample_rate * 0.45)
    if upper <= low or x.shape[-1] < 32:
        return x.copy()
    sos = signal.butter(3, (low, upper), btype="bandpass", fs=sample_rate, output="sos")
    try:
        return signal.sosfiltfilt(sos, x, axis=-1)
    except ValueError:
        return x.copy()


def _od_from_intensity(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.where(values > 0, values, np.nan)
    baseline = np.nanmedian(valid, axis=0, keepdims=True)
    output = -np.log(np.maximum(valid, 1e-12) / np.maximum(baseline, 1e-12))
    for column in range(output.shape[1]):
        finite = np.isfinite(output[:, column])
        if finite.sum() >= 2:
            output[:, column] = np.interp(np.arange(output.shape[0]), np.flatnonzero(finite), output[finite, column])
        else:
            output[:, column] = 0.0
    return output


def fnirs_proxy(path: Path, variant: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    recording = read_fnirs(path, None)
    # A few vendor exports use (1, n) row vectors for a channel while others
    # use (n,) vectors.  Flatten at the boundary so every downstream array is
    # unambiguously [time, physical_channel].
    a = _od_from_intensity(np.column_stack([np.ravel(c.intensity_a) for c in recording.channels]))
    b = _od_from_intensity(np.column_stack([np.ravel(c.intensity_b) for c in recording.channels]))
    diagnostics: dict[str, float] = {}
    if variant == "robust":
        a, spikes_a = hampel_filter(a.T, window=9, n_sigma=6.0)
        b, spikes_b = hampel_filter(b.T, window=9, n_sigma=6.0)
        a, steps_a = derivative_repair(a)
        b, steps_b = derivative_repair(b)
        a = kalman_smooth(a)
        b = kalman_smooth(b)
        diagnostics["hampel_percent"] = float(100 * np.mean(np.r_[spikes_a, spikes_b]))
        diagnostics["derivative_step_percent"] = float(100 * np.mean(np.r_[steps_a, steps_b]))
    else:
        diagnostics["hampel_percent"] = 0.0
        diagnostics["derivative_step_percent"] = 0.0
    # Wavelength balancing reduces channel-specific gain differences.  No DPF
    # or extinction coefficients are assumed, so this remains a relative proxy.
    # The robust branch works in [channel, time] orientation; the standard
    # branch remains [time, channel].  Normalize the orientation explicitly at
    # this boundary so event windows always index the time axis.
    a_ct, b_ct = (a, b) if variant == "robust" else (a.T, b.T)
    proxy = (robust_z(b_ct, axis=1) - robust_z(a_ct, axis=1)) / np.sqrt(2.0)
    proxy = safe_bandpass(proxy, recording.sample_rate_hz, 0.01, 0.2)
    return proxy, recording.time - recording.time[0], diagnostics


def network_metric(proxy: np.ndarray) -> float:
    if proxy.shape[0] < 4:
        return float("nan")
    corr = np.clip(np.corrcoef(proxy), -0.999, 0.999)
    z = np.arctanh(corr)
    values = np.abs(z[np.triu_indices_from(z, 1)])
    return float(np.nanmean(values))


def event_values(proxy: np.ndarray, time: np.ndarray, event_times: Iterable[float]) -> np.ndarray:
    values: list[float] = []
    for onset in event_times:
        pre = (time >= onset - 5.0) & (time < onset)
        post = (time >= onset + 2.0) & (time < onset + 12.0)
        if pre.sum() < 5 or post.sum() < 5:
            continue
        values.append(float(np.nanmean(proxy[:, post]) - np.nanmean(proxy[:, pre])))
    return np.asarray(values, dtype=float)


def summarize_vector(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan, f"{prefix}_mad": np.nan, f"{prefix}_n": 0}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mad": float(mad(values)),
        f"{prefix}_n": int(values.size),
    }


def emg_summary(path: Path, max_seconds: float = 120.0) -> dict[str, float]:
    # Pass the cap into the parser as well as slicing below; otherwise
    # genfromtxt would still load the complete multi-million-row export.
    raw_series = read_emg_csv(path, max_seconds)
    mapping = resolve_emg_mapping([item.sensor_id for item in raw_series], path)
    # Sensitivity summaries use the same logical-mapping gate as the primary
    # analysis. Unknown shifted-looking files remain QC-only.
    if not mapping["mapping_valid_for_primary_features"]:
        return {
            "emg_channels": 0.0,
            "emg_channels_observed": float(len(raw_series)),
            "emg_quality_eligible_channels": 0.0,
            "emg_quality_excluded_channels": float(len(raw_series)),
            "emg_mapping_valid_for_primary_features": 0.0,
            "emg_line_ratio_before": float("nan"),
            "emg_line_ratio_after": float("nan"),
            "emg_line_reduction_percent": float("nan"),
            "emg_rms_ratio_after_before": float("nan"),
        }
    allowed = {int(value) for value in mapping["physical_to_logical"]}
    series = [item for item in raw_series if item.sensor_id in allowed and item.muscle_index is not None]
    # Sensitivity diagnostics must consume the same channel-level QC gate as
    # primary features.  In particular, a short recording or a failed QC
    # assessment is not re-admitted by a small-channel fallback.
    quality_status: list[str] = []
    eligible: list = []
    for item in series:
        try:
            status = str(assess_emg_channel(item).status)
        except Exception:
            status = "UNKNOWN"
        quality_status.append(status)
        if status in {"GOOD", "CAUTION"}:
            eligible.append(item)
    rows: dict[str, float] = {
        "emg_channels": float(len(eligible)),
        "emg_channels_observed": float(len(series)),
        "emg_quality_eligible_channels": float(len(eligible)),
        "emg_quality_excluded_channels": float(len(series) - len(eligible)),
        "emg_quality_poor_channels": float(sum(status == "POOR" for status in quality_status)),
        "emg_quality_unknown_channels": float(sum(status == "UNKNOWN" for status in quality_status)),
        "emg_mapping_valid_for_primary_features": 1.0,
    }
    if not eligible:
        rows.update({
            "emg_line_ratio_before": float("nan"),
            "emg_line_ratio_after": float("nan"),
            "emg_line_reduction_percent": float("nan"),
            "emg_rms_ratio_after_before": float("nan"),
        })
        return rows
    raw_lines: list[float] = []
    clean_lines: list[float] = []
    raw_rms: list[float] = []
    clean_rms: list[float] = []
    for item in eligible:
        # Line-noise statistics are stationary enough to estimate on a fixed
        # two-minute prefix.  Capping the diagnostic window keeps the
        # exploratory pass tractable for multi-million-sample CSV exports;
        # downstream event features still use the original continuous file.
        limit = min(item.values.size, int(round(max_seconds * item.sample_rate_hz)))
        values = np.asarray(item.values[:limit], dtype=float)
        raw = values - np.nanmedian(values)
        cleaned = harmonic_cancel(raw, item.sample_rate_hz, upper=450.0)
        cleaned, _ = hampel_filter(cleaned, window=7, n_sigma=8.0)
        def line_ratio(x: np.ndarray) -> float:
            if x.size < 256:
                return float("nan")
            freqs, power = signal.welch(x, fs=item.sample_rate_hz, nperseg=min(4096, x.size))
            line = np.zeros_like(freqs, dtype=bool)
            for frequency in np.arange(LINE_HZ, min(450.0, item.sample_rate_hz * 0.45) + 0.1, LINE_HZ):
                line |= np.abs(freqs - frequency) <= 1.0
            band = (freqs >= 20.0) & (freqs <= min(450.0, item.sample_rate_hz * 0.45))
            return float(100 * np.sum(power[line & band]) / max(np.sum(power[band]), 1e-12))
        raw_lines.append(line_ratio(raw))
        clean_lines.append(line_ratio(cleaned))
        raw_rms.append(float(np.sqrt(np.mean(raw ** 2))))
        clean_rms.append(float(np.sqrt(np.mean(cleaned ** 2))))
    rows["emg_line_ratio_before"] = float(np.nanmedian(raw_lines))
    rows["emg_line_ratio_after"] = float(np.nanmedian(clean_lines))
    rows["emg_line_reduction_percent"] = float(100 * (1 - rows["emg_line_ratio_after"] / max(rows["emg_line_ratio_before"], 1e-9)))
    rows["emg_rms_ratio_after_before"] = float(np.nanmedian(clean_rms) / max(np.nanmedian(raw_rms), 1e-9))
    return rows


def asr_like(data: np.ndarray, sample_rate: float, cutoff: float = 5.0, window_seconds: float = 2.0) -> tuple[np.ndarray, float]:
    """A transparent ASR-like subspace shrinker for exploratory EEG only.

    Calibration covariance is estimated from the first 30 s.  Components with
    window variance above a robust calibration threshold are attenuated, not
    deleted.  This is deliberately *not* presented as EEGLAB ASR and should be
    compared with manual/ICA cleaning before use in a manuscript.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 64:
        return x.copy(), 0.0
    x = x - np.nanmedian(x, axis=1, keepdims=True)
    calibration = x[:, : min(x.shape[1], int(round(30 * sample_rate)))]
    scale = np.maximum(mad(calibration, axis=1, keepdims=True), 1e-6)
    normalized = calibration / scale
    covariance = normalized @ normalized.T / max(normalized.shape[1] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    calibration_projection = eigenvectors.T @ normalized
    baseline_variance = np.var(calibration_projection, axis=1)
    threshold = np.median(baseline_variance) + cutoff * max(float(mad(baseline_variance)), 1e-6)
    output = x.copy()
    window = max(32, int(round(window_seconds * sample_rate)))
    affected = 0
    total = 0
    for start in range(0, x.shape[1], window):
        stop = min(x.shape[1], start + window)
        if stop - start < 16:
            continue
        projected = eigenvectors.T @ (x[:, start:stop] / scale)
        variance = np.var(projected, axis=1)
        gain = np.minimum(1.0, np.sqrt(threshold / np.maximum(variance, 1e-9)))
        affected += int(np.count_nonzero(gain < 0.99))
        total += gain.size
        output[:, start:stop] = (eigenvectors @ (projected * gain[:, None])) * scale
    return output, float(100 * affected / max(total, 1))


def eeg_summary(path: Path) -> dict[str, float]:
    header, physical, _ = read_bdf(path, None)
    labels = [str(v) for v in header.labels]
    configured, _ = eeg_configuration_for_recording(path)
    wanted = {str(v).strip().lower() for v in configured or FULL_EEG_CHANNELS}
    indices = [i for i, label in enumerate(labels) if label.strip().lower() in wanted]
    if len(indices) < 8:
        return {"eeg_channels": float(len(indices)), "eeg_status": 0.0}
    raw = physical[indices].astype(float)
    referenced = raw - np.median(raw, axis=0, keepdims=True)
    before = harmonic_cancel(referenced, header.sample_rate, upper=45.0)
    before = safe_bandpass(before, header.sample_rate, 0.5, 45.0)
    cleaned, affected = asr_like(before, header.sample_rate)
    def line_power(x: np.ndarray) -> float:
        freqs, power = signal.welch(x, fs=header.sample_rate, nperseg=min(int(header.sample_rate * 4), x.shape[1]), axis=-1)
        index = int(np.argmin(np.abs(freqs - LINE_HZ)))
        band = (freqs >= 1) & (freqs <= 45)
        return float(np.median(power[:, index]) / max(np.median(np.sum(power[:, band], axis=1)), 1e-9))
    before_p2p = np.ptp(before, axis=0)
    after_p2p = np.ptp(cleaned, axis=0)
    return {
        "eeg_channels": float(len(indices)),
        "eeg_line_ratio_before": line_power(referenced),
        "eeg_line_ratio_after": line_power(cleaned),
        "eeg_line_reduction_percent": float(100 * (1 - line_power(cleaned) / max(line_power(referenced), 1e-9))),
        "eeg_high_amp_epoch_percent_before": float(100 * np.mean(before_p2p > 200)),
        "eeg_high_amp_epoch_percent_after": float(100 * np.mean(after_p2p > 200)),
        "eeg_asr_component_percent": affected,
        "eeg_variance_ratio_after_before": float(np.nanmedian(np.var(cleaned, axis=1)) / max(np.nanmedian(np.var(before, axis=1)), 1e-9)),
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, include_eeg: bool = False, data_root: Path | None = None) -> None:
    if data_root is not None:
        expanded.set_data_root(data_root)
    identity = expanded.load_identity(expanded.IDENTITY_XLSX)
    discovered_sessions = expanded.discover_sessions(identity)
    sessions, excluded_sessions = expanded.canonicalize_sessions(discovered_sessions)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for subject, name in identity.items():
        for session in sessions.get(subject, []):
            path = expanded.robust_find_recording(subject, session, "FNIRS", "运动")
            # This inventory is scoped to movement-task fNIRS.  Keep a
            # defensive eye-task guard in case path discovery returns a
            # misfiled recording; the eye branch is handled by the main
            # masked pipeline and is not silently relabelled as movement.
            if path is None or str(path).lower() in seen or is_infrared_eye_tracking_task(path):
                continue
            seen.add(str(path).lower())
            events = read_fnirs_events(path)
            event_times = (events.times - events.times[0]).tolist() if events is not None and events.times.size else []
            labels = infer_movement_event_labels(np.asarray(event_times), subject) if event_times else []
            standard, time, standard_diag = fnirs_proxy(path, "standard")
            robust, _, robust_diag = fnirs_proxy(path, "robust")
            row: dict[str, object] = {
                "subject": subject,
                "name": name,
                "session": session,
                "source": str(path),
                "subtask_labels_observed_or_inferred": ";".join(str(item.get("task_name", "")) for item in labels),
                "network_standard": network_metric(standard),
                "network_robust": network_metric(robust),
                "network_delta_robust_minus_standard": network_metric(robust) - network_metric(standard),
                **standard_diag,
                **{f"robust_{key}": value for key, value in robust_diag.items()},
            }
            row.update(summarize_vector(event_values(standard, time, event_times), "event_standard"))
            row.update(summarize_vector(event_values(robust, time, event_times), "event_robust"))
            row["event_mad_reduction_percent"] = float(100 * (1 - row["event_robust_mad"] / max(row["event_standard_mad"], 1e-9)))
            emg = expanded.robust_find_recording(subject, session, "EMG", "")
            if emg is not None:
                try:
                    row.update(emg_summary(emg))
                except Exception as exc:
                    row["emg_error"] = str(exc)
            if include_eeg:
                eeg = expanded.robust_find_recording(subject, session, "EEG", "运动")
                if eeg is not None:
                    try:
                        row.update(eeg_summary(eeg))
                    except Exception as exc:
                        row["eeg_error"] = str(exc)
            rows.append(row)
    _write_rows(output / "low_snr_recovery_inventory.csv", rows)
    metadata = {
        "raw_root": str(expanded.RAW_ROOT),
        "identity_source": str(expanded.IDENTITY_XLSX),
        "excluded_duplicate_sessions": excluded_sessions,
        "method_status": "sensitivity_analysis_only",
        "methods": {
            "fNIRS": ["local Hampel", "derivative step repair (TDDR-like)", "robust local-level Kalman smoothing", "wavelength-balanced OD proxy"],
            "EMG": ["blockwise harmonic regression", "local Hampel"],
            "EEG": ["blockwise harmonic regression", "transparent ASR-like PCA subspace shrinkage"],
        },
        "limitations": [
            "No short-separation fNIRS channel is present in the inspected montage; global-component regression is therefore a sensitivity analysis, not a validated superficial-noise correction.",
            "This inventory intentionally covers movement-task fNIRS only; eye-tracker fNIRS is retained by the main pipeline as EYE_IR_DEGRADED_MASKED with sample/channel masks and is not relabelled as a movement recovery result.",
            "EEG ASR-like output is not a substitute for calibrated EEGLAB ASR/ICA and must not be used as primary evidence without visual/component review.",
            "Event features use the supplied marker protocol; inferred labels remain tagged in the source event table.",
        ],
    }
    (output / "low_snr_recovery_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"records={len(rows)} output={output.resolve()}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="低信噪比 EEG/fNIRS/EMG 实验性恢复与敏感性分析")
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments/low_snr_20260901"))
    parser.add_argument("--include-eeg", action="store_true", help="额外运行 EEG ASR-like 指标，耗时较长")
    parser.add_argument("--data-root", type=Path, default=None, help="raw-data root (default: historical J: path)")
    args = parser.parse_args()
    run(args.out, include_eeg=args.include_eeg, data_root=args.data_root)


if __name__ == "__main__":
    main_cli()
