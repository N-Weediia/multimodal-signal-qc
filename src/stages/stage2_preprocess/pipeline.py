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
import hashlib
import json
import math
import re
import shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, linalg, signal

from emg_qc import (
    EMG_BAND_HZ,
    EMG_EPOCH_SECONDS,
    EXPECTED_SENSOR_IDS,
    SENSOR_TO_MUSCLE_INDEX,
    _band_power,
    assess_emg_channel,
    emg_window_quality,
    read_emg_csv,
    resolve_emg_mapping,
)
from fnirs_qc import (
    _fill_invalid,
    _scalp_coupling_index,
    build_fnirs_sample_masks,
    detect_fnirs_uncoupled_boundary_mask,
    is_infrared_eye_tracking_task,
    motion_artifact_level,
    read_fnirs,
)
from main import (
    DEFAULT_DATA_ROOT,
    EPOCH_SECONDS,
    POSITION_MAP,
    assert_writable_output_path,
    clamp_eeg_disconnect_epochs,
    configure_matplotlib_font,
    discover_recordings,
    eeg_configuration_for_recording,
    line_statistics,
    load_impedances,
    load_subject_identity_map,
    normalize_label,
    output_path_for,
    read_bdf,
    read_bdf_task_events,
    recording_metadata,
    recording_modality,
    session_for,
    task_label_for,
)


EEG_BAND_HZ = (0.5, 45.0)
EEG_TARGET_SAMPLE_RATE_HZ = 250.0
EEG_EPOCH_SECONDS = EPOCH_SECONDS
EEG_MIN_DURATION_SECONDS = 10.0
EEG_BAD_ARTIFACT_PERCENT = 50.0
EEG_BAD_FLAT_PERCENT = 10.0
EEG_ARTIFACT_PEAK_TO_PEAK_UV = 200.0
EEG_FLAT_PEAK_TO_PEAK_UV = 2.0
EEG_BRIDGE_ED_UV2 = 0.05
EEG_BRIDGE_REL_ED = 0.005
FNIRS_BAND_HZ = (0.01, 0.2)
FNIRS_MIN_DURATION_SECONDS = 30.0
FNIRS_BAD_SCI = 0.50
FNIRS_BAD_CV_PERCENT = 15.0
EMG_PROTOCOL_COMPLETE_SECONDS = 120.0
EMG_PROTOCOL_DEGRADED_SECONDS = 60.0
LINE_FREQUENCY_HZ = 50.0
PREPROCESS_PIPELINE_VERSION = "2026-11-09.v8-quality-tiered-recovery"


class PreprocessExclusionError(ValueError):
    """Structured conservative exclusion (short fragment or unrecoverable hardware failure)."""

    def __init__(self, status: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.details = details or {}


def _preprocess_parameter_hash() -> str:
    payload = {
        "pipeline_version": PREPROCESS_PIPELINE_VERSION,
        "eeg_band_hz": EEG_BAND_HZ,
        "eeg_target_sample_rate_hz": EEG_TARGET_SAMPLE_RATE_HZ,
        "eeg_min_duration_seconds": EEG_MIN_DURATION_SECONDS,
        "fnirs_band_hz": FNIRS_BAND_HZ,
        "fnirs_min_duration_seconds": FNIRS_MIN_DURATION_SECONDS,
        "emg_band_hz": EMG_BAND_HZ,
        "emg_expected_sensor_ids": EXPECTED_SENSOR_IDS,
        "emg_sensor_to_muscle_index": {str(k): v for k, v in SENSOR_TO_MUSCLE_INDEX.items()},
        "emg_mapping_profiles": {
            "standard_v1": {str(k): v for k, v in SENSOR_TO_MUSCLE_INDEX.items()},
            "shifted_v1_2026080901_base_line": {str(k): v for k, v in zip((1, 2, 3, 4, 5, 9, 10), range(1, 8))},
        },
        "line_frequency_hz": LINE_FREQUENCY_HZ,
        "missing_policy": "short_gap_only_long_gap_nan",
        "eeg_quality_policy": "disconnect_clamped_reref_bridge_dedup_trimmed_ref_v8",
        "fnirs_motion_policy": "windowed_admission_fishburn_tddr_v8",
        "emg_quality_policy": "hann_wola_harmonic_cancel_burst_gate_v8",
        "marker_alignment_policy": "ransac_piecewise_eeg_companion_v8",
        "provenance_policy": "sha256_cross_subject_and_task_quarantine_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preprocess_cache_is_current(destination: Path, recording: Path, source_hash: str | None = None) -> bool:
    summary_path = destination / "preprocessing_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        summary.get("source_sha256") == (source_hash or _sha256_file(recording))
        and summary.get("preprocess_pipeline_version") == PREPROCESS_PIPELINE_VERSION
        and summary.get("preprocess_parameter_hash") == _preprocess_parameter_hash()
    )


def _cached_status(destination: Path) -> str:
    """Expose whether a cache hit was a ready or intentionally excluded file."""
    try:
        summary = json.loads((destination / "preprocessing_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return "CACHED"
    status = str(summary.get("processing_status", ""))
    return "CACHED_EXCLUDED" if status.startswith("EXCLUDED") else "CACHED_READY"


def _cached_row_from_summary(destination: Path, metadata: dict[str, Any], modality: str) -> dict[str, Any]:
    try:
        summary = json.loads((destination / "preprocessing_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        summary = {}
    return {
        **metadata,
        "modality": modality,
        "status": _cached_status(destination),
        "known_ir_interference": summary.get("known_ir_interference"),
        "eye_tracker_occlusion_status": (summary.get("eye_tracker_occlusion") or {}).get("status"),
        "analysis_branch": summary.get("analysis_branch"),
        "quality_degradation": summary.get("quality_degradation"),
        "is_primary_for_task": summary.get("is_primary_for_task"),
        "recording_role": summary.get("recording_role"),
        "output_directory": str(destination),
        "message": summary.get("exclusion_reason") or "源文件哈希、流程版本和参数哈希均一致",
    }


def _as_list(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def _epoch_percent(
    values: np.ndarray, sample_rate: float, predicate: float, lower: bool, valid_epoch_mask: np.ndarray | None = None,
) -> float:
    samples = int(round(EEG_EPOCH_SECONDS * sample_rate))
    count = values.size // samples
    if not count:
        return 100.0
    peak_to_peak = np.ptp(values[: count * samples].reshape(count, samples), axis=1)
    if valid_epoch_mask is not None and valid_epoch_mask.size == count and np.any(valid_epoch_mask):
        peak_to_peak = peak_to_peak[valid_epoch_mask]
    return float(np.mean(peak_to_peak < predicate if lower else peak_to_peak > predicate) * 100.0)


def eeg_bad_channels(
    raw_eeg: np.ndarray,
    labels: list[str],
    sample_rate: float,
    *,
    digital: np.ndarray | None = None,
    digital_min: np.ndarray | None = None,
    digital_max: np.ndarray | None = None,
    impedances: dict[str, float] | None = None,
    valid_epoch_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Reject channels for direct signal failures before building the scalp reference."""
    bad = np.zeros(len(labels), dtype=bool)
    reasons: dict[str, list[str]] = {label: [] for label in labels}
    ep_samples = max(1, int(round(EEG_EPOCH_SECONDS * sample_rate)))
    ep_count = raw_eeg.shape[1] // ep_samples
    use_ep = (
        valid_epoch_mask
        if valid_epoch_mask is not None and valid_epoch_mask.size == ep_count and np.any(valid_epoch_mask)
        else np.ones(ep_count, dtype=bool)
    )
    rob_stds = np.zeros(len(labels), dtype=float)
    med_p2ps = np.zeros(len(labels), dtype=float)
    for index, label in enumerate(labels):
        artifact = _epoch_percent(raw_eeg[index], sample_rate, EEG_ARTIFACT_PEAK_TO_PEAK_UV, lower=False, valid_epoch_mask=use_ep)
        flat = _epoch_percent(raw_eeg[index], sample_rate, EEG_FLAT_PEAK_TO_PEAK_UV, lower=True, valid_epoch_mask=use_ep)
        sig = raw_eeg[index]
        if ep_count:
            ep_view = sig[: ep_count * ep_samples].reshape(ep_count, ep_samples)[use_ep]
            med_p2ps[index] = float(np.median(np.ptp(ep_view, axis=1)))
            eval_samples = ep_view.reshape(-1)
        else:
            eval_samples = sig
        rob_stds[index] = float(np.median(np.abs(eval_samples - np.median(eval_samples))) * 1.4826) if eval_samples.size else 0.0
        if artifact >= EEG_BAD_ARTIFACT_PERCENT:
            bad[index] = True
            reasons[label].append(f"{artifact:.1f}% 2 秒窗持续超过 {EEG_ARTIFACT_PEAK_TO_PEAK_UV:.0f} uV")
        if flat >= EEG_BAD_FLAT_PERCENT:
            bad[index] = True
            reasons[label].append(f"{flat:.1f}% 2 秒窗接近平线")
        if digital is not None and digital_min is not None and digital_max is not None:
            dig = digital[index]
            rail = (dig <= digital_min[index]) | (dig >= digital_max[index])
            if ep_count:
                rail_ep = rail[: ep_count * ep_samples].reshape(ep_count, ep_samples)[use_ep]
                clip_sample_pct = float(np.mean(rail_ep) * 100.0)
                clip_ep_pct = float(np.mean(np.any(rail_ep, axis=1)) * 100.0)
            else:
                clip_sample_pct = float(np.mean(rail) * 100.0)
                clip_ep_pct = clip_sample_pct
            if clip_sample_pct >= 0.5 or clip_ep_pct >= 5.0:
                bad[index] = True
                reasons[label].append(f"数字量程饱和（{clip_sample_pct:.2f}% 样本 / {clip_ep_pct:.1f}% 2 秒窗）")
        imp = impedances.get(normalize_label(label)) if impedances else None
        if imp is not None and imp >= 500.0 and (artifact >= 20.0 or rob_stds[index] > 45.0):
            bad[index] = True
            reasons[label].append(f"开路级高阻抗 {imp:.0f} kOhm 伴随持续伪迹")
    usable_pool = ~bad
    if int(np.count_nonzero(usable_pool)) >= 6:
        cap_rob_std = max(float(np.median(rob_stds[usable_pool])), 1.0)
        cap_p2p = max(float(np.median(med_p2ps[usable_pool])), 5.0)
        for index, label in enumerate(labels):
            if bad[index]:
                continue
            if rob_stds[index] > max(80.0, 6.0 * cap_rob_std) or med_p2ps[index] > max(250.0, 8.0 * cap_p2p):
                bad[index] = True
                reasons[label].append(
                    f"稳健幅值异常偏离全帽基线（MAD-SD {rob_stds[index]:.1f} vs 全帽中位 {cap_rob_std:.1f} uV）"
                )
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


def detect_eeg_gel_bridges(
    residual_eeg: np.ndarray,
    labels: list[str],
    bad_mask: np.ndarray,
    sample_rate: float,
    valid_epoch_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[str]]]:
    """Identify conductive-gel salt bridges via pairwise low-variance electrical distance."""
    n_ch = len(labels)
    cluster_ids = np.arange(n_ch, dtype=np.int16)
    usable = [i for i in range(n_ch) if not bad_mask[i]]
    if len(usable) < 2 or residual_eeg.shape[1] < 16:
        return cluster_ids, []
    ep_samples = max(1, int(round(EEG_EPOCH_SECONDS * sample_rate)))
    ep_count = residual_eeg.shape[1] // ep_samples
    if valid_epoch_mask is not None and valid_epoch_mask.size == ep_count and np.any(valid_epoch_mask):
        eval_data = residual_eeg[:, : ep_count * ep_samples].reshape(n_ch, ep_count, ep_samples)[:, valid_epoch_mask, :].reshape(n_ch, -1)
    else:
        eval_data = residual_eeg
    step = max(1, eval_data.shape[1] // 8000)
    sub = eval_data[:, ::step]
    med = np.median(sub, axis=1, keepdims=True)
    mad_var = (np.median(np.abs(sub - med), axis=1) * 1.4826) ** 2
    parent = list(range(n_ch))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for idx_a in range(len(usable)):
        i = usable[idx_a]
        if mad_var[i] <= 1e-6:
            continue
        for idx_b in range(idx_a + 1, len(usable)):
            j = usable[idx_b]
            if mad_var[j] <= 1e-6:
                continue
            diff = sub[i] - sub[j]
            ed = float(np.median(diff * diff))
            rel_ed = ed / max(float(mad_var[i] + mad_var[j]), 1e-6)
            if ed <= EEG_BRIDGE_ED_UV2 and rel_ed <= EEG_BRIDGE_REL_ED:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in usable:
        root = find(i)
        cluster_ids[i] = root
        groups.setdefault(root, []).append(i)
    bridge_clusters = [[labels[idx] for idx in members] for members in groups.values() if len(members) > 1]
    return cluster_ids, bridge_clusters


def _robust_scalp_reference(
    raw: np.ndarray, bad_mask: np.ndarray, bridge_cluster_ids: np.ndarray | None = None, trim_proportion: float = 0.20,
) -> np.ndarray:
    """Compute a 20% trimmed-mean scalp reference across independent spatial units."""
    usable = np.flatnonzero(~bad_mask)
    if usable.size == 0:
        return np.median(raw, axis=0, keepdims=True).astype(np.float32)
    if bridge_cluster_ids is not None and bridge_cluster_ids.size == raw.shape[0]:
        units: list[np.ndarray] = []
        for cid in np.unique(bridge_cluster_ids[usable]):
            members = usable[bridge_cluster_ids[usable] == cid]
            units.append(raw[members[0]] if members.size == 1 else np.mean(raw[members], axis=0))
        pool = np.stack(units, axis=0)
    else:
        pool = raw[usable]
    k = int(pool.shape[0] * trim_proportion)
    if pool.shape[0] >= 5 and k >= 1 and pool.shape[0] - 2 * k >= 3:
        ordered = np.sort(pool, axis=0)
        return np.mean(ordered[k : pool.shape[0] - k], axis=0, keepdims=True).astype(np.float32)
    return np.median(pool, axis=0, keepdims=True).astype(np.float32)


def _inverse_distance_interpolate(
    data: np.ndarray, labels: list[str], bad: np.ndarray, bridge_cluster_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Interpolate only a minority of failed EEG channels from independent known positions."""
    result = data.copy()
    candidates = [i for i, label in enumerate(labels) if not bad[i] and label in POSITION_MAP]
    if bridge_cluster_ids is not None and bridge_cluster_ids.size == len(labels):
        seen_clusters: set[int] = set()
        good_indices: list[int] = []
        for i in candidates:
            cid = int(bridge_cluster_ids[i])
            if cid not in seen_clusters:
                seen_clusters.add(cid)
                good_indices.append(i)
    else:
        good_indices = candidates
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
        nearest = np.argsort(distances)[: min(4, len(good_indices))]
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
    channel_mask, _ = eeg_epoch_quality(data, sample_rate)
    # Keep an epoch when at least half of the directly observed channels are
    # usable. Channel-level masks let downstream features use the remaining
    # electrodes instead of discarding an entire time segment.
    return np.mean(channel_mask, axis=0) >= 0.5


def eeg_epoch_quality(data: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return channel x epoch validity and robust quality weights.

    Peak-to-peak catches clipping/motion bursts, while a MAD-scaled first
    difference catches sharp transients that may have modest absolute range.
    No samples are overwritten; the result is metadata for weighted analyses.
    """
    samples = int(round(EEG_EPOCH_SECONDS * sample_rate))
    count = data.shape[1] // samples
    if not count:
        return np.zeros((data.shape[0], 0), dtype=bool), np.zeros((data.shape[0], 0), dtype=np.float32)
    epochs = data[:, :count * samples].reshape(data.shape[0], count, samples)
    p2p = np.ptp(epochs, axis=2)
    differences = np.diff(epochs, axis=2)
    diff_scale = np.median(np.abs(differences - np.median(differences, axis=2, keepdims=True)), axis=2) * 1.4826
    diff_peak = np.max(np.abs(differences), axis=2)
    mad_limit = np.maximum(25.0, 12.0 * np.maximum(diff_scale, np.finfo(float).eps))
    valid = (p2p <= EEG_ARTIFACT_PEAK_TO_PEAK_UV) & (diff_peak <= mad_limit)
    p2p_weight = np.clip(1.0 - p2p / max(EEG_ARTIFACT_PEAK_TO_PEAK_UV, np.finfo(float).eps), 0.0, 1.0)
    diff_weight = np.clip(1.0 - diff_peak / np.maximum(mad_limit, np.finfo(float).eps), 0.0, 1.0)
    return valid, (p2p_weight * diff_weight).astype(np.float32)


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


def protocol_aware_subspace_shrinkage(
    filtered_250hz: np.ndarray,
    labels: list[str],
    bad_mask: np.ndarray,
    bridge_cluster_ids: np.ndarray,
    disconnect_epochs: np.ndarray,
    sample_rate: float = 250.0,
    z_cutoff: float = 4.0,
) -> np.ndarray:
    """Sensitivity branch: ASR-style generalized eigen-subspace shrinkage anchored on P10-P50 clean epochs.

    Instead of assuming the first 30 s is resting baseline (movement tasks begin at 6-13 s),
    selects non-disconnect 2 s epochs whose cap-wide median peak-to-peak across usable channels
    falls within P10-P50 of the recording to build a shrinkage-regularized baseline covariance C0.
    Transient non-disconnect epochs with p2p > 200 uV are projected through generalized eigenvectors
    of (C_epoch, C0), shrinking eigenvalues above z_cutoff^2 back to z_cutoff. True disconnect
    epochs are never altered or unmasked.
    """
    out = filtered_250hz.copy()
    usable_idx = np.flatnonzero(~bad_mask)
    ep_len = max(1, int(round(EPOCH_SECONDS * sample_rate)))
    n_ep = out.shape[1] // ep_len
    if usable_idx.size < 8 or n_ep < 6:
        return out

    ep_data = out[usable_idx, : n_ep * ep_len].reshape(usable_idx.size, n_ep, ep_len)
    ep_p2p = np.ptp(ep_data, axis=2)
    med_p2p = np.median(ep_p2p, axis=0)

    non_disc = np.ones(n_ep, dtype=bool)
    if disconnect_epochs.size:
        n_sh = min(n_ep, disconnect_epochs.size)
        non_disc[:n_sh] = ~disconnect_epochs[:n_sh]

    valid_pool = np.flatnonzero(non_disc)
    if valid_pool.size < 4:
        return out

    q10, q50 = np.percentile(med_p2p[valid_pool], [10, 50])
    clean_eps = valid_pool[(med_p2p[valid_pool] >= q10) & (med_p2p[valid_pool] <= q50) & (med_p2p[valid_pool] <= 150.0)]
    if clean_eps.size < 3:
        clean_eps = valid_pool[np.argsort(med_p2p[valid_pool])[: max(3, valid_pool.size // 3)]]

    clean_concat = ep_data[:, clean_eps, :].transpose(0, 2, 1).reshape(usable_idx.size, -1).astype(np.float64)
    clean_concat -= np.mean(clean_concat, axis=1, keepdims=True)
    n_clean_samples = max(1, clean_concat.shape[1] - 1)
    c0 = (clean_concat @ clean_concat.T) / float(n_clean_samples)
    tr0 = float(np.trace(c0)) / float(usable_idx.size)
    c0_reg = 0.90 * c0 + 0.10 * max(tr0, 1.0) * np.eye(usable_idx.size, dtype=np.float64)

    evals0, evecs0 = linalg.eigh(c0_reg)
    evals0 = np.maximum(evals0, 1e-4)
    w_mat = (evecs0 * (1.0 / np.sqrt(evals0))) @ evecs0.T
    winv_mat = (evecs0 * np.sqrt(evals0)) @ evecs0.T

    for ep_i in valid_pool:
        if np.max(ep_p2p[:, ep_i]) <= 180.0:
            continue
        seg = ep_data[:, ep_i, :].astype(np.float64)
        seg_mean = np.mean(seg, axis=1, keepdims=True)
        seg_c = seg - seg_mean
        z_seg = w_mat @ seg_c
        c_z = (z_seg @ z_seg.T) / float(max(1, ep_len - 1))
        lam, u_mat = linalg.eigh(c_z)
        lam = np.maximum(lam, 1e-6)
        gain = np.where(lam > (z_cutoff ** 2), z_cutoff / np.sqrt(lam), 1.0)
        shrink_op = (u_mat * gain) @ u_mat.T
        repaired_seg = winv_mat @ (shrink_op @ z_seg) + seg_mean
        out[usable_idx, ep_i * ep_len : (ep_i + 1) * ep_len] = repaired_seg.astype(np.float32)

    out, _ = _inverse_distance_interpolate(out, labels, bad_mask, bridge_cluster_ids)
    return out


def _plot_eeg_preview(data: np.ndarray, labels: list[str], sample_rate: float, destination: Path) -> None:
    configure_matplotlib_font()
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


def preprocess_eeg(recording: Path, destination: Path, *, save_debug_arrays: bool = False) -> dict[str, Any]:
    assert_writable_output_path(destination)
    header, physical, digital = read_bdf(recording, None)
    sample_rate = header.sample_rate
    duration_seconds = float(physical.shape[1] / max(sample_rate, 1e-6))
    if duration_seconds < EEG_MIN_DURATION_SECONDS:
        raise PreprocessExclusionError(
            "EXCLUDED_SHORT_FRAGMENT",
            f"EEG 记录仅 {duration_seconds:.2f} s，短于 {EEG_MIN_DURATION_SECONDS:.0f} s 最小分析阈值。",
            {"duration_seconds": round(duration_seconds, 3)},
        )
    label_to_index = {normalize_label(label): index for index, label in enumerate(header.labels)}
    configured_channels, configuration_name = eeg_configuration_for_recording(recording)
    missing = [label for label in configured_channels if normalize_label(label) not in label_to_index]
    if missing:
        raise ValueError("缺少配置 EEG 导联：" + ", ".join(missing))
    indices = [label_to_index[normalize_label(label)] for label in configured_channels]
    labels = list(configured_channels)
    raw = physical[indices].astype(np.float32)
    dig = digital[indices]
    dig_min = header.digital_minimums[indices]
    dig_max = header.digital_maximums[indices]
    impedances = load_impedances(recording.with_name("recordInformation.json"))
    clamped_raw, disconnect_epochs = clamp_eeg_disconnect_epochs(raw, dig, dig_min, dig_max, sample_rate)
    non_disconnect_mask = ~disconnect_epochs if disconnect_epochs.size else None
    clamped_raw_centered = clamped_raw - np.median(clamped_raw, axis=1, keepdims=True)
    robust_referenced = clamped_raw_centered - np.median(clamped_raw_centered, axis=0, keepdims=True)
    residual_for_detection = _bandpass(
        _notch_harmonics(robust_referenced, sample_rate, EEG_BAND_HZ[1]), sample_rate, EEG_BAND_HZ,
    )
    bad, bad_reasons = eeg_bad_channels(
        residual_for_detection,
        labels,
        sample_rate,
        digital=dig,
        digital_min=dig_min,
        digital_max=dig_max,
        impedances=impedances,
        valid_epoch_mask=non_disconnect_mask,
    )
    usable_count = int(np.count_nonzero(~bad))
    if usable_count < 16:
        raise PreprocessExclusionError(
            "EXCLUDED_INSUFFICIENT_USABLE_CHANNELS",
            f"直接信号失败的 EEG 导联过多（仅 {usable_count}/{len(labels)} 可用，少于 16 个下限）；不生成伪恢复数据。",
            {"usable_channels": usable_count, "bad_channel_reasons": bad_reasons},
        )
    bridge_cluster_ids, bridge_clusters = detect_eeg_gel_bridges(
        residual_for_detection, labels, bad, sample_rate, valid_epoch_mask=non_disconnect_mask,
    )
    independent_spatial_units = int(np.unique(bridge_cluster_ids[~bad]).size)
    if independent_spatial_units < 12:
        raise PreprocessExclusionError(
            "EXCLUDED_INSUFFICIENT_USABLE_CHANNELS",
            f"导电膏桥接去重后独立头皮空间单元仅 {independent_spatial_units} 个（少于 12 个下限）。",
            {"usable_channels": usable_count, "independent_spatial_units": independent_spatial_units, "gel_bridge_clusters": bridge_clusters},
        )
    pre_line = _line_amplitude_median(clamped_raw_centered[~bad], sample_rate)
    scalp_ref = _robust_scalp_reference(clamped_raw_centered, bad, bridge_cluster_ids)
    analysis_referenced = clamped_raw_centered - scalp_ref
    filtered = _bandpass(
        _notch_harmonics(analysis_referenced, sample_rate, EEG_BAND_HZ[1]), sample_rate, EEG_BAND_HZ,
    )
    cleaned, interpolated = _inverse_distance_interpolate(filtered, labels, bad, bridge_cluster_ids)
    cleaned, output_rate = _resample(cleaned, sample_rate, EEG_TARGET_SAMPLE_RATE_HZ)
    epoch_channel_mask, epoch_quality = eeg_epoch_quality(cleaned, output_rate)
    epoch_channel_mask[bad, :] = False
    if disconnect_epochs.size:
        n_shared = min(disconnect_epochs.size, epoch_channel_mask.shape[1])
        epoch_channel_mask[:, :n_shared][:, disconnect_epochs[:n_shared]] = False
        epoch_quality[:, :n_shared][:, disconnect_epochs[:n_shared]] = 0.0
    valid_epochs = np.mean(epoch_channel_mask, axis=0) >= 0.5 if epoch_channel_mask.shape[1] else np.zeros(0, dtype=bool)
    channel_quality = np.mean(epoch_quality * epoch_channel_mask, axis=1) if epoch_quality.shape[1] else np.zeros(len(labels))
    post_line = _line_amplitude_median(cleaned[~bad], output_rate)

    # Subspace shrinkage sensitivity branch
    subspace_cleaned = protocol_aware_subspace_shrinkage(
        cleaned, labels, bad, bridge_cluster_ids, disconnect_epochs, output_rate
    )
    ep_ch_mask_sub, ep_q_sub = eeg_epoch_quality(subspace_cleaned, output_rate)
    ep_ch_mask_sub[bad, :] = False
    if disconnect_epochs.size:
        n_sh = min(disconnect_epochs.size, ep_ch_mask_sub.shape[1])
        ep_ch_mask_sub[:, :n_sh][:, disconnect_epochs[:n_sh]] = False
        ep_q_sub[:, :n_sh][:, disconnect_epochs[:n_sh]] = 0.0
    valid_epochs_sub = np.mean(ep_ch_mask_sub, axis=0) >= 0.5 if ep_ch_mask_sub.shape[1] else np.zeros(0, dtype=bool)
    post_line_sub = _line_amplitude_median(subspace_cleaned[~bad], output_rate)

    evt_info = (
        read_bdf_task_events(recording.with_name("evt.bdf"), max_duration_seconds=duration_seconds)
        if recording.stem.casefold() == "data"
        else {"companion_evt_bdf": None, "observed_event_times_seconds": [], "observed_event_descriptions": [], "excluded_impedance_event_count": 0}
    )
    evt_times = np.asarray(evt_info["observed_event_times_seconds"], dtype=np.float64)
    evt_descs = np.asarray(evt_info["observed_event_descriptions"], dtype="U32")
    npz_payload: dict[str, Any] = {
        "data_uv": cleaned,
        "data_uv_subspace": subspace_cleaned,
        "labels": np.asarray(labels, dtype="U8"),
        "sample_rate_hz": np.asarray(output_rate),
        "bad_channel_mask": bad,
        "bridge_cluster_id": bridge_cluster_ids,
        "interpolated_channel_mask": np.asarray([label in interpolated for label in labels]),
        "disconnect_clamped_2s_epoch_mask": disconnect_epochs[: epoch_channel_mask.shape[1]] if disconnect_epochs.size else np.zeros(epoch_channel_mask.shape[1], dtype=bool),
        "channel_valid_2s_epoch_mask": epoch_channel_mask,
        "eeg_epoch_quality_weight": epoch_quality,
        "channel_quality_weight": channel_quality.astype(np.float32),
        "valid_2s_epoch_mask": valid_epochs,
        "subspace_valid_2s_epoch_mask": valid_epochs_sub,
        "event_times_seconds": evt_times,
        "event_descriptions": evt_descs,
    }
    if save_debug_arrays:
        npz_payload["raw_data_uv"] = raw
        npz_payload["filtered_data_uv"] = filtered
    destination.mkdir(parents=True, exist_ok=True)
    _save_npz(destination / "eeg_cleaned.npz", **npz_payload)
    _plot_eeg_preview(cleaned, labels, output_rate, destination / "signal_preview_cleaned.png")
    return {
        "modality": "EEG",
        "processing_status": "READY_WITH_MASKS",
        "duration_seconds": round(duration_seconds, 3),
        "source_sample_rate_hz": sample_rate,
        "output_sample_rate_hz": output_rate,
        "bandpass_hz": EEG_BAND_HZ,
        "eeg_configuration": configuration_name,
        "configured_scalp_channel_count": len(configured_channels),
        "independent_spatial_unit_count": independent_spatial_units,
        "gel_bridge_clusters": bridge_clusters,
        "clamped_disconnect_2s_epochs": int(np.count_nonzero(disconnect_epochs)),
        "notch_hz": [LINE_FREQUENCY_HZ],
        "bad_channels_direct_signal": [labels[i] for i in np.flatnonzero(bad)],
        "bad_channel_reasons": bad_reasons,
        "interpolated_channels": interpolated,
        "valid_2s_epochs": int(np.count_nonzero(valid_epochs)),
        "total_2s_epochs": int(valid_epochs.size),
        "channel_quality_weights": {label: round(float(weight), 4) for label, weight in zip(labels, channel_quality)},
        "usable_epoch_percent": round(float(np.mean(valid_epochs) * 100.0), 2) if valid_epochs.size else 0.0,
        "subspace_valid_2s_epochs": int(np.count_nonzero(valid_epochs_sub)),
        "subspace_usable_epoch_percent": round(float(np.mean(valid_epochs_sub) * 100.0), 2) if valid_epochs_sub.size else 0.0,
        "subspace_line_noise_median_after_uv": round(post_line_sub, 3),
        "line_noise_median_before_uv": round(pre_line, 3),
        "line_noise_median_after_uv": round(post_line, 3),
        "line_noise_reduction_percent": round(100 * (1 - post_line / max(pre_line, np.finfo(float).eps)), 2),
        "task_events": evt_info,
        "analysis_rule": "下游统计应使用 channel_valid_2s_epoch_mask 与 eeg_epoch_quality_weight 做窗口级降权；bad_channel_mask 永久排除；桥接簇按独立空间单元去重参考；主分支提供 data_uv，敏感性分支提供 data_uv_subspace。",
    }


def fishburn_tddr(od: np.ndarray, sample_rate: float) -> tuple[np.ndarray, float]:
    """Standard Fishburn et al. (2019) iterative Tukey's biweight TDDR.

    Applies robust M-estimation with Tukey's biweight (c = 4.685, sigma = 1.4826 * MAD)
    on the temporal derivative of optical density, flattening step motion artifacts,
    then reintegrates and calculates the residual post-TDDR spike percentage.
    """
    x = np.asarray(od, dtype=np.float64).copy()
    n = x.size
    if n < 8 or not np.all(np.isfinite(x)):
        return x, 0.0

    if n >= 7:
        med5 = signal.medfilt(x, kernel_size=5)
        resid5 = x - med5
        mad5 = float(np.median(np.abs(resid5 - np.median(resid5))) * 1.4826)
        glitch = np.abs(resid5) > max(0.03, 6.0 * max(mad5, np.finfo(float).eps))
        glitch[[0, -1]] = False
        isolated = glitch & ~(np.r_[False, glitch[:-1]] & np.r_[glitch[1:], False])
        x[isolated] = med5[isolated]

    deriv = np.diff(x, prepend=x[0])
    w = np.ones(n, dtype=np.float64)
    mu = 0.0
    for _ in range(50):
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            break
        mu_new = float(np.sum(w * deriv) / w_sum)
        resid = deriv - mu_new
        sigma = 1.4826 * float(np.median(np.abs(resid)))
        if sigma <= 1e-12:
            mu = mu_new
            break
        u = resid / (4.685 * sigma)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if abs(mu_new - mu) < 1e-6:
            mu = mu_new
            break
        mu = mu_new

    deriv_repaired = w * (deriv - mu) + mu
    repaired = x[0] + np.cumsum(deriv_repaired)
    repaired = repaired - np.mean(repaired) + np.mean(x)

    post_d = np.abs(np.diff(repaired, prepend=repaired[0]))
    post_med = float(np.median(post_d))
    post_mad = float(np.median(np.abs(post_d - post_med)) * 1.4826)
    post_thresh = max(0.03, post_med + 12.0 * max(post_mad, 1e-12))
    post_spike_pct = float(np.mean(post_d > post_thresh) * 100.0)
    return repaired, round(post_spike_pct, 3)


def _robust_derivative_repair(optical_density: np.ndarray, sample_rate: float) -> tuple[np.ndarray, float]:
    """Iterative Tukey's biweight TDDR (Fishburn et al., 2019) + outlier glitch cleaning."""
    return fishburn_tddr(optical_density, sample_rate)


def _fnirs_channel_quality(
    channel: Any, sample_rate: float, coupled_mask: np.ndarray | None = None,
) -> tuple[bool, list[str]]:
    raw_a = channel.intensity_a
    raw_b = channel.intensity_b
    eval_mask = (
        coupled_mask
        if coupled_mask is not None and coupled_mask.size == raw_a.size and int(np.count_nonzero(coupled_mask)) >= max(22, int(round(sample_rate * 10.0)))
        else np.ones(raw_a.size, dtype=bool)
    )
    a_sub = raw_a[eval_mask]
    b_sub = raw_b[eval_mask]
    valid = np.isfinite(a_sub) & np.isfinite(b_sub) & (a_sub > 0) & (b_sub > 0)
    reasons: list[str] = []
    hard_reject = False
    valid_percent = float(np.mean(valid) * 100.0) if valid.size else 0.0
    if valid_percent < 80.0:
        reasons.append(f"非正或缺失原始光强 {100.0 - valid_percent:.1f}%（低于 80% 可恢复下限）")
        return False, reasons
    a = _fill_invalid(a_sub)
    b = _fill_invalid(b_sub)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return False, ["光强缺失段无法构建有限双波长序列"]
    cv = max(
        float(np.std(a) / max(abs(float(np.mean(a))), np.finfo(float).tiny)),
        float(np.std(b) / max(abs(float(np.mean(b))), np.finfo(float).tiny)),
    ) * 100.0
    sci = _scalp_coupling_index(a, b, sample_rate)
    ep_samples = max(2, int(round(2.0 * sample_rate)))
    ep_count = a.size // ep_samples
    if ep_count:
        ea = a[: ep_count * ep_samples].reshape(ep_count, ep_samples)
        eb = b[: ep_count * ep_samples].reshape(ep_count, ep_samples)
        tol_a = max(np.finfo(float).eps * max(1.0, float(np.max(np.abs(a)))) * 10.0, 1e-12)
        tol_b = max(np.finfo(float).eps * max(1.0, float(np.max(np.abs(b)))) * 10.0, 1e-12)
        flat_pct = float(np.mean((np.ptp(ea, axis=1) <= tol_a) | (np.ptp(eb, axis=1) <= tol_b)) * 100.0)
    else:
        flat_pct = 0.0
    rep_ext = 100.0 / max(a.size, 1) * max(
        int(np.count_nonzero(a == np.min(a))) - 1,
        int(np.count_nonzero(a == np.max(a))) - 1,
        int(np.count_nonzero(b == np.min(b))) - 1,
        int(np.count_nonzero(b == np.max(b))) - 1,
    )
    if not np.isfinite(sci) or sci < FNIRS_BAD_SCI:
        hard_reject = True
        reasons.append("SCI 无效（平线或死通道）" if not np.isfinite(sci) else f"SCI {sci:.2f} < {FNIRS_BAD_SCI:.2f}")
    if not np.isfinite(cv) or cv > FNIRS_BAD_CV_PERCENT:
        hard_reject = True
        reasons.append(f"双波长最大 CV {cv:.1f}% > {FNIRS_BAD_CV_PERCENT:.1f}%")
    if flat_pct >= 10.0:
        hard_reject = True
        reasons.append(f"平线分段 {flat_pct:.1f}% >= 10.0%")
    if rep_ext >= 5.0 or (rep_ext >= 1.0 and sci < 0.80):
        hard_reject = True
        reasons.append(f"重复极值饱和 {rep_ext:.2f}%")

    w10_samples = max(20, int(round(10.0 * sample_rate)))
    w10_count = a.size // w10_samples
    if w10_count >= 3:
        w10_a = a[: w10_count * w10_samples].reshape(w10_count, w10_samples)
        w10_b = b[: w10_count * w10_samples].reshape(w10_count, w10_samples)
        cv_w10 = np.maximum(
            np.std(w10_a, axis=1) / np.maximum(np.abs(np.mean(w10_a, axis=1)), 1e-30),
            np.std(w10_b, axis=1) / np.maximum(np.abs(np.mean(w10_b, axis=1)), 1e-30),
        ) * 100.0
        sci_w10 = np.array([
            _scalp_coupling_index(w10_a[i], w10_b[i], sample_rate)
            for i in range(w10_count)
        ])
        w10_pass = (sci_w10 >= FNIRS_BAD_SCI) & (cv_w10 <= FNIRS_BAD_CV_PERCENT)
        w10_pass_pct = float(np.mean(w10_pass) * 100.0)
    else:
        w10_pass_pct = 100.0 if (sci >= FNIRS_BAD_SCI and cv <= FNIRS_BAD_CV_PERCENT) else 0.0

    if hard_reject and w10_pass_pct >= 75.0 and flat_pct < 10.0 and rep_ext < 5.0:
        hard_reject = False
        reasons.append(f"10s分窗准入挽救 (75%合格窗: {w10_pass_pct:.1f}%)")

    return not hard_reject, reasons


def _ransac_one_to_one_match(
    fnirs_m: np.ndarray, eeg_m: np.ndarray, tol: float = 1.0
) -> tuple[list[int], float, list[float]]:
    """Find a global offset delta_t maximizing monotonic one-to-one inliers within `tol` seconds."""
    if fnirs_m.size == 0 or eeg_m.size == 0:
        return [-1] * int(fnirs_m.size), 0.0, []
    best_matched = [-1] * int(fnirs_m.size)
    best_inliers = 0
    best_med_err = float("inf")
    best_max_err = float("inf")
    best_offset = 0.0
    best_errs: list[float] = []

    for i in range(fnirs_m.size):
        for j in range(eeg_m.size):
            cand = float(eeg_m[j] - fnirs_m[i])
            matched = [-1] * int(fnirs_m.size)
            used: set[int] = set()
            last_k = -1
            for m_idx, m_val in enumerate(fnirs_m):
                diffs = np.abs(eeg_m - (m_val + cand))
                k = int(np.argmin(diffs))
                if diffs[k] <= tol and k not in used and k > last_k:
                    used.add(k)
                    last_k = k
                    matched[m_idx] = k
            inlier_pairs = [(m_idx, k) for m_idx, k in enumerate(matched) if k >= 0]
            if not inlier_pairs:
                continue
            ref_offset = float(np.median([eeg_m[k] - fnirs_m[m_idx] for m_idx, k in inlier_pairs]))
            matched_ref = [-1] * int(fnirs_m.size)
            used_ref: set[int] = set()
            errs_ref: list[float] = []
            last_k_ref = -1
            for m_idx, m_val in enumerate(fnirs_m):
                diffs = np.abs(eeg_m - (m_val + ref_offset))
                k = int(np.argmin(diffs))
                if diffs[k] <= tol and k not in used_ref and k > last_k_ref:
                    used_ref.add(k)
                    last_k_ref = k
                    matched_ref[m_idx] = k
                    errs_ref.append(float(diffs[k]))
            cnt = len(errs_ref)
            if cnt == 0:
                continue
            med_e = float(np.median(errs_ref))
            max_e = float(np.max(errs_ref))
            if (
                cnt > best_inliers
                or (cnt == best_inliers and med_e < best_med_err - 1e-6)
                or (cnt == best_inliers and abs(med_e - best_med_err) <= 1e-6 and max_e < best_max_err)
            ):
                best_inliers = cnt
                best_matched = matched_ref
                best_offset = ref_offset
                best_med_err = med_e
                best_max_err = max_e
                best_errs = errs_ref

    return best_matched, best_offset, best_errs


def _split_blocks_by_gap(times: np.ndarray, gap_threshold: float = 32.0) -> list[list[int]]:
    if times.size == 0:
        return []
    blocks: list[list[int]] = [[0]]
    for idx in range(1, times.size):
        if float(times[idx] - times[idx - 1]) > gap_threshold:
            blocks.append([idx])
        else:
            blocks[-1].append(idx)
    return blocks


def align_fnirs_markers_with_eeg(recording: Path, fnirs_markers: np.ndarray) -> dict[str, Any]:
    """Upgraded cross-modal event marker aligner (v8).

    Supports:
      1. COMPLETE_MATCH / CONTIGUOUS_SUBSEQUENCE_ALIGNED / OFFSET_ALIGNED_WITH_MISSED_TRIGGER
      2. PIECEWISE_BLOCK_ALIGNED (independent block-wise constant offsets for multi-block protocols)
      3. RANSAC_INLINE_ALIGNED (global RANSAC inlier consensus with outlier rejection)
      4. Single-modality mutual fallback
    """
    markers = np.asarray(fnirs_markers, dtype=float).reshape(-1)
    t_label = task_label_for(recording)
    evt_path: Path | None = None
    for ancestor in recording.parents:
        if ancestor.name.upper() == "FNIRS":
            eeg_root = ancestor.parent / "EEG"
            if eeg_root.is_dir():
                for sub in sorted(eeg_root.iterdir()):
                    if sub.is_dir() and task_label_for(sub / "data.bdf") == t_label and (sub / "evt.bdf").is_file():
                        evt_path = sub / "evt.bdf"
                        break
            break

    eeg_dedup: list[float] = []
    if evt_path is not None and evt_path.is_file():
        raw_eeg_times = read_bdf_task_events(evt_path)["observed_event_times_seconds"]
        for onset in sorted(float(t) for t in raw_eeg_times):
            if not eeg_dedup or onset - eeg_dedup[-1] > 1.0:
                eeg_dedup.append(onset)
    eeg_arr = np.asarray(eeg_dedup, dtype=float)

    base: dict[str, Any] = {
        "companion_eeg_evt_bdf": str(evt_path) if evt_path else None,
        "observed_fnirs_marker_times_seconds": [round(float(x), 3) for x in markers],
        "observed_fnirs_marker_count": int(markers.size),
        "companion_eeg_event_times_seconds": [round(float(x), 3) for x in eeg_arr],
        "companion_eeg_event_count": int(eeg_arr.size),
        "alignment_status": "UNALIGNED_NEEDS_REVIEW",
        "aligned_protocol_marker_indices": [-1] * int(markers.size),
        "eeg_sync_offset_seconds": None,
        "block_offsets_seconds": [],
        "inlier_count": 0,
        "median_abs_residual_seconds": None,
        "max_abs_residual_seconds": None,
        "per_marker_residuals_seconds": [],
        "replaced_outlier_indices": [],
    }

    if markers.size == 0 and eeg_arr.size == 0:
        base["alignment_status"] = "NO_TASK_MARKERS_BOTH_MODALITIES"
        return base
    if markers.size == 0 and eeg_arr.size > 0:
        base["alignment_status"] = "EEG_COMPANION_MARKERS_AVAILABLE"
        base["inlier_count"] = int(eeg_arr.size)
        base["median_abs_residual_seconds"] = 0.0
        base["max_abs_residual_seconds"] = 0.0
        return base
    if markers.size > 0 and eeg_arr.size == 0:
        base["alignment_status"] = (
            "FNIRS_NATIVE_MARKERS_VALID" if markers.size >= 3 else "NO_COMPANION_EEG_EVENTS"
        )
        base["inlier_count"] = int(markers.size)
        base["aligned_protocol_marker_indices"] = list(range(int(markers.size)))
        base["median_abs_residual_seconds"] = 0.0
        base["max_abs_residual_seconds"] = 0.0
        return base

    # Step 1: Check contiguous exact match
    if markers.size >= 2 and eeg_arr.size >= markers.size:
        best_start = -1
        best_max_err = float("inf")
        best_offset = 0.0
        for start in range(eeg_arr.size - markers.size + 1):
            window = eeg_arr[start : start + markers.size]
            offset = float(np.median(window - markers))
            errs = np.abs((markers + offset) - window)
            max_err = float(np.max(errs))
            if max_err < best_max_err:
                best_max_err = max_err
                best_start = start
                best_offset = offset
        if best_start >= 0 and best_max_err <= 1.5:
            window = eeg_arr[best_start : best_start + markers.size]
            resids = ((markers + best_offset) - window).tolist()
            status = "COMPLETE_MATCH" if markers.size == eeg_arr.size else "CONTIGUOUS_SUBSEQUENCE_ALIGNED"
            return {
                **base,
                "alignment_status": status,
                "aligned_protocol_marker_indices": list(range(best_start, best_start + int(markers.size))),
                "eeg_sync_offset_seconds": round(best_offset, 4),
                "block_offsets_seconds": [round(best_offset, 4)],
                "inlier_count": int(markers.size),
                "median_abs_residual_seconds": round(float(np.median(np.abs(resids))), 4),
                "max_abs_residual_seconds": round(best_max_err, 4),
                "per_marker_residuals_seconds": [round(float(r), 4) for r in resids],
            }

    # Step 2: Check single missed trigger
    matched_15, offset_15, errs_15 = _ransac_one_to_one_match(markers, eeg_arr, tol=1.5)
    if len(errs_15) >= max(2, int(markers.size) - 1) and max(errs_15, default=99.0) <= 1.5:
        resids = [
            round(float((markers[i] + offset_15) - eeg_arr[k]), 4) if k >= 0 else None
            for i, k in enumerate(matched_15)
        ]
        outliers = [i for i, k in enumerate(matched_15) if k < 0]
        return {
            **base,
            "alignment_status": "OFFSET_ALIGNED_WITH_MISSED_TRIGGER",
            "aligned_protocol_marker_indices": matched_15,
            "eeg_sync_offset_seconds": round(offset_15, 4),
            "block_offsets_seconds": [round(offset_15, 4)],
            "inlier_count": len(errs_15),
            "median_abs_residual_seconds": round(float(np.median(errs_15)), 4),
            "max_abs_residual_seconds": round(float(np.max(errs_15)), 4),
            "per_marker_residuals_seconds": resids,
            "replaced_outlier_indices": outliers,
        }

    # Step 3: Piecewise block alignment
    candidate_partitions: list[list[tuple[list[int], list[int]]]] = []
    if t_label == "运动任务" and markers.size >= 14 and eeg_arr.size >= 19:
        if markers.size == 20 and eeg_arr.size == 20:
            candidate_partitions.append(
                [
                    (list(range(0, 7)), list(range(0, 7))),
                    (list(range(7, 13)), list(range(7, 13))),
                    (list(range(13, 20)), list(range(13, 20))),
                ]
            )
            candidate_partitions.append(
                [
                    (list(range(0, 5)), list(range(0, 5))),
                    (list(range(5, 13)), list(range(5, 13))),
                    (list(range(13, 20)), list(range(13, 20))),
                ]
            )
        elif markers.size == 15 and eeg_arr.size == 20:
            candidate_partitions.append(
                [
                    (list(range(0, 7)), list(range(0, 7))),
                    (list(range(7, 15)), list(range(7, 15))),
                ]
            )

    f_blocks = _split_blocks_by_gap(markers, gap_threshold=32.0 if t_label == "运动任务" else 90.0)
    e_blocks = _split_blocks_by_gap(eeg_arr, gap_threshold=32.0 if t_label == "运动任务" else 90.0)
    if len(f_blocks) >= 2 and len(f_blocks) == len(e_blocks):
        candidate_partitions.append(list(zip(f_blocks, e_blocks)))

    best_pw: dict[str, Any] | None = None
    for part in candidate_partitions:
        pw_matched = [-1] * int(markers.size)
        pw_resids: list[float | None] = [None] * int(markers.size)
        pw_inlier_errs: list[float] = []
        pw_offsets: list[float] = []
        blocks_ok = True
        for f_idxs, e_idxs in part:
            f_sub = markers[f_idxs]
            e_sub = eeg_arr[e_idxs]
            m_sub, off_sub, errs_sub = _ransac_one_to_one_match(f_sub, e_sub, tol=0.6)
            min_req = 4 if t_label == "运动任务" else max(2, len(f_idxs) - 1)
            if len(errs_sub) < min_req or max(errs_sub, default=99.0) > 0.6:
                blocks_ok = False
                break
            pw_offsets.append(round(off_sub, 4))
            for local_i, global_f_i in enumerate(f_idxs):
                if m_sub[local_i] >= 0:
                    global_e_i = e_idxs[m_sub[local_i]]
                    pw_matched[global_f_i] = global_e_i
                    r_val = float((markers[global_f_i] + off_sub) - eeg_arr[global_e_i])
                    pw_resids[global_f_i] = round(r_val, 4)
                    pw_inlier_errs.append(abs(r_val))
        if blocks_ok and pw_inlier_errs:
            if best_pw is None or len(pw_inlier_errs) > best_pw["inlier_count"] or (
                len(pw_inlier_errs) == best_pw["inlier_count"]
                and float(np.median(pw_inlier_errs)) < best_pw["median_abs_residual_seconds"]
            ):
                outliers = [i for i, k in enumerate(pw_matched) if k < 0]
                best_pw = {
                    **base,
                    "alignment_status": "PIECEWISE_BLOCK_ALIGNED",
                    "aligned_protocol_marker_indices": pw_matched,
                    "eeg_sync_offset_seconds": pw_offsets[0],
                    "block_offsets_seconds": pw_offsets,
                    "inlier_count": len(pw_inlier_errs),
                    "median_abs_residual_seconds": round(float(np.median(pw_inlier_errs)), 4),
                    "max_abs_residual_seconds": round(float(np.max(pw_inlier_errs)), 4),
                    "per_marker_residuals_seconds": pw_resids,
                    "replaced_outlier_indices": outliers,
                }

    # Step 4: Global RANSAC alignment
    matched_r, offset_r, errs_r = _ransac_one_to_one_match(markers, eeg_arr, tol=1.0)
    min_n = min(int(markers.size), int(eeg_arr.size))
    req_inliers = max(3, int(math.ceil(0.75 * min_n)))
    ransac_ok = (
        len(errs_r) >= req_inliers
        and float(np.median(errs_r)) <= 0.25
    )

    if best_pw is not None and (not ransac_ok or best_pw["inlier_count"] > len(errs_r)):
        return best_pw

    if ransac_ok:
        resids = [
            round(float((markers[i] + offset_r) - eeg_arr[k]), 4) if k >= 0 else None
            for i, k in enumerate(matched_r)
        ]
        outliers = [i for i, k in enumerate(matched_r) if k < 0]
        return {
            **base,
            "alignment_status": "RANSAC_INLINE_ALIGNED",
            "aligned_protocol_marker_indices": matched_r,
            "eeg_sync_offset_seconds": round(offset_r, 4),
            "block_offsets_seconds": [round(offset_r, 4)],
            "inlier_count": len(errs_r),
            "median_abs_residual_seconds": round(float(np.median(errs_r)), 4),
            "max_abs_residual_seconds": round(float(np.max(errs_r)), 4),
            "per_marker_residuals_seconds": resids,
            "replaced_outlier_indices": outliers,
        }

    if best_pw is not None:
        return best_pw

    # Step 5: Companion EEG protocol markers fallback
    if eeg_arr.size >= 6 and len(errs_r) >= 2 and float(np.median(errs_r)) <= 0.40:
        resids = [
            round(float((markers[i] + offset_r) - eeg_arr[k]), 4) if k >= 0 else None
            for i, k in enumerate(matched_r)
        ]
        outliers = [i for i, k in enumerate(matched_r) if k < 0]
        return {
            **base,
            "alignment_status": "RANSAC_PARTIAL_EEG_COMPANION_ANCHORED",
            "aligned_protocol_marker_indices": matched_r,
            "eeg_sync_offset_seconds": round(offset_r, 4),
            "block_offsets_seconds": [round(offset_r, 4)],
            "inlier_count": len(errs_r),
            "median_abs_residual_seconds": round(float(np.median(errs_r)), 4),
            "max_abs_residual_seconds": round(float(np.max(errs_r)), 4),
            "per_marker_residuals_seconds": resids,
            "replaced_outlier_indices": outliers,
        }

    if eeg_arr.size >= 6:
        return {
            **base,
            "alignment_status": "EEG_COMPANION_MARKERS_AVAILABLE",
            "eeg_sync_offset_seconds": round(offset_r, 4) if errs_r else None,
            "inlier_count": len(errs_r),
            "median_abs_residual_seconds": round(float(np.median(errs_r)), 4) if errs_r else None,
            "max_abs_residual_seconds": round(float(np.max(errs_r)), 4) if errs_r else None,
        }

    return base


def preprocess_fnirs(recording: Path, destination: Path) -> dict[str, Any]:
    assert_writable_output_path(destination)
    infrared_eye_tracker_interference = is_infrared_eye_tracking_task(recording)
    source = read_fnirs(recording, None)
    duration_seconds = float(source.time[-1] - source.time[0]) if source.time.size >= 2 else 0.0
    if duration_seconds < FNIRS_MIN_DURATION_SECONDS:
        raise PreprocessExclusionError(
            "EXCLUDED_SHORT_FRAGMENT",
            f"fNIRS 记录仅 {duration_seconds:.2f} s，短于 {FNIRS_MIN_DURATION_SECONDS:.0f} s 最小血液动力学滤波窗口。",
            {"duration_seconds": round(duration_seconds, 3)},
        )
    uncoupled_boundary = detect_fnirs_uncoupled_boundary_mask(source)
    coupled_mask = ~uncoupled_boundary
    kept: list[np.ndarray] = []
    labels: list[str] = []
    rejected: dict[str, list[str]] = {}
    spike_rates: dict[str, float] = {}
    wavelength_valid_masks: list[np.ndarray] = []
    motion_masks: list[np.ndarray] = []
    repair_masks: list[np.ndarray] = []
    analysis_masks: list[np.ndarray] = []
    motion_levels: list[np.ndarray] = []
    short_gap_masks: list[np.ndarray] = []
    for channel in source.channels:
        usable, reasons = _fnirs_channel_quality(channel, source.sample_rate_hz, coupled_mask=coupled_mask)
        if not usable:
            rejected[channel.label] = reasons
            continue
        wavelength_valid, motion_mask, repair_mask, analysis_mask = build_fnirs_sample_masks(
            channel, source.sample_rate_hz, time=source.time, uncoupled_sample_mask=uncoupled_boundary,
        )
        wavelengths = []
        per_wavelength_spikes = []
        for values in (channel.intensity_a, channel.intensity_b):
            intensity = _fill_invalid(values, max_gap_samples=max(1, int(round(source.sample_rate_hz))))
            filter_intensity = _fill_invalid(values)
            med_ref = float(np.median(filter_intensity[coupled_mask])) if np.any(coupled_mask) else float(np.median(filter_intensity))
            od = -np.log(filter_intensity / max(med_ref, np.finfo(float).tiny))
            repaired, spike_rate = _robust_derivative_repair(od, source.sample_rate_hz)
            filtered_od = _bandpass(repaired[None, :], source.sample_rate_hz, FNIRS_BAND_HZ)[0]
            invalid = ~np.isfinite(intensity)
            filtered_od[invalid] = np.nan
            wavelengths.append(filtered_od)
            per_wavelength_spikes.append(spike_rate)
        kept.append(np.stack(wavelengths))
        labels.append(channel.label)
        spike_rates[channel.label] = round(float(np.mean(per_wavelength_spikes)), 3)
        wavelength_valid_masks.append(wavelength_valid)
        motion_masks.append(motion_mask)
        repair_masks.append(repair_mask)
        analysis_masks.append(analysis_mask)
        motion_levels.append(motion_artifact_level(channel.intensity_a, channel.intensity_b, source.sample_rate_hz))
        valid_a = np.isfinite(channel.intensity_a) & (channel.intensity_a > 0)
        valid_b = np.isfinite(channel.intensity_b) & (channel.intensity_b > 0)
        short_gap_masks.append((~valid_a) | (~valid_b))
    if len(kept) < 20:
        raise PreprocessExclusionError(
            "EXCLUDED_INSUFFICIENT_USABLE_CHANNELS",
            f"可用 fNIRS 通道仅 {len(kept)}/{len(source.channels)} 个（少于 20 个下限），无法形成可靠分析集。",
            {"kept_channel_count": len(kept), "rejected_channels": rejected},
        )
    data = np.stack(kept).astype(np.float32)
    wavelength_valid = np.stack(wavelength_valid_masks).astype(bool)
    motion_mask = np.stack(motion_masks).astype(bool)
    repair_mask = np.stack(repair_masks).astype(bool)
    analysis_mask = np.stack(analysis_masks).astype(bool)
    motion_level = np.stack(motion_levels).astype(np.int8)
    raw_invalid_mask = np.stack(short_gap_masks).astype(bool)
    marker_alignment = align_fnirs_markers_with_eeg(recording, source.marker_times_seconds)
    sync_offset_val = (
        float(marker_alignment["eeg_sync_offset_seconds"])
        if marker_alignment["eeg_sync_offset_seconds"] is not None
        else float("nan")
    )
    destination.mkdir(parents=True, exist_ok=True)
    _save_npz(
        destination / "fnirs_cleaned_od.npz",
        raw_intensity=np.stack([
            np.stack([channel.intensity_a, channel.intensity_b]) for channel in source.channels
            if channel.label in labels
        ]).astype(np.float32),
        optical_density=data,
        labels=np.asarray(labels, dtype="U20"),
        wavelengths_nm=np.asarray([source.channels[0].wavelength_a_nm, source.channels[0].wavelength_b_nm]),
        time_seconds=source.time - source.time[0],
        sample_rate_hz=np.asarray(source.sample_rate_hz),
        wavelength_valid_sample_mask=wavelength_valid,
        sample_valid_mask=np.all(wavelength_valid, axis=1),
        cap_coupled_sample_mask=coupled_mask.astype(bool),
        motion_artifact_mask=motion_mask,
        repair_mask=repair_mask,
        analysis_sample_mask=analysis_mask,
        motion_artifact_level=motion_level,
        raw_invalid_sample_mask=raw_invalid_mask,
        channel_quality_weight=np.mean(analysis_mask, axis=1).astype(np.float32),
        known_ir_interference=np.asarray(infrared_eye_tracker_interference),
        observed_marker_times_seconds=np.asarray(source.marker_times_seconds, dtype=np.float64),
        aligned_protocol_marker_indices=np.asarray(marker_alignment["aligned_protocol_marker_indices"], dtype=np.int16),
        eeg_sync_offset_seconds=np.asarray(sync_offset_val, dtype=np.float64),
    )
    if infrared_eye_tracker_interference:
        processing_status = "READY_WITH_MASKS_KNOWN_IR_INTERFERENCE"
        quality_degradation = "KNOWN_IR_INTERFERENCE_PARTIALLY_OCCLUDED"
        analysis_branch = "EYE_IR_DEGRADED_MASKED"
        occlusion = {
            "status": "PARTIAL_REPORTED",
            "reported": True,
            "mask_available": False,
            "basis": "用户确认已做一定程度遮挡；.nirs 文件未提供逐样本物理遮挡日志。",
        }
    else:
        processing_status = "READY_CHANNEL_FILTERED"
        quality_degradation = None
        analysis_branch = "STANDARD_MASKED"
        occlusion = {
            "status": "NOT_APPLICABLE",
            "reported": False,
            "mask_available": False,
            "basis": None,
        }
    return {
        "modality": "FNIRS",
        "processing_status": processing_status,
        "duration_seconds": round(duration_seconds, 3),
        "uncoupled_boundary_sample_percent": round(float(np.mean(uncoupled_boundary) * 100.0), 3),
        "marker_alignment": marker_alignment,
        "analysis_branch": analysis_branch,
        "quality_degradation": quality_degradation,
        "known_ir_interference": infrared_eye_tracker_interference,
        "eye_tracker_occlusion": occlusion,
        "output": "双波长去噪光密度；未转换 HbO/HbR，避免在未确认 DPF、探头距离和消光系数设置时引入模型假设。",
        "bandpass_hz": FNIRS_BAND_HZ,
        "kept_channels": labels,
        "rejected_channels": rejected,
        "motion_spike_percent_after_detection": spike_rates,
        "mask_summary": {
            "mask_file": "fnirs_cleaned_od.npz",
            "kept_channel_count": len(labels),
            "mean_repair_sample_percent": round(float(np.mean(repair_mask) * 100.0), 3),
            "mean_analysis_sample_percent": round(float(np.mean(analysis_mask) * 100.0), 3),
        "channels_with_repaired_samples": int(np.count_nonzero(np.any(repair_mask, axis=1))),
        "channel_quality_weights": {label: round(float(weight), 4) for label, weight in zip(labels, np.mean(analysis_mask, axis=1))},
        "motion_artifact_tiers": {"0": "无明显突变", "1": "轻度，建议降权", "2": "重度，默认排除"},
        },
        "sensitivity_branches": {
            "default_masked": "使用 analysis_sample_mask；原始非正/缺失样本与运动突变样本不进入默认统计。",
            "unmasked_debug": "仅用于波形审计，不得作为生理估计。",
            "eye_task_excluded": "可作为敏感性分析分支；不是默认数据路径。" if infrared_eye_tracker_interference else "不适用",
        },
        "analysis_rule": "下游分析仅使用 analysis_sample_mask，并按 channel_quality_weight 降权；motion_artifact_level=1 可保留作敏感性分析，=2 默认排除；长缺失在 optical_density 中保持 NaN，不跨缺失段解释慢变化。",
    }


def overlap_add_harmonic_cancel(
    raw_2d: np.ndarray,
    sample_rate: float,
    win_sec: float = 10.0,
    hop_sec: float = 2.5,
    line_freq: float = 50.0,
    max_freq: float = 450.0,
    half_bw_hz: float = 1.35,
) -> np.ndarray:
    """Hann WOLA (weighted overlap-add) sinusoidal harmonic regression + 20-450 Hz Butterworth bandpass.

    Applies a Hann analysis window before rfft (suppressing rectangular-window 1/f spectral
    sidelobe leakage by >31 dB) and projects out the narrowband sinusoidal subspace at
    50*k +/- half_bw_hz, then reconstructs via Hann synthesis window (normalized by sum(w^2))
    with reflection boundary padding and 20-450 Hz 4th-order zero-phase Butterworth bandpass.
    """
    x = np.asarray(raw_2d, dtype=np.float64)
    single_1d = x.ndim == 1
    if single_1d:
        x = x[None, :]
    n_ch, n_samples = x.shape
    x_det = signal.detrend(x, axis=-1, type="linear")
    harmonics = [f for f in np.arange(line_freq, min(max_freq, sample_rate * 0.45) + 0.1, line_freq)]
    upper = min(EMG_BAND_HZ[1], sample_rate * 0.45)
    sos = signal.butter(4, (EMG_BAND_HZ[0], upper), btype="bandpass", fs=sample_rate, output="sos")
    if not harmonics or n_samples < 64:
        out = signal.sosfiltfilt(sos, x_det, axis=-1).astype(np.float32)
        return out[0] if single_1d else out

    win_len = min(max(256, int(round(win_sec * sample_rate))), max(256, n_samples))
    if win_len % 2 == 1:
        win_len += 1
    hop_len = max(16, min(int(round(hop_sec * sample_rate)), win_len // 4))
    pad_len = min(win_len // 2, n_samples - 1)
    x_pad = np.pad(x_det, ((0, 0), (pad_len, pad_len)), mode="reflect")
    n_pad = x_pad.shape[1]

    if win_len >= n_pad:
        starts = [0]
        win_len = n_pad
    else:
        starts = list(range(0, n_pad - win_len + 1, hop_len))
        if starts[-1] + win_len < n_pad:
            starts.append(n_pad - win_len)

    hann = np.maximum(np.hanning(win_len).astype(np.float64), 1e-3)
    hann_sq = hann * hann

    freqs = np.fft.rfftfreq(win_len, d=1.0 / sample_rate)
    keep_bins = np.ones(freqs.size, dtype=bool)
    for h_freq in harmonics:
        keep_bins &= np.abs(freqs - h_freq) > half_bw_hz

    acc = np.zeros((n_ch, n_pad), dtype=np.float64)
    w_sum = np.zeros(n_pad, dtype=np.float64)
    t_norm = np.linspace(-1.0, 1.0, win_len, dtype=np.float64)
    t_norm_denom = float(np.sum(t_norm * t_norm))

    for s in starts:
        seg = x_pad[:, s : s + win_len]
        seg_mean = np.mean(seg, axis=1, keepdims=True)
        seg_slope = ((seg - seg_mean) @ t_norm)[:, None] / max(t_norm_denom, 1e-12)
        trend = seg_mean + seg_slope * t_norm[None, :]
        seg_win = (seg - trend) * hann[None, :]
        spec = np.fft.rfft(seg_win, axis=-1)
        spec[:, ~keep_bins] = 0.0
        clean_win = np.fft.irfft(spec, n=win_len, axis=-1)
        acc[:, s : s + win_len] += clean_win * hann[None, :] + trend * hann_sq[None, :]
        w_sum[s : s + win_len] += hann_sq

    deharm_pad = acc / np.maximum(w_sum[None, :], 1e-12)
    deharm = deharm_pad[:, pad_len : pad_len + n_samples]
    filtered = signal.sosfiltfilt(sos, deharm, axis=-1).astype(np.float32)
    return filtered[0] if single_1d else filtered


def preprocess_emg(recording: Path, destination: Path, *, save_debug_arrays: bool = False) -> dict[str, Any]:
    assert_writable_output_path(destination)
    series = read_emg_csv(recording, None)
    mapping = resolve_emg_mapping([item.sensor_id for item in series], recording)
    labels = [item.label for item in series]
    target_rate = min(1000.0, min(item.sample_rate_hz for item in series))
    spans = [float(item.time[-1] - item.time[0]) for item in series if item.time.size >= 2]
    max_span = max(spans) if spans else 0.0
    active_spans = [s for s in spans if s >= 0.80 * max_span]
    common_duration = min(active_spans) if active_spans else (min(spans) if spans else 0.0)
    if not np.isfinite(common_duration) or common_duration <= 0:
        raise ValueError("EMG 通道没有共同的正记录时长。")
    channel_metrics = [assess_emg_channel(item) for item in series]
    common_samples = max(3, int(np.floor(common_duration * target_rate)) + 1)
    cleaned: list[np.ndarray] = []
    raw_cropped: list[np.ndarray] = []
    valid_masks_out: list[np.ndarray] = []
    pre_line: list[float] = []
    post_line: list[float] = []
    early_terminated_sensors: list[int] = []
    gap_repaired_sensors: list[int] = []
    for item in series:
        elapsed = item.time - item.time[0]
        finite = np.isfinite(elapsed) & np.isfinite(item.values)
        el = elapsed[finite]
        val = item.values[finite]
        if el.size < 3:
            raise ValueError(f"传感器 {item.sensor_id} 的有效样本不足。")
        dt = np.diff(el)
        gap_limit_s = max(0.020, 3.0 / max(item.sample_rate_hz, 1e-6))
        has_gap = bool(np.any(dt <= 0) or np.any(dt > gap_limit_s))
        ended_early = bool(el[-1] < common_duration - max(0.05, 3.0 / max(item.sample_rate_hz, 1e-6)))
        if has_gap or ended_early:
            order = np.argsort(el, kind="mergesort")
            el_s = el[order]
            val_s = val[order]
            uniq_keep = np.r_[True, np.diff(el_s) > 1e-9]
            el_u = el_s[uniq_keep]
            val_u = val_s[uniq_keep]
            n_native = max(3, int(np.floor(common_duration * item.sample_rate_hz)) + 1)
            grid_t = np.arange(n_native, dtype=np.float64) / item.sample_rate_hz
            values = np.interp(grid_t, el_u, val_u)
            native_valid = grid_t <= (el_u[-1] + 0.5 / max(item.sample_rate_hz, 1e-6))
            gap_idx = np.flatnonzero(np.diff(el_u) > gap_limit_s)
            if gap_idx.size:
                gap_repaired_sensors.append(item.sensor_id)
                for g in gap_idx:
                    native_valid[(grid_t >= el_u[g] - 0.15) & (grid_t <= el_u[g + 1] + 0.15)] = False
            if ended_early:
                early_terminated_sensors.append(item.sensor_id)
        else:
            keep = elapsed <= common_duration + max(1e-9, 0.5 / item.sample_rate_hz)
            values = item.values[keep]
            grid_t = elapsed[keep]
            native_valid = np.ones(values.size, dtype=bool)
        if values.size < 3:
            raise ValueError(f"传感器 {item.sensor_id} 的共同时间段样本不足。")
        obs_slice = values[native_valid] if int(np.count_nonzero(native_valid)) >= int(item.sample_rate_hz * 2) else values
        before = signal.detrend(obs_slice)
        raw_resampled, _ = _resample(values[None, :], item.sample_rate_hz, target_rate)
        raw_cropped.append(raw_resampled[0][:common_samples])
        pre_line.append(_line_amplitude_median(before[None, :], item.sample_rate_hz))
        filtered = overlap_add_harmonic_cancel(values[None, :], item.sample_rate_hz)[0]
        filtered_res, output_rate = _resample(filtered[None, :], item.sample_rate_hz, target_rate)
        filtered = filtered_res[0][:common_samples]
        out_t = np.arange(filtered.size, dtype=np.float64) / output_rate
        out_valid = np.interp(out_t, grid_t if grid_t.size == native_valid.size else np.arange(native_valid.size) / item.sample_rate_hz, native_valid.astype(np.float64)) >= 0.999
        filtered_valid = filtered[out_valid] if int(np.count_nonzero(out_valid)) >= int(output_rate * 2) else filtered
        post_line.append(_line_amplitude_median(filtered_valid[None, :], output_rate))
        cleaned.append(filtered)
        valid_masks_out.append(out_valid)
    length = min(item.size for item in cleaned)
    data = np.stack([item[:length] for item in cleaned]).astype(np.float32)
    raw_data = np.stack([item[:length] for item in raw_cropped]).astype(np.float32)
    sample_valid_mask = np.stack([mask[:length] for mask in valid_masks_out]).astype(bool)
    env_cutoff = min(5.0, target_rate * 0.40)
    if env_cutoff > 0.2 and length > 18:
        env_sos = signal.butter(4, env_cutoff, btype="low", fs=target_rate, output="sos")
        envelope_5hz = np.maximum(0.0, signal.sosfiltfilt(env_sos, np.abs(data), axis=-1)).astype(np.float32)
    else:
        envelope_5hz = np.abs(data).astype(np.float32)
    common_duration_output = (length - 1) / target_rate
    quality_scores = np.asarray([metric.quality_score for metric in channel_metrics], dtype=np.float32)
    quality_status = np.asarray([metric.status for metric in channel_metrics], dtype="U24")

    rescued_sensors: list[int] = []
    for ch_idx, metric in enumerate(channel_metrics):
        if metric.status == "POOR" and metric.line_noise_power_percent >= 85.0:
            if (
                metric.flat_epoch_percent < 10.0
                and metric.repeated_extreme_percent < 1.0
                and metric.missing_sample_percent < 5.0
            ):
                ch_filt = data[ch_idx]
                ch_mask = sample_valid_mask[ch_idx]
                ch_active = ch_filt[ch_mask] if np.count_nonzero(ch_mask) >= int(target_rate * 2) else ch_filt
                env_ch = envelope_5hz[ch_idx]
                env_active = env_ch[ch_mask] if np.count_nonzero(ch_mask) >= int(target_rate * 2) else env_ch
                p20, p95 = np.percentile(env_active, [20, 95])
                burst_mod = float(p95 / max(p20, 1e-12))

                nper = min(int(round(target_rate * 2.0)), max(16, ch_active.size))
                f_welch, p_welch = signal.welch(ch_active, fs=target_rate, nperseg=nper)
                upper = min(EMG_BAND_HZ[1], target_rate * 0.45)
                broad = _band_power(f_welch, p_welch, EMG_BAND_HZ[0], upper)
                harm_post = 0.0
                h = LINE_FREQUENCY_HZ
                while h <= upper:
                    harm_post += _band_power(f_welch, p_welch, h - 1.0, h + 1.0)
                    h += LINE_FREQUENCY_HZ
                post_line_pct = 100.0 * harm_post / max(broad, 1e-30)

                if post_line_pct < 5.0 and burst_mod >= 2.0:
                    quality_status[ch_idx] = "CAUTION_HIGH_RAW_LINE"
                    quality_scores[ch_idx] = 65.0
                    rescued_sensors.append(series[ch_idx].sensor_id)

    window_size = max(1, int(round(target_rate * EMG_EPOCH_SECONDS)))
    window_count = int(np.ceil(length / window_size))
    window_mask = np.zeros((len(series), window_count), dtype=bool)
    window_weight = np.zeros((len(series), window_count), dtype=np.float32)
    for channel_index, metric in enumerate(channel_metrics):
        window_mask[channel_index], window_weight[channel_index] = emg_window_quality(
            data[channel_index],
            target_rate,
            float(quality_scores[channel_index]),
            raw_values=raw_data[channel_index],
            valid_sample_mask=sample_valid_mask[channel_index],
        )
    sensor_ids = np.asarray([item.sensor_id for item in series], dtype=np.int16)
    muscle_indices = np.asarray(
        [item.muscle_index if item.muscle_index is not None else -1 for item in series], dtype=np.int16
    )
    expected_sensor_mask = np.asarray(
        [mapping["mapping_valid_for_primary_features"] and item.muscle_index is not None for item in series],
        dtype=bool,
    )
    signal_gate = np.isin(quality_status, ("GOOD", "CAUTION", "CAUTION_HIGH_RAW_LINE")) & expected_sensor_mask
    missing_sensor_ids = mapping["mapping_missing_sensor_ids"]
    unexpected_sensor_ids = mapping["mapping_unexpected_sensor_ids"]
    canonical_missing_sensor_ids = mapping["canonical_missing_sensor_ids"]
    canonical_unexpected_sensor_ids = mapping["canonical_unexpected_sensor_ids"]
    in_session_tree = bool(session_for(recording))
    protocol_complete = bool(common_duration_output >= EMG_PROTOCOL_COMPLETE_SECONDS)
    quality_degradation = (
        "INCOMPLETE_PROTOCOL_DURATION"
        if in_session_tree and common_duration_output < EMG_PROTOCOL_DEGRADED_SECONDS
        else None
    )
    npz_payload: dict[str, Any] = {
        "data_source_unit": data,
        "data_raw_source_unit": raw_data,
        "envelope_5hz_source_unit": envelope_5hz,
        "sample_valid_mask": sample_valid_mask,
        "labels": np.asarray(labels, dtype="U80"),
        "logical_muscle_labels": np.asarray([item.muscle for item in series], dtype="U40"),
        "sensor_ids": sensor_ids,
        "muscle_indices": muscle_indices,
        "canonical_sensor_ids": np.asarray(
            [item.canonical_sensor_id if item.canonical_sensor_id is not None else -1 for item in series], dtype=np.int16
        ),
        "expected_sensor_ids": np.asarray(mapping["mapping_expected_sensor_ids"], dtype=np.int16),
        "canonical_expected_sensor_ids": np.asarray(EXPECTED_SENSOR_IDS, dtype=np.int16),
        "expected_sensor_mask": expected_sensor_mask,
        "missing_sensor_ids": np.asarray(missing_sensor_ids, dtype=np.int16),
        "unexpected_sensor_ids": np.asarray(unexpected_sensor_ids, dtype=np.int16),
        "canonical_missing_sensor_ids": np.asarray(canonical_missing_sensor_ids, dtype=np.int16),
        "canonical_unexpected_sensor_ids": np.asarray(canonical_unexpected_sensor_ids, dtype=np.int16),
        "mapping_profile_id": np.asarray(mapping["mapping_profile_id"]),
        "mapping_status": np.asarray(mapping["mapping_status"]),
        "mapping_confidence": np.asarray(mapping["mapping_confidence"]),
        "mapping_valid_for_primary_features": np.asarray(mapping["mapping_valid_for_primary_features"]),
        "sample_rate_hz": np.asarray(target_rate),
        "time_seconds": np.arange(length) / target_rate,
        "channel_quality_score": quality_scores,
        "channel_quality_status": quality_status,
        "channel_signal_gate": signal_gate,
        "window_seconds": np.asarray(EMG_EPOCH_SECONDS),
        "window_mask": window_mask,
        "window_quality_weight": window_weight,
    }
    if save_debug_arrays:
        npz_payload["data_filtered_source_unit"] = data
    destination.mkdir(parents=True, exist_ok=True)
    _save_npz(destination / "emg_cleaned.npz", **npz_payload)
    return {
        "modality": "EMG",
        "processing_status": "READY_CONTINUOUS",
        "quality_degradation": quality_degradation,
        "protocol_duration_complete": protocol_complete,
        "early_terminated_sensor_ids": early_terminated_sensors,
        "rescued_high_raw_line_sensor_ids": rescued_sensors,
        "gap_repaired_sensor_ids": gap_repaired_sensors,
        "bandpass_hz": EMG_BAND_HZ,
        "notch_hz": list(np.arange(LINE_FREQUENCY_HZ, EMG_BAND_HZ[1] + 0.1, LINE_FREQUENCY_HZ)),
        "source_amplitude_unit": "未声明，保持源文件单位",
        "source_line_noise_median": round(float(np.nanmedian(pre_line)), 6),
        "output_line_noise_median": round(float(np.nanmedian(post_line)), 6),
        "analysis_rule": "CSV 未提供任务事件；输出仅为连续信号，任何分段必须用外部同步日志确定。",
        "common_duration_seconds": round(float(common_duration_output), 6),
        "sensor_ids": sensor_ids.tolist(),
        "muscle_indices": muscle_indices.tolist(),
        "logical_muscle_labels": [item.muscle for item in series],
        "canonical_sensor_ids": [item.canonical_sensor_id if item.canonical_sensor_id is not None else -1 for item in series],
        "mapping_profile_id": mapping["mapping_profile_id"],
        "mapping_status": mapping["mapping_status"],
        "mapping_confidence": mapping["mapping_confidence"],
        "mapping_basis": mapping["mapping_basis"],
        "mapping_source": mapping["mapping_source"],
        "mapping_valid_for_primary_features": mapping["mapping_valid_for_primary_features"],
        "physical_to_logical": mapping["physical_to_logical"],
        "physical_to_canonical": mapping["physical_to_canonical"],
        "expected_sensor_ids": mapping["mapping_expected_sensor_ids"],
        "canonical_expected_sensor_ids": list(EXPECTED_SENSOR_IDS),
        "missing_sensor_ids": missing_sensor_ids,
        "unexpected_sensor_ids": unexpected_sensor_ids,
        "canonical_missing_sensor_ids": canonical_missing_sensor_ids,
        "canonical_unexpected_sensor_ids": canonical_unexpected_sensor_ids,
        "channel_quality_scores": {str(item.sensor_id): round(float(quality_scores[idx]), 3) for idx, item in enumerate(series)},
        "channel_quality_status": {str(item.sensor_id): str(quality_status[idx]) for idx, item in enumerate(series)},
        "analysis_rule_channel_gate": "保留所有传感器及质量证据；主特征要求 mapping_valid_for_primary_features=True 且非 POOR，CAUTION 传感器需携带质量分数或降权；跨记录按逻辑肌肉索引对齐。",
        "window_quality": {
            "window_seconds": EMG_EPOCH_SECONDS,
            "valid_window_percent_by_channel": {
                str(item.sensor_id): round(float(np.mean(window_mask[index]) * 100.0), 3)
                for index, item in enumerate(series)
            },
            "weight_rule": "窗口权重=通道质量分×鲁棒幅度稳定性；窗口级 mask 仅作为默认统计门控，原始/滤波数据均保留。",
        },
    }


def _preprocess_output_path(recording: Path, data_root: Path, output_root: Path) -> Path:
    return output_path_for(recording, data_root, output_root)


def _observed_task_event_count(recording: Path, modality: str | None) -> int | None:
    try:
        if modality == "EEG":
            return len(read_bdf_task_events(recording.with_name("evt.bdf"))["observed_event_times_seconds"])
        if modality == "FNIRS":
            return int(read_fnirs(recording, None).marker_times_seconds.size)
    except Exception:
        return None
    return None


def audit_batch_provenance(
    recordings: list[Path], data_root: Path,
) -> tuple[dict[Path, str], dict[Path, tuple[str, str, dict[str, Any]]], dict[Path, str]]:
    """Pre-compute SHA-256 hashes and quarantine cross-subject or cross-task byte duplicates."""
    hashes: dict[Path, str] = {}
    by_hash: dict[str, list[Path]] = {}
    for rec in recordings:
        if recording_modality(rec) is None:
            continue
        digest = _sha256_file(rec)
        hashes[rec] = digest
        by_hash.setdefault(digest, []).append(rec)

    identity_map = load_subject_identity_map(data_root)
    exclusions: dict[Path, tuple[str, str, dict[str, Any]]] = {}
    notes: dict[Path, str] = {}

    def _rel_str(path: Path) -> str:
        try:
            return str(path.relative_to(data_root))
        except ValueError:
            return str(path)

    def _subject_of(path: Path) -> str:
        try:
            return path.relative_to(data_root).parts[0]
        except (ValueError, IndexError):
            return "外部输入"

    for rec, digest in hashes.items():
        subj = _subject_of(rec)
        id_prefix_match = re.search(r"(20\d{8})", rec.name)
        if id_prefix_match and id_prefix_match.group(1) != subj and len(by_hash.get(digest, [])) == 1:
            expected_name = identity_map.get(subj, "")
            if expected_name and expected_name in rec.name:
                notes[rec] = f"文件名含编号 {id_prefix_match.group(1)} 笔误，但受试者姓名（{expected_name}）与唯一 SHA-256 确认归属外层目录 {subj}"

    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        subjects = {_subject_of(p) for p in group}
        if len(subjects) > 1:
            def _owner_score(path: Path) -> tuple[int, int, str]:
                subj = _subject_of(path)
                cname = identity_map.get(subj, "")
                name_hit = 1 if (cname and cname in path.name) else 0
                id_hit = 1 if subj in path.name else 0
                return (-name_hit, -id_hit, f"{subj}/{_rel_str(path)}")

            ordered = sorted(group, key=_owner_score)
            winner = ordered[0]
            winner_subj = _subject_of(winner)
            winner_rel = _rel_str(winner)
            for dup in ordered[1:]:
                if _subject_of(dup) != winner_subj:
                    exclusions[dup] = (
                        "EXCLUDED_CROSS_SUBJECT_DUPLICATE",
                        f"跨受试者字节级重复：SHA-256 与 {winner_rel} 完全一致，姓名/编号归属核验指向 {winner_subj}",
                        {"canonical_subject": winner_subj, "canonical_source_relative_path": winner_rel},
                    )
        remaining = [p for p in group if p not in exclusions]
        by_slot: dict[tuple[str, str, str | None], list[Path]] = {}
        for p in remaining:
            by_slot.setdefault((_subject_of(p), session_for(p, data_root), recording_modality(p)), []).append(p)
        for (_subj, _sess, mod), slot_files in by_slot.items():
            tasks = {task_label_for(p) for p in slot_files}
            if len(tasks) <= 1:
                continue
            task_order = {"静息态": 0, "眼动任务": 1, "运动任务": 2}

            def _task_score(path: Path) -> tuple[int, int, int, str]:
                tlabel = task_label_for(path)
                ev_cnt = _observed_task_event_count(path, mod)
                proto_fit = 0
                if ev_cnt == 0 and tlabel == "静息态":
                    proto_fit = 2
                elif ev_cnt is not None and 4 <= ev_cnt <= 8 and tlabel == "眼动任务":
                    proto_fit = 2
                elif ev_cnt is not None and ev_cnt >= 12 and tlabel == "运动任务":
                    proto_fit = 2
                stem_fit = 1 if any(k in path.stem for k in ("静息", "眼动", "运动") if k in tlabel) else 0
                return (-proto_fit, -stem_fit, task_order.get(tlabel, 9), _rel_str(path))

            ordered_tasks = sorted(slot_files, key=_task_score)
            winner = ordered_tasks[0]
            winner_task = task_label_for(winner)
            winner_rel = _rel_str(winner)
            for dup in ordered_tasks[1:]:
                if task_label_for(dup) != winner_task:
                    exclusions[dup] = (
                        "EXCLUDED_CROSS_TASK_DUPLICATE",
                        f"同受试者跨任务目录字节级重复：SHA-256 与 {winner_rel} 完全一致，事件标记协议指向 {winner_task}",
                        {"canonical_task": winner_task, "canonical_source_relative_path": winner_rel},
                    )
    return hashes, exclusions, notes


def _write_exclusion_summary(
    destination: Path,
    metadata: dict[str, Any],
    modality: str,
    status: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert_writable_output_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "modality": modality,
        "processing_status": status,
        "exclusion_reason": reason,
        "exclusion_details": details or {},
        "analysis_branch": "EXCLUDED",
        "quality_degradation": status,
        **metadata,
    }
    (destination / "preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def assign_primary_recordings(rows: list[dict[str, Any]]) -> None:
    """Select one canonical primary recording per (subject, session, modality, task)."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("subject", ""), row.get("session", ""), row.get("modality", ""), row.get("task", ""))
        groups.setdefault(key, []).append(row)

    for _key, members in groups.items():
        ready_items: list[tuple[tuple[float, float, float, int], dict[str, Any], dict[str, Any]]] = []
        for row in members:
            status = str(row.get("status", ""))
            out_dir = Path(row["output_directory"]) if row.get("output_directory") else None
            summary_path = out_dir / "preprocessing_summary.json" if out_dir else None
            summary: dict[str, Any] = {}
            if summary_path and summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    summary = {}
            if status.startswith("READY") or status == "CACHED_READY":
                mod = row.get("modality", "")
                fname = str(row.get("source_filename", "")).lower()
                if mod == "EEG":
                    valid_ep = float(summary.get("valid_2s_epochs", 0))
                    rank = (1.0 if valid_ep >= 30 else 0.0, valid_ep, float(summary.get("usable_epoch_percent", 0.0)), 1 if fname == "data.bdf" else 0)
                elif mod == "FNIRS":
                    dur = float(summary.get("duration_seconds", 0.0))
                    cov = float((summary.get("mask_summary") or {}).get("mean_analysis_sample_percent", 0.0)) / 100.0
                    rank = (1.0 if dur >= 60.0 else 0.0, dur * cov, float(len(summary.get("kept_channels") or [])), 0)
                else:
                    dur = float(summary.get("common_duration_seconds", 0.0))
                    scores = list((summary.get("channel_quality_scores") or {}).values())
                    mean_q = float(np.mean(scores)) if scores else 0.0
                    rank = (1.0 if summary.get("protocol_duration_complete") else 0.0, dur, mean_q, 0)
                ready_items.append((rank, row, summary))
            elif status.startswith("EXCLUDED") or status == "CACHED_EXCLUDED":
                row["is_primary_for_task"] = False
                row["recording_role"] = status.replace("CACHED_", "")
            else:
                row["is_primary_for_task"] = False
                row["recording_role"] = "FAILED_QC"

        ready_items.sort(key=lambda item: item[0], reverse=True)
        for idx, (_rank, row, summary) in enumerate(ready_items):
            if idx == 0:
                row["is_primary_for_task"] = True
                row["recording_role"] = "PRIMARY"
            else:
                fname = str(row.get("source_filename", "")).lower()
                align_status = str((summary.get("marker_alignment") or {}).get("alignment_status", ""))
                is_cont = (row.get("modality") == "EEG" and fname != "data.bdf") or align_status == "CONTIGUOUS_SUBSEQUENCE_ALIGNED"
                row["is_primary_for_task"] = False
                row["recording_role"] = "CONTINUATION_FRAGMENT" if is_cont else "SECONDARY_VALID_RECORDING"

        for row in members:
            out_dir = Path(row["output_directory"]) if row.get("output_directory") else None
            summary_path = out_dir / "preprocessing_summary.json" if out_dir else None
            if summary_path and summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if (
                        summary.get("is_primary_for_task") != row["is_primary_for_task"]
                        or summary.get("recording_role") != row["recording_role"]
                    ):
                        summary["is_primary_for_task"] = row["is_primary_for_task"]
                        summary["recording_role"] = row["recording_role"]
                        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                except (OSError, ValueError, TypeError):
                    pass


def _write_index(output_root: Path, rows: list[dict[str, Any]], data_root: Path | None = None) -> None:
    assert_writable_output_path(output_root, data_root)
    fields = [
        "subject", "session", "task", "modality", "is_primary_for_task", "recording_role",
        "source_relative_path", "source_sha256",
        "preprocess_pipeline_version", "preprocess_parameter_hash", "status",
        "known_ir_interference", "eye_tracker_occlusion_status", "analysis_branch", "quality_degradation",
        "output_directory", "message",
    ]
    with (output_root / "preprocessing_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "preprocessing_index.json").write_text(json.dumps({
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "preprocess_pipeline_version": PREPROCESS_PIPELINE_VERSION,
        "records": rows,
        "boundary": "READY 表示已生成保守预处理数组和掩码，不表示临床有效或可跳过人工波形复核；EXCLUDED_* 表示按溯源或最小信号门槛保守隔离。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成可追溯的 EEG、sEMG、fNIRS 预处理数据。")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--subject", default="", help="默认处理 data 下全部受试者；可指定一个目录名。")
    parser.add_argument("--input", type=Path, help="单个文件或目录；指定后忽略 --subject。")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/preprocessed/current"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug-arrays", action="store_true", help="额外保存全采样率中间调试数组（显著增大磁盘占用）。")
    args = parser.parse_args()
    assert_writable_output_path(args.output_root, args.data_root)
    input_path = args.input if args.input else (args.data_root / args.subject if args.subject else None)
    recordings = discover_recordings(args.data_root, input_path)
    if not recordings:
        parser.error("未发现可处理记录。")
    args.output_root.mkdir(parents=True, exist_ok=True)
    param_hash = _preprocess_parameter_hash()
    hashes, provenance_exclusions, provenance_notes = audit_batch_provenance(recordings, args.data_root)
    rows: list[dict[str, Any]] = []
    for recording in recordings:
        modality = recording_modality(recording)
        if modality is None:
            continue
        destination = _preprocess_output_path(recording, args.data_root, args.output_root)
        assert_writable_output_path(destination, args.data_root)
        source_hash = hashes.get(recording) or _sha256_file(recording)
        metadata = {
            **recording_metadata(recording, args.data_root, destination),
            "source_sha256": source_hash,
            "preprocess_pipeline_version": PREPROCESS_PIPELINE_VERSION,
            "preprocess_parameter_hash": param_hash,
        }
        if recording in provenance_notes:
            metadata["provenance_note"] = provenance_notes[recording]
        if destination.exists() and args.overwrite:
            shutil.rmtree(destination)
        if destination.exists() and any(destination.iterdir()):
            if _preprocess_cache_is_current(destination, recording, source_hash):
                rows.append(_cached_row_from_summary(destination, metadata, modality))
                continue
            shutil.rmtree(destination)
        if recording in provenance_exclusions:
            exc_status, exc_reason, exc_details = provenance_exclusions[recording]
            _write_exclusion_summary(destination, metadata, modality, exc_status, exc_reason, exc_details)
            rows.append({
                **metadata,
                "modality": modality,
                "status": exc_status,
                "analysis_branch": "EXCLUDED",
                "quality_degradation": exc_status,
                "output_directory": str(destination),
                "message": exc_reason,
            })
            print(f"{modality} {recording}: {exc_status}: {exc_reason}")
            continue
        try:
            if modality == "EEG":
                summary = preprocess_eeg(recording, destination, save_debug_arrays=args.save_debug_arrays)
            elif modality == "EMG":
                summary = preprocess_emg(recording, destination, save_debug_arrays=args.save_debug_arrays)
            else:
                summary = preprocess_fnirs(recording, destination)
            summary.update(metadata)
            summary.update(
                {
                    "source_sha256": source_hash,
                    "preprocess_pipeline_version": PREPROCESS_PIPELINE_VERSION,
                    "preprocess_parameter_hash": param_hash,
                }
            )
            (destination / "preprocessing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({
                **metadata,
                "modality": modality,
                "status": summary["processing_status"],
                "known_ir_interference": summary.get("known_ir_interference"),
                "eye_tracker_occlusion_status": summary.get("eye_tracker_occlusion", {}).get("status"),
                "analysis_branch": summary.get("analysis_branch"),
                "quality_degradation": summary.get("quality_degradation"),
                "output_directory": str(destination),
                "message": metadata.get("provenance_note", ""),
            })
            print(f"{modality} {recording}: {summary['processing_status']}")
        except PreprocessExclusionError as exc:
            _write_exclusion_summary(destination, metadata, modality, exc.status, str(exc), exc.details)
            rows.append({
                **metadata,
                "modality": modality,
                "status": exc.status,
                "analysis_branch": "EXCLUDED",
                "quality_degradation": exc.status,
                "output_directory": str(destination),
                "message": str(exc),
            })
            print(f"{modality} {recording}: {exc.status}: {exc}")
        except Exception as exc:
            rows.append({**metadata, "modality": modality, "status": "FAILED", "output_directory": str(destination), "message": str(exc)})
            print(f"{modality} {recording}: FAILED: {exc}")
    assign_primary_recordings(rows)
    _write_index(args.output_root, rows, args.data_root)
    return 0 if any(row["status"].startswith(("READY", "CACHED")) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
